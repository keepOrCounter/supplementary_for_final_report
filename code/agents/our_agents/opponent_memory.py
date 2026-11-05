"""Cross-game opponent memory for the Prince (our method).

Motivation: every existing configuration models the
opponents from scratch each game. When the SAME people play a series of games
(mixed opponent types, roles re-dealt every game), a human Prince gradually
"learns each opponent's style". This module is that mechanism, folded into our
belief-tracking framework instead of a separate multi-LLM system (cf. the EMO
baseline, which spends ~70 LLM calls per game and cannot revise a prediction
until its player acts again):

    - After each COMPLETED game the runner calls ``record_game`` with the final
        state, i.e. with FULL information (the post-game "review"). For every
        opponent we store their true role and every answer they gave, annotated
        with the ground truth: the truthful answer, whether they lied, and which
        outcome slot (PRINCESS / QUEEN / OTHER) the answer pointed at.
    - The records are rendered as text and fed to the LLM in later games:
        per-opponent for the memory-aware soft-likelihood judge
        (``Kripke.MemoryKripkeState``), or all together for the ablation Prince's
        LLM question/guess prompts. What gets rendered is the ``report_mode``:
        the RAW ground-truth records, the LLM-written opponent ``profile`` of the
        opponent (maintained by ``OpponentProfiler``, one call per opponent after
        each game), or both.

Recording is purely symbolic (no LLM calls); LLM cost sits in the cached
judge / decision calls that consume the memory, plus (only when profiling is
enabled) the per-opponent profile updates between games.
"""
from __future__ import annotations
import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from game.predefined_data_classes import GameState, QAPair, QuestionType
from agents.our_agents.Kripke import PossibleWorld, implication, KripkeState, ThirdOrderKripkeState, \
    NeuroSymbolicKripkeState
from agents.our_agents.schemas import OpponentProfile
from core.llm_client import LLMClient
from core.helper_functions import SeedSource


# Candidate opponent TYPES for the Bayesian type memory (likelihood=bayes): "which of our likelihood models fits this opponent
# best", and the adaptive Prince then predicts them with exactly that model.
#   uniform = 0-order random opponent
#   strategic3 = 2-order opponent (the exact model of SymbolicPlayerAgent)
#   llm = a generic LLM-style opponent, scored by the memoryless soft-likelihood judge.
#     Its distribution is the same for every opponent.
TYPE_MODELS = ("uniform", "strategic3", "llm")
# The opponent agent (its config `type`) each candidate was built for. An opponent
# whose type is not a key here is one no candidate models exactly.
BUILT_FOR = {"random": "uniform", "symbolic": "strategic3", "llm": "llm"}


@dataclass
class ObservedQA:
    """One question-answer exchange by one opponent, annotated post-game with ground truth."""
    question: str  # human-readable question text (Q3 rendered with its target)
    answer: str
    truth: str  # the truthful answer in that game
    truthful: bool  # answer == truth
    answer_slot: str  # outcome slot the answer implicated (PRINCESS/QUEEN/OTHER)
    truth_slot: str  # outcome slot of the truthful answer


@dataclass
class OpponentGameRecord:
    """Everything one opponent revealed in one finished game."""
    game_index: int
    role: str  # true role name
    faction: str  # Faction value, e.g. "queen"
    constraint: str  # Constraint value, e.g. "must_lie"
    answers: list[ObservedQA] = field(default_factory=list)


def _question_text(qa: QAPair) -> str:
    """Question as asked (Q3's generic 'Player X' rendered with the real target)."""
    if qa.question_type == QuestionType.Q3:
        about = "the center card" if qa.question_param == -1 else f"Player {qa.question_param}"
        return f"Who is {about}?"
    return qa.question_type.value


class CrossGameMemory:
    """Append-only store of opponent behaviour across the games of one run.

    Keyed by player id, so it assumes each opponent keeps their seat across the
    run (the runner rejects ``shuffle_opponents`` when memory is enabled).

    Attributes:
        records:  player id -> list of game records
        games_recorded: int, number of games recorded so far (used to version the memory)
        window: int, render only each opponent's last `window` games in prompt reports
            (0 = all)
        report_mode: what ``opponent_report`` / ``full_report`` render per opponent
        type_log_likelihoods: player id -> {type model name: cumulative log-likelihood of that
            opponent's observed answers under that model} (update_type_posteriors)
        type_history: one snapshot of type_log_likelihoods + posteriors per scored game,
            the game-by-game learning trajectory of the type belief
    """

    REPORT_MODES = ("raw", "profile", "both")

    def __init__(self, window: int = 0, report_mode: str = "raw"):
        if report_mode not in self.REPORT_MODES:
            raise ValueError(f"Unknown report_mode: '{report_mode}' (expected one of {self.REPORT_MODES})")
        self.records: dict[int, list[OpponentGameRecord]] = defaultdict(list) # player id -> list of game records
        self.games_recorded: int = 0
        self.window = window
        self.report_mode = report_mode
        self.profiles: dict[int, str] = {}
        self.type_log_likelihoods: dict[int, dict[str, float]] = {}
        # player id -> {type model name: cumulative log-likelihood}
        self.type_history: list[dict] = []
        # one snapshot per scored game: the learning trajectory of the type belief

    def type_posterior(self, player_id: int) -> dict[str, float]:
        """P(type t | opponent's answers so far): exact Bayes over TYPE_MODELS.

            P(t | D) = P(t) * L_t(D) / sum over T of  P(T) * L_T(D)

            symbols:
            t, T = a candidate model in TYPE_MODELS
            D = every answer this opponent has given, across ALL games
            L_t(D) = likelihood of D under t = product over answers a of P_t(a)
            P(t) = prior, taken uniform (so it cancels)

        type_log_likelihoods stores log L_t(D), the running sum of per-answer
        log-likelihoods that update_type_posteriors keeps adding to (never reset).
        So this is one continuous game-by-game Bayes update, not a per-game
        restart. An unscored opponent returns the uniform prior.
        """
        scores = self.type_log_likelihoods.get(player_id)
        if not scores:
            return {name: 1.0 / len(TYPE_MODELS) for name in TYPE_MODELS}
        missing = [name for name in TYPE_MODELS if name not in scores]
        if missing:
            # Stored scores were built for a different candidate set (e.g. a warm
            # start from an older memory.json). Their sums are not comparable.
            raise ValueError(f"type_posterior: Player {player_id} is missing candidates {missing}; "
                f"stored {sorted(scores)} does not match TYPE_MODELS {list(TYPE_MODELS)}")
        top = max(scores[name] for name in TYPE_MODELS)  # log-sum-exp normalisation
        exps = {name: math.exp(scores[name] - top) for name in TYPE_MODELS}
        total = sum(exps.values())
        return {name: e / total for name, e in exps.items()}

    def snapshot_type_scores(self) -> None:
        """Append the current per-opponent type log-likelihoods and posteriors.

        Called once per scored game (update_type_posteriors), so type_history is the
        game-by-game learning trajectory of the type belief: it shows whether each
        opponent's posterior converges to the right model over the run.
        """
        # String player ids so the snapshot is JSON-ready as-is (save/load need no conversion).
        log_likelihoods = {}
        posteriors = {}
        for pid in sorted(self.type_log_likelihoods):
            log_likelihoods[str(pid)] = dict(self.type_log_likelihoods[pid])
            posteriors[str(pid)] = self.type_posterior(pid)
        self.type_history.append({"games_recorded": self.games_recorded, \
            "log_likelihoods": log_likelihoods, "posteriors": posteriors})

    @property
    def version(self) -> int:
        """Bumps once per recorded game; part of the judge cache key, so cached
        judgments never outlive the memory snapshot they were made from."""
        return self.games_recorded

    def record_game(self, state: GameState) -> None:
        """Post-game review with full information: store every opponent's true
        role and their answers annotated with the ground truth."""
        true_world = PossibleWorld({pid: r.name for pid, r in state.roles.items()},
            center_card=state.center_card.name if state.center_card else None)

        game_records: dict[int, OpponentGameRecord] = {}
        for qa in state.qa_history:
            if qa.answer is None:
                continue
            record = game_records.get(qa.target_id)
            if record is None:
                role = state.roles[qa.target_id]
                record = OpponentGameRecord(game_index=self.games_recorded, role=role.name,
                    faction=role.faction.value, constraint=role.constraint.value)
                game_records[qa.target_id] = record

            truth = true_world.get_truth(qa.question_type, qa.target_id, qa.question_param)
            record.answers.append(ObservedQA(
                question=_question_text(qa),
                answer=qa.answer,
                truth=truth,
                truthful=qa.answer == truth,
                answer_slot=implication(qa.answer, qa.question_type, qa.question_param, true_world),
                truth_slot=implication(truth, qa.question_type, qa.question_param, true_world),
            ))

        for pid, record in game_records.items():
            self.records[pid].append(record)
        self.games_recorded += 1

    @staticmethod
    def _record_a_game(record: OpponentGameRecord) -> list[str]:
        """One game record as text: "was <role> (...)" + one line per answer."""
        lines = [f"was {record.role} ({record.faction} faction, {record.constraint.replace('_', ' ')})"]
        for a in record.answers:
            verdict = "TRUTH" if a.truthful else "LIE"
            note = f"{verdict}, pointed at {a.answer_slot}"
            if not a.truthful:
                note += f"; truth was \"{a.truth}\" = {a.truth_slot}"
            lines.append(f"  Asked \"{a.question}\" -> answered \"{a.answer}\" ({note})")
        return lines

    def _raw_report(self, player_id: int, records: list[OpponentGameRecord]) -> str:
        """The ground-truth annotated game records, windowed."""
        header = f"Player {player_id} across {len(records)} previous game(s):"
        if self.window and len(records) > self.window:
            header = f"Player {player_id} across {self.window} previous game(s):"
            records = records[-self.window:]
        lines = [header]
        # Numbered relative to the rendered window, so the header count and the game
        # labels always agree and a truncated history stays imperceptible to the model
        for i, record in enumerate(records, 1):
            body = self._record_a_game(record)
            lines.append(f"  Game {i}: {body[0]}")
            lines.extend("  " + line for line in body[1:])
        return "\n".join(lines)

    def opponent_report(self, player_id: int) -> str:
        """Memory text for ONE opponent per report_mode ("" if nothing recorded yet)."""
        records = self.records.get(player_id, [])
        if not records:
            return ""
        raw = self._raw_report(player_id, records)
        if self.report_mode == "raw":
            return raw
        profile = self.profiles.get(player_id, "")
        if not profile:
            # No profile yet (or a failed update kept none).
            return "" if self.report_mode == "profile" else raw
        notes = f"Your accumulated notes on Player {player_id}:\n{profile}"
        if self.report_mode == "profile":
            return notes
        return raw + "\n" + notes  # "both"

    def raw_opponent_report(self, player_id: int) -> str:
        """The windowed ground-truth records regardless of report_mode ("" if none).

        The profiler reads THIS rather than opponent_report: in profile mode
        opponent_report returns the profile itself, and a profiler fed its own
        output would drift away from the ground truth.
        """
        records = self.records.get(player_id, [])
        return self._raw_report(player_id, records) if records else ""

    def full_report(self) -> str:
        """Memory text for every observed opponent (for the Prince's LLM prompts)."""
        reports = [self.opponent_report(pid) for pid in sorted(self.records)]
        return "\n".join(r for r in reports if r)

    def to_dict(self) -> dict:
        """turn the memory into a JSON-friendly dictionary (inverse of the parsing in ``load``)."""
        opponents = {}
        for pid, records in sorted(self.records.items()):
            opponents[str(pid)] = [asdict(record) for record in records]
        result = {"games_recorded": self.games_recorded, "opponents": opponents}
        if self.profiles:
            result["profiles"] = {str(pid): text for pid, text in sorted(self.profiles.items())}
        if self.type_log_likelihoods:
            result["type_log_likelihoods"] = {str(pid): scores for pid, scores in sorted(self.type_log_likelihoods.items())}
            # Derived, for human reading of memory.json: what the scores mean.
            posteriors = {}
            for pid in sorted(self.type_log_likelihoods):
                posteriors[str(pid)] = {name: round(p, 4) for name, p in self.type_posterior(pid).items()}
            result["type_posteriors"] = posteriors
        if self.type_history:
            result["type_history"] = self.type_history  # already JSON-ready (string player ids)
        return result

    def save(self, path: str) -> None:
        """Persist the store as JSON (the runner writes memory.json in the results dir)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, window: int = 0, report_mode: str = "raw") -> CrossGameMemory:
        """Rebuild memory from a memory.json written by ``save``.

        Warm start (config agents.prince.memory.load_path): the run keeps
        recording on top of the loaded games. A missing/invalid file raises.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        memory = cls(window=window, report_mode=report_mode)
        memory.games_recorded = data["games_recorded"]
        for pid, records in data["opponents"].items():
            memory.records[int(pid)] = [OpponentGameRecord(game_index=r["game_index"], role=r["role"], \
                faction=r["faction"], constraint=r["constraint"], answers=[ObservedQA(**a) for a in r["answers"]]) \
                    for r in records]
        memory.profiles = {int(pid): text for pid, text in data.get("profiles", {}).items()}
        memory.type_log_likelihoods = {int(pid): dict(scores) for pid, scores in data.get("type_log_likelihoods", {}).items()}
        memory.type_history = data.get("type_history", [])
        return memory


def update_type_posteriors(memory: CrossGameMemory, state: GameState, *, \
    llm_state: NeuroSymbolicKripkeState = None, tom_beta: float = None, \
    tom_smoothing: str = "softmax", tom_eps: float = None) -> None:
    """Add the finished game's answers to each candidate's running log-likelihood.

    The true roles are known post-game, so for each answer a the opponent gave we
    can compute P_t(a | true world) under each candidate t, and add
    log P_t(a | true world) to type_log_likelihoods[opponent][t]. Summed over all
    answers and games this is log L_t(D) that type_posterior normalises.

        symbols:
        t          = a candidate model (uniform, strategic3 or llm)
        a          = one observed answer
        true world = the real role assignment, revealed after the game

    strategic3 conditions on the public QA history, so answers are scored in
    transcript order and its belief is advanced (advance_shadow) between them.
    The "llm" candidate is scored by the memoryless soft-likelihood judge, so this
    needs an `llm_state` and makes LLM calls (cached by situation). Use the same
    tom knobs as the in-game BayesKripkeState.
    """
    use_center_card = state.center_card is not None
    true_world = PossibleWorld({pid: r.name for pid, r in state.roles.items()}, \
        center_card=state.center_card.name if state.center_card else None)

    uniform = KripkeState(state.num_players, use_center_card)
    strategic3 = ThirdOrderKripkeState(state.num_players, use_center_card, beta=tom_beta, \
        smoothing=tom_smoothing, eps=tom_eps)
    symbolic = {"uniform": uniform, "strategic3": strategic3}
    if "llm" in TYPE_MODELS and llm_state is None:
        raise ValueError("update_type_posteriors: the 'llm' candidate needs a memoryless llm_state judge")

    for qa in state.qa_history:
        if qa.answer is None:
            continue
        logliks = memory.type_log_likelihoods.setdefault(qa.target_id, {name: 0.0 for name in TYPE_MODELS})
        for name in TYPE_MODELS:
            if name == "llm":
                p = llm_state.soft_likelihood(true_world, qa.answer, qa.question_type, qa.target_id, \
                    qa.question_param)
            else:
                p = symbolic[name].answer_likelihood(true_world, qa.answer, qa.question_type, qa.target_id, \
                    qa.question_param)
            if p <= 0.0:
                # The env enforces the answer constraints, so every recorded answer
                # has p > 0 under every candidate (uniform >= 1/A; eps_greedy >=
                # eps/A; softmax > 0; the judge blends in a uniform floor). Zero
                # means the store or the environment is inconsistent.
                raise ValueError(f"type scoring: {name} gave zero probability to "
                    f"Player {qa.target_id}'s answer {qa.answer!r}")
            logliks[name] += math.log(p)
        strategic3.advance_shadow(qa)
    memory.snapshot_type_scores()  # record this game's point on the learning trajectory


class OpponentProfiler:
    """Maintains an LLM-written portrait (profile) per opponent.

    After each completed game, one LLM call per opponent rewrites that
    opponent's profile from their full (windowed) ground-truth records. The
    profile is recomputed from the raw data every game.
    """

    def __init__(self, llm_client: LLMClient, seed_source: SeedSource = None, reasoning_effort: str = "none", \
        print_out: bool = False):
        self.llm_client = llm_client
        self.seed_source = seed_source or SeedSource(None)

        self.reasoning_effort = reasoning_effort
        self.print_out = print_out
        self.failed_updates: int = 0

    def update(self, memory: CrossGameMemory) -> None:
        """Rewrite the profile of every opponent from their records so far.

        A failed LLM call keeps that opponent's previous profile (warned +
        counted), so one bad call cannot abort a whole learning series.
        """
        from agents.our_agents.prompts import build_profile_prompt

        last_game = memory.games_recorded - 1
        for pid in sorted(memory.records):
            latest = memory.records[pid][-1]
            if latest.game_index != last_game:
                # Every opponent answers in every recorded game, stale latest
                # record means the store is inconsistent.
                raise RuntimeError(f"profile update: Player {pid}'s latest record is from game "
                    f"{latest.game_index + 1}, expected game {last_game + 1}")
            system_prompt, user_prompt = build_profile_prompt(pid, memory.raw_opponent_report(pid))
            seed = self.seed_source.next_seed()
            try:
                result, _reasoning = self.llm_client.chat_structured_model(user_prompt=user_prompt, \
                    system_prompt=system_prompt, response_model=OpponentProfile, seed=seed, \
                        reasoning_effort=self.reasoning_effort)
            except Exception as e:
                self.failed_updates += 1
                warnings.warn(f"profile update for Player {pid} failed (previous profile kept): {e}", \
                    RuntimeWarning)
                continue
            memory.profiles[pid] = result.profile.strip()
            if self.print_out:
                print(f"[Profiler] Player {pid}: {memory.profiles[pid]}")
