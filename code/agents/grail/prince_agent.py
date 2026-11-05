"""GRAIL-style Prince agent (the analogue of GRAIL's agent_acl.py + policy_models/heuristic.py)."""
from __future__ import annotations

import numpy as np

from game.predefined_data_classes import Observation, PrinceObservation, Role, Phase, QuestionType
from agents.base import BaseAgent
from agents.grail.tofu_factor_graph import TofuFactorGraph, CENTER_ID
from agents.grail.schemas import VibesResponse
from agents.grail.prompts import build_vibes_prompt
from core.llm_client import LLMClient


class GrailPrinceAgent(BaseAgent):
    """GRAIL-style Prince baseline (ported from shahabrrad/Bayesian-Avalon):
    passive belief tracking (max-product BP on TofuFactorGraph) + optional
    one-shot LLM 'vibes' prior + belief-greedy policy. Unlike the symbolic IG
    Prince, it consumes beliefs but never optimises which question is informative.
    """

    def __init__(self, player_id: int, role: Role, num_players: int, use_center_card: bool, \
        llm_client: LLMClient = None, use_llm_prior: bool = False, algorithm: str = "max", \
        likelihood_codebook=None, posterior_model=None, vibes_reasoning_effort: str = "none", \
        print_out: bool = False):
        super().__init__(player_id, role)
        # Data-driven factor options (the analogue of GRAIL's trained NN factor):
        #   likelihood_codebook -> (B) learn the LIKELIHOOD p(answer|roles)
        #   posterior_model     -> (A) learn the POSTERIOR p(role|history), faithful
        # Both None -> the uniform likelihood (shared with the symbolic Prince).
        self.graph_model = TofuFactorGraph(num_players, use_center_card, \
            likelihood_codebook=likelihood_codebook, posterior_model=posterior_model)
        self.llm_client = llm_client
        self.use_llm_prior = use_llm_prior and llm_client is not None
        self.algorithm = algorithm
        # Thinking effort for the 'vibes' prior sub-call ONLY.
        self.vibes_reasoning_effort = vibes_reasoning_effort
        self.print_out = print_out
        self.num_players = num_players

        self.latest_beliefs: dict[int, dict[str, float]] | None = None
        # per-step context filled in by _update_beliefs (logged for later analysis)
        self._num_answered: int = 0
        self._llm_prior_used: bool = False
        # Why the LLM prior was / wasn't applied this step, so a `false`
        # llm_prior_used is not ambiguous in the log. One of:
        #   disabled / skipped_no_history / llm_error / all_same / applied
        self._llm_prior_status: str = "disabled"
        self._llm_reasoning: str | None = None  # vibes CoT (for debugging)
        self._llm_error: str | None = None      # exception string if the vibes call failed
        self.decision_log: list[dict] = []


    def _update_beliefs(self, obs: PrinceObservation) -> None:
        # Reset per-step LLM context (populated by _llm_vibes when it runs).
        self._llm_reasoning = None
        self._llm_error = None
        priors = None
        if not self.use_llm_prior:
            self._llm_prior_status = "disabled"
        elif not obs.qa_history:
            # First questioning: no answers to reason about -> skip the LLM entirely.
            self._llm_prior_status = "skipped_no_history"
        else:
            priors = self._llm_vibes(obs)  # sets _llm_prior_status / _llm_reasoning / _llm_error

        # Track context for the decision log
        self._num_answered = sum(1 for qa in obs.qa_history if qa.answer is not None)
        self._llm_prior_used = priors is not None
        # As in GRAIL, the LLM pseudo-prior is used for this inference only;
        # the next call starts from uniform priors again (agent_acl.py:531-538).
        self.latest_beliefs = self.graph_model.predict_probs(obs.qa_history, priors=priors, \
            algorithm=self.algorithm)
        if self.print_out:
            dist = self.graph_model.princess_distribution(self.latest_beliefs)
            print(f"[GrailPrince] P(Princess): { {k: round(v, 3) for k, v in sorted(dist.items())} }")

    def _llm_vibes(self, obs: PrinceObservation) -> dict[int, np.ndarray] | None:
        """One-shot LLM prior adjustment, GRAIL-style.

        GRAIL maps increase/decrease to fixed pseudo-priors over the binary
        good/evil variable (0.6/0.4, or 0.75/0.25 late game). The Tofu role
        variable is multiple dimension, so the same idea is expressed as reweighting the
        Princess weight of the player's prior by a fixed factor.
        """
        current = self.graph_model.predict_probs(obs.qa_history, algorithm=self.algorithm)
        dist = self.graph_model.princess_distribution(current)

        prompt = build_vibes_prompt(
            obs=obs,
            player_ids=self.graph_model.player_ids,
            princess_dist=dist,
            has_center=CENTER_ID in self.graph_model.player_ids,
        )

        try:
            response, reasoning = self.llm_client.chat_structured_model(prompt, VibesResponse, \
                reasoning_effort=self.vibes_reasoning_effort)
        except Exception as e:
            # Vibes are advisory -> fall back to uniform priors, but make the failure
            # VISIBLE (status + message + print) instead of swallowing it silently, so a
            # `false` llm_prior_used can be told apart from an all-"same" response.
            self._llm_prior_status = "llm_error"
            self._llm_error = f"{type(e).__name__}: {e}"
            if self.print_out:
                print(f"[GrailPrince] vibes LLM call FAILED -> uniform priors: {self._llm_error}")
            return None

        self._llm_reasoning = reasoning
        if self.print_out:
            print(f"{'='*60}")
            print(f"[GrailPrince] vibes LLM response")
            if reasoning:
                print(f"Thinking: {reasoning}\n")
            print(f"Adjustments: {[(a.player, a.direction) for a in response.adjustments]}")
            stats = self.llm_client.last_call_stats
            if stats:
                print(f"Tokens: {stats.prompt_tokens} prompt + {stats.completion_tokens} completion"
                    f" = {stats.total_tokens} total (seed={stats.seed_used})")

        num_of_roles = self.graph_model.num_of_roles
        princess_idx = self.graph_model.role_index["Princess Tofu"]
        priors: dict[int, np.ndarray] = {}
        late_game = len([qa for qa in obs.qa_history if qa.answer]) >= self.num_players - 1
        boost = 3.0 if late_game else 1.5  # analogue of GRAIL's 0.75/0.25 vs 0.6/0.4
        for a in response.adjustments:
            pid, adj = a.player, a.direction
            # 'same' (and any unknown player) is left out of `priors`, so _build_graph
            # keeps it at the uniform prior -- i.e. no-op, exactly what 'same' means.
            if pid not in self.graph_model.player_ids or adj not in ("increase", "decrease"):
                continue
            p = np.full(num_of_roles, 1.0 / num_of_roles)
            p[princess_idx] *= boost if adj == "increase" else 1.0 / boost
            priors[pid] = p / p.sum()
        # applied = at least one nudge took effect; all_same = LLM answered but advised
        # no change (empty priors -> uniform -> logged distinctly from an llm_error).
        self._llm_prior_status = "applied" if priors else "all_same"
        return priors or None


    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        if not isinstance(obs, PrinceObservation):
            raise TypeError("GrailPrinceAgent expects PrinceObservation")

        self._update_beliefs(obs)

        if obs.phase == Phase.QUESTIONING:
            return self._question_belief_greedy(valid_actions)
        elif obs.phase == Phase.FINAL_QUESTION:
            return self._final_question_greedy(valid_actions)
        elif obs.phase == Phase.GUESSING:
            return self._guess_map(valid_actions)
        raise Exception("Unexpected phase: " + str(obs.phase))

    def _question_belief_greedy(self, valid_actions: list[dict]) -> dict:
        """Belief-greedy questioning (the faithful GRAIL transliteration).

        GRAIL's HeuristicOracle never optimises information; it simply ACTS ON
        THE ARGMAX of its current belief (propose the top-k most-trusted party;
        assassinate the argmax-evil player). The direct Tofu analogue is: direct
        the question at the current MAP Princess candidate and ask the player with
        the highest belief to be Princess -- ask it "Who are you?" (Q2).
        """
        dist = self.graph_model.princess_distribution(self.latest_beliefs)
        askable_targets = {a["target"] for a in valid_actions}
        # rank askable_targets targets by current P(Princess); take the leading hypothesis
        for target in sorted((t for t in dist if t in askable_targets), key=dist.get, reverse=True):
            q2 = [a for a in valid_actions if a["target"] == target \
                and QuestionType(a["question_type"]) == QuestionType.Q2]
            
            if q2:
                self._log("questioning", q2[0])
                return q2[0]
            same_target = [a for a in valid_actions if a["target"] == target]
            if same_target: # if no Q2 valid for this player, ask any question if exists
                self._log("questioning", same_target[0])
                return same_target[0]
            
        # action = valid_actions[0] # fallback: pick first valid action if no MAP target found (it should not happen!)
        # self._log("questioning", action)
        # return action

    def _final_question_greedy(self, valid_actions: list[dict]) -> dict:
        """Belief-greedy final question: ask the current MAP princess candidate
        'Where is Princess Tofu?' (analogue of GRAIL proposing the party it
        currently believes best, not the most informative one)."""
        dist = self.graph_model.princess_distribution(self.latest_beliefs)
        ranked = sorted(dist, key=dist.get, reverse=True)
        for target in ranked:
            for a in valid_actions:
                if a["target"] == target and QuestionType(a["question_type"]) == QuestionType.Q1:
                    self._log("final_question", a)
                    return a
        # action = valid_actions[0]
        # self._log("final_question", action)
        # return action

    def _guess_map(self, valid_actions: list[dict]) -> dict:
        """Guess = argmax max-marginal P(Princess) (analogue of GRAIL's
        chose_assassin_target: argmax belief)."""
        dist = self.graph_model.princess_distribution(self.latest_beliefs)
        valid_targets = {a["target"] for a in valid_actions}
        best = max((player_id for player_id in dist if player_id in valid_targets), key=dist.get)
        for a in valid_actions:
            if a["target"] == best:
                entry = self._belief_snapshot("guessing")
                entry["guess"] = best
                self.decision_log.append(entry)
                return a
        # return valid_actions[0]

    def _belief_snapshot(self, phase: str) -> dict:
        """Per-step belief record for offline analysis (the GRAIL analogue of the
        Symbolic Prince's per-step num_worlds/entropy log). Keeps the full
        per-entity role beliefs -- not just the marginal Princess distribution --
        so the whole belief trajectory can be reconstructed after the run."""
        dist = self.graph_model.princess_distribution(self.latest_beliefs)
        entry = {
            "phase": phase,
            "num_answered": self._num_answered,
            "llm_prior_used": self._llm_prior_used,
            # Why the prior was / wasn't applied (disambiguates a `false` above):
            # disabled / skipped_no_history / llm_error / all_same / applied.
            "llm_prior_status": self._llm_prior_status,
            "princess_dist": {str(k): round(v, 4) for k, v in dist.items()},
            "beliefs": {str(p): {r: round(b, 4) for r, b in roles.items()}
                        for p, roles in (self.latest_beliefs or {}).items()},
        }
        # Only attach the LLM debug fields when present, to avoid noise on non-LLM steps.
        if self._llm_error is not None:
            entry["llm_error"] = self._llm_error
        if self._llm_reasoning is not None:
            entry["llm_reasoning"] = self._llm_reasoning
        return entry

    def _log(self, phase: str, action: dict) -> None:
        entry = self._belief_snapshot(phase)
        entry["action_chosen"] = action
        self.decision_log.append(entry)
