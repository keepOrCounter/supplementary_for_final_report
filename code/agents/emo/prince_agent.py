"""EMO-style Prince agent: drives the atomic opponent models and central validator
through the paper's Algorithm 1 (per-opponent update) and Algorithm 2 (CoT + EMO)."""
from __future__ import annotations

from game.predefined_data_classes import Observation, PrinceObservation, Role, Phase
from agents.base import BaseAgent, LLMParseError
from agents.emo.opponent_model import AtomicOpponentModel, CentralValidator
from agents.emo.prompts import build_reasoning_prompt
from core.llm_client import LLMClient
from core.helper_functions import action_description, SeedSource


class EMOPrinceAgent(BaseAgent):
    """EMO-style Prince baseline (ported from Yu, Zhang & Lu, NAACL 2025:
    "LLM-Based Explicit Models of Opponents for Multi-Agent Games").

    Mapping to the paper: the Prince is agent i; each non-prince player is an
    opponent j whose hidden Role is modelled by its own Atomic Opponent Model
    (M_j). A Central Validator (M_c) enforces global consistency across those
    models and summarises them for a CoT reasoning module that selects the
    Prince's action. Because each player answers only when the Prince questions
    them, an opponent "acts" (a_{j,t}) exactly when a new answer appears in the
    transcript; ``_sync_models`` runs EMO's Algorithm 1 for each such new answer
    before the Prince reasons.
    """

    def __init__(self, player_id: int, role: Role, num_players: int, use_center_card: bool, llm_client: LLMClient, \
        self_refine_steps: int = 2, global_refine_steps: int = 2, use_reasoning_history: bool = True, \
            model_reasoning_effort: str = "none", print_out: bool = False, seed_source: SeedSource = None):
        super().__init__(player_id, role)
        self.llm_client = llm_client
        # Iteration counts for the two refinement loops. The paper fixes both to 2
        # (Appendix A); exposed here so cost can be tuned.
        self.self_refine_steps = self_refine_steps
        self.global_refine_steps = global_refine_steps
        self.use_reasoning_history = use_reasoning_history
        self.print_out = print_out

        # Independent seed sub-streams: reasoning calls, the validator, and one per atomic model, so one component's draw
        # count never shifts another's.
        src = seed_source or SeedSource(None)
        self._reason_seeds = src.spawn(0)
        # Same with the paper, default to reasoning_effort="none" in EMO.
        self.validator = CentralValidator(llm_client, seed_source=src.spawn(1), print_out=print_out, \
            reasoning_effort=model_reasoning_effort)
        self._models: dict[int, AtomicOpponentModel] = {
            pid: AtomicOpponentModel(pid, llm_client, seed_source=src.spawn(pid + 1), print_out=print_out, \
                reasoning_effort=model_reasoning_effort)
            for pid in range(1, num_players)
        }

        # Reasoning history (same shape as LLMAgent, for comparable logging).
        self.reasoning_history: list[str] = []
        self.current_summary: str = ""
        self.summary_history: list[str] = []
        self.decision_log: list[dict] = []

    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        if not isinstance(obs, PrinceObservation):
            raise TypeError("EMOPrinceAgent expects PrinceObservation")
        if self.llm_client is None:
            raise RuntimeError(f"{self.agent_name}: llm_client is not assigned!!!")
        if obs.phase not in (Phase.QUESTIONING, Phase.FINAL_QUESTION, Phase.GUESSING):
            raise Exception("Unexpected phase: " + str(obs.phase))

        try:
            # Fold any newly-answered players into their opponent models (Algorithm 1),
            # then run the CoT reasoning module (Algorithm 2).
            self._sync_models(obs)
            if len(valid_actions) == 1:
                return valid_actions[0]
            return self._reason_and_act(obs, valid_actions)
        except Exception as e:
            # Any LLM parse/inference failure in a model / validator / reasoning call
            # will cause the runner to skip just this game.
            raise LLMParseError(self.agent_name, e)

    def _sync_models(self, obs: PrinceObservation) -> None:
        """Run EMO (Algorithm 1) for each of a player's answers not yet modelled.

        A player can answer up to twice (once in QUESTIONING, maybe once in the
        FINAL_QUESTION), and each answer is a fresh 'action' to model. Answers for
        a given player are chronological in the transcript, so anything past
        ``num_processed`` is new.
        """
        answers_by_player: dict[int, int] = {}
        for qa in obs.qa_history:
            if qa.answer is not None:
                answers_by_player[qa.target_id] = answers_by_player.get(qa.target_id, 0) + 1

        for pid, count in answers_by_player.items():
            model = self._models.get(pid)
            if model is None:
                continue
            while model.num_processed < count:
                self._run_emo_for_player(pid, obs)

    def _run_emo_for_player(self, pid: int, obs: PrinceObservation) -> None:
        """EMO's Algorithm 1 for one opponent: self-refine then global-refine."""
        model = self._models[pid]
        if self.print_out:
            print(f"  [EMO] updating opponent model for Player {pid} ", \
                f"(S={self.self_refine_steps}, G={self.global_refine_steps})")

        # 1. Self-Refine (Eqs. 1-3)
        pred = model.initialize(obs) # 1.1 initialization
        for _ in range(self.self_refine_steps):
            feedback = model.self_feedback(obs, pred) # 1.2 self-feedback phase
            pred = model.self_refine(obs, pred, feedback) # 1.3 self-refinement phase

        # 2. Global-Refine (Eqs. 4-5): validator checks this model against the others.
        for _ in range(self.global_refine_steps):
            global_feedback = self.validator.global_feedback(obs, pid, pred, self._current_predictions(pid, pred))
            # 2.1 global-feedback phase
            
            pred = model.global_refine(obs, pred, global_feedback) # 2.2 global-refinement phase

        model.commit(pred)

    def _current_predictions(self, current_pid: int = None, current_pred = None) -> dict[int, object]:
        """All committed predictions, overlaid with the in-progress one for ``current_pid``."""
        preds = {pid: m.prediction for pid, m in self._models.items() if m.prediction is not None}
        if current_pred is not None and current_pid is not None:
            preds[current_pid] = current_pred
        return preds

    def _reason_and_act(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """CoT reasoning module (Algorithm 2): summarise the models, then decide."""
        committed = self._current_predictions()
        summary = self.validator.summarize(obs, committed) if committed else ""  # Eq. 6

        previous_summary = self.current_summary if self.use_reasoning_history else ""
        system_prompt, user_prompt = build_reasoning_prompt(
            obs, valid_actions, opponent_summary=summary, previous_summary=previous_summary)

        if self.print_out:
            print(f"\n{'~'*60}\n[{self.agent_name}] EMO reasoning prompt\n{'~'*60}")
            print("System Prompt:\n" + system_prompt)
            print(f"\n{'-'*60}\nUser Prompt:\n" + user_prompt + f"\n{'~'*60}")

        # The final CoT selection (Algorithm 2 in paper) keeps the client's configured reasoning_effort.
        max_index = len(valid_actions) - 1
        choice, reasoning = self.llm_client.chat_structured(user_prompt, system_prompt=system_prompt, \
            max_index=max_index, seed=self._reason_seeds.next_seed())

        if reasoning:
            self.reasoning_history.append(reasoning)
        if choice.reasoning_summary:
            self.current_summary = choice.reasoning_summary
            self.summary_history.append(choice.reasoning_summary)

        if self.print_out:
            print(f"{'='*60}\n[{self.agent_name}] EMO decision")
            print(f"Opponent summary: {summary}")
            if reasoning:
                print(f"Thinking: {reasoning}")
            print(f"Summary: {choice.reasoning_summary}")
            print(f"Action index: {choice.action_index}")

        # idx = max(0, min(choice.action_index, max_index))
        idx = choice.action_index
        action = valid_actions[idx]

        self.decision_log.append({
            "phase": obs.phase.value,
            "num_models": len(committed),
            "opponent_summary": summary,
            "action_chosen": action_description(action),
        })
        return action
