"""Atomic opponent models and the central validator (EMO's two components).

Paper: Yu, Zhang & Lu, "LLM-Based Explicit Models of Opponents for Multi-Agent
Games" (NAACL 2025). This module implements the LLM steps of the bi-level
feedback-refinement framework; the ``EMOPrinceAgent`` in ``agents/emo/prince_agent.py`` drives
them through Algorithm 1 (per-opponent update) and Algorithm 2 (CoT + EMO).

Every step is a structured ``LLMClient`` call.
"""
from __future__ import annotations

from game.predefined_data_classes import PrinceObservation
from core.helper_functions import SeedSource
from core.llm_client import LLMClient

from agents.emo.schemas import RolePrediction, FeedbackResponse, SummaryResponse
from agents.emo import prompts


class AtomicOpponentModel:
    """LLM-driven model of a single player's hidden role (the paper's M_j).

    AtomicOpponentModel: one memory-enhanced self-refine model per player 
        (Eq. 1 initialize, Eq. 2 self-feedback,Eq. 3 self-refine, Eq. 5 global-refine).
        
    Holds this player's running prediction (q_hat_j), its prediction history.
    
    Attributes:
        ``num_processed``: how many of the player's answers have been folded in, so the agent can tell when a new answer (a_{j,t}) needs an update.
        ``prediction_trace``: = H_j, the past committed predictions fed back into every
    prompt
    """

    def __init__(self, player_id: int, llm_client: LLMClient, seed_source: SeedSource = None, \
        print_out: bool = False, reasoning_effort: str = "none"):
        self.player_id = player_id
        self.llm_client = llm_client
        self._seeds = seed_source or SeedSource(None)
        self.print_out = print_out

        self.reasoning_effort = reasoning_effort # opponent sub-calls skip the thinking mode by default
        self.system_prompt = prompts.build_opponent_system(player_id)  # OPPONENT_ROLE

        self.prediction: RolePrediction | None = None
        self.num_processed: int = 0 # answers by this player already modelled
        self.prediction_trace: list[dict] = [] # H_j: committed predictions, for prompts + logging
        self.reasoning_log: list[dict] = [] # per sub-call: {step, thinking, output}

    def _LLM_reasoning_log(self, step: str, thinking: str | None, output: str) -> None:
        self.reasoning_log.append({"step": step, "thinking": thinking, "output": output})
        if self.print_out:
            print("~" * 60)
            print(f" [EMO model P{self.player_id} | {step}] -> {output}")
            if thinking:
                print(f"  thinking: {thinking}")

    def _predict(self, user_prompt: str, step: str) -> RolePrediction:
        pred, thinking = self.llm_client.chat_structured_model(user_prompt, RolePrediction, \
            system_prompt=self.system_prompt, seed=self._seeds.next_seed(), \
                reasoning_effort=self.reasoning_effort)
        self._LLM_reasoning_log(step, thinking, prompts.role_prediction_format(pred))
        return pred


    def initialize(self, obs: PrinceObservation) -> RolePrediction:
        """Eq. 1: first prediction q_hat^0 from the public record + H_j."""
        return self._predict(prompts.build_init_prompt(obs, self.player_id, self.prediction_trace), "init")

    def self_feedback(self, obs: PrinceObservation, pred: RolePrediction) -> str:
        """Eq. 2: self-feedback F^s on the current prediction."""
        resp, thinking = self.llm_client.chat_structured_model(\
            prompts.build_self_feedback_prompt(obs, self.player_id, pred, self.prediction_trace),\
                FeedbackResponse, system_prompt=self.system_prompt, seed=self._seeds.next_seed(), \
                    reasoning_effort=self.reasoning_effort)
        self._LLM_reasoning_log("self_feedback", thinking, resp.feedback)
        return resp.feedback

    def self_refine(self, obs: PrinceObservation, pred: RolePrediction, feedback: str) -> RolePrediction:
        """Eq. 3: refined prediction q_hat^{s+1} incorporating self-feedback."""
        return self._predict(\
            prompts.build_self_refine_prompt(obs, self.player_id, pred, feedback, self.prediction_trace), "self_refine")

    def global_refine(self, obs: PrinceObservation, pred: RolePrediction, global_feedback: str) -> RolePrediction:
        """Eq. 5: refined prediction q_hat^{g+1} incorporating the validator's feedback."""
        return self._predict(\
            prompts.build_global_refine_prompt(obs, self.player_id, pred, global_feedback, self.prediction_trace), \
                "global_refine")

    def commit(self, pred: RolePrediction) -> None:
        """Store the finished prediction (merge q_hat_{j,t} into H_j) for logging + reuse."""
        self.prediction = pred
        self.num_processed += 1
        self.prediction_trace.append({
            "after_answers": self.num_processed,
            "faction": pred.faction,
            "role": pred.role,
            "rationale": pred.rationale,
        })
        if self.print_out:
            print(f" [EMO model P{self.player_id} | commit] -> {prompts.role_prediction_format(pred)}")


class CentralValidator:
    """Coordinates the atomic models to enforce global consistency (the paper's M_c).
    
    CentralValidator: global consistency across the atomic models (Eq. 4 global-feedback, Eq. 6 summary).
    """

    def __init__(self, llm_client: LLMClient, seed_source: SeedSource = None, print_out: bool = False, \
        reasoning_effort: str = "none"):
        self.llm_client = llm_client
        self._seeds = seed_source or SeedSource(None)
        self.print_out = print_out
        self.reasoning_effort = reasoning_effort # validator sub-calls skip the thinking mode by default
        self.reasoning_log: list[dict] = []  # per sub-call: {step, thinking, output}

    def _LLM_reasoning_log(self, step: str, thinking: str | None, output: str) -> None:
        self.reasoning_log.append({"step": step, "thinking": thinking, "output": output})
        if self.print_out:
            print(f" [EMO validator | {step}] -> {output}")
            if thinking:
                print(f"  thinking: {thinking}")

    def global_feedback(self, obs: PrinceObservation, player_id: int, pred: RolePrediction, \
        all_predictions: dict[int, RolePrediction]) -> str:
        """Eq. 4: global feedback F^g for the ONE model under review, given all the others."""
        resp, thinking = self.llm_client.chat_structured_model(\
            prompts.build_global_feedback_prompt(obs, player_id, pred, all_predictions),\
                FeedbackResponse, system_prompt=prompts.VALIDATOR_SYSTEM, seed=self._seeds.next_seed(), \
                    reasoning_effort=self.reasoning_effort)
        self._LLM_reasoning_log(f"global_feedback(P{player_id})", thinking, resp.feedback)
        return resp.feedback

    def summarize(self, obs: PrinceObservation, all_predictions: dict[int, RolePrediction]) -> str:
        """Eq. 6: aggregate all atomic predictions into q_hat_{-i} for the reasoning module."""
        resp, thinking = self.llm_client.chat_structured_model(\
            prompts.build_summary_prompt(obs, all_predictions), SummaryResponse, \
                system_prompt=prompts.VALIDATOR_SYSTEM, seed=self._seeds.next_seed(), \
                    reasoning_effort=self.reasoning_effort)
        self._LLM_reasoning_log("summary", thinking, resp.summary)
        return resp.summary
