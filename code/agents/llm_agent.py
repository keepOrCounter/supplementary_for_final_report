"""Baseline LLM agent: picks actions via a structured LLM call (no explicit belief state)."""
from __future__ import annotations

from game.predefined_data_classes import Observation, Role
from agents.base import BaseAgent, LLMParseError
from agents.prompts import build_prompt
from core.llm_client import LLMClient
from core.helper_functions import SeedSource


class LLMAgent(BaseAgent):
    """Calls llm_client to get actions"""

    def __init__(self, player_id: int, role: Role, llm_client: LLMClient=None, \
        use_reasoning_history: bool = True, print_out: bool = False, seed_source: SeedSource = None):

        super().__init__(player_id, role)
        self.llm_client = llm_client
        self.use_reasoning_history = use_reasoning_history
        self.print_out = print_out
        # Independent per-agent seed stream for LLM calls (see seeding.SeedSource).
        self._seed_source = seed_source or SeedSource(None)

        # Full reasoning from model CoT: saved for logging
        self.reasoning_history: list[str] = []
        
        # Iterative summary: single string updated each turn, fed back into prompt
        self.current_summary: str = ""
        # History of all summaries for logging
        self.summary_history: list[str] = []

    def _next_llm_seed(self) -> int | None:
        """Fresh seed for this LLM call, or None if this agent is seedless."""
        return self._seed_source.next_seed()

    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        if self.llm_client is None:
            raise RuntimeError(f"{self.agent_name}: llm_client is not assigned!!!")
        
        if len(valid_actions) == 1:
            # Only one valid action, no need to call LLM
            return valid_actions[0]

        # Pass current iterative summary (single string, not a list)
        previous_summary = self.current_summary if self.use_reasoning_history else ""

        system_prompt, user_prompt = build_prompt(obs, valid_actions, previous_summary=previous_summary)
        
        if self.print_out:
            print(f"\n{'~'*60}")
            print(f"[{self.agent_name}] LLM Prompt")
            print(f"{'~'*60}")
            print("System Prompt:")
            print(system_prompt)
            print(f"\n{'-'*60}")
            print("User Prompt:")
            print(user_prompt)
            print(f"{'~'*60}")
        
        max_index = len(valid_actions) - 1
        call_seed = self._next_llm_seed()

        try:
            choice, reasoning = self.llm_client.chat_structured(user_prompt, system_prompt=system_prompt, \
                max_index=max_index, seed=call_seed)

            # Store full reasoning (thinking content) for logging
            if reasoning:
                self.reasoning_history.append(reasoning)

            # Update iterative summary (replaces previous, not appends)
            if choice.reasoning_summary:
                self.current_summary = choice.reasoning_summary
                self.summary_history.append(choice.reasoning_summary)

            if self.print_out:
                print(f"{'='*60}")
                print(f"\n[{self.agent_name}] LLM Response")
                if reasoning:
                    print(f"Thinking: {reasoning}")
                    print()
                print(f"Summary: {choice.reasoning_summary}")
                print(f"Action index: {choice.action_index}")
                stats = self.llm_client.last_call_stats
                if stats:
                    print(f"Tokens: {stats.prompt_tokens} prompt + {stats.completion_tokens} completion", \
                        f" = {stats.total_tokens} total (seed={stats.seed_used})")

            idx = choice.action_index
            # idx = max(0, min(idx, max_index)) # clamp to valid range
            return valid_actions[idx]

        except Exception as e:
            # DO NOT fallback silently
            raise LLMParseError(self.agent_name, e)
