"""Generic structured LLM outputs shared by every LLM-driven agent.

`ActionChoice` (and its index-bounded variant from `bounded_action_choice`) is the
default response model of ``LLMClient.chat_structured``: one action index plus an
iterative reasoning summary. Method-specific response models live in each method's
own schemas.py (agents/our_agents, agents/grail, agents/emo).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ActionChoice(BaseModel):
    """structured response from LLM used by instructor"""
    reasoning_summary: str = Field(description="Update your reasoning summary so far: combine your previous reasoning summary"
        "with new deductions of this turn into ONE concise summary (3-6 sentences). "
        "If you are prince: include your current suspicion of each player's role and faction. "
        "If you are not prince: summarize how your answers so far help your faction to win, and how this will guide your response if you are later asked in the Final Question phase.")
    action_index: int = Field(description="The index number of your chosen action from the list.")


def bounded_action_choice(max_index: int) -> type[BaseModel]:
    """create ActionChoice model with action_index bounded to [0, max_index]"""

    class BoundedActionChoice(BaseModel):
        reasoning_summary: str = Field(description="Update your reasoning summary so far: combine your previous reasoning summary"
            "with new deductions of this turn into ONE concise summary (3-6 sentences). "
            "If you are prince: include your current suspicion of each player's role and faction. "
            "If you are not prince: summarize how your answers so far help your faction to win, and how this will guide your response if you are later asked in the Final Question phase.")
        action_index: int = Field(ge=0, le=max_index, \
            description=f"The index number of your chosen action (must be between 0 and {max_index})")
            # choosed index between 0 and max_index inclusive
            
            
    return BoundedActionChoice
