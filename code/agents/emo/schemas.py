"""Structured LLM outputs for the EMO baseline.

All EMO LLM calls go through ``LLMClient.chat_structured_model`` (instructor),
so every step -- prediction, feedback, refinement, summary -- returns one of the
pydantic models below. Routing every call through instructor is deliberate: it
keeps EMO's (substantial) token usage inside the same cumulative-stats hooks the
rest of the project relies on for cost accounting.

The atomic-model output ``RolePrediction`` follows the paper's Avalon format
``['allegiance', role]`` (Appendix D): the hidden variable in Tofu Kingdom is a
player's Role, so we emit a coarse ``faction`` plus a single most-likely specific
``role``.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The seven canonical Tofu Kingdom roles (game.predefined_data_classes.ROLES) plus
# "unknown". Constraining the role output to this closed set: stops the model from inventing role names, 
# the analogue of the fixed card/role vocabulary the paper's outputs draw from.
RoleName = Literal["Princess Tofu", "Tofu Chef", "Queen Tofu", "Tofu Minister", "Tofu Guard", "Tofu Maid", \
    "Pudding Spy", "unknown"]


class RolePrediction(BaseModel):
    """An atomic opponent model's belief about one player's hidden role.
    """

    faction: Literal["princess", "queen", "other"] = Field(
        description="Predicted faction: 'princess', 'queen', or 'other'. Always give your best guess.")
    role: RoleName = Field(description="Single most-likely specific role (exact name), consistent with the faction. "\
        "Use 'unknown' only when the answers pin down the faction but not the exact role.")
    rationale: str = Field(description="ONE short sentence (at most 25 words): why this player's answer(s) imply "\
        "this role, given the faction truth/lie constraint. Do not restate the rules or enumerate other players.")


class FeedbackResponse(BaseModel):
    """Self-feedback or global-feedback critique text."""

    feedback: str = Field(description="At most 4 short sentences (~80 words total): describe the single key issue step-by-step with the "\
        "current prediction and one concrete adjustment. Do NOT walk through every player or restate the rules.")


class SummaryResponse(BaseModel):
    """Central validator's aggregated summary for the reasoning module."""

    summary: str = Field(
        description="ONE short line per player (role + faction), then one line for where Princess Tofu most " \
            "likely is (a player id, or the center card). At most ~10 words per line; no step-by-step " \
                "reasoning, no rule restatement.")
