"""Structured LLM outputs for our method's soft-likelihood judge.

``LikelihoodJudgment`` is what ``NeuroSymbolicKripkeState._judge`` asks the LLM for
(one ordinal ``SlotRating`` per outcome-slot plus an overall confidence); see
Kripke.NeuroSymbolicKripkeState for how it is mapped to a distribution.
``JudgeAnalysis`` is the reasoning pass of the manual-CoT judge (ablation.judge_cot);
``OpponentProfile`` is the cross-game profiler's output (opponent_memory.OpponentProfiler).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SlotRating(BaseModel):
    """One ordinal rating for a single outcome-slot (used by the soft-likelihood judge)."""
    
    label: Literal["PRINCESS", "QUEEN", "OTHER"] = Field(description="The outcome-slot label being rated (one of the labels you were shown).")
    rating: Literal["LIKELY", "NEUTRAL", "UNLIKELY"] = Field(description="How likely the answerer is to give an answer of this kind.")


# TODO LikelihoodJudgment docstring leakage!!!
class LikelihoodJudgment(BaseModel):
    """Structured response for the Option A soft-likelihood judge.

    See Kripke.NeuroSymbolicKripkeState for how this is mapped to a distribution.
    """
    reasoning: str = Field(description="One or two sentences of strategic rationale.")
    ratings: list[SlotRating] = Field(description="One rating per outcome-slot label shown.")
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Overall confidence in these ratings.")


# NB: instructor shows response-model docstrings to the LLM as the tool-schema
# description, so the docstrings of the two models below are written as intentional
# instructions, not internal notes.

class JudgeAnalysis(BaseModel):
    """Your step-by-step reasoning about the situation. Do not give any rating here."""
    analysis: str = Field(description="Step-by-step strategic reasoning in 3-6 short sentences, no ratings.")


class OpponentProfile(BaseModel):
    """Your updated profile of this player."""
    profile: str = Field(description="The rewritten profile: answering tendencies per constraint, "
        "consistency, and skill level (novice / intermediate / expert). At most 6 sentences.")
