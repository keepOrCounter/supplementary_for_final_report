# [Resistance] structured LLM outputs for our method's likelihood judge
#
# Written to match agents/our_agents/schemas.py, which does the same for Tofu
# Kingdom's judge; the code is this game's. The options a judge rates change
# from question to question, so each response model is built from them, one
# field per option. pydantic then refuses an answer that skips an option or
# adds one, and instructor asks again with the reason.
#
# NB: instructor shows a response model's docstring to the model as the
# tool-schema description, so JudgeAnalysis's docstring is an instruction and
# not a note.
from __future__ import annotations
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, create_model

Rating = Literal["LIKELY", "NEUTRAL", "UNLIKELY"]
Confidence = Literal["LOW", "MEDIUM", "HIGH"]
Suspicion = Literal["SUSPECT", "UNSURE", "TRUSTED"]


@lru_cache(maxsize=None)
def judgment_model(options: tuple[str, ...]) -> type[BaseModel]:
    """one rating per option, then one overall confidence"""
    fields = {"reasoning": (str, Field(description="One or two sentences of strategic rationale."))}
    for option in options:
        fields[option] = (Rating, Field(description=f"How likely this player would want to take {option}."))
    fields["confidence"] = (Confidence, Field(description="Overall confidence in these ratings."))
    return create_model("LikelihoodJudgment", **fields)


@lru_cache(maxsize=None)
def suspicion_model(num_players: int) -> type[BaseModel]:
    """one suspicion rating per player, read from the public record"""
    fields = {"reasoning": (str, Field(description="One or two sentences on what the record shows."))}
    for player_id in range(num_players):
        fields[f"player_{player_id}"] = (Suspicion, Field(description=f"How trustworthy Player {player_id} looks, i.e. how likely it is not a spy."))
    return create_model("SuspicionRatings", **fields)


class JudgeAnalysis(BaseModel):
    """Your step-by-step reasoning about the situation. Do not give any rating here."""
    analysis: str = Field(description="Step-by-step strategic reasoning in 3-6 short sentences, no ratings.")
