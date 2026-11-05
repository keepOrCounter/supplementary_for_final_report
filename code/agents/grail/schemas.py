"""Structured LLM outputs for the GRAIL baseline (the 'vibes' pseudo-prior call)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# valid player ids: -1 (center) .. num_players-1 (<= 7 for the max 8-player game)
PlayerId = Annotated[int, Field(ge=-1, le=7)]
Direction = Literal["increase", "decrease", "same"]


# NB: keep developer rationale in `#` comments, NOT docstrings -- pydantic puts a
# class docstring and every Field(description=...) into the JSON schema that instructor
# sends to the model (Mode.TOOLS), but code comments never leak there.
#
# Adjustment/VibesResponse are a list of flat {player, direction} records rather than a
# dict keyed by PlayerId: instructor's TOOLS mode cannot constrain object KEYS, so an
# int-keyed (and negative "-1") map lets the model emit keys like "center" or player
# names that fail validation and trigger retries; a list of records avoids that.
class Adjustment(BaseModel):
    player: PlayerId = Field(
        description="Candidate being judged: a player ID, or -1 for the center card.")
    direction: Direction = Field(
        description="increase / decrease / same for whether this candidate is Princess "
                    "Tofu; use 'same' when the evidence is insufficient.")


class VibesResponse(BaseModel):
    adjustments: list[Adjustment] = Field(description="One judgement for every candidate listed.")
