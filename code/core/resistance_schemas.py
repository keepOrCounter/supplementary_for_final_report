# structured LLM outputs for The Resistance, one response model per decision
# core/schemas.py is the same thing for Tofu Kingdom, and both are named after
# the game the way core/logger.py and core/resistance_logger.py are.
#
# The Tofu agents answer with an index into valid_actions (ActionChoice). That
# does not carry over: the propose menu is every team of the required size, 20
# entries in a six player game, and a small model reads "send players 0, 2 and
# 5" better than it reads the twentieth line of a list of near identical ones.
# So each decision returns the decision itself and the menu never reaches the
# prompt.
#
# Every limit is declared on the field, so pydantic refuses an impossible answer
# and instructor asks again with the reason: the range on each id and how many
# there must be come from the field, and "no id twice" from a validator.
#
# The lists are list and not set. instructor validates the tool call arguments
# in pydantic's STRICT mode, and strict mode will not turn a JSON array into a
# set, so every answer was refused with "Input should be a valid array" however
# well formed it was. The model even started sending the string "{4}" trying to
# give us a set. uniqueItems stays in the schema through json_schema_extra, so
# the model is still told the ids must differ.
#
# The models are built with create_model rather than a class body, because the
# player id type is computed from the table size. In a class body that reads as
# `team: set[_player_id(n)]`, and a call inside a type annotation is not valid
# typing even though pydantic accepts it. create_model takes the type as a
# value, so the type checker has nothing to complain about.
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, create_model, field_validator

# The first two sentences are Tofu Kingdom's ActionChoice.reasoning_summary word
# for word, so the two games ask for the same thing and their traces compare.
# The per faction split is Tofu's too: it tells the prince and a non prince
# player what to put in the summary, and this tells each faction here.
REASONING_DESCRIPTION = ("Update your reasoning summary so far: combine your previous reasoning "
    "summary with new deductions of this turn into ONE concise summary (3-6 sentences). "
    "If you are in the Resistance faction: say which players you trust, which players you think "
    "are spies, and what made you think so. "
    "If you are in the Spy faction: say how the votes and the missions so far help your faction "
    "win, and how that guides your next move.")


def _player_id(num_players: int):
    """the type of one player id, checked against the table size"""
    return Annotated[int, Field(ge=0, le=num_players - 1)]


def _no_duplicates(field_name: str):
    """reject a repeated player id

    A list keeps the repeat, so without this the environment would be handed a
    team that names one player twice.
    """
    def check(cls, value):
        if isinstance(value, list) and len(value) != len(set(value)):
            raise ValueError("name each player id at most once")
        return value

    return field_validator(field_name, mode="before")(classmethod(check))


def _decision_base(num_players: int, num_spies: int) -> type[BaseModel]:
    """the two fields every decision carries

    spy_suspects is asked on every decision, a spy's own included, where the
    answer is simply the players it was shown at the start. One shape for every
    decision keeps one code path, and the runner only scores resistance players.
    """
    suspects_description = (f"The player ids you think are in the Spy faction. This game has "
        f"{num_spies} spies, so name at most {num_spies} players. Name fewer only if you "
        "cannot tell.")

    return create_model("Decision", reasoning_summary=(str, Field(description=REASONING_DESCRIPTION)),
        spy_suspects=(list[_player_id(num_players)], Field(max_length=num_spies, \
            description=suspects_description, json_schema_extra={"uniqueItems": True})),
        __validators__={"_suspects_no_duplicates": _no_duplicates("spy_suspects")})


# The reasoning pass of a manual chain of thought (ablation decision_cot). It
# carries only the reasoning: the decision comes back on a second call that is
# given this analysis, and asking for the two separately is what bounds the
# reasoning to the length the prompt asks for. It takes no table size, because
# it holds no player id.
#
# NB: instructor shows a response model's docstring to the model as the
# tool-schema description, so the docstring below is written as an intentional
# instruction and not as a note. A note there cost a whole run: the model
# answered the reasoning prompt in prose, never called the tool, and every retry
# did the same. The Tofu analogue is JudgeAnalysis, which carries the same NB.

class DecisionAnalysis(BaseModel):
    """Your step-by-step reasoning about this decision. Do not make the decision here."""
    analysis: str = Field(description="Step-by-step strategic reasoning in 4-8 short "
        "sentences, no decision.")


def final_guess_model(num_players: int, num_spies: int) -> type[BaseModel]:
    """the closing guess, asked once after the game is over

    Only the two fields every decision already carries. There is no action left
    to take, and spy_suspects is the whole answer.
    """
    return create_model("FinalGuess", __base__=_decision_base(num_players, num_spies))


def team_proposal_model(num_players: int, num_spies: int, team_size: int) -> type[BaseModel]:
    """the leader's mission team

    Any team of team_size different player ids is a legal proposal, because the
    environment offers every one of them, so the field limits are the whole
    check.
    """
    return create_model("TeamProposal", __base__=_decision_base(num_players, num_spies),
        team=(list[_player_id(num_players)], Field(min_length=team_size, max_length=team_size, \
            description=f"The {team_size} player ids you send on this mission. You may include yourself.", \
            json_schema_extra={"uniqueItems": True})),
        __validators__={"_team_no_duplicates": _no_duplicates("team")})


def team_vote_model(num_players: int, num_spies: int) -> type[BaseModel]:
    """one player's vote on the proposed team"""
    return create_model("TeamVote", __base__=_decision_base(num_players, num_spies),
        approve=(bool, Field(description="True to approve this mission team, False to reject it.")))


def mission_card_model(num_players: int, num_spies: int) -> type[BaseModel]:
    """one team member's mission card"""
    return create_model("MissionCard", __base__=_decision_base(num_players, num_spies),
        play_success=(bool, Field(description="True to play the Success card, False to play the "
            "Fail card. Fail cards make the mission fail.")))
