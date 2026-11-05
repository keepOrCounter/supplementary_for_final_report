# [Resistance] prompts for our method's likelihood judge
#
# Written to match agents/our_agents/prompts.py, which holds Tofu Kingdom's judge
# prompts; the code is this game's. The game rules come from
# agents/resistance/prompts.py.
#
# Three kinds of question:
#   spy          how a spy acts, told how many spies are on the team
#   resistance   how a resistance player acts, told how the other players on the
#                team look from the public record
#   suspicion    how every player looks from the public record
# The first two name no player, so the same question asked about another player
# or in another world is the same text, and it is asked only once.
#
# Every builder returns (system, body, ask). The ask is the closing instruction.
# It is kept apart from the body so that the reasoning pass of judge_cot can be
# sent the body without being told to give the ratings.
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

from agents.resistance.prompts import game_rules, history_lines
from game.resistance_data_classes import MISSIONS_TO_WIN, MissionRecord, ProposalRecord

VOTE_OPTIONS = ("APPROVE", "REJECT")
CARD_OPTIONS = ("SUCCESS", "FAIL")
RATE_ASK = "Rate each option (LIKELY / NEUTRAL / UNLIKELY), then give one overall confidence."
SUSPICION_ASK = "Rate every player: SUSPECT, UNSURE or TRUSTED."


@dataclass(frozen=True)
class Situation:
    """the public facts every spy or resistance question starts from"""
    num_players: int
    successes: int
    failures: int
    team_size: int
    fails_required: int # Fail cards that make this mission fail
    last_chance: bool = False # the last team allowed on this mission


_RATE_OPTIONS = """\
YOUR TASK: the next message gives one situation and its options. Rate EACH option by
how likely this player would want to take it:
    LIKELY = clearly helps this player's faction
    NEUTRAL = neither clearly helps nor hurts it
    UNLIKELY = clearly works against it
Give exactly one rating per option, then one overall confidence (LOW / MEDIUM / HIGH)."""

_SPY = """\
You are NOT a player. Estimate how a rational SPY player acts, so a deduction engine can
weight the possible actions. Every question is about ONE spy player, called "this spy
player" below.
A spy knows who all the spies are. It wants missions to fail. It also wants to stay
hidden, because the others keep a spy they have found off every team. These two
goals often pull in different directions."""

_SPY_CHANNEL = {
    "vote": """\
This spy player votes on a proposed team. The options are APPROVE and REJECT.
If the last team allowed on a mission is rejected, the spies win the game at once.""",

    "proposal": """\
This spy player is the leader and picks the team. Each option is a KIND of team, named
from this spy player's own point of view:
    WITH_ME_k_PARTNERS: this spy player and k other spies are on the team
    WITHOUT_ME_k_PARTNERS: this spy player is not on the team, and k other spies are
Many teams can be of one kind. Rate how likely this spy player would want to put up a
team of each kind.""",

    "mission": """\
This spy player is on the mission team and plays a card. The options are SUCCESS and FAIL.
Everyone learns if the mission failed, but nobody knows who played which card.
After a failed mission, the others suspect everyone who was on the team.""",
}

# What the three suspicion levels mean. Both the question that rates every player
# and the resistance questions that read those ratings say it with these lines.
_LEVELS = """\
    SUSPECT = probably a spy
    UNSURE = could be either
    TRUSTED = probably not a spy"""

_RESISTANCE = """\
You are NOT a player. Estimate how a rational RESISTANCE player acts, so a deduction
engine can weight the possible actions. Every question is about ONE resistance player,
called "this resistance player" below.
A resistance player does not know who the spies are, but it knows it is not a spy
itself. It wants missions to succeed, so it wants teams without spies.
You are told how trustworthy the other players look from the public record:
""" + _LEVELS

_RESISTANCE_CHANNEL = {
    "vote": """\
This resistance player votes on a proposed team. The options are APPROVE and REJECT.
If the last team allowed on a mission is rejected, the spies win the game at once.""",

    "proposal": """\
This resistance player is the leader and picks the team. Each option is a KIND of team.
The number in front of each word counts the other players of that kind on the team:
    WITH_ME_1SUSPECT_0UNSURE_2TRUSTED: this resistance player is on the team, with
        1 SUSPECT, 0 UNSURE and 2 TRUSTED other players
    WITHOUT_ME_1SUSPECT_0UNSURE_2TRUSTED: the same, but this resistance player is
        not on the team
Many teams can be of one kind. Rate how likely this resistance player would want to put
up a team of each kind.""",
}

# One example per kind of question: a real question from a 6 player game, copied
# as the judge was sent it, and the answer a rational player would give, one field
# per line in the order of the response model.
_EXAMPLE = {
    ("spy", "vote"): """\
The following block is an EXAMPLE:
----------------------------------
Question:
Situation to rate:
- Current Score: 1 missions succeeded, 1 failed. 3 successes win for the Resistance, 3 failures for the Spies.
- The mission team has 4 players. 1 or more Fail cards make the mission fail.
- This spy player is on the team: yes. Spies on the team other than this spy player: 0.
- This spy player proposed this team: no.
- Options: APPROVE, REJECT
Rate each option (LIKELY / NEUTRAL / UNLIKELY), then give one overall confidence.
Answer:
reasoning = This spy player is on the team, so its Fail card can make the mission fail.
APPROVE = LIKELY
REJECT = UNLIKELY
confidence = MEDIUM
----------------------------------""",

    ("spy", "proposal"): """\
The following block is an EXAMPLE:
----------------------------------
Question:
Situation to rate:
- Current Score: 0 missions succeeded, 1 failed. 3 successes win for the Resistance, 3 failures for the Spies.
- The mission team has 3 players. 1 or more Fail cards make the mission fail.
- Options: WITHOUT_ME_0_PARTNERS, WITHOUT_ME_1_PARTNERS, WITH_ME_0_PARTNERS, WITH_ME_1_PARTNERS
Rate each option (LIKELY / NEUTRAL / UNLIKELY), then give one overall confidence.
Answer:
reasoning = A team with no spy on it cannot fail, and one spy is enough to make it fail. Two spies on one failed team are easy to catch.
WITHOUT_ME_0_PARTNERS = UNLIKELY
WITHOUT_ME_1_PARTNERS = LIKELY
WITH_ME_0_PARTNERS = LIKELY
WITH_ME_1_PARTNERS = NEUTRAL
confidence = MEDIUM
----------------------------------""",

    ("spy", "mission"): """\
The following block is an EXAMPLE:
----------------------------------
Question:
Situation to rate:
- Current Score: 1 missions succeeded, 1 failed. 3 successes win for the Resistance, 3 failures for the Spies.
- The mission team has 2 players. 1 or more Fail cards make the mission fail.
- This spy player is on the team. Spies on the team other than this spy player: 0.
- Options: SUCCESS, FAIL
Rate each option (LIKELY / NEUTRAL / UNLIKELY), then give one overall confidence.
Answer:
reasoning = Only this spy player's card can make the mission fail. On a team of 2, a failed mission points at it.
SUCCESS = UNLIKELY
FAIL = LIKELY
confidence = MEDIUM
----------------------------------""",

    ("resistance", "vote"): """\
The following block is an EXAMPLE:
----------------------------------
Question:
Situation to rate:
- Current Score: 1 missions succeeded, 1 failed. 3 successes win for the Resistance, 3 failures for the Spies.
- The mission team has 4 players. 1 or more Fail cards make the mission fail.
- This resistance player is on the team: no. This resistance player proposed this team: no.
- How trustworthy the players on the team other than this resistance player look: 1 SUSPECT, 1 UNSURE, 2 TRUSTED.
- Options: APPROVE, REJECT
Rate each option (LIKELY / NEUTRAL / UNLIKELY), then give one overall confidence.
Answer:
reasoning = A SUSPECT player is on the team, so the mission would likely fail.
APPROVE = UNLIKELY
REJECT = LIKELY
confidence = MEDIUM
----------------------------------""",

    ("resistance", "proposal"): """\
The following block is an EXAMPLE:
----------------------------------
Question:
Situation to rate:
- Current Score: 1 missions succeeded, 1 failed. 3 successes win for the Resistance, 3 failures for the Spies.
- The mission team has 4 players. 1 or more Fail cards make the mission fail.
- Options: WITHOUT_ME_1SUSPECT_0UNSURE_3TRUSTED, WITHOUT_ME_2SUSPECT_0UNSURE_2TRUSTED, \
WITH_ME_0SUSPECT_0UNSURE_3TRUSTED, WITH_ME_1SUSPECT_0UNSURE_2TRUSTED, WITH_ME_2SUSPECT_0UNSURE_1TRUSTED
Rate each option (LIKELY / NEUTRAL / UNLIKELY), then give one overall confidence.
Answer:
reasoning = This resistance player knows it is not a spy, so itself and 3 TRUSTED players is the safest team. Every other kind has a SUSPECT player.
WITHOUT_ME_1SUSPECT_0UNSURE_3TRUSTED = UNLIKELY
WITHOUT_ME_2SUSPECT_0UNSURE_2TRUSTED = UNLIKELY
WITH_ME_0SUSPECT_0UNSURE_3TRUSTED = LIKELY
WITH_ME_1SUSPECT_0UNSURE_2TRUSTED = UNLIKELY
WITH_ME_2SUSPECT_0UNSURE_1TRUSTED = UNLIKELY
confidence = MEDIUM
----------------------------------""",

}

_SUSPICION_EXAMPLE = """\
The following block is an EXAMPLE:
----------------------------------
Question:
Game history:

--- Mission 1 ---
Mission 1, proposal 1: APPROVED, and the mission FAILED. Player 2 proposed the team Player 0, Player 2.
  Approved by Player 0, Player 1, Player 2, Player 3, Player 5. Rejected by Player 4.
Rate every player: SUSPECT, UNSURE or TRUSTED.
Answer:
reasoning = Mission 1 failed, so Player 0 or Player 2 is probably a spy. Nothing yet tells the other players apart.
Player 0 = SUSPECT
Player 1 = UNSURE
Player 2 = SUSPECT
Player 3 = UNSURE
Player 4 = UNSURE
Player 5 = UNSURE
----------------------------------
"""


@lru_cache(maxsize=None)
def _system(faction: str, channel: str, num_players: int) -> str:
    """the system prompt for one kind of spy or resistance question"""
    if faction == "spy":
        who, block = _SPY, _SPY_CHANNEL[channel]
    else:
        who, block = _RESISTANCE, _RESISTANCE_CHANNEL[channel]
    return "\n\n".join(["You are estimating how players act in The Resistance, a social deduction game.",
        game_rules(num_players).strip(), who, block, _RATE_OPTIONS, _EXAMPLE[faction, channel]])


def _situation_lines(situation: Situation) -> list[str]:
    """the facts every question shares, as prompt lines"""
    lines = ["Situation to rate:",
        f"- Current Score: {situation.successes} missions succeeded, {situation.failures} failed. "
            f"{MISSIONS_TO_WIN} successes win for the Resistance, {MISSIONS_TO_WIN} failures for the Spies.",
        f"- The mission team has {situation.team_size} players. "
            f"{situation.fails_required} or more Fail cards make the mission fail."]
    if situation.last_chance:
        lines.append("- This is the last team allowed on this mission. If it is rejected, the spies win the game.")
    return lines


def _body(lines: list[str], options: tuple[str, ...]) -> str:
    """the situation lines, then the options to rate"""
    return "\n".join(lines + [f"- Options: {', '.join(options)}"])


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def spy_team_option(with_me: bool, partners: int) -> str:
    """the kind of team a spy leader sees: is it on it, and how many other spies are"""
    return f"{'WITH' if with_me else 'WITHOUT'}_ME_{partners}_PARTNERS"


def resistance_team_option(with_me: bool, looks: tuple[int, int, int]) -> str:
    """the kind of team a resistance leader sees

    looks: how many of the other players on the team look SUSPECT, UNSURE and TRUSTED
    """
    suspects, unsures, trusted = looks
    # Each number is glued to the word it counts: with a number between two words,
    # a small model reads 0_SUSPECT_1_UNSURE as one SUSPECT.
    return f"{'WITH' if with_me else 'WITHOUT'}_ME_{suspects}SUSPECT_{unsures}UNSURE_{trusted}TRUSTED"


def _spy_vote_lines(situation: Situation, on_team: bool, partners: int, proposed_by_me: bool) -> list[str]:
    lines = _situation_lines(situation)
    lines.append(f"- This spy player is on the team: {_yes(on_team)}. "
        f"Spies on the team other than this spy player: {partners}.")
    lines.append(f"- This spy player proposed this team: {_yes(proposed_by_me)}.")
    return lines


def _spy_mission_lines(situation: Situation, partners: int) -> list[str]:
    lines = _situation_lines(situation)
    lines.append(f"- This spy player is on the team. Spies on the team other than this spy player: {partners}.")
    return lines


def _resistance_vote_lines(situation: Situation, on_team: bool, proposed_by_me: bool, \
    looks: tuple[int, int, int]) -> list[str]:
    suspects, unsures, trusted = looks
    lines = _situation_lines(situation)
    lines.append(f"- This resistance player is on the team: {_yes(on_team)}. "
        f"This resistance player proposed this team: {_yes(proposed_by_me)}.")
    lines.append(f"- How trustworthy the players on the team other than this resistance player look: "
        f"{suspects} SUSPECT, {unsures} UNSURE, {trusted} TRUSTED.")
    return lines


def spy_vote(situation: Situation, on_team: bool, partners: int, proposed_by_me: bool) -> tuple[str, str, str]:
    """(system, body, ask) for how a spy votes

    partners: the number of other spies on the team
    """
    body = _body(_spy_vote_lines(situation, on_team, partners, proposed_by_me), VOTE_OPTIONS)
    return _system("spy", "vote", situation.num_players), body, RATE_ASK


def spy_proposal(situation: Situation, options: tuple[str, ...]) -> tuple[str, str, str]:
    """(system, body, ask) for which kind of team a spy leader puts up"""
    return _system("spy", "proposal", situation.num_players), _body(_situation_lines(situation), options), RATE_ASK


def spy_mission(situation: Situation, partners: int) -> tuple[str, str, str]:
    """(system, body, ask) for which card a spy plays

    partners: the number of other spies on the team
    """
    body = _body(_spy_mission_lines(situation, partners), CARD_OPTIONS)
    return _system("spy", "mission", situation.num_players), body, RATE_ASK


def resistance_vote(situation: Situation, on_team: bool, proposed_by_me: bool, \
    looks: tuple[int, int, int]) -> tuple[str, str, str]:
    """(system, body, ask) for how a resistance player votes

    looks: how many of the other players on the team look SUSPECT, UNSURE and TRUSTED
    """
    body = _body(_resistance_vote_lines(situation, on_team, proposed_by_me, looks), VOTE_OPTIONS)
    return _system("resistance", "vote", situation.num_players), body, RATE_ASK


def resistance_proposal(situation: Situation, options: tuple[str, ...]) -> tuple[str, str, str]:
    """(system, body, ask) for which kind of team a resistance leader puts up"""
    return _system("resistance", "proposal", situation.num_players), \
        _body(_situation_lines(situation), options), RATE_ASK


@lru_cache(maxsize=None)
def _suspicion_system(num_players: int, failed_mission_has_spy: bool) -> str:
    """the system prompt for rating every player from the public record"""
    lines = ["You are watching a game of The Resistance, a social deduction game, from outside.", "",
        game_rules(num_players).strip(), "",
        "You do not know who the spies are. Using only the public record, rate how trustworthy each player is:",
        _LEVELS]
    if failed_mission_has_spy:
        lines.append("A resistance player never plays a Fail card, so every failed mission had at least one spy on its team.")
    lines.append("")
    lines.append(_SUSPICION_EXAMPLE)
    return "\n".join(lines)


def _history_body(missions: list[MissionRecord], rejected_this_round: list[ProposalRecord]) -> str:
    return "\n".join(history_lines(missions, rejected_this_round, len(missions))).strip()


def suspicion(num_players: int, missions: list[MissionRecord], rejected_this_round: list[ProposalRecord], \
    failed_mission_has_spy: bool) -> tuple[str, str, str]:
    """(system, body, ask) for how every player looks from the public record

    Args:
        rejected_this_round: proposals rejected on the mission being played now
        failed_mission_has_spy: whether a resistance player never plays Fail,
            which is what the belief asking this assumes
    """
    return _suspicion_system(num_players, failed_mission_has_spy), \
        _history_body(missions, rejected_this_round), SUSPICION_ASK


# Manual chain of thought (judge_cot): one question is split into a reasoning
# pass and a rating pass. The Tofu analogues have the same names.

def judge_cot_analysis_prompt(body: str) -> str:
    """the reasoning pass: the question without its ask, and a request to reason only"""
    return body + "\n\nDo NOT give the ratings yet. First reason step by step, " \
        + "in 3-6 short sentences, about the situation above."


def judge_cot_final_prompt(body: str, analysis: str, ask: str) -> str:
    """the rating pass: the question, the reasoning pass's analysis, then the ask"""
    return body + "\n\nYour own step-by-step analysis:\n" + analysis \
        + "\n\nBased on this analysis: " + ask


def rating_prompt(body: str, ask: str) -> str:
    """the question in one pass, when judge_cot is off"""
    return body + "\n" + ask
