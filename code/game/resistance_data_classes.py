# shared data classes for The Resistance environment
# Rules come from the official rulebook: Don Eskridge, Indie Boards and Cards.
#
# The counterpart of game/predefined_data_classes.py, and new code throughout:
# the two games share no state. The convention is what carries over, one frozen
# dataclass per thing the environment makes public.
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    """game phases"""
    PROPOSE = "propose" # leader assigns team cards, then calls for the vote
    TEAM_VOTE = "team_vote" # every player approves or rejects the team
    MISSION_VOTE = "mission_vote" # team members play success or fail
    DONE = "done"


class Team(Enum):
    """the two sides"""
    RESISTANCE = "resistance"
    SPIES = "spies"


@dataclass(frozen=True)
class PlayerCount:
    """the rulebook charts for one player count"""
    num_spies: int
    team_sizes: tuple[int, ...] # team size for each of the five missions
    fails_required: tuple[int, ...] # fail cards needed to sabotage each mission


# Rulebook charts, one entry per supported player count.
# Two fail cards are needed on the 4th mission with 7 or more players.
PLAYER_PRESET = {
    5: PlayerCount(2, (2, 3, 2, 3, 3), (1, 1, 1, 1, 1)),
    6: PlayerCount(2, (2, 3, 4, 3, 4), (1, 1, 1, 1, 1)),
    7: PlayerCount(3, (2, 3, 3, 4, 4), (1, 1, 1, 2, 1)),
    8: PlayerCount(3, (3, 4, 4, 5, 5), (1, 1, 1, 2, 1)),
    9: PlayerCount(3, (3, 4, 4, 5, 5), (1, 1, 1, 2, 1)),
    10: PlayerCount(4, (3, 4, 4, 5, 5), (1, 1, 1, 2, 1)),
}

MAX_REJECTIONS = 5 # "The Spies win the game if five Mission Teams are rejected in a single round"
MISSIONS_TO_WIN = 3
# A leader may revise its team before calling the vote, which is not a rule but
# what GRAIL's server allows. That server caps an agent at 10 actions in one
# turn and kills the game past it; we cap the revisions instead and let the
# game go on. Its released games never went past 6 proposals before a vote.
MAX_PROPOSALS_BEFORE_VOTE = 10


@dataclass
class ProposalRecord:
    """one team a leader put to the vote, and how the table voted on it

    Who proposed a team is public and is evidence: a leader that keeps naming
    the players a failed mission was made of is not reading as resistance. The
    state's own leader field is the current one and moves on every round, so
    without this the proposer of every past team would be lost.
    """
    leader: int # player id that proposed this team
    team: frozenset[int] # the players it named
    votes: dict[int, bool] # player id to approve(True)/reject(False)


@dataclass
class MissionRecord:
    """what the environment made public about one mission"""
    index: int # which mission, starting from 0
    leader: int # player id that proposed the approved team
    team: frozenset[int] # the approved team
    team_votes: dict[int, bool] # player id to approve(True)/reject(False)
    rejected: list[ProposalRecord] # proposals rejected before this one
    success: bool
    num_fails: int = None # None when the config hides the count


@dataclass
class ResistanceState:
    """complete game state. maintained by environment"""
    num_players: int
    spies_ids: frozenset[int] # which player_ids were dealt a spy card
    phase: Phase
    leader: int # player_id that proposes this round
    mission_index: int # which mission, starting from 0

    mission_results: list[bool] = field(default_factory=list)
    missions: list[MissionRecord] = field(default_factory=list)
    rejections: int = 0 # teams rejected so far in this round
    rejected_this_round: list[ProposalRecord] = field(default_factory=list)

    proposed_team: frozenset[int] = None
    proposals_since_vote: int = 0 # teams the leader has assigned since the last vote closed
    player_voted: dict[int, bool] = field(default_factory=dict) # votes cast so far this phase,
    # secret until the phase resolves, so observation_for empties it
    approved_votes: dict[int, bool] = field(default_factory=dict) # votes that approved the current team
    winner: Team = None
    win_reason: str = None


@dataclass
class ResistanceObservation:
    """what a resistance operative can see

    The public state plus this player's own card. state.spies_ids is emptied
    here: the deal is not public, and a resistance operative is shown nothing
    about it.
    """
    player_id: int
    team: Team # the card this player looked at
    team_size: int # size the current mission team must have, None once the game is over
    fails_required: int # fail cards needed to sabotage the current mission, None once it is over
    resistance_must_pass: bool # rule in force; decides whether a failed mission
    # proves a spy was on the team, which a belief model needs to know
    state: ResistanceState # spies_ids emptied


@dataclass
class SpyObservation(ResistanceObservation):
    """what a spy can see: the same, plus its teammates

    Rulebook: "Spies open your eyes. Spies look around and make sure that you
    know all the other Spies."
    """
    known_spies: frozenset[int] # the other spies, never including this player
