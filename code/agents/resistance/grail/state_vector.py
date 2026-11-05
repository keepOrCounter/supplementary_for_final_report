# [Resistance port] ours. The encoder that feeds GRAIL's factor graph.
# It replaces GameInfo.get_state_vector in upstream's agent_acl.py, which reads
# their server's name keyed dictionaries. We read our own mission records.
from __future__ import annotations
from itertools import combinations

from game.resistance_data_classes import MissionRecord

NUM_PLAYERS = 6 # GRAIL's graph is built for six players and nothing else
NUM_MISSIONS = 5
MIN_APPROVALS = 4 # a team of six passes on a strict majority, so four or more


def _vote_compositions() -> list[tuple[int, ...]]:
    """every set of seats that can approve a team, in upstream's order"""
    compositions = []
    for size in range(MIN_APPROVALS, NUM_PLAYERS + 1):
        for voters in combinations(range(NUM_PLAYERS), size):
            compositions.append(voters)
    return compositions


VOTE_COMPOSITIONS = _vote_compositions()


def encode_state(missions: list[MissionRecord]) -> list[int]:
    """the 21 entry state vector GRAIL's factor graph reads

    Layout:
        entries 0-5: one per seat, that seat's role. Always 0 here, because
            predict_probs writes the reader's own entry and infers the rest.
        entries 6-20: three per mission, in mission order
            party: 1 + the approved team's place in combinations(seats, size)
            vote: 1 + the approving seats' place in VOTE_COMPOSITIONS
            outcome: 1 if the mission failed, 2 if it succeeded
        A mission that has not been played contributes 0, 0, 0.

    Seats are numbered from 0 here and from 1 upstream. A team's place in the
    combinations list is the same either way, because the two lists are the
    same tuples with every seat shifted by one.

    Only finished missions are encoded, and of each one only the proposal that
    was approved. Rejected proposals never enter the vector. That is upstream's
    behaviour, and it is why GRAIL does not see the 76 rejected proposals in
    its own released games.
    """
    vector = [0] * NUM_PLAYERS

    for index in range(NUM_MISSIONS):
        if index >= len(missions):
            vector.extend([0, 0, 0])
            continue

        record = missions[index]
        team = tuple(sorted(record.team))
        approvers = []
        for seat in sorted(record.team_votes):
            if record.team_votes[seat]:
                approvers.append(seat)

        teams = list(combinations(range(NUM_PLAYERS), len(team)))
        vector.append(teams.index(team) + 1) # party
        vector.append(VOTE_COMPOSITIONS.index(tuple(approvers)) + 1) # vote
        vector.append(int(record.success) + 1) # outcome

    return vector
