# game environment for The Resistance (the 6-player variant GRAIL plays)
# Quoted lines in the docstrings are from the official rulebook.
#
# The counterpart of game/tofu_env.py. Nothing is shared: the two games have
# different phases, actions and win conditions. What carries over is the shape
# every agent here is written against, reset / observation_for / step, so a
# runner written for one reads like a runner written for the other.
from __future__ import annotations
import random
from copy import deepcopy
from itertools import combinations

from game.resistance_data_classes import (
    Phase, Team, PlayerCount, PLAYER_PRESET,
    MAX_REJECTIONS, MISSIONS_TO_WIN, MAX_PROPOSALS_BEFORE_VOTE,
    MissionRecord, ProposalRecord, ResistanceState, ResistanceObservation, SpyObservation,
)


class ResistanceEnv:
    """
    The Resistance game environment.

    Game flow:
        1. reset(seed) to initialise game
        2. PROPOSE: the leader proposes a team, revises it if it wants, then
            calls for the vote
        3. TEAM_VOTE: every player votes; a rejection passes leadership clockwise
        4. MISSION_VOTE: the team plays success or fail cards
        5. steps 2-4 repeat until one side wins three missions, or five teams
            are rejected in a single round

    Every action goes through step(player_id, action). pending_actors says who
    the environment is waiting on, so a runner never tracks phases itself.

    Discussion is not modelled here. The runner is expected to handle that.

    Two rules are flags because GRAIL's environment does not implement them.
    Both default to GRAIL's behaviour so that the default config reproduces the
    setting its published numbers come from.
    """

    def __init__(self, num_players: int = 6, enforce_resistance_must_pass: bool = False, \
        reveal_fail_count: bool = False):
        """
        enforce_resistance_must_pass: rulebook says "The Resistance Operatives
            must select the Mission Success card". When True a resistance fail
            card is rejected as an illegal action, which makes "mission failed
            implies at least one spy on the team" a hard constraint.
        reveal_fail_count: rulebook has the mission cards shuffled and revealed,
            so the number of fail cards is public. When False only success or
            failure is published.
        """
        if num_players not in PLAYER_PRESET:
            raise Exception(f"number of players must be between 5-10 !!!!")

        self.num_players = num_players
        self.enforce_resistance_must_pass = enforce_resistance_must_pass
        self.reveal_fail_count = reveal_fail_count
        self.state: ResistanceState = None

    @property
    def preset(self) -> PlayerCount:
        """the rulebook charts for this player count"""
        return PLAYER_PRESET[self.num_players]

    def reset(self, seed: int = None, spies_id=None, leader: int = None) -> ResistanceState:
        """initialise a new game

            deal the spies and pick the first leader. Pass spies and leader to
            replay a recorded game instead of dealing a fresh one.
        """
        ran_seed = random.Random(seed)
        player_ids = range(self.num_players)

        # "Randomly select Leader and spies; the Leader receives the Leader card."
        if spies_id is None:
            spies_id = ran_seed.sample(player_ids, self.preset.num_spies)
        if leader is None:
            leader = ran_seed.randrange(self.num_players)

        spies_id = frozenset(spies_id)
        if len(spies_id) != self.preset.num_spies:
            raise Exception(f"{self.num_players} players game need {self.preset.num_spies} spies, got {sorted(spies_id)}")
        if not spies_id.issubset(player_ids):
            raise Exception(f"spies has unknown players: {sorted(spies_id)}")
        if leader not in player_ids:
            raise Exception(f"unknown leader {leader}")

        self.state = ResistanceState(
            num_players=self.num_players,
            spies_ids=spies_id,
            phase=Phase.PROPOSE,
            leader=leader,
            mission_index=0,
        )
        return self.snapshot()

    def snapshot(self) -> ResistanceState:
        """a copy of the state, safe to hand to an agent"""
        return deepcopy(self.state)

    @property
    def done(self) -> bool:
        return self.state.phase == Phase.DONE

    @property
    def team_size(self) -> int:
        """required size of the current mission team"""
        return self.preset.team_sizes[self.state.mission_index]

    @property
    def fails_required(self) -> int:
        """fail cards needed to sabotage the current mission"""
        return self.preset.fails_required[self.state.mission_index]

    def is_spy(self, player: int) -> bool:
        return player in self.state.spies_ids

    def team_of(self, player: int) -> Team:
        """the card this player was dealt"""
        return Team.SPIES if player in self.state.spies_ids else Team.RESISTANCE

    def observation_for(self, player: int) -> ResistanceObservation:
        """what this player can see

        A spy gets a SpyObservation carrying its teammates. Rulebook: "Spies
        look around and make sure that you know all the other Spies", while
        resistance operatives are shown nothing. Either way the state handed
        out has spies_ids emptied, so it never carries the real deal.

        Only valid while the game is running: team_size reads the current
        mission, and after the fifth one there is no current mission.
        """
        if self.done:
            raise Exception("the game is over, there is no decision to take. "
                "Use final_observation_for to ask a finished game a question")
        return self.__build_observation(player, self.team_size, self.fails_required)

    def final_observation_for(self, player: int) -> ResistanceObservation:
        """what this player can see once the game is over

        observation_for describes a pending decision and a finished game has
        none, so team_size and fails_required are None here: they read the
        current mission, and after the last one there is no current mission.
        Everything else is the same, so a closing question reads the same
        observation every decision read.
        """
        if not self.done:
            raise Exception("the game is still running, use observation_for")
        return self.__build_observation(player, None, None)

    def __build_observation(self, player: int, team_size: int, fails_required: int) \
        -> ResistanceObservation:
        """the observation itself, the same whether the game is running or over"""
        public = self.snapshot()
        public.spies_ids = frozenset()
        # Voting is simultaneous. player_voted holds what has been cast so far
        # this phase, so handing it over would let a player who votes later read
        # the votes already in. What becomes public on resolution is kept:
        # approved_votes for a team that passed, rejected_this_round for one that
        # did not.
        public.player_voted = {}
        common = {
            "player_id": player,
            "team": self.team_of(player),
            "team_size": team_size,
            "fails_required": fails_required,
            "resistance_must_pass": self.enforce_resistance_must_pass,
            "state": public,
        }
        if not self.is_spy(player):
            return ResistanceObservation(**common)
        return SpyObservation(**common, known_spies=self.state.spies_ids - {player})

    @property
    def pending_actors(self) -> frozenset[int]:
        """players the environment is waiting on right now"""
        state = self.state
        if state.phase == Phase.PROPOSE:
            return frozenset({state.leader})
        elif state.phase == Phase.TEAM_VOTE:
            return frozenset(range(self.num_players)) - set(state.player_voted)
        elif state.phase == Phase.MISSION_VOTE:
            return state.proposed_team - set(state.player_voted)
        return frozenset()

    def get_valid_actions(self, player_id: int) -> list[dict]:
        """every action this player may legally take right now

        Empty when the environment is not waiting on this player. step() still
        validates on its own, so this is a menu and not the authority; the two
        agreeing is checked in test/test_resistance.py.
        """
        if self.state is None or player_id not in self.pending_actors:
            return []

        phase = self.state.phase

        if phase == Phase.PROPOSE:
            actions = []
            if self.state.proposals_since_vote < MAX_PROPOSALS_BEFORE_VOTE:
                for team in combinations(range(self.num_players), self.team_size):
                    actions.append({"type": "propose_team", "team": list(team)})
            if self.state.proposed_team is not None:
                actions.append({"type": "call_vote"})
            return actions

        elif phase == Phase.TEAM_VOTE:
            return [{"type": "team_vote", "vote": True}, {"type": "team_vote", "vote": False}]

        elif phase == Phase.MISSION_VOTE:
            actions = [{"type": "mission_vote", "vote": True}]
            # "The Resistance Operatives must select the Mission Success card"
            if self.is_spy(player_id) or not self.enforce_resistance_must_pass:
                actions.append({"type": "mission_vote", "vote": False})
            return actions

        return []

    def step(self, player_id: int, action: dict) -> None:
        """process an action. Validates, updates state, proceed to next phase

        Args:
            player_id: player who take action
            action: dict with a "type" key, one of
                {"type": "propose_team", "team": [0, 1]} leader, PROPOSE
                {"type": "call_vote"} leader, PROPOSE
                {"type": "team_vote", "vote": True} everyone, TEAM_VOTE
                {"type": "mission_vote", "vote": True} the team, MISSION_VOTE
        """
        if self.state is None:
            raise Exception("game has not started, call reset() first !!!!")
        if player_id not in self.pending_actors:
            raise Exception(f"Not Player {player_id}'s turn. Expected one of {sorted(self.pending_actors)}.")

        action_type = action["type"]

        if action_type == "propose_team":
            self.__step_propose_team(action)
        elif action_type == "call_vote":
            self.__step_call_vote()
        elif action_type == "team_vote":
            self.__step_team_vote(player_id, action)
        elif action_type == "mission_vote":
            self.__step_mission_vote(player_id, action)
        else:
            raise ValueError(f"Invalid action type: {action_type}!!!!!!!!")

    def __step_propose_team(self, action: dict) -> None:
        """the leader assigns team cards

        "a player may only be assigned one Team card", so the team is a set.

        The vote only starts on a call_vote action, so the leader may propose
        again to revise the team. GRAIL's leaders do this in 129 of the 324
        proposals in its released games. Revisions are capped so that a leader
        which never calls the vote cannot stall the game for ever.
        """
        self._phase_check(Phase.PROPOSE)
        if self.state.proposals_since_vote >= MAX_PROPOSALS_BEFORE_VOTE:
            raise Exception(f"leader has assigned {MAX_PROPOSALS_BEFORE_VOTE} teams without "
                "calling the vote, it may only call the vote now")
        team = frozenset(action["team"])
        if len(team) != self.team_size:
            raise Exception(f"team needs {self.team_size} distinct members, got {sorted(team)}")
        if not team.issubset(range(self.num_players)):
            raise Exception(f"team has unknown players: {sorted(team)}")

        self.state.proposed_team = team
        self.state.proposals_since_vote += 1

    def __step_call_vote(self) -> None:
        """the leader calls for the vote on the team it has assigned"""
        self._phase_check(Phase.PROPOSE)
        if self.state.proposed_team is None:
            raise Exception("no team has been proposed to vote on")

        self.state.player_voted = {}
        self.state.proposals_since_vote = 0
        self.state.phase = Phase.TEAM_VOTE

    def __step_team_vote(self, player_id: int, action: dict) -> None:
        """one player's vote on the proposed team"""
        self._phase_check(Phase.TEAM_VOTE)
        self.state.player_voted[player_id] = bool(action["vote"])
        if not self.pending_actors:
            self.__resolve_team_vote()

    def __step_mission_vote(self, player_id: int, action: dict) -> None:
        """one team member's mission card"""
        self._phase_check(Phase.MISSION_VOTE)
        vote = bool(action["vote"])
        if self.enforce_resistance_must_pass and not vote and player_id not in self.state.spies_ids:
            raise Exception(f"resistance must play success, but {player_id} played fail")

        self.state.player_voted[player_id] = vote
        if not self.pending_actors:
            self.__resolve_mission()

    def __resolve_team_vote(self) -> None:
        """count the votes

        "The Mission Team is approved if the majority of Votes are affirmative
        and fails if a majority rejected the assignments; a tied vote is also a
        rejection."
        """
        state = self.state
        votes = state.player_voted

        approvals = 0
        for vote in votes.values():
            if vote:
                approvals += 1
        approved = approvals * 2 > self.num_players # more than half approved is passed

        if approved:
            state.approved_votes = dict(votes)
            state.player_voted = {}
            state.phase = Phase.MISSION_VOTE
            return

        state.rejections += 1
        # state.leader is still the player that proposed this team; the rotation
        # below moves it on, and the record has to keep the proposer.
        state.rejected_this_round.append(ProposalRecord(leader=state.leader, \
            team=state.proposed_team, votes=dict(votes)))
        state.proposed_team = None
        state.player_voted = {}
        # "If the Mission Team is rejected, the Leader passes in a clockwise manner"
        state.leader = (state.leader + 1) % self.num_players
        if state.rejections >= MAX_REJECTIONS:
            self._finish(winner=Team.SPIES, reason=f"{MAX_REJECTIONS} teams rejected in a single round")
        else:
            state.phase = Phase.PROPOSE

    def __resolve_mission(self) -> None:
        """count the mission cards and check whether the game is over

        "The Mission is successful only if all the Cards revealed are Mission
        Success cards. The Mission fails if one (or more) Mission Fail cards
        have been played."
        """
        state = self.state

        num_fails = 0
        for vote in state.player_voted.values():
            if not vote:
                num_fails += 1
        success = num_fails < self.fails_required

        state.missions.append(MissionRecord(
            index=state.mission_index,
            # An approved team does not pass the leadership on, so this is still
            # the player that proposed it. The rotation below happens after.
            leader=state.leader,
            team=state.proposed_team,
            team_votes=state.approved_votes,
            rejected=state.rejected_this_round,
            success=success,
            num_fails=num_fails if self.reveal_fail_count else None,
        ))
        state.mission_results.append(success)

        state.mission_index += 1
        state.rejections = 0
        state.rejected_this_round = []
        state.proposed_team = None
        state.player_voted = {}
        # approved_votes means "the votes that approved the team now on a
        # mission", and no team is on one any more. Rebound and not cleared in
        # place: the mission record above holds this same dict.
        state.approved_votes = {}
        # "The Leader passes in a clockwise manner and the Team building phase
        # of the next Round begins."
        state.leader = (state.leader + 1) % self.num_players

        successes = 0
        for result in state.mission_results:
            if result:
                successes += 1
        failures = len(state.mission_results) - successes
        if failures >= MISSIONS_TO_WIN:
            self._finish(winner=Team.SPIES, reason=f"{MISSIONS_TO_WIN} missions failed")
        elif successes >= MISSIONS_TO_WIN:
            self._finish(winner=Team.RESISTANCE, reason=f"{MISSIONS_TO_WIN} missions succeeded")
        else:
            state.phase = Phase.PROPOSE

    def _phase_check(self, phase: Phase) -> None:
        if self.state.phase != phase:
            raise Exception(f"expected phase {phase.value}, in phase {self.state.phase.value}")

    def _finish(self, winner: Team, reason: str) -> None:
        self.state.winner = winner
        self.state.win_reason = reason
        self.state.phase = Phase.DONE
