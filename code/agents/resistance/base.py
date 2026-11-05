# base agent interface for The Resistance, plus the random baseline
#
# The counterpart of agents/base.py. The interface is new, because the two games
# ask a player for different things, but the contract is the same one: the
# environment decides what is legal and an agent only chooses.
from __future__ import annotations
from abc import ABC, abstractmethod

from core.helper_functions import SeedSource
from game.resistance_data_classes import Phase, ProposalRecord, ResistanceObservation, Team
from game.resistance_env import ResistanceEnv


def open_proposal(state) -> ProposalRecord:
    """the team on the table, as a record a belief can read

    None while a leader is still assigning one, because a team it may still
    revise is not what the table was shown. From the vote onwards it is public,
    and its votes are empty until they pass, so a player voting has seen the
    team and a player about to play a mission card has also seen the votes that
    let it run. No past record carries it: a team being voted on has not been
    rejected and has not run.
    """
    if state.phase not in (Phase.TEAM_VOTE, Phase.MISSION_VOTE) or state.proposed_team is None:
        return None
    # state.leader still holds the proposer here: the rotation happens when a
    # team is rejected or when its mission resolves, never in between.
    return ProposalRecord(leader=state.leader, team=frozenset(state.proposed_team), \
        votes=dict(state.approved_votes))


def refuse_impossible_fail_counts(env: ResistanceEnv, resistance_fail_rate: float) -> None:
    """a published Fail count no world can explain voids the game it appears in

    A belief that says a resistance player never plays Fail cannot explain more
    Fail cards than the team holds spies. With the count published and the rules
    still allowing the card, one resistance Fail rules out every world and the
    game ends in an exception, hours into a run rather than before it.

    The rulebook flag is the other way out: it forbids the card, so the count
    can never say what the belief cannot.
    """
    if not env.reveal_fail_count or env.enforce_resistance_must_pass:
        return
    if resistance_fail_rate > 0.0:
        return
    raise ValueError("reveal_fail_count with resistance_fail_rate 0 and the resistance still "
        "allowed to play Fail: a published count that only a resistance Fail card explains "
        "rules out every world. Set enforce_resistance_must_pass to forbid the card, or give "
        "the belief a resistance_fail_rate above 0")


class ResistanceAgent(ABC):
    """one player_id's policy

    Free discussion is switched off, so the environment has no speech action
    and an agent only picks game actions.
    """


    def __init__(self, player_id: int, seed_source: SeedSource):
        """
        seed_source: required, and it must carry a seed. A null source runs
            nondeterministically, which is fine in a notebook and never right
            in an experiment, so both a missing and a null source raise here
            rather than quietly costing a run its reproducibility.
        """
        if seed_source is None:
            raise ValueError(f"player {player_id}: {type(self).__name__} was built without a seed source")
        if seed_source.is_null:
            raise ValueError(f"player {player_id}: {type(self).__name__} was given an unseeded source")
        self.player_id = player_id
        self.seed_source = seed_source

    @classmethod
    def needs_llm(cls, agent_config: dict) -> bool:
        """does an agent built from this config entry need an LLM client

        Asked of the entry and not of the class, because talking to a model is a
        setting rather than a type: the ablation agent with every decision on
        random never asks anything, and would otherwise make a run that needs no
        Ollama demand an llm block and build a client it never used.
        """
        return False

    @classmethod
    def check_rules(cls, env: ResistanceEnv, team: Team, agent_config: dict = None) -> None:
        """raise if this agent cannot play this player id under these rules

        Reads the environment rather than the config, so it checks what will
        actually run, and takes the faction the player was dealt so an agent that
        only plays one side is caught here. agent_config is the entry it will be
        built from, for an agent that only one of its settings cannot play a side
        with. Called while the agents are built, which is outside the runner's
        per game try block, so a mismatch stops the experiment.
        """

    @abstractmethod
    def take_action(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """return one action for ResistanceEnv.step()

        valid_actions comes from env.get_valid_actions(), so legality is the
        environment's job and an agent only chooses. In PROPOSE the leader is
        asked twice: first the menu is the possible teams, then it also holds
        call_vote.
        """

    def identity_belief(self) -> dict[int, float]:
        """this agent's probability that each player holds a spy card, or None
        """
        return None

    def final_belief(self, obs: ResistanceObservation) -> dict[int, float]:
        """this agent's belief once the game is over, on the finished game

        identity_belief is only read when the agent acts, and nobody acts after
        the last mission resolves, so the belief scored at the end of a game
        never carried the result that ended it. This is the one look that does.

        obs comes from ResistanceEnv.final_observation_for, so its team_size and
        fails_required are None: no mission is current any more.
        """
        return self.identity_belief()

    def __str__(self) -> str:
        return type(self).__name__


class RandomResistanceAgent(ResistanceAgent):
    """picks uniformly among the legal actions, except that it never revises a team

    GRAIL uses the same lower bound: "a Random agent that performs random
    actions at each timestep". It knows no rules at all, so it can never play
    an illegal card whatever the rule flags say.

    The leader's second turn is the exception. There the menu holds call_vote
    next to every team again, so picking uniformly would revise about twenty
    times per proposal and fill the log with teams nobody ever voted on. Only
    the last team proposed reaches the vote, and every proposal is uniform, so
    calling the vote at once leaves that team uniform too: the games this agent
    plays are distributed exactly as before and only the log gets shorter.
    """

    def __init__(self, player_id: int, seed_source: SeedSource, plays_success: bool = True):
        """
        plays_success: a resistance player plays the Success card without drawing
            for it. ON by default, because the belief reads a resistance Fail
            card as impossible evidence: RESISTANCE_FAIL_RATE is 0, so a mission
            that failed with no spy on it rules out every world and ends the
            game in an exception.

            Turn it on to compare a belief model against GRAIL. Under GRAIL's own
            rule settings a resistance player MAY play Fail and this one does,
            half the time, which makes the mission outcome carry no information
            about the deal at all: P(the mission fails) is 1 - 2^-|team| whatever
            the deal is, so every belief model scores the no information baseline
            and none of them can be told apart. GRAIL's own agent never plays
            Fail on a resistance seat (heuristic.py, vote_for_quest), so this
            flag is what makes the two line-ups agree on that one card rather
            than a change to how the baseline plays.
        """
        super().__init__(player_id, seed_source)
        self.plays_success = plays_success
        self.rng = seed_source.to_random()

    def take_action(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        # call_vote is only ever offered on the leader's second turn of a proposal.
        for action in valid_actions:
            if action["type"] == "call_vote":
                return action
        if self.plays_success and obs.team == Team.RESISTANCE \
            and valid_actions[0]["type"] == "mission_vote":
            return {"type": "mission_vote", "vote": True}
        return self.rng.choice(valid_actions)
