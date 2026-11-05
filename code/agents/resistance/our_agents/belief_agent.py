# [Resistance] our method, belief only: the exact belief on the random policy.
#
# GRAIL's baseline has the same shape, its Graph Only ablation, and for the same
# reason: it separates "does the belief model read the game correctly" from "does
# the policy play well". This agent takes no decision of its own, so a run of it
# is the random baseline's games with our belief measured on top of them.
from __future__ import annotations

from agents.resistance.base import (
    RandomResistanceAgent, open_proposal, refuse_impossible_fail_counts,
)
from agents.resistance.our_agents.kripke import (
    KripkeState, MODEL_EPSILON, RESISTANCE_FAIL_RATE, SPY_APPROVES_SPY_TEAM, SPY_FAIL_RATE,
    SPY_STEERS_ITS_TEAM,
)
from core.helper_functions import SeedSource
from game.resistance_data_classes import ResistanceObservation, Team
from game.resistance_env import ResistanceEnv


class BeliefResistanceAgent(RandomResistanceAgent):
    """the exact belief over deals, carried by the random baseline's policy

    Subclassed from the random agent and adding no draw of its own, so given the
    same seed it plays exactly the games the random baseline plays. That is the
    point: the belief is then measured on a fixed set of games rather than on
    games its own policy chose, and the two line-ups are directly comparable.

    Resistance players only, the way GRAIL's port is. The spy side needs the
    public belief rather than a private one and a policy that hides rather than
    infers, which is a later step.
    """

    def __init__(self, player_id: int, seed_source: SeedSource, spy_fail_rate: float = SPY_FAIL_RATE, \
        resistance_fail_rate: float = RESISTANCE_FAIL_RATE, \
            spy_approves_spy_team: float = SPY_APPROVES_SPY_TEAM, \
                spy_steers_its_team: float = SPY_STEERS_ITS_TEAM, \
                    read_votes: bool = True, read_proposals: bool = True, \
                        model_epsilon: float = MODEL_EPSILON, plays_success: bool = True):
        """
        The four rates are the opponent model, model_epsilon is how much of it is
        conceded to be wrong, and the two read flags pick which channels are read
        at all. They are passed straight to KripkeState, which documents what
        each one means; they are constructor arguments here so a config entry can
        sweep them.

        plays_success goes to the random policy underneath, which is what makes a
        run of this agent comparable with GRAIL's Graph Only ablation.
        """
        super().__init__(player_id, seed_source, plays_success=plays_success)
        self.belief_options = {
            "spy_fail_rate": spy_fail_rate,
            "resistance_fail_rate": resistance_fail_rate,
            "spy_approves_spy_team": spy_approves_spy_team,
            "spy_steers_its_team": spy_steers_its_team,
            "read_votes": read_votes,
            "read_proposals": read_proposals,
            "model_epsilon": model_epsilon,
        }
        self.belief: KripkeState = None # built on the first observation, which carries the table size
        # mission index to the card this agent played on it. Its own card is not
        # a random variable to it, and a mission it sabotaged itself says nothing about anyone else.
        self.own_plays: dict[int, bool] = {}

    @classmethod
    def check_rules(cls, env: ResistanceEnv, team: Team, agent_config: dict = None) -> None:
        if team != Team.RESISTANCE:
            raise ValueError("the belief agent has no spy policy yet, so it cannot take a spy "
                "seat. A spy knows the deal and needs the public belief instead, which is a later step")
        rate = RESISTANCE_FAIL_RATE if agent_config is None \
            else agent_config.get("resistance_fail_rate", RESISTANCE_FAIL_RATE)
        refuse_impossible_fail_counts(env, rate)

    def take_action(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """the random policy, with the belief brought up to date first"""
        self.__sync(obs)
        action = super().take_action(obs, valid_actions)
        if action["type"] == "mission_vote":
            self.own_plays[obs.state.mission_index] = action["vote"]
        return action

    def identity_belief(self) -> dict[int, float]:
        """P(spy) per player, straight out of the world distribution

        None before this agent has acted, because the table size arrives with
        the first observation.
        """
        if self.belief is None:
            return None
        return self.belief.spy_probability()

    def final_belief(self, obs: ResistanceObservation) -> dict[int, float]:
        """sync on the finished game before answering

        The belief is rebuilt when this agent acts, and nobody acts after the
        last mission resolves, so answering without this would report a belief
        one mission out of date.
        """
        self.__sync(obs)
        return self.identity_belief()

    def __sync(self, obs: ResistanceObservation) -> None:
        """rebuild the belief from everything the table has made public"""
        if self.belief is None:
            self.belief = KripkeState(obs.state.num_players, viewer=self.player_id, viewer_is_spy=False, \
                **self.belief_options)
        self.belief.sync_with_history(obs.state.missions, obs.state.rejected_this_round, \
            self.own_plays, obs.state.leader, open_proposal(obs.state))
