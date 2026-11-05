# [Resistance port] ours. Wires GRAIL's belief model and policy to our agent
# interface. It replaces upstream's agent_acl.py, which is mostly their
# websocket protocol, TypeChat schemas and discussion handling.
from __future__ import annotations
from pathlib import Path

from agents.resistance.base import ResistanceAgent
from agents.resistance.grail.ateam import ATEAM
from agents.resistance.grail.heuristic import HeuristicOracle
from agents.resistance.grail.model_reduced_categories import FactorGraphModelV2
from agents.resistance.grail.state_vector import NUM_PLAYERS, encode_state
from core.helper_functions import SeedSource
from game.resistance_data_classes import MAX_REJECTIONS, Phase, ResistanceObservation, Team

WEIGHTS_DIR = Path(__file__).resolve().parent / "models" / "v2"
ALGORITHMS = ("max", "sum") # max-product or sum-product belief propagation
DEFAULT_ALGORITHM = "max" # what upstream's agent asks for

# The rules GRAIL's factors were trained on. Its state vector cannot express
# anything else: the outcome entry is a single success or failure, and the
# graph has one variable per seat for six seats.
#
# enforce_resistance_must_pass is deliberately not here, see UPSTREAM.md. Both
# values are fine: the state vector holds no rule flag, and GRAIL's own
# resistance policy already plays Success on every mission it is sent on.
REQUIRED_RULES = {
    "num_players": NUM_PLAYERS,
    "reveal_fail_count": False,
}

_MODEL = None


def load_belief_model() -> FactorGraphModelV2:
    """the factor graph, built once and shared by every agent

    Building it takes about a second and predict_probs only reads it, so one
    copy serves the whole experiment instead of one per seat per game.

    The weights path is absolute on purpose. Upstream's loader joins a
    hardcoded "our/models/" in front of what it is given, and os.path.join
    drops that prefix when the second part is absolute.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = FactorGraphModelV2()
        _MODEL.construct()
        _MODEL.load_from_file(str(WEIGHTS_DIR))
    return _MODEL


class GrailResistanceAgent(ResistanceAgent):
    """GRAIL's Graph Only ablation

    Upstream's Graph Only agent is the full GRAIL agent with the language prior
    replaced by a uniform one. A uniform prior leaves the graph's own role
    marginals unchanged, so this agent never calls update_priors and the belief
    is the factor graph's alone.

    Resistance seats only. Upstream's policy proposes the seats it trusts most
    and rejects a team holding anyone it suspects, which is resistance
    reasoning; its own source says the policy has to change before a spy can
    use it. So a spy seat raises rather than playing a policy that was never
    written for it.
    """

    def __init__(self, player_id: int, seed_source: SeedSource, algorithm: str = DEFAULT_ALGORITHM):
        """
        algorithm: which belief propagation to run, "max" or "sum". Upstream's
            agent asks for max-product; sum-product is the other thing their
            engine supports and changing it cannot break anything.
        """
        super().__init__(player_id, seed_source)
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
        self.algorithm = algorithm
        self.oracle = HeuristicOracle(ATEAM.GOOD, rng=seed_source.to_random())
        self.model = load_belief_model()
        self.belief: dict[int, dict[str, float]] = None # seat to {'good': float, 'evil': float}
        self.state_vector: list[int] = None # the vector self.belief was computed from

    @classmethod
    def check_rules(cls, env, team: Team, agent_config: dict = None) -> None:
        if team != Team.RESISTANCE:
            raise ValueError("GRAIL has no spy policy, so it cannot take a spy seat. "
                "Upstream's own policy file says so, see agents/resistance/grail/UPSTREAM.md")
        for name, expected in REQUIRED_RULES.items():
            actual = getattr(env, name)
            if actual != expected:
                raise ValueError(f"GRAIL needs {name}={expected}, this experiment runs {name}={actual}. "
                    "Its factors were trained on that configuration and its state vector encodes it, "
                    "see agents/resistance/grail/UPSTREAM.md")

    def take_action(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        if obs.team != Team.RESISTANCE:
            raise Exception(f"GRAIL has no spy policy, but player {self.player_id} was dealt a spy card")

        self.__refresh_belief(obs) # equivalent to upstream's update_predictions, without the server protocol
        phase = obs.state.phase

        if phase == Phase.PROPOSE:
            # The leader is asked twice. With no discussion there is nothing to
            # revise on, so the second turn just calls the vote.
            for action in valid_actions:
                if action["type"] == "call_vote":
                    return action
            team = self.oracle.propose_party(obs.team_size, self.belief)
            return {"type": "propose_team", "team": sorted(team)}

        elif phase == Phase.TEAM_VOTE:
            vote = self.oracle.vote_for_party(sorted(obs.state.proposed_team), self.belief)
            # Upstream's agent overrides its own policy on the last proposal of
            # a round, because rejecting that one hands the game to the spies.
            # It lives in their agent and not in the policy, so it is here.
            if obs.state.rejections >= MAX_REJECTIONS - 1:
                vote = True
            return {"type": "team_vote", "vote": vote}

        elif phase == Phase.MISSION_VOTE:
            return {"type": "mission_vote", "vote": self.oracle.vote_for_quest()}

        raise Exception(f"nothing to decide in phase {phase.value}")

    def identity_belief(self) -> dict[int, float]:
        """the graph's posterior, as one probability of being a spy per seat"""
        if self.belief is None:
            return None
        spy_probability = {}
        for seat, probabilities in self.belief.items():
            spy_probability[seat] = probabilities["evil"]
        return spy_probability

    def final_belief(self, obs: ResistanceObservation) -> dict[int, float]:
        """recompute on the finished game, so the last mission is in the belief

        The graph reads the mission records and nothing else, so a finished game
        is an ordinary input to it. One more belief update, which is what the LLM
        baseline pays for the same look.
        """
        self.__refresh_belief(obs)
        return self.identity_belief()

    def __refresh_belief(self, obs: ResistanceObservation) -> None:
        """recompute the belief, but only when a mission has moved the vector

        One update takes seconds, and the vector only changes when a mission
        finishes, so every proposal and vote inside a round reuses the same
        belief. Upstream recomputes on every chat message; with no discussion
        there is nothing between two missions that could change the answer.
        """
        vector = encode_state(obs.state.missions)
        if vector == self.state_vector:
            return

        # predict_probs writes the reader's own role into the list it is given
        # and returns seats numbered from 1.
        probabilities = self.model.predict_probs(list(vector), ATEAM.GOOD, self.player_id, \
            algorithm=self.algorithm)
        belief = {}
        for seat in range(NUM_PLAYERS):
            belief[seat] = probabilities[seat + 1]

        self.belief = belief
        self.state_vector = vector
