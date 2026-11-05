# [Resistance] our method's belief state: which players hold a Spy card.
#
# The analogue of agents/our_agents/Kripke.py, which does the same for Tofu
# Kingdom, and named to match it: a set of possible worlds with a weight each,
# updated by Bayes on every public observation.
#
# The two games differ in one number that decides everything about this file.
# Tofu Kingdom has 7! = 5040 worlds, so its state prunes and approximates. Here a
# world is the set of players that were dealt a Spy card, so a six player game has
# C(6,2) = 15 of them, and a resistance player that knows its own card has
# C(5,2) = 10 left. Nothing needs pruning: every update touches every world and
# the posterior is exact.
from __future__ import annotations
import math
from itertools import combinations, product
from random import Random

from game.resistance_data_classes import (
    MAX_REJECTIONS, MISSIONS_TO_WIN, PLAYER_PRESET, MissionRecord, ProposalRecord,
)
from agents.resistance.our_agents.lookahead import mission_value, swing, team_gain, win_values

# The opponent model, one number per thing a player chooses. A Resistance game
# makes three things public, so the belief reads three channels and each needs its
# own P(what was done | this world):
#
#   the mission result   SPY_FAIL_RATE, RESISTANCE_FAIL_RATE
#   the votes on a team  SPY_APPROVES_SPY_TEAM
#   who proposed a team  SPY_STEERS_ITS_TEAM
#
# Every one of them describes only the SPIES, because only a spy knows the deal
# and can act on it. A resistance player in a candidate world does not know which
# world it is in, so nothing it does can depend on the world, and at this rung it
# is modelled as acting the same way in every world: see vote_likelihood. That
# asymmetry is what makes a vote informative at all.
#
# They are a model of the opponents, so the right values depend on who is at the
# table. A random line-up does everything with probability one half, which none of
# these defaults describes; check_resistance_belief.py reports what that costs.
# Replacing the constants with a best response computed per player and per
# position is the ToM ladder, and is not this class.
SPY_FAIL_RATE = 0.8 # a spy on a mission plays the Fail card
# A resistance player never plays the Fail card. It is a simplification of the
# game and not a measurement of one: the card stops being a decision, so nothing
# below has to price it and nothing above has to model it. Set it back above 0
# for a table where some resistance seat might still play Fail, or a mission that
# fails with no spy on it becomes impossible evidence and rules out the world it
# actually happened in.
RESISTANCE_FAIL_RATE = 0.0
SPY_APPROVES_SPY_TEAM = 0.8 # a spy approves a team that carries at least one spy
# How often a spy leader steers, rather than how often its team ends up carrying a
# spy. Those are two different numbers: a team is proposed either on purpose or by
# picking uniformly like anybody else, and most uniform picks carry a spy too, so
# steering 0.8 of the time puts a spy on 0.92 of a six player game's two player
# teams and on 0.99 of its four player ones. See proposal_likelihood.
SPY_STEERS_ITS_TEAM = 0.8

# The two numbers the rung above needs. Neither describes this game: the first is
# the noise every quantal response model has, the second is an exchange rate.
#
# Sharpness is in odds per mission of value. 2.5 is where the rung above best
# matches the four constants above, on the smaller total relative error over the
# two Fail rates: it derives 0.880 against 0.8 and 0.047 against 0.05. The vote
# is not a target, and tops out near 0.6 where the rung below says 0.8: a vote
# moves the game by a fraction of a mission and a Fail card by a whole one, so a
# player picks its vote less sharply. See README.md for the sweep.
PLAYER_SHARPNESS = 2.5
# What one unit of the table's suspicion costs, in missions. The one number the
# rung above cannot derive: its gain prices a single mission, suspicion lasts the
# rest of the game, and converting needs a lookahead over the belief itself. Past
# about 1.0 a spy starts voting its own teams down.
BEING_SUSPECTED_COSTS = 0.5
# The most value, in missions, a modelled spy will give up to keep the table
# guessing. A budget and not a rate: the term it bounds is normalised, so the
# option that gives most away loses exactly this and the one that gives least
# away loses nothing, whatever the bits on the table happen to be worth.
#
# The rate this replaces had no bound and failed by mechanism: the cheapest way
# to teach the table nothing is to reject, so a spy given one rejected itself
# into the open. Its approval rate fell 40.9% to 18.0% and the resistance read
# it 99.9% of the time. README.md has the table.
#
# Zero until it is swept: a spy that spends nothing here is the spy every
# measurement in this repository was taken against.
SPY_HIDE_BUDGET = 0.0

# How much of an opponent model is conceded to be wrong. Off, because it was
# added to rescue a mismatched model and measurement says it does not: the win
# rate is flat and the belief is worse at every setting. See README.md. Turn it
# on for a model sharp enough to rule a world out, such as a large beta.
MODEL_EPSILON = 0.0


def smooth(chances: dict, epsilon: float) -> dict:
    """mix a policy with the uniform one over its own support

        P(a) = (1 - e) model(a) + e / k
            a   an action the model puts weight on
            e   epsilon, the argument
            k   how many actions the model puts weight on, the support

    Caps what one observation can be worth at log2((1 - e + e/k) / (e/k)) bits.

    Uniform over the support and not over every action, because a zero can come
    from the rules rather than from the model, and a rule is not a thing to
    smooth away.

    Laplace smoothing, used as a likelihood floor.
    """
    if epsilon <= 0.0:
        return chances
    support = 0
    for chance in chances.values():
        support += chance > 0.0
    if support == 0:
        return chances
    floor = epsilon / support # explore
    keep = 1.0 - epsilon # exploit
    return {key: keep * chance + (floor if chance > 0.0 else 0.0) for key, chance in chances.items()}


class KripkeState:
    """a distribution over which players were dealt a Spy card

    A world is a frozenset of player ids and nothing else, because that is all
    there is to know: everything else about a Resistance game is public.

    The viewer decides which worlds are possible at all:
        viewer=None     every deal, C(6,2) = 15 worlds. This is the public
            belief, what the table as a whole can infer, and it is what a spy
            reads when it wants to know how it is seen.
        viewer=3        the deals that agree with Player 3's own card, 10 of
            them for a resistance player. This is a private belief.

    Three observation channels are read, one per thing a Resistance game makes
    public: who proposed a team, how everyone voted on it, and whether the
    mission then failed. read_proposals and read_votes turn the first two off,
    which is how an ablation asks what each channel is worth and how a cell
    comparable with GRAIL is built, since GRAIL's state vector carries no
    proposals at all.

    A possible worlds model from epistemic logic, holding an exact posterior over
    fifteen worlds. The deal never changes, so this is sequential Bayesian
    inference and not a filter. Multiplying the three channels is a naive Bayes
    factorisation, which is an assumption. See docs/standard_names.md.
    """

    # The rung of the opponent model this class implements, and the name the
    # config and docs/opponent_models.md use for it. Every class in this
    # repository that assumes something about how other players behave carries
    # one, so grepping for it lists them all.
    OPPONENT_MODEL = "uniform"
    # Whether building this belief needs an llm_client. The agent reads it to
    # decide what to pass in.
    NEEDS_LLM = False

    # What one unit of the table's suspicion costs a player, in missions. Zero
    # here and only here: exposure is what a player looks like to the rung below,
    # and this rung has no rung below to look with. An agent reads its own price
    # off the belief it holds, so that the number it pays and the number its
    # model of everybody else pays cannot drift apart.
    hide_weight = 0.0

    def __init__(self, num_players: int, viewer: int = None, viewer_is_spy: bool = False, \
        spy_fail_rate: float = SPY_FAIL_RATE, resistance_fail_rate: float = RESISTANCE_FAIL_RATE, \
            spy_approves_spy_team: float = SPY_APPROVES_SPY_TEAM, \
                spy_steers_its_team: float = SPY_STEERS_ITS_TEAM, \
                    read_votes: bool = True, read_proposals: bool = True, \
                        model_epsilon: float = MODEL_EPSILON, info_weight: float = 0.0, \
                        adaptive_info_weight: bool = False):
        """
        viewer: the player this belief belongs to, or None for the public belief
        viewer_is_spy: the card the viewer was dealt, which decides which worlds
            it can rule out at the deal
        read_votes, read_proposals: whether to read those two channels at all.
            Off leaves the mission result as the only evidence, which is the
            evidence GRAIL's factor graph gets.
        model_epsilon: how much of every opponent model is conceded to be wrong.
            0 trusts it completely, which is what collapses onto a confidently
            wrong deal when it is not. See smooth().
        info_weight, adaptive_info_weight: what one bit is worth to whoever holds
            this belief, in win probability. Kept here rather than on the agent
            so that the agent and its model of everyone else spend one number,
            which the agent reads back off the belief as it does hide_weight.

            Read only where actions get priced, which is bits_are_worth, and
            that is StrategicKripkeState. A KripkeState reads them when it is an
            agent's own belief and never when it is a shadow: the rung above
            weighs with its OWN copy and asks the shadow only for the bits. So
            the bottom of a nested ladder carries these and ignores them.
        """
        if num_players not in PLAYER_PRESET:
            raise ValueError(f"no rulebook chart for {num_players} players, "
                f"known: {sorted(PLAYER_PRESET)}")
        if viewer is not None and viewer not in range(num_players):
            raise ValueError(f"viewer {viewer} is not a player id of a {num_players} player game")
        for name, rate in (("spy_fail_rate", spy_fail_rate), ("resistance_fail_rate", resistance_fail_rate), \
            ("spy_approves_spy_team", spy_approves_spy_team), ("spy_steers_its_team", spy_steers_its_team), \
                ("model_epsilon", model_epsilon)):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be a probability, got {rate}")

        self.num_players = num_players
        self.num_spies = PLAYER_PRESET[num_players].num_spies
        self.viewer = viewer
        self.viewer_is_spy = viewer_is_spy
        self.spy_fail_rate = spy_fail_rate # TODO what about random?
        self.resistance_fail_rate = resistance_fail_rate
        self.spy_approves_spy_team = spy_approves_spy_team
        self.spy_steers_its_team = spy_steers_its_team
        self.read_votes = read_votes
        self.read_proposals = read_proposals
        self.model_epsilon = model_epsilon
        self.info_weight = info_weight
        self.adaptive_info_weight = adaptive_info_weight
        self.leader: int = None # who put up the team being voted on now
        # How many teams this round has already lost. Carried for the same
        # reason as leader: rejecting prices what the NEXT leader would put up,
        # and whether that next team is the last one a mission allows changes
        # what it would be. Nothing in the record says it.
        self.rejections: int = 0

        self.worlds: list[frozenset[int]] = self.__possible_worlds()
        self.weights: list[float] = []
        # P(m Fail cards), keyed by how many players of each kind are on the team.
        # A six player game asks for at most a few dozen of these and asks for
        # each of them once per world per mission.
        self.__fail_counts: dict[tuple, list[float]] = {}
        # information_gain by the weights it was computed from. A few hundred
        # entries a game, and it does not survive the belief.
        self.__IG_cache: dict[tuple, float] = {}
        self.reset()

    def reset(self) -> None:
        """back to the deal: every possible world equally likely"""
        self.weights = [1.0 / len(self.worlds)] * len(self.worlds)
        # The score, carried because the rung above prices every action off the
        # value table and that table is keyed by the score. Advanced by
        # update_on_mission, so replaying a game through sync_with_history leaves
        # it at the score each observation was actually made at.
        self.successes = 0
        self.failures = 0
        self.leader = None
        self.rejections = 0
        self.__expected_success = None

    @property
    def num_worlds(self) -> int:
        return len(self.worlds)

    def sync_with_history(self, missions: list[MissionRecord], rejected_this_round: list[ProposalRecord] = (), \
        own_plays: dict[int, bool] = None, leader: int = None, \
            open_proposal: ProposalRecord = None) -> None:
        """rebuild the whole belief from everything the table has made public

        Rebuilt and not advanced, so syncing twice on the same record gives the
        same answer as syncing once and no caller has to track what it has
        already folded in.

        The order is the order it happened in, which is what makes the three
        channels multiply into one posterior: a team is proposed, voted on, and
        then either sunk or played.

        Args:
            rejected_this_round: proposals of the mission being played now, which
                has no record of its own yet. Dropping them would lose most of
                the votes of a game that ends on rejections.
            own_plays: mission index to the card the viewer played on it, True
                for Success. The viewer's own card is not a random variable to
                it, so a mission it was on says more to it than to the table.
            leader: who proposes this round. Set even before a team is on the
                table, because a rejection is priced by what the NEXT leader
                would put up and nothing in the record says who that is yet.
            open_proposal: the team ON THE TABLE and not yet resolved, which no
                past record carries: a team being voted on has not been rejected
                and has not run. It is public the moment it is named, so a player
                deciding anything after it has seen it, and leaving it out costs
                the proposal channel its most recent observation. Its votes are
                empty while the vote is open and hold the approving votes once it
                has passed, which is what a player about to play a mission card
                has also seen.
        """
        own_plays = own_plays or {}
        self.reset()
        for mission in missions:
            for number, proposal in enumerate(mission.rejected):
                self.__observe_proposal(proposal.leader, proposal.team, proposal.votes, number)
            # The approved team is the last proposal of its mission, so the ones
            # sunk before it are how many chances the round had already used.
            self.__observe_proposal(mission.leader, mission.team, mission.team_votes, len(mission.rejected))
            self.update_on_mission(mission.team, self.__fails_required(mission.index), failed=not mission.success, \
                num_fails=mission.num_fails, own_play=own_plays.get(mission.index))

        for number, proposal in enumerate(rejected_this_round):
            self.__observe_proposal(proposal.leader, proposal.team, proposal.votes, number)
        if open_proposal is not None:
            self.__observe_proposal(open_proposal.leader, open_proposal.team, \
                open_proposal.votes, len(rejected_this_round))
        self.rejections = len(rejected_this_round)
        if leader is not None:
            self.leader = leader

    def update_on_mission(self, team: frozenset[int], fails_required: int, failed: bool, num_fails: int = None, \
        own_play: bool = None) -> None:
        """Bayes on one finished mission

        Args:
            fails_required: Fail cards needed to sabotage this mission
            failed: whether it did fail
            num_fails: the Fail cards played, when the table announced them.
                None when the config hides the count, and then only the outcome is read.
            own_play: the card the viewer played, True for Success, None when it was not on the team
        """
        self.__update(lambda world: self.mission_likelihood(world, team, fails_required, failed, num_fails, \
            own_play), f"mission {sorted(team)}, failed={failed}, num_fails={num_fails}")
        # After the update, not before: the mission was played at the old score.
        self.failures += failed
        self.successes += not failed

    def update_on_proposal(self, leader: int, team: frozenset[int], \
        rejections: int = 0) -> None:
        """Bayes on which team a leader put forward

        rejections is how many teams this round has already lost, which says
        both whether THIS team is the last one allowed and whether the next one
        would be. The two are different questions and both are asked.
        """
        self.leader = leader
        self.rejections = rejections
        if not self.read_proposals:
            return
        last_chance = rejections >= MAX_REJECTIONS - 1
        self.__update(lambda world: self.proposal_likelihood(world, leader, team, last_chance), \
            f"Player {leader} proposing {sorted(team)}")

    def update_on_votes(self, team: frozenset[int], votes: dict[int, bool], last_chance: bool = False) -> None:
        """Bayes on how the table voted on one team

        Every player's vote is a separate observation and they are independent
        given the world, so they fold in one at a time. The viewer's own vote is
        skipped: it is not a random variable to the viewer, and at this rung a
        resistance player votes the same way in every world anyway, so it would cancel in the normalisation.
        """
        if not self.read_votes:
            return
        for voter in sorted(votes):
            if voter == self.viewer:
                continue
            self.__update(lambda world, voter=voter: self.vote_likelihood(world, voter, team, votes[voter], \
                last_chance), f"Player {voter} voting {'approve' if votes[voter] else 'reject'} on {sorted(team)}")

    def __observe_proposal(self, leader: int, team: frozenset[int], votes: dict[int, bool], \
        rejections_before: int) -> None:
        """one team put to the vote: who proposed it, then how the table voted

        rejections_before is how many teams this mission had already lost, which
        is what tells vote_likelihood whether this was the last one allowed.
        """
        self.update_on_proposal(leader, team, rejections_before)
        self.update_on_votes(team, votes, rejections_before >= MAX_REJECTIONS - 1)

    def __update(self, likelihood, observed: str) -> None:
        """Bayes on one observation: weigh every world by it, then normalise

        Every channel goes through here, so the rule that a world's weight is its
        prior times P(what happened | that world) is written once.

        Args:
            likelihood: world -> P(what happened | world)
            observed: what happened, for the message if no world can explain it
        """
        posterior = []
        for world, weight in zip(self.worlds, self.weights):
            posterior.append(weight * likelihood(world)) # P(world) * P(observed | world)

        total = sum(posterior)
        if total <= 0.0:
            raise Exception(f"every world was ruled out by {observed}. No deal could have produced it "
                f"under this opponent model (spy_fail_rate={self.spy_fail_rate}, "
                f"resistance_fail_rate={self.resistance_fail_rate}, "
                f"spy_approves_spy_team={self.spy_approves_spy_team}, "
                f"spy_steers_its_team={self.spy_steers_its_team}), so the model is wrong")

        self.weights = [weight / total for weight in posterior] # normalize to sum to 1
        self.__expected_success = None



    def mission_likelihood(self, world: frozenset[int], team: frozenset[int], fails_required: int, \
        failed: bool, num_fails: int = None, own_play: bool = None) -> float:
        """P(this mission result | this world)"""
        distribution = self.__team_fail_counts(world, team, own_play)
        if num_fails is not None:
            counts = smooth(dict(enumerate(distribution)), self.model_epsilon)
            return counts.get(num_fails, 0.0)

        sabotaged = sum(distribution[fails_required:])
        # smooth() over the two outcomes, written out because this is the most
        # called line in the file. The support is both of them unless the rules
        # leave one, and a rule is not a thing to smooth away.
        if 0.0 < sabotaged < 1.0:
            sabotaged = (1.0 - self.model_epsilon) * sabotaged + self.model_epsilon / 2.0
        return sabotaged if failed else 1.0 - sabotaged

    def vote_likelihood(self, world: frozenset[int], voter: int, team: frozenset[int], approved: bool, \
        last_chance: bool = False) -> float:
        """P(this player voted this way | this world)

        A player that is resistance in this world does not know the deal, so
        nothing it does can depend on the deal: it votes one half either way, the
        same number in every world, and cancels. Only a spy conditions on the
        truth, and that asymmetry is the whole reason a vote says anything.

        A spy wants a spy on the mission, so it approves a team that carries one.
        The exception is the last team a mission allows: rejecting that one wins
        the game for the spies immediately, which beats any mission, so there the
        preference flips whatever the team is made of.
        """
        if voter not in world:
            return 0.5 # opponent model

        wants_it_approved = bool(team & world) and not last_chance # opponent model
        approves = self.spy_approves_spy_team if wants_it_approved else 1.0 - self.spy_approves_spy_team # opponent model
        if 0.0 < approves < 1.0: # smooth() over two outcomes, as in mission_likelihood
            approves = (1.0 - self.model_epsilon) * approves + self.model_epsilon / 2.0
        return approves if approved else 1.0 - approves

    def proposal_likelihood(self, world: frozenset[int], leader: int, team: frozenset[int], \
        last_chance: bool = False) -> float:
        """P(this leader proposed this team | this world)

        A resistance leader does not know the deal, so it is modelled as picking
        uniformly among the teams of the required size, the same number in every
        world.

        A spy leader picks in one of two ways, and the two are what is summed:
            steering, with probability spy_steers_its_team, uniformly among the
                teams that carry a spy
            not steering, otherwise, uniformly among every team
        A team carrying a spy can come out of either, so it collects both terms;
        a team carrying none can only come out of the second, so it collects one.
        Summing them marginalises out which way the leader picked, which is the
        thing that is unobserved.
        """
        all_teams = math.comb(self.num_players, len(team))
        if leader not in world:
            return 1.0 / all_teams # opponent model

        # Every team except the ones drawn entirely from the players this world
        # calls resistance carries at least one spy.
        with_a_spy = all_teams - math.comb(self.num_players - len(world), len(team))
        chance = (1.0 - self.spy_steers_its_team) / all_teams # opponent model
        if team & world:
            chance += self.spy_steers_its_team / with_a_spy # opponent model
        # smooth(), written out, because the choice set here is every team of this
        # size and there is no need to enumerate it to know how large it is. The
        # support is all of them, or only the ones carrying a spy if this model
        # steers every single time and so rules the rest out itself.
        if chance <= 0.0:
            return 0.0
        support = all_teams if self.spy_steers_its_team < 1.0 else with_a_spy
        return (1.0 - self.model_epsilon) * chance + self.model_epsilon / support

    def seat_weights(self, seat: int) -> list[float]:
        """these weights as that seat holds them, with the worlds it rules out cut

        A player knows its own card, so its belief is this one conditioned on it,
        which is the same list with every world that calls the seat a spy set to
        zero and the rest renormalised. Measured against a belief really built
        for that seat and replayed on the same public record, the two agree to
        3.3e-16, so a rung modelling a seat needs no second belief to hold.

        A world the seat rules out keeps its place in the list and loses its
        weight, because every caller reads the two lists in step.
        """
        kept = []
        allowed = 0 # worlds that leave this seat out, whatever they now weigh
        total = 0.0
        for world, weight in zip(self.worlds, self.weights):
            if seat in world:
                kept.append(0.0)
                continue
            kept.append(weight)
            allowed += 1
            total += weight
        if allowed == 0:
            raise ValueError(f"every world of this belief holds seat {seat} as a spy, so it "
                "has no belief of its own to read. Only a belief built FOR that seat as a spy "
                "looks like this, and a rung above never models a seat with one")
        if total > 0.0:
            return [weight / total for weight in kept]

        # Uniform, because this belief has ruled out every world the seat
        # allows: it has proved the seat holds a Spy card, and the seat knows
        # better, so all that is left to condition on is its own card. Nothing
        # computed from this reaches an answer either way - the same zero
        # weights multiply it in whatever update asked.
        return [0.0 if seat in world else 1.0 / allowed for world in self.worlds]

    def information_gain(self, team: frozenset[int], fails_required: int, \
        last_chance: bool = False, seat: int = None) -> float:
        """bits this belief expects to lose by putting this team up

        Memoised on gain_signature and not cleared with the weights, because
        this is a pure function of that signature and sync_with_history rebuilds
        from scratch on every observation: the belief walks the same states again
        on every turn, and 77% of the calls in a game were exact repeats.

            IG(T) = I(W; O) = H(W) - sum over o of P(o) H(W | o)
                W   the deal: which players hold a Spy card
                T   the team put up, the argument
                o   what the table then does about T. Either the votes alone, if
                    they reject it, or the votes and the mission result
                V   how every player except this belief's owner votes on T
                M   whether the mission failed

            P(o) = sum over w of P(w) P(V=v | w) P(M=m | w)   if v approves
                    = sum over w of P(w) P(V=v | w)              if v rejects

        P(M=m | w), not P(M=m | w, v): the cards played come from fail_rates,
        which reads the score and the table's belief as they stand now, not this
        round's votes, so M and V are independent given w in this model. A real
        level 2 spy would pick its card after watching the vote move the table.

        The mutual information between the deal and what the table does about
        this team. Maximising it is Bayesian experimental design.

        Jointly and not one channel at a time, because I(W; V, M) is not
        I(W; V) + I(W; M): both channels point at the same worlds. Measured on
        one state the two sum to 0.089 bits where the joint is 0.065.

        The proposal channel is not in the sum. P(this leader put up T | w) is the
        same number in every world when the leader is this belief's owner, so it
        cancels. Which team to put up still changes the gain, through the votes
        and the mission that team draws.
        """
        key = (self.gain_signature(), team, fails_required, last_chance, seat)
        cached = self.__IG_cache.get(key)
        if cached is not None:
            return cached

        weights, owner, rows = self.__vote_outcomes(team, last_chance, seat)
        outcomes = [] # one unnormalised posterior over the worlds per outcome o
        for votes, after_votes in rows:
            # The owner put this team up, so it is counted as approving it.
            approvals = sum(votes) + (owner is not None)
            if self.read_votes and approvals * 2 <= self.num_players:
                outcomes.append(after_votes) # rejected, so there is no mission to watch
                continue
            for failed in (True, False): # mission phase
                after_mission = [] # P(w, V=votes, M=failed)
                for world, weight in zip(self.worlds, after_votes):
                    after_mission.append(weight * self.mission_likelihood(world, team, fails_required, failed))
                outcomes.append(after_mission)

        gain = self.entropy(weights) - self.__expected_entropy(outcomes)
        self.__IG_cache[key] = gain
        return gain

    def __vote_outcomes(self, team: frozenset[int], last_chance: bool, seat: int) -> tuple:
        """P(w, V=v) for every way the other players can vote on this team

            (weights, owner, rows)
                weights  the prior this gain is read against
                owner    whose gain it is, and so the one player not in V
                rows     one (v, P(w, V=v) world by world) pair per vote pattern

        Shared by the two gains that watch one team: what its votes teach on
        their own, and what they teach together with the mission they let
        through. The two have to walk the same patterns in the same order or
        subtracting one from the other means nothing.
        """
        # Whose gain this is. The seat argument asks for it as one player reads
        # it, and that player is then the one whose own vote teaches it nothing.
        owner = self.viewer if seat is None else seat
        weights = self.weights if seat is None else self.seat_weights(seat)
        voters = []
        for player in range(self.num_players):
            if player != owner:
                voters.append(player)
        approves = [] # P(this player approves | this world), one row per world
        for world in self.worlds:
            row = []
            if self.read_votes:
                for voter in voters:
                    row.append(self.vote_likelihood(world, voter, team, True, last_chance))
            approves.append(row)

        rows = []
        for votes in product((True, False), repeat=len(voters) if self.read_votes else 0):
            after_votes = [] # P(w, V=votes), world by world
            for index, weight in enumerate(weights):
                for voter, approved in enumerate(votes):
                    chance = approves[index][voter]
                    weight *= chance if approved else 1.0 - chance # P(this player approves/rejects | this world) x P(this world)
                after_votes.append(weight)
            rows.append((votes, after_votes))
        return weights, owner, rows

    def vote_information_gain(self, team: frozenset[int], last_chance: bool = False, \
        seat: int = None) -> float:
        """bits the votes on this team teach, whichever way this player votes

            IG = I(W; V) = H(W) - sum over v of P(v) H(W | v)
                W  the deal: which players hold a Spy card
                V  how every player except this belief's owner votes on the team

        The votes are cast at once, so this player sees them whether it approves
        or rejects. That makes this term common to the two sides of a vote, and
        a side that carries it while the other does not is a thumb on the scale:
        it is worth 1.7 to 3.4 times the whole rejecting side on its own. Both
        sides carry it now, so it cancels where it should and only what follows
        the votes decides.

        Not subtracted from a PROPOSAL, where it does not cancel: two teams draw
        two different sets of votes, so what they teach is part of what tells
        them apart.
        """
        key = ("votes", self.gain_signature(), team, last_chance, seat)
        cached = self.__IG_cache.get(key)
        if cached is not None:
            return cached

        weights, _, rows = self.__vote_outcomes(team, last_chance, seat)
        outcomes = []
        for _, after_votes in rows:
            outcomes.append(after_votes) # P(w, V=v)
        gain = self.entropy(weights) - self.__expected_entropy(outcomes)
        self.__IG_cache[key] = gain
        return gain

    def mission_information_gain(self, team: frozenset[int], fails_required: int) -> float:
        """bits the mission alone teaches, for a team that is already going out

            IG = I(W; M) = H(W) - sum over m of P(m) H(W | m)
                W  the deal: which players hold a Spy card
                M  whether the mission failed

        What is left to learn at the mission card. By then the team has been
        approved and the votes are public and already folded in, so the votes are
        not an outcome any more and the mission is certain to run. Taking
        information_gain and subtracting vote_information_gain gives neither:
        its votes are a fresh hypothetical draw rather than the ones on the
        table, and it still carries the branch where the team is voted down.
        On one measured state the subtraction reads 0.1560 bits where this reads
        0.5515.

        Like information_gain it does not hold this player's own card fixed. Its
        own Success is certain, so the count it contributes is known, and leaving
        that out overstates what the mission can teach.
        """
        outcomes = []
        for failed in (True, False):
            after_mission = [] # P(w, M=failed)
            for world, weight in zip(self.worlds, self.weights):
                after_mission.append(weight \
                    * self.mission_likelihood(world, team, fails_required, failed))
            outcomes.append(after_mission)
        return self.entropy() - self.__expected_entropy(outcomes)

    def gain_signature(self) -> tuple:
        """everything information_gain reads that can change on one belief

        The memo above is only sound if this names all of it. Anything added to
        this class that information_gain reads and that a caller can change has
        to be added here too, or a stale answer comes back silently. The score is
        in it because a rung that prices the mission card off the value table
        reads it; read_votes because it is an ablation switch and the tests flip
        it on a belief that has already answered; rejections because a rung above
        prices a rejection by whether the NEXT team is the last one a mission
        allows, and vote_likelihood reads that through rejection_information_gain.
        """
        return (tuple(self.weights), self.successes, self.failures, self.read_votes, \
            self.rejections)

    def rejection_information_gain(self, leader: int, seat: int = None) -> float:
        """bits this belief expects to lose by rejecting and seeing what comes next

            IG = H(W) - sum over t of P(t) H(W | t)
                W       the deal
                leader  the player that proposes after this rejection, the argument
                t       the team it puts up
                seat    read the gain as that seat reads it, over the worlds it
                        allows. None reads it as whoever holds this belief

        The other half of a team vote: without it, the vote compares a measured
        gain against a zero.

        Only the proposal itself, not the votes and mission that would follow it.
        Those depend on a team this belief cannot see yet.

        The team that comes next may itself be the last one a mission allows,
        and then its leader names it into a vote that ends the game rather than
        an ordinary one, so it names a different team. self.rejections is what
        says so: this is asked while voting on the team after that many, so the
        next one is number rejections + 2. Worth 0.0919 bits on one measured
        rung 2 state, where the whole term is 0.1983.
        """
        preset = PLAYER_PRESET[self.num_players]
        index = min(self.successes + self.failures, len(preset.team_sizes) - 1)
        owner = self.viewer if seat is None else seat
        if not self.read_proposals or leader == owner:
            return 0.0 # its own proposal is the same number in every world

        next_is_last = self.rejections + 1 >= MAX_REJECTIONS - 1
        weights = self.weights if seat is None else self.seat_weights(seat)
        outcomes = []
        for candidate in combinations(range(self.num_players), preset.team_sizes[index]):
            option = frozenset(candidate)
            after_proposal = [] # P(w, team=option)
            for world, weight in zip(self.worlds, weights):
                after_proposal.append(weight \
                    * self.proposal_likelihood(world, leader, option, next_is_last))
            outcomes.append(after_proposal)

        return self.entropy(weights) - self.__expected_entropy(outcomes)

    def __expected_entropy(self, outcomes: list[list[float]]) -> float:
        """sum over o of P(o) H(W | o), from one unnormalised posterior per outcome

        Each row holds P(w, o) for every world w. Its sum is P(o), and dividing
        by that sum gives P(w | o). The rows partition the outcomes, so the sums
        are a distribution and no further normalising is needed.
        """
        expected = 0.0
        for unposteriors in outcomes:
            marginal = sum(unposteriors)
            if marginal < 0.0:
                raise Exception(f"P(outcome) came out negative ({marginal}), which a product of "
                    "probabilities cannot be, so a likelihood is wrong")
            if marginal == 0.0:
                continue # no world can produce this outcome, so it contributes P(o) H = 0
            entropy = 0.0
            for unposterior in unposteriors:
                if unposterior > 0.0:
                    posterior = unposterior / marginal
                    entropy -= posterior * math.log2(posterior)
            expected += marginal * entropy
        return expected

    def expected_mission_successes(self) -> tuple[float, ...]:
        """P(a mission succeeds) for average team, one per mission

        The input lookahead.py's value table takes, and reading it here is what
        keeps it from being a hyperparameter: "a mission with no team chosen yet"
        is an average team of the size that mission asks for, and this belief
        already has an opinion about every one of them.

        One per mission and not one number, because a six player game sends teams
        of 2, 3, 4, 3 and 4, and an average four player team out of six almost
        always carries a spy where a two player one often does not.

        The average over teams is taken by how many spies a team holds and not
        team by team, which is the same number here because fail_rates at this
        rung does not depend on which team is asked. A rung whose rates do
        depend on the team overrides this. It matters: sync_with_history rebuilds
        this on every observation, and team by team it was the slowest thing in
        the file at eight seconds of every fourteen.
        """
        if self.__expected_success is not None:
            return self.__expected_success

        preset = PLAYER_PRESET[self.num_players]
        by_size = {}
        chances = []
        for index, size in enumerate(preset.team_sizes):
            key = (size, preset.fails_required[index])
            if key not in by_size:
                by_size[key] = 1.0 - self.__average_fails(size, preset.fails_required[index])
            chances.append(by_size[key])

        self.__expected_success = tuple(chances)
        return self.__expected_success

    def __average_fails(self, size: int, fails_required: int) -> float:
        """P(a mission fails) averaged over every team of this size

            sum over w of P(w) sum over j of H(j) P(fails | j spies, size - j others)
                w     a world this belief still allows
                j     how many of the team hold a Spy card
                H(j)  C(spies in w, j) C(the rest, size - j) / C(num_players, size),
                        the chance a team drawn at random holds exactly j of them

        H is the hypergeometric distribution. Grouping by j works because at this
        rung the spy count is a sufficient statistic for the team.
        """
        total_teams = math.comb(self.num_players, size)
        fails = 0.0
        for world, weight in zip(self.worlds, self.weights):
            for spies in range(min(len(world), size) + 1):
                others = size - spies
                if others > self.num_players - len(world):
                    continue # not enough non spies to fill the team
                share = math.comb(len(world), spies) * math.comb(self.num_players - len(world), others)
                spy_rate, resistance_rate = self._average_team_rates(size, fails_required, spies)
                distribution = self._fail_counts_for(spies, others, 0, spy_rate, resistance_rate)
                fails += weight * (share / total_teams) * sum(distribution[fails_required:])
        return fails

    def _average_team_rates(self, size: int, fails_required: int, spies: int) -> tuple[float, float]:
        """P(plays the Fail card) for a spy and for a resistance player on an average team

        At this rung the rates do not depend on the team, so every argument is
        ignored. A rung whose spy rate depends on the team size, the Fail cards
        needed or how many spies are on it overrides this.
        """
        return self.fail_rates(None, None)

    def mission_fails(self, team: frozenset[int], fails_required: int, \
        own_play: bool = None) -> float:
        """P(this team's mission fails), over every world this belief still allows

        The posterior predictive: a prediction, not a belief about the deal.

            sum over w of P(w) P(this mission fails | w)
                w a world this belief still allows
                own_play  the card the viewer plays, when it is on the team. None
                    leaves its own card a random variable like anybody
                        else's, which is what a viewer asking about a team it has not been put on wants

        What a one step payoff is read off. The complement is the chance the
        mission succeeds, which is what the resistance is trying to buy.
        """
        chance = 0.0
        for world, weight in zip(self.worlds, self.weights):
            chance += weight * self.mission_likelihood(world, team, fails_required, failed=True, \
                own_play=own_play)
        return chance

    def spy_probability(self) -> dict[int, float]:
        """P(this player holds a Spy card), one entry per player id

        The shape ResistanceAgent.identity_belief returns, so it goes straight
        into beliefs.csv next to GRAIL's.
        """
        probability = {}
        for player_id in range(self.num_players):
            probability[player_id] = 0.0
        for world, weight in zip(self.worlds, self.weights):
            for player_id in world:
                probability[player_id] += weight
        return probability

    def most_likely_world(self) -> frozenset[int]:
        """the deal this belief thinks most likely, ties broken by player id"""
        best = None
        best_weight = -1.0
        for world in sorted(self.worlds, key=sorted):
            weight = self.weights[self.worlds.index(world)]
            if weight > best_weight:
                best, best_weight = world, weight
        return best

    def entropy(self, weights: list[float] = None) -> float:
        """Shannon entropy over the worlds, in bits

            H(W) = - sum over w of P(w) log2 P(w)
                W        the deal
                w        a world this belief still allows
                weights  P(w) to measure over. None is this belief's own, and a
                         seat's own weighting is what seat_weights returns

        What an information gain is measured against: it starts at log2 of the
        number of possible deals and falls as the missions rule them out.
        """
        total = 0.0
        for weight in (self.weights if weights is None else weights):
            if weight > 0.0:
                total -= weight * math.log2(weight)
        return total

    def __possible_worlds(self) -> list[frozenset[int]]:
        """every deal the viewer cannot already rule out

        The public belief rules out nothing. A viewer knows its own card, which
        halves the count or better: a resistance player in a six player game is
        left with C(5,2) = 10 of the 15 deals.
        """
        worlds = []
        for spies in combinations(range(self.num_players), self.num_spies):
            world = frozenset(spies)
            if self.viewer is not None and (self.viewer in world) != self.viewer_is_spy:
                continue
            worlds.append(world)
        if not worlds:
            raise Exception(f"no deal of {self.num_spies} spies among {self.num_players} players "
                f"agrees with viewer={self.viewer}, viewer_is_spy={self.viewer_is_spy}")
        return worlds

    def __team_fail_counts(self, world: frozenset[int], team: frozenset[int], own_play: bool) -> list[float]:
        """P(exactly m Fail cards) for this team in this world, m from 0"""
        spies = len(team & world)
        others = len(team) - spies
        known_fails = 0
        mine = None
        # The viewer knows its own card and knows what it played, so it takes
        # itself out of the two counts and adds what it did as a certainty.
        if own_play is not None and self.viewer is not None and self.viewer in team:
            if self.viewer in world:
                spies -= 1
            else:
                others -= 1
            known_fails = 0 if own_play else 1
        elif self.viewer is not None and self.viewer in team and self.viewer in world:
            # A spy of ours still to play. Its card is its OWN rung's, while the
            # rest of the team is what it predicts of them, one rung below that.
            # One rate for both would price its own card as somebody else's.
            spies -= 1
            mine = self.own_fail_rate(team, len(team & world))
        # The rates are asked for the whole team, before own_play takes this
        # player out of the counts, because how many spies are on it is what
        # decides whether one more Fail card changes anything.
        spy_rate, other_rate = self.fail_rates(team, len(team & world))
        return self._fail_counts_for(spies, others, known_fails, spy_rate, other_rate, mine)

    def fail_rates(self, team: frozenset[int], spies: int = None) -> tuple[float, float]:
        """P(plays the Fail card) for a spy and for a resistance player on this team

        Args:
            team: who is on the mission
            spies: how many of them hold a Spy card, which a rung above needs to
                know whether one more Fail card would change anything

        A method and not two attributes so that a rung above can derive them
        instead of being handed them. At this rung they are the constants, so
        neither argument is read and both may be None.
        """
        return self.spy_fail_rate, self.resistance_fail_rate # opponent model

    def own_fail_rate(self, team: frozenset[int], spies: int = None) -> float:
        """P(this belief's own holder plays the Fail card), it being a spy

        fail_rates answers for a MODELLED spy. At this rung the two are the same
        constant, because nothing here is derived from anything. A rung above
        derives them and they part company: see StrategicKripkeState.
        """
        return self.fail_rates(team, spies)[0]

    def _fail_counts_for(self, spies: int, others: int, known_fails: int, spy_rate: float, \
        other_rate: float, own_rate: float = None) -> list[float]:
        """P(exactly m Fail cards) from this many players of each kind

            P(m) = the convolution of one Bernoulli per player on the team
                m           Fail cards played
                spies       team members holding a Spy card, at spy_rate each
                others      the rest, at other_rate each
                known_fails cards the viewer already knows were Fail
                own_rate    one more player at this rate, which is this belief's
                            own holder when it is a spy on the team and has not
                            played yet. Its card is its own rung's and the rest
                            are what it predicts of them, so the two rates differ

        Every player plays its card independently, which is what makes the count
        a convolution rather than a sum over assignments.
        """
        key = (spies, others, known_fails, spy_rate, other_rate, own_rate)
        cached = self.__fail_counts.get(key)
        if cached is not None:
            return cached

        groups = [(spies, spy_rate), (others, other_rate)]
        if own_rate is not None:
            groups.append((1, own_rate))
        distribution = [0.0] * known_fails + [1.0]
        for count, rate in groups:
            for _ in range(count):
                grown = [0.0] * (len(distribution) + 1)
                for played, probability in enumerate(distribution):
                    grown[played] += probability * (1.0 - rate)
                    grown[played + 1] += probability * rate
                distribution = grown

        self.__fail_counts[key] = distribution
        return distribution

    def __fails_required(self, mission_index: int) -> int:
        """Fail cards needed to sabotage the mission at this index"""
        return PLAYER_PRESET[self.num_players].fails_required[mission_index]


class RandomKripkeState(KripkeState):
    """the rung below the rung below: the opponents draw uniformly

    KripkeState is called uniform but is not: its four constants say a spy plays
    Fail 0.8 of the time and steers its team 0.8 of the time, which describes a
    player trying to win. This one describes a player drawing from
    valid_actions, which is exactly what RandomResistanceAgent does, so against
    the random floor it is not an approximation.

    resistance_fail_rate stays at the class default rather than 0.5, because our
    own resistance seats play Success by choice. That asymmetry is the only thing
    left for the belief to read: with both sides at 0.5 the mission channel would
    carry nothing at all, since P(the mission fails) is 1 - 2^-|team| whatever
    the deal is. Set it to 0.5 to describe a table of random players on both
    sides, and the belief will correctly learn nothing.

    The vote and proposal channels carry nothing here by construction: a number
    that is the same in every world cancels out of Bayes. That is the point.
    Reading a channel with a model that is wrong is worse than not reading it.
    """

    OPPONENT_MODEL = "random"

    def __init__(self, num_players: int, viewer: int = None, viewer_is_spy: bool = False, \
        spy_fail_rate: float = 0.5, spy_approves_spy_team: float = 0.5, \
            spy_steers_its_team: float = 0.0, resistance_fail_rate: float = 0.0, **options):
        super().__init__(num_players, viewer=viewer, viewer_is_spy=viewer_is_spy, \
            spy_fail_rate=spy_fail_rate, resistance_fail_rate=resistance_fail_rate, \
                spy_approves_spy_team=spy_approves_spy_team, \
                    spy_steers_its_team=spy_steers_its_team, **options)

class StrategicKripkeState(KripkeState):
    """the rung above: every player maximises a value, and no constant describes one

    Every player is modelled as choosing the action worth the most, "worth" comes
    from lookahead.py's value table, and the belief it reads that value through
    is the rung below, held in self.shadow.

        value(a) = gain(a) - hide_weight x exposure(a)
            a           the action: a vote, a team to put up, or a mission card
            gain(a)     what a does to P(this player's faction wins), divided by
                        swing(score) so that one unit is one mission
            exposure(a) P(the table calls this player a spy | it did a)
            hide_weight what one unit of exposure costs, in missions

    Both factions use that one line, and differ only in what they know: a spy
    reads P(this mission fails) off the world being scored, a resistance player
    off the shadow with its own card ruled in.

    Order 2 in the naming docs/opponent_models.md uses. order=3 builds the shadow
    out of this same class at order 2, with no new code.
    """

    OPPONENT_MODEL = "strategic"

    def __init__(self, num_players: int, viewer: int = None, viewer_is_spy: bool = False, \
        beta: float = PLAYER_SHARPNESS, hide_weight: float = BEING_SUSPECTED_COSTS, \
            order: int = 2, spy_hide_budget: float = SPY_HIDE_BUDGET, \
                greedy: bool = False, **options):
        """
        beta: the quantal response precision, an inverse temperature and not a
            learning rate: how sharply a player picks the better option, in odds
            per mission of value. Kept finite: a deterministic model rules a world out on a
            mismatch instead of only downweighting it. Ignored when greedy.
        hide_weight: what one unit of the table's suspicion costs, in missions.
        order: 2 gives the shadow the rung below, a KripkeState. 3 gives it
            another StrategicKripkeState at order 2, and so on. Level-k thinking
            over what is formally a finitely nested I-POMDP, since what nests is
            a belief about beliefs and not a strategy.
        spy_hide_budget: the most value, in missions, a modelled SPY will give
            up to keep the table guessing. A spy saw the deal, so a mission
            teaches it nothing, but it teaches the table, and this is the wish
            for the table to stay confused, bounded so that wanting it can never
            cost more than the budget. See SPY_HIDE_BUDGET and __hidden.

            A modelled RESISTANCE player's information term is not here, because
            it is not a knob: it is info_weight, that player's own policy, over
            swing. See bits_are_worth.
        greedy: pick the best option with probability 1 - model_epsilon and
            uniformly otherwise, instead of a softmax. Needs model_epsilon above
            0, or the model would be deterministic and rule worlds out.

            Measured the same way: 62.1% at epsilon 0.1 and 54.8% at 0.25,
            against the softmax's 73.8%. It grades options by which is best and
            not by how much, so it says 0.95 to one of two options a hair apart,
            and against opponents that are not maximising at all that is a
            confident wrong answer forty times a game.
        """
        super().__init__(num_players, viewer=viewer, viewer_is_spy=viewer_is_spy, **options)
        if beta < 0.0:
            raise ValueError(f"beta must not be negative, got {beta}")
        if hide_weight < 0.0:
            raise ValueError(f"hide_weight must not be negative, got {hide_weight}")
        if order < 2:
            raise ValueError(f"order must be at least 2, got {order}. Order 1 is KripkeState")
        if spy_hide_budget < 0.0:
            raise ValueError(f"spy_hide_budget must not be negative, got {spy_hide_budget}")
        if greedy and self.model_epsilon <= 0.0:
            raise ValueError("greedy needs model_epsilon above 0, or a modelled player would be "
                "deterministic and rule a world out the moment it did something unexpected")
        self.beta = beta
        self.hide_weight = hide_weight
        self.order = order
        self.spy_hide_budget = spy_hide_budget
        self.greedy = greedy
        # The table, which every player here weighs itself against. No viewer,
        # because it is nobody's private belief. viewer_is_spy is passed although
        # viewer=None makes it moot, so that a later change to how the worlds are
        # filtered cannot hand the shadow a card the table was never shown.
        if order > 2:
            # Every knob of this rung, not some of them: the rung below is the
            # same construction one step down, and a shadow that dropped half of
            # them would model a player this line-up does not contain.
            self.shadow = StrategicKripkeState(num_players, viewer=None, viewer_is_spy=False, \
                beta=beta, hide_weight=hide_weight, order=order - 1, \
                    spy_hide_budget=spy_hide_budget, greedy=greedy, **options)
        else:
            self.shadow = KripkeState(num_players, viewer=None, viewer_is_spy=False, **options)
        self.__clear()

    def __clear(self) -> None:
        """everything derived from the shadow, which every observation moves"""
        self.__exposures: dict[tuple, float] = {}
        self.__public_fails: dict[tuple, float] = {}
        self.__rates: dict[frozenset[int], tuple[float, float]] = {}
        self.__votes: dict[tuple, dict] = {}
        self.__proposals: dict[tuple, dict] = {}
        self.__IG_cache: dict[frozenset[int], float] = {}
        self.__rejection_gains: dict[int, float] = {}
        self.__own_rates: dict[tuple, float] = {}
        self.__owner_success = None

    def reset(self) -> None:
        super().reset()
        if getattr(self, "shadow", None) is not None: # __init__ resets before the shadow exists
            self.shadow.reset()
            self.__clear()

    def update_on_mission(self, team: frozenset[int], fails_required: int, failed: bool, \
        num_fails: int = None, own_play: bool = None) -> None:
        super().update_on_mission(team, fails_required, failed, num_fails, own_play)
        self.shadow.update_on_mission(team, fails_required, failed, num_fails) # own_play is private
        self.__clear()

    def update_on_proposal(self, leader: int, team: frozenset[int], \
        rejections: int = 0) -> None:
        super().update_on_proposal(leader, team, rejections)
        self.shadow.update_on_proposal(leader, team, rejections)
        self.__clear()

    def sync_with_history(self, missions: list[MissionRecord], rejected_this_round: list[ProposalRecord] = (), \
        own_plays: dict[int, bool] = None, leader: int = None, \
            open_proposal: ProposalRecord = None) -> None:
        # Who leads now reaches the shadow with everything else. A rung 3 shadow
        # is a StrategicKripkeState and prices a rejection by what the next
        # leader would put up, so a stale leader there is a stale price.
        super().sync_with_history(missions, rejected_this_round, own_plays, leader, open_proposal)
        shadow = self.shadow
        while shadow is not None:
            shadow.rejections = self.rejections
            if leader is not None:
                shadow.leader = leader
            shadow = getattr(shadow, "shadow", None)

    def update_on_votes(self, team: frozenset[int], votes: dict[int, bool], \
        last_chance: bool = False) -> None:
        # Read the votes, then let the table read them: a player chose its vote
        # against the table as it stood before the vote.
        super().update_on_votes(team, votes, last_chance)
        self.shadow.update_on_votes(team, votes, last_chance)
        self.__clear()

    # the one decision rule, in three channels

    def vote_likelihood(self, world: frozenset[int], voter: int, team: frozenset[int], \
        approved: bool, last_chance: bool = False) -> float:
        """P(this player voted this way | this world), for a player that maximises

            P(vote) = response({approve: value(approve), reject: value(reject)})
        """
        # Keyed by the world only for a spy. A resistance player cannot condition
        # on the deal, so its answer is the same in every world it could be
        # resistance in.
        key = (world if voter in world else None, voter, team, last_chance)
        chances = self.__votes.get(key)
        if chances is None:
            values = {}
            values[True] = self.action_value(world, voter, team, "approve", last_chance)
            values[False] = self.action_value(world, voter, team, "reject", last_chance)
            if voter in world and self.spy_hide_budget > 0.0:
                # Rejecting the last team a mission allows ends the game, so
                # there is no next proposal left for the table to read.
                gains = {True: self.__gain(team, last_chance),
                    False: 0.0 if last_chance else self.__votes_gain(team) + self.__rejection_gain()}
                values = self.__hidden(values, gains)
            chances = self.__response(values)
            self.__votes[key] = chances
        return chances[approved]

    def proposal_likelihood(self, world: frozenset[int], leader: int, team: frozenset[int], \
        last_chance: bool = False) -> float:
        """P(this leader proposed this team | this world), for a player that maximises

            P(t) = response({t: value(propose t) for every team t of this size})

        A leader is modelled as sure its team will be approved, so this is a
        choice among teams and not a second vote.
        """
        key = (world if leader in world else None, leader, len(team), last_chance) # as in vote_likelihood
        chances = self.__proposals.get(key)
        if chances is None:
            values = {}
            for candidate in combinations(range(self.num_players), len(team)):
                option = frozenset(candidate)
                values[option] = self.action_value(world, leader, option, "propose", last_chance)
            if leader in world and self.spy_hide_budget > 0.0:
                gains = {}
                for option in values:
                    gains[option] = self.__gain(option, last_chance)
                values = self.__hidden(values, gains)
            chances = self.__response(values)
            self.__proposals[key] = chances
        return chances[team]

    def fail_rates(self, team: frozenset[int], spies: int = None) -> tuple[float, float]:
        """P(player plays the Fail card) for a spy and for a resistance player on this team

            value(Fail) = pivotal(team, spies) x (1 - hide_weight x team_exposure(team))
            value(Success) = 0
            P(Fail) = response({Fail: value(Fail), Success: value(Success)})   spy
                = resistance_fail_rate, which is 0                 resistance

        pivotal is P(the rest of the team play exactly fails_required - 1 Fail
        cards), which is when one more card changes the outcome and the only
        time it is worth anything. It scales what being seen costs as well as
        what the mission is worth, and for one reason: a card that changes
        nothing leaves the table looking at the same outcome either way, so it
        gives nothing away either. Only the decisive card both wins a mission
        and points at whoever played it. A lone spy on a mission needing two Fail
        cards is wasting its card 89% of the time; a second spy on a mission
        needing one is wasting it 91% of the time, because its partner is
        already sinking the mission. Pinned at 1, as it was, both of those
        played Fail as eagerly as a spy whose card decides.

        Two softmaxes, one per faction, and not one softmax across the two. The
        choice a player makes here is which card to play; which faction it is in
        was dealt to it. Whether it is a spy at all is the belief's question, and
        Bayes over the worlds answers it. So this returns a pair, and
        _fail_counts_for convolves the spy rate over the spies on the team and
        the resistance rate over the rest: the two rates never compete.

        value(Success) is 0 because a softmax reads only differences, so one of
        the two options can be put anywhere and the other measured from it.
        Both options still have to be passed, or there is nothing to normalise
        against and response returns 1.

        Only the spy branch is derived, because only a spy has a card to choose:
        a resistance player plays Success by the rules of the simplified game, so
        its rate is read off resistance_fail_rate rather than softmaxed.

        That is also what ends an old asymmetry here. This used to price a
        resistance Fail card at a flat -1 while the spy got pivotal, and pricing
        it properly instead made things worse, not better: a resistance player on
        a doomed team is near indifferent, so a softmax handed its Fail card a
        third of the mass, the two rates went to 0.596 and 0.334, and the gap
        between a team with a spy and a clean one, which is the whole of what the
        mission channel reads, fell from 0.748 to 0.629. With the card fixed by
        rule there is nothing left to price either way.

        Two approximations are left, both from treating the cards on a team as
        independent, which is what lets the count of them stay a convolution:

        1. Every spy on the team gets the same rate. What a failed mission gives
            away differs between team members, and each is charged the mean.
        2. The spies on a team are solved for together, by substituting the rate
            back into pivotal until it settles. Every spy on the team is given
            the same rate, so this is a mean field and not a full equilibrium.
        """
        if self.successes >= MISSIONS_TO_WIN or self.failures >= MISSIONS_TO_WIN:
            return super().fail_rates(team, spies) # the game is over, nothing left to price
        key = (team, spies)
        cached = self.__rates.get(key)
        if cached is not None:
            return cached # asked once per world, and the answer depends only on the count

        # "fail" and "success" spelled out, because True means the Success card
        # in every mission_vote action and would mean the Fail card here.
        # What a DECISIVE Fail card is worth: one mission, less what the failure
        # it causes gives away. A card that is not decisive is worth neither.
        decisive = 1.0 - self.hide_weight * self.__team_exposure(team)
        # Not derived, because it is not a choice: a resistance player plays
        # Success by the rules of the simplified game. See RESISTANCE_FAIL_RATE.
        resistance = self.resistance_fail_rate
        if spies:
            # What the modelled player believes its team mates play, and that is
            # the rung below it: a rung k belief models every other player at
            # k-1, in this channel exactly as __believed_fails does in the vote.
            # This used to substitute THIS rung's rate back into itself until it
            # settled, which modelled a spy that thinks its team mates are its
            # own rung. Two channels of one modelled player then answered the
            # same question two ways: 0.6824 here against the 0.8000 its vote
            # was given.
            below_spy, below_resistance = self.shadow.fail_rates(team, spies)
            decisive *= self.__pivotal(team, spies, below_spy, below_resistance)
        spy = self.__response({"fail": decisive, "success": 0.0})["fail"] # opponent model
        rates = (spy, resistance)
        self.__rates[key] = rates
        return rates

    def own_fail_rate(self, team: frozenset[int], spies: int = None) -> float:
        """P(this belief's own holder plays the Fail card), it being a spy

            value(Fail) = pivotal(team mates at fail_rates) x (1 - hide x exposure)

        fail_rates answers for a MODELLED spy, which is a rung k-1 player, so its
        pivotal reads the rung BELOW for its team mates. The holder of this
        belief is rung k, so its team mates are the rung k reading, and that is
        fail_rates itself. One response step above, and closed form, because
        fail_rates no longer chases itself.

        One measured team of three holding two spies: a modelled spy plays 0.6072
        where the holder plays 0.7017, the gap being team mates read at 0.8000
        against 0.6072.
        """
        if not spies or self.successes >= MISSIONS_TO_WIN or self.failures >= MISSIONS_TO_WIN:
            return self.fail_rates(team, spies)[0]
        key = (team, spies)
        cached = self.__own_rates.get(key)
        if cached is not None:
            return cached

        decisive = 1.0 - self.hide_weight * self.__team_exposure(team)
        modelled, resistance = self.fail_rates(team, spies)
        decisive *= self.__pivotal(team, spies, modelled, resistance)
        rate = self.__response({"fail": decisive, "success": 0.0})["fail"]
        self.__own_rates[key] = rate
        return rate

    def __pivotal(self, team: frozenset[int], spies: int, spy_rate: float, resistance_rate: float) -> float:
        """P(one more Fail card changes whether this mission fails)

            P(the rest of the team play exactly fails_required - 1 Fail cards)
                the rest  spies - 1 other spies at spy_rate, and len(team) - spies others at resistance_rate

        The partners' rate comes from the rung below, so this is read once and
        not substituted back into itself. See fail_rates.

        Decisiveness, from voting theory: P(my action changes the outcome).
        """
        required = self.__fails_required_now()
        distribution = self._fail_counts_for(spies - 1, len(team) - spies, 0, spy_rate, resistance_rate)
        return distribution[required - 1] if required - 1 < len(distribution) else 0.0

    # the same rule read as a policy, for an agent that wants to be one

    def sample_vote(self, world: frozenset[int], voter: int, team: frozenset[int], \
        rng: Random, last_chance: bool = False) -> bool:
        """a very simple policy for approve or reject, drawn from what this rung says this player does

        An agent calling this is not imitating the opponent model, it is the
        opponent model. A spy that plays differently from the spy the resistance
        models has added a rung nobody described.

        Drawn and not maximised, on all three channels. The argmax throws away
        the noise that made the likelihood a distribution: taken that way the spy
        played Fail 157 times out of 157 where its own model says 0.88, and the
        resistance beat those spies 150 games out of 150.

        rng is passed in rather than held, so a belief owns no random state and
        the caller keeps one stream.
        """
        return rng.random() < self.vote_likelihood(world, voter, team, True, last_chance)

    def sample_team(self, world: frozenset[int], leader: int, size: int, rng: Random, \
        last_chance: bool = False) -> frozenset[int]:
        """the team of this size this leader puts up, drawn from proposal_likelihood

        The most likely team is worth about 7% of the distribution, so a spy that
        always named it would be a spy the resistance could read off a table.
        """
        chances = {}
        for candidate in combinations(range(self.num_players), size):
            option = frozenset(candidate)
            chances[option] = self.proposal_likelihood(world, leader, option, last_chance)

        # Inverse transform sampling: walk the teams, keeping a running total,
        # and take the one the draw lands in. chosen holds the last team walked,
        # which is what comes back if the total ends a hair under 1, and that is
        # rounding rather than a choice.
        draw = rng.random()
        total = 0.0
        chosen = None
        for option, chance in chances.items():
            total += chance
            chosen = option
            if draw < total:
                break
        return chosen

    def sample_card(self, world: frozenset[int], player: int, team: frozenset[int], rng: Random) -> bool:
        """Success (True) or Fail, drawn from the rate that belongs to this player

        Its own rung when the agent holding this belief is the one playing,
        fail_rates when it is simulating somebody else. The two differ by what
        the player is assumed to believe of its team mates: see own_fail_rate.
        """
        spies = len(team & world)
        if player not in world:
            fails = self.fail_rates(team, spies)[1]
        elif player == self.viewer:
            fails = self.own_fail_rate(team, spies)
        else:
            fails = self.fail_rates(team, spies)[0]
        return not rng.random() < fails

    # what an action is worth, in missions

    def action_value(self, world: frozenset[int], player: int, team: frozenset[int], action: str, \
        last_chance: bool = False) -> float:
        """value(this action) for this player, in missions

            value(a) = gain(a) - hide_weight x exposure(a)
                a            "approve", "reject" or "propose"
                gain(a)      what a does to P(this player's faction wins), in
                             missions. See __action_gain
                exposure(a)  P(the table calls this player a spy | it did a).
                             See exposure

        One function for the two channels a player chooses among alternatives,
        and fail_rates writes the same two lines for the third. An agent that
        wants to BE this model calls this, so that a rung above changes what a
        player wants and never how it is priced.

        A signalling game utility: a payoff, minus the cost of the belief the
        action induces in whoever is watching. hide_weight is the exchange rate
        between the two, which is why the ladder cannot derive it.
        """
        if action not in ("approve", "reject", "propose"):
            raise ValueError(f"action must be approve, reject or propose, got {action!r}")
        gain = self.__action_gain(world, player, team, action, last_chance)
        if last_chance and action in ("approve", "reject"):
            # On the last team a mission allows both factions have a dominant
            # vote: rejecting loses the game outright, so the resistance can only
            # approve and a spy can only reject. Nothing the table thinks can
            # reverse a dominated action, so pricing it here can only add error.
            # The agent drops the term too, in __team_vote_eu, and the two have
            # to agree. A PROPOSAL is not dominated and keeps it.
            return gain
        return gain - self.hide_weight * self.exposure(player, team, action, last_chance)

    def __action_gain(self, world: frozenset[int], player: int, team: frozenset[int], \
        action: str, last_chance: bool) -> float:
        """what this action does to P(this player's faction wins), in missions

            approve, propose    q - p
                q   P(this team's mission succeeds), as this player sees it
                p   P(an average team of this size succeeds)
            reject              0                 the score does not move
            reject, last team   -V / swing        the game ends and is lost
                V       P(the resistance eventually wins) at the score now
                swing   V(one more success) - V(one more failure), which is what
                        one mission is worth here and so is the unit

        Written for the resistance and flipped for a spy, because the two want
        the same number to go opposite ways.
        """
        if action == "reject":
            if last_chance:
                gain = -self.__win_now() / self.__swing()
            else:
                # The score does not move, but the votes are cast and the next
                # leader still has to name a team, and both are watched.
                return self.__reject_information(player, world, team)
        else:
            gain = self.__mission_gain(world, player, team) \
                + self.__information(world, player, team, last_chance)
            return gain # __mission_gain has already flipped the sign
        return -gain if player in world else gain

    def __mission_gain(self, world: frozenset[int], player: int, team: frozenset[int]) -> float:
        """what running this team's mission is worth to this player, in missions

            lookahead.team_gain(score, 1 - __believed_fails(world, player, team))

        team_gain is lookahead.team_value divided by swing, and team_value is the
        line the agent scores a team with too. Divided, because beta multiplies
        this and a sharpness has to mean the same thing at every score.

        THE CLAIM this makes is that a rung k model of a seat is what a rung k-1
        AGENT sitting in that seat would decide. Measured by building that agent
        and handing it the shadow, it agrees on 98.4% of votes, worst gap 0.50
        missions. It was 65.3% before the information term moved here.
        docs/tom_agent.md has what the remaining 1.6% is made of.

        q is read off the shadow restricted to the worlds this player allows.
        A rung k-1 agent in that seat holds its own belief: the same prior over
        the same worlds updated on the same public history, so the two are the
        same number. Measured over 3297 modelled resistance seats in 40 games,
        for a seat that has played no mission card yet, the worst gap between
        them is 5.6e-16, which is float addition order and nothing else.

        Once the seat HAS played a card the two part company, by 0.003 on average
        and 0.093 at worst, because that card is private and a model cannot read
        it. That is the game's information structure, not this file's.

        p, the average team, agrees to 0.0000 always: every world holds the same
        number of spies, so a team drawn at random carries the same spy count
        distribution whichever world is true, and at this rung p does not depend
        on the belief at all.

        What is left is the ladder doing its job. Our agent is rung 2 and the
        model of it is rung 1, worth 0.101 on q and 0.030 on p, and that gap is
        the point: a rung k model of a player is a rung k-1 player.

        What is left, apart from argmax against softmax, which cannot be closed:
        __information reads the PUBLIC shadow where a player would condition it
        on its own card, worth 0.12 bits; rung 1 has no shadow and so cannot
        price exposure at all, which is why order 2 cannot test the claim; and
        bits_are_worth divides by swing, which is unbounded as swing goes to
        zero. fail_rates is deliberately not aligned: pricing a resistance Fail
        card by pivotal was measured to collapse the mission channel.
        """
        succeeds = 1.0 - self.__believed_fails(world, player, team)
        gain = team_gain(self.successes, self.failures, succeeds, self.__chance())
        return -gain if player in world else gain

    def __information(self, world: frozenset[int], player: int, team: frozenset[int], \
        last_chance: bool = False) -> float:
        """what running this team would teach this player, in missions

            bits_are_worth() x IG(team)      a resistance player
            0                                a spy, which learns nothing from a
                                             deal it was shown. What a spy wants
                                             is the TABLE to learn nothing, and
                                             that is bounded, so it is priced
                                             once per decision in __hidden and
                                             not once per action here
                IG(team)   bits the SHADOW expects to lose by putting this team
                           up. The shadow is one rung below this one, whatever
                           the order: a KripkeState at order 2, another
                           StrategicKripkeState at order 3

        The resistance half is DERIVED from info_weight, not a knob of its own,
        which is what makes this the modelled player's own policy rather than a
        second opinion about it. See bits_are_worth.

        The spy half is a knob, off by default and measured to be a bad idea:
        the cheapest way to stop teaching the table is to reject, so a spy given
        it rejects itself into the open. README.md has the numbers.

        The shadow is the PUBLIC belief, not the modelled player's own, so this
        is read as THIS PLAYER reads it: the shadow restricted to the worlds its
        card allows, which is what a rung k-1 belief of that seat holds. The
        payoff half conditions the same way, so both halves of one modelled
        player's value now come from one belief. Measured on one history the six
        seats run 0.500 to 0.589 bits where the table alone reads 0.561.
        """
        if player in world:
            return 0.0 # a spy was shown the deal, so no mission can teach it one
        weight = self.bits_are_worth()
        if weight == 0.0:
            return 0.0
        return weight * self.__gain(team, last_chance, seat=player)

    def __gain(self, team: frozenset[int], last_chance: bool = False, seat: int = None) -> float:
        """bits putting this team up teaches, to that seat or to the table

        last_chance changes it and is not decoration: on the last team a mission
        allows, a spy would rather sink the round than approve a clean team, so
        the votes the team draws are a different distribution and teach something
        else. The agent passes it, so this has to as well or the two price one
        team two ways. It is in the key for the same reason.

        Cached per decision because the shadow does not move inside one, and the
        cache is cleared with everything else the shadow feeds. This is the most
        expensive call in the file and the propose channel asks for one per team.
        """
        key = (seat, team, last_chance)
        gain = self.__IG_cache.get(key)
        if gain is None:
            gain = self.shadow.information_gain(team, self.__fails_required_now(), \
                last_chance, seat=seat)
            self.__IG_cache[key] = gain
        return gain

    def __votes_gain(self, team: frozenset[int], seat: int = None) -> float:
        """bits the votes this team draws teach, to that seat or to the table

        Only ever asked for an ordinary round: on the last team a mission allows
        a rejection ends the game, and a game that is over has no use for what
        the votes taught. Cached beside __gain and cleared with it.
        """
        key = ("votes", seat, team)
        gain = self.__IG_cache.get(key)
        if gain is None:
            gain = self.shadow.vote_information_gain(team, seat=seat)
            self.__IG_cache[key] = gain
        return gain

    def __hidden(self, values: dict, gains: dict) -> dict:
        """the same options, each pulled down by what doing it would teach the table

            hidden(a) = value(a) - budget x (IG(a) - min IG) / (max IG - min IG)
                a       an option: approve and reject, or every team a leader may
                        name
                IG(a)   bits the TABLE expects to learn if this player does a
                budget  spy_hide_budget, in missions

        A spy wants to win AND to stay hidden. Traded linearly the second eats
        the first without limit, because the cheapest way to teach the table
        nothing is to reject and let no mission run at all, which is how the
        rate this replaces rejected a spy into the open.

        Normalised and capped instead. The option that gives most away loses
        exactly the budget, the one that gives least away loses nothing, and
        nothing loses more than the budget however many bits are on the table.
        So two options further apart in value than the budget keep their order,
        and the budget is exactly what wanting to hide is allowed to cost.

        A tie is where it decides alone, which is the point: with nothing to
        choose on value, a spy takes the action that says less.
        """
        least = min(gains.values())
        spread = max(gains.values()) - least
        if spread <= 0.0:
            return values # every option tells the table the same, so none hides better

        hidden = {}
        for option, value in values.items():
            hidden[option] = value - self.spy_hide_budget * (gains[option] - least) / spread
        return hidden

    def bits_are_worth(self) -> float:
        """what one bit is worth to a modelled resistance player, in missions

            info_weight / swing                 adaptive_info_weight off
            info_weight x (1 - swing) / swing   on
                swing   V(one more success) - V(one more failure)

        info_weight is in win probability per bit, because that is the unit the
        agent's own two sides of a vote are in. Everything here is in missions,
        so it is divided by swing, which is exactly what the agent's term becomes
        when its comparison is divided by swing too. That division is the whole
        of why this needs no knob of its own.
        """
        if self.info_weight <= 0.0:
            return 0.0
        step = self.__swing()
        if self.adaptive_info_weight:
            return self.info_weight * (1.0 - step) / step
        return self.info_weight / step

    def __reject_information(self, player: int, world: frozenset[int], \
        team: frozenset[int]) -> float:
        """what sending this team back would teach this player, in missions

            bits_are_worth() x (IG(the votes this team draws)
                                + IG(the team the NEXT leader puts up))

        A rejection is watched too: the next leader has to name a team, and which
        one it names says something about its own card. Zero for a spy, which is
        the spy half of __information left off, and zero when no leader has been
        recorded, which is a belief nobody is voting with.

        The votes come first and are the same term approving carries, because
        they are cast at once and this player reads them either way. Carried on
        both sides so that it cancels: only what follows the votes should decide
        a vote. See vote_information_gain.
        """
        weight = 0.0 if player in world else self.bits_are_worth()
        if weight == 0.0:
            return 0.0
        return weight * (self.__votes_gain(team, seat=player) + self.__rejection_gain(seat=player))

    def __rejection_gain(self, seat: int = None) -> float:
        """bits the next leader's proposal teaches, to that seat or to the table

        Zero when no leader has been recorded, which is a belief nobody is
        voting with, and zero when the seat is the one that would propose:
        its own team is the same number in every world it allows.
        """
        if self.leader is None:
            return 0.0
        following = (self.leader + 1) % self.num_players
        key = (seat, following, self.rejections)
        gain = self.__rejection_gains.get(key)
        if gain is None:
            gain = self.shadow.rejection_information_gain(following, seat=seat)
            self.__rejection_gains[key] = gain
        return gain

    def __believed_fails(self, world: frozenset[int], player: int, team: frozenset[int]) -> float:
        """P(this team's mission fails), as this player sees it

            a spy P(fails | the world being scored)
            a resistance player sum over w of P(w) P(fails | w) / sum over w of P(w)
                w   every world the SHADOW still allows with this player not a spy

        A spy knows the deal, so it reads the answer off one world. A resistance
        player knows only its own card, so it averages over the worlds that agree
        with it.
        """
        required = self.__fails_required_now()
        if player in world:
            if self.viewer == player:
                # Its own decision, not a model of somebody else's: a player
                # decides on the belief it holds. Reading the shadow here would
                # drop this agent a rung at the moment it acts, and it would be
                # wrong about its OWN card, which no ladder argument covers.
                return self.mission_likelihood(world, team, required, failed=True)
            return self.shadow.mission_likelihood(world, team, required, failed=True)

        key = (player, team)
        cached = self.__public_fails.get(key)
        if cached is not None:
            return cached

        chance = 0.0
        total = 0.0
        for other, weight in zip(self.shadow.worlds, self.shadow.weights):
            if player in other:
                continue # this player knows it is not a spy
            total += weight # P(this player is not a spy)
            chance += weight * self.shadow.mission_likelihood(other, team, required, failed=True) # P(this team's mission fails, this player is not a spy)
        fails = chance / total if total > 0.0 else 0.0 # P(this team's mission fails | this player is not a spy)
        self.__public_fails[key] = fails
        return fails

    def exposure(self, player: int, team: frozenset[int], action: str, \
        last_chance: bool = False) -> float:
        """P(this player is considered as a spy | it did this)

            suspicion({player}, P(this action | w))
                w       a world the SHADOW still allows
                action  "approve", "reject" or "propose", as in action_value

        One Bayes step and nothing else, run from the observer's side: the
        receiver's posterior in a signalling game. Subtracted for both factions:
        a resistance player that looks like a spy is kept off teams too.

        Only the action itself. Approving a team also puts the voter on a mission
        that may then fail, and that further exposure is not counted.

        The world being scored is not in the key, because the table does not know
        which world it is in.
        """
        key = (player, team, action, last_chance)
        cached = self.__exposures.get(key)
        if cached is not None:
            return cached

        def likelihood(world):
            if action == "propose":
                return self.shadow.proposal_likelihood(world, player, team, last_chance)
            return self.shadow.vote_likelihood(world, player, team, action == "approve", last_chance)

        exposure = self.__suspicion(frozenset({player}), likelihood)
        self.__exposures[key] = exposure
        return exposure

    def __team_exposure(self, team: frozenset[int]) -> float:
        """what a failed mission costs the players who were on it, averaged

            after(m) = suspicion(team, P(the mission came out m | w))
                m   whether the mission failed
            team_exposure = after(failed) - after(succeeded)

        The mission card is anonymous: the table learns that somebody on the team
        played Fail and never which one, so the cost is a team sized quantity and
        every member is charged the mean of it. On one measured team the three
        members are worth 0.649, 0.300 and 0.300 after a failure, and each is
        charged their mean of 0.416.

        The difference the failure makes, not the suspicion after it, because
        going on the mission at all was not the choice being priced.
        """
        key = (team, "mission")
        cached = self.__exposures.get(key)
        if cached is not None:
            return cached

        required = self.__fails_required_now()
        after = {}
        for failed in (True, False):
            after[failed] = self.__suspicion(team, lambda world, failed=failed: self.shadow.mission_likelihood(world, team, required, failed))

        exposure = after[True] - after[False]
        self.__exposures[key] = exposure
        return exposure

    def __suspicion(self, members: frozenset[int], likelihood) -> float:
        """the mean over these players of P(that player holds a Spy card | the observation)

            P(w | o) = P(w) L(w) / sum over w of P(w) L(w)      the posterior
            answer   = mean over p in members of sum over w with p in w of P(w | o)
                w        a world the SHADOW still allows
                L(w)     P(the observation | w), at the rung below
                o        the observation
                p        one of the players being read
                members  the players being read

        The one Bayes step both exposure terms are: normalise once, then read
        each player's probability off the posterior and average them. For one
        member that is just that player's probability.
        """
        posterior = []
        total = 0.0
        for world, weight in zip(self.shadow.worlds, self.shadow.weights):
            joint = weight * likelihood(world)
            posterior.append(joint)
            total += joint
        if total <= 0.0:
            return 0.0 # no world explains it, so there is nothing to read

        spied = 0.0
        for player in members:
            for world, joint in zip(self.shadow.worlds, posterior):
                if player in world:
                    spied += joint / total # P(this player holds a Spy card | o)
        return spied / len(members)

    # the quantal response, and the units it is read in

    def __response(self, values: dict) -> dict:
        """how likely each option is, given what each is worth

            softmax   P(a) = exp(beta value(a)) / sum over b of exp(beta value(b))
            greedy    P(a) = e / k + (1 - e if a is the best option else 0)
                a, b  options
                e     model_epsilon
                k     how many options there are

        A softmax, so a preference never rules an option out: a deterministic
        model would eliminate a world the moment a real player did something it
        thought pointless. smooth() on top is a separate thing and is off by
        default: the softmax is the opponent's own noise, the smoothing is our
        doubt about the whole model.

        greedy replaces it with epsilon greedy: the best option with probability
        1 - e, and uniform otherwise. Not smooth() on top of an argmax, because
        smooth() spreads over the support and an argmax has a support of one. It
        grades options by which is best and not by how much, so two options a
        hair apart get (1 - e) + e/k and e/k.
        """
        top = max(values.values()) # subtracted for stability, cancels in the ratio
        if self.greedy:
            floor = self.model_epsilon / len(values)
            best = max(values, key=values.get)
            chances = {}
            for option in values:
                chances[option] = floor + (1.0 - self.model_epsilon if option == best else 0.0)
            return chances

        weights = {}
        total = 0.0
        for option, value in values.items():
            weights[option] = math.exp(self.beta * (value - top)) # softmax
            total += weights[option]
        chances = {}
        for option, weight in weights.items():
            chances[option] = weight / total
        return smooth(chances, self.model_epsilon)

    def gain_signature(self) -> tuple:
        """the rung below's, plus what this rung's vote likelihood also reads

        vote_likelihood softmaxes approving against rejecting, and rejecting is
        worth what the NEXT leader's proposal would teach, so the leader is part
        of the answer here where it is not below.

        The shadow's signature is in it because everything this rung prices is
        read through the shadow, and the two do not move together: this belief
        skips its own viewer's vote and the shadow does not, so a vote by the
        viewer leaves these weights alone and moves the shadow's. Without it the
        key would repeat while the answer had changed.
        """
        return super().gain_signature() + (self.leader,) + self.shadow.gain_signature()

    def expected_mission_successes(self) -> tuple[float, ...]:
        """P(a mission succeeds) for an average team, one per mission

        Team by team, unlike the rung below, because fail_rates here reads the
        team: two teams of the same size holding the same number of spies are
        priced differently when the table is more suspicious of one of them.
        """
        if self.__owner_success is not None:
            return self.__owner_success

        preset = PLAYER_PRESET[self.num_players]
        by_size = {}
        chances = []
        for index, size in enumerate(preset.team_sizes):
            key = (size, preset.fails_required[index])
            if key not in by_size:
                teams = list(combinations(range(self.num_players), size))
                fails = 0.0
                for team in teams:
                    fails += self.mission_fails(frozenset(team), preset.fails_required[index])
                by_size[key] = 1.0 - fails / len(teams)
            chances.append(by_size[key])

        self.__owner_success = tuple(chances)
        return self.__owner_success

    def __chance(self) -> tuple[float, ...]:
        """P(a mission succeeds) for an average team, one per mission, as the TABLE sees it

        Off the shadow, because everything here prices somebody else's action and
        that somebody does not hold this belief's private card.
        expected_mission_successes is the owner's own answer and stays inherited.

        Rounded, as the agent rounds its own: lookahead caches a value table per
        distinct tuple, and a chance read off a belief is a different float on
        every observation, so unrounded the cache grows without bound and never
        hits. Measured over 20 games it reached 185 tables.
        """
        rounded = []
        for chance in self.shadow.expected_mission_successes():
            rounded.append(round(chance, 3))
        return tuple(rounded)

    def __win_now(self) -> float:
        """P(the resistance eventually wins) at the score as it stands, from lookahead.py"""
        return win_values(self.__chance())[(self.successes, self.failures)]

    def __swing(self) -> float:
        """one mission's worth of win probability, the unit every value here is in

            swing = V(one more success) - V(one more failure)
                V   P(the resistance eventually wins) at that score

        Floored: a value table built on a mission that always goes the same way
        has no swing, and this divides by it.
        """
        return max(swing(self.successes, self.failures, self.__chance()), 1e-9)

    def __fails_required_now(self) -> int:
        """Fail cards needed to sabotage the mission being played"""
        preset = PLAYER_PRESET[self.num_players]
        return preset.fails_required[min(self.successes + self.failures, len(preset.fails_required) - 1)]
