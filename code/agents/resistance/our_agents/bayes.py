# [Resistance] our method's belief when the other players are of different types
#
# Written to match BayesKripkeState in agents/our_agents/Kripke.py with its joint
# adapt: one belief over the deal and the type of every other player, updated by
# Bayes on every public observation. The code is this game's. There every
# observation is one opponent's answer, so a type posterior per opponent and
# world is exact. Here a mission result is the cards of every spy on the team
# together, which ties their types, so each world keeps one table over every
# combination of types.
from __future__ import annotations
import inspect
from itertools import combinations, product
from typing import TYPE_CHECKING

from agents.resistance.our_agents.kripke import KripkeState, RandomKripkeState, StrategicKripkeState
from agents.resistance.our_agents.neuro_symbolic import LikelihoodStats, NeuroSymbolicKripkeState
from core.helper_functions import SeedSource
from core.llm_client import LLMClient
from game.resistance_data_classes import MAX_REJECTIONS, PLAYER_PRESET

if TYPE_CHECKING: # type only; opponent_memory imports this module
    from agents.resistance.our_agents.opponent_memory import TypeMemory

# The types a player can be, each with the model that is exact for it:
#   random      RandomResistanceAgent
#   strategic3  an eu player reading an order 2 StrategicKripkeState
#   llm         an LLM player, read by the LLM judge
TYPE_MODELS = ("random", "strategic3", "llm")

# A type whose P(type | previous games) is below this is skipped for one game:
# it is left out of the table and its model is not asked. Speed only: what is
# skipped is at most this much probability at the start of the game. TypeMemory
# keeps the type and scores it after every game, so it can come back.
PRUNE_WEIGHT = 1e-6


def build_type_models(types, num_players: int, viewer: int, *, strategic3: dict, llm_client: LLMClient, \
    seed_source: SeedSource, judge_cot: bool, judge_reasoning_effort: str, print_out: bool) -> dict:
    """one belief per type, read only for what it says a player does

    random and llm keep no posterior on votes and proposals, which nothing reads;
    for llm that saves the judge calls it would make. strategic3 keeps both,
    because its read flags also reach its shadow, which is part of the model.
    """
    models = {}
    if "random" in types:
        models["random"] = RandomKripkeState(num_players, viewer=viewer, read_votes=False, read_proposals=False)
    if "strategic3" in types:
        models["strategic3"] = StrategicKripkeState(num_players, viewer=viewer, order=3, **strategic3)
    if "llm" in types:
        models["llm"] = NeuroSymbolicKripkeState(num_players, viewer=viewer, llm_client=llm_client, \
            seed_source=seed_source, judge_cot=judge_cot, judge_reasoning_effort=judge_reasoning_effort, \
                print_out=print_out, read_votes=False, read_proposals=False)
    return models


def strategic_knobs(options: dict) -> dict:
    """every setting a StrategicKripkeState is built with: from options where given, the class default otherwise

    Compares nothing. check_strategic3 in run_resistance.py calls it on strategic3
    and on every order 2 eu player, and compares the two results.
    """
    knobs = {}
    for rung in (StrategicKripkeState, KripkeState):
        for name, parameter in inspect.signature(rung.__init__).parameters.items():
            if parameter.default is inspect.Parameter.empty or name in ("viewer", "viewer_is_spy"):
                continue
            knobs.setdefault(name, options.get(name, parameter.default))
    return knobs


def fail_count_distribution(rates: list[float], known_fails: int) -> list[float]:
    """P(exactly m Fail cards), m from 0, with one Bernoulli per player still to play

        rates P(this player plays Fail), one per player
        known_fails  Fail cards already known to be played
    """
    distribution = [0.0] * known_fails + [1.0]
    for rate in rates:
        grown = [0.0] * (len(distribution) + 1)
        for played, chance in enumerate(distribution):
            grown[played] += chance * (1.0 - rate)
            grown[played + 1] += chance * rate
        distribution = grown
    return distribution


class BayesKripkeState(KripkeState):
    """one belief over the deal and the types of every other player

        P(w, c | h)   the table: one row per world w, one column per type combination c
            w   a world: which players hold a Spy card
            c   a type combination: one TYPE_MODELS entry for every other player
            h   what the table has made public this game

    At the start of a game, and of every replay of it:

        P(w, c) = P(w) P(c | past)
            P(w)          uniform over the worlds the viewer allows
            P(c | past)   the product over players of P(that player's type in c | past),
                          each from TypeMemory, which scores the games before this one.
                          A type below PRUNE_WEIGHT is left out for this game and the
                          rest of that player's types are renormalised

    On every observation o:

        P(w, c | h, o) = P(w, c | h) P(o | w, c) / P(o | h)
            P(o | w, c)   read off the model of the type c gives whoever acted: one
                          player for a vote or a proposal, every spy on the team for
                          a mission result
            P(o | h)      the sum of the products over every w and c

    P(w | h) is a row's sum, and P(c | w, h) a row divided by its sum.

    The value table asks about one action at a time, with that player's type
    summed out in that world:

        P(a | w, h) = sum over t of P(t | w, h) P_t(a | w)
            a   one action of player j
            t   a type player j may be

    A predicted mission sums out the types of the spies on the team together,
    and the information gain does the same for a spy's vote and card; see
    information_gain. So every prediction reads the table as the update writes it.

    Held by a resistance player only: check_rules refuses eu, and so this
    belief, for any player that may be dealt a Spy card.
    """

    OPPONENT_MODEL = "bayes"
    NEEDS_LLM = True

    def __init__(self, num_players: int, viewer: int = None, viewer_is_spy: bool = False, \
        llm_client: LLMClient = None, *, memory: TypeMemory, seed_source: SeedSource, strategic3: dict = None, \
            judge_cot: bool = False, judge_reasoning_effort: str = None, print_out: bool = False, \
                info_weight: float = 0.0, adaptive_info_weight: bool = False):
        """
        memory: every other player's type posterior, from the games before this one
        seed_source: the stream the llm model's judge seeds come from
        strategic3: settings of the strategic3 model, those of the players it models
        judge_cot, judge_reasoning_effort: passed to the llm model
        info_weight, adaptive_info_weight: this player's own policy, see KripkeState
        """
        self.others: list[int] = [] # every player id but the viewer
        for player in range(num_players):
            if player != viewer:
                self.others.append(player)
        # player id -> type name -> P(type | previous games), without the types below PRUNE_WEIGHT
        self.type_prior: dict[int, dict[str, float]] = {}
        for player in self.others:
            kept = {}
            for type_name, probability in memory.posterior(player).items():
                if probability >= PRUNE_WEIGHT:
                    kept[type_name] = probability
            total = sum(kept.values())
            self.type_prior[player] = {}
            for type_name, probability in kept.items():
                self.type_prior[player][type_name] = probability / total
        type_names = [] # per other player, the types it may be
        for player in self.others:
            type_names.append(list(self.type_prior[player]))
        self.type_combs: list[dict[int, str]] = [] # every combination, as player id -> type name
        for names in product(*type_names):
            self.type_combs.append(dict(zip(self.others, names)))
        self.type_comb_priors: list[float] = []
        # P(type_combination | previous games): what the table starts from. The table is what gets updated.
        for comb in self.type_combs:
            prior = 1.0
            for player, type_name in comb.items():
                prior *= self.type_prior[player][type_name] # opponent model: types independent
            self.type_comb_priors.append(prior)

        used = set() # the types some player may be
        for player in self.others:
            used.update(self.type_prior[player])
        self.seed_source = seed_source
        self.model_settings = {"strategic3": strategic3 or {}, "llm_client": llm_client, \
            "judge_cot": judge_cot, "judge_reasoning_effort": judge_reasoning_effort, "print_out": print_out}
        self.models = build_type_models(used, num_players, viewer, seed_source=self.seed_source.spawn(0), \
            **self.model_settings) # type name -> its model
        super().__init__(num_players, viewer=viewer, viewer_is_spy=viewer_is_spy, info_weight=info_weight, \
            adaptive_info_weight=adaptive_info_weight)
        self.world_index = {} # world -> its row in the table
        for index, world in enumerate(self.worlds):
            self.world_index[world] = index
        self._gains = {} # (gain_signature(), team, ...) -> bits; the key holds the table, so never dropped
        self.scoring_models = {} # type_models(), once the game is scored

    def reset(self) -> None:
        super().reset()
        for model in self.models.values():
            model.reset()
        # table[i][c] = P(worlds, type_combs | h). It starts at P(world) P(combination | previous
        # games), since the deal does not depend on anybody's type.
        self.table: list[list[float]] = []
        for weight in self.weights:
            row = []
            for prior in self.type_comb_priors:
                row.append(weight * prior) # table[i][c] = P(w, type_comb) = P(w) × P(type_comb | past)
            self.table.append(row)
        self._forget()

    def _forget(self) -> None:
        """drop the caches, which are read off the table and the models"""
        self._type_marginals: dict[tuple, dict] = {} # (world, players) -> P(their types | world, h)
        self._table_signature: tuple = None # the table as a tuple, for gain_signature
        self._average_success: tuple = None # expected_mission_successes()

    def type_models(self) -> dict:
        """fresh models for every type, built as these were, to score a finished game

        Also the types this game skipped: TypeMemory scores all of them.
        """
        self.scoring_models = build_type_models(TYPE_MODELS, self.num_players, self.viewer, \
            seed_source=self.seed_source.spawn(1), **self.model_settings)
        return self.scoring_models

    @property
    def stats(self) -> LikelihoodStats:
        """the judge lookups of the llm models, in the game and in scoring it; None when neither was built

        The logger reads it into summary.csv, after the game is scored.
        """
        counted = []
        for models in (self.models, self.scoring_models):
            if "llm" in models:
                counted.append(models["llm"].stats)
        if not counted:
            return None
        total = LikelihoodStats()
        for stats in counted:
            total.queries += stats.queries
            total.judge_calls += stats.judge_calls
            total.cache_hits += stats.cache_hits
        return total

    @property
    def judge_log(self) -> list:
        """every question the llm models sent to the judge, the game's and then scoring's

        The last scoring_judge_calls entries, in the game's memory record, are scoring's.
        """
        log = []
        for models in (self.models, self.scoring_models):
            if "llm" in models:
                log += models["llm"].judge_log
        return log

    # the public record

    def sync_with_history(self, missions, rejected_this_round=(), own_plays=None, leader=None, \
        open_proposal=None) -> None:
        super().sync_with_history(missions, rejected_this_round, own_plays, leader, open_proposal)
        # Who leads now and how many teams this round has lost reach the models the
        # way StrategicKripkeState passes them to its shadows.
        for model in self.models.values():
            below = model
            while below is not None:
                below.rejections = self.rejections
                if leader is not None:
                    below.leader = leader
                below = getattr(below, "shadow", None)

    def update_on_proposal(self, leader: int, team: frozenset[int], rejections: int = 0) -> None:
        self.leader = leader
        self.rejections = rejections
        # Its own proposal is the same number in every world, so there is nothing to learn from it.
        if leader != self.viewer:
            for model in self.models.values():
                # What each model sets before it reads its own proposal likelihood.
                model.leader = leader
                model.rejections = rejections
            last_chance = rejections >= MAX_REJECTIONS - 1

            def proposal(model, world, player): # P(actions | world, types)
                return model.proposal_likelihood(world, player, team, last_chance)

            by_joint = self._likelihood_by_joint(leader, proposal)

            def likelihood(i: int, comb: dict) -> float:
                # P(actions | worlds, types_comb): only the leader's type matters
                return by_joint[i][comb[leader]]

            self._bayes_update(likelihood, f"Player {leader} proposing {sorted(team)}")
        for model in self.models.values():
            model.update_on_proposal(leader, team, rejections)
        self._forget()

    def update_on_votes(self, team: frozenset[int], votes: dict[int, bool], last_chance: bool = False) -> None:
        voters: list[int] = []
        for voter in sorted(votes):
            if voter != self.viewer: # its own vote is not a random variable to it
                voters.append(voter)
        if voters:
            def vote(model, world, player):
                return model.vote_likelihood(world, player, team, votes[player], last_chance)

            by_voter = {} # voter -> by_joint, as _likelihood_by_joint returns it
            for voter in voters:
                by_voter[voter] = self._likelihood_by_joint(voter, vote)

            def likelihood(i: int, comb: dict) -> float:
                # P(all votes | worlds[i], type_comb): given both, each voter votes on its own
                # comb: player id -> type name
                all_votes = 1.0
                for voter in voters:
                    all_votes *= by_voter[voter][i][comb[voter]]
                return all_votes

            self._bayes_update(likelihood, f"the votes on {sorted(team)}")
        for model in self.models.values():
            model.update_on_votes(team, votes, last_chance)
        self._forget()

    def update_on_mission(self, team: frozenset[int], fails_required: int, failed: bool, num_fails: int = None, \
        own_play: bool = None) -> None:
        by_spy_types: list[dict[tuple[str, ...], float]] = [] # per world, as _mission_likelihood_by_joints returns it
        spies: list[list[int]] = [] # per world, the players on the team holding a Spy card in it
        for world in self.worlds:
            by_spy_types.append(self._mission_likelihood_by_joints(world, team, fails_required, failed, \
                num_fails, own_play))
            spies.append(sorted(team & world))

        def likelihood(i: int, comb: dict) -> float:
            # P(this mission result | worlds[i], type_comb): only the types of the spies on the team matter
            # comb: player id -> type name
            # i: the world
            spy_types = []
            for spy in spies[i]: # spies in this world
                spy_types.append(comb[spy]) # the type of that spy in this combination
            return by_spy_types[i][tuple(spy_types)]

        self._bayes_update(likelihood, f"mission {sorted(team)}, failed={failed}, num_fails={num_fails}")
        for model in self.models.values():
            model.update_on_mission(team, fails_required, failed, num_fails, own_play)
        # After the update, not before: the mission was played at the old score.
        self.failures += failed
        self.successes += not failed
        self._forget()

    def _likelihood_by_joint(self, player: int, likelihood) -> list[dict[str, float]]:
        """by_joint[i][type name] = P(what this player did | worlds[i], the player is of that type)

        likelihood(model, world, player): what that type's model says, for one world
        """
        by_joint = [] # per world, type name of this player -> P(action | world, type)
        for world in self.worlds:
            by_name = {}
            for type_name in self.type_prior[player]:
                by_name[type_name] = likelihood(self.models[type_name], world, player) # P(action | world, type)
            by_joint.append(by_name)
        return by_joint

    def _bayes_update(self, likelihood, observed: str) -> None:
        """Bayes on one observation/action, over the worlds and the type combinations together

            table[i][c] <- table[i][c] x likelihood(i, type_combs[c]) / total
                likelihood(i, comb)   P(action | worlds, type_comb)
                total P(action | h): the products summed over every i and c
                table[i][c] P(worlds, type_combs | h)

        Afterwards weights[i], P(worlds[i] | h), is the sum P(worlds, type_combs | h) over type_combs.
        """
        total = 0.0
        for i, row in enumerate(self.table):
            for c, comb in enumerate(self.type_combs):
                row[c] *= likelihood(i, comb) # P(world, combination, observation | h)
            total += sum(row)
        if total <= 0.0:
            raise Exception(f"every world and every type was ruled out by {observed}, so the models are wrong")
        self.weights = []
        for row in self.table:
            for c in range(len(row)):
                row[c] /= total # P(world, combination | h and the observation)
            self.weights.append(sum(row)) # P(world | h)

    def _mission_likelihood_by_joints(self, world: frozenset[int], team: frozenset[int], fails_required: int, \
        failed: bool, num_fails: int, own_play: bool) -> dict[tuple, float]:
        """P(this mission result | world, the types combination of the spies on the team)

        Keyed by those types, the spies in player id order. Each spy plays Fail at
        the rate its own type's model gives a spy on this team. How it expects its
        team mates to play is part of that model, as it is for a real player, which
        does not know their types either. The resistance players play at the rule's
        rate.
        """
        spies = sorted(team & world)
        resistance_players = len(team) - len(spies)
        known_fails = 0
        if own_play is not None and self.viewer in team:
            # The viewer knows the card it played, so its card is not a random variable.
            resistance_players -= 1
            known_fails = 0 if own_play else 1
        spy_fail_rates = {} # type name -> P(this spy plays Fail | type, h)
        for spy in spies:
            for type_name in self.type_prior[spy]:
                if type_name not in spy_fail_rates:
                    # fail_rates gives (spy rate, resistance rate); [0] is the spy's
                    spy_fail_rates[type_name] = self.models[type_name].fail_rates(team, len(spies))[0]

        type_names = [] # per spy, the types it may be
        for spy in spies:
            type_names.append(list(self.type_prior[spy]))
        by_spy_types = {}
        for spy_types in product(*type_names):
            rates = []
            for type_name in spy_types:
                rates.append(spy_fail_rates[type_name])
            for _ in range(resistance_players):
                rates.append(self.resistance_fail_rate)
            distribution = fail_count_distribution(rates, known_fails)
            if num_fails is not None:
                by_spy_types[spy_types] = distribution[num_fails] if num_fails < len(distribution) else 0.0
            else:
                sabotaged = sum(distribution[fails_required:])
                by_spy_types[spy_types] = sabotaged if failed else 1.0 - sabotaged
        return by_spy_types

    def _type_marginal(self, world: frozenset[int], players: list[int]) -> dict[tuple, float]:
        """P(the types of these players | world, h), keyed by their types in the order given

        The table's row for this world, summed over the types of everybody else.
        """
        key = (world, tuple(players))
        marginal = self._type_marginals.get(key)
        if marginal is not None:
            return marginal
        row = self.table[self.world_index[world]] # P(world, type_comb | h)
        world_weight = sum(row) # P(world | h)
        marginal = {}
        for c, comb in enumerate(self.type_combs):
            types = []
            for player in players:
                types.append(comb[player])
            types = tuple(types)
            if world_weight > 0.0:
                share = row[c] / world_weight # P(type_comb | world, h)
            else:
                # A ruled out world. The information gain still reads it, weighted by zero, so any distribution does.
                share = self.type_comb_priors[c]
            marginal[types] = marginal.get(types, 0.0) + share
        self._type_marginals[key] = marginal
        return marginal

    # what one action is worth to predict

    def vote_likelihood(self, world: frozenset[int], voter: int, team: frozenset[int], approved: bool, \
        last_chance: bool = False) -> float:
        """P(this vote | world, h), the voter's type summed out"""
        likelihood = 0.0
        for (type_name,), weight in self._type_marginal(world, [voter]).items():
            likelihood += weight * self.models[type_name].vote_likelihood(world, voter, team, approved, last_chance)
            # P(vote | world, h) = sum over types of P(type | world, h) * P(vote | world, type, h)
        return likelihood

    def proposal_likelihood(self, world: frozenset[int], leader: int, team: frozenset[int], \
        last_chance: bool = False) -> float:
        """P(this proposal | world, h), the leader's type summed out"""
        likelihood = 0.0
        for (type_name,), weight in self._type_marginal(world, [leader]).items():
            likelihood += weight * self.models[type_name].proposal_likelihood(world, leader, team, last_chance)
        return likelihood

    def mission_likelihood(self, world: frozenset[int], team: frozenset[int], fails_required: int, \
        failed: bool, num_fails: int = None, own_play: bool = None) -> float:
        """P(this mission result | world, h), the types of the spies on the team summed out"""
        by_spy_types = self._mission_likelihood_by_joints(world, team, fails_required, failed, num_fails, own_play)
        likelihood = 0.0
        for spy_types, weight in self._type_marginal(world, sorted(team & world)).items():
            likelihood += weight * by_spy_types[spy_types]
        return likelihood

    def expected_mission_successes(self) -> tuple[float, ...]:
        """P(a mission succeeds) for an average team, one per mission

            P(a mission succeeds) = sum over teams T of P(T succeeds | h) / the number of teams
                T   every team of the size that mission asks for

        Team by team, as StrategicKripkeState does: which players are on a team
        decides which types are, so two teams holding as many spies differ.
        """
        if self._average_success is not None:
            return self._average_success

        preset = PLAYER_PRESET[self.num_players]
        by_size = {} # (team size, Fail cards needed) -> P(an average team succeeds)
        by_mission = []
        for index, size in enumerate(preset.team_sizes):
            key = (size, preset.fails_required[index])
            if key not in by_size:
                teams = list(combinations(range(self.num_players), size))
                total_success = 0.0 # summed over the teams, so up to len(teams)
                for team in teams:
                    total_success += 1.0 - self.mission_fails(frozenset(team), preset.fails_required[index])
                by_size[key] = total_success / len(teams)
            by_mission.append(by_size[key])

        self._average_success = tuple(by_mission)
        return self._average_success

    # what a team put to the vote teaches

    def information_gain(self, team: frozenset[int], fails_required: int, last_chance: bool = False) -> float:
        """bits this belief expects to lose by putting this team up, as in KripkeState

            IG(T) = H(W) - sum over o of P(o) H(W | o)
                W   the deal
                T   the team put up, the argument
                o   how every other player votes on T and, if the votes pass
                    it, whether the mission fails

            P(w, o) = P(w | h) x sum over s of P(s | w, h) P(the spies' votes and M | w, s)
                               x product over j of P(j's vote | w, h)
                w   a world
                h   what the table has made public this game
                s   the types of the other players holding a Spy card in w, together
                M   whether the mission fails, which only the spies on T decide
                j   every other player holding no Spy card in w

        A spy's vote and its card come from one type, so they are read under
        the same s. KripkeState multiplies P(votes | w) by P(M | w), which would
        let the spy change type between the two. The other players can be read
        one by one: in one world only a mission ties types together, and on a
        mission only the spies' types matter.
        """
        return self._team_gain(team, last_chance, fails_required)

    def vote_information_gain(self, team: frozenset[int], last_chance: bool = False) -> float:
        """bits the votes on this team teach, as in KripkeState, the spies' types read together"""
        return self._team_gain(team, last_chance, None)

    def _team_gain(self, team: frozenset[int], last_chance: bool, fails_required: int = None) -> float:
        """information_gain, or vote_information_gain when fails_required is None"""
        key = (self.gain_signature(), team, last_chance, fails_required)
        cached = self._gains.get(key)
        if cached is not None:
            return cached
        patterns = list(product((True, False), repeat=len(self.others))) # True: approves, one per self.others
        outcomes = {} # outcome -> P(w, outcome), one entry per world in self.worlds order
        for world, weight in zip(self.worlds, self.weights):
            for outcome, likelihood in self._joint_likelihood(world, team, last_chance, fails_required, patterns).items():
                outcomes.setdefault(outcome, []).append(weight * likelihood) # P(w, outcome | h)
        expected = 0.0 # sum over o of P(o) H(W | o)
        for joint in outcomes.values():
            total = sum(joint) # P(o)
            if total > 0.0:
                posterior = []
                for value in joint:
                    posterior.append(value / total)
                expected += total * self.entropy(posterior)
        gain = self.entropy() - expected
        self._gains[key] = gain
        return gain

    def _joint_likelihood(self, world: frozenset[int], team: frozenset[int], last_chance: bool, fails_required: int, \
        patterns: list[tuple[bool, ...]]) -> dict[tuple, float]:
        """joint likelihood P(o | world, h) for every outcome o, keyed (vote pattern, failed)

        failed is None when no mission is watched: the votes reject the team, or
        fails_required is None.
        """
        spies: list[int] = [] # the other players holding a Spy card in this world, in player id order
        for player in self.others:
            if player in world:
                spies.append(player)
        spy_types = self._type_marginal(world, spies) # their types, together -> P(those types | world, h)
        approves: dict[tuple[int, str | None], float] = {} # (player id, type name) -> P(approves | world, type); None: the type summed out
        for player in self.others:
            if player not in world:
                approves[(player, None)] = self.vote_likelihood(world, player, team, True, last_chance)
        for spy in spies:
            for type_name in self.type_prior[spy]:
                approves[(spy, type_name)] = self.models[type_name].vote_likelihood(world, spy, team, True, \
                    last_chance)
        fails: dict[tuple[str, ...], float] = {} # the spies' types -> P(the mission fails | world, those types)
        if fails_required is not None:
            by_team_types = self._mission_likelihood_by_joints(world, team, fails_required, True, None, None)
            for types in spy_types:
                team_types = [] # the types of the spies on the team, in player id order as by_team_types' key
                for spy, type_name in zip(spies, types):
                    if spy in team:
                        team_types.append(type_name)
                fails[types] = by_team_types[tuple(team_types)] # P(the mission fails | world, spies' types in the team, h)

        outcomes = {}
        for pattern in patterns:
            votes: dict[int, bool] = dict(zip(self.others, pattern)) # player id -> True/False
            resistance_votes = 1.0 # P(the votes of the players holding no Spy card | world, h)
            for player in self.others:
                if player not in world:
                    approve = approves[(player, None)] # P(resistance player approves | world, h)
                    resistance_votes *= approve if votes[player] else 1.0 - approve
            voted = 0.0 # P(this pattern | world, h)
            failed = 0.0 # P(this pattern, the mission fails | world, h)
            passed = 0.0 # P(this pattern, the mission succeeds | world, h)
            for types, share in spy_types.items():
                likelihood = share * resistance_votes
                for spy, type_name in zip(spies, types):
                    approve = approves[(spy, type_name)] # P(spy approves | world, type, h)
                    likelihood *= approve if votes[spy] else 1.0 - approve
                voted += likelihood
                if fails_required is not None:
                    failed += likelihood * fails[types]
                    passed += likelihood * (1.0 - fails[types])
            # The viewer put this team up, so it approves it.
            if fails_required is None or (sum(pattern) + 1) * 2 <= self.num_players: # the mission is not played
                outcomes[(pattern, None)] = voted
            else:
                outcomes[(pattern, True)] = failed
                outcomes[(pattern, False)] = passed
        return outcomes

    def gain_signature(self) -> tuple:
        """the rung below's, plus the models' and the table

        The table can move while the world weights do not, and the models move on
        this player's own vote, which the table skips.
        """
        if self._table_signature is None:
            rows = []
            for row in self.table:
                rows.append(tuple(row))
            self._table_signature = tuple(rows)
        models = []
        for model in self.models.values():
            models.append(model.gain_signature())
        return super().gain_signature() + tuple(models) + (self._table_signature,)
