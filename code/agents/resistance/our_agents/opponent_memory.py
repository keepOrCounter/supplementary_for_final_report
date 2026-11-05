# [Resistance] what the Bayes player learns about every other player's type
#
# Written to match the type half of CrossGameMemory and update_type_posteriors in
# agents/our_agents/opponent_memory.py: a log likelihood per player id and type,
# summed over every finished game and never reset. The code is this game's. The
# other half of that store, the records an LLM is shown, is not ported.
from __future__ import annotations
import json
import math
import time

from agents.resistance.our_agents.bayes import TYPE_MODELS
from game.resistance_data_classes import MAX_REJECTIONS, PLAYER_PRESET, ProposalRecord, ResistanceState

# The kinds of action a game is scored on, kept apart in each game's record.
CHANNELS = ("proposal", "vote", "card")


class TypeMemory:
    """P(type | the games so far) for every player id, by exact Bayes over TYPE_MODELS

        P(t | D) = P(t) L_t(D) / sum over u of P(u) L_u(D)
            t, u    a type, one TYPE_MODELS entry
            D       everything the player did in the games scored so far
            L_t(D)  the likelihood of D if the player is of type t: the product
                    over its actions of P_t(action)
            P(t)    the prior, uniform, so it cancels

    log_likelihoods holds log L_t(D). No type is ever removed: every game scores
    all of them, so a type BayesKripkeState skips can come back.
    """

    def __init__(self, players: list[int]):
        self.log_likelihoods: dict[int, dict[str, float]] = {} # player id -> type name -> log L_t(D)
        for player in players:
            self.log_likelihoods[player] = dict.fromkeys(TYPE_MODELS, 0.0)
        self.games_recorded = 0 # games scored so far
        self.history: list[dict] = [] # one record per scored game, see add_game
        self.agents: dict[int, dict] = {} # player id -> its agent and the type built for it, set by the runner

    def posterior(self, player: int) -> dict[str, float]:
        """P(type | the games so far) for one player"""
        scores = self.log_likelihoods[player]
        top = max(scores.values()) # subtracted so exp cannot underflow to all zeros
        exps = {}
        for type_name, score in scores.items():
            exps[type_name] = math.exp(score - top)
        total = sum(exps.values())
        posterior = {}
        for type_name, value in exps.items():
            posterior[type_name] = value / total
        return posterior

    def add_game(self, scores: dict[int, dict[str, dict[str, float]]], actions: dict[int, dict[str, int]], \
        game_index: int, seed: int, scoring_judge_calls: int, scoring_seconds: float) -> None:
        """add one game's log likelihoods, and keep a record of the game

        scores: player id -> type name -> channel -> log likelihood of that channel's actions in this game
        actions: player id -> channel -> how many actions were scored
        scoring_judge_calls: judge questions that scoring this game asked
        scoring_seconds: wall time of scoring this game, which the game's duration_seconds leaves out
        """
        for player, by_type in scores.items():
            for type_name, by_channel in by_type.items():
                # opponent model: a player keeps its type
                self.log_likelihoods[player][type_name] += sum(by_channel.values())
        self.games_recorded += 1
        log_likelihoods = {} # after this game, so the evidence still shows once a posterior reads 1
        posteriors = {}
        for player in sorted(self.log_likelihoods):
            log_likelihoods[str(player)] = dict(self.log_likelihoods[player])
            posteriors[str(player)] = self.posterior(player)
        game_scores = {}
        game_actions = {}
        for player in sorted(scores):
            game_scores[str(player)] = scores[player]
            game_actions[str(player)] = actions[player]
        self.history.append({"games_recorded": self.games_recorded, "game_index": game_index, "seed": seed, \
            "log_likelihoods": log_likelihoods, "posteriors": posteriors, "scores": game_scores, \
            "actions": game_actions, "scoring_judge_calls": scoring_judge_calls, "scoring_seconds": scoring_seconds})

    def save(self, path: str) -> None:
        """write the store as JSON, the runner's memory.json"""
        log_likelihoods = {}
        posteriors = {}
        for player in sorted(self.log_likelihoods):
            log_likelihoods[str(player)] = self.log_likelihoods[player]
            posteriors[str(player)] = self.posterior(player)
        agents = {}
        for player in sorted(self.agents):
            agents[str(player)] = self.agents[player]
        data = {"games_recorded": self.games_recorded, "agents": agents, "type_log_likelihoods": log_likelihoods, \
            "type_posteriors": posteriors, "type_history": self.history}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


def update_type_posteriors(memory: TypeMemory, state: ResistanceState, cards: list[dict[int, bool]], \
    models: dict, game_index: int, seed: int) -> None:
    """score one finished game against every type each player may still be

        score[j][t] = sum over the actions a of player j of log P_t(a | w*)
            j   a player id in memory
            t   a type player j may still be
            w*  the deal, shown once the game is over

    Kept apart by channel (CHANNELS) in the game's record; the memory adds the sum.

    The actions are the public record and every spy's mission card. A
    resistance player's card is the rule's, the same under every type, so it is
    not scored. The models advance along the record as BayesKripkeState advances
    them, so each action is scored against what was public just before it.

    Args:
        cards: per mission index, player id to the card played, True for Success
        models: type name -> its model, BayesKripkeState.type_models() of the Bayes player of this game,
            one model for every type, also the types that game skipped
        game_index, seed: which game this is, for its record
    """
    started = time.time()
    world = frozenset(state.spies_ids)
    scores = {} # player id -> type name -> channel -> log likelihood of this game
    actions = {} # player id -> channel -> actions scored
    for player, by_type in memory.log_likelihoods.items():
        scores[player] = {}
        for type_name in by_type:
            scores[player][type_name] = dict.fromkeys(CHANNELS, 0.0)
        actions[player] = dict.fromkeys(CHANNELS, 0)

    def add_log_likelihoods(player: int, channel: str, by_type: dict[str, float]) -> None:
        """by_type: type name -> P(the player's action | w*, that type)"""
        for type_name, likelihood in by_type.items():
            if likelihood <= 0.0:
                raise ValueError(f"the {type_name} model gives Player {player}'s action a probability of "
                    f"{likelihood}, but every model allows every legal action, so the record and the models disagree")
            scores[player][type_name][channel] += math.log(likelihood)
        actions[player][channel] += 1

    def team_setup_replay(proposal: ProposalRecord, rejections: int) -> None:
        last_chance = rejections >= MAX_REJECTIONS - 1
        for model in models.values():
            # What each model sets before it reads its own proposal likelihood.
            model.leader = proposal.leader
            model.rejections = rejections
        if proposal.leader in scores: # not the Bayes player itself
            by_type = {}
            for type_name in scores[proposal.leader]:
                by_type[type_name] = models[type_name].proposal_likelihood(world, proposal.leader, \
                    proposal.team, last_chance)
            add_log_likelihoods(proposal.leader, "proposal", by_type)
        for model in models.values():
            model.update_on_proposal(proposal.leader, proposal.team, rejections)
        for voter, approved in sorted(proposal.votes.items()):
            if voter in scores:
                by_type = {}
                for type_name in scores[voter]:
                    by_type[type_name] = models[type_name].vote_likelihood(world, voter, proposal.team, \
                        approved, last_chance)
                add_log_likelihoods(voter, "vote", by_type)
        for model in models.values():
            model.update_on_votes(proposal.team, proposal.votes, last_chance)

    # The order sync_with_history replays a game in.
    for mission in state.missions:
        for rejections, proposal in enumerate(mission.rejected):
            team_setup_replay(proposal, rejections)
        team_setup_replay(ProposalRecord(leader=mission.leader, team=mission.team, votes=mission.team_votes), \
            len(mission.rejected))
        spies = len(mission.team & world)
        for player, success in sorted(cards[mission.index].items()):
            if player in world and player in scores:
                by_type = {}
                for type_name in scores[player]:
                    fails = models[type_name].fail_rates(mission.team, spies)[0] # [0]: the spy's rate
                    by_type[type_name] = 1.0 - fails if success else fails
                add_log_likelihoods(player, "card", by_type)
        fails_required = PLAYER_PRESET[state.num_players].fails_required[mission.index]
        for model in models.values():
            model.update_on_mission(mission.team, fails_required, not mission.success, mission.num_fails)
    # A game the spies win on rejections ends with these, which no mission record holds.
    for rejections, proposal in enumerate(state.rejected_this_round):
        team_setup_replay(proposal, rejections)

    judge = models.get("llm") # missing only when a caller scores fewer types
    scoring_judge_calls = 0 if judge is None else judge.stats.judge_calls
    memory.add_game(scores, actions, game_index, seed, scoring_judge_calls, time.time() - started)
