# ablation agent for The Resistance: every decision is a separate knob, so a
# grid can tell which decision an LLM actually helps on.
#
# The Resistance asks a player for three different things, and this agent picks
# a strategy per thing the way agents/our_agents/ablation_agent.py, the Tofu
# ablation Prince, picks one per question and per guess. All random is the random baseline; all llm is the pure LLM
# baseline; the mixed cells isolate one decision at a time.
from __future__ import annotations
import inspect

from agents.resistance import prompts
from agents.resistance.base import ResistanceAgent, open_proposal, refuse_impossible_fail_counts
from agents.resistance.our_agents.bayes import BayesKripkeState
from agents.resistance.our_agents.kripke import (
    KripkeState, RandomKripkeState, RESISTANCE_FAIL_RATE, StrategicKripkeState,
)
from agents.resistance.our_agents.lookahead import swing, team_value, win_values
from agents.resistance.our_agents.neuro_symbolic import NeuroSymbolicKripkeState
from core.helper_functions import SeedSource
from core.llm_client import LLMClient
from core.resistance_schemas import (
    DecisionAnalysis, final_guess_model, mission_card_model, team_proposal_model, team_vote_model,
)
from game.resistance_data_classes import (
    MAX_REJECTIONS, PLAYER_PRESET, Phase, ResistanceObservation, Team,
)

STRATEGIES = ("random", "llm", "eu")
DEFAULT_STRATEGY = "llm" # what a decision runs on when the config does not say
STRATEGY_KEYS = ("propose_strategy", "team_vote_strategy", "mission_strategy")

# Which rung of the opponent model an eu strategy reads the game with. See
# docs/opponent_models.md; uniform is the constants, strategic is the spy that
# weighs a vote against what it gives away, llm asks an LLM judge, bayes learns
# which of random, strategic and llm each other player is, across games.
BELIEF_MODELS = {"random": RandomKripkeState, "uniform": KripkeState, "strategic": StrategicKripkeState,
    "llm": NeuroSymbolicKripkeState, "bayes": BayesKripkeState}
DEFAULT_BELIEF_MODEL = "uniform"


def unknown_belief_options(agent_config: dict) -> list[str]:
    """config keys that are neither this agent's own nor its belief rung's

    The agent keeps what its signature names and hands the rest to the belief,
    so a key that rung does not take arrives there as a keyword it has never
    heard of. That raises where the belief is built, which is the first
    observation of the first game, and then of every game after it, instead of
    before the run starts.
    """
    taken = set(inspect.signature(AblationResistanceAgent.__init__).parameters)
    belief = BELIEF_MODELS[agent_config.get("belief_model", DEFAULT_BELIEF_MODEL)]
    accepted = set()
    for rung in belief.__mro__:
        if rung is object:
            continue
        parameters = inspect.signature(rung.__init__).parameters
        accepted.update(parameters)
        # A rung without **options hands nothing to its parents, so their keys cannot be set through it.
        passes_on = False
        for parameter in parameters.values():
            passes_on = passes_on or parameter.kind == inspect.Parameter.VAR_KEYWORD
        if not passes_on:
            break

    unknown = []
    for key in agent_config:
        if key in ("type", "faction"): # read by the runner, not by the agent
            continue
        if key not in taken and key not in accepted:
            unknown.append(key)
    return sorted(unknown)


class AblationResistanceAgent(ResistanceAgent):
    """one strategy per decision, plus the context flags the LLM strategies read

    propose_strategy: "random" | "llm" | "eu" which players to send on the mission
    team_vote_strategy: "random" | "llm" | "eu" approve or reject the proposed team
    mission_strategy: "random" | "llm" | "eu" play the Success or the Fail card

    eu is our method. It keeps a belief over the deals and scores an action by
    what it is worth two steps out, plus what it would teach:

        V(action) = E[the game's value after it] + info_weight x IG(action)

    info_weight is the one knob the whole comparison turns on. At 0 the agent
    consumes its belief greedily, which is what GRAIL's policy does, so the
    baseline is this method's own special case. Above 0 it will pay some win
    probability for a mission whose result it cannot predict, which is what
    actively seeking information means here.

    Both factions. A spy is handed its teammates in the system prompt and is the
    only player with a real choice of mission card, so it needs its own prompts
    rather than a resistance policy with the labels swapped.

    An LLM strategy returns the decision itself, not an index into valid_actions:
    see core/resistance_schemas.py for why.
    """

    @classmethod
    def check_rules(cls, env, team: Team, agent_config: dict = None) -> None:
        """an eu spy needs the rung that has a spy in it

        The value table is the resistance's chance of winning the game, and a spy
        maximising it would play to lose; what a spy maximises is one minus it.
        StrategicKripkeState prices both sides off that one table, so an eu spy
        is that model applied to itself and needs nothing else. The uniform rung
        describes its spies with four constants and has no policy to hand back,
        so eu there is a resistance strategy only.
        """
        if agent_config is None:
            return
        unknown = unknown_belief_options(agent_config)
        if unknown:
            only_these = {}
            for key in unknown:
                only_these[key] = agent_config[key]
            takers = []
            for name in BELIEF_MODELS:
                if not unknown_belief_options(dict(only_these, belief_model=name)):
                    takers.append(name)
            fix = "no belief_model takes all of them, drop the key"
            if takers:
                fix = f"set belief_model: {' or '.join(takers)}, or drop the key"
            raise ValueError(f"{', '.join(unknown)}: belief_model "
                f"{agent_config.get('belief_model', DEFAULT_BELIEF_MODEL)!r} does not take this; {fix}")
        strategies = []
        for key in STRATEGY_KEYS:
            strategies.append(agent_config.get(key, DEFAULT_STRATEGY))
        # The runner scores every finished game with the belief, and only eu builds one.
        if agent_config.get("belief_model") == "bayes" and "eu" not in strategies:
            raise ValueError("belief_model: bayes needs at least one strategy on eu, the only one that reads it")
        if "eu" in strategies:
            refuse_impossible_fail_counts(env, \
                agent_config.get("resistance_fail_rate", RESISTANCE_FAIL_RATE))
        if team != Team.SPIES:
            return
        # An eu spy draws its card from the belief's own fail_rates, so pinning
        # the card makes it play what its own opponent model says it would not,
        # and the resistance reads that spy with a model of a player that no
        # longer exists. For a floor, where nothing infers anything, pin it.
        if agent_config.get("spies_play_fail") \
            and agent_config.get("mission_strategy", DEFAULT_STRATEGY) == "eu":
            raise ValueError("spies_play_fail with mission_strategy: eu pins the card an eu spy "
                "draws from its own belief, so the agent and the model of it come apart. Drop one: "
                "the switch is for a floor, eu is for the method")
        if agent_config.get("belief_model", DEFAULT_BELIEF_MODEL) == "strategic":
            return
        for key in STRATEGY_KEYS:
            if agent_config.get(key, DEFAULT_STRATEGY) == "eu":
                raise ValueError(f"{key}: eu on a spy seat needs belief_model: strategic, which is the "
                    "rung that prices a spy's actions. Set it, pin this seat with "
                    "faction: resistance, put eu under agents.resistance, or use random or llm here")

    @classmethod
    def needs_llm(cls, agent_config: dict) -> bool:
        """true when a decision is on llm, or an eu decision reads a belief that asks one"""
        strategies = []
        for key in STRATEGY_KEYS:
            strategies.append(agent_config.get(key, DEFAULT_STRATEGY))
        belief = BELIEF_MODELS[agent_config.get("belief_model", DEFAULT_BELIEF_MODEL)]
        return "llm" in strategies or (belief.NEEDS_LLM and "eu" in strategies)

    def __init__(self, player_id: int, seed_source: SeedSource, \
        propose_strategy: str = DEFAULT_STRATEGY, team_vote_strategy: str = DEFAULT_STRATEGY, \
            mission_strategy: str = DEFAULT_STRATEGY, llm_client: LLMClient = None, \
                use_reasoning_history: bool = True, decision_cot: bool = False, \
                    decision_reasoning_effort: str = None, resistance_plays_success: bool = True, \
                        resistance_approves_last_team: bool = False, spies_play_fail: bool = False, \
                        belief_model: str = DEFAULT_BELIEF_MODEL, mission_success: float = None, \
                        print_out: bool = False, **belief_options):
        """
        use_reasoning_history: feed the agent's own running summary back into its next prompt.
        decision_cot: manual chain of thought. Split every decision into a reasoning
            pass and a decision pass, 2 LLM calls instead of 1. Meant to be used
            with decision_reasoning_effort: none, where the reasoning is bounded
            by what the prompt asks for and a runaway thinking trace is
            impossible. The model's own thinking has no such bound: it ran to
            94% of max_tokens on one measured team vote, and a trace that
            reaches the cap returns no decision at all and costs the whole game.
            Tofu Kingdom's judge uses the same two pass shape (ablation.judge_cot).
        decision_reasoning_effort: the model's own thinking on this agent's calls,
            "none" to switch it off. None keeps whatever the llm block set, which
            is what every agent gets by default. It is this agent's own setting
            and not the llm block's, so one line-up can hold an agent with the
            thinking off next to an agent with it on. Tofu Kingdom pairs its
            manual CoT with the same kind of setting (ablation.judge_reasoning_effort).
        resistance_plays_success: a resistance player plays the Success card
            without being asked, whatever mission_strategy says. On by default,
            because the simplified game says a resistance player never plays
            Fail, which is the same thing RESISTANCE_FAIL_RATE at 0 says on the
            belief side. The environment still allows the card, so this is a
            policy that matches a rule and not the rule itself: turn it off only
            together with a non-zero resistance_fail_rate, or the belief will
            call a mission that failed with no spy on it impossible.
        spies_play_fail: a spy plays the Fail card without being asked, whatever
            mission_strategy says. The spy side's half of
            resistance_plays_success, and what makes a floor symmetric: a random
            spy plays Success about half the time, which loses the spies the game
            as surely as a random resistance player playing Fail loses it for the
            resistance.
            Unlike resistance_plays_success it takes away a real decision. A
            resistance player gains nothing from a Fail card, but a spy that
            plays Success hides, and over 68 measured spy cards an llm spy chose
            Success 63% of the time. So turn it on for a floor, where the point
            is that no inference happens, and read an llm cell that has it on as
            a different agent rather than a cheaper one.
            NOT for an eu spy: that one draws its card from the belief's own
            fail_rates, and pinning the card makes the agent play something its
            own opponent model says it would not. check_rules refuses the pair.
        belief_model: which rung of the opponent model an eu strategy reads with,
            one of BELIEF_MODELS. See docs/opponent_models.md.
        mission_success: what a mission with no team chosen yet is assumed to
            succeed with, the one input the value table takes. None reads it off
            the belief, which is an average team of the size the next mission
            asks for. Pin it to a number only to sweep it.
        belief_options: passed through to the belief, which documents them.
            Three of them are this agent's own policy and not only the model's:
            info_weight and adaptive_info_weight, what a bit is worth, and
            hide_weight, what the table's suspicion costs. They live on the
            belief so that a rung above prices this player the way this player
            actually plays. Two knobs for one quantity is how an agent and its
            own opponent model come apart.
        resistance_approves_last_team: a resistance player approves the last team
            a mission allows without being asked. Rejecting that team hands the
            game to the spies there and then, so approving is never worse and
            sometimes still wins the mission. GRAIL's agent overrides its own
            policy the same way, so this is what makes the two comparable on
            that ending. The resistance side only: a spy that rejects the last
            team wins the game, which is its best move and not a formality.
        """
        super().__init__(player_id, seed_source)
        self.strategies = {
            Phase.PROPOSE: propose_strategy,
            Phase.TEAM_VOTE: team_vote_strategy,
            Phase.MISSION_VOTE: mission_strategy,
        }
        for phase, strategy in self.strategies.items():
            if strategy not in STRATEGIES:
                raise ValueError(f"{phase.value} strategy must be one of {STRATEGIES}, got {strategy!r}")
        if llm_client is None and "llm" in self.strategies.values():
            raise ValueError("an llm strategy needs an llm_client")
        # Manual CoT is only bounded with the model's own thinking off, and a run
        # that keeps it on pays for two calls a decision and still loses games to
        # a runaway trace. YAML null reads like "none" and means the opposite, so
        # leaving this unset is refused rather than taken as a choice.
        if decision_cot and decision_reasoning_effort is None:
            raise ValueError("decision_cot needs decision_reasoning_effort set as well: "
                "\"none\" switches the model's own thinking off, which is what bounds the "
                "reasoning. YAML null is not \"none\": it leaves llm.reasoning_effort in force.")

        if belief_model not in BELIEF_MODELS:
            raise ValueError(f"belief_model must be one of {sorted(BELIEF_MODELS)}, got {belief_model!r}")
        if llm_client is None and "eu" in self.strategies.values() and BELIEF_MODELS[belief_model].NEEDS_LLM:
            raise ValueError(f"belief_model {belief_model!r} asks an LLM, so it needs an llm_client")
        self.belief_model = belief_model
        self.mission_success = mission_success
        self.belief_options = belief_options
        self.belief: KripkeState = None # built on the first observation, which carries the table size
        self.own_plays: dict[int, bool] = {} # mission index to the card this agent played

        self.llm_client = llm_client
        self.use_reasoning_history = use_reasoning_history
        self.decision_cot = decision_cot
        self.decision_reasoning_effort = decision_reasoning_effort
        self.resistance_plays_success = resistance_plays_success
        self.resistance_approves_last_team = resistance_approves_last_team
        self.spies_play_fail = spies_play_fail
        self.print_out = print_out

        # Two sub-streams, so switching one decision from random to llm does not
        # shift what the decisions still on random draw.
        self.rng = seed_source.spawn(0).to_random()
        self._llm_seeds = seed_source.spawn(1)
        self._judge_seeds = seed_source.spawn(2) # only a belief that asks an LLM draws from it

        self._llm_suspects: list[int] = None # player ids the last LLM call called spies
        self._num_players: int = None # table size, kept for identity_belief which has no observation

        # Same shape as the Tofu LLM agent, so the logger reads both the same way.
        self.reasoning_history: list[str] = [] # the model's own thinking, per call
        self.current_summary: str = "" # the running summary fed back into the next prompt
        self.summary_history: list[str] = []
        self.decision_log: list[dict] = [] # per call: phase, suspects, summary

    def __str__(self) -> str:
        parts = []
        for phase in (Phase.PROPOSE, Phase.TEAM_VOTE, Phase.MISSION_VOTE):
            parts.append(self.strategies[phase])
        name = f"{type(self).__name__}({'/'.join(parts)}"
        if "eu" in parts:
            # Only eu reads the belief. No space: the summary's agents column is split on spaces.
            name += f",belief={self.belief_model}"
        return name + ")"

    def take_action(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        self._num_players = obs.state.num_players
        if "eu" in self.strategies.values():
            self.__sync_belief(obs)
        if len(valid_actions) == 1:
            # Nothing to decide, but a mission card the rules force is still this
            # player's own card and its belief has to hold it. Recorded here as
            # well as below, because enforce_resistance_must_pass leaves a
            # resistance player one action and it would never reach that branch.
            if obs.state.phase == Phase.MISSION_VOTE:
                self.own_plays[obs.state.mission_index] = valid_actions[0]["vote"]
            return valid_actions[0]

        phase = obs.state.phase
        if phase == Phase.PROPOSE:
            # The leader is asked twice, and the second turn is not a decision:
            # nobody speaks between the two, so there is nothing new to revise
            # the team on. This is what GRAIL's agent does too.
            for action in valid_actions:
                if action["type"] == "call_vote":
                    return action
            return self.__select_propose(obs, valid_actions)

        elif phase == Phase.TEAM_VOTE:
            return self.__select_team_vote(obs, valid_actions)

        elif phase == Phase.MISSION_VOTE:
            action = self.__select_mission(obs, valid_actions)
            self.own_plays[obs.state.mission_index] = action["vote"]
            return action

        raise Exception(f"nothing to decide in phase {phase.value}")

    def __select_propose(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """PROPOSE: hand the decision to this run's propose strategy

        One dispatcher per phase, the way the Tofu ablation Prince splits into
        __select_question and __select_guess. A new strategy adds one branch
        here and one method, instead of a row per phase in a shared table.
        """
        strategy = self.strategies[Phase.PROPOSE]
        if strategy == "random":
            return self.__random_action(obs, valid_actions)
        elif strategy == "llm":
            return self.__propose_llm(obs)
        elif strategy == "eu":
            return self.__propose_eu(obs, valid_actions)
        raise ValueError(f"Unknown propose_strategy: '{strategy}'")

    def __select_team_vote(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """TEAM_VOTE: hand the decision to this run's team vote strategy

        A resistance player can be told to approve the last team a mission
        allows, in which case there is no decision left for a strategy to make.
        A spy is asked as usual: rejecting that team wins it the game.
        """
        if self.resistance_approves_last_team and obs.team == Team.RESISTANCE \
            and obs.state.rejections >= MAX_REJECTIONS - 1:
            return {"type": "team_vote", "vote": True}

        strategy = self.strategies[Phase.TEAM_VOTE]
        if strategy == "random":
            return self.__random_action(obs, valid_actions)
        elif strategy == "llm":
            return self.__team_vote_llm(obs)
        elif strategy == "eu":
            return self.__team_vote_eu(obs, valid_actions)
        raise ValueError(f"Unknown team_vote_strategy: '{strategy}'")

    def __select_mission(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """MISSION_VOTE: hand the decision to this run's mission strategy

        Either faction can be told which card to play, in which case there is no
        decision left for a strategy to make. A spy always holds both cards, so
        the Fail card below is legal whatever the rule flags say.
        """
        if self.resistance_plays_success and obs.team == Team.RESISTANCE:
            return {"type": "mission_vote", "vote": True}

        if self.spies_play_fail and obs.team == Team.SPIES:
            return {"type": "mission_vote", "vote": False}

        strategy = self.strategies[Phase.MISSION_VOTE]
        if strategy == "random":
            return self.__random_action(obs, valid_actions)
        elif strategy == "llm":
            return self.__mission_llm(obs)
        elif strategy == "eu":
            return self.__mission_eu(obs, valid_actions)
        raise ValueError(f"Unknown mission_strategy: '{strategy}'")

    def identity_belief(self) -> dict[int, float]:
        """what this agent currently believes about who the spies are

        With one strategy per decision there can be more than one belief in the
        agent at once, so the rule is: report the belief that is defined at every
        decision point, whichever decision is being made. Ranked, most defined
        first:

        1. eu: the belief it keeps over the deals, rebuilt from every public
            observation. It is defined even on turns no eu strategy acts on,
            which is what makes it the most defined answer here.
        2. llm: the players the last LLM answer named. Hard 0 or 1, because the
            model names players rather than scoring them; the logger calls a
            player a spy above 50%, so this scores the same way a graded belief does.
        3. nothing: None, which is the answer when every decision is on random.

        A mixed line-up reports a belief that is only as fresh as its last LLM
        call. That is not a gap in the record: an agent that has not thought
        since mission 2 does still hold what it thought then, and beliefs.csv
        stamps the snapshot with the mission count it was taken at.
        """
        if self.belief is not None:
            return self.belief.spy_probability()
        if self._llm_suspects is None:
            return None
        belief = {}
        for player_id in range(self._num_players):
            belief[player_id] = 1.0 if player_id in self._llm_suspects else 0.0
        return belief

    def final_belief(self, obs: ResistanceObservation) -> dict[int, float]:
        """sync the belief, or ask the model once more, now the last mission resolved

        An eu line-up answers for free: its belief is rebuilt from the record, so
        the closing look is one more rebuild and no call.

        The suspects come back with every decision, but the last decision of a
        game is made before the last mission resolves, so no answer the agent
        ever gave saw the result that ended the game. This call is the one that
        does, and it costs one call per resistance player per game.

        A line-up with no llm strategy has nothing to ask, so it reports what it
        already holds, which for all random is nothing.
        """
        if "eu" in self.strategies.values():
            self.__sync_belief(obs)
            return self.identity_belief()
        if "llm" not in self.strategies.values():
            return super().final_belief(obs)

        self._num_players = obs.state.num_players
        model = final_guess_model(obs.state.num_players, self.__num_spies(obs))
        self.__ask(obs, model, prompts.build_final_guess_prompt(obs, self.__previous_summary()))
        return self.identity_belief()

    def __propose_eu(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """PROPOSE: send the team worth the most, less what naming it gives away

            V(team) - hide_weight x exposure(proposal) x swing(score)

        A resistance leader is modelled as naming a team uniformly, so nothing it
        proposes says anything about the deal on its own. The Bayes step still
        moves, because it weighs that against what a SPY in this seat would name,
        and a spy steers its own onto the team.
        """
        # The last team a mission allows draws a different vote, so it is worth a
        # different number. The agent has always said so to its own gain and the
        # model has to hear it too, or one team is priced two ways.
        last_chance = obs.state.rejections >= MAX_REJECTIONS - 1
        if obs.team == Team.SPIES:
            team = self.belief.sample_team(self.__own_world(obs), self.player_id, obs.team_size, \
                self.rng, last_chance)
            return {"type": "propose_team", "team": sorted(team)}

        best = None
        best_value = None
        for action in valid_actions:
            team = frozenset(action["team"])
            value = self.__team_value(obs, team) - self.__exposure_cost(obs, team, "propose", last_chance)
            if best_value is None or value > best_value:
                best, best_value = action, value
        return {"type": "propose_team", "team": sorted(best["team"])}

    def __team_vote_eu(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """TEAM_VOTE: approve when running this mission beats sending it back

        Rejecting does not change the score, so it is worth what the game is
        worth now, except on the last team a mission allows, where a rejection
        hands the game to the spies and is worth nothing.

        V(score) is by construction
        chance x V(one more success) + (1 - chance) x V(one more failure), so
        approving beats rejecting exactly when this team beats an average one.
        Reading chance off the belief is what makes "average" mean the teams this
        belief expects; pinned high, the same line rejects every team in the game.

        The whole comparison, in win probability:

            approving  = P(succeeds) V(one more success)
                       + (1 - P(succeeds)) V(one more failure)
                       + info_weight x IG(this team)
                       - hide_weight x exposure(approve) x swing
            rejecting  = V(the score now)
                       + info_weight x (IG(this team's votes)
                                        + IG(the next leader's proposal))
                       - hide_weight x exposure(reject) x swing
                       = 0 on the last team a mission allows

        Both sides carry an information term. Approving is watched through the
        votes and the mission, rejecting through the same votes and then the team
        the next leader has to name.

        IG(this team) already holds IG(this team's votes), so writing it on both
        sides is what makes it cancel. The votes are cast at once: this player
        reads them whichever way it votes, so crediting them to approving alone
        pays for information the vote does not buy. Measured before this was
        written on both sides, the votes term alone was 1.7 to 3.4 times the
        whole rejecting side, and a resistance agent approved 56.4% of teams at
        info_weight 0 against 83.6% at 0.5.

        Both sides carry an exposure term too, once hide_weight is on. It is the
        same term the strategic belief prices every other player with, times
        swing because that term is in missions and these two are in win
        probability.

        One approximation: a rejection also burns one of the five a round allows,
        and this prices it as free until the last one, where it costs everything.
        """
        state = obs.state
        successes = state.mission_results.count(True)
        failures = state.mission_results.count(False)
        last_chance = state.rejections >= MAX_REJECTIONS - 1
        if obs.team == Team.SPIES:
            vote = self.belief.sample_vote(self.__own_world(obs), self.player_id, state.proposed_team, \
                self.rng, last_chance)
            return {"type": "team_vote", "vote": vote}

        # A rejection leaves the score where it is, so it is worth what the game
        # is worth now. On the last team a mission allows it hands the game over,
        # and there is no next proposal to learn anything from either.
        rejecting = 0.0
        if not last_chance:
            rejecting = win_values(self.__chance())[(successes, failures)]
            weight = self.__info_weight(obs)
            if weight > 0.0:
                next_leader = (state.leader + 1) % state.num_players
                rejecting += weight * (self.belief.vote_information_gain(state.proposed_team) \
                    + self.belief.rejection_information_gain(next_leader))
            rejecting -= self.__exposure_cost(obs, state.proposed_team, "reject", last_chance)
        approving = self.__team_value(obs, state.proposed_team)
        if not last_chance:
            # On the last team a mission allows, rejecting loses the game, so
            # approving is dominant and looking like a spy cannot change that.
            # The belief drops the term on the same vote, in action_value.
            approving -= self.__exposure_cost(obs, state.proposed_team, "approve", last_chance)
        return {"type": "team_vote", "vote": approving >= rejecting}

    def __exposure_cost(self, obs: ResistanceObservation, team: frozenset[int], \
        action: str, last_chance: bool = False) -> float:
        """what this action would cost by making the table suspect this agent

            hide_weight x P(the table calls me a spy | I did this) x swing(score)
                action  "approve", "reject" or "propose", the same three words
                        the belief prices every other player's actions with

        Times swing to get from missions, which is what hide_weight is in, into
        win probability, which is what the two sides of the vote are in.

        hide_weight is read off the belief and not off this agent, so that the
        price this agent pays for looking like a spy is the price its model of
        every other player pays. A rung with no shadow cannot see what the table
        thinks and carries 0, which turns this term off.

        Multiplying by swing does NOT make the agent bolder when it is ahead.
        Both sides of the vote carry this term and the payoff term is itself
        swing times (q - p), so swing divides out of the comparison: the vote
        approves above q = p + hide_weight x (exposure(approve) - exposure(reject)),
        and that shift is the same number at every score. The one exception is
        the last team a mission allows, where rejecting ends the game and so
        exposes nobody, and the term stands alone against an absolute value.
        """
        hide_weight = self.belief.hide_weight
        if hide_weight <= 0.0:
            return 0.0
        successes = obs.state.mission_results.count(True)
        failures = obs.state.mission_results.count(False)
        exposure = self.belief.exposure(self.player_id, team, action, last_chance)
        return hide_weight * exposure * swing(successes, failures, self.__chance())

    def __mission_eu(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """MISSION_VOTE: score both cards and play the better one

        It comes out Success every time on a resistance seat: the Fail card drops
        the value to whatever a failed mission is worth and teaches nothing,
        because a mission this player sank itself says nothing about anybody
        else. Computed rather than asserted, because a value that has to be
        overridden by a rule is a value nobody can check.
        """
        state = obs.state
        successes = state.mission_results.count(True)
        failures = state.mission_results.count(False)
        if obs.team == Team.SPIES:
            card = self.belief.sample_card(self.__own_world(obs), self.player_id, state.proposed_team, self.rng)
            return {"type": "mission_vote", "vote": card}

        # Both cards priced through the belief with this player's own card held
        # fixed, so that what is compared is what the rest of the team does.
        # The votes are already in the belief by now, so the Success card is
        # priced on what the MISSION still teaches and not on them as well.
        # The Fail card carries no information term, because a mission this
        # player sank itself teaches it nothing. That is exact only when one Fail
        # card is enough; where two are needed the outcome still says whether
        # somebody else played one. Leaving it out understates the Fail card,
        # which is the safe direction and cannot flip the choice, since the
        # payoff term already favours Success.
        playing_success = self.__team_value(obs, state.proposed_team, own_play=True, \
            votes_are_in=True)
        fails = self.belief.mission_fails(state.proposed_team, obs.fails_required, own_play=False)
        playing_fail = team_value(successes, failures, 1.0 - fails, self.__chance())
        return {"type": "mission_vote", "vote": playing_success >= playing_fail}

    def __team_value(self, obs: ResistanceObservation, team: frozenset[int], \
        own_play: bool = None, votes_are_in: bool = False) -> float:
        """what running this team's mission is worth, plus what it would teach

            V(team) = team_value(score, P(success)) + info_weight x IG(team)
                P(success)   1 - belief.mission_fails(team), this team's chance
                team_value   P(success) V(one more success)
                            + (1 - P(success)) V(one more failure)
                IG(team)     bits the belief expects to lose by putting this team
                            up, over the votes it draws and the mission it runs
                votes_are_in the votes have already been cast and folded in, so
                            only the mission is still to come and IG(team) drops
                            to what it teaches BEYOND them

        lookahead.team_value is the SAME line the belief scores a team with when
        it models somebody else deciding. The belief reads it as team_gain, which
        is this divided by swing, because a softmax sharpness needs a
        dimensionless number.

        The model of a seat feeds that line the belief a rung lower agent in
        that seat would hold, which is the same belief to 5.6e-16 until that seat
        plays a mission card nobody else can see. What is left between this agent
        and the model of it is the rung, which is the ladder working, and argmax
        against softmax, which cannot be closed: see __mission_gain in kripke.py.

        The two step part is what a one step P(success) cannot see: the same
        mission is worth six times as much at 2 successes against 2 failures as
        it is at 2 against 0.

        This is QMDP, and QMDP never pays to learn anything, which is what
        info_weight corrects. See docs/standard_names.md.
        """
        state = obs.state
        successes = state.mission_results.count(True)
        failures = state.mission_results.count(False)
        success_chance = self.__belief_success(obs, team, own_play) # P(this team succeeds)
        value = team_value(successes, failures, success_chance, self.__chance())
        weight = self.__info_weight(obs)
        if weight > 0.0:
            last_chance = state.rejections >= MAX_REJECTIONS - 1
            gain = self.belief.information_gain(team, obs.fails_required, last_chance)
            if votes_are_in:
                # By the mission card the votes are public and already in the
                # belief and the team has been approved, so neither the votes nor
                # the chance of being voted down is still an outcome. What is
                # left is the mission. Subtracting the votes out of IG(team)
                # gives a third number and not this one: 0.1560 bits against
                # 0.5515 on one measured state.
                gain = self.belief.mission_information_gain(team, obs.fails_required)
            value += weight * gain
        return value

    def __belief_success(self, obs: ResistanceObservation, team: frozenset[int], \
        own_play: bool = None) -> float:
        """P(this team's mission succeeds), under the belief

        own_play holds this player's own card fixed, which is what the mission
        card decision needs and what proposing or voting on a team does not.
        """
        return 1.0 - self.belief.mission_fails(team, obs.fails_required, own_play)

    def __info_weight(self, obs: ResistanceObservation) -> float:
        """what one bit is worth right now

            info_weight x (1 - swing(score))    when adaptive_info_weight is on
            info_weight                         otherwise
                swing(score)  V(one more success) - V(one more failure)

        The price this agent puts on a bit, so a scalarisation weight over two
        objectives. Scaling by 1 - swing makes it a state dependent schedule.

        Fixed unless adaptive_info_weight is on, and then scaled by 1 - swing:
        information is cheap while the score leaves room to lose a mission, and
        worth nothing at 2 successes against 2 failures, where losing this one
        ends the game.
        """
        if not self.belief.adaptive_info_weight:
            return self.belief.info_weight
        successes = obs.state.mission_results.count(True)
        failures = obs.state.mission_results.count(False)
        return self.belief.info_weight * (1.0 - swing(successes, failures, self.__chance()))

    def __chance(self):
        """P(a mission succeeds) for average team, one per mission, from the belief or pinned

            pinned      the number the config gave, used for every mission
            otherwise   belief.expected_mission_successes(), one per mission

        A number pinned by the config, or one per mission read off the belief.
        Rounded, because the table is cached per value and a chance read off a
        belief is a different float on every turn of every game.
        """
        if self.mission_success is not None:
            return self.mission_success
        rounded = []
        for chance in self.belief.expected_mission_successes():
            rounded.append(round(chance, 3))
        return tuple(rounded)

    def __own_world(self, obs: ResistanceObservation) -> frozenset[int]:
        """the deal, for a player that was shown it

        A spy is told who its partners are, so there is no belief to hold about
        the thing this game is otherwise about. What it needs a belief for is the
        other direction: how the table is reading it, which is the shadow.
        """
        return frozenset({self.player_id} | set(obs.known_spies))

    def __sync_belief(self, obs: ResistanceObservation) -> None:
        """rebuild the belief from everything the table has made public"""
        if self.belief is None:
            belief_class = BELIEF_MODELS[self.belief_model]
            extra = {}
            if belief_class.NEEDS_LLM:
                extra = {"llm_client": self.llm_client, "seed_source": self._judge_seeds, \
                    "print_out": self.print_out}
            self.belief = belief_class(obs.state.num_players, viewer=self.player_id, \
                viewer_is_spy=obs.team == Team.SPIES, **self.belief_options, **extra)
        self.belief.sync_with_history(obs.state.missions, obs.state.rejected_this_round, \
            self.own_plays, obs.state.leader, open_proposal(obs.state))

    def __random_action(self, obs: ResistanceObservation, valid_actions: list[dict]) -> dict:
        """any phase: pick uniformly among the legal actions

        The same policy RandomResistanceAgent plays, so an all random line-up of
        this agent plays the same distribution of games as the random baseline.
        """
        return self.rng.choice(valid_actions)

    def __propose_llm(self, obs: ResistanceObservation) -> dict:
        """PROPOSE: ask the model which players to send

        Only reached on the leader's first turn of a proposal. Any team of the
        required size is legal, so the response model's limits are the whole check.
        """
        model = team_proposal_model(obs.state.num_players, self.__num_spies(obs), obs.team_size)
        decision = self.__ask(obs, model, prompts.build_propose_prompt(obs, self.__previous_summary()))
        return {"type": "propose_team", "team": sorted(decision.team)}

    def __team_vote_llm(self, obs: ResistanceObservation) -> dict:
        """TEAM_VOTE: ask the model to approve or reject the proposed team"""
        model = team_vote_model(obs.state.num_players, self.__num_spies(obs))
        decision = self.__ask(obs, model, prompts.build_team_vote_prompt(obs, self.__previous_summary()))
        return {"type": "team_vote", "vote": decision.approve}

    def __mission_llm(self, obs: ResistanceObservation) -> dict:
        """MISSION_VOTE: ask the model for the Success or the Fail card"""
        model = mission_card_model(obs.state.num_players, self.__num_spies(obs))
        decision = self.__ask(obs, model, prompts.build_mission_prompt(obs, self.__previous_summary()))
        return {"type": "mission_vote", "vote": decision.play_success}

    def __num_spies(self, obs: ResistanceObservation) -> int:
        """how many spies this table was dealt, which bounds spy_suspects"""
        return PLAYER_PRESET[obs.state.num_players].num_spies

    def __previous_summary(self) -> str:
        return self.current_summary if self.use_reasoning_history else ""

    def __ask(self, obs: ResistanceObservation, response_model, prompt: tuple[str, str]):
        """one structured LLM call, plus the bookkeeping every call shares

        Each llm method builds its own response model and its own prompt. What
        is left is the call itself and recording the answer, and those are the
        same for all three.

        A parse failure after all retries is left to propagate: the runner ends
        that game and records why, rather than the agent picking something.
        """
        system_prompt, user_prompt = prompt
        if self.print_out:
            print(f"\n{'~' * 60}\n[{self}] Player {self.player_id} "
                f"{obs.state.phase.value} prompt\n{'~' * 60}")
            print("System Prompt:\n" + system_prompt)
            print(f"\n{'-' * 60}\nUser Prompt:\n" + user_prompt + f"\n{'~' * 60}")

        cot_analysis = None # the reasoning pass, None when decision_cot is off
        if self.decision_cot:
            analysis, analysis_reasoning = self.llm_client.chat_structured_model( \
                prompts.decision_cot_analysis_prompt(user_prompt), DecisionAnalysis, \
                    system_prompt=system_prompt, seed=self._llm_seeds.next_seed(), \
                        reasoning_effort=self.decision_reasoning_effort)
            cot_analysis = analysis.analysis
            if analysis_reasoning:
                self.reasoning_history.append(analysis_reasoning)
            user_prompt = prompts.decision_cot_final_prompt(user_prompt, cot_analysis)

        decision, reasoning = self.llm_client.chat_structured_model(user_prompt, response_model, \
            system_prompt=system_prompt, seed=self._llm_seeds.next_seed(), \
                reasoning_effort=self.decision_reasoning_effort)

        self._llm_suspects = sorted(decision.spy_suspects)
        if reasoning:
            self.reasoning_history.append(reasoning)
        if decision.reasoning_summary:
            self.current_summary = decision.reasoning_summary
            self.summary_history.append(decision.reasoning_summary)
        self.decision_log.append({
            "phase": obs.state.phase.value,
            "spy_suspects": self._llm_suspects,
            "reasoning_summary": decision.reasoning_summary,
            "cot_analysis": cot_analysis, # manual-CoT reasoning pass, null when decision_cot is off
        })

        if self.print_out:
            print(f"{'=' * 60}\n[{self}] Player {self.player_id} decision")
            if cot_analysis:
                print(f"Analysis: {cot_analysis}")
            if reasoning:
                print(f"Thinking: {reasoning}")
            print(f"Suspects: {self._llm_suspects}")
            print(f"Summary: {decision.reasoning_summary}")
        return decision
