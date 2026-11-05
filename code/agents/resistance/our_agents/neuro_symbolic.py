# [Resistance] our method with an LLM judge as the opponent model
#
# Written to match NeuroSymbolicKripkeState in agents/our_agents/Kripke.py: the
# same exact belief over deals, with P(what a player did | world) asked of an LLM
# judge instead of read off constants. The code is this game's. Tofu Kingdom's
# answerers all know the deal; here only the spies do, so a resistance player is
# asked about in two steps. See docs/resistance_llm_likelihood_plan.md.
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations

from agents.resistance.our_agents import prompts
from agents.resistance.our_agents.kripke import KripkeState, smooth
from agents.resistance.our_agents.schemas import JudgeAnalysis, judgment_model, suspicion_model
from core.helper_functions import SeedSource
from core.llm_client import LLMClient
from game.resistance_data_classes import PLAYER_PRESET, MissionRecord, ProposalRecord

# The codebook, the same numbers as Tofu Kingdom's judge. ANSWER_WEIGHT is
# normalised, so only its ratios matter. CONFIDENCE_SHRINK is the share of the
# judge's answer; the rest goes to the uniform distribution.
ANSWER_WEIGHT = {"LIKELY": 4, "NEUTRAL": 2, "UNLIKELY": 1}
CONFIDENCE_SHRINK = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}


@dataclass
class LikelihoodStats:
    """how many judge answers were looked up, and how many questions were asked

    A question is one LLM call, or two with judge_cot.
    """
    queries: int = 0
    judge_calls: int = 0
    cache_hits: int = 0


class NeuroSymbolicKripkeState(KripkeState):
    """the same belief over deals, with how other players act judged by an LLM

    Every observation still goes through Bayes over the worlds. What changes is
    P(what this player did | world), on all three channels:
        the rules a resistance player's mission card is resistance_fail_rate, and a team with no spy 
            on it plays no Fail card
        a spy knows the deal, so the judge is told the team as the spy sees it in the world being scored
        a resistance does not know the deal. The judge first rates how every
            player looks from the public record, once per point in the game. Then it is asked how 
            a resistance player acts on a team, told how the other players on that team look. The
            player's own rating is left out: it knows it is not a spy.
    The resistance branch never reads the world being scored, so it is the same
    number in every world where that player is resistance.

    The spy constants of the rung below (spy_fail_rate and the rest) are not read: the judge replaces them.
    """

    OPPONENT_MODEL = "llm"
    NEEDS_LLM = True

    # (system prompt, user prompt, judge_cot, judge_reasoning_effort) to the
    # parsed answer. Shared by every belief, so players asking the same question
    # ask it once; the runner clears it before every game.
    _judge_cache: dict[tuple, object] = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._judge_cache = {}

    def __init__(self, num_players: int, viewer: int = None, viewer_is_spy: bool = False, \
        llm_client: LLMClient = None, *, seed_source: SeedSource = None, judge_cot: bool = False, \
            judge_reasoning_effort: str = None, answer_weight: dict[str, float] = None, \
                confidence_shrink: dict[str, float] = None, print_out: bool = False, **options):
        """
        seed_source: the stream the judge's call seeds come from
        judge_cot: manual chain of thought, a reasoning pass before every rating
            pass. Needs judge_reasoning_effort, as decision_cot needs its own.
        judge_reasoning_effort: the model's own thinking on judge calls, "none"
            to switch it off. None keeps the llm block's setting.
        answer_weight, confidence_shrink: the codebook, ANSWER_WEIGHT and
            CONFIDENCE_SHRINK when None
        print_out: print every prompt sent to the judge and every answer, very long
        options: passed to KripkeState
        """
        if judge_cot and judge_reasoning_effort is None:
            raise ValueError("judge_cot needs judge_reasoning_effort set as well: \"none\" switches "
                "the model's own thinking off. YAML null is not \"none\".")
        self.llm_client = llm_client
        self.seed_source = seed_source or SeedSource(None)
        self.judge_cot = judge_cot
        self.judge_reasoning_effort = judge_reasoning_effort
        self.answer_weight = answer_weight or ANSWER_WEIGHT
        self.confidence_shrink = confidence_shrink or CONFIDENCE_SHRINK
        self.print_out = print_out
        self.stats = LikelihoodStats()
        self.judge_log: list[dict] = [] # one entry per question this belief sent to the LLM
        super().__init__(num_players, viewer=viewer, viewer_is_spy=viewer_is_spy, **options)

    def reset(self) -> None:
        super().reset()
        # The public record, rebuilt from the observations as they are replayed,
        # so that each one is judged on what came before it.
        self._missions: list[MissionRecord] = []
        self._rejected: list[ProposalRecord] = [] # rejected teams on current mission
        self._approved_votes: dict[int, bool] = {} # the votes on the team of current mission
        self._suspicion_level: dict[int, str] = None # the suspicion ratings of the record as it stands
        # Team distributions, keyed by everything they depend on.
        self._cached_proposals_likelihood: dict[tuple, dict] = {}

    # the public record

    def update_on_votes(self, team: frozenset[int], votes: dict[int, bool], last_chance: bool = False) -> None:
        super().update_on_votes(team, votes, last_chance)
        if not votes:
            return # still being voted on
        if sum(votes.values()) * 2 > self.num_players: # approved votes
            self._approved_votes = dict(votes)
            return
        self._rejected.append(ProposalRecord(leader=self.leader, team=team, votes=dict(votes)))
        self._suspicion_level = None

    def update_on_mission(self, team: frozenset[int], fails_required: int, failed: bool, \
        num_fails: int = None, own_play: bool = None) -> None:
        super().update_on_mission(team, fails_required, failed, num_fails, own_play)
        self._missions.append(MissionRecord(index=len(self._missions), leader=self.leader, team=team, \
            team_votes=self._approved_votes, rejected=self._rejected, success=not failed, num_fails=num_fails))
        self._rejected = []
        self._approved_votes = {}
        self._suspicion_level = None

    def gain_signature(self) -> tuple:
        """the rung below's, plus the leader and how much of the record there is

        vote_likelihood reads the leader, and the resistance branch reads the
        record, which can grow without moving the weights: this belief skips its
        own viewer's vote, and the record does not.
        """
        return super().gain_signature() + (self.leader, len(self._missions), len(self._rejected))

    # the three channels

    def vote_likelihood(self, world: frozenset[int], voter: int, team: frozenset[int], approved: bool, \
        last_chance: bool = False) -> float:
        """P(this player voted this way | this world)"""
        situation = self._situation(len(team), self._fails_required_now(), last_chance)
        if voter in world:
            partners = len(team & world) - (voter in team)
            question = prompts.spy_vote(situation, voter in team, partners, voter == self.leader)
        else:
            question = prompts.resistance_vote(situation, voter in team, voter == self.leader, \
                self._team_suspicion_level(voter, team))
        answer = self._judge(question, judgment_model(prompts.VOTE_OPTIONS)) # llm judge's answer
        chances = self._chances({"APPROVE": "APPROVE", "REJECT": "REJECT"}, answer)
        return chances["APPROVE" if approved else "REJECT"]

    def proposal_likelihood(self, world: frozenset[int], leader: int, team: frozenset[int], \
        last_chance: bool = False) -> float:
        """P(this leader proposed this team | this world)

        The judge rates kinds of team, and every team of one kind gets that
        kind's rating.
        """
        spy = leader in world
        key = (world if spy else None, leader, len(team), last_chance, self.successes, self.failures, \
            len(self._missions), len(self._rejected))
        chances = self._cached_proposals_likelihood.get(key)
        if chances is None:
            kinds = {}
            for candidate in combinations(range(self.num_players), len(team)):
                option = frozenset(candidate)
                if spy:
                    kinds[option] = prompts.spy_team_option(leader in option, len(option & world) - (leader in option))
                else:
                    kinds[option] = prompts.resistance_team_option(leader in option, self._team_suspicion_level(leader, option))
            options = tuple(sorted(set(kinds.values())))
            situation = self._situation(len(team), self._fails_required_now(), last_chance)
            if spy:
                question = prompts.spy_proposal(situation, options)
            else:
                question = prompts.resistance_proposal(situation, options)
            chances = self._chances(kinds, self._judge(question, judgment_model(options)))
            self._cached_proposals_likelihood[key] = chances
        return chances[team]

    def fail_rates(self, team: frozenset[int], spies: int = None) -> tuple[float, float]:
        """P(plays the Fail card) for a spy and for a resistance player on this team

        spies: how many spies are on the team, which the judge is told
        """
        if spies is None:
            raise ValueError("the llm rung asks the judge about a team with a known number of spies")
        return self._average_team_rates(len(team), self._fails_required_now(), spies)

    def _average_team_rates(self, size: int, fails_required: int, spies: int) -> tuple[float, float]:
        """the same two rates for a team of this size holding this many spies

        The judge is told only counts, so a team is as good as any other of its
        size with the same number of spies.
        """
        if spies == 0:
            return self.spy_fail_rate, self.resistance_fail_rate # no spy in teams, so these will be ignored
        question = prompts.spy_mission(self._situation(size, fails_required), spies - 1)
        answer = self._judge(question, judgment_model(prompts.CARD_OPTIONS))
        spy = self._chances({"SUCCESS": "SUCCESS", "FAIL": "FAIL"}, answer)["FAIL"]
        return spy, self.resistance_fail_rate # a resistance player's card is the rule's

    # the two steps for a resistance player

    def _team_suspicion_level(self, player: int, team: frozenset[int]) -> tuple[int, int, int]:
        """how many of the team, leaving this player out, look SUSPECT, UNSURE and TRUSTED"""
        looks = self._suspicion()
        others = []
        for member in team:
            if member != player:
                others.append(looks[member])
        return others.count("SUSPECT"), others.count("UNSURE"), others.count("TRUSTED")

    def _suspicion(self) -> dict[int, str]:
        """how every player looks from the public record as it stands, asked once per record"""
        if self._suspicion_level is not None:
            return self._suspicion_level
        if not self._missions and not self._rejected:
            # Nothing has happened, so there is nothing to tell the players apart by.
            self._suspicion_level = {player: "UNSURE" for player in range(self.num_players)}
            return self._suspicion_level
        question = prompts.suspicion(self.num_players, self._missions, self._rejected, \
            failed_mission_has_spy=self.resistance_fail_rate == 0.0)
        answer = self._judge(question, suspicion_model(self.num_players))
        self._suspicion_level = {player: getattr(answer, f"player_{player}") for player in range(self.num_players)}
        return self._suspicion_level

    # the judge

    def _chances(self, kinds: dict, answer) -> dict:
        """P(each option), from the rating the judge gave the option's kind

            P(a) = s w(a) / (sum over b of w(b)) + (1 - s) / n
                a, b  options
                w(a)  answer_weight of the rating the kind of a got
                s     confidence_shrink of the judge's confidence
                n     how many options there are

        Then smoothed by model_epsilon, as every other rung's likelihoods are.
        """
        shrink = self.confidence_shrink[answer.confidence]
        weights = {}
        for option, kind in kinds.items():
            weights[option] = self.answer_weight[getattr(answer, kind)]
        total = sum(weights.values())
        chances = {}
        for option, weight in weights.items():
            chances[option] = shrink * weight / total + (1.0 - shrink) / len(weights)
        return smooth(chances, self.model_epsilon)

    def _judge(self, question: tuple[str, str, str], response_model):
        """the judge's answer to this question, asked of the LLM once per game

        question: (system, user_prompts, ask), as the builders in prompts.py return it
        """
        self.stats.queries += 1
        key = question + (self.judge_cot, self.judge_reasoning_effort)
        cached = NeuroSymbolicKripkeState._judge_cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        system, user_prompts, ask = question
        cot_analysis = None
        prompt = prompts.rating_prompt(user_prompts, ask)
        if self.judge_cot:
            analysis_prompt = prompts.judge_cot_analysis_prompt(user_prompts)
            analysis, _ = self.llm_client.chat_structured_model(analysis_prompt, JudgeAnalysis, \
                system_prompt=system, seed=self.seed_source.next_seed(), \
                    reasoning_effort=self.judge_reasoning_effort)
            cot_analysis = analysis.analysis
            self._print_call("reasoning pass", system, analysis_prompt, cot_analysis)
            prompt = prompts.judge_cot_final_prompt(user_prompts, cot_analysis, ask)
        answer, thinking = self.llm_client.chat_structured_model(prompt, response_model, system_prompt=system, \
            seed=self.seed_source.next_seed(), reasoning_effort=self.judge_reasoning_effort)
        self._print_call("rating pass" if self.judge_cot else "rating", system, prompt, answer.model_dump())

        self.stats.judge_calls += 1
        self.judge_log.append({"question": prompts.rating_prompt(user_prompts, ask), "cot_analysis": cot_analysis, \
            "thinking": thinking, "answer": answer.model_dump()})
        NeuroSymbolicKripkeState._judge_cache[key] = answer
        return answer

    def _print_call(self, what: str, system: str, user: str, answer) -> None:
        """one judge call as the LLM saw it, when print_out is on"""
        if not self.print_out:
            return
        print(f"\n{'~' * 60}\n[likelihood judge] Player {self.viewer} belief, {what}\n{'~' * 60}")
        print("System Prompt:\n" + system)
        print(f"\n{'-' * 60}\nUser Prompt:\n" + user)
        print(f"\n{'-' * 60}\nAnswer:\n{answer}\n{'~' * 60}")

    def _situation(self, team_size: int, fails_required: int, last_chance: bool = False) -> prompts.Situation:
        return prompts.Situation(self.num_players, self.successes, self.failures, team_size, fails_required, \
            last_chance)

    def _fails_required_now(self) -> int:
        """Fail cards needed to sabotage the mission being played"""
        preset = PLAYER_PRESET[self.num_players]
        return preset.fails_required[min(self.successes + self.failures, len(preset.fails_required) - 1)]
