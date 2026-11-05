"""
Kripke Structure module for Tofu Kingdom.

Models Prince Mochi's epistemic state as a weighted set of possible worlds
(complete role assignments). Provides:
    - Enumeration of all initially possible worlds
    - Bayesian belief update on observed Q&A
    - Information-gain computation for question selection
    - Faction-specific answer-value evaluation

Terminology:
    - "Possible world" = one complete assignment of roles to players (+center)
    - "Accessibility relation R_Prince" = the set of worlds the Prince considers possible
    - "Weight" = Bayesian posterior probability of each world

This is a Kripke structure (not full DEL with action models), augmented
with Bayesian weights for probabilistic reasoning.
"""

from __future__ import annotations
import math
from itertools import permutations
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
import copy
import warnings

if TYPE_CHECKING:  # type-only; avoids a circular import (opponent_memory imports Kripke)
    from agents.our_agents.opponent_memory import CrossGameMemory

from game.predefined_data_classes import (
    Role, ROLES, ROLE_CONFIGS, ROLE_CONFIGS_NO_CENTER,
    Constraint, QuestionType, QAPair, Faction,
)

from core.llm_client import LLMClient

from agents.our_agents.schemas import LikelihoodJudgment, JudgeAnalysis

from core.helper_functions import player_id_from_answer

from agents.base import BeliefUpdateError



class PossibleWorld:
    """A single possible world: a complete role assignment."""

    __slots__ = ("roles", "center_card", "__princess", "__queen", "hash")

    def __init__(self, roles: dict[int, str], center_card: str | None = None):
        """
        Args:
            roles: {player_id: role_name} for every non-prince player.
            center_card: role_name placed in the center (None if disabled).
        """
        self.roles = roles
        self.center_card = center_card
        self.__princess = self.__find_role("Princess Tofu")
        self.__queen = self.__find_role("Queen Tofu")
        self.hash = hash((tuple(sorted(self.roles.items())), self.center_card))

    def __find_role(self, role_name: str) -> int:
        """Return pid holding *role_name*, or -1 if in center, None if absent."""
        for pid, rn in self.roles.items():
            if rn == role_name:
                return pid
        if self.center_card == role_name:
            return -1
        return None

    @property
    def princess_holder(self) -> int:
        return self.__princess

    @property
    def queen_holder(self) -> int:
        return self.__queen

    def get_truth(self, q_type: QuestionType, target_id: int, q_param: int | None) -> str: # TODO: redundant?
        """Return the truthful answer to a question in this world."""
        if q_type == QuestionType.Q1:          # "Where is Princess Tofu?"
            if self.__princess > 0:
                return f"Player {self.__princess}"
            return "center"

        elif q_type == QuestionType.Q2:        # "Who are you?"
            return self.roles[target_id]

        elif q_type == QuestionType.Q3:        # "Who is Player X?"
            if q_param == -1:
                return self.center_card
            return self.roles[q_param]

        return "unknown"

    def get_constraint(self, player_id: int) -> Constraint:
        return ROLES[self.roles[player_id]].constraint

    def get_faction(self, player_id: int) -> Faction:
        return ROLES[self.roles[player_id]].faction

    def __repr__(self):
        parts = ", ".join(f"P{i}:{role}" for i, role in sorted(self.roles.items()))
        if self.center_card:
            parts += f", center:{self.center_card}"
        return f"W({parts})"

    def __eq__(self, other):
        return (self.roles == other.roles and self.center_card == other.center_card)

    def __hash__(self):
        return self.hash



Rating = Literal["LIKELY", "NEUTRAL", "UNLIKELY"]
Confidence = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass
class JudgeQuery:
    """Canonical, game-invariant description of a soft-likelihood situation.

    Everything the judge is allowed to see lives here; nothing references an
    absolute player id, so equal queries -> equal cache key.
    """
    q_type: str # QuestionType value, e.g. "Q1"
    answerer_faction: str # Faction of the hypothesised answerer, e.g. "QUEEN"
    constraint: str # Constraint of the answerer, e.g. "MUST_LIE"
    truth_label: str # outcome-slot label the truthful answer implicates
    slots: tuple[str, ...] # unique outcome-slot labels among the allowed answers

    def key(self) -> tuple:
        return (self.q_type, self.answerer_faction, self.constraint, self.truth_label, self.slots)


@dataclass
class MemoryJudgeQuery(JudgeQuery):
    """JudgeQuery about ONE CONCRETE opponent, carrying their cross-game memory.

    Unlike the base query this is deliberately NOT game-invariant: equal
    situations for different opponents (or different memory snapshots) must be
    judged separately, hence opponent_id and memory_version join the cache key.
    """
    opponent_id: int = -1
    memory_version: int = 0  # how many games this opponent has played so far
    memory: str = ""  # this opponent's raw past-game behaviour ("" = none)

    def key(self) -> tuple:
        return super().key() + (self.opponent_id, self.memory_version)


@dataclass
class JudgeResult:
    """A rating per outcome-slot label plus an overall confidence."""
    rating_by_label: dict[str, Rating]
    confidence: Confidence


@dataclass
class LikelihoodStats:
    """Lightweight observability: how many soft queries were served, and how many
    of those actually hit the judge (the rest came from cache). Used to quantify
    LLM cost / verify the cache works."""
    queries: int = 0 # soft_distribution calls
    judge_calls: int = 0 # actual judge invocations (cache misses)
    cache_hits: int = 0
    defaulted_slots: int = 0 # slots the LLM left unrated -> defaulted to NEUTRAL (should stay near 0)

    def hit_rate(self) -> float:
        return self.cache_hits / self.queries if self.queries else 0.0



# Only these two roles are *pivotal*: the guess outcome (Princess / Queen / Other
# faction wins) is decided solely by whether the Prince picks Princess Tofu or
# Queen Tofu. Every other role (Chef, Minister, Guard, Maid, Spy) -> Other wins.
PRINCESS_ROLE = "Princess Tofu"
QUEEN_ROLE = "Queen Tofu"


def _role_label(role_name: str | None) -> str:
    """Outcome class of a role for the Prince's guess (rule-faithful slot label)."""
    if role_name == PRINCESS_ROLE:
        return "PRINCESS"
    if role_name == QUEEN_ROLE:
        return "QUEEN"
    return "OTHER"


def implication(answer: str, q_type: QuestionType, q_param: int | None, world: PossibleWorld) -> str:
    """Outcome-class slot that *answer* implicates in *world* (game-invariant).

    NB: keyed on the *role*, not the faction — Tofu Chef is PRINCESS-faction but
    guessing Chef makes the OTHER faction win, so Chef -> OTHER here.
    """
    if q_type == QuestionType.Q1:
        # answer points at a location (where the Princess is claimed to be)
        if answer == "center":
            return _role_label(world.center_card)
        return _role_label(world.roles[player_id_from_answer(answer)])
    # Q2 / Q3: the answer *is* a role name -> outcome class of the claimed role
    return _role_label(answer)



def uniform_answer_likelihood(truthful: bool, constraint: Constraint, num_answers: int) -> float:
    """Uniform-rule answer likelihood P(answer | possible answers, constraint).

    Args:
        truthful:   whether the observed answer is the honest answer in context.
        constraint: the answering role's answer constraint.
        num_answers:  number of conceivable answers for this question type.

    Model:
        MUST_TRUTH:    1 if truthful else 0
        MUST_LIE:      0 if truthful else 1/(num_answers - 1)
        NO_CONSTRAINT: 1/num_answers
    """
    if constraint == Constraint.MUST_TRUTH:
        return 1.0 if truthful else 0.0
    if constraint == Constraint.MUST_LIE:
        if truthful:
            return 0.0
        return 1.0 / (num_answers - 1) if num_answers > 1 else 0.0
    return 1.0 / num_answers  # NO_CONSTRAINT


class KripkeState: # TODO: packaging is confusing
    """
    Weighted Kripke structure representing Prince's epistemic state.

    worlds[i]  – i-th possible world still in the accessibility relation
    weights[i] – posterior probability of that world (sums to 1), P(wi) or P(wi | answeer)
    
    Currently uniform initial weights, and update only depends on hard constraints.
    """

    def __init__(self, num_players: int, use_center_card: bool = True):
        self.num_players = num_players
        self.use_center_card = use_center_card
        self.worlds: list[PossibleWorld] = []
        self.weights: list[float] = []
        self.__num_qa: int = 0 # how many QA pairs already applied
        # Per-instance answer-list cache, keyed on q_type (num_players / use_center_card
        # are fixed for the life of the instance).
        self._answers_cache: dict[QuestionType, list[str]] = {}
        self.__enumerate_worlds()


    def __enumerate_worlds(self): # Initialisation
        """Enumerate every legal role assignment -> initial uniform prior."""
        configs = ROLE_CONFIGS if self.use_center_card else ROLE_CONFIGS_NO_CENTER
        role_names = copy.deepcopy(configs[self.num_players])
        pids = list(range(1, self.num_players))

        if self.use_center_card and self.num_players != 8:
            # Enumerate all possible center-card assignments.
            for single_role_n in range(len(role_names)):
                center = role_names[single_role_n]
                rest = role_names[:single_role_n] + role_names[single_role_n + 1:]
                for perm in permutations(rest):
                    self.worlds.append(PossibleWorld({pid: perm[i] for i, pid in enumerate(pids)}, \
                        center_card=center))
        else:
            for perm in permutations(role_names):
                self.worlds.append(PossibleWorld({pid: perm[i] for i, pid in enumerate(pids)}, \
                    center_card=None))

        n = len(self.worlds)
        self.weights = [1.0 / n] * n # uniform prior probability: P(w) = 1/|W| for all w


    def _all_answers(self, q_type: QuestionType) -> list[str]: # TODO: redundant? also used in env for answer validation
        """All conceivable answers for a question type (cached per q_type)."""
        cached = self._answers_cache.get(q_type)
        if cached is not None:
            return cached

        configs = ROLE_CONFIGS if self.use_center_card else ROLE_CONFIGS_NO_CENTER
        if q_type == QuestionType.Q1:
            ans = [f"Player {p}" for p in range(1, self.num_players)]
            if self.use_center_card and self.num_players != 8:
                ans.append("center")
        else:  # Q2, Q3
            ans = list(configs[self.num_players])  # shallow copy: flat list[str] -> safe

        self._answers_cache[q_type] = ans
        return ans


    def answer_likelihood(self, world: PossibleWorld, answer: str, q_type: QuestionType, target_id: int,\
        q_param: int | None) -> float:
        """
        Compute P(answer | world) under Prince's rational-player model. 

        Current model (uniform):
            MUST_TRUTH: P(ans | w) = 1[ans = truth(w)]
            MUST_LIE: P(ans | w) = 0  if ans = truth(w),  else 1/(lies_answers)
            NO_CONSTRAINT: P(ans | w) = 1/A

        where A = set of all conceivable answers for the question type, truth(w) = the honest answer in 
        world w.
        """
        truth = world.get_truth(q_type, target_id, q_param)
        constraint = world.get_constraint(target_id)
        all_ans = self._all_answers(q_type) # A
        return uniform_answer_likelihood(answer == truth, constraint, len(all_ans))


    def update_on_answer(self, qa: QAPair, likelihood=None) -> None:
        """
        Bayesian posterior update after observing one QA pair.

        Formula (Bayes' rule applied to each world):
            P(w | ans) = P(ans | w) * P(w) / P(ans)

        where:
            P(w) = prior weight of world w (self.weights[i] is the prior weight from last update P(w | ans))
            P(ans | w) = likelihood(w, ans, ...) (lk)
            P(ans) = sum_w'[P(ans | w') * P(w')] (= total, the normalisation constant)

        likelihood: the P(ans | w) function; defaults to self.answer_likelihood (uniform).
        NeuroSymbolicKripkeState passes its soft likelihood here.
        Worlds where P(ans | w) = 0 are removed from the accessibility set R_Prince.
        """
        likelihood = likelihood if likelihood is not None else self.answer_likelihood
        new_w, new_wt = [], []
        for world, wt in zip(self.worlds, self.weights):
            lk = likelihood(world, qa.answer, qa.question_type, qa.target_id, qa.question_param)
            nw = wt * lk # numerator: P(ans, w) = P(w) * P(ans | w)
            if nw > 0:
                new_w.append(world)
                new_wt.append(nw)

        total = sum(new_wt) # P(ans) = sum_w P(ans | w) * P(w)
        if total <= 0:
            # Every world was eliminated, so the environment produced an answer no
            # role assignment allows, or a likelihood model returned zero where it
            # must not. This should not happen in a valid game!!!
            raise BeliefUpdateError(qa.answer, qa.target_id, qa.question_type.value, qa.round)
        self.worlds = new_w
        self.weights = [w / total for w in new_wt] # P(w | ans) = numerator / P(ans)
        self.__num_qa += 1


    def sync_with_history(self, qa_history: list[QAPair]) -> None: 
        """Apply any QA pairs in *qa_history* that haven't been applied yet."""
        for qa in qa_history[self.__num_qa:]:
            self.update_on_answer(qa)


    def princess_distribution(self) -> dict[int, float]:
        """P(princess = x) for each player x (−1 = center).
        
        Formula: P(princess = x) = sum_{w: princess(w)=x} P(w)
        """
        dist: dict[int, float] = defaultdict(float)
        for w, wt in zip(self.worlds, self.weights):
            dist[w.princess_holder] += wt # P(princess=x) += P(w)
        return dict(dist)

    def queen_distribution(self) -> dict[int, float]:
        """P(queen = x) for each player x.
        
        Formula:P(queen = x) = sum_{w: queen(w)=x} P(w)
        """
        dist: dict[int, float] = defaultdict(float)
        for w, wt in zip(self.worlds, self.weights):
            dist[w.queen_holder] += wt # P(queen=x) += P(w)
        return dict(dist)

    def role_distribution(self, player_id: int) -> dict[str, float]:
        """P(player x has role R) for every role R.
        
        Formula: P(role(player) = R) = sum_{w: role_w(player)=R} P(w)
        """
        dist: dict[str, float] = defaultdict(float)
        for w, wt in zip(self.worlds, self.weights):
            if player_id == -1:
                if w.center_card:
                    dist[w.center_card] += wt
                else:
                    raise ValueError("Center card is disabled but role_distribution called with player_id=-1")
                    # TODO add more safety checks
            else:
                dist[w.roles[player_id]] += wt
        return dict(dist)


    @staticmethod
    def entropy(dist: dict) -> float: # TODO: can be moved to helper utils
        """Shannon entropy.
        
        Formula:
            H(X) = − sum_x [P(x) * log(P(x))]

        where dist = {x: P(x)} is a probability distribution (values sum to 1).
        """
        # H(X) = − sum_x P(x) * log(P(x))
        return -sum(p * math.log2(p) for p in dist.values() if p > 0)

    def princess_entropy(self) -> float:
        """H(Princess is someone) = − sum_x [P(princess=x) * log(P(princess=x))]"""
        return self.entropy(self.princess_distribution())

    def information_gain(self, q_type: QuestionType, target_id: int, q_param: int | None, \
        likelihood=None) -> float:
        """
        Expected information gain about Princess identity for a candidate question.

        likelihood selects the P(answer | world) function used to weight each
        hypothetical answer; defaults to self.answer_likelihood (cheap uniform IG,
        used to rank ALL candidates). NeuroSymbolicKripkeState.soft_information_gain
        passes its soft likelihood to re-score only the top-k candidates -- this is
        what lets Top-k forward IG decouple LLM cost from the action-space size.

        Formula:
            IG = H(Princess is someone) − Expected_Ans[H(Princess is someone | Ans)]
               = H(Princess is someone) − sum_a [P(Ans = a) * H(Princess is someone | Ans = a)]

        where:
            H(Princess) = prior Shannon entropy over who holds Princess
            P(a) = sum_w P(w) * P(a | w) (marginal probability of answer a)
            H(Princess | a) = entropy of posterior princess distribution given answer a

        The question that maximises IG should, on average, reduces uncertainty the most.
        """
        h_prior = self.princess_entropy() # H(Princess)
        # if h_prior < threshold: TODO: tune threshold is required
        #     Entropy ≈ 0 means we already (almost) know who Princess is.
        #     No question can reduce uncertainty further; return 0 to avoid floating-point noise in the 
        #     subtraction below.
        #     return 0.0

        likelihood = likelihood if likelihood is not None else self.answer_likelihood
        all_ans = self._all_answers(q_type)
        expected_h_post = 0.0 # will accumulate sum_a P(a)*H(Princess|a)

        for ans in all_ans: # calculate IG for all possible answers and take expectation
            post_w: list[PossibleWorld] = []
            post_wt: list[float] = []
            p_ans = 0.0 # P(a) = sum_w [P(w)·P(a|w)]
            for world, wt in zip(self.worlds, self.weights):
                lk = likelihood(world, ans, q_type, target_id, q_param)  # P(a | w)
                nw = wt * lk # P(w) * P(a | w)
                if nw > 0:
                    post_w.append(world)
                    post_wt.append(nw)
                p_ans += nw # P(a)

            if p_ans <= 0:
                # This answer is impossible with current beliefs, result in 0 H(Princess|A)
                continue

            # Build posterior princess distribution: P(princess=x | a)
            # For each world:  P(w | a) = P(w)·P(a|w) / P(a) = pw / p_ans
            post_dist: dict[int, float] = defaultdict(float)
            for world, pw in zip(post_w, post_wt):
                post_dist[world.princess_holder] += pw / p_ans # P(princess=x | a) += P(w | a)

            # sum: P(a) * H(Princess | a)
            expected_h_post += p_ans * self.entropy(dict(post_dist)) # += P(A=a) * H(Princess | A=a)

        return h_prior - expected_h_post # IG = H(Princess) − sum_a [P(a) * H(Princess | a)]



    def answer_value_queen(self, answer: str, q_type: QuestionType, target_id: int, q_param: int | None, \
        queen_holder_id: int) -> float: # Answer-value evaluation (for strategic Queen faction agents)
        # TODO: can input directly with QAPair
        """
        Value of answer for Queen faction.

        Simulates Prince's Bayesian update and computes:
            value(a) = P(princess = queen_holder | a)

        i.e. the probability that Prince will believe the actual Queen player is Princess after hearing this
        answer. Higher is better for Queen.

        Formula:
            value(a) = sum_{w: princess(w)=queen_id}  P(w|a)

        where P(w|a) = P(a|w)·P(w) / P(a) (Bayes' rule)
                P(a) = sum_w P(a|w)·P(w) (marginal)
        """
        post_w, post_wt = [], []
        for world, wt in zip(self.worlds, self.weights):
            lk = self.answer_likelihood(world, answer, q_type, target_id, q_param) # P(a | w)
            nw = wt * lk # P(a|w) * P(w)
            if nw > 0:
                post_w.append(world)
                post_wt.append(nw)

        total = sum(post_wt) # P(a) = sum_w P(a|w)·P(w)
        if total <= 0:
            # This answer is impossible under Prince's beliefs
            return 0.0

        # value(a) = sum_{w: princess(w)=queen_id} P(w|a)
        # = sum_{w: princess(w)=queen_id} [P(a|w)·P(w)] / P(a)
        return sum(pw for world, pw in zip(post_w, post_wt) \
            if world.princess_holder == queen_holder_id) / total


    def answer_value_other(self, answer: str, q_type: QuestionType, target_id: int, q_param: int | None, \
        princess_holder_id: int, queen_holder_id: int) -> float: # TODO: can input directly with QAPair
        """
        Value of *answer* for Other faction.

        Goal: Prince picks neither Princess nor Queen.

        Heuristic scoring formula:
            value(a) = 0.5 * P_safe(a) + 0.5 * H_norm(a)

        where:
            P_safe(a)  = 1 − P(princess=real_princess | a) − P(princess=real_queen | a) probability Prince
            guesses someone who is neither Princess nor Queen
            
            H_norm(a)  = H(Princess | a) / log₂(N) normalised posterior entropy (confusion level, 0~1)
            N = number of places the Princess can be: the non-prince players, plus
            the center card when there is one (used to normalise entropy to [0,1])

        P_safe is the primary objective; H_norm is secondary (prefer more confusion).
        Weights 0.5/0.5 are tunable hyperparameters.
        """
        post_w, post_wt = [], []
        for world, wt in zip(self.worlds, self.weights):
            lk = self.answer_likelihood(world, answer, q_type, target_id, q_param) # P(a | w)
            nw = wt * lk # P(a|w) * P(w)  (unnormalised posterior)
            if nw > 0:
                post_w.append(world)
                post_wt.append(nw)

        total = sum(post_wt) # P(a) = sum_w P(a|w)·P(w)
        if total <= 0:
            # This answer is impossible under Prince's beliefs (shouldn't happen in valid game).
            return 0.0

        # P(princess = real_princess | a) = sum_{w: princess(w)=princess_id} P(w|a)
        p_princess = sum(pw for w, pw in zip(post_w, post_wt) \
            if w.princess_holder == princess_holder_id) / total

        # P(princess = real_queen | a) = sum_{w: princess(w)=queen_id} P(w|a)
        p_queen = sum(pw for w, pw in zip(post_w, post_wt) \
            if w.princess_holder == queen_holder_id) / total

        # P_safe(a) = 1 − P(princess=real_princess|a) − P(princess=real_queen|a)
        safe_prob = 1.0 - p_princess - p_queen

        # H(Princess | a): posterior entropy for confusion measurement
        post_dist: dict[int, float] = defaultdict(float)
        for w, pw in zip(post_w, post_wt):
            post_dist[w.princess_holder] += pw / total # P(princess=x | a) = P(w|a)
        h_post = self.entropy(dict(post_dist)) # H(Princess | a)

        # Number of possible Princess locations:
        # (num_players - 1) non-prince players + (1 if center card exists, 0 otherwise)
        num_candidates = (self.num_players - 1) + (1 if self.use_center_card and self.num_players != 8 else 0)

        # H_norm(a) = H(Princess | a) / log₂(num_candidates)   (normalise to [0,1])
        h_norm = h_post / max(math.log2(num_candidates), 1.0)

        # value(a) = 0.5 * P_safe(a) + 0.5 * H_norm(a)
        return 0.5 * safe_prob + 0.5 * h_norm # TODO: tune weights is required!!!
        # return safe_prob


    def num_worlds(self) -> int:
        return len(self.worlds)


    @classmethod
    def from_history(cls, num_players: int, use_center_card: bool, qa_history: list[QAPair]) -> KripkeState:
        """Reconstruct Prince's belief state from scratch given full QA history."""
        # TODO
        ks = cls(num_players, use_center_card)
        for qa in qa_history:
            ks.update_on_answer(qa)
        return ks

    def get_summary_Princess(self) -> str:
        """Human-readable summary of belief state."""
        lines = [f"Number of accessible worlds: {self.num_worlds()}"]

        p_dist = self.princess_distribution()
        lines.append("Princess distribution:")
        for holder in sorted(p_dist, key=lambda x: -p_dist[x]): # sort by probability from high to low
            label = "center" if holder == -1 else f"Player {holder}"
            lines.append(f"  {label}: {p_dist[holder]:.4f}")
        lines.append(f"Princess entropy: {self.princess_entropy():.4f} bits")

        return "\n".join(lines)

    def get_full_report(self) -> str:
        """Detailed report including every role distributions."""
        lines = [self.get_summary_Princess(), ""]

        lines.append("All role distributions:")
        for pid in range(1, self.num_players):
            role_dist = self.role_distribution(pid)
            parts = ", ".join(f"{r}: {p:.3f}" for r, p in sorted(role_dist.items(), key=lambda x: -x[1]))
            lines.append(f"  Player {pid}: {parts}")

        if self.use_center_card and self.num_players != 8:
            role_dist = self.role_distribution(-1)
            parts = ", ".join(f"{r}: {p:.3f}" for r, p in sorted(role_dist.items(), key=lambda x: -x[1]))
            lines.append(f"  Center: {parts}")

        return "\n".join(lines)

    def get_top_worlds_report(self, top_n: int = 5) -> str:
        """Report showing the most probable possible worlds.

        Alternative belief representation: instead of role distributions,
        show world assignments sorted by posterior probability.

        Args:
            top_n: number of top worlds to show (default 5).
        """
        if not self.worlds:
            return "No possible worlds remain."

        # Sort worlds by weight descending
        ranked = sorted(zip(self.weights, self.worlds), key=lambda x: -x[0])
        show = ranked[:top_n]

        lines = [f"Top {min(top_n, len(ranked))} most probable worlds (out of {len(ranked)} total):"]

        for i, (wt, world) in enumerate(show, 1):
            assignments = ", ".join(f"P{pid}={role}" for pid, role in sorted(world.roles.items()))
            if world.center_card:
                assignments += f", center={world.center_card}"
            lines.append(f"{i}. (prob={wt:.4f}): {assignments}")

        return "\n".join(lines)



class NeuroSymbolicKripkeState(KripkeState):
    # Codebook (static; tuned once). ANSWER_WEIGHT is normalised, so its scale is
    # irrelevant (integers fine; 4:2:1 = log-symmetric +/-1 notch). CONF is the
    # lambda of  p' = lam*p + (1-lam)*uniform, a blend ratio in [0,1] (must be fractional).
    ANSWER_WEIGHT: dict[str, int] = {"LIKELY": 4, "NEUTRAL": 2, "UNLIKELY": 1}
    CONF: dict[str, float] = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}

    _judge_cache: dict[tuple, JudgeResult] = {}  # class-level: shared across instances/games

    def __init__(self, num_players: int, use_center_card: bool = True, llm_client: LLMClient=None, *,\
        weights: dict[str, int] = None, shrink: dict[str, float] = None, use_cache: bool = True, \
            seed_source=None, judge_reasoning_effort: str = None, judge_cot: bool = False):
        super().__init__(num_players, use_center_card)
        self.llm_client = llm_client
        # NB: distinct from self.weights (the inherited world posterior). These are
        # the codebook (answer-rating weights / confidence shrinkage).
        self.answer_weight = weights if weights is not None else self.ANSWER_WEIGHT
        self.conf = shrink if shrink is not None else self.CONF
        self.use_cache = use_cache
        self.seed_source = seed_source
        self.judge_reasoning_effort = judge_reasoning_effort
        # Thinking effort for the judge sub-calls only
        # judge_cot: manual chain-of-thought.
        self.judge_cot = judge_cot
        self.stats = LikelihoodStats()  # per-instance (per-game) observability
        # Full record of every judge LLM call: the situation, the manual-CoT reasoning, the model thinking, and the
        # ratings.
        self.judge_log: list[dict] = []

    @classmethod
    def clear_cache(cls) -> None:
        cls._judge_cache = {}

    def update_on_answer(self, qa: QAPair, likelihood=None) -> None:
        super().update_on_answer(qa, likelihood=self.soft_likelihood)

    def soft_information_gain(self, q_type: QuestionType, target_id: int, q_param: int | None) -> float:
        """LLM-informed IG"""
        return self.information_gain(q_type, target_id, q_param, likelihood=self.soft_likelihood)

    def soft_likelihood(self, world: PossibleWorld, answer: str, q_type: QuestionType, target_id: int, \
        q_param: int | None) -> float:
        """P(answer | world) with the hard mask kept and the soft branch judged by the LLM."""
        truth = world.get_truth(q_type, target_id, q_param)
        constraint = world.get_constraint(target_id)
        all_ans = self._all_answers(q_type)
        allowed = [] # all allowed answers under the constraint

        if constraint == Constraint.MUST_TRUTH:
            return 1.0 if answer == truth else 0.0
        if constraint == Constraint.MUST_LIE:
            if answer == truth:
                return 0.0
            allowed = [a for a in all_ans if a != truth]
        else:  # NO_CONSTRAINT (truth itself is allowed)
            allowed = all_ans

        return self._soft_distribution(world, q_type, target_id, q_param, allowed, truth).get(answer, 0.0)

    def soft_distribution(self, world: PossibleWorld, q_type: QuestionType, target_id: int, \
        q_param: int | None) -> dict[str, float]:
        """Full P(answer | world) over the answers the constraint allows.

        The same distribution soft_likelihood indexes into (for offline analysis of
        what the judge predicted). MUST_TRUTH returns the single truthful answer.
        """
        truth = world.get_truth(q_type, target_id, q_param)
        constraint = world.get_constraint(target_id)
        all_ans = self._all_answers(q_type)
        if constraint == Constraint.MUST_TRUTH:
            return {truth: 1.0}
        if constraint == Constraint.MUST_LIE:
            allowed = [a for a in all_ans if a != truth]
        else:  # NO_CONSTRAINT (truth itself is allowed)
            allowed = all_ans
        return self._soft_distribution(world, q_type, target_id, q_param, allowed, truth)

    def _soft_distribution(self, world: PossibleWorld, q_type: QuestionType, target_id: int, \
        q_param: int | None, allowed: list[str], truth: str) -> dict[str, float]:
        """q(answer | world) over `allowed` (sums to 1): cached judge ratings -> codebook -> shrink."""
        self.stats.queries += 1
        query = self._build_query(world, q_type, q_param, allowed, truth, world.roles[target_id], target_id)
        key = query.key()

        if self.use_cache and key in self._judge_cache:
            self.stats.cache_hits += 1
            result = self._judge_cache[key]
        else:
            result = self._judge(query)
            self.stats.judge_calls += 1
            if self.use_cache:
                self._judge_cache[key] = result

        return self._to_distribution(allowed, q_type, q_param, world, result)

    def _judge(self, query: JudgeQuery) -> JudgeResult: # TODO
        """Ask the LLM to rate the outcome-slots. Override in tests to mock offline."""

        system_prompt, user_prompt = self._judge_prompt(query)

        # the runner skips a game only on LLMParseError, treating anything else as fatal to the whole experiment.
        cot_analysis = None  # manual-CoT reasoning pass text
        cot_thinking = None  # model thinking on the reasoning pass
        try:
            if self.judge_cot:
                # Manual CoT step 1: a dedicated reasoning pass (no ratings). Step 2
                # then rates conditioned on that analysis.
                from agents.our_agents.prompts import judge_cot_analysis_prompt, judge_cot_final_prompt
                analysis_seed = self.seed_source.next_seed() if self.seed_source is not None else None
                analysis: JudgeAnalysis
                analysis, cot_thinking = self.llm_client.chat_structured_model(user_prompt=judge_cot_analysis_prompt(user_prompt), \
                    system_prompt=system_prompt, response_model=JudgeAnalysis, seed=analysis_seed, \
                        reasoning_effort=self.judge_reasoning_effort)
                cot_analysis = analysis.analysis
                user_prompt = judge_cot_final_prompt(user_prompt, cot_analysis)

            seed = self.seed_source.next_seed() if self.seed_source is not None else None

            judgment: LikelihoodJudgment
            judgment, reasoning = self.llm_client.chat_structured_model(user_prompt=user_prompt, \
                system_prompt=system_prompt, response_model=LikelihoodJudgment, seed=seed, \
                    reasoning_effort=self.judge_reasoning_effort)
        except Exception as e:
            from agents.base import LLMParseError
            raise LLMParseError("SoftLikelihoodJudge", e)
        
        rating_by_label = {r.label: r.rating.upper() for r in judgment.ratings}
        # Fill any slot the LLM left unrated with NEUTRAL, but COUNT it (not silent):
        # a persistently high defaulted_slots means the judge is omitting slots and the
        # soft signal is silently degrading toward uniform.
        missing = [lbl for lbl in query.slots if lbl not in rating_by_label]
        if missing:
            self.stats.defaulted_slots += len(missing)
            for lbl in missing:
                rating_by_label[lbl] = "NEUTRAL"
        self._log_judgment(query, cot_analysis, cot_thinking, judgment, reasoning, rating_by_label)
        return JudgeResult(rating_by_label=rating_by_label, confidence=judgment.confidence.upper())

    def _log_judgment(self, query: JudgeQuery, cot_analysis, cot_thinking, judgment, reasoning, \
        rating_by_label: dict) -> None:
        """Record one judge call (situation + all LLM outputs) for later analysis.

        The judge's raw output is the per-slot ratings + confidence; the answer
        distribution is derived from them later in _to_distribution.
        """
        entry = {
            "situation": {
                "q_type": query.q_type, "answerer_faction": query.answerer_faction,
                "constraint": query.constraint, "truth_label": query.truth_label,
                "slots": list(query.slots),
            },
            "cot_analysis": cot_analysis,  # manual-CoT reasoning pass (null if judge_cot off)
            "cot_thinking": cot_thinking,  # model thinking on that pass (null if none)
            "reasoning": judgment.reasoning,  # the judge's own one-line rationale
            "thinking": reasoning,  # model thinking on the rating call (null if none)
            "ratings": rating_by_label,
            "confidence": judgment.confidence.upper(),
        }
        # A MemoryJudgeQuery also identifies which opponent / memory snapshot this was.
        opponent_id = getattr(query, "opponent_id", None)
        if opponent_id is not None:
            entry["situation"]["opponent_id"] = opponent_id
            entry["situation"]["memory_version"] = query.memory_version
        self.judge_log.append(entry)

    def _judge_prompt(self, query: JudgeQuery) -> tuple[str, str]:
        """Prompt for one judge call (overridden by MemoryKripkeState)."""
        from agents.our_agents.prompts import build_judge_prompt
        return build_judge_prompt(query)

    def _build_query(self, world: PossibleWorld, q_type: QuestionType, q_param: int | None, allowed: list[str], \
        truth: str, role_hyp: int, target_id: int = None) -> JudgeQuery:
        labels = [implication(a, q_type, q_param, world) for a in allowed]
        
        return JudgeQuery(q_type=q_type.value, answerer_faction=ROLES[role_hyp].faction.name, \
            constraint=ROLES[role_hyp].constraint.name, truth_label=implication(truth, q_type, q_param, world),\
                slots=tuple(sorted(set(labels))))

    def _to_distribution(self, allowed: list[str], q_type: QuestionType, q_param: int | None, \
        world: PossibleWorld, result: JudgeResult) -> dict[str, float]:
        
        confidence_score = self.conf.get(result.confidence, 0.6)
        
        raw_rate = {a: self.answer_weight.get(result.rating_by_label.get(implication(a, q_type, q_param, world), "NEUTRAL"), 1.0) \
            for a in allowed} # raw weight per allowed answer
        total = sum(raw_rate.values())
        num_ans = len(allowed)
        
        uniform = 1.0 / num_ans
        dist = {a: confidence_score * (raw_rate[a] / total) + (1.0 - confidence_score) * uniform for a in allowed}
        # λ x normalised LLM weight + (1-λ) x uniform score
        s = sum(dist.values())  # guard fp drift
        return {a: p / s for a, p in dist.items()}



class MemoryKripkeState(NeuroSymbolicKripkeState):
    """Soft likelihood conditioned on CROSS-GAME opponent memory.

    Same neuro-symbolic pipeline as NeuroSymbolicKripkeState (hard constraint
    mask kept, LLM-judged slot ratings, confidence shrink towards uniform), but
    the judge rates ONE CONCRETE opponent and is shown that opponent's raw
    behaviour from previous games with the same people (true role + every
    answer, ground-truth annotated; see opponent_memory.CrossGameMemory). The
    judged likelihood drives BOTH the Bayesian belief update and the IG
    question-selection forward simulation, exactly like the parent class.

    Design notes vs the EMO baseline this replaces:
        - Revision without new actions: the memory only shapes P(answer | world);
            every observed answer still re-weights ALL worlds via Bayes, so beliefs
            about a silent opponent keep moving as others answer.
        -   Bounded LLM output: the judge emits one ordinal rating per outcome
            slot, never free-form predictions, so small-model inconsistency cannot
            corrupt the probabilistic bookkeeping.
        - Cost: judged situations are cached per (situation, opponent, memory
            version); with no recorded games this degrades to exactly the
            opponent-agnostic NeuroSymbolicKripkeState judgment (plus the
            per-opponent cache split).
    """

    def __init__(self, num_players: int, use_center_card: bool = True, llm_client: LLMClient = None, *, \
        memory: CrossGameMemory, weights: dict[str, int] = None, shrink: dict[str, float] = None, \
            use_cache: bool = True, seed_source=None, judge_reasoning_effort: str = None, \
                judge_cot: bool = False):
        super().__init__(num_players, use_center_card, llm_client, weights=weights, shrink=shrink, \
            use_cache=use_cache, seed_source=seed_source, judge_reasoning_effort=judge_reasoning_effort, \
                judge_cot=judge_cot)
        self.memory = memory

    def _build_query(self, world: PossibleWorld, q_type: QuestionType, q_param: int | None, allowed: list[str], \
        truth: str, role_hyp: int, target_id: int = None) -> MemoryJudgeQuery:
        base = super()._build_query(world, q_type, q_param, allowed, truth, role_hyp, target_id)
        return MemoryJudgeQuery(q_type=base.q_type, answerer_faction=base.answerer_faction, \
            constraint=base.constraint, truth_label=base.truth_label, slots=base.slots, \
                opponent_id=target_id, memory_version=self.memory.version, \
                    memory=self.memory.opponent_report(target_id))

    def _judge_prompt(self, query: MemoryJudgeQuery) -> tuple[str, str]:
        from agents.our_agents.prompts import build_memory_judge_prompt
        return build_memory_judge_prompt(query)


class LogicalKripkeState(KripkeState):
    """Order-0 Prince belief: only the *set of logically-possible worlds*.

    This Prince "completely ignores the opponents". It rules out a world only when
    the observed answer is impossible in it (a MUST_TRUTH role that lied, a MUST_LIE
    role that told the truth) and keeps every surviving world equally weighted (no likelihood).

    This is the belief a 1st-order opponent best-responds to (see StrategicKripkeState).
    """

    def answer_likelihood(self, world: PossibleWorld, answer: str, q_type: QuestionType, target_id: int, \
        q_param: int | None) -> float:
        """1.0 if `answer` is logically possible in `world`, else 0.0 (renormalised by
        the caller -> uniform over the surviving worlds)."""
        truth = world.get_truth(q_type, target_id, q_param)
        constraint = world.get_constraint(target_id)
        if constraint == Constraint.MUST_TRUTH:
            return 1.0 if answer == truth else 0.0
        if constraint == Constraint.MUST_LIE:
            return 0.0 if answer == truth else 1.0  # every lie equally possible (no 1/(A-1))
        return 1.0 # NO_CONSTRAINT: any answer possible (no 1/A)


class StrategicKripkeState(KripkeState):
    """Prince belief with a *2nd-order ToM (strategic-opponent) answer likelihood*.

    This subclass is a 2nd-order Prince: it models each opponent
    as a *1st-order* best-responder -- an opponent that assumes the Prince is order-0
    (only tracks the logically-possible worlds, ``LogicalKripkeState``, otherwise
    ignoring the opponents) and picks answers that help its faction against that belief.
    """

    # Inverse-temperature of the quantal best-response (p(a) ∝ exp(beta·value(a))).
    # beta -> ∞ recovers a deterministic arg-max best-response; beta = 0 collapses back
    # to the uniform model. Kept finite so the soft branch never hard-eliminates a
    # world on a modelling mismatch. TODO: tune.
    BETA: float = 8.0
    # eps_greedy smoothing: P = (1-eps)*[argmax, split over ties] + eps*uniform.
    # Scale-free (only the value ORDER matters), so eps transfers across player counts /
    # center-card configs, and it has a closed-form estimate from logged games:
    # eps_hat = (1 - argmax_match) / (1 - 1/|A|), reported by beta_mle.py.
    EPS: float = 0.1

    def __init__(self, num_players: int, use_center_card: bool = True, beta: float = None, \
        smoothing: str = "softmax", eps: float = None):
        super().__init__(num_players, use_center_card)
        self.beta = beta if beta is not None else self.BETA
        self.eps = eps if eps is not None else self.EPS
        if smoothing not in ("softmax", "eps_greedy"):
            raise ValueError(f"Unknown smoothing: '{smoothing}'")
        if smoothing == "eps_greedy" and not 0.0 < self.eps <= 1.0:
            # eps = 0 would hard-eliminate every non-argmax answer (the failure mode
            # smoothing exists to prevent); eps = 1 is the uniform model.
            raise ValueError(f"eps_greedy needs 0 < eps <= 1, got {self.eps}")
        self.smoothing = smoothing
        # The order-0 (logical) Prince belief the 1st-order opponent imagines.
        self._shadow = LogicalKripkeState(num_players, use_center_card)
        # Per-question answer distribution, keyed on the world facts it depends on
        # (see _strategic_distribution). Valid only while the shadow belief is
        # unchanged; cleared whenever it advances.
        self._dist_cache: dict[tuple, dict[str, float]] = {}

    def update_on_answer(self, qa: QAPair, likelihood=None) -> None:
        # Base update uses self.answer_likelihood (strategic) via its default arg.
        super().update_on_answer(qa)
        # Advance the opponent's imagined (order-0 logical) belief AFTER the update
        # above, so that update saw only the pre-answer public history.
        self.advance_shadow(qa)

    def advance_shadow(self, qa: QAPair) -> None:
        """Advance only the opponent-side shadow (opponent model) belief.

        answer_likelihood depends on the shadow alone, so a caller that uses this
        state as a component model (BayesKripkeState, update_type_posteriors) can
        advance it with this instead of paying for the unused own-belief update.
        """
        self._shadow.update_on_answer(qa)
        self._dist_cache.clear()  # shadow advanced -> cached distributions are stale

    def answer_likelihood(self, world: PossibleWorld, answer: str, q_type: QuestionType, target_id: int, \
        q_param: int | None) -> float:
        """P(answer | world) under the strategic (2nd-order) opponent model."""
        truth = world.get_truth(q_type, target_id, q_param)
        constraint = world.get_constraint(target_id)

        if constraint == Constraint.MUST_TRUTH:
            return 1.0 if answer == truth else 0.0
        if constraint == Constraint.MUST_LIE:
            if answer == truth:
                return 0.0
            allowed = [a for a in self._all_answers(q_type) if a != truth]
        else:  # NO_CONSTRAINT (truth itself is allowed)
            allowed = self._all_answers(q_type)

        return self._strategic_distribution(world, q_type, target_id, q_param, allowed).get(answer, 0.0)

    def _strategic_distribution(self, world: PossibleWorld, q_type: QuestionType, target_id: int, \
        q_param: int | None, allowed: list[str]) -> dict[str, float]:
        """Smoothed best-response over `allowed` for the answerer in `world`.

        Faction value is scored on ``self._shadow`` (the order-0 logical Prince belief
        the 1st-order opponent imagines), reusing the exact answer-value functions the
        symbolic players use:
            QUEEN -> answer_value_queen  (push P(Prince guesses the Queen holder) up)
            OTHER -> answer_value_other  (keep the Prince away from Princess & Queen)
        PRINCESS-faction answerers are MUST_TRUTH and never reach the soft branch.

        `smoothing` turns the values into a distribution:
            softmax -> p(a) ∝ exp(beta * value(a)) (gap-sensitive, needs beta)
            eps_greedy -> (1-eps)*[argmax, split over ties] + eps*uniform (scale-free; eps estimable in closed form, see class docs)
        """
        if not allowed:
            return {}
        faction = world.get_faction(target_id)
        # key = (q_type, target_id, q_param, world.hash)
        key = (q_type, target_id, q_param, faction, world.princess_holder, world.queen_holder, world.get_truth(q_type, target_id, q_param))
        # likelihood is called once per (candidate answer x world) inside
        # information_gain: so for ONE candidate question the SAME world's full answer
        # distribution is recomputed once per candidate answer; the cache removes that
        # repeat.
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached

        if faction == Faction.QUEEN:
            queen_id = world.queen_holder
            values = [self._shadow.answer_value_queen(a, q_type, target_id, q_param, queen_id) for a in allowed]
        elif faction == Faction.OTHER:
            princess_id, queen_id = world.princess_holder, world.queen_holder
            values = [self._shadow.answer_value_other(a, q_type, target_id, q_param, princess_id, queen_id) \
                for a in allowed]
        # else:  # PRINCESS faction only reaches here if a role is not MUST_TRUTH -> stay uniform
        #     dist = {a: 1.0 / len(allowed) for a in allowed}
        #     self._dist_cache[key] = dist
        #     return dist
        # This should never happen in a valid game
        else:
            raise ValueError(f"Unexpected faction {faction} for player {target_id} in world {world}")

        top = max(values)
        if self.smoothing == "eps_greedy":
            ties = [v == top for v in values]
            n_ties = sum(ties)
            dist = {a: (1.0 - self.eps) * t / n_ties + self.eps / len(allowed) \
                for a, t in zip(allowed, ties)}
        else:  # softmax
            exps = [math.exp(self.beta * (v - top)) for v in values]  # subtract max for numerical stability
            total = sum(exps)
            dist = {a: e / total for a, e in zip(allowed, exps)}
        self._dist_cache[key] = dist
        return dist


class ThirdOrderKripkeState(StrategicKripkeState):
    """Prince belief with a *3rd-order ToM* answer likelihood.
    """

    def __init__(self, num_players: int, use_center_card: bool = True, beta: float = None, \
        smoothing: str = "softmax", eps: float = None):
        super().__init__(num_players, use_center_card, beta, smoothing=smoothing, eps=eps)
        # Replace the order-0 logical shadow with the order-1 uniform Prince belief the
        # symbolic opponents actually best-respond to. update_on_answer/_dist_cache
        # handling is inherited unchanged.
        self._shadow = KripkeState(num_players, use_center_card)


class BayesKripkeState(KripkeState):
    """Bayesian opponent-TYPE memory: per-opponent adaptive answer likelihood.

    Each opponent has a hidden strategy profile: which of the candidate answer models
    (opponent_memory.TYPE_MODELS) best describes how they answer. The belief over it is
    learned across games by update_type_posteriors, and read once per game here.

        symbols:
        j = one opponent, identified by seat (seats keep their profile across games)
        theta = one candidate answer model, i.e. one entry of TYPE_MODELS
        world = one hypothesised role assignment
        past = the answers j gave in PREVIOUS games
        seen = the answers j has given SO FAR in this game

        P(answer | world, theta, j) = the answer probability candidate theta assigns
        P(theta | j, past) = belief that theta describes j, from previous games only
            (memory.type_posterior, read once per game into self._weights)
        P(theta | j, past, seen, world) = the same belief after also using `seen`,
            computed inside `world` (self._joint_post; adapt="joint" only)

    The answer likelihood comes from marginalising over the profile. Writing D for
    whatever evidence the belief has absorbed:

        P(answer | world, j, D) = sum over theta of P(theta | j, D) * P(answer | world, theta, j)

    D drops out of the second factor: once theta is fixed, the candidate model does
    not look at what j answered before, so theta screens D off. It cannot drop out of
    the first factor, which is where all the learning sits.

    The modes differ only in D, so their left-hand sides are different quantities:

        mixture: P(answer | world, j, past) = sum over theta of P(theta | j, past) * P(answer | world, theta, j)
        map: P(answer | world, j, past) = P(answer | world, theta, j) for the
            single theta with the largest P(theta | j, past) (ties -> earlier TYPE_MODELS entry; no data -> uniform)
        joint: P(answer | world, j, past, seen) = sum over theta of P(theta | j, past, seen, world) * P(answer | world, theta, j)

    That difference is the approximation: after j has answered once, the belief
    update needs P(answer | world, j, past, seen), and mixture keeps supplying
    P(answer | world, j, past). `seen` is empty for an opponent's first answer, which
    is why the two agree there.

    P(answer | world, theta, j) is not history-free. The symbolic candidates condition
    on the PUBLIC QA history through their own shadow belief, advanced in
    update_on_answer. That is every player's answers, while `seen` is only j's,
    because only j's answers say anything about j's profile.

    Why "mixture" and "joint" differ. Given the world and the profile, one opponent's
    answers are independent of each other: the profile is the only thing that ties
    them together. So an earlier answer can inform a later one only by saying
    something about the profile. "mixture" holds b(theta_j) fixed for the whole game,
    which cuts that link and makes the answers independent -- the same model you get
    if the opponent draws a fresh profile before every answer. "joint" updates the
    belief after each answer, per world, so a candidate that explained the earlier
    answers well counts for more on the next one. "joint" is the belief over (role
    assignment, profile) that the framework defines; "mixture" approximates it.

    The two agree exactly in two cases: an opponent who answers at most once (nothing
    to carry over), and a b(theta_j) that sits on a single candidate (which is what
    "map" does). In Tofu Kingdom the Prince asks each opponent once and re-asks one
    opponent for the final question, so they can differ only on that one opponent's
    second answer.

    The "llm" candidate is the memoryless soft-likelihood judge (``llm_state``): a
    generic LLM-style opponent. It is one candidate among the others, so an opponent
    that no symbolic model explains gets weight on "llm" through the ordinary type
    posterior -- there is no separate threshold. ``llm_state`` is required whenever
    "llm" is in TYPE_MODELS.
    """

    ADAPT_MODES = ("mixture", "map", "joint")
    # Speed only. In mixture mode an opponent's weights are a posterior over the
    # candidates; after enough games the losing candidate's weight becomes tiny.
    # Dropping a candidate of weight w changes the mixed answer probability by at
    # most w, because that candidate's own probability is <= 1. So at 1e-6 the
    # mixture is correct to 6 decimal places. Not a tuned parameter: any small
    # value gives the same results. 
    #   w = the candidate's posterior weight for this opponent
    PRUNE_WEIGHT = 1e-6

    def __init__(self, num_players: int, use_center_card: bool = True, *, memory: CrossGameMemory, \
        adapt: str = "mixture", tom_beta: float = None, tom_smoothing: str = "softmax", \
            tom_eps: float = None, llm_state: NeuroSymbolicKripkeState = None):
        super().__init__(num_players, use_center_card)
        from agents.our_agents.opponent_memory import TYPE_MODELS
        if adapt not in self.ADAPT_MODES:
            raise ValueError(f"Unknown adapt: '{adapt}' (expected one of {self.ADAPT_MODES})")
        if "llm" in TYPE_MODELS and llm_state is None:
            raise ValueError("BayesKripkeState: the 'llm' type candidate needs a memoryless llm_state judge")
        self.memory = memory
        self.adapt = adapt
        # The memoryless soft-likelihood judge that scores the "llm" candidate.
        self.llm_state = llm_state
        # Symbolic candidate models. They share the tom parametrisation with the
        # post-game scorer (update_type_posteriors); the "uniform" candidate is
        # the inherited answer_likelihood.
        self._symbolic_candidates = {
            "strategic3": ThirdOrderKripkeState(num_players, use_center_card, beta=tom_beta, \
                smoothing=tom_smoothing, eps=tom_eps),
        }
        # Per-opponent weights are fixed within a game: the posterior only moves
        # between games, when update_type_posteriors folds the finished game in.
        self._weights: dict[int, dict[str, float]] = {}
        for pid in range(1, num_players):
            posterior = memory.type_posterior(pid)
            if adapt == "map":
                best = max(posterior, key=posterior.get)
                self._weights[pid] = {best: 1.0}
            else:
                self._weights[pid] = posterior
        # Only a symbolic candidate owns a shadow belief that has to be advanced each
        # QA; "uniform" and "llm" have none. Collect the ones some opponent uses.
        self._shadow_candidates = set()
        for weights in self._weights.values():
            for name, weight in weights.items():
                if name in self._symbolic_candidates and weight >= self.PRUNE_WEIGHT:
                    self._shadow_candidates.add(name)

        # joint mode only: P(theta | j, past, seen, world) for every opponent and every
        # world, normalised over theta. It starts as P(theta | j, past), the same in
        # every world, and each answer reweights it inside each world.
        #   self._joint_post[j][theta][i] = that belief in the world at position i
        #   self._joint_index[id(world)]  = the position i of that world
        # Positions come from the INITIAL world list. self.worlds only ever shrinks as
        # the base update prunes, and a surviving world keeps its identity, so id()
        # always maps it back.
        self._joint_index: dict[int, int] = {}
        self._joint_post: dict[int, dict[str, list[float]]] = {} # j -> theta -> belief per world
        if adapt == "joint":
            self._joint_index = {id(world): i for i, world in enumerate(self.worlds)}
            n = len(self.worlds)
            for pid in range(1, num_players):
                kept = {name: weight for name, weight in self._weights[pid].items() if weight >= self.PRUNE_WEIGHT}
                total = sum(kept.values())  # pruning can drop a little mass; renormalise
                self._joint_post[pid] = {name: [weight / total] * n for name, weight in kept.items()}

    def _candidate_likelihood(self, name: str, world: PossibleWorld, answer: str, q_type: QuestionType, \
        target_id: int, q_param: int | None) -> float:
        """P_t(answer | world) for the single candidate model t = `name`."""
        if name == "uniform":
            return super().answer_likelihood(world, answer, q_type, target_id, q_param)
        if name == "llm":
            return self.llm_state.soft_likelihood(world, answer, q_type, target_id, q_param)
        return self._symbolic_candidates[name].answer_likelihood(world, answer, q_type, target_id, q_param)

    def answer_likelihood(self, world: PossibleWorld, answer: str, q_type: QuestionType, target_id: int, \
        q_param: int | None) -> float:
        """Posterior-weighted mixture (or single routed model) for this opponent."""
        if self.adapt == "joint":
            return self._joint_answer_likelihood(world, answer, q_type, target_id, q_param)
        p = 0.0
        for name, weight in self._weights[target_id].items():
            if weight < self.PRUNE_WEIGHT:
                continue
            p += weight * self._candidate_likelihood(name, world, answer, q_type, target_id, q_param)
        return p

    def _joint_answer_likelihood(self, world: PossibleWorld, answer: str, q_type: QuestionType, \
        target_id: int, q_param: int | None) -> float:
        """Answer probability for adapt="joint".

            P(answer | world, j, past, seen) = sum over theta of P(theta | j, past, seen, world) * P(answer | world, theta, j)

            j = target_id, the opponent being asked
            theta = one candidate answer model (a TYPE_MODELS entry)
            past = the answers j gave in previous games
            seen = the answers j has given so far in this game
            P(theta | j, past, seen, world) = self._joint_post[j][theta][index of world]
            P(answer | world, theta, j) = self._candidate_likelihood(theta, world, ...)

        Same candidates as the mixture, but weighted by how well each one explained
        this opponent's earlier answers IN THIS WORLD. Before the opponent has
        answered, every world still holds P(theta | j, past), so this equals the
        mixture.
        """
        index = self._joint_index[id(world)]
        p = 0.0
        for profile, by_world in self._joint_post[target_id].items():
            weight = by_world[index]
            if weight <= 0.0:
                # This world has ruled the candidate out. Skipping saves a judge call.
                continue
            p += weight * self._candidate_likelihood(profile, world, answer, q_type, target_id, q_param)
        return p

    def update_on_answer(self, qa: QAPair, likelihood=None) -> None:
        # The base update uses self.answer_likelihood (the adapted one). The extra
        # calls below do NOT touch the Prince's belief: they advance the candidates'
        # shadow states (the Prince belief each modelled opponent imagines), which
        # their answer distributions condition on.
        super().update_on_answer(qa)
        # Both advances run AFTER the belief update, so that update saw only the
        # history before this answer.
        if self.adapt == "joint":
            self._advance_joint(qa)
        for name in self._shadow_candidates:
            self._symbolic_candidates[name].advance_shadow(qa)

    def _advance_joint(self, qa: QAPair) -> None:
        """Update the answering opponent's profile belief with their new answer.

            P(theta | j, past, seen, world) <- P(theta | j, past, seen, world)
                * P(answer | world, theta, j), then normalised over theta, for every
            world still in play.

            j = qa.target_id, the opponent who answered
            theta = one candidate answer model (a TYPE_MODELS entry)
            seen = the answers j had already given before this one

        Only the opponent who answered is touched; every other opponent keeps their
        belief. Worlds the base update already pruned are not visited.
        """
        beliefs = self._joint_post[qa.target_id]
        for world in self.worlds:
            index = self._joint_index[id(world)]
            reweighted = {}
            for profile, by_world in beliefs.items():
                likelihood = self._candidate_likelihood(profile, world, qa.answer, qa.question_type, \
                    qa.target_id, qa.question_param)
                reweighted[profile] = by_world[index] * likelihood
            # The base update already raised if this sum was zero in every world, and
            # it prunes any world where it is zero, so every world left here is safe
            # to divide by.
            total = sum(reweighted.values())
            for profile, value in reweighted.items():
                beliefs[profile][index] = value / total
