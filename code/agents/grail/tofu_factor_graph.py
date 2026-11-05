"""Factor-graph belief tracking for Tofu Kingdom, GRAIL-style (the analogue of
GRAIL's model_reduced_categories.py).

One role variable per entity (player ids 1..num_players-1, and -1 for the center
card); a "roles form a permutation" hard constraint (the analogue of GRAIL's
"exactly 2 Evil"); answer-likelihood factors p(answer | roles) playing the role
of GRAIL's data-trained NN factors; max-product BP via the vendored pomegranate
fork. See docs/GRAIL_code_analysis.md for the full GRAIL <-> Tofu mapping.
"""
from __future__ import annotations

import itertools
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from pomegranate import FactorGraph, Categorical, JointCategorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from game.predefined_data_classes import ROLES, ROLE_CONFIGS, ROLE_CONFIGS_NO_CENTER, QuestionType, QAPair
from agents.our_agents.Kripke import uniform_answer_likelihood
from core.helper_functions import player_id_from_answer

CENTER_ID = -1
PRINCESS_ROLE = "Princess Tofu"




def valid_player_ids(num_players: int, use_center_card: bool) -> list[int]:
    """players 1..num_players-1, plus the center card (-1).
    
        Returns:
            a list of valid player ids
    """
    ids = list(range(1, num_players))
    if use_center_card and num_players != 8:
        ids.append(CENTER_ID)
    return ids


def QA_referenced_player(q_type: QuestionType, answer: str, question_param: int | None) -> int | None:
    """The player/center card the answer is ABOUT: Q1 -> the pointed-at entity (from the
    answer), Q3 -> question_param, Q2 -> None (only constrains the answerer).
    
    Returns:
        player id
    """
    if q_type == QuestionType.Q1:
        return player_id_from_answer(answer, center_id=CENTER_ID)
    if q_type == QuestionType.Q3:
        return question_param
    return None


def answer_is_truthful(q_type: QuestionType, answer: str, target_role: str, ref_role: str | None) -> bool:
    """Whether the observed answer is the honest one, given the answerer's role
    and the referenced entity's role. Q1 points at an entity (true iff that
    entity is the Princess); Q2 claims own role; Q3 claims entity X's role."""
    if q_type == QuestionType.Q1:
        return ref_role == PRINCESS_ROLE
    if q_type == QuestionType.Q2:
        return answer == target_role
    return answer == ref_role  # Q3


class TofuFactorGraph:
    """Belief over role assignments via max-product BP on a factor graph.

    The graph is rebuilt from the QA history on every inference call. This is
    cheap (graph construction is milliseconds) and avoids the bookkeeping
    GRAIL needs for its fixed-shape state vector; the inference algorithm that
    runs on the graph is GRAIL's own.
    """

    def __init__(self, num_players: int, use_center_card: bool = True, likelihood_codebook=None, \
        posterior_model=None):
        
        self.num_players = num_players
        self.use_center_card = use_center_card and num_players != 8
        role_config = ROLE_CONFIGS if use_center_card else ROLE_CONFIGS_NO_CENTER
        self.role_names: list[str] = list(role_config[num_players])
        self.num_of_roles = len(self.role_names)
        # role_index: role name -> its index in a role variable's probability
        # vector / its axis position in a factor table (the inverse of
        # role_names).
        self.role_index = {r: i for i, r in enumerate(self.role_names)}

        # role-holding entities, in factor-graph variable order
        self.player_ids = valid_player_ids(num_players, self.use_center_card)
        assert len(self.player_ids) == self.num_of_roles, "Tofu Kingdom always has as many role-holding entities as roles"
        # player_id -> its position among the marginal nodes
        self.player_index = {p_id: idx for idx, p_id in enumerate(self.player_ids)}

        self.likelihood_codebook = likelihood_codebook # a table of p(answer | answerer role, referenced entity role) leant from data

        self.posterior_model = posterior_model # a learned NN model to predict p(role_e | history)

        self._permutation_factor_array = self._build_permutation_array()


    def _build_permutation_array(self) -> np.ndarray:
        """Hard constraint: the num_of_roles entities hold the num_of_roles roles, one each.

        This is the Tofu analogue of GRAIL's ``array_filler_evil`` (the
        "exactly two Evil players" JointCategorical): a table that is uniform
        over all legal assignments and zero elsewhere.
        """
        num_of_roles = self.num_of_roles
        table = np.zeros((num_of_roles,) * num_of_roles, dtype=np.float64)
        for perm in itertools.permutations(range(num_of_roles)):
            table[perm] = 1.0
        return table / table.sum()

    def _q1_answers(self) -> list[str]:
        ans = [f"Player {p}" for p in range(1, self.num_players)]
        if self.use_center_card:
            ans.append("center")
        return ans

    def _likelihood(self, q_type: QuestionType, answer: str, target_role: str, ref_role: str | None) -> float:
        """P(answer | answerer role, referenced entity role).
        """
        if self.likelihood_codebook is not None:
            return self.likelihood_codebook(q_type, answer, target_role, ref_role) # learned likelihood
        
        num_answers = len(self._q1_answers()) if q_type == QuestionType.Q1 else self.num_of_roles
        truthful = answer_is_truthful(q_type, answer, target_role, ref_role)
        return uniform_answer_likelihood(truthful, ROLES[target_role].constraint, num_answers)

    def _answer_factor(self, qa: QAPair):
        """P(answer | answerer role, referenced entity role)
        
        factor table for likelihood of an observed answer, 
        conditioned on the roles of the target and referenced entity (if any).
        
        Returns:
            table: np.ndarray, the factor table
            connected_entities: list[int], the player ids of the entities this factor connects to
        """
        q_type, answer, target = qa.question_type, qa.answer, qa.target_id
        ref = QA_referenced_player(q_type, answer, qa.question_param)

        if ref is None or ref == target:
            # unary: when ref == target, the referenced role is the answerer's own role
            table = np.array([
                self._likelihood(q_type, answer, role, role if ref == target else None)
                for role in self.role_names
            ])
            return table, [target]

        table = np.zeros((self.num_of_roles, self.num_of_roles))
        for i, t_role in enumerate(self.role_names):
            for j, r_role in enumerate(self.role_names):
                if i == j:
                    continue  # impossible under the permutation constraint
                table[i, j] = self._likelihood(q_type, answer, t_role, r_role)
        return table, [target, ref]

    def _build_graph(self, qa_history: list[QAPair], priors: dict[int, np.ndarray] | None = None) -> FactorGraph:
        model = FactorGraph()
        
        marginals = []
        for p_id in self.player_ids:
            if priors and p_id in priors:
                role_distribution = np.asarray(priors[p_id], dtype=np.float64)
                role_distribution = role_distribution / role_distribution.sum()
                # normalize just in case and type check
            else:
                role_distribution = np.full(self.num_of_roles, 1.0 / self.num_of_roles)
                # default to uniform if no prior
            marginals.append(Categorical([role_distribution.tolist()])) 
            # variable node: role distribution for every player

        perm_factor = JointCategorical(self._permutation_factor_array, frozen=True)
        # constraint factor: role only exist once
        model.add_factor(perm_factor)

        if self.posterior_model is not None:
            # (A) "posterior factor" mode: one learned unary factor per player/center_card, with
            # p(role_e | whole QA history) -- the faithful analogue of GRAIL's
            # per-player NN factor.
            posteriors = self.posterior_model.predict(qa_history)
            answer_factors = []
            for p_id in self.player_ids:
                dist = np.array([posteriors[p_id][r] for r in self.role_names], dtype=np.float64)
                dist = dist / dist.sum()
                answer_factors.append((Categorical([dist.tolist()]), [p_id]))
                model.add_factor(answer_factors[-1][0])
        else:
            # (B)/likelihood mode: one factor per observed answer (the answer's
            # likelihood, conditioned into a table over the role(s) it touches).
            answer_factors = [] # likelihood, corresponds to GRAIL's NN factors
            for qa in qa_history:
                if qa.answer is None:
                    continue
                table, connected_entities = self._answer_factor(qa)
                if len(connected_entities) == 1:
                    f = Categorical([(table / table.sum()).tolist()])
                else:
                    f = JointCategorical(table / table.sum(), frozen=True)
                answer_factors.append((f, connected_entities))
                model.add_factor(f)

        for m in marginals:
            model.add_marginal(m)

        # edge order must match factor dimension order (see FactorGraph docs)
        for p_id, m in zip(self.player_ids, marginals):
            model.add_edge(m, perm_factor)
        for f, ents in answer_factors:
            for p_id in ents:
                model.add_edge(marginals[self.player_index[p_id]], f)

        return model



    def predict_probs(self, qa_history: list[QAPair], priors: dict[int, np.ndarray] | None = None, \
        algorithm: str = "max") -> dict[int, dict[str, float]]:
        """Run belief propagation; return {player/center_card: {role_name: belief}}.

        ``algorithm`` is "max" (max-product, GRAIL's choice) or "sum"
        (sum-product). All role variables are unobserved, so the evidence
        mask is all-False; the observed answers live inside the factor
        tables (conditioning on an observed leaf is equivalent to reducing
        the factor).
        
        Returns:
            dict: {player_id: {role_name: belief}}
        """
        assert algorithm in ("max", "sum")
        model = self._build_graph(qa_history, priors)

        X = torch.zeros((1, self.num_of_roles), dtype=torch.int32)
        mask = torch.zeros((1, self.num_of_roles), dtype=torch.bool)
        X_masked = torch.masked.MaskedTensor(X, mask=mask)

        beliefs = model.predict_proba(X_masked, alg=algorithm)

        result: dict[int, dict[str, float]] = {}
        for p_id in self.player_ids:
            b = beliefs[self.player_index[p_id]][0]
            result[p_id] = {r: float(b[i]) for i, r in enumerate(self.role_names)}
        return result

    def princess_distribution(self, beliefs: dict[int, dict[str, float]]) -> dict[int, float]:
        """Normalised P(Princess = entity) from per-entity beliefs.
        
        Returns:
            dict: {player_id: belief of player being Princess Tofu}
        """
        raw = {player_id: b.get(PRINCESS_ROLE, 0.0) for player_id, b in beliefs.items()}
        total = sum(raw.values())
        if total <= 0:
            return {player_id: 1.0 / len(raw) for player_id in raw}
        return {player_id: b / total for player_id, b in raw.items()}
