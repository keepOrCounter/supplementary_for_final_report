"""Our method's symbolic agents: exact Kripke belief tracking + information-gain-driven
active question selection (SymbolicPrinceAgent), its LLM-soft-likelihood variant
(SoftSymbolicPrinceAgent), and the Kripke-driven non-prince opponent (SymbolicPlayerAgent).
"""
from __future__ import annotations

from game.predefined_data_classes import Observation, PrinceObservation, FullyObservation, Role, Phase, \
    QuestionType, Faction
from agents.base import BaseAgent, SymbolicGuessError
from agents.our_agents.Kripke import KripkeState, LogicalKripkeState, NeuroSymbolicKripkeState
from core.helper_functions import action_description


class SymbolicPrinceAgent(BaseAgent): 
    """
    Prince agent driven entirely by Kripke-structure reasoning.

    Question selection: pick the question with highest expected information gain about Princess identity.
    Guess: pick the player (or center) with highest posterior P(Princess).
    """

    def __init__(self, player_id: int, role: Role, num_players: int, use_center_card: bool, \
        print_out: bool = False):
        super().__init__(player_id, role)
        self.kripke = KripkeState(num_players, use_center_card)
        self.print_out = print_out
        self.decision_log: list[dict] = []   # for logging / analysis

    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        if not isinstance(obs, PrinceObservation):
            raise TypeError("SymbolicPrinceAgent expects PrinceObservation")

        # update belief state with any new Q&A
        self.kripke.sync_with_history(obs.qa_history)

        if obs.phase in (Phase.QUESTIONING, Phase.FINAL_QUESTION):
            return self.__select_question(obs, valid_actions)
        elif obs.phase == Phase.GUESSING:
            return self.__guess(obs, valid_actions)
        else:
            raise Exception("Unexpected phase: " + str(obs.phase))

    def __select_question(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """Pick the question with highest expected information gain."""
        best_action = valid_actions[0]
        best_ig = -1.0
        ig_scores = []

        for action in valid_actions:
            q_type = QuestionType(action["question_type"])
            target = action["target"]
            q_param = action.get("question_param")

            ig = self.kripke.information_gain(q_type, target, q_param)
            ig_scores.append((action, ig))

            if ig > best_ig:
                best_ig = ig
                best_action = action

        if self.print_out:
            print(f"\n[SymbolicPrince] number of Worlds: {self.kripke.num_worlds()}")
            print(f"[SymbolicPrince] Princess entropy: ", f"{self.kripke.princess_entropy():.4f} bits")
            ig_scores.sort(key=lambda x: -x[1])
            print("[SymbolicPrince] Top questions by info gain:")
            for act, ig in ig_scores[:3]:
            # for act, ig in ig_scores:
                print(f"  IG={ig:.4f}  {action_description(act)}")

        self.decision_log.append({
            "phase": obs.phase.value,
            "num_worlds": self.kripke.num_worlds(),
            "Princess_entropy": round(self.kripke.princess_entropy(), 4),
            "action_chosen": action_description(best_action),
            "info_gain": round(best_ig, 4),
        })

        return best_action

    def __guess(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """Guess the player with highest P(Princess)."""
        dist = self.kripke.princess_distribution()

        if self.print_out:
            print(f"\n[SymbolicPrince] Final belief state:")
            print(self.kripke.get_full_report())

        best_target = max(dist, key=dist.get)

        for action in valid_actions:
            if action["target"] == best_target:
                self.decision_log.append({
                    "phase": "guessing",
                    "worlds_remaining": self.kripke.num_worlds(),
                    "princess_dist": {str(k): round(v, 4) for k, v in dist.items()},
                    "guess": best_target,
                })
                return action

        # MAP target not found in valid actions, should not happen in normal game!!!!
        raise SymbolicGuessError(self.agent_name, best_target, [a['target'] for a in valid_actions])


class SoftSymbolicPrinceAgent(SymbolicPrinceAgent):
    """Symbolic Prince + LLM powered soft likelihood, selecting via Top-k forward IG.

    Reuses the symbolic baseline unchanged for belief update and guessing. The ONLY
    difference is question selection:
        1. rank ALL candidates by the cheap uniform IG (no LLM),
        2. re-score just the top_k with the LLM-informed soft IG, pick the best.
    """

    def __init__(self, player_id: int, role: Role, num_players: int, use_center_card: bool, \
        llm_client, top_k: int = 3, print_out: bool = False, weights: dict = None, shrink: dict = None, \
            seed_source=None):
        super().__init__(player_id, role, num_players, use_center_card, print_out=print_out)
        
        self.top_k = top_k
        self.llm_client = llm_client  # exposed for logger LLM-stats extraction
        # Replace the baseline's pure-symbolic belief state with the neuro-symbolic one.
        self.kripke = NeuroSymbolicKripkeState(num_players, use_center_card, llm_client, weights=weights, \
            shrink=shrink, seed_source=seed_source)

    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        if not isinstance(obs, PrinceObservation):
            raise TypeError("SoftSymbolicPrinceAgent expects PrinceObservation")
        
        self.kripke.sync_with_history(obs.qa_history)
        if obs.phase in (Phase.QUESTIONING, Phase.FINAL_QUESTION):
            return self.__select_question_soft(obs, valid_actions)
        
        return super().take_action(obs, valid_actions)

    def __select_question_soft(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        def uniform_ig(a):
            return self.kripke.information_gain(
                QuestionType(a["question_type"]), a["target"], a.get("question_param"))
        def soft_ig(a):
            return self.kripke.soft_information_gain(
                QuestionType(a["question_type"]), a["target"], a.get("question_param"))

        # 1. cheap uniform IG ranking for ALL candidates (no LLM)
        uniform_scores = sorted(((a, uniform_ig(a)) for a in valid_actions), key=lambda x: -x[1])
        
        # 2. re-score only the top_k with the soft (LLM-informed) IG
        k = max(1, self.top_k)
        best_action = uniform_scores[0][0]
        best_ig = -1.0
        topk_detail = []
        
        for action, u_ig in uniform_scores[:k]:
            s_ig = soft_ig(action)
            topk_detail.append((action_description(action), round(u_ig, 4), round(s_ig, 4)))
            if s_ig > best_ig:
                best_ig, best_action = s_ig, action

        if self.print_out:
            print(f"\n[SoftSymbolicPrince] worlds={self.kripke.num_worlds()}, "
                    f"Princess entropy={self.kripke.princess_entropy():.4f} bits")
            print("[SoftSymbolicPrince] top_k re-score (uniform IG -> soft IG):")
            for desc, u, s in topk_detail:
                print(f"  {u:.4f} -> {s:.4f}  {desc}")

        self.decision_log.append({
            "phase": obs.phase.value,
            "num_worlds": self.kripke.num_worlds(),
            "Princess_entropy": round(self.kripke.princess_entropy(), 4),
            "scoring": "soft_topk",
            "action_chosen": action_description(best_action),
            "info_gain": round(best_ig, 4),
            "topk_detail": topk_detail,
        })
        return best_action


class SymbolicPlayerAgent(BaseAgent):
    """
    Non-prince agent driven by Kripke-structure reasoning (2nd-order ToM).

    Reconstructs Prince's belief state from public Q&A history, then evaluates each candidate answer by
        simulating Prince's Bayesian update and scoring the result for this player's faction.
    The imagined Prince is ORDER-1: a uniform-likelihood KripkeState (the Prince weighs answers with
        the rational-player model), so this player is a 2nd-order best-responder -- the exact opponent
        the strategic3 Prince likelihood (ThirdOrderKripkeState) assumes. For the 1st-order variant
        (best-response to an order-0 logical Prince) see FirstOrderSymbolicPlayerAgent.

    PRINCESS faction: only one valid answer (truth).
    QUEEN faction: pick lie that maximises P(Prince guesses Queen).
    OTHER faction: pick answer that maximises confusion/avoids Prince finding Princess or Queen.
    """

    def __init__(self, player_id: int, role: Role, num_players: int, use_center_card: bool, \
        print_out: bool = False):
        
        super().__init__(player_id, role)
        self.num_players = num_players
        self.use_center_card = use_center_card
        self.print_out = print_out
        self.decision_log: list[dict] = []
        # Persistent model of the Prince's belief, built lazily on the first answer that
        # needs it.
        self.prince_ks: KripkeState | None = None

    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        if not isinstance(obs, FullyObservation):
            raise TypeError("SymbolicPlayerAgent expects FullyObservation")

        if len(valid_actions) == 1: # if PRINCESS faction, only one valid answer (truth)
            return valid_actions[0]

        faction = obs.player_role.faction

        if faction == Faction.PRINCESS:
            # Princess faction must tell truth -> single answer
            return valid_actions[0]

        # Build the Prince belief model on first use, then advance it with only the QA
        # pairs not yet folded in -- identical result to rebuilding from the full history,
        # but avoids re-enumerating every world and replaying the whole history each turn.
        if self.prince_ks is None:
            self.prince_ks = self._make_prince_model()
        self.prince_ks.sync_with_history(obs.qa_history)
        prince_ks = self.prince_ks
        
        # Current question info
        q = obs.current_question
        q_type = q.question_type
        target_id = q.target_id
        q_param = q.question_param

        # Find true Princess and Queen holders (this player has full info)
        princess_id = self.__find_holder(obs, "Princess Tofu")
        queen_id = self.__find_holder(obs, "Queen Tofu")

        best_action = valid_actions[0]
        best_value = -float("inf")
        value_scores = []

        for action in valid_actions:
            answer = action["answer"]

            if faction == Faction.QUEEN:
                value = prince_ks.answer_value_queen(answer, q_type, target_id, q_param, queen_id)
            else:  # OTHER
                value = prince_ks.answer_value_other(answer, q_type, target_id, q_param,princess_id, \
                    queen_id)

            value_scores.append((action, value))
            if value > best_value:
                best_value = value
                best_action = action

        if self.print_out:
            print(f"\n[SymbolicP{self.player_id} ({obs.player_role.name})] " \
                f"Prince thinks {prince_ks.num_worlds()} possible worlds")
            print(f"Faction: {faction.value}; Constraint: {obs.answer_constraint.value}")
            value_scores.sort(key=lambda x: -x[1])
            print(f"Answer evaluations:")
            for act, val in value_scores[:5]:
                print(f"value={val:.4f}  \"{act['answer']}\"")

        self.decision_log.append({
            "player_id": self.player_id,
            "role": obs.player_role.name,
            "faction": faction.value,
            "prince_worlds": prince_ks.num_worlds(),
            "chosen_answer": best_action["answer"],
            "chosen_value": round(best_value, 4),
        })

        return best_action

    def _make_prince_model(self) -> KripkeState:
        """The Prince belief this player imagines and best-responds to (order-1 uniform)."""
        return KripkeState(self.num_players, self.use_center_card)

    @staticmethod
    def __find_holder(obs: FullyObservation, role_name: str) -> int: # TODO move to helper functions?
        """Find player holding a specific role. -1 if in center."""
        for pid, role in obs.all_roles.items():
            if role.name == role_name:
                return pid
        if obs.center_card and obs.center_card.name == role_name:
            return -1
        return None


class FirstOrderSymbolicPlayerAgent(SymbolicPlayerAgent):
    """1st-order ToM non-prince agent (config type: symbolic1).

    Same best-response machinery as SymbolicPlayerAgent, but the imagined Prince is
    ORDER-0: a LogicalKripkeState that only rules out logically-impossible worlds and
    keeps the survivors uniformly weighted (no answer-likelihood weighting).

    This is EXACTLY the opponent the 2nd-order strategic Prince assumes
    """

    def _make_prince_model(self) -> KripkeState:
        return LogicalKripkeState(self.num_players, self.use_center_card)
