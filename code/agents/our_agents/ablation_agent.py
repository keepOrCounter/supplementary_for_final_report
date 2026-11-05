"""Ablation Prince for our method: question selection, guess strategy and answer
likelihood are all configurable, so each component's contribution can be isolated."""
from __future__ import annotations

from game.predefined_data_classes import Observation, PrinceObservation, Role, Phase, QuestionType
from agents.base import BaseAgent, LLMParseError, SymbolicGuessError
from agents.our_agents.Kripke import KripkeState, NeuroSymbolicKripkeState, StrategicKripkeState, \
    ThirdOrderKripkeState, MemoryKripkeState, BayesKripkeState
from agents.our_agents.opponent_memory import CrossGameMemory
from agents.our_agents.prompts import build_ablation_prince_prompt
from core.llm_client import LLMClient
from core.helper_functions import action_description, SeedSource


class AblationPrinceAgent(BaseAgent):
    """
    Prince agent for ablation experiments.

    Both question selection and guess are configurable:

    question_strategy: "ig" | "random" | "llm"
    guess_strategy:    "map" | "random" | "llm"
    likelihood:        "uniform" | "strategic" | "strategic3" | "llm" | "memory"  (opponent answer model)

    The `likelihood` factor IS the opponent model (answer likelihood). It governs BOTH
    the Bayesian belief update (MAP guess + belief context shown to the LLM) AND the IG
    question-selection forward simulation, so question selection uses the same opponent
    model:
        uniform -> KripkeState (1st-order: opponents answer uniformly at random)
        strategic -> StrategicKripkeState (2nd-order ToM: opponents best-respond to an order-0 logical Prince)
        strategic3 -> ThirdOrderKripkeState (3rd-order ToM: opponents best-respond to a uniform order-1 Prince, 
            the exact model of SymbolicPlayerAgent)
        llm -> NeuroSymbolicKripkeState  (LLM-judged soft answer likelihood)
        memory -> MemoryKripkeState (per-opponent LLM judge conditioned on cross-game memory; needs `memory`)
        bayes -> BayesKripkeState (per-opponent Bayesian posterior over the answer models
            uniform / strategic3 / llm, learned across games; adaptive mixture/MAP likelihood;
            needs `memory` and an LLM client, since the "llm" candidate uses the memoryless judge)

    For LLM strategies, flags control what the LLM context:
        - give_belief_state: whether to include Kripke posterior in prompt
        - give_history: whether to include QA history in prompt
        - belief_mode: "distribution" (role probabilities) or "worlds" (top possible worlds)
        - use_reasoning_history: whether to feed back LLM's reasoning summary (like LLMAgent)

    LLM context flags examples:
        llm + history + belief(dist) -> LLM with distributions
        llm + history + belief(worlds) -> LLM with top worlds
        llm + history only -> LLM with game history, no symbolic info
        llm + belief only -> LLM with belief state, no raw history
        llm (no context) -> LLM with only game rules + action list

    Ablation grid examples:
        question=ig, guess=map -> full symbolic baseline
        question=random, guess=map -> belief-update-only baseline
        question=ig, guess=random -> IG questions selection, random guess (isolate guess value)
        question=ig, guess=llm -> IG questions selection, LLM guesses
        question=llm, guess=map -> LLM questions selection, MAP guess
        question=llm, guess=llm -> LLM questions selection, LLM guesses
    """

    def __init__(self, player_id: int, role: Role, num_players: int, use_center_card: bool, question_strategy: str = "ig", \
        guess_strategy: str = "map", likelihood: str = "uniform", give_belief_state: bool = True, give_history: bool = True, \
            belief_mode: str = "distribution", llm_client: LLMClient = None, use_reasoning_history: bool = True, \
                print_out: bool = False, seed_source: SeedSource = None, tom_beta: float = None, \
                    tom_smoothing: str = "softmax", tom_eps: float = None, memory: CrossGameMemory = None, give_memory: bool = True, \
                        judge_reasoning_effort: str = None, judge_cot: bool = False, bayes_adapt: str = "mixture"):
        super().__init__(player_id, role)
        self.question_strategy = question_strategy # "ig" | "random" | "llm"
        self.guess_strategy = guess_strategy # "map" | "random" | "llm"
        self.likelihood = likelihood # "uniform" | "strategic" | "llm" | "memory"
        self.give_belief_state = give_belief_state
        self.give_history = give_history
        self.belief_mode = belief_mode # "distribution" | "worlds"
        self.llm_client = llm_client
        self.memory = memory # Cross-game opponent memory (shared per-run store);
        self.give_memory = give_memory
        # give_memory additionally shows its raw report in the LLM question/guess prompts.
        self.use_reasoning_history = use_reasoning_history # shared with llm.reasoning_history.enabled
        self.print_out = print_out
        # Independent sub-streams so that toggling one concern's strategy
        # (e.g. question_strategy ig<->llm) does not shift the other's draws:
        #   child(0) -> random question/guess choices
        #   child(1) -> per-call LLM seeds (question/guess)
        #   child(2) -> soft-likelihood judge seeds (only when likelihood == "llm")
        src = seed_source or SeedSource(None)
        self.rng = src.spawn(0).to_random()
        self._llm_source = src.spawn(1)

        # Belief state: the uniform answer likelihood (KripkeState, shared with the
        # symbolic Prince), the 2nd/3rd-order strategic-opponent likelihoods
        # (StrategicKripkeState / ThirdOrderKripkeState), or the LLM-judged soft
        # likelihood (NeuroSymbolicKripkeState).
        if likelihood == "uniform":
            self.kripke = KripkeState(num_players, use_center_card)
        elif likelihood == "strategic":
            self.kripke = StrategicKripkeState(num_players, use_center_card, beta=tom_beta, \
                smoothing=tom_smoothing, eps=tom_eps)
        elif likelihood == "strategic3":
            self.kripke = ThirdOrderKripkeState(num_players, use_center_card, beta=tom_beta, \
                smoothing=tom_smoothing, eps=tom_eps)
        elif likelihood == "llm":
            self.kripke = NeuroSymbolicKripkeState(num_players, use_center_card, llm_client, \
                seed_source=src.spawn(2), judge_reasoning_effort=judge_reasoning_effort, \
                    judge_cot=judge_cot)
        elif likelihood == "memory":
            if memory is None:
                raise ValueError("likelihood='memory' needs a CrossGameMemory store "
                    "(enable agents.prince.memory in the config)")
            self.kripke = MemoryKripkeState(num_players, use_center_card, llm_client, memory=memory, \
                seed_source=src.spawn(2), judge_reasoning_effort=judge_reasoning_effort, \
                    judge_cot=judge_cot)
        elif likelihood == "bayes":
            if memory is None:
                raise ValueError("likelihood='bayes' needs a CrossGameMemory store (enable agents.prince.memory in the config)")
            if llm_client is None:
                raise ValueError("likelihood='bayes' needs an LLM client (the 'llm' type candidate uses the memoryless judge)")
            # The "llm" type candidate: a memoryless soft-likelihood judge, shared
            # by the in-game mixture and the post-game update_type_posteriors scorer.
            llm_state = NeuroSymbolicKripkeState(num_players, use_center_card, llm_client, \
                seed_source=src.spawn(2), judge_reasoning_effort=judge_reasoning_effort, \
                    judge_cot=judge_cot)
            self.kripke = BayesKripkeState(num_players, use_center_card, memory=memory, \
                adapt=bayes_adapt, tom_beta=tom_beta, tom_smoothing=tom_smoothing, tom_eps=tom_eps, \
                    llm_state=llm_state)
        else:
            raise ValueError(f"Unknown likelihood: '{likelihood}'")
        self.decision_log: list[dict] = []

        # Reasoning history (same structure as LLMAgent for comparability)
        self.reasoning_history: list[str] = []
        self.current_summary: str = ""
        self.summary_history: list[str] = []

    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        if not isinstance(obs, PrinceObservation):
            raise TypeError("AblationPrinceAgent expects PrinceObservation")

        self.kripke.sync_with_history(obs.qa_history)

        if obs.phase in (Phase.QUESTIONING, Phase.FINAL_QUESTION):
            return self.__select_question(obs, valid_actions)
        elif obs.phase == Phase.GUESSING:
            return self.__select_guess(obs, valid_actions)
        else:
            raise Exception("Unexpected phase: " + str(obs.phase))

    def _next_llm_seed(self) -> int | None:
        """Fresh seed for this LLM call, or None if this agent is seedless."""
        return self._llm_source.next_seed()

    def _memory_report(self) -> str:
        """Cross-game memory block for the LLM prompts ("" when absent or disabled)."""
        if self.memory is None or not self.give_memory:
            return ""
        return self.memory.full_report()


    def __select_question(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        strategy = self.question_strategy

        if strategy == "ig":
            return self.__question_by_ig(obs, valid_actions)
        elif strategy == "random":
            return self.__question_by_random(obs, valid_actions)
        elif strategy == "llm":
            return self.__question_by_llm(obs, valid_actions)
        else:
            raise ValueError(f"Unknown question_strategy: '{strategy}'")

    def __question_by_ig(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """Pick the question with highest expected information gain, using the SAME
        opponent model (answer likelihood) as the belief update -- so the IG forward
        simulation follows `likelihood`: uniform for a plain KripkeState, the LLM-judged
        soft likelihood for a NeuroSymbolicKripkeState."""
        best_action = valid_actions[0]
        best_ig = -1.0
        ig_scores = []
        soft = isinstance(self.kripke, NeuroSymbolicKripkeState)

        for action in valid_actions:
            q_type = QuestionType(action["question_type"])
            target = action["target"]
            q_param = action.get("question_param")

            ig = self.kripke.soft_information_gain(q_type, target, q_param) if soft \
                else self.kripke.information_gain(q_type, target, q_param)
            ig_scores.append((action, ig))

            if ig > best_ig:
                best_ig = ig
                best_action = action

        if self.print_out:
            print(f"\n[AblationPrince] number of Worlds: {self.kripke.num_worlds()}")
            print(f"[AblationPrince] Princess entropy: ", f"{self.kripke.princess_entropy():.4f} bits")
            ig_scores.sort(key=lambda x: -x[1])
            print("[AblationPrince] Top questions by info gain:")
            for act, ig in ig_scores[:3]:
                print(f"  IG={ig:.4f}  {action_description(act)}")

        self.decision_log.append({
            "phase": obs.phase.value, "strategy": "ig",
            "num_worlds": self.kripke.num_worlds(),
            "Princess_entropy": round(self.kripke.princess_entropy(), 4),
            "action_chosen": action_description(best_action),
            "info_gain": round(best_ig, 4),
        })

        return best_action

    def __question_by_random(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """Uniform random question selection."""
        action = self.rng.choice(valid_actions)
        self.decision_log.append({
            "phase": obs.phase.value, "strategy": "random",
            "num_worlds": self.kripke.num_worlds(),
            "action_chosen": action_description(action),
        })
        return action

    def __question_by_llm(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """LLM selects question with configurable context."""
        if self.llm_client is None:
            raise RuntimeError(f"{self.agent_name}: llm_client is not assigned!!!")

        # Build prompt via prompts.py (reuses PRINCE_SYSTEM, same formatting as LLMAgent)
        previous_summary = self.current_summary if self.use_reasoning_history else ""

        system_prompt, user_prompt = build_ablation_prince_prompt(
            obs, valid_actions,
            give_history=self.give_history,
            give_belief_state=self.give_belief_state,
            belief_mode=self.belief_mode,
            kripke_state=self.kripke,
            previous_summary=previous_summary,
            memory_report=self._memory_report(),
        )

        if self.print_out:
            print(f"\n{'~'*60}")
            print(f"[{self.agent_name}] Ablation LLM Prompt (belief={self.give_belief_state}, history={self.give_history}, ",\
                f"mode={self.belief_mode})")
            print(f"{'~'*60}")
            print("System Prompt:")
            print(system_prompt)
            print(f"\n{'-'*60}")
            print("User Prompt:")
            print(user_prompt)
            print(f"{'~'*60}")

        max_index = len(valid_actions) - 1
        call_seed = self._next_llm_seed()
        try:
            choice, reasoning = self.llm_client.chat_structured(user_prompt, system_prompt=system_prompt, \
                max_index=max_index, seed=call_seed)

            # Store reasoning history (same as LLMAgent)
            if reasoning:
                self.reasoning_history.append(reasoning)
            
            # Update iterative summary (same as LLMAgent)
            if choice.reasoning_summary:
                self.current_summary = choice.reasoning_summary
                self.summary_history.append(choice.reasoning_summary)

            if self.print_out:
                print(f"{'='*60}")
                print(f"\n[{self.agent_name}] LLM Response")
                if reasoning:
                    print(f"Thinking: {reasoning}")
                    print()
                print(f"Summary: {choice.reasoning_summary}")
                print(f"Action index: {choice.action_index}")
                stats = self.llm_client.last_call_stats
                if stats:
                    print(f"Tokens: {stats.prompt_tokens} prompt + {stats.completion_tokens} completion", \
                        f" = {stats.total_tokens} total (seed={stats.seed_used})")

            # idx = max(0, min(choice.action_index, max_index))
            idx = choice.action_index
            action = valid_actions[idx]
        except Exception as e:
            raise LLMParseError(self.agent_name, e)

        # # Compute regret: how far is LLM's choice from the IG-optimal question?
        # chosen_q = QuestionType(action["question_type"])
        # chosen_ig = self.kripke.information_gain(chosen_q, action["target"], action.get("question_param"))
        # best_ig = -1.0
        # for a in valid_actions:
        #     ig = self.kripke.information_gain(
        #         QuestionType(a["question_type"]), a["target"], a.get("question_param"))
        #     if ig > best_ig:
        #         best_ig = ig
        # ig_regret = round(best_ig - chosen_ig, 4)  # 0 = optimal, >0 = suboptimal

        self.decision_log.append({
            "phase": obs.phase.value, "strategy": "llm",
            "belief_mode": self.belief_mode,
            "give_belief": self.give_belief_state, "give_history": self.give_history,
            "num_worlds": self.kripke.num_worlds(),
            "action_chosen": action_description(action),
            # "chosen_ig": round(chosen_ig, 4),
            # "best_ig": round(best_ig, 4),
            # "ig_regret": ig_regret,
        })
        return action


    def __select_guess(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        strategy = self.guess_strategy

        if strategy == "map":
            return self.__guess_by_map(obs, valid_actions)
        elif strategy == "random":
            return self.__guess_by_random(obs, valid_actions)
        elif strategy == "llm":
            return self.__guess_by_llm(obs, valid_actions)
        else:
            raise ValueError(f"Unknown guess_strategy: '{strategy}'")

    def __guess_by_map(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """Guess using Kripke posterior MAP: pick max P(Princess=x)."""
        dist = self.kripke.princess_distribution()

        if self.print_out:
            print(f"\n[AblationPrince] Final belief (map guess):")
            print(self.kripke.get_full_report())

        best_target = max(dist, key=dist.get)

        for action in valid_actions:
            if action["target"] == best_target:
                self.decision_log.append({
                    "phase": "guessing", "question_strategy": self.question_strategy,
                    "guess_strategy": "map",
                    "worlds_remaining": self.kripke.num_worlds(),
                    "princess_dist": {str(k): round(v, 4) for k, v in dist.items()},
                    "guess": best_target,
                })
                return action

        # MAP target not found in valid actions, should not happen in normal game!!!!
        raise SymbolicGuessError(self.agent_name, best_target, [a['target'] for a in valid_actions])

    def __guess_by_random(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """Uniform random guess."""
        action = self.rng.choice(valid_actions)

        if self.print_out:
            print(f"\n[AblationPrince] Random guess: {action_description(action)}")

        self.decision_log.append({
            "phase": "guessing", "question_strategy": self.question_strategy,
            "guess_strategy": "random",
            "worlds_remaining": self.kripke.num_worlds(),
            "guess": action["target"],
        })
        return action

    def __guess_by_llm(self, obs: PrinceObservation, valid_actions: list[dict]) -> dict:
        """LLM selects guess with configurable context (same flags as question selection)."""
        if self.llm_client is None:
            raise RuntimeError(f"{self.agent_name}: llm_client is not assigned!!!")

        previous_summary = self.current_summary if self.use_reasoning_history else ""

        system_prompt, user_prompt = build_ablation_prince_prompt(obs, valid_actions, \
            give_history=self.give_history, give_belief_state=self.give_belief_state, \
                belief_mode=self.belief_mode, kripke_state=self.kripke, \
                    previous_summary=previous_summary, memory_report=self._memory_report())

        if self.print_out:
            print(f"\n{'~'*60}")
            print(f"[{self.agent_name}] Ablation LLM Prompt (belief={self.give_belief_state}, history={self.give_history}, ",\
                f"mode={self.belief_mode})")
            print(f"{'~'*60}")
            print("System Prompt:")
            print(system_prompt)
            print(f"\n{'-'*60}")
            print("User Prompt:")
            print(user_prompt)
            print(f"{'~'*60}")

        max_index = len(valid_actions) - 1
        call_seed = self._next_llm_seed()
        try:
            choice, reasoning = self.llm_client.chat_structured(user_prompt, system_prompt=system_prompt, \
                max_index=max_index, seed=call_seed)

            if reasoning:
                self.reasoning_history.append(reasoning)
            if choice.reasoning_summary:
                self.current_summary = choice.reasoning_summary
                self.summary_history.append(choice.reasoning_summary)

            if self.print_out:
                print(f"{'='*60}")
                print(f"\n[{self.agent_name}] LLM Response")
                if reasoning:
                    print(f"Thinking: {reasoning}")
                    print()
                print(f"Summary: {choice.reasoning_summary}")
                print(f"Action index: {choice.action_index}")
                stats = self.llm_client.last_call_stats
                if stats:
                    print(f"Tokens: {stats.prompt_tokens} prompt + {stats.completion_tokens} completion", \
                        f" = {stats.total_tokens} total (seed={stats.seed_used})")

            idx = choice.action_index
            # idx = max(0, min(idx, max_index)) # clamp to valid range
            action = valid_actions[idx]
        except Exception as e:
            # DO NOT fallback silently
            raise LLMParseError(self.agent_name, e)

        guessed_target = action["target"]
        # # Compute how good the LLM's guess is relative to Kripke posterior
        # dist = self.kripke.princess_distribution()
        # guessed_target = action["target"]
        # map_target = max(dist, key=dist.get)
        # guessed_prob = round(dist.get(guessed_target, 0.0), 4)  # P(Princess = LLM's guess)
        # map_prob = round(dist.get(map_target, 0.0), 4)          # P(Princess = MAP target)
        # # rank: 1 = LLM picked the most probable, 2 = second most probable, etc.
        # sorted_targets = sorted(dist, key=lambda x: -dist[x])
        # guess_rank = sorted_targets.index(guessed_target) + 1 if guessed_target in sorted_targets else -1
        # agrees_with_map = guessed_target == map_target

        self.decision_log.append({
            "phase": "guessing", "question_strategy": self.question_strategy,
            "guess_strategy": "llm",
            "belief_mode": self.belief_mode,
            "give_belief": self.give_belief_state, "give_history": self.give_history,
            "worlds_remaining": self.kripke.num_worlds(),
            "guess": guessed_target,
            # "guessed_prob": guessed_prob,
            # "map_target": map_target,
            # "map_prob": map_prob,
            # "guess_rank": guess_rank,
            # "agrees_with_map": agrees_with_map,
        })
        return action
