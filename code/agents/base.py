"""Base agent interface, the simple built-in agents (random / human) and the
shared agent exceptions. Method-specific agents live in their own modules:
`llm_agent`, `our_agents/`, `grail/`, `emo/` (all re-exported by `agents/__init__.py`).
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from game.predefined_data_classes import Observation, PrinceObservation, FullyObservation, Role, Phase, \
    QuestionType
from core.helper_functions import action_description, SeedSource


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, player_id: int, role: Role):
        self.player_id = player_id
        self.role = role
        self.agent_name = self.__class__.__name__ + "(Pid"+str(player_id)+")"

    @abstractmethod
    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        """Choose an action from valid_actions based on observation.

        Args:
            valid_actions: list of legal actions in dict

        Returns:
            action in dict from valid_actions.
        """
        pass

    def __str__(self) -> str:
        return self.agent_name


class RandomAgent(BaseAgent):
    """choose random valid action"""

    def __init__(self, player_id: int, role: Role, seed_source: SeedSource = None):
        super().__init__(player_id, role)
        # One independent stream per agent: this agent's choices are unaffected
        # by how many seeds any other agent (or the LLM Prince) draws.
        self.ran_seed = (seed_source or SeedSource(None)).to_random()

    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        return self.ran_seed.choice(valid_actions)


class HumanAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role):
        super().__init__(player_id, role)
        
    def take_action(self, obs: Observation, valid_actions: list[dict]) -> dict:
        # Print observation
        print("-" * 50)
        print(f"It's {self.agent_name}'s turn")
        print()
        self.__print_observation(obs)

        # Print actions
        print("Valid actions:")
        for i, action in enumerate(valid_actions):
            print(f"{i}. {action_description(action)}")

        # Get input
        while True:
            try:
                choice = int(input(f"\nChoose action (input a number 0-{len(valid_actions)-1})>>>"))
                if 0 <= choice < len(valid_actions):
                    return valid_actions[choice]
                
                print("Invalid choice!!!")
                
            except (ValueError):
                print("Please enter a number.")

    def __print_observation(self, obs: Observation) -> None:
        if isinstance(obs, PrinceObservation):
            match obs.phase:
                case Phase.QUESTIONING:
                    print("You are the Prince. You can ask a question to a player.")
                case Phase.FINAL_QUESTION:
                    print("You are the Prince. You can ask a final question to any player.")
                case Phase.GUESSING:
                    print("You are the Prince. You can make a guess about who is the Princess.")
            
            print(f"Current Phase: {obs.phase.value}")
            print(f"Player you can choose: {obs.remaining_players}")
            print()
            print("Roles in this game:")
            for role in obs.available_roles:
                print(f"  - {role.name} (faction: {role.faction.value}, constraint: {role.constraint.value})")
            if obs.has_center_card:
                print(f"  ({len(obs.available_roles)} roles for {obs.num_players - 1} players, 1 is center card)")
            print()
            print("QA History:")
            if obs.qa_history:
                for qa in obs.qa_history:
                    if qa.question_type==QuestionType.Q3:
                        print(f"Round {qa.round}: Prince ask P{qa.target_id} {qa.question_type.value}", \
                                f"{f' about P{qa.question_param}' if qa.question_param and qa.question_param != -1 \
                                    else 'about center'}", f" --> {qa.answer}")
                    else:
                        print(f"Round {qa.round}: Prince ask P{qa.target_id} {qa.question_type.value}", \
                            f" --> {qa.answer}")
            print()
        elif isinstance(obs, FullyObservation):
            print(f"You are: {obs.player_role.name} ({obs.answer_constraint.value})")
            print(f"Current Phase: {obs.phase.value}")
            print(f"All roles: { {pid: r.name for pid, r in obs.all_roles.items()} }")
            if obs.center_card:
                print(f"Center card: {obs.center_card.name}")
            print()
            current_question = obs.current_question
            print(current_question)
            if current_question.question_type==QuestionType.Q3:
                print(f"Current question: {current_question.question_type.value}", \
                        f"{f' about P{current_question.question_param}' if current_question.question_param \
                            and current_question.question_param != -1 else 'about center'}")
            else:
                print(f"Current question: {current_question.question_type.value}")
            print()



class LLMParseError(Exception):
    """raised Exception when LLM output cannot be parsed after all retries"""
    def __init__(self, agent_name: str, original_error: Exception):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"{agent_name}: LLM output parse failed after all retries ({original_error})")


class BeliefUpdateError(Exception):
    """raised when an observed answer has zero likelihood in EVERY possible world"""
    def __init__(self, answer: str, target_id: int, q_type: str, round_index: int):
        self.answer = answer
        self.target_id = target_id
        self.q_type = q_type
        self.round_index = round_index
        super().__init__(
            f"Bayesian update: total likelihood = 0 for answer {answer!r} "
            f"(round {round_index}, target P{target_id}, {q_type}). "
            f"No possible world can produce this answer, so the belief cannot be "
            f"updated. There maybe a bug in the environment or Kripke state!!!!"
        )


class SymbolicGuessError(Exception):
    """raised when symbolic agent's MAP guess target is not in valid actions"""
    def __init__(self, agent_name: str, best_target: int, valid_targets: list[int]):
        self.agent_name = agent_name
        self.best_target = best_target
        self.valid_targets = valid_targets
        super().__init__(
            f"{agent_name}: MAP target {best_target} not in valid actions "
            f"(valid targets: {valid_targets}). "
            f"There maybe a bug in the environment or Kripke state!!!!"
        )
