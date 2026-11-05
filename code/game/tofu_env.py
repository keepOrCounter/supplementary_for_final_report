#game Environment
#game loop for tofu kingdom
import random
from copy import deepcopy

from game.predefined_data_classes import (
    Phase, Faction, Constraint, QuestionType,
    Role, ROLES, ROLE_CONFIGS, ROLE_CONFIGS_NO_CENTER,
    QAPair, GameState, GameResult, 
    PrinceObservation, FullyObservation, Observation
)

# Prince is always player 0
PRINCE_ID = 0

class TofuKingdomEnv:
    """
    Tofu Kingdom game environment.

    Game flow:
        1. reset(seed) to initialise game
        2. QUESTIONING: Prince asks each non-prince player one question,
            and each questioned player answers immediately after.
        3. FINAL_QUESTION: Prince asks one more question to any non-prince player,
            that player answers immediately after.
        4. GUESSING: Prince guesses who is Princess Tofu.
        5. get_result() to finish game
    """

    def __init__(self, num_players: int = 5, use_center_card: bool = True):
        configs = ROLE_CONFIGS if use_center_card else ROLE_CONFIGS_NO_CENTER
        if num_players not in configs:
            raise Exception(f"number of players must be between 3-8 !!!!")
        
        self.num_players = num_players
        self.use_center_card = use_center_card
        self.__state: GameState = None

        # Turn tracking
        self.waiting_for: str = "question" # "question" | "answer" | "guess" TODO remove

    def reset(self, seed: int = None) -> GameState:
        """initialise a new game
        
            shuffle roles, assign to players (and center card if enabled)
        """
        ran_seed = random.Random(seed)

        # get role names for current player numbers
        configs = ROLE_CONFIGS if self.use_center_card else ROLE_CONFIGS_NO_CENTER
        role_names: list[str] = list(configs[self.num_players])
        ran_seed.shuffle(role_names)

        # assign agents with roles
        non_prince_ids = list(range(1, self.num_players)) # 1 - N-1
        roles_assigned: dict[int, Role] = {}
        for i, pid in enumerate(non_prince_ids):
            roles_assigned[pid] = ROLES[role_names[i]]
        
        center_card = None
        if self.use_center_card and self.num_players != 8:
            center_card = ROLES[role_names[-1]]  # last extra one goes to center card

        self.__state = GameState(
            done=False,
            result=None,
            current_actor=PRINCE_ID,
            questioned_players = [],
            roles=roles_assigned,
            phase=Phase.QUESTIONING,
            qa_history=[],
            current_question=None,
            center_card=center_card,
            num_players=self.num_players,
            current_target=None,
            round=0,
        )

        # Reset turn tracking TODO remove
        self.waiting_for = "question"

        return deepcopy(self.__state)


    def get_observation(self, player_id: int) -> Observation:
        self.is_game_start() # is game start?

        if player_id == PRINCE_ID:
            return self.get_prince_observation()
        else:
            return self.get_fully_observation(player_id)

    def get_prince_observation(self) -> PrinceObservation:

        if self.__state.phase == Phase.QUESTIONING:
            remaining_players = [p for p in self.__state.roles.keys() \
                if p not in self.__state.questioned_players]
        elif self.__state.phase == Phase.FINAL_QUESTION or self.__state.phase == Phase.GUESSING:
            remaining_players = list(self.__state.roles.keys())
        else:
            remaining_players = []

        available_roles = sorted(self.__state.roles.values(), key=lambda r: r.name)
        # All roles exist in the game
        if self.__state.center_card is not None:
            available_roles.append(self.__state.center_card)
            available_roles.sort(key=lambda r: r.name)

        return deepcopy(PrinceObservation(
            phase=self.__state.phase,
            qa_history=self.__state.qa_history,
            remaining_players=remaining_players,
            num_players=self.__state.num_players,
            available_roles=available_roles,
            has_center_card=self.__state.center_card is not None,
        ))

    def get_fully_observation(self, player_id: int) -> FullyObservation:
        current_qa = self.__state.current_question  # The question just asked to a player

        return deepcopy(FullyObservation( # TODO add current target
            player_id=player_id,
            player_role=self.__state.roles[player_id],
            answer_constraint=self.__state.roles[player_id].constraint,
            all_roles=self.__state.roles,
            center_card=self.__state.center_card,
            current_question=current_qa,
            qa_history=self.__state.qa_history,  
            phase=self.__state.phase,
        ))

    def get_valid_actions(self, player_id: int) -> list[dict]:
        """return all legal actions in dict for the given player in current state"""
        self.is_game_start()

        if player_id == PRINCE_ID:
            if self.waiting_for == "question": #  TODO remove
                return self.get_question_actions()
            elif self.waiting_for == "guess": #  TODO remove
                return self.get_guess_actions()
            else:
                return []
        else:
            if self.waiting_for == "answer": #  TODO remove
                return self.get_answer_actions(player_id)
            else:
                return []

    def get_question_actions(self) -> list[dict]:
        """all possible questions the prince can ask"""
        actions = []
        

        if self.__state.phase == Phase.QUESTIONING:
            # players have not been asked
            targets = [p for p in self.__state.roles.keys() if p not in self.__state.questioned_players]
        elif self.__state.phase == Phase.FINAL_QUESTION:
            targets = list(self.__state.roles.keys())
        else:
            return []

        for target in targets:
            # Q1: Where is Princess Tofu?
            actions.append({
                "type": "question", # TODO add action class
                "target": target,
                "question_type": QuestionType.Q1.value,
                "question_param": None,
            })
            # Q2: Who are you?
            actions.append({
                "type": "question",
                "target": target,
                "question_type": QuestionType.Q2.value,
                "question_param": None,
            })
            # Q3: Who is Player X? (for each other non-prince player, excluding the target)
            for other in range(1, self.__state.num_players):
                if other != target:
                    actions.append({
                        "type": "question",
                        "target": target,
                        "question_type": QuestionType.Q3.value,
                        "question_param": other,
                    })
                    
            if self.__state.center_card is not None: # Q3 target to center card
                actions.append({
                    "type": "question",
                    "target": target,
                    "question_type": QuestionType.Q3.value,
                    "question_param": -1,
                })
            

        return actions

    def get_guess_actions(self) -> list[dict]:
        result = [{"type": "guess", "target": pid}for pid in list(self.__state.roles.keys())]
        if self.__state.center_card is not None:
            result.append({"type": "guess", "target": -1}) # -1 for center card
        return result

    def get_answer_actions(self, player_id: int) -> list[dict]:
        """all valid answers for a player with their constraint"""
        player_role = self.__state.roles[player_id]
        current_qa = self.__state.current_question
        q_type = current_qa.question_type

        truth, all_answers = self.__get_answers(player_id, q_type, current_qa)

        # apply constraint filter
        match player_role.constraint:
            case Constraint.MUST_TRUTH:
                valid_answers = [truth]
            case Constraint.MUST_LIE:
                valid_answers = [a for a in all_answers if a != truth]
            case Constraint.NO_CONSTRAINT:
                valid_answers = all_answers

        return [{"type": "answer", "target": player_id, "answer": ans}for ans in valid_answers]

    def __get_answers(self, player_id: int, q_type: QuestionType, qa: QAPair) -> tuple[str, list[str]]:
        """
        Returns:
            truth: The honest answer.
            all_answers: All possible answers the player could reply without constraint.
        """

        configs = ROLE_CONFIGS if self.use_center_card else ROLE_CONFIGS_NO_CENTER
        all_role_names = deepcopy(configs[self.num_players])

        if q_type == QuestionType.Q1:
            # "Where is Princess Tofu?" Answer: reply a player ID or "center"
            princess_pid = self.__find_princess()
            if princess_pid != -1:
                truth = f"Player {princess_pid}"
            else:
                truth = "center"

            all_answers = [f"Player {pid}" for pid in list(self.__state.roles.keys())]
            if self.__state.center_card is not None:
                all_answers += ["center"]
            return truth, all_answers

        elif q_type == QuestionType.Q2:
            # "Who are you?" Answer: reply a role name
            truth = self.__state.roles[player_id].name
            return truth, all_role_names

        elif q_type == QuestionType.Q3:
            # "Who is Player X?" Answer: reply Player X's role name
            if qa.question_param == -1:
                truth = self.__state.center_card.name
            elif qa.question_param in self.__state.roles:
                truth = self.__state.roles[qa.question_param].name
            else:
                truth = "unknown"
            return truth, all_role_names


    def step(self, player_id: int, action: dict) -> None:
        """process an action. Validates, updates state, proceed to next round

        Args:
            player_id: player who take action
            action: dict (from get_valid_actions)
        """
        self.is_game_start()

        if player_id != self.__state.current_actor:
            raise Exception(f"Not Player {player_id}'s turn. \
                Expected Player {self.__state.current_actor}.")

        action_type = action["type"]

        if action_type == "question":
            self.__step_question(action)
        elif action_type == "answer":
            self.__step_answer(action)
        elif action_type == "guess":
            self.__step_guess(action)
        else:
            raise ValueError(f"Invalid action type: {action_type}!!!!!!!!")

    def __step_question(self, action: dict) -> None:
        target = action["target"]
        q_type_str = action["question_type"]
        q_param = action.get("question_param")

        if target == PRINCE_ID or target >= self.__state.num_players or target < 1:
            raise ValueError(f"Invalid question target: {target}")

        qa = QAPair(
            round=self.__state.round,
            # questioner_id=PRINCE_ID,
            target_id=target,
            question_type=QuestionType(q_type_str),
            question_param=q_param,
            answer=None,
        )
        self.__state.current_question = qa
        self.__state.current_target = target

        # wait for the target player's answer
        self.__state.current_actor = target
        self.waiting_for = "answer"

    def __step_answer(self, action: dict) -> None:
        """updates current QAPair, proceed to next round"""
        answer_text = action["answer"]

        # fill answer into current QA record
        current_qa = self.__state.current_question
        current_qa.answer = answer_text

        self.__state.current_target = None
        
        self.__state.qa_history.append(current_qa)

        # update game phase
        if self.__state.phase == Phase.QUESTIONING:
            self.__state.questioned_players.append(current_qa.target_id)
            if len(self.__state.questioned_players) >= self.num_players - 1:
                # all players questioned
                self.__state.phase = Phase.FINAL_QUESTION
                self.__state.round += 1
            else:
                self.__state.round += 1

            self.__state.current_actor = PRINCE_ID
            self.waiting_for = "question" # TODO remove

        elif self.__state.phase == Phase.FINAL_QUESTION:
            # Final question answered
            self.__state.phase = Phase.GUESSING
            self.__state.current_actor = PRINCE_ID
            self.waiting_for = "guess" # TODO remove

    def __step_guess(self, action: dict) -> None:
        """this method will determines winner, game will ends"""
        guessed_pid = action["target"]

        if guessed_pid == PRINCE_ID or guessed_pid >= self.__state.num_players or \
            (guessed_pid == -1 and self.__state.center_card is None) or guessed_pid < -1:
                
            raise ValueError(f"Invalid guess target: {guessed_pid}")

        princess_pid = self.__find_princess()
        queen_pid = self.__find_queen()
        # guessed_role = self.__state.roles[guessed_pid]

        # determine outcome
        if guessed_pid == princess_pid:
            winner = Faction.PRINCESS
        elif guessed_pid == queen_pid:
            winner = Faction.QUEEN
        else:
            winner = Faction.OTHER

        # winners get 1 mark
        scores = {}
        for pid, role in self.__state.roles.items():
            if role.faction == winner:
                scores[pid] = 1
            else:
                scores[pid] = 0
        # prince scores same as PRINCESS faction
        if winner == Faction.PRINCESS:
            scores[PRINCE_ID] = 1
        else:
            scores[PRINCE_ID] = 0

        self.__state.done = True
        self.__state.result = GameResult(
            guessed_target=guessed_pid,
            actual_princess=princess_pid,
            actual_queen=queen_pid,
            all_roles=deepcopy(self.__state.roles),
            scores=scores,
            winner_faction=winner,
        )


    def get_result(self) -> GameResult:
        """only valid after game finish"""
        if not self.__state.done or self.__state.result is None:
            raise Exception("Game is not finished!")
        return deepcopy(self.__state.result)

    @property
    def done(self) -> bool:
        return self.__state.done
    
    @property
    def state(self) -> GameState:
        return deepcopy(self.__state)


    def __find_princess(self) -> int:
        """find which player_id is princess tofu. -1 if in center."""
        for pid, role in self.__state.roles.items():
            if role.name == "Princess Tofu":
                return pid
        return -1
    
    def __find_queen(self) -> int:
        """find which player_id is queen tofu. -1 if in center."""
        for pid, role in self.__state.roles.items():
            if role.name == "Queen Tofu":
                return pid
        return -1

    def is_game_start(self) -> None:
        if self.__state is None:
            raise Exception("Game not started. Call reset() first.")
        if self.__state.done:
            raise Exception("Game is already over. Call reset() to restart.")