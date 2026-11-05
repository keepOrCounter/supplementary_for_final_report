from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from abc import ABC



class Phase(Enum):
    """game phases"""
    QUESTIONING = "questioning" # prince asks each player one question
    FINAL_QUESTION = "final_question" # prince asks one last question to any player
    GUESSING = "guessing" # player prince guesses who is Princess Tofu


class Faction(Enum):
    """player factions"""
    PRINCESS = "princess"
    QUEEN = "queen"
    OTHER = "other" # Wins if Prince picks neither Princess nor Queen


class Constraint(Enum):
    """Answer constraint for each role."""
    MUST_TRUTH = "must_truth"
    MUST_LIE = "must_lie"
    NO_CONSTRAINT = "no_constraint"


class QuestionType(Enum):
    Q1 = "Where is Princess Tofu?"
    Q2 = "Who are you?"
    Q3 = "Who is Player X"
    
    
@dataclass(frozen=True)
class Role:
    """Role for each player"""
    name: str
    faction: Faction
    constraint: Constraint

    def __str__(self) -> str:
        return self.name



@dataclass
class QAPair:
    """single question-answer record in game history."""
    round: int # which round (start from 0)
    # questioner_id: int
    target_id: int # player being questioned
    question_type: QuestionType
    question_param: int # For Q3: which player is being asked about, -1 if center
    answer: str  = None # target's answer (None until answered)


@dataclass
class GameState:
    """complete game state. maintained by environment"""
    done: bool
    result: GameResult
    current_actor: int # Who should act next, player id
    questioned_players: list[int] # Players already questioned in QUESTIONING stage
    
    roles: dict[int, Role] # player_id to Role (start from Prince is 0)
    phase: Phase
    qa_history: list[QAPair]
    current_question: QAPair
    num_players: int

    round: int = 0 # number questions asked
    current_target: int= None # Who is currently being questioned
    center_card: Role = None # role placed in the center without player


@dataclass
class GameResult:
    """Final result of a game"""
    guessed_target: int
    actual_princess: int
    actual_queen: int
    all_roles: dict[int, Role] # roles reveal, id to Role
    scores: dict[int, int] # player_id to score
    winner_faction: Faction
    

class Observation(ABC): # ABC Abstract Base Class
    pass



@dataclass
class PrinceObservation(Observation):
    """What the prince can see. prince does not know anyone's role, but knows which roles exist in the game
    """
    phase: Phase
    qa_history: list[QAPair]
    remaining_players: list[int] # player IDs the prince can still question/guess
    num_players: int
    available_roles: list[Role] # all roles in the game (assigned + center), order shuffled
    has_center_card: bool # whether a center card exists


@dataclass
class FullyObservation(Observation): # TODO add current target
    """what a non-prince player can see when asked a question.

    mon-prince players have full information
    """
    player_id: int
    player_role: Role
    answer_constraint: Constraint
    all_roles: dict[int, Role]
    center_card: Role
    current_question: QAPair # question being asked to this player
    qa_history: list[QAPair]
    phase: Phase
    

# Pre-defined roles
ROLES = {
    "Princess Tofu": Role("Princess Tofu", Faction.PRINCESS, Constraint.MUST_TRUTH),
    "Tofu Chef": Role("Tofu Chef", Faction.PRINCESS, Constraint.MUST_TRUTH),
    "Queen Tofu": Role("Queen Tofu", Faction.QUEEN, Constraint.MUST_LIE),
    "Tofu Minister": Role("Tofu Minister", Faction.QUEEN, Constraint.MUST_LIE),
    "Tofu Guard": Role("Tofu Guard", Faction.QUEEN, Constraint.MUST_LIE),
    "Tofu Maid": Role("Tofu Maid", Faction.OTHER, Constraint.NO_CONSTRAINT),
    "Pudding Spy": Role("Pudding Spy", Faction.OTHER, Constraint.NO_CONSTRAINT),
}


# Prince + these roles = total player
# one extra role goes to center card
ROLE_CONFIGS = {
    3: ["Princess Tofu", "Queen Tofu", "Tofu Maid"],
    4: ["Princess Tofu", "Queen Tofu", "Tofu Guard", "Tofu Maid"],
    5: ["Princess Tofu", "Tofu Chef", "Queen Tofu", "Tofu Guard", "Tofu Maid"],
    6: ["Princess Tofu", "Tofu Chef", "Queen Tofu", "Tofu Minister", "Tofu Guard", "Tofu Maid"],
    7: ["Princess Tofu", "Tofu Chef", "Queen Tofu", "Tofu Minister", "Tofu Guard", "Tofu Maid", "Pudding Spy"],
    8: ["Princess Tofu", "Tofu Chef", "Queen Tofu", "Tofu Minister", "Tofu Guard", "Tofu Maid", "Pudding Spy"],
}

# No center card: exactly (num_players - 1) roles, all assigned to non-prince players
ROLE_CONFIGS_NO_CENTER = {
    3: ["Princess Tofu", "Queen Tofu"],
    4: ["Princess Tofu", "Queen Tofu", "Tofu Maid"],
    5: ["Princess Tofu", "Queen Tofu", "Tofu Guard", "Tofu Maid"],
    6: ["Princess Tofu", "Tofu Chef", "Queen Tofu", "Tofu Guard", "Tofu Maid"],
    7: ["Princess Tofu", "Tofu Chef", "Queen Tofu", "Tofu Minister", "Tofu Guard", "Tofu Maid"],
    8: ["Princess Tofu", "Tofu Chef", "Queen Tofu", "Tofu Minister", "Tofu Guard", "Tofu Maid", "Pudding Spy"],
}