# Prompt Templates shared by every LLM-driven agent: game rules / system prompts,
# the generic action prompts and reusable prompt blocks. Method-specific prompts
# live in each method's own prompts.py (agents/our_agents, agents/grail, agents/emo).
from __future__ import annotations

from game.predefined_data_classes import PrinceObservation, FullyObservation, Observation, Phase, QuestionType, \
    QAPair, Constraint

from core.helper_functions import action_description



GAME_FACTIONS = """\
Roles and factions:
- PRINCESS faction (Prince Mochi, Princess Tofu, Tofu Chef): MUST tell the truth. Wins if Prince finds Princess Tofu.
- QUEEN faction (Queen Tofu, Tofu Minister, Tofu Guard): MUST lie. Wins if Prince picks Queen Tofu.
- OTHER faction (Tofu Maid, Pudding Spy): Can say anything (truth or lie). Wins if Prince picks neither Princess nor Queen."""


GAME_RULES = f"""\
You are playing Tofu Kingdom, a social deduction game.

{GAME_FACTIONS}

Game flow:
1. Roles are secretly assigned. One extra role (if exist) is placed face-down as the center card.
2. Prince Mochi does NOT know anyone's role. All other players can see everyone's role.
3. QUESTIONING: Prince asks each player one question. The player answers immediately.
4. FINAL QUESTION: Prince asks one more question to any player.
5. GUESSING: Prince guesses which player (or guesses center card if exist) is Princess Tofu given with the previous answers.

Question types:
- Q1 "Where is Princess Tofu?" -> answer with a player ID or "center"
- Q2 "Who are you?" -> answer with a role name
- Q3 "Who is Player X?" -> answer with Player X's role name
"""

PRINCE_SYSTEM = GAME_RULES + """
You are Prince Mochi (Player 0). Your goal is to find Princess Tofu.

Key deduction principles:
- If two players' answers are consistent, they are more likely to be from the same faction.
- If a player's answer contradicts others, consider whether they MUST lie (Queen faction), MUST tell truth (Princess faction), or can say anything (Other faction).
- Princess faction members will actively help you find Princess Tofu with truthful answers.
- Queen faction members will try to mislead you to choose Queen by always lying.
- Other faction members are the hardest to read: they will prevent you from choosing the either Princess or Queen player and may mix truth and lies to create confusion.
"""

PLAYER_SYSTEM = GAME_RULES + """
You are a non-Prince player. You can see all roles. You must answer within your constraint to help your faction win.
"""


def build_prompt(obs: Observation, valid_actions: list[dict], previous_summary: str = "") -> tuple[str, str]:
    """Build system_prompt+user_prompt from observation and valid actions

    Args:
        previous_summary: single iterative summary string from previous turns

    Returns:
        system_prompt: Game rules+role-specific instructions
        user_prompt: Current game state + action choices
    """
    if isinstance(obs, PrinceObservation):
        return PRINCE_SYSTEM, build_prince_prompt(obs, valid_actions, previous_summary)
    elif isinstance(obs, FullyObservation):
        return PLAYER_SYSTEM, build_non_prince_prompt(obs, valid_actions, previous_summary)
    else:
        raise ValueError(f"Unknown observation type: {type(obs)}")


def build_prince_prompt(obs: PrinceObservation, valid_actions: list[dict], previous_summary: str = "") -> str:
    lines = []

    lines.extend(prince_basic_game_info(obs))
    lines.extend(qa_history(obs))

    # Include iterative reasoning summary from previous turns
    if previous_summary:
        lines.append("")
        lines.append("Your reasoning summary so far:")
        lines.append(previous_summary)

    # phase-specific instruction
    lines.append("")
    if obs.phase == Phase.QUESTIONING:
        lines.append("Choose which question to ask next.")
    elif obs.phase == Phase.FINAL_QUESTION:
        lines.append("You can ask ONE final question to any player. Choose wisely.")
    elif obs.phase == Phase.GUESSING:
        lines.append("Based on the Question & Answer history, guess which player is Princess Tofu.")

    lines.extend(action_choices(valid_actions))

    return "\n".join(lines)


def build_non_prince_prompt(obs: FullyObservation, valid_actions: list[dict], previous_summary: str = "") -> str:
    lines = []

    lines.append(f"You are Player {obs.player_id}.")
    lines.append(f"Your role: {obs.player_role.name} (faction: {obs.player_role.faction.value})")
    lines.append(f"Your constraint: {constraint_description(obs.answer_constraint)}")

    # Visible roles
    lines.append("")
    lines.append(" All Roles (visible to you):")
    for pid, role in sorted(obs.all_roles.items()):
        label = " (you)" if pid == obs.player_id else ""
        lines.append(f" Player {pid}: {role.name}{label}")
    if obs.center_card:
        lines.append(f" Center card: {obs.center_card.name}")

    # QA history
    # if obs.qa_history:
    #     lines.append("")
    #     lines.append(" Question & Answer History:")
    #     for qa in obs.qa_history:
    #         lines.append(format_qa_pair(qa))
    lines.extend(qa_history(obs))

    # Include iterative reasoning summary from previous turns
    if previous_summary:
        lines.append("")
        lines.append("Your reasoning summary from your previous answer:")
        lines.append(previous_summary)

    # Current question
    lines.append("")
    q = obs.current_question
    if q.question_type==QuestionType.Q3:
        lines.append(f"Prince is asking you: {q.question_type.value}"+\
                f"{f' about Player {q.question_param}' if q.question_param and q.question_param != -1 \
                    else 'about center'}")
    else:
        lines.append(f"Prince is asking you: {q.question_type.value}")

    # Strategy hint based on faction
    lines.append("")
    lines.append(strategy_hint(obs))

    # Actions
    # lines.append("")
    # lines.append(" Your choices:")
    # for i, action in enumerate(valid_actions):
    #     lines.append(f"{i}. {action_description(action)}")

    # lines.append("")
    # lines.append("Think step by step about which choice is best. (but keep it short, 3-6 sentences of reasoning is enough)")
    lines.extend(action_choices(valid_actions))

    return "\n".join(lines)



def format_qa_pair(qa: QAPair) -> str:
    """Format a QA record for the prompt"""
    q_desc = f"Prince asked Player {qa.target_id}: \"{qa.question_type.value}\""
    
    if qa.question_type == QuestionType.Q3:
        if qa.question_param == -1:
            q_desc += " about center card"
        else:
            q_desc += f" about Player {qa.question_param}"
    return f"Round {qa.round}: {q_desc} --> The Player answered: {qa.answer}"


def constraint_description(constraint: Constraint) -> str:
    match constraint:
        case Constraint.MUST_TRUTH:
            return "MUST tell the truth."
        case Constraint.MUST_LIE:
            return "MUST lie."
        case Constraint.NO_CONSTRAINT:
            return "Can say anything (truth or lie)."


def strategy_hint(obs: FullyObservation) -> str:
    """Strategy guidance based on faction, emphasizing how to help/hinder the Prince."""
    match obs.player_role.faction.value:
        case "princess":
            return ("Strategy: You are on the Prince's side. Your goal is to help the Prince to find Princess Tofu. "
                    "You MUST tell the truth — use this to give the Prince clear, helpful information. "
                    "The Prince's victory is your victory.")
        case "queen":
            return ("Strategy: You are AGAINST the Prince. Your goal is to trick the Prince into choosing Queen Tofu. "
                    "You MUST lie — use this to mislead the Prince. ")
        case "other":
            return ("Strategy: You want NEITHER Princess Tofu NOR Queen Tofu to be chosen. "
                    "You can say anything — truth or lies, whatever creates the most confusion. "
                    "Try to make the Prince doubt everyone. Point him toward a player (or yourself) who is neither Princess nor Queen. ")


def prince_basic_game_info(obs: PrinceObservation) -> list[str]:
    """Basic game info block: player count, center card, phase, roles list.

    This is public knowledge available to all prince agents.
    """
    lines = []
    lines.append(f"There are {obs.num_players} players (Player 0 is you, the Prince).")
    lines.append(f"Players 1-{obs.num_players - 1} each have a hidden role.")
    if obs.has_center_card:
        lines.append("There is also 1 center card (face-down, no one holds it).")
    else:
        lines.append("There is no center card in this game. All roles are held by the players.")
    lines.append(f"Current phase: {str(obs.phase.value).upper()}")

    lines.append("")
    lines.append("Roles in this game:")
    for role in obs.available_roles:
        lines.append(f"  - {role.name} (faction: {role.faction.value}, {constraint_description(role.constraint)})")
    center_note = " (1 is the center card)" if obs.has_center_card else ""
    lines.append(f"Total: {len(obs.available_roles)} roles for {obs.num_players - 1} players{center_note}.")
    
    return lines


def qa_history(obs: Observation) -> list[str]:
    """QA history block. Works with both PrinceObservation and FullyObservation."""
    lines = []
    if obs.qa_history:
        lines.append("")
        lines.append("Question & Answer History:")
        for qa in obs.qa_history:
            lines.append(format_qa_pair(qa))
    else:
        lines.append("")
        lines.append("No questions have been finished yet.")
    return lines


def action_choices(valid_actions: list[dict]) -> list[str]:
    """Action choices block + reasoning instruction."""
    lines = []
    lines.append("")
    lines.append(" Your choices:")
    for i, action in enumerate(valid_actions):
        lines.append(f"{i}. {action_description(action)}")

    lines.append("")
    lines.append("Think step by step about which choice is best. (but keep it short, 3-6 sentences of reasoning is enough)")
    
    return lines
