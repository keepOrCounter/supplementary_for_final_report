# Prompt templates for our agents: the soft-likelihood judge (SoftSymbolicPrinceAgent /
# NeuroSymbolicKripkeState) and the ablation Prince. Shared blocks come from agents/prompts.py.
from __future__ import annotations

from typing import TYPE_CHECKING

from game.predefined_data_classes import PrinceObservation, Phase
from agents.prompts import GAME_FACTIONS, PRINCE_SYSTEM, prince_basic_game_info, qa_history, \
    action_choices
from agents.our_agents.Kripke import KripkeState

if TYPE_CHECKING:  # type-only; JudgeQuery (the judge prompt's input) is defined in Kripke
    from agents.our_agents.Kripke import JudgeQuery, MemoryJudgeQuery


JUDGE_SYSTEM = "You are estimating player strategy in Tofu Kingdom, a social deduction game.\n\n" \
    + GAME_FACTIONS + """

You are NOT a player. Estimate how a strategic player would answer, so a deduction
engine can weight the possible answers. A player's faction fixes HOW they may answer:
Princess faction always tells the truth, Queen faction always lies, Other faction may
do either.

Every possible answer falls into one OUTCOME SLOT, named by the role the answer points
at or claims (below, "slot" always means one of these):
- PRINCESS -> the answer points at / claims Princess Tofu
- QUEEN -> the answer points at / claims Queen Tofu
- OTHER -> the answer points at / claims any other role

What a slot does to the Prince depends on the question:
- "Where is Princess Tofu?": if the Prince believes the answer, he GUESSES that spot, 
whatever role is really there.
- "Who are you?" / "Who is player X?": if the Prince believes the answer, claiming a
player is the PRINCESS makes him guess that player, while claiming they are the QUEEN
or any other role makes him AVOID that player.

A strategic answerer prefers whichever slot best serves their faction's win condition
(Queen wins ONLY if the Prince guesses Queen Tofu; Princess ONLY if Princess Tofu).

YOUR TASK: the next message gives one situation: the answerer's faction, the question,
and the slots that are still allowed (the hard rule "a liar never tells the truth" has
already removed forbidden ones). Rate EACH listed slot for how much this answerer would
want to give an answer of that slot:
    LIKELY = clearly helps answerer's faction win
    NEUTRAL = neither clearly helps nor hurts
    UNLIKELY = clearly works against answerer's faction
Give exactly one rating per listed slot (use only those three labels), then an overall
confidence (LOW / MEDIUM / HIGH).

Example
    Situation: answerer is QUEEN faction (must lie); question "Where is Princess Tofu?";
        allowed slots: OTHER, QUEEN.
    Reasoning: the Queen faction wins only if the Prince guesses Queen Tofu. Pointing at
        the QUEEN slot makes him guess Queen Tofu (a win); the OTHER slot just
        sends him to a harmless player.
    Answer: QUEEN = LIKELY, OTHER = NEUTRAL, confidence = MEDIUM
"""

# Just the question text; the slots, faction goals and slot meanings are all defined
# in JUDGE_SYSTEM, so the per-query prompt stays lean and only states THIS situation.
_JUDGE_QUESTION = {
    "Q1": '"Where is Princess Tofu?"',
    "Q2": '"Who are you?"',
    "Q3": '"Who is some player X?"',
}


def _judge_situation_lines(query: JudgeQuery) -> list[str]:
    """The 'situation to rate' block shared by the plain and memory judge prompts."""
    return [
        f"- Answerer: {query.answerer_faction} faction ({query.constraint})",
        f"- Question: {_JUDGE_QUESTION.get(query.q_type)}",
        f"- The truthful answer is in the {query.truth_label} slot.",
        f"- Allowed slots: {', '.join(query.slots)}",
        "Rate each allowed slot (LIKELY / NEUTRAL / UNLIKELY), then give one overall confidence.",
    ]


def build_judge_prompt(query: JudgeQuery) -> tuple[str, str]:
    """Render one JudgeQuery as a lean 'situation to rate' (mirrors the SYSTEM example)."""
    return JUDGE_SYSTEM, "\n".join(["Situation to rate:"] + _judge_situation_lines(query))


MEMORY_JUDGE_SYSTEM = JUDGE_SYSTEM + """
CROSS-GAME MEMORY: here you rate ONE CONCRETE PLAYER, and you may be shown how that
player answered in PREVIOUS games with the same people (each game's true roles were
revealed afterwards, so every past answer is labelled TRUTH or LIE and with the slot
it pointed at). Roles are re-dealt every game: the player's current role is unknown
and may differ, but the PERSON persists, so use their past behaviour to judge how
they would answer IF they held the stated faction/constraint now. For example, a
player who told the truth whenever their role allowed lying is likely to do so again;
a must-lie player who kept pointing at the QUEEN slot will likely repeat that.
"""


def build_memory_judge_prompt(query: MemoryJudgeQuery) -> tuple[str, str]:
    """Memory-aware judge prompt: the situation plus THIS opponent's past-game record."""
    lines = [f"You are rating Player {query.opponent_id}."]
    if query.memory:
        lines += ["", "Their observed behaviour in previous games:", query.memory]
    else:
        lines += ["", "No previous games observed for this player yet."]
    lines += ["", "Situation to rate (a hypothesis about the CURRENT game):"]
    lines += _judge_situation_lines(query)
    return MEMORY_JUDGE_SYSTEM, "\n".join(lines)


# Manual chain-of-thought for the judge (ablation.judge_cot): with the model's native
# thinking off, one judge call is split into an explicit reasoning pass and a rating
# pass, so the CoT length is bounded by the prompt instead of the model's thinking mode.

def judge_cot_analysis_prompt(judge_user_prompt: str) -> str:
    """Step 1: the judge prompt, asking for reasoning only (no ratings yet)."""
    return judge_user_prompt + "\n\n" \
        + "Do NOT give the ratings yet. First reason step by step, in 3-6 short sentences: " \
        + "what and why would an answer in each allowed slot do for this answerer's faction, and " \
        + "what do their past games (if shown) suggest they actually do?"


def judge_cot_final_prompt(judge_user_prompt: str, analysis: str) -> str:
    """Step 2: the judge prompt plus the step-1 analysis, asking for the ratings."""
    return judge_user_prompt + "\n\nYour own step-by-step analysis:\n" + analysis \
        + "\n\nBased on this analysis, give the final ratings now."


PROFILER_SYSTEM = "You maintain an opponent profile of one player in Tofu Kingdom, a social deduction game.\n\n" \
    + GAME_FACTIONS + """

The SAME people play many games in a row; roles are re-dealt every game, and after
each game everyone's true role is revealed. You are shown, ground-truth annotated,
everything ONE player did in the previous games. Write the profile that best
predicts how this PERSON will answer in FUTURE games:
- tendencies per constraint: when free to answer anything (OTHER faction), do they
    tell the truth, lie randomly, or mislead deliberately? when forced to lie (QUEEN
    faction), which slot do they prefer to point at?
- consistency: do they repeat the same patterns, or vary?
- skill level: novice / intermediate / expert -- do their answers actually help their faction?
Keep only what helps prediction, in at most 6 sentences. Do not mention game counts
or game numbers."""


def build_profile_prompt(player_id: int, raw_report: str) -> tuple[str, str]:
    """One profile call per opponent: their full (windowed) ground-truth records."""
    return PROFILER_SYSTEM, "\n".join([raw_report, "", f"Write your profile of Player {player_id}."])


def build_ablation_prince_prompt(obs: PrinceObservation, valid_actions: list[dict], give_history: bool, \
    give_belief_state: bool, belief_mode: str, kripke_state: KripkeState, previous_summary: str = "", \
        memory_report: str = "") -> tuple[str, str]:
    """Build system + user prompt for AblationPrinceAgent.

    Reuses PRINCE_SYSTEM as system prompt. User prompt is configurable:
        - give_history: whether to include QA history
        - give_belief_state: whether to include Kripke belief info
        - belief_mode: "distribution" (role probabilities) or "worlds" (top possible worlds)
        - previous_summary: reasoning summary from prior turns (for reasoning history)
        - memory_report: raw cross-game opponent memory ("" = omit the block)

    Args:
        kripke_state: KripkeState instance (needed for belief info).

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    lines = []

    # Basic game info, always provided
    lines.extend(prince_basic_game_info(obs))

    # QA history, optional
    if give_history:
        lines.extend(qa_history(obs))

    # Belief state
    lines.append("")
    if give_belief_state and kripke_state is not None:
        if belief_mode == "worlds":
            lines.append("Current belief state (top possible worlds):")
            lines.append(kripke_state.get_top_worlds_report())
        else:
            lines.append("Current belief state (role probability distributions):")
            lines.append(kripke_state.get_full_report())

    # Cross-game opponent memory (same people across games, roles re-dealt each game)
    if memory_report:
        lines.append("")
        lines.append("You have played previous games with the SAME players (roles are re-dealt")
        lines.append("every game, but each player's answering style may persist). What you")
        lines.append("observed, with the truth revealed after each game:")
        lines.append(memory_report)

    # Reasoning summary from previous turns
    if previous_summary:
        lines.append("")
        lines.append("Your reasoning summary so far:")
        lines.append(previous_summary)


    lines.append("")
    if obs.phase == Phase.QUESTIONING:
        lines.append("Choose which question to ask next.")
    elif obs.phase == Phase.FINAL_QUESTION:
        lines.append("You can ask ONE final question to any player. Choose wisely.")
    elif obs.phase == Phase.GUESSING:
        lines.append("Based on all information above, guess which player is Princess Tofu.")

    lines.extend(action_choices(valid_actions))

    return PRINCE_SYSTEM, "\n".join(lines)
