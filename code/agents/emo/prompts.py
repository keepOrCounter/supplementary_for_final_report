"""EMO prompt templates, adapted to Tofu Kingdom.

The prompts are assembled from BLOCKS that map one-to-one to the symbols in the
paper's equations, so each ``build_*`` function conditions on exactly what its
equation specifies (Yu, Zhang & Lu, NAACL 2025):
"""
from __future__ import annotations

from game.predefined_data_classes import PrinceObservation, Phase
from agents.prompts import (
    PRINCE_SYSTEM, GAME_RULES,
    prince_basic_game_info, qa_history, action_choices, format_qa_pair,
)

def build_opponent_system(player_id: int) -> str:
    """OPPONENT_ROLE (R), specialised to one player (mirrors the paper's per-player framing)."""
    return GAME_RULES + f"""
You are the Prince's explicit opponent model for Player {player_id}. Your only task is to
infer the hidden role Player {player_id} holds, and to adjust that estimate as Player
{player_id} answers and as you receive feedback. You see ONLY what the Prince sees: the
public questions and answers, never any player's private view; inferring Player {player_id}'s
hidden role from that public record is exactly your job. An answer's truthfulness is fixed by
the answerer's faction: Princess faction MUST tell the truth, Queen faction MUST lie, Other
faction may do either. Reason about which role would make Player {player_id} answer as they did.
"""

VALIDATOR_SYSTEM = GAME_RULES + """
You are the central validator coordinating the Prince's explicit models of the players.
Each player is modelled separately, so individual models can end up mutually inconsistent
(e.g. two of them claiming to be Princess Tofu). You enforce the global rules: every role
in the game is UNIQUE, there is exactly one Princess Tofu and one Queen Tofu, and one role
may be the face-down center card that no player holds.
"""


def _center_note(obs: PrinceObservation) -> str:
    return (" (one role is the face-down center card that no player holds, so Princess Tofu "\
        "may be in the center rather than held by any player)" \
            if obs.has_center_card else " (there is no center card; every role is held by a player)")


def _player_answers(obs: PrinceObservation, player_id: int) -> list[str]:
    """This player's answered questions, formatted (their action a_{j,t})."""
    return [format_qa_pair(qa) for qa in obs.qa_history if qa.target_id == player_id and qa.answer is not None]


def role_prediction_format(pred) -> str:
    """Render a RolePrediction as a plain-prose noun phrase (no dict/JSON)."""
    if getattr(pred, "role", "unknown") not in (None, "", "unknown"):
        return f"most likely the {pred.faction} faction, specifically {pred.role}"
    return f"most likely the {pred.faction} faction (exact role still unclear)"


def _compose(*blocks: list[str]) -> str:
    """Join non-empty blocks (each a list of lines) with a blank line between them."""
    out: list[str] = []
    for block in blocks:
        if not block:
            continue
        if out:
            out.append("")
        out.extend(block)
    return "\n".join(out)




def block_public_info(obs: PrinceObservation) -> list[str]:
    """p_tilde_{j,t}: the CURRENT public info the Prince sees: the game setup
    (players, roles in play, phase, center card).

    Per the paper p_tilde_{j,t} = p_{i,t} is the current-turn environment info.
    """
    lines = ["Public information (the game setup the Prince can see):"]
    lines.extend(prince_basic_game_info(obs))
    return lines


def block_action(obs: PrinceObservation, player_id: int) -> list[str]:
    """a_{j,t}: the modelled player's own answers, the actions to explain."""
    answers = _player_answers(obs, player_id)
    lines = [f"Player {player_id}'s own answer(s) so far (the actions to explain):"]
    if answers:
        lines.extend(f"- {a}" for a in answers)
    else:
        lines.append("- (none yet)")
    return lines


def block_history(obs: PrinceObservation, player_id: int = None, prediction_trace=()) -> list[str]:
    """H_{j,t}: the accumulated history: the Q&A transcript (past interactions /
    observations) plus, for an atomic model, its own past predictions.

    The paper defines H_{i,t} = (p_0, q_0, a_0, ..., p_{t-1}, q_{t-1}, a_{t-1}), i.e.
    the running transcript; for the opponent model it additionally "contains ...
    predictions ... of the opponent model". The validator (H_N) passes no
    ``prediction_trace``: it just needs the shared transcript.
    """
    lines: list[str] = []
    transcript = qa_history(obs) # past interactions (the transcript)
    if transcript and transcript[0] == "":
        transcript = transcript[1:] # drop qa_history's leading blank
    lines.extend(transcript)
    if prediction_trace:
        lines.append("")
        lines.append(f"Your earlier predictions for Player {player_id} (from previous rounds):")
        for entry in prediction_trace:
            role = entry.get("role") or "unknown"
            if role not in (None, "", "unknown"):
                desc = f"the {entry['faction']} faction, specifically {role}"
            else:
                desc = f"the {entry['faction']} faction (exact role unclear)"
            lines.append(f"- after {entry['after_answers']} answer(s), you thought Player {player_id} was {desc}.")
    return lines


def block_current_prediction(player_id: int, pred) -> list[str]:
    """q_hat: the current prediction being critiqued / refined."""
    return [f"Your current prediction for Player {player_id}: {role_prediction_format(pred)}."]


def block_feedback(feedback: str, label: str) -> list[str]:
    """F^s / F^g: the feedback to act on (``label`` names its source)."""
    return [f"{label} to act on: {feedback}"]


def block_all_predictions(all_predictions: dict[int, object], reviewing_pid: int = None, \
    with_rationale: bool = False) -> list[str]:
    """{q_hat_j}: every model's current prediction (validator context / summary input).

    ``with_rationale`` appends each prediction's one-line reason: useful for the
    summary (Eq. 6), off for the validator's consistency check (Eq. 4).
    """
    lines = ["Current predictions from all the player models:"]
    for pid, pred in sorted(all_predictions.items()):
        mark = "  (the model you are reviewing)" if pid == reviewing_pid else ""
        reason = f". Reason: {pred.rationale}" if with_rationale else ""
        lines.append(f"- Player {pid}: {role_prediction_format(pred)}{mark}{reason}")
    return lines



def build_init_prompt(obs: PrinceObservation, player_id: int, prediction_trace=()) -> str:
    """INIT (Eq. 1): R, H, p_tilde."""
    instruction = [f"Task: using the record above, infer the hidden role Player {player_id} holds. Give your best "\
        "single guess for the faction and the specific role (use 'unknown' for the role only if the "\
            "faction is clear but the exact role is not)."]
    return _compose(block_public_info(obs), block_history(obs, player_id, prediction_trace), instruction)


def build_self_feedback_prompt(obs: PrinceObservation, player_id: int, pred, prediction_trace=()) -> str:
    """SELF_FEEDBACK (Eq. 2): R, H, p_tilde, a, q_hat."""
    instruction = [f"Task(in AT MOST 4 short sentences): analyze step by step whether your current prediction is consistent with "\
        f"Player {player_id}'s answer(s) under the faction truth/lie constraint, and give one adjustment to make "\
            "(if any). Be brief: no long walkthrough."]
    return _compose(block_public_info(obs), block_history(obs, player_id, prediction_trace), \
        block_action(obs, player_id), block_current_prediction(player_id, pred), instruction)


def build_self_refine_prompt(obs: PrinceObservation, player_id: int, pred, feedback: str, prediction_trace=()) -> str:
    """SELF_REFINE (Eq. 3): R, H, p_tilde, a, q_hat, F^s."""
    instruction = ["Task: using the feedback, output your revised prediction (faction + specific role) for this player."]
    return _compose(block_public_info(obs), block_history(obs, player_id, prediction_trace), \
        block_action(obs, player_id), block_current_prediction(player_id, pred), \
            block_feedback(feedback, "Your self-feedback"), instruction)


def build_global_refine_prompt(obs: PrinceObservation, player_id: int, pred, global_feedback: str, prediction_trace=()) -> str:
    """GLOBAL_REFINE (Eq. 5): R, H, p_tilde, a, q_hat, F^g."""
    instruction = ["Task: using the central validator's feedback, output your revised prediction " \
        "(faction + specific role) so it is consistent with the whole game (unique roles, " \
            "exactly one Princess Tofu and one Queen Tofu)."]
    return _compose(block_public_info(obs), block_history(obs, player_id, prediction_trace), \
        block_action(obs, player_id), block_current_prediction(player_id, pred), \
            block_feedback(global_feedback, "The central validator's feedback"), instruction)




def build_global_feedback_prompt(obs: PrinceObservation, player_id: int, pred, all_predictions: dict[int, object]) -> str:
    """GLOBAL_FEEDBACK (Eq. 4): R, H_N, p_tilde_N, a_N, {q_hat}. Feedback for the ONE reviewed model.

    The validator reads the whole public record and every model's prediction, but
    produces feedback only for Player ``player_id``: the opponent whose turn
    triggered this update (Algorithm 2 runs EMO for the acting opponent alone).
    """
    instruction = [f"Task: focus on the model for Player {player_id}. Every role is unique; there is exactly one "\
        f"Princess Tofu and one Queen Tofu{_center_note(obs)}. In AT MOST 4 short sentences, if any, point out the "\
            f"biggest inconsistency (e.g. two models claiming the same unique role) and the one adjustment for "\
                f"Player {player_id}'s model. Be brief: do NOT walk through every player."]
    return _compose(block_public_info(obs), block_history(obs), block_all_predictions(all_predictions, player_id), instruction)


def build_summary_prompt(obs: PrinceObservation, all_predictions: dict[int, object]) -> str:
    """SUMMARY (Eq. 6): {q_hat} only -> the aggregated brief for the reasoning module.

    Reuses ``block_all_predictions`` (with rationales, which help the summary):
    Eq. 6 conditions on the predictions alone, so no transcript here.
    """
    instruction = ["Task: give the Prince a BRIEF summary: ONE short line per player (most likely role + faction), "\
        "flag any contradiction in a few words, then one line for where Princess Tofu most likely is" \
            f"{_center_note(obs)}. No step-by-step reasoning, no rule restatement."]
    return _compose(block_all_predictions(all_predictions, with_rationale=True), instruction)



def build_reasoning_prompt(obs: PrinceObservation, valid_actions: list[dict], opponent_summary: str, \
    previous_summary: str = "") -> tuple[str, str]:
    """The CoT reasoning module that consumes q_hat_{-i} (the validator's prose summary).

    Reuses ``PRINCE_SYSTEM`` and returns ``(system_prompt, user_prompt)``. The
    opponent-model summary (plain prose) is injected between the transcript and the
    action list: acting on modelled opponent roles IS the point of EMO.
    """
    lines = list(block_public_info(obs)) # p_{i,t}: game setup
    lines.append("")
    lines.extend(block_history(obs)) # H_{i,t}: the Q&A transcript

    lines.append("")
    lines.append("Explicit models of your opponents (summary of your per-player opponent models):")
    lines.append(opponent_summary if opponent_summary else "No player has answered yet, so there is nothing to model.")

    if previous_summary:
        lines.append("")
        lines.append("Your reasoning summary so far:")
        lines.append(previous_summary)

    lines.append("")
    if obs.phase == Phase.QUESTIONING:
        lines.append("Using the opponent models above, choose which question to ask next to best " \
            "pin down who is Princess Tofu.")
    elif obs.phase == Phase.FINAL_QUESTION:
        lines.append("Using the opponent models above, ask ONE final question to any player. Choose wisely.")
    elif obs.phase == Phase.GUESSING:
        lines.append("Using the opponent models above, guess which player (or the center card) is Princess Tofu.")

    lines.extend(action_choices(valid_actions))
    return PRINCE_SYSTEM, "\n".join(lines)
