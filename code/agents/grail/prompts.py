# GRAIL-specific prompt: the one-shot LLM 'vibes' pseudo-prior (paper's Prompt E.7).
# Shared blocks come from agents/prompts.py; the response schema is in agents/grail/schemas.py.
from __future__ import annotations

from game.predefined_data_classes import Observation
from agents.prompts import GAME_RULES, qa_history

def build_vibes_prompt(obs: Observation, player_ids: list[int], princess_dist: dict[int, float], \
    has_center: bool) -> str:
    """GRAIL-style 'vibes' prompt (analogue of the paper's Prompt E.7, 'Prior
    Probability Generation'): the LLM reads the rules, the Q&A transcript and the
    current per-candidate belief, then nudges each candidate's 'is the Princess'
    probability increase/decrease/same.

    Deviations from GRAIL's E.7 (deliberate, domain-forced):
      1. Semantic axis: E.7 adjusts P(player is Evil) on a binary good/evil variable;
         Tofu has no such axis, so we adjust P(candidate is Princess Tofu) instead.
      2. Output shape: E.7 returns a dict keyed by player NAME; Tofu players have no
         names and instructor's TOOLS mode can't constrain integer keys, so we return
         a list of {player, direction} records instead (see VibesResponse).
    """
    def label(pid: int) -> str:
        return "the center card" if pid == -1 else f"Player {pid}"

    targets = ", ".join(label(p) for p in player_ids)
    belief_lines = "\n".join(
        f"- {label(p)}: {round(princess_dist.get(p, 0.0), 2)}" for p in player_ids
    )
    center_note = " (no player holds the center card, but Princess Tofu may be hidden there)" \
        if has_center else ""

    return (
        GAME_RULES + "\n"
        "You are advising Prince Mochi. Read the Q&A below and judge, for EACH candidate, whether the\n"
        "answers make that candidate MORE or LESS likely to be Princess Tofu.\n\n"
        "Candidates: " + targets + center_note + "\n"
        + "\n".join(qa_history(obs)) + "\n\n"
        "Current estimate of P(is Princess Tofu), higher = more likely:\n"
        + belief_lines + "\n\n"
        # Domain reasoning cues (analogue of E.7's 'consider whether they behave
        # suspiciously...'), mirroring PRINCE_SYSTEM's deduction principles but framed
        # around the Princess question rather than good/evil alignment.
        "When judging, use how each role must answer (see the rules above):\n"
        "- Princess-faction players (Princess Tofu, Tofu Chef) always answer truthfully:\n"
        "  a candidate that truthful players point to as the Princess, or whose own\n"
        "  answers stay consistent with truthful play, is MORE likely to be Princess Tofu.\n"
        "- Queen-faction players always lie, and Other-faction players may mix truth and\n"
        "  lies to hide the Princess: a candidate contradicted by otherwise-consistent\n"
        "  testimony, or who claims a non-Princess role, is LESS likely.\n"
        "- If two players' answers are consistent, they tend to share a faction.\n\n"
        "For each candidate, choose exactly one direction:\n"
        "  increase: the answers make this candidate MORE likely to be Princess Tofu\n"
        "  decrease: the answers make this candidate LESS likely to be Princess Tofu\n"
        "  same: not enough evidence either way (use this when unsure)\n\n"
        # The output shape (a list of {player, direction}) is carried by the response
        # schema instructor sends, so we only state the task-level constraint the schema
        # can't express: cover every candidate exactly once.
        "Give exactly one judgement for EACH candidate listed above."
    )
