# Prompt templates shared by every LLM-driven Resistance agent: the game rules,
# the system prompts, and the reusable prompt blocks.
#
# Mirrors agents/prompts.py, which does the same for Tofu Kingdom. Block names
# and wording are kept the same wherever the two games ask for the same thing,
# so the prompts and the reasoning traces of the two games compare. A ported
# method keeps its own prompts in its own package and reuses the blocks here for
# the parts that describe the game itself.
#
# The model is small, so every line is short and plain: no long sentences, no
# figures of speech, one fact per line. A player is always written "Player 3", the
# way the Tofu prompts name them, and never "seat 3".
from __future__ import annotations

from game.resistance_data_classes import (
    MAX_REJECTIONS, MISSIONS_TO_WIN, PLAYER_PRESET,
    MissionRecord, ProposalRecord, ResistanceObservation, SpyObservation,
)

# The counts come from the rules, not from the text, so the prompt cannot drift
# away from what the environment actually does.
GAME_FACTIONS = f"""\
Two factions:
- RESISTANCE: does not know which faction any other player is in. Wins when {MISSIONS_TO_WIN} missions succeed.
- SPIES: knows who the other spies are. Wins when {MISSIONS_TO_WIN} missions fail, or when {MAX_REJECTIONS} teams are rejected on one mission."""


def game_rules(num_players: int) -> str:
    """the rules of a table this size

    A function and not a constant, because the team sizes change with the
    number of players.
    """
    preset = PLAYER_PRESET[num_players]
    team_sizes = ", ".join(str(size) for size in preset.team_sizes)
    return f"""\
There are {num_players} players. Their player ids are 0 to {num_players - 1}.
{preset.num_spies} players belong to the Spy faction. {num_players - preset.num_spies} players belong to the Resistance faction.

{GAME_FACTIONS}

Game flow:
1. PROPOSE: the leader picks the players for the mission team. The team sizes, mission by mission, are {team_sizes}.
2. TEAM VOTE: every player votes to approve or reject that team. A tie rejects it.
3. If the team is rejected, the next player becomes leader and step 1 happens again. If {MAX_REJECTIONS} teams are rejected on one mission, the spies win the game.
4. MISSION: each player on the team plays a Success card or a Fail card in secret. The mission fails if enough Fail cards are played.

Nobody talks in this game. After a team vote, everyone is told who voted which way.
After a mission, everyone is told if it succeeded or failed. There is no other information.
"""


def strategy_hint(obs: ResistanceObservation) -> str:
    """what this player should be trying to do, and how to read the game

    The Tofu prompts give one of these per faction. This gives one per side.
    """
    if isinstance(obs, SpyObservation):
        return f"""\
Your goal is to make {MISSIONS_TO_WIN} missions fail. You already know who the spies are.
- A mission can only fail if a spy is on the team and plays a Fail card. So try to get a spy on the team.
- Do not be obvious. The resistance players watch who proposed and who approved the teams that failed.
- If you play a Fail card every time you are on a team, they will find you. So keep your faction hidden."""

    return f"""\
Your goal is to make {MISSIONS_TO_WIN} missions succeed. You must work out who the spies are.
- {_failed_mission_note(obs)}
- A spy wants a spy on the team. A player who proposed or approved a team that then failed may be a spy.
- A player who rejects teams made of players you trust may be a spy.
- Players who vote the same way again and again are often on the same faction."""


def _failed_mission_note(obs: ResistanceObservation) -> str:
    """what a failed mission proves, which depends on a rule flag

    obs.resistance_must_pass carries the flag. With it off a resistance player
    may play a Fail card, so a failed mission stops being proof of a spy.
    """
    if obs.resistance_must_pass:
        return ("A resistance player cannot play a Fail card. So a failed mission proves that at "
            "least one player on that team belongs to the Spy faction.")
    return ("A resistance player is allowed to play a Fail card here. So a failed mission usually "
        "means a spy was on that team, but it is not proof.")


def system_prompt(obs: ResistanceObservation) -> str:
    """the rules, who this player is, and what it is trying to do"""
    faction = "Spy" if isinstance(obs, SpyObservation) else "Resistance"
    lines = ["You are playing The Resistance, a social deduction game.\n\n" + game_rules(obs.state.num_players),
        f"You are Player {obs.player_id}. You belong to the {faction} faction."]
    if isinstance(obs, SpyObservation):
        lines.append(_known_spies_line(obs))
    lines.append("")
    lines.append(strategy_hint(obs))
    return "\n".join(lines)


def _known_spies_line(obs: SpyObservation) -> str:
    """the teammates a spy was shown at the start

    Every player count in PLAYER_PRESET deals at least two spies, so a spy
    always has at least one teammate.
    """
    if not obs.known_spies:
        raise Exception(f"Player {obs.player_id} is a spy with no teammates, "
            "every player count deals at least two spies, this should never happen in a valid game.")
    return f"The other spies are {_player_list(obs.known_spies)}."


def _player_list(player_ids) -> str:
    """player ids as prose, always written out as 'Player 3'"""
    if not player_ids:
        return "nobody"
    return ", ".join(f"Player {player_id}" for player_id in sorted(player_ids))


def basic_game_info(obs: ResistanceObservation) -> list[str]:
    """where the game stands right now

    The Tofu analogue is prince_basic_game_info: the public setup every agent
    starts its user prompt with.
    """
    state = obs.state
    successes = state.mission_results.count(True)
    failures = state.mission_results.count(False)
    # The three counters read the same way, "X: n. WHO wins at k", so the
    # number and the faction it is good for are never in doubt. The rules
    # themselves are in the system prompt; these lines carry the state.
    return [
        f"You are Player {obs.player_id}.",
        f"Current phase: {state.phase.value.upper()}",
        f"Missions succeeded: {successes}. The Resistance wins at {MISSIONS_TO_WIN}.",
        f"Missions failed: {failures}. The Spies win at {MISSIONS_TO_WIN}.",
        f"Teams rejected on this mission: {state.rejections}. The Spies win at {MAX_REJECTIONS}.",
        f"This is mission {state.mission_index + 1}. Its team must have {obs.team_size} players. "
            f"{_fail_cards_line(obs.fails_required)}",
        f"The leader is Player {state.leader}.",
    ]


def _fail_cards_line(fails_required: int) -> str:
    """how many Fail cards sabotage this mission, which is 2 on one mission of a
    seven or more player game and 1 everywhere else"""
    if fails_required == 1:
        return "1 or more Fail card(s) makes it fail."
    return f"{fails_required} or more Fail cards make it fail."


def _mission_header(mission_index: int) -> str:
    """the line that opens one mission's block

    A whole history runs to about 36 lines of near identical entries. The
    header and the blank line before it cut that into one block per mission,
    so the model reads a few entries at a time. Every entry still names its
    own mission, so a line read on its own is still complete.
    """
    return f"--- Mission {mission_index + 1} ---"


def _entry_label(mission_index: int, proposal_number: int) -> str:
    """which mission an entry belongs to, and its place among that mission's proposals

    One mission takes as many proposals as it needs to get a team approved, so
    the two numbers together name one entry. Without both of them a rejected
    proposal reads as a mission of its own, and every later mission is numbered
    wrong.
    """
    return f"Mission {mission_index + 1}, proposal {proposal_number}"


def format_mission(mission: MissionRecord) -> list[str]:
    """one finished mission, formatted for the prompt

    The Tofu analogue is format_qa_pair. num_fails is only printed when the
    table announced it. The approved team is the last proposal of its mission,
    so its number is one past the rejected ones.
    """
    outcome = "SUCCEEDED" if mission.success else "FAILED"
    fails = "" if mission.num_fails is None else f", with {mission.num_fails} Fail card(s) played"
    label = _entry_label(mission.index, len(mission.rejected) + 1)
    return [
        f"{label}: APPROVED, and the mission {outcome}{fails}. Player {mission.leader} "
            f"proposed the team {_player_list(mission.team)}.",
        f"  {_vote_line(mission.team_votes)}",
    ]


def format_rejected(proposal: ProposalRecord, mission_index: int, \
    proposal_number: int) -> list[str]:
    """one team that was proposed and voted down, in the same two lines

    A rejected team uses up a proposal and not a mission, so the entry carries
    the mission it was proposed for.
    """
    label = _entry_label(mission_index, proposal_number)
    return [
        f"{label}: REJECTED, so no mission was played. Player {proposal.leader} "
            f"proposed the team {_player_list(proposal.team)}.",
        f"  {_vote_line(proposal.votes)}",
    ]


def _vote_line(votes: dict[int, bool]) -> str:
    """who approved and who rejected, in player id order"""
    approved = []
    rejected = []
    for player_id in sorted(votes):
        if votes[player_id]:
            approved.append(player_id)
        else:
            rejected.append(player_id)
    return f"Approved by {_player_list(approved)}. Rejected by {_player_list(rejected)}."


def mission_history(obs: ResistanceObservation) -> list[str]:
    """every proposal and every mission the table has seen, in order

    The Tofu analogue is qa_history.
    """
    state = obs.state
    return history_lines(state.missions, state.rejected_this_round, state.mission_index)


def history_lines(missions: list[MissionRecord], rejected_this_round: list[ProposalRecord], \
    mission_index: int) -> list[str]:
    """these records as prompt lines, in the order they happened

    A mission carries the proposals rejected before it, so walking the missions
    and then this mission's rejections replays the whole game.

    Args:
        rejected_this_round: proposals rejected on the mission being played now
        mission_index: the mission being played now
    """
    lines = [""]
    if not missions and not rejected_this_round:
        lines.append("Nothing has happened yet. This is the first proposal of the game.")
        return lines

    lines.append("Game history:")
    for mission in missions:
        lines.append("")
        lines.append(_mission_header(mission.index))
        for number, proposal in enumerate(mission.rejected, start=1):
            lines.extend(format_rejected(proposal, mission.index, number))
        lines.extend(format_mission(mission))

    # These belong to the mission being played now, which has no record yet.
    # It only opens a block once it has an entry to put in it.
    if rejected_this_round:
        lines.append("")
        lines.append(_mission_header(mission_index))
        for number, proposal in enumerate(rejected_this_round, start=1):
            lines.extend(format_rejected(proposal, mission_index, number))
    return lines


def think_step_by_step(what: str) -> list[str]:
    """the closing instruction, worded as the Tofu action_choices block words it"""
    return ["", f"Think step by step about {what}. "
        "(but keep it short, 3-6 sentences of reasoning is enough)"]


# Manual chain of thought (ablation decision_cot): with the model's own thinking
# off, one decision is split into a reasoning pass and a decision pass, so the
# length of the reasoning is bounded by the prompt instead of the thinking mode.
# The Tofu analogues are judge_cot_analysis_prompt and judge_cot_final_prompt.

def decision_cot_analysis_prompt(user_prompt: str) -> str:
    """the reasoning pass: the same prompt, asking for reasoning and no decision

    The prompt already says which decision is being made, so this only has to
    stop the model making it yet and say how long to reason for.
    """
    return user_prompt + "\n\n" \
        + "Do NOT decide yet. First reason step by step, in 4-8 short sentences: " \
        + "what the proposals, the votes and the missions so far say about which faction " \
        + "each player is in, and what that means for the decision above."


def decision_cot_final_prompt(user_prompt: str, analysis: str) -> str:
    """the decision pass: the same prompt plus the reasoning pass's own analysis"""
    return user_prompt + "\n\nYour own step-by-step analysis:\n" + analysis \
        + "\n\nBased on this analysis, make your decision now."


def _build_prompt(obs: ResistanceObservation, previous_summary: str, task: list[str], \
    what: str) -> tuple[str, str]:
    """(system_prompt, user_prompt) for one decision

    The user prompt is the same four blocks for every decision, the way the Tofu
    prompts are: the setup, the history, the running summary, then the task.
    """
    lines = list(basic_game_info(obs))
    lines.extend(mission_history(obs))
    if previous_summary:
        lines.append("")
        lines.append("Your reasoning summary so far:")
        lines.append(previous_summary)
    lines.append("")
    lines.extend(task)
    lines.extend(think_step_by_step(what))
    return system_prompt(obs), "\n".join(lines)


def build_propose_prompt(obs: ResistanceObservation, previous_summary: str = "") -> tuple[str, str]:
    """PROPOSE: the leader picks the mission team"""
    if isinstance(obs, SpyObservation):
        aim = "Your faction needs a spy on this team. But a team the others will reject is useless."
    else:
        aim = "Pick the players you think belong to the Resistance faction."
    task = [f"You are the leader. Pick {obs.team_size} players for mission "
        f"{obs.state.mission_index + 1}. You may pick yourself.", aim]
    return _build_prompt(obs, previous_summary, task, "which players to put on the team")


def build_team_vote_prompt(obs: ResistanceObservation, previous_summary: str = "") -> tuple[str, str]:
    """TEAM_VOTE: approve or reject the team the leader picked"""
    task = [f"Player {obs.state.leader} proposes this team: {_player_list(obs.state.proposed_team)}.",
        "Vote to approve this team or to reject it."]
    # Rejecting the last proposal of a mission ends the game there and then, so
    # each side is told what that means for it.
    if obs.state.rejections >= MAX_REJECTIONS - 1:
        if isinstance(obs, SpyObservation):
            task.append("This is the last team allowed on this mission. If the team is rejected, your "
                "faction wins the game now.")
        else:
            task.append("This is the last team allowed on this mission. If the team is rejected, the "
                "spies win the game now.")
    return _build_prompt(obs, previous_summary, task, "whether to approve this team")


def build_mission_prompt(obs: ResistanceObservation, previous_summary: str = "") -> tuple[str, str]:
    """MISSION_VOTE: play the Success card or the Fail card"""
    if isinstance(obs, SpyObservation):
        aim = (f"{obs.fails_required} or more Fail cards make this mission fail. Everyone will see the "
            "result and will suspect the players on this team if it fails.")
    else:
        aim = "A Fail card from you would hurt your own faction."
    task = [f"You are on the mission team: {_player_list(obs.state.proposed_team)}."]
    task.extend(_approval_lines(obs))
    task.append("Play the Success card or the Fail card.")
    task.append(aim)
    return _build_prompt(obs, previous_summary, task, "which card to play")


def _approval_lines(obs: ResistanceObservation) -> list[str]:
    """who put this team on the mission, and who voted it through

    Both are public the moment the vote closes, and both are what everyone will
    weigh once the result is in. A team only reaches a mission by winning a
    vote, so the votes are always there.
    """
    state = obs.state
    if not state.approved_votes:
        raise Exception("a mission team with no approving votes, a team only reaches a "
            "mission by winning a vote, this should never happen in a valid game.")
    return [f"Player {state.leader} proposed this team.", f"  {_vote_line(state.approved_votes)}"]


def build_final_guess_prompt(obs: ResistanceObservation, previous_summary: str = "") -> tuple[str, str]:
    """the game is over: name the spies

    Asked once per game, of a resistance player. Every other prompt is built
    while a decision is pending, so none of them can carry the last mission's
    result; this one does, and identity accuracy is scored on the answer.

    The Tofu analogue is the prince's final guess, which is what that game
    measures too. It does not reuse basic_game_info: those lines describe the
    mission being played, and there is no mission being played.
    """
    state = obs.state
    lines = [f"You are Player {obs.player_id}.",
        "The game is over.",
        f"Missions succeeded: {state.mission_results.count(True)}.",
        f"Missions failed: {state.mission_results.count(False)}."]
    lines.extend(mission_history(obs))
    if previous_summary:
        lines.append("")
        lines.append("Your reasoning summary so far:")
        lines.append(previous_summary)
    lines.append("")
    lines.append(f"Name the players you think were in the Spy faction. This game had "
        f"{PLAYER_PRESET[state.num_players].num_spies} spies.")
    lines.extend(think_step_by_step("who the spies were"))
    return system_prompt(obs), "\n".join(lines)
