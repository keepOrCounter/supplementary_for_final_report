# per-game records and summary rows for Resistance experiments
#
# The counterpart of core/logger.py, which does this for Tofu Kingdom. The two
# genuinely share create_output_dir and save_summary_csv, which is why those
# live in core/output.py and both import them from there. Everything else is
# per game, because the two games record different things.
from __future__ import annotations
import csv
import json
import os

from core.output import create_output_dir, save_summary_csv
from game.resistance_data_classes import MissionRecord, ProposalRecord, ResistanceState


def serialize_mission(mission: MissionRecord) -> dict:
    """one mission, including the proposals that were rejected before it

    The key names say which team and whose votes, because "team" also names a
    faction elsewhere and "votes on the team" is not "the team's votes".
    """
    return {
        "mission_index": mission.index, # which mission, from 0
        "proposed_by": mission.leader, # player id of the leader whose team was approved
        "mission_team": sorted(mission.team), # player ids that went on the mission
        "votes_on_team": _sorted_votes(mission.team_votes), # every player's approve or reject
        "rejected_proposals": serialize_proposals(mission.rejected), # sunk before this team passed
        "mission_succeeded": mission.success,
        "fail_cards": mission.num_fails, # Fail cards played, null when the config hides it
    }


def serialize_action(action: dict) -> dict:
    """one action as JSON, with the team as a sorted list"""
    record = dict(action)
    if "team" in record:
        record["team"] = sorted(record["team"])
    return record


def serialize_proposals(proposals: list[ProposalRecord]) -> list[dict]:
    """rejected proposals, each with the leader that made it and the votes that sank it"""
    serialized = []
    for proposal in proposals:
        serialized.append({
            "proposed_by": proposal.leader, # player id of the leader that put it forward
            "proposed_team": sorted(proposal.team), # player ids put forward, never approved
            "votes_on_team": _sorted_votes(proposal.votes), # every player's approve or reject
        })
    return serialized


def serialize_beliefs(belief_log: list[dict]) -> list[dict]:
    """belief snapshots, with the player id keys turned into strings for JSON
    Args:
        belief_log: one snapshot per reporting agent per number of missions played
            [{"player_id": int, "missions_seen": int, "spy_probability": dict[int, float]}]
    """
    serialized = []
    for record in belief_log:
        probabilities = {}
        for player_id in sorted(record["spy_probability"]):
            probabilities[str(player_id)] = record["spy_probability"][player_id]
        serialized.append({
            "player_id": record["player_id"],
            "missions_seen": record["missions_seen"],
            "spy_probability": probabilities,
        })
    return serialized


def score_snapshot(record: dict, spies_ids: frozenset[int]) -> tuple[int, int]:
    """(players called correctly, players guessed) for one belief snapshot

    A player is called a spy when its probability is above 50%, and called
    resistance otherwise. The holder's own player id is not guessed.
    """
    correct = 0
    guesses = 0
    for player_id, probability in record["spy_probability"].items():
        if player_id == record["player_id"]:
            continue
        guesses += 1
        if (probability > 0.5) == (player_id in spies_ids):
            correct += 1
    return correct, guesses


def build_belief_rows(game_index: int, seed: int, state: ResistanceState, agents: dict, \
    belief_log: list[dict], crashed: bool = False) -> list[dict]:
    """one row per belief snapshot, for beliefs.csv

    Only resistance players appear; the runner does not snapshot a spy, which
    knows its teammates already.

    Long format on purpose: one row per (game, agent, number of missions seen),
    with the agent's type in a column. Grouping by that column gives a score
    per method, grouping by it and missions_seen gives the curve over a game,
    and neither needs the columns to change when the lineup does.

    crashed marks every row of a game that raised. Its last snapshot was taken
    mid game rather than on the closing question, so a breakdown that compares
    agents at the end of a game has to leave that game out.
    """
    rows = []
    for record in belief_log:
        correct, guesses = score_snapshot(record, state.spies_ids)
        rows.append({
            "game_index": game_index,
            "seed": seed,
            "player_id": record["player_id"],
            "agent": str(agents[record["player_id"]]),
            "missions_seen": record["missions_seen"],
            "guesses": guesses,
            "correct": correct,
            "accuracy": correct / guesses if guesses else None,
            "crashed": crashed, # the game raised, so its last snapshot is mid game
        })
    return rows


def save_belief_csv(output_dir: str, rows: list[dict]) -> None:
    """write every game's belief rows to <output_dir>/beliefs.csv"""
    if not rows:
        return

    path = os.path.join(output_dir, "beliefs.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def identity_accuracy(belief_log: list[dict], spies_ids: frozenset[int]) -> float:
    """how many players each reporting agent labelled correctly, at its last look

    Read it against the no information baseline. An agent that answers one half
    everywhere calls every player resistance, so from a resistance player it scores
    the share of other players that really are resistance: three of five in a six
    player game, 60%. Only above that is the agent telling the sides apart.

    Only each agent's last snapshot counts, which in a finished game is the
    closing one: an agent is asked for its belief when it acts, so a belief read
    during play never carries the result that ended the game, and the runner
    asks every resistance agent once more after the last mission resolves. So
    this is accuracy given everything the game showed, and
    identity_missions_seen says how many missions that answer saw.

    The earlier snapshots stay in the game record, so a curve over the mission
    count can be built from there.

    None when no agent in the game held a belief.
    """
    last = {}
    for record in belief_log:
        last[record["player_id"]] = record

    correct = 0
    guesses = 0
    for record in last.values():
        player_correct, player_guesses = score_snapshot(record, spies_ids)
        correct += player_correct
        guesses += player_guesses

    if guesses == 0:
        return None
    return correct / guesses


def identity_missions_seen(belief_log: list[dict]) -> int:
    """how many missions the newest belief snapshot in this game had seen

    The companion of identity_accuracy, which scores that snapshot. Read it
    against missions_played in the same row: the two are equal in a finished
    game, and a difference means some agent was never asked the closing
    question, which is a bug in the runner and not a property of the game.

    None when no agent in the game held a belief.
    """
    seen = None
    for record in belief_log:
        if seen is None or record["missions_seen"] > seen:
            seen = record["missions_seen"]
    return seen


def extract_llm_stats(agents: dict) -> dict:
    """token and call counts per player, and the game's totals

    Reads whatever agent carries an llm_client, so a new LLM method is counted
    without touching this. A player without a client does not appear at all; a
    player that holds one but never used it reports zeros.
    """
    per_agent = {}
    game_total = {
        "total_calls": 0,
        "retry_count": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }
    tracking_complete = True # false as soon as one response came back without usage data

    for player_id, agent in agents.items():
        client = getattr(agent, "llm_client", None)
        if client is None:
            continue
        stats = client.stats.to_dict()
        per_agent[str(player_id)] = stats
        for key in game_total:
            game_total[key] += stats.get(key, 0)
        if not stats.get("token_tracking_complete", True):
            tracking_complete = False

    game_total["token_tracking_complete"] = tracking_complete
    return {"per_agent": per_agent, "game_total": game_total}


def extract_likelihood_stats(agents: dict) -> dict:
    """the likelihood judge's lookups per player, and the game's totals

    Reads whatever agent holds a belief with judge stats, so a player without
    one does not appear. The judge's answers are shared by every player of a
    game, so game_total's judge_calls is the number of questions the game asked.
    """
    per_agent = {}
    game_total = {"queries": 0, "judge_calls": 0, "cache_hits": 0}
    for player_id, agent in agents.items():
        stats = getattr(getattr(agent, "belief", None), "stats", None)
        if stats is None:
            continue
        per_agent[str(player_id)] = {"queries": stats.queries, "judge_calls": stats.judge_calls, \
            "cache_hits": stats.cache_hits}
        for key in game_total:
            game_total[key] += per_agent[str(player_id)][key]
    return {"per_agent": per_agent, "game_total": game_total}


def extract_reasoning(agents: dict) -> dict:
    """what each LLM agent thought, per player

    decision_log is one entry per LLM call; full_reasoning is the model's own
    thinking when the model returns it. Both are absent for an agent that does
    not keep them, so a non LLM player contributes nothing.
    """
    reasoning = {}
    for player_id, agent in agents.items():
        traces = {}
        if getattr(agent, "decision_log", None):
            traces["decisions"] = agent.decision_log
        if getattr(agent, "reasoning_history", None):
            traces["full_reasoning"] = agent.reasoning_history
        if getattr(agent, "summary_history", None):
            traces["summary_iterations"] = agent.summary_history
        if getattr(getattr(agent, "belief", None), "judge_log", None):
            traces["judge_log"] = agent.belief.judge_log
        if traces:
            reasoning[str(player_id)] = traces
    return reasoning


def save_game_record(output_dir: str, game_index: int, seed: int, state: ResistanceState, \
    agents: dict, action_log: list[dict], belief_log: list[dict] = None, crashed: bool = False, \
        crash_reason: str = "", duration_seconds: float = None) -> None:
    """write one game to <output_dir>/game_<index>.json"""
    belief_log = belief_log or []
    missions = []
    for mission in state.missions:
        missions.append(serialize_mission(mission))

    player_agents = {}
    for player_id, agent in agents.items():
        player_agents[str(player_id)] = str(agent)

    record = {
        "game_index": game_index,
        "seed": seed,
        # the game itself raised, which is not a mission failing
        "crashed": crashed,
        "crash_reason": crash_reason,
        "duration_seconds": duration_seconds,
        "num_players": state.num_players,
        "spies": sorted(state.spies_ids),
        "agents": player_agents,
        "winner": state.winner.value if state.winner else None,
        "win_reason": state.win_reason,
        "mission_successes": list(state.mission_results), # true means that mission succeeded
        "missions": missions,
        # Proposals rejected after the last completed mission. A game lost to
        # five rejections ends inside a round, so these belong to no mission
        # record and would otherwise be dropped.
        "rejected_after_last_mission": serialize_proposals(state.rejected_this_round),
        # one snapshot per reporting agent per number of missions played
        "beliefs": serialize_beliefs(belief_log),
        "identity_accuracy": identity_accuracy(belief_log, state.spies_ids),
        "identity_missions_seen": identity_missions_seen(belief_log),
        "actions": action_log,
        # both empty when no player in this game used an LLM
        "llm_stats": extract_llm_stats(agents),
        # empty when no player's belief asks the likelihood judge
        "likelihood_stats": extract_likelihood_stats(agents),
        "reasoning": extract_reasoning(agents),
    }

    path = os.path.join(output_dir, f"game_{game_index:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4, ensure_ascii=False)


def build_summary_row(game_index: int, seed: int, state: ResistanceState, agents: dict, \
    belief_log: list[dict] = None, crashed: bool = False, crash_reason: str = "", \
        duration_seconds: float = None) -> dict:
    """one row of summary.csv"""
    belief_log = belief_log or []
    llm_total = extract_llm_stats(agents)["game_total"]
    judge_total = extract_likelihood_stats(agents)["game_total"]
    successes = state.mission_results.count(True)
    failures = state.mission_results.count(False)

    total_rejections = 0
    for mission in state.missions:
        total_rejections += len(mission.rejected)
    # A game lost to five rejections has no mission record for the last round.
    total_rejections += state.rejections

    player_types = []
    for player_id in sorted(agents):
        player_types.append(str(agents[player_id]))

    return {
        "game_index": game_index,
        "seed": seed,
        "crashed": crashed, # the game raised, which is not a mission failing
        "crash_reason": crash_reason,
        "duration_seconds": duration_seconds,
        "winner": state.winner.value if state.winner else None,
        "win_reason": state.win_reason,
        "missions_played": len(state.mission_results),
        "missions_succeeded": successes,
        "missions_failed": failures,
        "total_rejections": total_rejections,
        "spies": " ".join(str(player_id) for player_id in sorted(state.spies_ids)),
        "agents": " ".join(player_types),
        # empty when no agent in this game holds a belief, the random baseline for one
        "identity_accuracy": identity_accuracy(belief_log, state.spies_ids),
        # missions the scored snapshot saw, which equals missions_played in a
        # finished game. See identity_accuracy.
        "identity_missions_seen": identity_missions_seen(belief_log),
        "identity_snapshots": len(belief_log),
        # zero when no player in this game used an LLM
        "llm_calls": llm_total["total_calls"],
        "llm_total_tokens": llm_total["total_tokens"],
        # zero when no player's belief asks the likelihood judge; its calls are
        # also counted in llm_calls
        "lk_queries": judge_total["queries"],
        "lk_judge_calls": judge_total["judge_calls"],
        "lk_cache_hits": judge_total["cache_hits"],
    }


def _sorted_votes(votes: dict[int, bool]) -> dict:
    """votes keyed by player_id, in player_id order, with string keys for JSON"""
    ordered = {}
    for player_id in sorted(votes):
        ordered[str(player_id)] = votes[player_id]
    return ordered
