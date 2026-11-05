"""
The Resistance - Experiment Analyzer

The counterpart of analyze.py, which does this for Tofu Kingdom. Same two modes
and the same matplotlib conventions; the columns it reads are this game's.

Four modes:
  1. Single experiment: python3 analyze_resistance.py results/<run>/summary.csv
  2. Batch experiment:  python3 analyze_resistance.py results/batch_resistance_<ts>/batch_summary.csv
  3. Type posteriors:   python3 analyze_resistance.py results/<run>/memory.json
  4. Learning curves:   python3 analyze_resistance.py --learning-curve results/<grid output_dir> ...
     (every run under the directories given: the tested player's results game by
     game, and how fast the Bayes player's memory finds each player's type)

Batch mode is auto-detected by the 'combination' column, which only
batch_runner_resistance.py writes.

analyze.py is the same thing for Tofu Kingdom, and the two are named after the
game the way the runners and the loggers are. They share no code: the games
report different things, and the little that is common (loading a CSV) is
smaller than the import would be.

Both modes read beliefs.csv (batch_beliefs.csv) next to the summary when it is
there. It carries one row per belief snapshot, which is what a per agent
breakdown and the accuracy curve are built from.
"""

from __future__ import annotations
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import MaxNLocator

from agents.resistance.our_agents.bayes import TYPE_MODELS
from game.resistance_data_classes import Team
from run_resistance import agent_label, modelled_type, turned_seats

# One colour per line in the charts, reused in order. The first is the colour
# analyze.py draws its own curves in.
COLOURS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
# Blue, red and purple, for the charts that compare up to three lines. They stay
# apart for colour blind readers too, which the red and green above do not. A
# line style for each as well, so no line is told apart by its colour alone.
LINE_COLOURS = ["#2196F3", "#F44336", "#9C27B0"]
LINE_STYLES = ["-", "--", ":"]


# ══════════════════════════════════════════════════════════════
#  Loading
# ══════════════════════════════════════════════════════════════

def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_beliefs(csv_path: str, name: str) -> list[dict]:
    """the belief snapshots written next to this summary, or nothing

    csv_path: the summary CSV that was given on the command line
    name: the belief file's name in that same directory

    A run whose agents hold no belief writes no such file, so an empty list is
    an ordinary answer and not an error.
    """
    path = os.path.join(os.path.dirname(csv_path), name)
    if not os.path.isfile(path):
        return []

    rows = []
    for row in load_csv(path):
        row["missions_seen"] = int(row["missions_seen"])
        row["guesses"] = int(row["guesses"])
        row["correct"] = int(row["correct"])
        row["accuracy"] = to_float(row["accuracy"])
        # A run from before the runner wrote this column marks nothing as
        # crashed, so all of its snapshots count, the way they used to.
        row["crashed"] = row.get("crashed", "") == "True"
        rows.append(row)
    return rows


def is_batch_csv(rows: list[dict]) -> bool:
    """batch_runner_resistance writes one row per grid cell, tagged with its label"""
    return len(rows) > 0 and "combination" in rows[0]


def to_float(value) -> float:
    """a CSV cell as a float, or None when the runner left it empty"""
    if value in ("", "None", None):
        return None
    return float(value)


def to_int(value) -> int:
    """a CSV cell as an int, or 0 when the runner left it empty"""
    if value in ("", "None", None):
        return 0
    return int(value)


# ══════════════════════════════════════════════════════════════
#  Measures
# ══════════════════════════════════════════════════════════════

def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """confidence interval for a win rate, the Wilson score interval

    wins: games won
    games: games played
    z: the standard normal quantile, 1.96 for a 95% interval
    rate: wins / games, the observed win rate
    centre, half: the interval is centre - half to centre + half

    An LLM run affords tens of games, not hundreds, and the textbook normal
    interval runs outside 0 to 1 at those counts. This one does not, so the
    width it reports is the honest one.
    """
    if games == 0:
        return 0.0, 0.0
    rate = wins / games
    denominator = 1 + z * z / games
    centre = (rate + z * z / (2 * games)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / denominator
    # The interval lies inside 0 to 1, and the clamp is only against a rounding
    # error at the ends, which would otherwise print as "-0.0%".
    return max(centre - half, 0.0), min(centre + half, 1.0)


def no_information_baseline(num_players: int, num_spies: int) -> float:
    """what a resistance player scores by calling every other player resistance

    num_players: players at the table
    num_spies: spies dealt
    others: the players this one is scored on, which is everyone but itself

    Identity accuracy has to be read against this. Three of the five others are
    resistance in a six player game, so 60% is what naming nobody scores, and
    only above it is an agent telling the factions apart.
    """
    others = num_players - 1
    return (others - num_spies) / others


def table_size(rows: list[dict]) -> tuple[int, int]:
    """(num_players, num_spies) read off a summary row

    The runner writes one agent per player and one id per spy, both as space
    separated lists, so counting them is what says how big the table was.
    Returns (0, 0) for a summary with no row that carries them.
    """
    for row in rows:
        agents = row.get("agents", "")
        spies = row.get("spies", "")
        if agents and spies:
            return len(agents.split()), len(spies.split())
    return 0, 0


def final_snapshots(beliefs: list[dict]) -> dict[tuple, dict]:
    """the last snapshot each agent held in each game that finished

    Keyed by (combination, game_index, player_id).

    Only finished games count, so every agent is scored on the same thing: the
    closing guess, made with the whole game in front of it. A crashed game's
    last snapshot was taken mid game, and keeping it would put two different
    amounts of information into one number.
    """
    last = {}
    for row in beliefs:
        if row["crashed"]:
            continue
        key = (row.get("combination", ""), row["game_index"], row["player_id"])
        if key not in last or row["missions_seen"] >= last[key]["missions_seen"]:
            last[key] = row
    return last


def mark_crashed_from_summary(beliefs: list[dict], rows: list[dict]) -> None:
    """copy each game's crashed flag from the summary onto its belief rows

    beliefs: the snapshot rows, changed in place
    rows: the summary rows, which have carried the flag since the runner was
        written, while beliefs.csv only started carrying it later

    So a run recorded before the column existed is still scored on its finished
    games alone. A batch has no per game summary to read and relies on the
    column instead.
    """
    crashed = {}
    for row in rows:
        crashed[row["game_index"]] = row["crashed"] == "True"
    for row in beliefs:
        if row["game_index"] in crashed:
            row["crashed"] = crashed[row["game_index"]]


def count_games(beliefs: list[dict], crashed: bool) -> int:
    """games in a belief log that did, or did not, crash

    beliefs: the snapshot rows
    crashed: True counts the games that raised, False the ones that finished
    """
    games = set()
    for row in beliefs:
        if row["crashed"] == crashed:
            games.add((row.get("combination", ""), row["game_index"]))
    return len(games)


# ══════════════════════════════════════════════════════════════
#  Single experiment
# ══════════════════════════════════════════════════════════════

def analyze_single(rows: list[dict], csv_path: str) -> None:
    """one run's summary.csv"""
    played = []
    crashed = []
    for row in rows:
        if row["crashed"] == "True":
            crashed.append(row)
        else:
            played.append(row)

    print()
    print("=" * 64)
    print("  RESISTANCE EXPERIMENT")
    print("=" * 64)
    print(f"  Source: {csv_path}")
    print()

    print(f"  Games:  {len(rows)} run, {len(played)} finished, {len(crashed)} crashed"
        f" ({len(crashed) / len(rows):.1%})")
    print_crash_reasons(crashed)
    print()

    print_win_rate(played)
    print()
    print_missions(played)
    print()

    num_players, num_spies = table_size(rows)
    beliefs = load_beliefs(csv_path, "beliefs.csv")
    mark_crashed_from_summary(beliefs, rows)
    print_identity(played, beliefs, num_players, num_spies)
    print()

    print_llm_usage(rows)
    print_durations(rows)
    print()

    if played:
        save_single_charts(played, beliefs, num_players, num_spies, csv_path)


def print_crash_reasons(crashed: list[dict]) -> None:
    """how many games each kind of crash cost

    Grouped by the exception type, which is what the runner writes in front of
    the message. A run that keeps losing games to one exception is a run whose
    win rate is measured on whatever survived, so the count belongs at the top.
    """
    if not crashed:
        return

    kinds = Counter()
    for row in crashed:
        reason = row["crash_reason"]
        kinds[reason.split(":")[0]] += 1
    print("  Crashed on:")
    for kind, count in kinds.most_common():
        print(f"    {count:>4}  {kind}")


def print_win_rate(played: list[dict]) -> None:
    """the resistance win rate, with the interval its game count supports"""
    if not played:
        print("  No finished game to score.")
        return

    wins = 0
    for row in played:
        if row["winner"] == "resistance":
            wins += 1
    low, high = wilson_interval(wins, len(played))
    print(f"  Resistance wins: {wins}/{len(played)} ({wins / len(played):.1%})"
        f"   95% CI [{low:.1%}, {high:.1%}]")

    reasons = Counter()
    for row in played:
        reasons[f"{row['winner']}: {row['win_reason']}"] += 1
    print("  Ended by:")
    for reason, count in reasons.most_common():
        print(f"    {count:>4}  {reason}")


def print_missions(played: list[dict]) -> None:
    """how the missions themselves went, per game"""
    if not played:
        return

    missions = []
    succeeded = []
    rejections = []
    for row in played:
        missions.append(to_int(row["missions_played"]))
        succeeded.append(to_int(row["missions_succeeded"]))
        rejections.append(to_int(row["total_rejections"]))
    games = len(played)
    print(f"  Missions per game: {sum(missions) / games:.1f}"
        f"   ({sum(succeeded) / games:.1f} succeeded, "
        f"{(sum(missions) - sum(succeeded)) / games:.1f} failed)")
    print(f"  Rejected proposals per game: {sum(rejections) / games:.1f}"
        f"   (max {max(rejections)} in one game)")


def print_identity(played: list[dict], beliefs: list[dict], num_players: int, \
    num_spies: int) -> None:
    """identity accuracy, against the baseline and per agent

    The per game number in summary.csv mixes every agent that held a belief, so
    the breakdown that says which method is doing the work comes from
    beliefs.csv instead.
    """
    scored = []
    for row in played:
        accuracy = to_float(row["identity_accuracy"])
        if accuracy is not None:
            scored.append(row)

    if not scored:
        print("  Identity accuracy: no agent in this run held a belief.")
        return

    accuracies = []
    seen = []
    missions = []
    for row in scored:
        accuracies.append(to_float(row["identity_accuracy"]))
        seen.append(to_int(row["identity_missions_seen"]))
        missions.append(to_int(row["missions_played"]))
    baseline = no_information_baseline(num_players, num_spies)
    print(f"  Identity accuracy: {sum(accuracies) / len(accuracies):.1%}"
        f"   over {len(scored)} games")
    print(f"    no-information baseline: {baseline:.1%}"
        f"   ({num_players} players, {num_spies} spies)")
    # These two are equal once every agent has answered the closing question.
    print(f"    the scored beliefs saw {sum(seen) / len(seen):.1f}"
        f" of {sum(missions) / len(missions):.1f} missions")

    if not beliefs:
        print("    (no beliefs.csv next to the summary, so no per agent breakdown)")
        return
    print()
    print_identity_by_agent(beliefs, baseline)
    print()
    print_identity_curve(beliefs)


def print_identity_by_agent(beliefs: list[dict], baseline: float) -> None:
    """each agent type's accuracy on its closing guess, against the baseline

    n is the number of (game, player) closing guesses the agent type
    contributed, so one agent per game makes it the game count.
    """
    by_agent = defaultdict(list)
    for row in final_snapshots(beliefs).values():
        by_agent[row["agent"]].append(row["accuracy"])

    dropped = count_games(beliefs, crashed=True)
    print(f"  Identity accuracy per agent, on its closing guess"
        f" ({count_games(beliefs, crashed=False)} finished games,"
        f" {dropped} crashed games left out):")
    for agent in sorted(by_agent):
        scores = by_agent[agent]
        mean = sum(scores) / len(scores)
        verdict = "above" if mean > baseline else "at or below"
        print(f"    {agent:<52} n={len(scores):>3}  {mean:>6.1%}  ({verdict} baseline)")


def print_identity_curve(beliefs: list[dict]) -> None:
    """accuracy against how many missions the belief had seen

    A method that reads the game gets better as the missions come in. One that
    does not can get worse, which the mean over a run hides.
    """
    counts = sorted({row["missions_seen"] for row in beliefs})
    by_agent = defaultdict(lambda: defaultdict(list))
    for row in beliefs:
        by_agent[row["agent"]][row["missions_seen"]].append(row["accuracy"])

    # Every snapshot counts here, a crashed game's included: a belief that had
    # seen two missions is comparable with any other that had seen two, whatever
    # the game did afterwards. Only the closing guess above needs the game over.
    header = "  Identity accuracy by missions seen (every snapshot):"
    print(header)
    line = f"    {'agent':<52}"
    for count in counts:
        line += f"  m={count}"
    print(line)
    for agent in sorted(by_agent):
        line = f"    {agent:<52}"
        for count in counts:
            scores = by_agent[agent][count]
            if scores:
                line += f"  {sum(scores) / len(scores):.2f}"
            else:
                line += "     -"
        print(line)


def print_llm_usage(rows: list[dict]) -> None:
    """what the run cost, over every game including the crashed ones"""
    calls = 0
    tokens = 0
    for row in rows:
        calls += to_int(row["llm_calls"])
        tokens += to_int(row["llm_total_tokens"])
    if calls == 0:
        return

    print("  LLM usage:")
    print(f"    calls:        {calls:>12,}")
    print(f"    tokens:       {tokens:>12,}")
    print(f"    tokens/call:  {tokens / calls:>12,.0f}")
    print(f"    calls/game:   {calls / len(rows):>12,.1f}")
    print()


def print_durations(rows: list[dict]) -> None:
    """how long the games took, which is what decides how many a run can afford"""
    durations = []
    for row in rows:
        value = to_float(row["duration_seconds"])
        if value is not None:
            durations.append(value)
    if not durations:
        return

    durations.sort()
    # An LLM game's length swings with how long the model thinks, so the spread
    # matters as much as the mean when a run is being budgeted.
    spread = statistics.stdev(durations) if len(durations) >= 2 else 0.0
    print("  Duration:")
    print(f"    mean:   {sum(durations) / len(durations):>8.1f}s")
    print(f"    std:    {spread:>8.1f}s")
    print(f"    min:    {durations[0]:>8.1f}s")
    print(f"    median: {durations[len(durations) // 2]:>8.1f}s")
    print(f"    max:    {durations[-1]:>8.1f}s")
    print(f"    total:  {sum(durations) / 3600:>8.2f}h  over {len(durations)} games")


def save_single_charts(played: list[dict], beliefs: list[dict], num_players: int, \
    num_spies: int, csv_path: str) -> None:
    """the win rate as the run went, and the accuracy curve per agent"""
    charts_dir = os.path.join(os.path.dirname(csv_path), "charts")
    os.makedirs(charts_dir, exist_ok=True)

    games = []
    rates = []
    wins = 0
    for index, row in enumerate(played, start=1):
        if row["winner"] == "resistance":
            wins += 1
        games.append(index)
        rates.append(wins / index)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(games, rates, color=COLOURS[0], linewidth=2)
    ax.axhline(y=rates[-1], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Games finished")
    ax.set_ylabel("Resistance win rate")
    ax.set_title(f"Resistance win rate over the run (n={len(played)})")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "resistance_win_rate.png"), dpi=150)
    plt.close()

    if beliefs:
        save_identity_chart(beliefs, num_players, num_spies, charts_dir)
    print(f"  Charts saved to: {charts_dir}/")


def save_identity_chart(beliefs: list[dict], num_players: int, num_spies: int, \
    charts_dir: str) -> None:
    """one line per agent: accuracy against the missions its belief had seen"""
    counts = sorted({row["missions_seen"] for row in beliefs})
    by_agent = defaultdict(lambda: defaultdict(list))
    for row in beliefs:
        by_agent[row["agent"]][row["missions_seen"]].append(row["accuracy"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for index, agent in enumerate(sorted(by_agent)):
        xs = []
        ys = []
        for count in counts:
            scores = by_agent[agent][count]
            if scores:
                xs.append(count)
                ys.append(sum(scores) / len(scores))
        ax.plot(xs, ys, marker="o", linewidth=2, label=agent, \
            color=COLOURS[index % len(COLOURS)])

    baseline = no_information_baseline(num_players, num_spies)
    ax.axhline(y=baseline, color="gray", linestyle="--", \
        label=f"no information ({baseline:.0%})")
    ax.set_xlabel("Missions the belief had seen")
    ax.set_ylabel("Identity accuracy")
    ax.set_title("Identity accuracy as the missions came in")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "identity_accuracy.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
#  Batch experiment
# ══════════════════════════════════════════════════════════════

def analyze_batch(rows: list[dict], csv_path: str) -> None:
    """one grid's batch_summary.csv, one row per cell"""
    print()
    print("=" * 64)
    print("  RESISTANCE BATCH EXPERIMENT")
    print("=" * 64)
    print(f"  Source: {csv_path}")
    print(f"  Cells:  {len(rows)}")
    print()

    cells = []
    for row in rows:
        played = to_int(row["games_played"])
        wins = to_int(row["resistance_wins"])
        low, high = wilson_interval(wins, played)
        cells.append({
            "label": row["combination"],
            "played": played,
            "crashed": to_int(row["games_crashed"]),
            "wins": wins,
            "rate": wins / played if played else 0.0,
            "low": low,
            "high": high,
            "identity": to_float(row["identity_accuracy"]),
            "calls": to_int(row["llm_calls"]),
            "tokens": to_int(row["llm_total_tokens"]),
            "duration": to_float(row["duration_seconds"]) or 0.0,
        })
    cells.sort(key=lambda cell: -cell["rate"])
    # A cell's label carries every setting the grid swept, so it runs to about
    # 90 characters and no column can hold it. Each cell gets a numbered line of
    # its own and its numbers on the line below, and the charts use the number.
    for number, cell in enumerate(cells, start=1):
        cell["number"] = number

    print("  Cells by resistance win rate:")
    for cell in cells:
        identity = "-" if cell["identity"] is None else f"{cell['identity']:.1%}"
        print(f"    [{cell['number']}] {cell['label']}")
        print(f"        resistance {cell['wins']}/{cell['played']} ({cell['rate']:.1%})"
            f"   95% CI [{cell['low']:.1%}, {cell['high']:.1%}]"
            f"   identity {identity}"
            f"   crashed {cell['crashed']}")
    print()

    print_batch_cost(cells)
    print()
    print_batch_durations(cells)
    beliefs = load_beliefs(csv_path, "batch_beliefs.csv")
    if beliefs:
        print()
        print_batch_identity(beliefs, cells)
    save_batch_charts(cells, csv_path)


def print_batch_cost(cells: list[dict]) -> None:
    """what the grid cost, and which cells carried it"""
    calls = 0
    tokens = 0
    for cell in cells:
        calls += cell["calls"]
        tokens += cell["tokens"]
    if calls == 0:
        return

    print("  LLM usage:")
    print(f"    calls:   {calls:>12,}")
    print(f"    tokens:  {tokens:>12,}")
    for cell in cells:
        if cell["calls"]:
            print(f"    [{cell['number']}] {cell['calls']:>8,} calls, {cell['tokens']:>12,} tokens,"
                f" {cell['tokens'] / cell['calls']:>6,.0f} tokens/call")


def print_batch_durations(cells: list[dict]) -> None:
    """how long the grid took, and what one game costs in each cell

    The per game figure is what a bigger grid is planned from: one cell that
    takes minutes a game decides how long the whole run will be. It counts the
    crashed games too, because they took their time before they raised.
    """
    total = 0.0
    for cell in cells:
        total += cell["duration"]
    if total == 0.0:
        return

    print("  Duration:")
    print(f"    total:  {total / 3600:.2f}h")
    for cell in cells:
        games = cell["played"] + cell["crashed"]
        per_game = cell["duration"] / games if games else 0.0
        print(f"    [{cell['number']}] {cell['duration'] / 60:>8.1f} min"
            f"   {per_game:>7.1f}s per game over {games} games")


def print_batch_identity(beliefs: list[dict], cells: list[dict]) -> None:
    """each cell's agents, scored at their last look

    The cell's own identity_accuracy mixes every agent in the line-up, so a
    mixed cell needs this breakdown before it says anything about a method.
    """
    numbers = {}
    for cell in cells:
        numbers[cell["label"]] = cell["number"]

    by_cell = defaultdict(list)
    for row in final_snapshots(beliefs).values():
        by_cell[(row.get("combination", ""), row["agent"])].append(row["accuracy"])

    print(f"  Identity accuracy per cell and agent, on its closing guess"
        f" ({count_games(beliefs, crashed=True)} crashed games left out):")
    for key in sorted(by_cell, key=lambda k: (numbers.get(k[0], 0), k[1])):
        combination, agent = key
        scores = by_cell[key]
        print(f"    [{numbers.get(combination, '?')}] {agent:<52} n={len(scores):>3}"
            f"  {sum(scores) / len(scores):>6.1%}")


def save_batch_charts(cells: list[dict], csv_path: str) -> None:
    """the win rate per cell, with the interval each cell's game count supports"""
    charts_dir = os.path.join(os.path.dirname(csv_path), "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # The cell numbers, not the labels: a label runs to about 90 characters and
    # would leave no room for the bars. The report above says which is which.
    labels = []
    rates = []
    lows = []
    highs = []
    for cell in cells:
        labels.append(f"[{cell['number']}]")
        rates.append(cell["rate"])
        lows.append(cell["rate"] - cell["low"])
        highs.append(cell["high"] - cell["rate"])

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.5 * len(cells)))
    positions = range(len(cells))
    ax.barh(positions, rates, color=COLOURS[0], xerr=[lows, highs], \
        error_kw={"ecolor": "gray", "capsize": 3})
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Resistance win rate (95% CI)")
    ax.set_xlim(0, 1.0)
    ax.set_title("Resistance win rate per cell")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "resistance_win_rate_by_cell.png"), dpi=150)
    plt.close()
    print(f"\n  Charts saved to: {charts_dir}/")


# ══════════════════════════════════════════════════════════════
#  Type posteriors (belief_model: bayes)
# ══════════════════════════════════════════════════════════════

def run_seats(run_dir: str) -> list[dict]:
    """this run's line-up by player id, turned as its trial turned it, from its config.yaml"""
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return turned_seats(config, config["game"]["seed"])["agents"]["seats"]


def player_labels(run_dir: str) -> dict[int, str]:
    """the agent that held each player id in this run"""
    labels = {}
    for player_id, entry in enumerate(run_seats(run_dir)):
        labels[player_id] = agent_label(entry)
    return labels


def analyze_types(memory_path: str) -> None:
    """every other player's type posterior, game by game, from one run's memory.json"""
    with open(memory_path, encoding="utf-8") as f:
        memory = json.load(f)
    labels = player_labels(os.path.dirname(memory_path))
    players = sorted(memory["type_posteriors"], key=int)

    print(f"\n  Type posterior after {memory['games_recorded']} games")
    for player in players:
        chances = []
        for name in TYPE_MODELS:
            chances.append(f"{name} {memory['type_posteriors'][player].get(name, 0.0):.3f}")
        print(f"    Player {player}  {labels[int(player)]:<24} {'  '.join(chances)}")

    # Game 0 is the prior, uniform over the types.
    games = [0]
    for snapshot in memory["type_history"]:
        games.append(snapshot["games_recorded"])
    fig, axes = plt.subplots(len(players), 1, figsize=(8, 2 * len(players)), sharex=True)
    for ax, player in zip(axes, players):
        for index, name in enumerate(TYPE_MODELS):
            chances = [1.0 / len(TYPE_MODELS)]
            for snapshot in memory["type_history"]:
                chances.append(snapshot["posteriors"][player].get(name, 0.0))
            ax.plot(games, chances, linewidth=2, label=name, color=LINE_COLOURS[index], \
                linestyle=LINE_STYLES[index])
        ax.set_ylim(0, 1.05) # above 1, so a line at 1 is not hidden by the frame
        ax.set_ylabel("P(type)")
        ax.set_title(f"Player {player}: {labels[int(player)]}", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Games scored")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    charts_dir = os.path.join(os.path.dirname(memory_path), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    plt.savefig(os.path.join(charts_dir, "type_posteriors.png"), dpi=150)
    plt.close()
    print(f"  Chart saved to: {charts_dir}/type_posteriors.png")


# ══════════════════════════════════════════════════════════════
#  Learning curves (every run under the directories given)
# ══════════════════════════════════════════════════════════════

def run_dirs(roots: list[str], name: str) -> list[str]:
    """every run directory under these directories that holds this file and a config.yaml"""
    found = []
    for root in roots:
        for path, _, files in sorted(os.walk(root)):
            if name in files and "config.yaml" in files:
                found.append(path)
    return found


def trend(points: list[tuple[int, float]]) -> dict:
    """how one measure moves with the game index

        slope = sum of (i - mean i)(v - mean v) / sum of (i - mean i)^2, the least squares line
            i   a game index
            v   the measure in that game
        error = sqrt(s1^2 / n1 + s2^2 / n2), the standard error of second half - first half
            s1, s2  the standard deviation of v in each half
            n1, n2  the games in each half

    The halves split the game indices at their midpoint. The error treats every
    game as independent, which games in one run are not quite.
    """
    indices = []
    for index, _ in points:
        indices.append(index)
    middle = (min(indices) + max(indices) + 1) / 2
    first = []
    second = []
    for index, value in points:
        if index < middle:
            first.append(value)
        else:
            second.append(value)
    mean_index = statistics.mean(indices)
    mean_value = statistics.mean(value for _, value in points)
    spread = 0.0 # sum of (i - mean i)^2
    together = 0.0 # sum of (i - mean i)(v - mean v)
    for index, value in points:
        spread += (index - mean_index) ** 2
        together += (index - mean_index) * (value - mean_value)
    result = {"mean": mean_value, "first": None, "second": None, "error": None, \
        "slope": together / spread if spread else 0.0}
    if len(first) > 1 and len(second) > 1:
        result["first"] = statistics.mean(first)
        result["second"] = statistics.mean(second)
        result["error"] = math.sqrt(statistics.variance(first) / len(first) \
            + statistics.variance(second) / len(second))
    return result


def outcome_learning_curve(roots: list[str]) -> None:
    """the tested player's results against the game index, one line per agent it is

    The tested player is the one entry pinned to faction: resistance. Two panels,
    each averaged over the runs at every game index (thin) and then over the five
    games up to it (thick):
        Resistance win rate
        the tested player's identity accuracy at the end of the game
    Crashed games are left out rather than counted as losses.
    """
    measures = {"Resistance win rate": defaultdict(list), "Identity accuracy": defaultdict(list)}
    runs = Counter() # label -> runs
    for run_dir in run_dirs(roots, "summary.csv"):
        seats = run_seats(run_dir)
        pinned = []
        for player_id, entry in enumerate(seats):
            if entry.get("faction") == Team.RESISTANCE.value:
                pinned.append(player_id)
        if len(pinned) != 1:
            raise ValueError(f"{run_dir}: the tested player is the one entry pinned to "
                f"faction: resistance, and this line-up pins {len(pinned)}")
        tested = pinned[0]
        label = agent_label(seats[tested])
        runs[label] += 1
        summary_path = os.path.join(run_dir, "summary.csv")
        rows = load_csv(summary_path)
        beliefs = load_beliefs(summary_path, "beliefs.csv")
        mark_crashed_from_summary(beliefs, rows)
        last = final_snapshots(beliefs)
        for row in rows:
            if row["crashed"] == "True":
                continue
            game = int(row["game_index"])
            measures["Resistance win rate"][label].append((game, float(row["winner"] == Team.RESISTANCE.value)))
            snapshot = last.get(("", row["game_index"], str(tested)))
            if snapshot is not None and snapshot["accuracy"] is not None:
                measures["Identity accuracy"][label].append((game, snapshot["accuracy"]))
    if not runs:
        print(f"No run with a summary.csv under: {', '.join(roots)}")
        return
    labels = sorted(runs)
    if len(labels) > len(LINE_COLOURS):
        raise ValueError(f"{len(labels)} different tested agents ({labels}); one chart holds "
            f"{len(LINE_COLOURS)}, so pass fewer directories")

    print()
    print("=" * 60)
    print("  LEARNING CURVES: the tested player, game by game")
    print("=" * 60)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (name, by_label) in zip(axes, measures.items()):
        print(f"\n  {name}")
        for index, label in enumerate(labels):
            points = by_label[label]
            if not points:
                continue
            by_game = defaultdict(list)
            for game, value in points:
                by_game[game].append(value)
            games = sorted(by_game)
            means = []
            for game in games:
                means.append(statistics.mean(by_game[game]))
            rolling = [] # the mean over the five games up to each one
            for position in range(len(means)):
                rolling.append(statistics.mean(means[max(0, position - 4):position + 1]))
            moved = trend(points)
            halves = "n/a"
            if moved["first"] is not None:
                halves = (f"{moved['first']:.1%} -> {moved['second']:.1%} "
                    f"({moved['second'] - moved['first']:+.1%} +- {moved['error']:.1%})")
            print(f"    {label:<20} {runs[label]} runs, {len(points)} games: overall {moved['mean']:.1%}, "
                f"halves {halves}, slope {moved['slope']:+.4f} per game")
            ax.plot(games, means, color=LINE_COLOURS[index], linestyle=LINE_STYLES[index], linewidth=1, \
                alpha=0.35)
            ax.plot(games, rolling, color=LINE_COLOURS[index], linestyle=LINE_STYLES[index], linewidth=2, \
                label=f"{label} (n={runs[label]} runs)")
        ax.set_title(name)
        ax.set_xlabel("Game index within a run")
        ax.set_ylim(0, 1.05) # above 1, so a line at 1 is not hidden by the frame
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean over runs (thin), over 5 games (thick)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(roots[0], "learning_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Chart saved to: {out_path}")


def type_learning_curve(roots: list[str]) -> None:
    """how fast the Bayes player's memory finds each player's type, over every memory.json

    Each other player is scored on the type built for its agent (modelled_type).
    Two panels against the games scored, one line per type, pooled over the
    players of that type in every run:
        P(that type | the games scored), with the lowest and highest run as a
            band, each run averaged over its own players
        the chance the most likely type is that one, ties split evenly, so the
            uniform start reads 1/3
    """
    # type -> games scored -> one (run, P(type), share of the top) per player
    by_type = defaultdict(lambda: defaultdict(list))
    players = Counter() # type -> players scored on it, over all runs
    uniform = dict.fromkeys(TYPE_MODELS, 1.0 / len(TYPE_MODELS))
    runs = [] # finished runs only: a run stopped part way has written memory.json but no summary.csv
    for run_dir in run_dirs(roots, "memory.json"):
        if os.path.isfile(os.path.join(run_dir, "summary.csv")):
            runs.append(run_dir)
    for run, run_dir in enumerate(runs):
        with open(os.path.join(run_dir, "memory.json"), encoding="utf-8") as f:
            memory = json.load(f)
        seats = run_seats(run_dir)
        for player in memory["type_posteriors"]:
            built_for = modelled_type(seats[int(player)])
            if built_for is None:
                continue
            players[built_for] += 1
            posteriors = [uniform] # before any game is scored
            for snapshot in memory["type_history"]:
                posteriors.append(snapshot["posteriors"][player])
            for scored, posterior in enumerate(posteriors):
                top = max(posterior.values())
                tied = [] # the types at the top
                for type_name, probability in posterior.items():
                    if probability == top:
                        tied.append(type_name)
                top_share = 1.0 / len(tied) if built_for in tied else 0.0
                by_type[built_for][scored].append((run, posterior[built_for], top_share))
    if not runs:
        print(f"\n  No memory.json under: {', '.join(roots)}")
        return

    print()
    print("=" * 60)
    print(f"  TYPE LEARNING: {len(runs)} runs with a memory.json")
    print("=" * 60)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for index, type_name in enumerate(TYPE_MODELS):
        by_scored = by_type[type_name]
        if not by_scored:
            continue
        scored_counts = sorted(by_scored)
        means = []
        lows = []
        highs = []
        top_shares = []
        for scored in scored_counts:
            by_run = defaultdict(list)
            for run, probability, _ in by_scored[scored]:
                by_run[run].append(probability)
            run_means = []
            for values in by_run.values():
                run_means.append(statistics.mean(values))
            means.append(statistics.mean(probability for _, probability, _ in by_scored[scored]))
            lows.append(min(run_means))
            highs.append(max(run_means))
            top_shares.append(statistics.mean(share for _, _, share in by_scored[scored]))
        colour = LINE_COLOURS[index]
        style = LINE_STYLES[index]
        label = f"{type_name} ({players[type_name]} players)"
        left.fill_between(scored_counts, lows, highs, color=colour, alpha=0.15, linewidth=0)
        left.plot(scored_counts, means, color=colour, linestyle=style, linewidth=2, label=label)
        right.plot(scored_counts, top_shares, color=colour, linestyle=style, linewidth=2, label=label)
        checkpoints = sorted({1, scored_counts[-1] // 2, scored_counts[-1]})
        readings = []
        for scored in checkpoints:
            position = scored_counts.index(scored)
            readings.append(f"after {scored}: {means[position]:.3f} (top {top_shares[position]:.0%})")
        print(f"  {type_name:<11} {players[type_name]} players   P(type) {'   '.join(readings)}")
    for ax in (left, right):
        ax.axhline(y=1.0 / len(TYPE_MODELS), color="gray", linestyle="--", alpha=0.5, label="uniform start")
        ax.set_xlabel("Games scored")
        ax.set_ylim(0, 1.05) # above 1, so a line at 1 is not hidden by the frame
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    left.set_title("Posterior probability of the type built for the player\nband: lowest and highest run")
    right.set_title("MAP classified accuracy")
    left.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(roots[0], "type_learning_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Chart saved to: {out_path}")


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_resistance.py <csv_file>")
        print()
        print("  Single experiment: python3 analyze_resistance.py results/<run>/summary.csv")
        print("  Batch experiment:  python3 analyze_resistance.py "
            "results/batch_resistance_<ts>/batch_summary.csv")
        print("  Type posteriors:   python3 analyze_resistance.py results/<run>/memory.json")
        print("  Learning curves:   python3 analyze_resistance.py --learning-curve results/<dir> ...")
        sys.exit(1)

    if sys.argv[1] == "--learning-curve":
        if len(sys.argv) < 3:
            print("Error: --learning-curve takes at least one directory of runs.")
            sys.exit(1)
        outcome_learning_curve(sys.argv[2:])
        type_learning_curve(sys.argv[2:])
        return

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"Error: '{csv_path}' not found.")
        sys.exit(1)
    if csv_path.endswith(".json"):
        analyze_types(csv_path)
        return

    rows = load_csv(csv_path)
    if not rows:
        print("Error: the CSV has no rows.")
        sys.exit(1)

    if is_batch_csv(rows):
        analyze_batch(rows, csv_path)
    else:
        analyze_single(rows, csv_path)


if __name__ == "__main__":
    main()
