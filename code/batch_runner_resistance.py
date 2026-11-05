# batch runner for Resistance experiments
#
#   python3 batch_runner_resistance.py <grid.yaml> [--dry-run]
#
# The grid machinery is shared with batch_runner.py, which runs Tofu Kingdom:
# load_grid_config, expand_grid, combo_label and the batch CSV writer are all
# game agnostic. Only the per-cell loop and the aggregation below are specific
# to this game, because they read the winner.
from __future__ import annotations
import csv
import os
import sys
import time
from datetime import datetime

from batch_runner import load_grid_config, expand_grid, combo_label
from game.resistance_data_classes import Team
from run_resistance import run_experiment


def save_batch_csv(path: str, results: list[dict]) -> None:
    """one row per grid cell, columns taken from the rows themselves

    batch_runner's writer names its columns after the Tofu Kingdom summary, so
    it drops every field this game reports. The rows here are already flat, so
    reading the columns off them keeps whatever summarise returns.
    """
    fieldnames = []
    for result in results:
        for name in result:
            if name not in fieldnames:
                fieldnames.append(name)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def summarise(rows: list[dict]) -> dict:
    """win counts over the games of one grid cell"""
    played = 0
    crashed = 0
    resistance_wins = 0
    accuracies = []
    llm_calls = 0
    llm_tokens = 0
    for row in rows:
        # A crashed game still cost whatever LLM calls it made before it raised.
        llm_calls += row["llm_calls"]
        llm_tokens += row["llm_total_tokens"]
        if row["crashed"]:
            crashed += 1
            continue
        played += 1
        if row["winner"] == Team.RESISTANCE.value:
            resistance_wins += 1
        if row["identity_accuracy"] is not None:
            accuracies.append(row["identity_accuracy"])

    return {
        "games_played": played,
        "games_crashed": crashed,
        "resistance_wins": resistance_wins,
        "spies_wins": played - resistance_wins,
        "resistance_win_rate": resistance_wins / played if played else 0.0,
        # blank when no agent in this cell holds a belief
        "identity_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "identity_games": len(accuracies),
        # zero when no player in this cell used an LLM
        "llm_calls": llm_calls,
        "llm_total_tokens": llm_tokens,
    }


def run_batch(grid_path: str, dry_run: bool = False) -> None:
    """load a grid, run every combination, write one row per combination"""
    grid_config = load_grid_config(grid_path)
    combinations = expand_grid(grid_config)
    num_games = grid_config["base"]["game"].get("num_games", 1)

    print("=" * 70)
    print("  RESISTANCE BATCH EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Grid config: {grid_path}")
    print(f"  Total combinations: {len(combinations)}")
    print(f"  Games per combination: {num_games}")
    print(f"  Total games: {len(combinations) * num_games}")

    if dry_run:
        for index, (combo, _) in enumerate(combinations, start=1):
            print(f"  [{index}] {combo_label(combo)}")
        return

    # Under the grid's own output_dir, not under results/ flat. The timestamp is
    # second granular and two grids started together share it, so a flat name
    # meant the second one to finish overwrote the first one's summary and said
    # nothing. A split runbook gives each file its own output_dir and they stop
    # colliding, which is what experiment_grids/consistency/README.md already
    # assumes of the Tofu runner.
    base_dir = grid_config["base"].get("logging", {}).get("output_dir", "results")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = os.path.join(base_dir, f"batch_resistance_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    results = []
    belief_rows = [] # every cell's belief snapshots, tagged with the cell
    for index, (combo, config) in enumerate(combinations, start=1):
        label = combo_label(combo)
        print("\n" + "-" * 70)
        print(f"  [{index}/{len(combinations)}] {label}")
        print("-" * 70)

        started = time.time()
        cell_beliefs = []
        rows = run_experiment(config, belief_rows=cell_beliefs)
        duration = time.time() - started

        for row in cell_beliefs:
            tagged = {"combination": label} # first column, so the cell reads left to right
            tagged.update(row)
            belief_rows.append(tagged)

        result = {"combination": label, "duration_seconds": round(duration, 1)}
        result.update(combo)
        result.update(summarise(rows))
        results.append(result)
        print(f"\n  -> resistance win rate {result['resistance_win_rate']:.1%} ({duration:.1f}s)")

    save_batch_csv(os.path.join(batch_dir, "batch_summary.csv"), results)
    # Every cell's belief snapshots in one table. Group by combination and
    # agent for a score per method, add missions_seen for the curve.
    save_batch_csv(os.path.join(batch_dir, "batch_beliefs.csv"), belief_rows)
    print(f"\nbatch summary written to {batch_dir}")


def main():
    if len(sys.argv) < 2:
        print("usage: python3 batch_runner_resistance.py <grid.yaml> [--dry-run]")
        sys.exit(1)
    run_batch(sys.argv[1], dry_run="--dry-run" in sys.argv)


if __name__ == "__main__":
    main()
