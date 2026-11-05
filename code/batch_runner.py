"""
Tofu Kingdom - Batch Experiment Runner

Reads experiment_grid.yaml, expands all parameter combinations,
and runs each one as a separate experiment.

Usage:
    python batch_runner.py experiment_grid.yaml
    python batch_runner.py experiment_grid.yaml --dry-run    # preview combinations only
    python batch_runner.py experiment_grid.yaml --parallel 4  # run 4 experiments in parallel
"""

from __future__ import annotations
import sys
import os
import yaml
import time
import itertools
from copy import deepcopy
from datetime import datetime

from main import run_experiment, print_summary
from game.predefined_data_classes import GameResult, Faction
from core.logger import save_summary_csv


def load_grid_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_nested(config: dict, dot_path: str, value) -> None:
    """Set a value in a nested dict using dot notation.

    Example: set_nested(config, "llm.temperature", 0.5)
    → config["llm"]["temperature"] = 0.5

    A numeric step indexes a list, so one agent of a line-up can be swept
    without writing the whole line-up out per cell:
    set_nested(config, "agents.resistance.0.beta", 2.5)
    → config["agents"]["resistance"][0]["beta"] = 2.5
    """
    keys = dot_path.split(".")
    d = config
    for key in keys[:-1]:
        if isinstance(d, list):
            d = d[int(key)]
            continue
        if key not in d:
            d[key] = {}
        d = d[key]
    if isinstance(d, list):
        d[int(keys[-1])] = value
    else:
        d[keys[-1]] = value


def expand_grid(grid_config: dict) -> list[dict]:
    """Expand sweep parameters into all combinations.

    Returns:
        List of (combo_dict, full_config) tuples where:
        - combo_dict: {param_path: value} for this combination
        - full_config: complete config ready to pass to run_experiment
    """
    base = deepcopy(grid_config["base"])
    sweep = grid_config.get("sweep", {})

    if not sweep:
        return [({}, base)]

    # Build list of (param_path, values_list) pairs
    param_paths = list(sweep.keys())
    param_values = [sweep[p] for p in param_paths]

    # Cartesian product
    combinations = []
    for combo_vals in itertools.product(*param_values):
        combo_dict = dict(zip(param_paths, combo_vals))
        config = deepcopy(base)

        # Apply this combination's values
        for path, val in combo_dict.items():
            set_nested(config, path, val)

        # Ensure llm section has minimum required fields
        _ensure_llm_defaults(config)

        combinations.append((combo_dict, config))

    return combinations


def _ensure_llm_defaults(config: dict) -> None:
    """Fill in default LLM settings if not specified."""
    llm = config.setdefault("llm", {})
    llm.setdefault("base_url", "http://localhost:11434/v1")
    llm.setdefault("api_key", "ollama")
    llm.setdefault("max_retries", 3)
    llm.setdefault("temperature", 0.7)
    llm.setdefault("top_p", 0.9)
    llm.setdefault("max_tokens", -1)  # output cap (-> Ollama num_predict); -1 = unlimited
    llm.setdefault("reasoning_effort", "high")  # thinking: "none" = off, "low"/"medium"/"high" = on
    # NOTE: num_ctx / repeat_penalty / keep_alive are NOT set here — Ollama's /v1
    # endpoint ignores them. Bake num_ctx/repeat_penalty into a model variant
    # (Modelfile.template); set keep_alive via OLLAMA_KEEP_ALIVE env var.
    # NOTE: llm.reasoning_history is NOT set here. main.py already defaults it
    # when it reads it, and writing it here put the key into every config the
    # grid produced — including The Resistance's, which does not read it, where
    # sweeping it would have run one cell twice and reported it as two.


def combo_label(combo: dict) -> str:
    """Short human-readable label for a parameter combination."""
    parts = []
    for path, val in combo.items():
        key_parts = path.split(".")
        # Use parent.child for generic keys like "type" or "enabled"
        if key_parts[-1] in ("type", "enabled") and len(key_parts) >= 2:
            short_key = key_parts[-2]  # e.g. "prince" from "agents.prince.type"
        else:
            short_key = key_parts[-1]

        # Agent player list: homogeneous [{type: llm}] → "llm"; a MIXED list is
        # run-length encoded in order (e.g. [random,random,llm,llm] → "random2-llm2")
        # so distinct mixes get distinct labels/folders and never collide.
        if isinstance(val, list) and len(val) > 0 and all(isinstance(x, dict) and "type" in x for x in val):
            types = [x["type"] for x in val]
            if len(set(types)) == 1:
                # An entry may carry settings beyond its type, and two cells that
                # run the same agent with different settings would otherwise share
                # a label. Read them off the first entry, which is the one every
                # seat of a homogeneous list is built from. The separator is ":"
                # and not "=", because complete_batch_results.label_params splits
                # a label on "=" and needs the values to hold none.
                settings = []
                for key in sorted(val[0]):
                    if key != "type":
                        settings.append(f"{key}:{val[0][key]}")
                suffix = "(" + "-".join(settings) + ")" if settings else ""
                parts.append(f"{short_key}={types[0]}{suffix}")
            else:
                runs: list[list] = []
                for t in types:
                    if runs and runs[-1][0] == t:
                        runs[-1][1] += 1
                    else:
                        runs.append([t, 1])
                mix = "-".join(t if n == 1 else f"{t}{n}" for t, n in runs)
                parts.append(f"{short_key}={mix}")
        elif isinstance(val, bool):
            parts.append(f"{short_key}={'Y' if val else 'N'}")
        elif isinstance(val, float):
            parts.append(f"{short_key}={val}")
        else:
            parts.append(f"{short_key}={val}")
    return "_".join(parts) if parts else "baseline"


def run_batch(grid_path: str, dry_run: bool = False) -> None:
    """Main entry point: load grid, expand, run all."""
    grid_config = load_grid_config(grid_path)
    combinations = expand_grid(grid_config)

    print("=" * 70)
    print("  TOFU KINGDOM BATCH EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Grid config: {grid_path}")
    print(f"  Total combinations: {len(combinations)}")

    num_games = grid_config["base"]["game"].get("num_games", 1)
    print(f"  Games per combination: {num_games}")
    print(f"  Total games: {len(combinations) * num_games}")
    print()

    # Preview all combinations
    print("  Combinations:")
    for i, (combo, _) in enumerate(combinations):
        print(f"    [{i+1}] {combo_label(combo)}")
    print()

    if dry_run:
        print("  --dry-run: no experiments will be run.")
        print()
        # Print one sample config for inspection
        print("  Sample config for combination [1]:")
        print("  " + "-" * 40)
        sample_config = combinations[0][1]
        for line in yaml.dump(sample_config, default_flow_style=False).split("\n"):
            print(f"    {line}")
        return

    # Run each combination
    all_experiment_results = []
    batch_start = time.time()

    for i, (combo, config) in enumerate(combinations):
        label = combo_label(combo)
        print()
        print("=" * 70)
        print(f"  Experiment [{i+1}/{len(combinations)}]: {label}")
        print("=" * 70)

        # Common random numbers (CRN): every combination shares the SAME base seed,
        # so game g draws the identical deal and the identical per-agent random
        # streams across all combinations -- only the swept factor differs. The
        # SeedSource tree (see helper_functions / create_agents_from_config) keeps
        # each agent's stream isolated, so toggling one agent's strategy never shifts
        # another's draws. This pairs combinations on the same games for low-variance
        # comparison (instead of offsetting the seed per combination).
        config["game"]["seed"] = config["game"].get("seed", 42000)

        exp_start = time.time()

        try:
            results, usage = run_experiment(config, return_stats=True)
            exp_duration = time.time() - exp_start

            # Collect summary
            n = len(results)
            wins = {Faction.PRINCESS: 0, Faction.QUEEN: 0, Faction.OTHER: 0}
            for r in results:
                wins[r.winner_faction] += 1

            correct = wins[Faction.PRINCESS]
            accuracy = correct / n if n > 0 else 0

            summary = {
                "combination": label,
                "params": combo,
                "num_games": num_games,
                "completed": n,
                "failed": num_games - n,
                "princess_wins": wins[Faction.PRINCESS],
                "queen_wins": wins[Faction.QUEEN],
                "other_wins": wins[Faction.OTHER],
                "prince_accuracy": round(accuracy, 4),
                "duration_seconds": round(exp_duration, 1),
                # LLM / soft-likelihood usage for this combination (0 for symbolic-only)
                "llm_calls": usage["llm"]["total_calls"],
                "llm_retries": usage["llm"]["retry_count"],
                "llm_total_tokens": usage["llm"]["total_tokens"],
                "lk_queries": usage["likelihood"]["queries"],
                "lk_judge_calls": usage["likelihood"]["judge_calls"],
                "lk_cache_hit_rate": usage["likelihood"]["cache_hit_rate"],
                "lk_defaulted_slots": usage["likelihood"]["defaulted_slots"],
            }
            all_experiment_results.append(summary)

            print(f"\n  → Accuracy: {correct}/{n} ({accuracy:.1%}), " f"Duration: {exp_duration:.1f}s")

        except Exception as e:
            exp_duration = time.time() - exp_start
            print(f"\n  [ERROR] Experiment failed: {e}")
            all_experiment_results.append({
                "combination": label,
                "params": combo,
                "num_games": num_games,
                "completed": 0,
                "failed": num_games,
                "error": str(e),
                "duration_seconds": round(exp_duration, 1),
            })

    batch_duration = time.time() - batch_start

    # ── Print final summary ──
    print()
    print("=" * 70)
    print("  BATCH RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total time: {batch_duration:.0f}s ({batch_duration/60:.1f}m)")
    print()

    # Sort by accuracy (best first)
    scored = [r for r in all_experiment_results if "prince_accuracy" in r]
    scored.sort(key=lambda x: x["prince_accuracy"], reverse=True)

    print(f"  {'Rank':<5} {'Combination':<45} {'Accuracy':>10} {'P/Q/O':>12} {'Time':>8}")
    print(f"  {'-'*5} {'-'*45} {'-'*10} {'-'*12} {'-'*8}")

    for rank, r in enumerate(scored, 1):
        acc = f"{r['prince_accuracy']:.1%}"
        pqo = f"{r['princess_wins']}/{r['queen_wins']}/{r['other_wins']}"
        dur = f"{r['duration_seconds']:.0f}s"
        marker = " ★" if rank == 1 else ""
        print(f"  {rank:<5} {r['combination']:<45} {acc:>10} {pqo:>12} {dur:>8}{marker}")

    # Print failed experiments if any
    failed = [r for r in all_experiment_results if "error" in r]
    if failed:
        print()
        print(f"  Failed experiments: {len(failed)}")
        for r in failed:
            print(f"    {r['combination']}: {r.get('error', 'unknown')}")

    # Save batch summary (YAML)
    output_dir = grid_config["base"].get("logging", {}).get("output_dir", "results")
    batch_summary_path = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    os.makedirs(output_dir, exist_ok=True)

    batch_summary = {
        "grid_config": grid_path,
        "total_combinations": len(combinations),
        "total_games": len(combinations) * num_games,
        "total_duration_seconds": round(batch_duration, 1),
        "results": all_experiment_results,
        "best": scored[0] if scored else None,
    }

    with open(batch_summary_path, "w") as f:
        yaml.dump(batch_summary, f, default_flow_style=False)
    print(f"\n  Batch summary (YAML): {batch_summary_path}")

    # Save batch results CSV (one row per combination, params as columns)
    batch_csv_path = os.path.join(output_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    _save_batch_csv(batch_csv_path, all_experiment_results)
    print(f"  Batch results (CSV):  {batch_csv_path}")
    print(f"\n  To analyze: python analyze.py {batch_csv_path}")
    print()


def _save_batch_csv(path: str, results: list[dict]) -> None:
    """Save batch results as a flat CSV with params as columns.

    Each row = one parameter combination.
    Param columns are named by their dot-path (e.g. "llm.temperature").
    Agent list params are flattened (e.g. players=[{type:llm}] → "players=llm").
    """
    import csv

    if not results:
        return

    # Collect all param keys across all results
    all_param_keys = set()
    for r in results:
        if "params" in r:
            all_param_keys.update(r["params"].keys())
    param_keys = sorted(all_param_keys)

    # Build rows
    fieldnames = ["combination"] + param_keys + [
        "num_games", "completed", "failed",
        "prince_accuracy", "princess_wins", "queen_wins", "other_wins",
        "duration_seconds",
        "llm_calls", "llm_retries", "llm_total_tokens",
        "lk_queries", "lk_judge_calls", "lk_cache_hit_rate", "lk_defaulted_slots",
        "error",
    ]

    rows = []
    for r in results:
        row = {"combination": r.get("combination", "")}

        # Flatten params into columns
        params = r.get("params", {})
        for key in param_keys:
            val = params.get(key)
            # Flatten agent list: [{type: llm}] → "llm"
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and "type" in val[0]:
                row[key] = val[0]["type"]
            else:
                row[key] = val

        # Result columns
        row["num_games"] = r.get("num_games", "")
        row["completed"] = r.get("completed", "")
        row["failed"] = r.get("failed", "")
        row["prince_accuracy"] = r.get("prince_accuracy", "")
        row["princess_wins"] = r.get("princess_wins", "")
        row["queen_wins"] = r.get("queen_wins", "")
        row["other_wins"] = r.get("other_wins", "")
        row["duration_seconds"] = r.get("duration_seconds", "")
        for k in ("llm_calls", "llm_retries", "llm_total_tokens",
                  "lk_queries", "lk_judge_calls", "lk_cache_hit_rate", "lk_defaulted_slots"):
            row[k] = r.get(k, 0)
        row["error"] = r.get("error", "")
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_runner.py experiment_grid.yaml")
        print("  python batch_runner.py experiment_grid.yaml --dry-run")
        sys.exit(1)

    grid_path = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    if not os.path.isfile(grid_path):
        print(f"Error: '{grid_path}' not found.")
        sys.exit(1)

    run_batch(grid_path, dry_run=dry_run)


if __name__ == "__main__":
    main()
