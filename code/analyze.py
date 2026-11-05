"""
Tofu Kingdom - Experiment Analyzer

Supports three modes:
  1. Single experiment:  python analyze.py results/<experiment>/summary.csv
  2. Batch experiment:   python analyze.py results/baseline_sweep/batch_results_xxx.csv
  3. Learning curve:     python analyze.py --learning-curve results/grid_memory
     (cross-game memory experiments: accuracy vs game index, averaged over the
      seed series found under the given directories, grouped by Prince config)

Batch mode auto-detected by checking for 'prince_accuracy' column (per-combination summary).
"""

from __future__ import annotations
import sys
import csv
import json
import os
import statistics
from collections import Counter, defaultdict

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.our_agents.opponent_memory import BUILT_FOR


# ══════════════════════════════════════════════════════════════
#  CSV Loading
# ══════════════════════════════════════════════════════════════

def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_batch_csv(rows: list[dict]) -> bool:
    """Detect batch results CSV by the presence of 'prince_accuracy' column."""
    return len(rows) > 0 and "prince_accuracy" in rows[0]


# ══════════════════════════════════════════════════════════════
#  Single Experiment Analysis (existing)
# ══════════════════════════════════════════════════════════════

def parse_single_row(row: dict) -> dict:
    """Normalize types from single-experiment CSV strings."""
    row["correct"] = row.get("correct", "").strip() == "True"
    for key in ("guessed_target", "actual_princess", "actual_queen", "num_players"):
        val = row.get(key, "")
        row[key] = int(val) if val not in ("", "None", None) else None
    val = row.get("duration_seconds", "")
    row["duration_seconds"] = float(val) if val not in ("", "None", None) else None
    return row


def analyze_single(rows: list[dict], csv_path: str):
    """Analyze a single experiment's summary.csv."""
    rows = [parse_single_row(r) for r in rows]
    completed = [r for r in rows if r["status"] == "completed"]
    failed = [r for r in rows if r["status"] == "failed"]
    n = len(completed)

    wins = Counter(r["winner_faction"] for r in completed)
    correct_count = sum(1 for r in completed if r["correct"])

    print()
    print("=" * 60)
    print("  SINGLE EXPERIMENT ANALYSIS")
    print("=" * 60)
    print(f"  Source: {csv_path}")
    print()

    print(f"  Total games:  {len(rows)}")
    print(f"  Completed:    {n}")
    print(f"  Failed:       {len(failed)} ({len(failed)/len(rows):.1%})")
    print()
    if n:
        print(f"  Prince accuracy: {correct_count}/{n} ({correct_count/n:.1%})")
    else:
        print("  No completed games.")
    print()
    print("  Win rates:")
    for faction in ["princess", "queen", "other"]:
        c = wins.get(faction, 0)
        if n:
            print(f"    {faction:>10}: {c:>4} wins ({c/n:.1%})")

    # Duration
    durations = [r["duration_seconds"] for r in completed if r["duration_seconds"] is not None]
    if durations:
        print()
        print(f"  Game duration:")
        print(f"    Mean:   {sum(durations)/len(durations):>8.1f}s")
        # sample std dev: how much per-game time fluctuates (needs >= 2 games)
        std = statistics.stdev(durations) if len(durations) >= 2 else 0.0
        print(f"    Std:    {std:>8.1f}s")
        print(f"    Min:    {min(durations):>8.1f}s")
        print(f"    Max:    {max(durations):>8.1f}s")
        sorted_d = sorted(durations)
        median = sorted_d[len(sorted_d)//2]
        print(f"    Median: {median:>8.1f}s")
        print(f"    Total:  {sum(durations):>8.1f}s")

    # LLM usage (printed only if any LLM calls were made; columns added by logger)
    def _num(r, key, cast=int):
        v = r.get(key, "")
        try:
            return cast(v)
        except (ValueError, TypeError):
            return cast(0)

    per_game_calls = [_num(r, "llm_calls") for r in rows]
    llm_calls = sum(per_game_calls)
    if llm_calls > 0:
        retries = sum(_num(r, "llm_retries") for r in rows)
        tot_tok = sum(_num(r, "llm_total_tokens") for r in rows)
        prompt_tok = sum(_num(r, "llm_prompt_tokens") for r in rows)
        completion_tok = sum(_num(r, "llm_completion_tokens") for r in rows)
        active_calls = [c for c in per_game_calls if c > 0]
        active_tokens = [_num(r, "llm_total_tokens") for r in rows if _num(r, "llm_calls") > 0]
        n_llm_games = len(active_calls)
        print()
        print("  LLM usage:")
        print(f"    Total calls:       {llm_calls:>12,}")
        print(f"    Total retries:     {retries:>12,}   ({retries / llm_calls:.1%} of calls)")
        print(f"    Games with LLM:    {n_llm_games:>12} / {len(rows)}")
        print(f"    Total tokens:      {tot_tok:>12,}")
        if prompt_tok or completion_tok:
            # prompt / completion split (may be 0 in older CSVs without these columns)
            base = tot_tok or (prompt_tok + completion_tok) or 1
            print(f"      Prompt:          {prompt_tok:>12,}   ({prompt_tok / base:.0%})")
            print(f"      Completion:      {completion_tok:>12,}   ({completion_tok / base:.0%})")
        print(f"    Tokens/call:       {tot_tok / llm_calls:>12,.0f}")
        if n_llm_games:
            print(f"    Calls/game:        {llm_calls / n_llm_games:>12,.1f}   "
                  f"(min {min(active_calls)}, max {max(active_calls)})")
            print(f"    Tokens/game:       {tot_tok / n_llm_games:>12,.0f}   "
                  f"(min {min(active_tokens):,}, max {max(active_tokens):,})")
        if any(str(r.get('llm_token_tracking_complete')).lower() == 'false' for r in rows):
            print("    (NOTE: token counts under-counted — some API responses lacked usage data)")

    # Soft-likelihood judge usage (printed only for a soft_symbolic Prince)
    lk_queries = sum(_num(r, "lk_queries") for r in rows)
    if lk_queries > 0:
        rates = [_num(r, "lk_cache_hit_rate", float) for r in rows if _num(r, "lk_queries") > 0]
        judge_calls = sum(_num(r, "lk_judge_calls") for r in rows)
        print()
        print("  Soft-likelihood (judge):")
        print(f"    Total queries:        {lk_queries:>10,}")
        print(f"    Total judge LLM calls:{judge_calls:>10,}")
        if judge_calls:
            print(f"    Queries/judge call:   {lk_queries / judge_calls:>10,.1f}")
        if rates:
            print(f"    Mean cache hit rate:  {sum(rates)/len(rates):>10.1%}")
        print(f"    Defaulted slots:      {sum(_num(r, 'lk_defaulted_slots') for r in rows):>10,}"
              "   (LLM left a slot unrated; should stay ~0)")

    # Princess location
    center = [r for r in completed if r["actual_princess"] == -1]
    player = [r for r in completed if r["actual_princess"] is not None and r["actual_princess"] != -1]
    print()
    print("  Princess location impact:")
    for label, subset in [("With a player", player), ("In center", center)]:
        if subset:
            acc = sum(1 for r in subset if r["correct"])
            print(f"    {label}: {len(subset)} games, accuracy {acc}/{len(subset)} ({acc/len(subset):.1%})")
        else:
            print(f"    {label}: 0 games")
    print()

    # Charts
    if n > 0:
        _save_single_charts(completed, wins, csv_path)


def _save_single_charts(completed, wins, csv_path):
    charts_dir = os.path.join(os.path.dirname(csv_path), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    n = len(completed)

    # Win distribution pie
    fig, ax = plt.subplots(figsize=(6, 6))
    factions = ["princess", "queen", "other"]
    counts = [wins.get(f, 0) for f in factions]
    colors = ["#4CAF50", "#F44336", "#9E9E9E"]
    labels = [f"{f}\n{c} ({c/sum(counts):.0%})" for f, c in zip(factions, counts)]
    ax.pie(counts, labels=labels, colors=colors, startangle=90, textprops={"fontsize": 12})
    ax.set_title(f"Win Distribution (n={n})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "win_distribution.png"), dpi=150)
    plt.close()

    # Cumulative accuracy
    fig, ax = plt.subplots(figsize=(8, 4))
    cum = 0
    xs, ys = [], []
    for i, r in enumerate(completed, 1):
        cum += 1 if r["correct"] else 0
        xs.append(i)
        ys.append(cum / i)
    ax.plot(xs, ys, color="#2196F3", linewidth=2)
    ax.set_xlabel("Games Played")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Prince Accuracy Over Time")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=ys[-1], color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "accuracy_over_time.png"), dpi=150)
    plt.close()

    print(f"  Charts saved to: {charts_dir}/")


# ══════════════════════════════════════════════════════════════
#  Batch Analysis
# ══════════════════════════════════════════════════════════════

def parse_batch_row(row: dict) -> dict:
    """Normalize types from batch results CSV."""
    for key in ("num_games", "completed", "failed", "princess_wins", "queen_wins", "other_wins"):
        val = row.get(key, "")
        row[key] = int(val) if val not in ("", "None", None) else 0
    val = row.get("prince_accuracy", "")
    row["prince_accuracy"] = float(val) if val not in ("", "None", None) else 0.0
    val = row.get("duration_seconds", "")
    row["duration_seconds"] = float(val) if val not in ("", "None", None) else 0.0
    # LLM / soft-likelihood usage columns (present only for LLM/soft batches)
    for key in ("llm_calls", "llm_retries", "llm_total_tokens",
                "lk_queries", "lk_judge_calls", "lk_defaulted_slots"):
        val = row.get(key, "")
        row[key] = int(float(val)) if val not in ("", "None", None) else 0
    val = row.get("lk_cache_hit_rate", "")
    row["lk_cache_hit_rate"] = float(val) if val not in ("", "None", None) else 0.0
    return row


def detect_param_columns(rows: list[dict]) -> list[str]:
    """Find which columns are sweep parameters (not result columns)."""
    result_cols = {
        "combination", "num_games", "completed", "failed",
        "prince_accuracy", "princess_wins", "queen_wins", "other_wins",
        "duration_seconds", "error",
        "llm_calls", "llm_retries", "llm_total_tokens",
        "lk_queries", "lk_judge_calls", "lk_cache_hit_rate", "lk_defaulted_slots",
    }
    all_cols = list(rows[0].keys()) if rows else []
    return [c for c in all_cols if c not in result_cols]


def analyze_batch(rows: list[dict], csv_path: str):
    """Analyze batch experiment results."""
    rows = [parse_batch_row(r) for r in rows]
    param_cols = detect_param_columns(rows)

    total_games = sum(r["completed"] + r["failed"] for r in rows)
    total_completed = sum(r["completed"] for r in rows)
    total_failed = sum(r["failed"] for r in rows)
    total_duration = sum(r["duration_seconds"] for r in rows)

    print()
    print("=" * 80)
    print("  BATCH EXPERIMENT ANALYSIS")
    print("=" * 80)
    print(f"  Source: {csv_path}")
    print(f"  Combinations: {len(rows)}")
    print(f"  Total games: {total_games} ({total_completed} completed, {total_failed} failed)")
    print(f"  Total time: {total_duration:.0f}s ({total_duration/60:.1f}m)")

    # LLM usage across the whole batch (shown only if any LLM calls were made)
    batch_llm_calls = sum(r.get("llm_calls", 0) for r in rows)
    if batch_llm_calls > 0:
        batch_tokens = sum(r.get("llm_total_tokens", 0) for r in rows)
        batch_retries = sum(r.get("llm_retries", 0) for r in rows)
        # completed games belonging to combinations that actually used the LLM
        llm_completed = sum(r["completed"] for r in rows if r.get("llm_calls", 0) > 0)
        print("  LLM usage:")
        print(f"    Total calls:    {batch_llm_calls:>14,}")
        print(f"    Total retries:  {batch_retries:>14,}   ({batch_retries / batch_llm_calls:.1%} of calls)")
        print(f"    Total tokens:   {batch_tokens:>14,}")
        print(f"    Tokens/call:    {batch_tokens / batch_llm_calls:>14,.0f}")
        if llm_completed:
            print(f"    Calls/game:     {batch_llm_calls / llm_completed:>14,.1f}   (over {llm_completed:,} LLM games)")
            print(f"    Tokens/game:    {batch_tokens / llm_completed:>14,.0f}")
    # Soft-likelihood usage across the batch (shown only for soft_symbolic batches)
    batch_lk_q = sum(r.get("lk_queries", 0) for r in rows)
    if batch_lk_q > 0:
        rates = [r["lk_cache_hit_rate"] for r in rows if r.get("lk_queries", 0) > 0]
        print(f"  Soft-likelihood: {batch_lk_q:,} queries, "
              f"{sum(r.get('lk_judge_calls', 0) for r in rows):,} judge LLM calls, "
              f"mean cache hit {sum(rates)/len(rates):.1%}, "
              f"{sum(r.get('lk_defaulted_slots', 0) for r in rows):,} defaulted slots")
    print()

    # Detected factors
    print(f"  Factors detected: {len(param_cols)}")
    for col in param_cols:
        values = sorted(set(str(r[col]) for r in rows))
        print(f"    {col}: {values}")
    print()

    # ── 1. Ranking Table ──────────────────────────────────────
    scored = [r for r in rows if r["completed"] > 0]
    scored.sort(key=lambda x: x["prince_accuracy"], reverse=True)

    print("─" * 80)
    print("  1. RANKING (by Prince accuracy)")
    print("─" * 80)

    # Build compact param display. LLM cost columns (calls / retries / tokens) are
    # shown per combination only when the batch actually used the LLM.
    header_params = "  ".join(f"{_short(c):>10}" for c in param_cols)
    llm_hdr = f"  {'Calls':>7} {'Retry':>5} {'Tokens':>12}" if batch_llm_calls > 0 else ""
    llm_sep = f"  {'-'*7} {'-'*5} {'-'*12}" if batch_llm_calls > 0 else ""
    print(f"  {'Rank':<5} {header_params}  {'Acc':>7} {'P/Q/O':>10} {'N':>4} {'Fail':>4} {'Time':>7}{llm_hdr}")
    print(f"  {'-'*5} {'  '.join('-'*10 for _ in param_cols)}  {'-'*7} {'-'*10} {'-'*4} {'-'*4} {'-'*7}{llm_sep}")

    for rank, r in enumerate(scored, 1):
        param_vals = "  ".join(f"{_short_val(r.get(c, '')):>10}" for c in param_cols)
        acc = f"{r['prince_accuracy']:.1%}"
        pqo = f"{r['princess_wins']}/{r['queen_wins']}/{r['other_wins']}"
        marker = " ★" if rank == 1 else ""
        llm_cell = (f"  {r['llm_calls']:>7,} {r['llm_retries']:>5,} {r['llm_total_tokens']:>12,}"
                    if batch_llm_calls > 0 else "")
        print(f"  {rank:<5} {param_vals}  {acc:>7} {pqo:>10} {r['completed']:>4} {r['failed']:>4} "
              f"{r['duration_seconds']:>6.0f}s{llm_cell}{marker}")

    # ── 2. Per-Factor Main Effects ────────────────────────────
    print()
    print("─" * 80)
    print("  2. MAIN EFFECTS (average accuracy per factor level)")
    print("─" * 80)

    factor_effects = {}

    for col in param_cols:
        unique_vals = sorted(set(str(r[col]) for r in scored))
        if len(unique_vals) < 2:
            continue  # Skip factors with only one level

        print(f"\n  {col}:")

        level_results = defaultdict(list)
        for r in scored:
            level_results[str(r[col])].append(r["prince_accuracy"])

        best_level = None
        best_acc = -1
        for level in unique_vals:
            accs = level_results[level]
            mean_acc = sum(accs) / len(accs)
            n_combos = len(accs)
            total_games_level = sum(r["completed"] for r in scored if str(r[col]) == level)

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_level = level

            marker = ""
            print(f"    {level:>20}: mean accuracy {mean_acc:>6.1%}  "
                  f"({n_combos} combos, {total_games_level} games){marker}")

        # Effect size (difference between levels for 2-level factors)
        if len(unique_vals) == 2:
            acc_a = sum(level_results[unique_vals[0]]) / len(level_results[unique_vals[0]])
            acc_b = sum(level_results[unique_vals[1]]) / len(level_results[unique_vals[1]])
            diff = abs(acc_a - acc_b)
            factor_effects[col] = {
                "diff": diff,
                "best_level": best_level,
                "best_acc": best_acc,
                "levels": {v: sum(level_results[v]) / len(level_results[v]) for v in unique_vals},
            }
            impact = "HIGH" if diff >= 0.10 else "MEDIUM" if diff >= 0.05 else "LOW"
            print(f"    → Effect size: {diff:.1%} ({impact}), best: {best_level}")

    # ── 3. Factor Impact Summary ──────────────────────────────
    if factor_effects:
        print()
        print("─" * 80)
        print("  3. FACTOR IMPACT RANKING")
        print("─" * 80)

        sorted_effects = sorted(factor_effects.items(), key=lambda x: x[1]["diff"], reverse=True)

        for rank, (col, eff) in enumerate(sorted_effects, 1):
            diff = eff["diff"]
            impact = "HIGH" if diff >= 0.10 else "MEDIUM" if diff >= 0.05 else "LOW"
            print(f"  {rank}. {col:<35} effect={diff:>6.1%} ({impact:>6})  best={eff['best_level']}")

        # Recommended combination
        print()
        print("  RECOMMENDED COMBINATION:")
        for col, eff in sorted_effects:
            print(f"    {col} = {eff['best_level']}")

    # ── 4. Agent Type Breakdown ───────────────────────────────
    prince_col = None
    players_col = None
    for col in param_cols:
        if "prince" in col.lower() and "type" in col.lower():
            prince_col = col
        if "players" in col.lower():
            players_col = col

    if prince_col and players_col:
        print()
        print("─" * 80)
        print("  4. AGENT MATCHUP ANALYSIS")
        print("─" * 80)

        matchups = defaultdict(list)
        for r in scored:
            key = (str(r[prince_col]), str(r[players_col]))
            matchups[key].append(r["prince_accuracy"])

        print(f"  {'Prince':>10} vs {'Players':<10}  {'Mean Acc':>10}  {'N Combos':>10}")
        print(f"  {'-'*10}    {'-'*10}  {'-'*10}  {'-'*10}")
        for (prince, players), accs in sorted(matchups.items()):
            mean = sum(accs) / len(accs)
            print(f"  {prince:>10} vs {players:<10}  {mean:>9.1%}  {len(accs):>10}")

    print()
    print("=" * 80)

    # ── Charts ────────────────────────────────────────────────
    if scored:
        _save_batch_charts(scored, param_cols, factor_effects, csv_path)


def _save_batch_charts(scored, param_cols, factor_effects, csv_path):
    """Generate batch analysis charts."""
    charts_dir = os.path.join(os.path.dirname(csv_path) or ".", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Factor main effects bar chart
    if factor_effects:
        factors_to_plot = [(col, eff) for col, eff in factor_effects.items()]
        n_factors = len(factors_to_plot)

        fig, axes = plt.subplots(1, n_factors, figsize=(4 * n_factors, 5), squeeze=False)
        axes = axes[0]

        for ax, (col, eff) in zip(axes, factors_to_plot):
            levels = list(eff["levels"].keys())
            accs = [eff["levels"][l] for l in levels]
            colors = ["#4CAF50" if l == eff["best_level"] else "#90CAF9" for l in levels]

            bars = ax.bar(levels, accs, color=colors)
            ax.set_title(_short(col), fontsize=11)
            ax.set_ylabel("Mean Accuracy")
            ax.set_ylim(0, max(max(accs) * 1.3, 0.3))

            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{acc:.1%}", ha="center", fontsize=10)

        plt.suptitle("Factor Main Effects on Prince Accuracy", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "factor_main_effects.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # 2. Ranking bar chart (top 15)
    top_n = min(15, len(scored))
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))

    labels = [r["combination"][:40] for r in scored[:top_n]]
    accs = [r["prince_accuracy"] for r in scored[:top_n]]
    colors = ["#4CAF50" if i == 0 else "#90CAF9" for i in range(top_n)]

    y_pos = range(top_n)
    ax.barh(y_pos, accs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Prince Accuracy")
    ax.set_title(f"Top {top_n} Combinations by Accuracy")
    ax.set_xlim(0, max(max(accs) * 1.2, 0.3))
    ax.invert_yaxis()

    for i, acc in enumerate(accs):
        ax.text(acc + 0.005, i, f"{acc:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "ranking.png"), dpi=150)
    plt.close()

    # 3. Factor impact ranking
    if factor_effects:
        sorted_eff = sorted(factor_effects.items(), key=lambda x: x[1]["diff"], reverse=True)
        fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_eff) * 0.6)))

        names = [_short(col) for col, _ in sorted_eff]
        diffs = [eff["diff"] for _, eff in sorted_eff]
        colors = ["#F44336" if d >= 0.10 else "#FF9800" if d >= 0.05 else "#4CAF50" for d in diffs]

        y_pos = range(len(names))
        ax.barh(y_pos, diffs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Effect Size (accuracy difference)")
        ax.set_title("Factor Impact Ranking")
        ax.invert_yaxis()

        for i, d in enumerate(diffs):
            label = "HIGH" if d >= 0.10 else "MED" if d >= 0.05 else "LOW"
            ax.text(d + 0.002, i, f"{d:.1%} ({label})", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "factor_impact.png"), dpi=150)
        plt.close()

    # 4. Accuracy vs. LLM token cost (efficiency frontier) — only if tokens tracked
    cost_rows = [r for r in scored if r.get("llm_total_tokens", 0) > 0 and r["completed"] > 0]
    if cost_rows:
        fig, ax = plt.subplots(figsize=(8, 6))
        xs = [r["llm_total_tokens"] / r["completed"] for r in cost_rows]
        ys = [r["prince_accuracy"] for r in cost_rows]
        ax.scatter(xs, ys, color="#2196F3", alpha=0.7, s=40)
        best = max(cost_rows, key=lambda r: r["prince_accuracy"])
        ax.scatter([best["llm_total_tokens"] / best["completed"]], [best["prince_accuracy"]],
                   color="#4CAF50", s=90, zorder=5, label=f"best acc: {best['prince_accuracy']:.0%}")
        ax.set_xlabel("Tokens per game")
        ax.set_ylabel("Prince Accuracy")
        ax.set_title("Accuracy vs. LLM Token Cost")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "accuracy_vs_cost.png"), dpi=150)
        plt.close()

    print(f"  Charts saved to: {charts_dir}/")


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _short(col_name: str) -> str:
    """Shorten column names for display: 'agents.prince.type' → 'prince'."""
    parts = col_name.split(".")
    if parts[-1] in ("type", "enabled") and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def _short_val(val) -> str:
    """Shorten values for display."""
    s = str(val)
    if s in ("True", "true"):
        return "Y"
    if s in ("False", "false"):
        return "N"
    return s


# ══════════════════════════════════════════════════════════════
#  Learning-Curve Analysis (cross-game memory experiments)
# ══════════════════════════════════════════════════════════════

def _series_label(config: dict) -> str:
    """Short label for one run's Prince config (the learning-curve grouping key)."""
    prince = config.get("agents", {}).get("prince", {})
    label = prince.get("type", "unknown")
    if label == "ablation":
        abl = prince.get("ablation", {})
        label = f"{abl.get('question_strategy', 'ig')}-{abl.get('guess_strategy', 'map')}" \
            f"-{abl.get('likelihood', 'uniform')}"
        # Memory representation arms (grid_memory_profile) must not pool together;
        # "raw" keeps the plain label so grid_memory runs group as before.
        if abl.get("likelihood") == "memory":
            report = prince.get("memory", {}).get("report", "raw")
            if report != "raw":
                label += f"-{report}"
        # Same for the bayes adapt modes (grid_memory_bayes vs _bayes_joint): they are
        # different arms, so they must not pool under one label.
        if abl.get("likelihood") == "bayes":
            adapt = abl.get("bayes_adapt", "mixture")
            if adapt != "mixture":
                label += f"-{adapt}"
            # An opponent no type candidate was built for makes it another experiment
            # (grid_memory_bayes_joint_unmodelled), so it must not pool either.
            unmodelled = []
            for player in config.get("agents", {}).get("players", []):
                if player["type"] not in BUILT_FOR and player["type"] not in unmodelled:
                    unmodelled.append(player["type"])
            for player_type in unmodelled:
                label += f"-vs-{player_type}"
    return label


def _find_series(root_dirs: list[str]) -> list[dict]:
    """Every run (summary.csv + sibling config.yaml) under the given directories.

    One run = one learning series, or one PART of a series when the run was
    warm-started from a previous run's memory.json (memory.load_path, e.g. via
    continue_runs.py). Such continuation runs restart their game numbering at 0,
    so each run also gets an ``offset``: how many games the loaded memory already
    contained (own memory.json total minus games recorded in this run). Adding it
    to the game index stacks chained parts onto one series axis.
    """
    series = []
    for root in root_dirs:
        for dirpath, _dirnames, filenames in sorted(os.walk(root)):
            if "summary.csv" not in filenames:
                continue
            rows = [parse_single_row(r) for r in load_csv(os.path.join(dirpath, "summary.csv"))]
            cfg_path = os.path.join(dirpath, "config.yaml")
            config = {}
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r") as f:
                    config = yaml.safe_load(f) or {}

            offset = 0
            memory_cfg = config.get("agents", {}).get("prince", {}).get("memory", {}) or {}
            memory_path = os.path.join(dirpath, "memory.json")
            if memory_cfg.get("load_path") and os.path.isfile(memory_path):
                with open(memory_path, "r") as f:
                    total_recorded = json.load(f).get("games_recorded", 0)
                recorded_here = sum(1 for r in rows if r["status"] == "completed")
                offset = max(0, total_recorded - recorded_here)

            series.append({"label": _series_label(config), "dir": dirpath, "rows": rows,
                           "offset": offset})
    return series


def analyze_learning_curve(root_dirs: list[str]) -> None:
    """Accuracy vs game index, averaged over series, grouped by Prince config.

    The question: does the Prince's accuracy RISE over a run's games (it can only
    do so if something persists across games — the cross-game memory)? Failed
    games are excluded from a game index's average rather than counted as wrong.
    """
    series = _find_series(root_dirs)
    if not series:
        print(f"No summary.csv found under: {', '.join(root_dirs)}")
        return

    # label -> game index -> list of 0/1 outcomes (one per series)
    outcomes: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    # A continuation run (offset > 0) is a later PART of an existing series, not a
    # new one, so count series by their offset-0 opening parts only.
    n_series = Counter(s["label"] for s in series if s["offset"] == 0)
    for s in series:
        for r in s["rows"]:
            if r["status"] != "completed":
                continue
            outcomes[s["label"]][s["offset"] + int(r["game"])].append(1 if r["correct"] else 0)

    print()
    print("=" * 60)
    print("  LEARNING-CURVE ANALYSIS (accuracy vs game index)")
    print("=" * 60)
    print(f"  Runs found: {len(series)} under {', '.join(root_dirs)}")
    print()

    # label -> one accuracy per RUN, for the spread across independent series in the
    # summary table. A continued series contributes its parts separately.
    run_accuracy: dict[str, list[float]] = defaultdict(list)
    for s in series:
        done = [1 if r["correct"] else 0 for r in s["rows"] if r["status"] == "completed"]
        if done:
            run_accuracy[s["label"]].append(statistics.mean(done))

    table_rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in sorted(outcomes):
        per_index = outcomes[label]
        indices = sorted(per_index)
        if not indices:
            print(f"  {label}: no completed games, skipped")
            continue
        means = [statistics.mean(per_index[i]) for i in indices]
        flat = []
        for i in indices:
            for outcome in per_index[i]:
                flat.append((i, outcome))
        overall = statistics.mean(o for _, o in flat)

        # First-half vs second-half accuracy + least-squares slope over all
        # (game index, outcome) points: a positive slope = a rising curve.
        mid = (indices[0] + indices[-1] + 1) / 2
        first = [o for i, o in flat if i < mid]
        second = [o for i, o in flat if i >= mid]
        halves = " -> ".join(f"{statistics.mean(h):.1%}" if h else "n/a" for h in (first, second))
        mean_i = statistics.mean(i for i, _ in flat)
        mean_o = statistics.mean(o for _, o in flat)
        var_i = sum((i - mean_i) ** 2 for i, _ in flat)
        slope = sum((i - mean_i) * (o - mean_o) for i, o in flat) / var_i if var_i else 0.0

        print(f"  {label}  ({n_series[label]} series, {len(flat)} completed games)")
        print(f"    overall accuracy:     {overall:.1%}")
        print(f"    first half -> second: {halves}")
        print(f"    trend slope:          {slope:+.4f} accuracy/game")
        print()

        # Spread over runs, not over games: games inside one run share a memory, so
        # the run is the independent unit. stderr = 0 when only one run carries the arm.
        accuracies = run_accuracy[label]
        stderr = statistics.stdev(accuracies) / (len(accuracies) ** 0.5) if len(accuracies) > 1 else 0.0
        table_rows.append((label, n_series[label], len(flat), overall, stderr, first, second, slope))

        line, = ax.plot(indices, means, alpha=0.35, linewidth=1)
        # Rolling mean smooths the per-index noise (few series per point)
        w = min(5, len(means))
        rolling = [statistics.mean(means[max(0, k - w + 1):k + 1]) for k in range(len(means))]
        ax.plot(indices, rolling, color=line.get_color(), linewidth=2.5,
                label=f"{label} (n={n_series[label]} series)")

    _print_learning_curve_table(table_rows)

    ax.set_xlabel("Game index within a series")
    ax.set_ylabel("Prince accuracy (mean over series)")
    ax.set_title("Cross-game learning curve (thin = per-index mean, thick = rolling mean)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(root_dirs[0], "learning_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Chart saved to: {out_path}")
    print()


def _print_learning_curve_table(table_rows: list[tuple]) -> None:
    """The same per-arm numbers as above, as one markdown table to paste into a paper."""
    print("  Accuracy by arm (+- standard error over series), best first:")
    print()
    print("  | Method | Series | Games | Accuracy | 1st half | 2nd half | Slope (acc/game) |")
    print("  |---|---|---|---|---|---|---|")
    for label, n, games, overall, stderr, first, second, slope in sorted(table_rows, key=lambda r: -r[3]):
        halves = []
        for half in (first, second):
            halves.append(f"{statistics.mean(half):.1%}" if half else "n/a")
        print(f"  | {label} | {n} | {games} | {overall:.1%} +- {stderr:.1%} | "
            f"{halves[0]} | {halves[1]} | {slope:+.4f} |")
    print()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze.py <csv_file>")
        print("  python analyze.py --learning-curve <results_dir> [...]")
        print()
        print("  Single experiment:  python analyze.py results/<experiment>/summary.csv")
        print("  Batch experiment:   python analyze.py results/batch_results_xxx.csv")
        print("  Learning curve:     python analyze.py --learning-curve results/grid_memory")
        sys.exit(1)

    if "--learning-curve" in sys.argv:
        dirs = [a for a in sys.argv[1:] if a != "--learning-curve"]
        missing = [d for d in dirs if not os.path.isdir(d)]
        if not dirs or missing:
            print(f"Error: --learning-curve needs existing results directories "
                  f"(missing: {', '.join(missing) if missing else 'none given'}).")
            sys.exit(1)
        analyze_learning_curve(dirs)
        return

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"Error: '{csv_path}' not found.")
        sys.exit(1)

    rows = load_csv(csv_path)
    if not rows:
        print("Error: CSV is empty.")
        sys.exit(1)

    if is_batch_csv(rows):
        analyze_batch(rows, csv_path)
    else:
        analyze_single(rows, csv_path)


if __name__ == "__main__":
    main()