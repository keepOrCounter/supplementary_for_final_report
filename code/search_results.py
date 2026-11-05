"""
Tofu Kingdom - Results Search

Find experiment run directories under results/ by their CONFIG, instead of
opening each config.yaml by hand. Every run is saved as

    results/<time>_<model>_<Np>/
        config.yaml        <- the strategy config lives here (buried in
        summary.csv           agents.prince.ablation: question_strategy,
        game_001.json         guess_strategy, likelihood, ...)
        ...

but the folder name only encodes time/model/#players, so locating "the ig +
map + strategic run" means grepping configs one by one. This script scans all
run dirs (recursively), reads each config, joins it with the run's accuracy
from summary.csv, and lets you filter on the config fields.

Usage:
    # list every run with a one-line config summary + accuracy
    python search_results.py

    # filter by the ablation knobs (comma = OR within a flag; flags AND together)
    python search_results.py --question ig --guess map --likelihood strategic
    python search_results.py -q llm -l uniform,strategic
    python search_results.py --prince grail --factor likelihood

    # generic nested match on anything in config.yaml (repeatable)
    python search_results.py --where agents.prince.ablation.tom_eps=0.2
    python search_results.py --where llm.reasoning_effort~=high      # ~= contains
    python search_results.py --where game.num_games>=50

    # quality / sorting / output
    python search_results.py -q ig --min-acc 0.5 --sort acc
    python search_results.py --model qwen3.5 --full        # dump all key fields
    python search_results.py -l bayes --json               # machine-readable

    # point at a different results root
    python search_results.py path/to/results -q ig
"""

from __future__ import annotations
import os
import sys
import csv
import json
import argparse

import yaml


CONFIG_NAMES = ("config.yaml", "copy_of_config.yaml")


# ══════════════════════════════════════════════════════════════
#  Scan run directories
# ══════════════════════════════════════════════════════════════

def find_run_dirs(root: str) -> list[str]:
    """Every directory under `root` that holds a saved config (i.e. a run).

    Walks recursively so it works whether runs sit flat in results/ or under
    named sub-folders (e.g. results/grid_memory/<run>/), the layout batch and
    learning-curve experiments use.
    """
    run_dirs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(name in filenames for name in CONFIG_NAMES):
            run_dirs.append(dirpath)
    return sorted(run_dirs)


def load_config(run_dir: str) -> dict | None:
    for name in CONFIG_NAMES:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError):
                return None
    return None


def load_run_stats(run_dir: str) -> dict:
    """Prince accuracy for a run, from summary.csv (falls back to game_*.json).

    Returns {completed, failed, correct, accuracy}. accuracy is None when the
    run has no completed games yet (e.g. still running / crashed early).
    """
    completed = failed = correct = 0

    summary = os.path.join(run_dir, "summary.csv")
    if os.path.isfile(summary):
        with open(summary, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "completed":
                    completed += 1
                    if str(row.get("correct", "")).strip().lower() == "true":
                        correct += 1
                elif row.get("status") == "failed":
                    failed += 1
    else:
        # No summary yet: reconstruct from the per-game JSON records.
        for name in sorted(os.listdir(run_dir)):
            if not (name.startswith("game_") and name.endswith(".json")):
                continue
            try:
                with open(os.path.join(run_dir, name), "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get("status") == "failed":
                failed += 1
            elif rec.get("result"):
                completed += 1
                if rec["result"].get("correct"):
                    correct += 1

    accuracy = (correct / completed) if completed else None
    return {"completed": completed, "failed": failed, "correct": correct, "accuracy": accuracy}


# ══════════════════════════════════════════════════════════════
#  Extract the searchable / displayable config fields
# ══════════════════════════════════════════════════════════════

def players_signature(players: list) -> str:
    """Compact opponent-lineup string, e.g. 'symbolic' or 'random2-llm2'."""
    if not isinstance(players, list) or not players:
        return "?"
    types = [p.get("type", "?") if isinstance(p, dict) else str(p) for p in players]
    runs: list[list] = []
    for t in types:
        if runs and runs[-1][0] == t:
            runs[-1][1] += 1
        else:
            runs.append([t, 1])
    if len(runs) == 1:
        return runs[0][0]
    return "-".join(t if n == 1 else f"{t}{n}" for t, n in runs)


def extract_key_config(cfg: dict) -> dict:
    """Flatten the config fields worth searching on into a single dict.

    Which Prince-specific block is pulled depends on the Prince type, so the
    friendly filters (--question / --guess / --likelihood / --factor / ...)
    all map onto keys returned here.
    """
    agents = cfg.get("agents", {}) or {}
    prince = agents.get("prince", {}) or {}
    ptype = prince.get("type")
    out: dict = {"prince": ptype}

    if ptype == "ablation":
        ab = prince.get("ablation", {}) or {}
        out.update({
            "question": ab.get("question_strategy"),
            "guess": ab.get("guess_strategy"),
            "likelihood": ab.get("likelihood"),
            "tom_smoothing": ab.get("tom_smoothing"),
            "tom_eps": ab.get("tom_eps"),
            "tom_beta": ab.get("tom_beta"),
            "bayes_adapt": ab.get("bayes_adapt"),
            "belief_mode": ab.get("belief_mode"),
            "give_belief_state": ab.get("give_belief_state"),
            "give_history": ab.get("give_history"),
            "give_memory": ab.get("give_memory"),
            "judge_reasoning_effort": ab.get("judge_reasoning_effort"),
            "judge_cot": ab.get("judge_cot"),
        })
    elif ptype == "grail":
        gr = prince.get("grail", {}) or {}
        out.update({
            "use_llm_prior": gr.get("use_llm_prior"),
            "algorithm": gr.get("algorithm"),
            "factor_function": gr.get("factor_function"),
            "vibes_reasoning_effort": gr.get("vibes_reasoning_effort"),
        })
    elif ptype == "emo":
        em = prince.get("emo", {}) or {}
        out.update({
            "self_refine_steps": em.get("self_refine_steps"),
            "global_refine_steps": em.get("global_refine_steps"),
            "model_reasoning_effort": em.get("model_reasoning_effort"),
            "use_reasoning_history": em.get("use_reasoning_history"),
        })
    elif ptype == "soft_symbolic":
        sl = prince.get("soft_likelihood", {}) or {}
        out.update({"top_k": sl.get("top_k")})

    # Cross-game memory applies to any Prince type.
    mem = prince.get("memory", {}) or {}
    out["memory"] = mem.get("enabled")
    if mem.get("enabled"):
        out["memory_report"] = mem.get("report")
        out["memory_window"] = mem.get("window")

    # Global run settings.
    llm = cfg.get("llm", {}) or {}
    game = cfg.get("game", {}) or {}
    out["model"] = llm.get("model")
    out["reasoning_effort"] = llm.get("reasoning_effort")
    out["num_players"] = game.get("num_players")
    out["num_games"] = game.get("num_games")
    out["seed"] = game.get("seed")
    out["players"] = players_signature(agents.get("players", []))

    # Drop keys that are absent so signatures/filters ignore them cleanly.
    return {k: v for k, v in out.items() if v is not None}


def signature(key: dict) -> str:
    """One-line, type-aware summary of the config for the results table."""
    p = key.get("prince", "?")
    parts: list[str] = []
    if p == "ablation":
        parts.append(f"q={key.get('question', '?')}")
        parts.append(f"g={key.get('guess', '?')}")
        parts.append(f"lk={key.get('likelihood', '?')}")
        lk = key.get("likelihood")
        if lk in ("strategic", "strategic3"):
            sm = key.get("tom_smoothing", "?")
            if sm == "eps_greedy":
                parts.append(f"tom={sm}/{key.get('tom_eps', '?')}")
            elif sm == "softmax":
                parts.append(f"tom={sm}/{key.get('tom_beta', '?')}")
            else:
                parts.append(f"tom={sm}")
        if lk == "bayes":
            parts.append(f"adapt={key.get('bayes_adapt', '?')}")
    elif p == "grail":
        parts.append(f"prior={_b(key.get('use_llm_prior'))}")
        parts.append(f"algo={key.get('algorithm', '?')}")
        parts.append(f"factor={key.get('factor_function', '?')}")
    elif p == "emo":
        parts.append(f"S={key.get('self_refine_steps', '?')}")
        parts.append(f"G={key.get('global_refine_steps', '?')}")
        parts.append(f"reff={key.get('model_reasoning_effort', '?')}")
    elif p == "soft_symbolic":
        parts.append(f"top_k={key.get('top_k', '?')}")

    if key.get("memory"):
        parts.append(f"mem={key.get('memory_report', 'raw')}")
    parts.append(f"vs {key.get('players', '?')}")
    return " ".join(parts)


def _b(v) -> str:
    return "Y" if v else "N"


# ══════════════════════════════════════════════════════════════
#  Filtering
# ══════════════════════════════════════════════════════════════

def _norm(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().lower()


_TRUEY = {"true", "yes", "y", "1", "on", "enabled"}
_FALSEY = {"false", "no", "n", "0", "off", "disabled"}


def match_equals(actual, wanted_csv: str) -> bool:
    """actual equals any of the comma-separated wanted values (case-insensitive).

    Booleans also accept yes/no/1/0 style spellings.
    """
    a = _norm(actual)
    wanted = [w.strip() for w in wanted_csv.split(",") if w.strip()]
    for w in wanted:
        wn = w.lower()
        if wn == a:
            return True
        if isinstance(actual, bool):
            if (wn in _TRUEY) == bool(actual) and wn in (_TRUEY | _FALSEY):
                return True
    return False


def get_nested(cfg: dict, dot_path: str):
    """Follow a dotted path into nested dicts; None if any hop is missing."""
    cur = cfg
    for part in dot_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def match_where(cfg: dict, expr: str) -> bool:
    """Evaluate one --where expression against the full config.

    Operators (checked longest-first): ~= contains, >= <= numeric, != not-equal,
    = equals (comma = OR).
    """
    for op in ("~=", ">=", "<=", "!=", "="):
        if op in expr:
            path, _, raw = expr.partition(op)
            path, raw = path.strip(), raw.strip()
            actual = get_nested(cfg, path)
            if op == "~=":
                return actual is not None and raw.lower() in _norm(actual)
            if op in (">=", "<="):
                try:
                    an, rn = float(actual), float(raw)
                except (TypeError, ValueError):
                    return False
                return an >= rn if op == ">=" else an <= rn
            if op == "!=":
                return actual is not None and not match_equals(actual, raw)
            return actual is not None and match_equals(actual, raw)
    # No operator: treat as "path exists".
    return get_nested(cfg, expr) is not None


def passes_filters(run: dict, args) -> bool:
    key, cfg, stats = run["key"], run["config"], run["stats"]

    # Named field filters (map flag -> extracted key).
    named = {
        "prince": args.prince,
        "question": args.question,
        "guess": args.guess,
        "likelihood": args.likelihood,
        "tom_smoothing": args.tom,
        "factor_function": args.factor,
        "bayes_adapt": args.adapt,
        "num_players": args.num_players,
    }
    for field, wanted in named.items():
        if wanted is None:
            continue
        if field not in key or not match_equals(key[field], wanted):
            return False

    # Substring filters.
    if args.model is not None:
        if "model" not in key or args.model.lower() not in _norm(key["model"]):
            return False
    if args.players is not None:
        if "players" not in key or args.players.lower() not in _norm(key["players"]):
            return False

    # Boolean toggles.
    if args.memory and not key.get("memory"):
        return False
    if args.no_memory and key.get("memory"):
        return False

    # Generic nested filters.
    for expr in args.where or []:
        if not match_where(cfg, expr):
            return False

    # Quality gates.
    acc = stats["accuracy"]
    if args.min_acc is not None and (acc is None or acc < args.min_acc):
        return False
    if args.max_acc is not None and (acc is None or acc > args.max_acc):
        return False
    if args.min_games is not None and stats["completed"] < args.min_games:
        return False

    return True


# ══════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════

def acc_cell(stats: dict) -> str:
    acc = stats["accuracy"]
    if acc is None:
        return "   —      "
    return f"{acc:5.1%} {stats['correct']:>3}/{stats['completed']:<3}"


def print_table(matches: list[dict], root: str) -> None:
    if not matches:
        print("  (no matching runs)")
        return
    print(f"  {'#':>3}  {'ACCURACY':<12} {'PRINCE':<12} {'CONFIG':<44} {'MODEL':<20} DIR")
    print(f"  {'-'*3}  {'-'*12} {'-'*12} {'-'*44} {'-'*20} {'-'*30}")
    for i, run in enumerate(matches, 1):
        key = run["key"]
        disp = os.path.relpath(run["dir"])
        model = str(key.get("model", "?"))
        if len(model) > 20:
            model = model[:17] + "..."
        print(f"  {i:>3}  {acc_cell(run['stats']):<12} {str(key.get('prince', '?')):<12} "
              f"{signature(key):<44} {model:<20} {disp}")


def print_full(matches: list[dict]) -> None:
    for i, run in enumerate(matches, 1):
        print()
        print(f"  [{i}] {os.path.relpath(run['dir'])}")
        s = run["stats"]
        acc = "—" if s["accuracy"] is None else f"{s['accuracy']:.1%}"
        print(f"      accuracy : {acc}  ({s['correct']}/{s['completed']} correct, "
              f"{s['failed']} failed)")
        for k, v in run["key"].items():
            print(f"      {k:<22}: {v}")


def as_json(matches: list[dict]) -> str:
    payload = []
    for run in matches:
        payload.append({
            "dir": os.path.relpath(run["dir"]),
            "accuracy": run["stats"]["accuracy"],
            "completed": run["stats"]["completed"],
            "failed": run["stats"]["failed"],
            "correct": run["stats"]["correct"],
            "config": run["key"],
        })
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Search Tofu Kingdom experiment runs by their config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Comma-separate a filter value for OR (e.g. -q ig,llm). "
               "Multiple flags AND together.",
    )
    p.add_argument("root", nargs="?", default="results",
                   help="results directory to scan (default: results)")

    g = p.add_argument_group("config filters")
    g.add_argument("--prince", help="Prince agent type (ablation|grail|emo|...)")
    g.add_argument("-q", "--question", help="ablation question_strategy (ig|random|llm)")
    g.add_argument("-g", "--guess", help="ablation guess_strategy (map|random|llm)")
    g.add_argument("-l", "--likelihood",
                   help="ablation likelihood (uniform|llm|strategic|strategic3|memory|bayes)")
    g.add_argument("--tom", help="ablation tom_smoothing (softmax|eps_greedy)")
    g.add_argument("--factor", help="grail factor_function (uniform|likelihood|posterior)")
    g.add_argument("--adapt", help="ablation bayes_adapt (mixture|map|joint)")
    g.add_argument("--model", help="substring match on llm.model")
    g.add_argument("--players", help="substring match on opponent lineup (e.g. symbolic)")
    g.add_argument("--num-players", dest="num_players", help="game.num_players")
    g.add_argument("--memory", action="store_true", help="only runs with memory.enabled")
    g.add_argument("--no-memory", action="store_true", help="only runs without memory")
    g.add_argument("--where", action="append", metavar="path OP value",
                   help="generic nested filter; OP is = ~= != >= <= "
                        "(e.g. agents.prince.ablation.tom_eps=0.2). Repeatable.")

    q = p.add_argument_group("quality gates")
    q.add_argument("--min-acc", type=float, help="min prince accuracy (0-1)")
    q.add_argument("--max-acc", type=float, help="max prince accuracy (0-1)")
    q.add_argument("--min-games", type=int, help="min completed games")

    o = p.add_argument_group("output")
    o.add_argument("--sort", choices=("date", "acc", "games", "dir"), default="date",
                   help="sort order (default: date, newest first)")
    o.add_argument("--limit", type=int, help="show at most N matches")
    o.add_argument("--full", action="store_true", help="dump every key field per match")
    o.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    o.add_argument("--count", action="store_true", help="print only the match count")
    return p


def sort_matches(matches: list[dict], key: str) -> list[dict]:
    if key == "acc":
        return sorted(matches, key=lambda r: (r["stats"]["accuracy"] is not None,
                                              r["stats"]["accuracy"] or 0.0), reverse=True)
    if key == "games":
        return sorted(matches, key=lambda r: r["stats"]["completed"], reverse=True)
    if key == "dir":
        return sorted(matches, key=lambda r: r["dir"])
    # date: newest first by mtime
    return sorted(matches, key=lambda r: os.path.getmtime(r["dir"]), reverse=True)


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.isdir(args.root):
        print(f"Error: results directory '{args.root}' not found.", file=sys.stderr)
        print("       (nothing has been run yet, or pass the right path: "
              "python search_results.py <dir>)", file=sys.stderr)
        sys.exit(1)

    run_dirs = find_run_dirs(args.root)

    runs = []
    for d in run_dirs:
        cfg = load_config(d)
        if cfg is None:
            continue
        runs.append({
            "dir": d,
            "config": cfg,
            "key": extract_key_config(cfg),
            "stats": load_run_stats(d),
        })

    matches = [r for r in runs if passes_filters(r, args)]
    matches = sort_matches(matches, args.sort)
    if args.limit is not None:
        matches = matches[:args.limit]

    if args.count:
        print(len(matches))
        return

    if args.json:
        print(as_json(matches))
        return

    print()
    print("=" * 70)
    print("  RESULTS SEARCH")
    print("=" * 70)
    print(f"  Scanned {len(runs)} run(s) under '{args.root}'  →  {len(matches)} match(es)")
    print()
    if args.full:
        print_full(matches)
    else:
        print_table(matches, args.root)
    print()


if __name__ == "__main__":
    main()
