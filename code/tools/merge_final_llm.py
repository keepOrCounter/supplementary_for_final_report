# merge the finished trials of the formal LLM likelihood run, check them, report
#
#   python3 tools/merge_final_llm.py [--grail results/final_grail_mixed]
#
# Run from the project root, as often as you like while trials are still
# running: only a trial with a summary.csv counts, and the rest are listed as
# pending. The trials are the ones in experiment_grids/grid_resistance_final_llm.yaml,
# trial k under results/final_llm_mixed/t{k}/, as experiment_grids/final_llm/ runs them.
#
# Nothing is written unless every check passes:
#   one finished run per trial, with the trial's own game seeds, each once
#   the tested agent (belief=llm) sits at player id k in trial k, in the
#     summary's agents column and in every belief row
#   each belief row's crashed flag agrees with its game in the summary
#   every finished game has a closing guess from the tested agent
#
#   each trial's batch_summary.csv counts the same games, wins and crashes
#
# Writes two pairs of files under results/final_llm_mixed/merged/, one for each
# mode of analyze_resistance.py:
#   batch_summary.csv, batch_beliefs.csv   one row per trial, so the batch mode
#       reports every trial on its own (a trial is a cell, seed=<trial seed>)
#   summary.csv, beliefs.csv               one row per game, so the single mode
#       reports all finished trials pooled
# In the pooled pair game_index is renumbered to trial * num_games + game_index:
# every trial counts its games from 0, and the single mode matches a game's
# crashed flag to its beliefs by game_index, so repeated numbers would mark the
# wrong games as crashed. The batch pair keeps the numbers, since the batch mode
# tells trials apart by their combination column.
#
# With --grail it pairs the run game by game, by seed, with the GRAIL arm:
# exact McNemar on who won, and an exact sign test on the tested seat's closing
# identity accuracy. A game crashed in either arm is left out of both.
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
from math import comb

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_resistance import no_information_baseline, wilson_interval

GRID = "experiment_grids/grid_resistance_final_llm.yaml"
RESULTS = "results/final_llm_mixed"
TESTED = "belief=llm" # in the tested agent's name, and in no other agent's
GRAIL = "GrailResistanceAgent"


def load_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def check_trial(trial: int, seed: int, num_games: int, summary: list[dict], beliefs: list[dict]) -> list[str]:
    """every way this trial's files disagree with the run it is meant to be"""
    problems = []
    seeds = []
    for row in summary:
        seeds.append(int(row["seed"]))
    expected = list(range(seed + 1, seed + num_games + 1))
    if sorted(seeds) != expected:
        problems.append(f"game seeds are not {expected[0]}..{expected[-1]} once each "
            f"({len(seeds)} rows, {len(set(seeds))} distinct)")

    crashed = {}
    for row in summary:
        crashed[int(row["seed"])] = row["crashed"] == "True"
        agents = row["agents"].split()
        tested_at = []
        for player_id, name in enumerate(agents):
            if TESTED in name:
                tested_at.append(player_id)
        if tested_at != [trial]:
            problems.append(f"seed {row['seed']}: the tested agent sits at {tested_at}, not [{trial}]")

    closing = set() # seeds where the tested agent left a belief
    for row in beliefs:
        game = int(row["seed"])
        if row["combination"] != f"seed={seed}":
            problems.append(f"seed {game}: belief row from {row['combination']}, not seed={seed}")
        if game not in crashed:
            problems.append(f"seed {game}: belief row for a game the summary does not hold")
            continue
        if (row["crashed"] == "True") != crashed[game]:
            problems.append(f"seed {game}: crashed is {row['crashed']} in beliefs and {crashed[game]} in the summary")
        if TESTED in row["agent"]:
            closing.add(game)
            if int(row["player_id"]) != trial:
                problems.append(f"seed {game}: tested agent's belief from player {row['player_id']}, not {trial}")
    for game, did_crash in crashed.items():
        if not did_crash and game not in closing:
            problems.append(f"seed {game}: finished, but the tested agent left no belief")
    return problems


def check_batch_row(seed: int, batch: list[dict], summary: list[dict]) -> list[str]:
    """every way the trial's batch_summary.csv row disagrees with its summary.csv"""
    if len(batch) != 1 or batch[0]["combination"] != f"seed={seed}":
        return [f"batch_summary.csv is not one row for seed={seed}"]
    crashed, wins = 0, 0
    for row in summary:
        crashed += row["crashed"] == "True"
        wins += row["winner"] == "resistance" and row["crashed"] != "True"
    counted = (int(batch[0]["games_played"]), int(batch[0]["games_crashed"]), int(batch[0]["resistance_wins"]))
    if counted != (len(summary) - crashed, crashed, wins):
        return [f"batch_summary.csv counts (played, crashed, resistance wins) {counted}, "
            f"summary.csv {(len(summary) - crashed, crashed, wins)}"]
    return []


def closing_accuracy(beliefs: list[dict]) -> dict[tuple[int, int], dict]:
    """(seed, player id) to the last snapshot that player took, in games that finished"""
    last = {}
    for row in beliefs:
        if row["crashed"] == "True":
            continue
        key = (int(row["seed"]), int(row["player_id"]))
        if key not in last or int(row["missions_seen"]) >= int(last[key]["missions_seen"]):
            last[key] = row
    return last


def exact_two_sided(first: int, second: int) -> float:
    """exact binomial test of first against second at one half each, two sided

    first, second: the games one side won and the other did not; ties are
    left out before this is called, which is what McNemar and the sign test do
    """
    n = first + second
    if n == 0:
        return 1.0
    tail = 0
    for k in range(min(first, second) + 1):
        tail += comb(n, k)
    return min(1.0, 2 * tail / 2 ** n)


def load_grail(grail_dir: str) -> tuple[dict[int, bool], dict[int, float]]:
    """GRAIL arm: seed to resistance won, and seed to GRAIL's closing accuracy, finished games only"""
    won, accuracy = {}, {}
    for path in sorted(glob.glob(os.path.join(grail_dir, "*", "summary.csv"))):
        _, rows = load_csv(path)
        for row in rows:
            game = int(row["seed"])
            if game in won:
                raise SystemExit(f"GRAIL arm: seed {game} appears twice; remove the stale run under {grail_dir}")
            if row["crashed"] != "True":
                won[game] = row["winner"] == "resistance"
        beliefs_path = os.path.join(os.path.dirname(path), "beliefs.csv")
        if not os.path.isfile(beliefs_path):
            continue
        _, beliefs = load_csv(beliefs_path)
        for (game, _), row in closing_accuracy(beliefs).items():
            if row["agent"] == GRAIL and game in won:
                accuracy[game] = float(row["accuracy"])
    return won, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grail", help="the GRAIL arm's results directory, to pair with")
    args = parser.parse_args()

    grid = yaml.safe_load(open(GRID, encoding="utf-8"))
    trial_seeds = grid["sweep"]["game.seed"]
    num_games = grid["base"]["game"]["num_games"]

    summary_header, belief_header, batch_header = None, None, None
    merged_summary, merged_beliefs = [], []
    merged_batch, merged_batch_beliefs = [], []
    finished, pending, problems = [], [], []
    for trial, seed in enumerate(trial_seeds):
        name = f"t{trial}"
        summaries = glob.glob(os.path.join(RESULTS, name, "2*", "summary.csv"))
        belief_files = glob.glob(os.path.join(RESULTS, name, "batch_resistance_*", "batch_beliefs.csv"))
        if not summaries:
            pending.append(name)
            continue
        if len(summaries) != 1 or len(belief_files) != 1:
            problems.append(f"{name}: {len(summaries)} summary.csv and {len(belief_files)} batch_beliefs.csv, "
                "expected one each; remove the run you do not want")
            continue
        batch_file = os.path.join(os.path.dirname(belief_files[0]), "batch_summary.csv")
        if not os.path.isfile(batch_file):
            problems.append(f"{name}: no batch_summary.csv next to {belief_files[0]}")
            continue
        header, summary = load_csv(summaries[0])
        b_header, beliefs = load_csv(belief_files[0])
        s_header, batch = load_csv(batch_file)
        if summary_header is None:
            summary_header, belief_header, batch_header = header, b_header, s_header
        if header != summary_header or b_header != belief_header or s_header != batch_header:
            problems.append(f"{name}: columns differ from the first finished trial")
            continue
        for problem in check_trial(trial, seed, num_games, summary, beliefs) + check_batch_row(seed, batch, summary):
            problems.append(f"{name}: {problem}")
        merged_batch.extend(batch)
        for row in beliefs:
            merged_batch_beliefs.append(dict(row)) # before the renumbering below
        for row in summary + beliefs:
            row["game_index"] = str(trial * num_games + int(row["game_index"]))
        merged_summary.extend(summary)
        merged_beliefs.extend(beliefs)
        finished.append((trial, name, summary, beliefs))

    if problems:
        print("NOT MERGED, the files disagree with the run:")
        for problem in dict.fromkeys(problems): # one line per problem, however many rows show it
            print("  " + problem)
        sys.exit(1)
    if not finished:
        sys.exit(f"no finished trial under {RESULTS} yet")

    out = os.path.join(RESULTS, "merged")
    os.makedirs(out, exist_ok=True)
    outputs = (("summary.csv", summary_header, merged_summary), ("beliefs.csv", belief_header, merged_beliefs), \
        ("batch_summary.csv", batch_header, merged_batch), ("batch_beliefs.csv", belief_header, merged_batch_beliefs))
    for name, header, rows in outputs:
        path = os.path.join(out, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    print(f"merged {len(finished)} trial(s) into {out}; pending: {', '.join(pending) or 'none'}\n")
    played, wins = 0, 0
    for trial, name, summary, beliefs in finished:
        trial_played, trial_wins, crashed = 0, 0, 0
        for row in summary:
            if row["crashed"] == "True":
                crashed += 1
                continue
            trial_played += 1
            trial_wins += row["winner"] == "resistance"
        scores = []
        for row in closing_accuracy(beliefs).values():
            if TESTED in row["agent"]:
                scores.append(float(row["accuracy"]))
        print(f"  {name} (tested agent at player {trial}): {trial_played} finished, {crashed} crashed, "
            f"resistance won {trial_wins} ({trial_wins / trial_played:.1%}), "
            f"tested agent's closing accuracy {sum(scores) / len(scores):.1%}")
        played += trial_played
        wins += trial_wins

    low, high = wilson_interval(wins, played)
    print(f"\n  resistance win rate: {wins}/{played} = {wins / played:.1%}  (95% CI {low:.1%} to {high:.1%})")
    by_agent = {}
    for row in closing_accuracy(merged_beliefs).values():
        by_agent.setdefault(row["agent"], []).append(float(row["accuracy"]))
    print(f"  identity accuracy on the closing guess (no-information baseline "
        f"{no_information_baseline(6, 2):.0%}):")
    for agent in sorted(by_agent):
        scores = by_agent[agent]
        print(f"    {agent:<52} n={len(scores):>3}  {sum(scores) / len(scores):.1%}")

    if not args.grail:
        return
    grail_won, grail_accuracy = load_grail(args.grail)
    ours_won, ours_accuracy = {}, {}
    for row in merged_summary:
        if row["crashed"] != "True":
            ours_won[int(row["seed"])] = row["winner"] == "resistance"
    for (game, _), row in closing_accuracy(merged_beliefs).items():
        if TESTED in row["agent"]:
            ours_accuracy[game] = float(row["accuracy"])

    common = sorted(set(ours_won) & set(grail_won))
    if not common:
        sys.exit(f"no game seed finished in both arms; is {args.grail} the GRAIL arm?")
    only_ours = sum(ours_won[g] and not grail_won[g] for g in common)
    only_grail = sum(grail_won[g] and not ours_won[g] for g in common)
    print(f"\n  paired with GRAIL on {len(common)} games finished in both arms:")
    print(f"    resistance win rate: ours {sum(ours_won[g] for g in common) / len(common):.1%}, "
        f"GRAIL {sum(grail_won[g] for g in common) / len(common):.1%}")
    print(f"    only ours won {only_ours}, only GRAIL won {only_grail}, "
        f"exact McNemar p = {exact_two_sided(only_ours, only_grail):.4f}")
    paired = sorted(set(ours_accuracy) & set(grail_accuracy))
    if not paired:
        print("    no closing guess of GRAIL's to pair with: no beliefs.csv next to its summaries")
        return
    higher = sum(ours_accuracy[g] > grail_accuracy[g] for g in paired)
    lower = sum(ours_accuracy[g] < grail_accuracy[g] for g in paired)
    print(f"    tested seat's closing accuracy on {len(paired)} games: ours "
        f"{sum(ours_accuracy[g] for g in paired) / len(paired):.1%}, GRAIL "
        f"{sum(grail_accuracy[g] for g in paired) / len(paired):.1%}; ours higher in {higher}, "
        f"lower in {lower}, exact sign test p = {exact_two_sided(higher, lower):.4f}")


if __name__ == "__main__":
    main()
