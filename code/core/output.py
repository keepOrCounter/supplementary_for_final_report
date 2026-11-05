# where an experiment writes its results, for both games
#
# The two loggers, core/logger.py for Tofu Kingdom and core/resistance_logger.py
# for The Resistance, record different things but write them to the same shape of
# directory. What is common lives here and nothing in it reads a game: importing
# it costs a Resistance run nothing, which importing the Tofu logger did, because
# that one names every Tofu agent class and one of them loads torch.
from __future__ import annotations
import csv
import os
from datetime import datetime

import yaml


def create_output_dir(config: dict) -> str:
    """Create timestamped output directory for this experiment.

    Structure:
        results/
            <time>_<model_name>/
                copy_of_config.yaml
                summary.csv
                game_001.json
                game_002.json
                ...

    Returns:
        Path to the created directory.
    """
    base_dir = config.get("logging", {}).get("output_dir", "results")

    # Build descriptive folder name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model = config.get("llm", {}).get("model", "unknown").replace(":", "_").replace("/", "_")
    num_players = config["game"]["num_players"]
    folder_name = f"{timestamp}_{model}_{num_players}p"

    # De-collide: the timestamp is second-granular, so back-to-back fast runs (e.g.
    # the per-seed learning-curve series) would otherwise overwrite each other.
    # Claimed by making it rather than by asking whether it exists, because grid
    # cells run in parallel processes and another one can take the name between
    # the question and the answer.
    output_dir = os.path.join(base_dir, folder_name)
    suffix = 1
    while True:
        try:
            os.makedirs(output_dir)
            break
        except FileExistsError:
            output_dir = os.path.join(base_dir, f"{folder_name}_{suffix}")
            suffix += 1

    # Save a copy of the config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_dir


def save_summary_csv(output_dir: str, all_results: list[dict]) -> None:
    """Save experiment summary as CSV.

    Args:
        output_dir: Directory to write to.
        all_results: List of per-game result dicts.
    """
    if not all_results:
        return

    filepath = os.path.join(output_dir, "summary.csv")
    fieldnames = list(all_results[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
