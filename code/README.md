# Supplementary code and data

This archive holds the code for both games in the paper, Tofu Kingdom and The
Resistance, and the result files of the reported experiments. Part 1 covers
Tofu Kingdom. Part 2 covers The Resistance. The section "What was left out of
this archive" lists what is not included.
The technical appendix is `technical_appendix.pdf` in this folder.

Some code comments cite files that are not in this archive: `docs/*.md`,
`test/*`, `agents/resistance/grail/UPSTREAM.md`,
`experiment_grids/consistency/README.md`, and the helper scripts `beta_mle.py`,
`check_resistance_belief.py` and `continue_runs.py`. These are internal notes
and tools. The `README.md` cited in `agents/resistance/our_agents/kripke.py` and
the README cited in `agents/emo/__init__.py` are an internal README, not this
file.

On Windows, set `PYTHONUTF8=1` so that the YAML files are read as UTF-8.

# Part 1: Tofu Kingdom

## The grid format

A grid file is a `base:` block, which is one complete game config with exactly
the schema of `Config.yaml`, plus a `sweep:` block. Sweep keys are **dotted paths
into `base`**, each holding a list of values. The runner takes the **Cartesian
product** of those lists, overrides the named paths on a copy of `base` for each
combination, and runs `num_games` games per combination.

The two annotated files in the archive root are the reference for the format
and for what the parameters mean. Read them rather than this page:

- **`Config.yaml`**: a complete single-run config with most fields documented
  inline: the agent types, the `ablation` factors (`question_strategy`,
  `guess_strategy`, and `likelihood` with what each value models), the GRAIL and
  EMO blocks, cross-game `memory`, the ToM smoothing knobs, and the `llm`
  settings.
- **`experiment_grid.yaml`**: a worked sweep, one comment per factor, showing
  the dotted-path convention and how conditions map onto combinations.

## Prerequisites

```bash
pip install -r requirements.txt        # Python >= 3.12
```

Anything with an LLM in it needs a live Ollama server at `llm.base_url`. This
covers `type: llm` opponents, `question_strategy: llm`, `guess_strategy: llm`,
`likelihood: llm`/`memory`/`bayes`, the `soft_symbolic` Prince, LLM-written
memory profiles, the EMO Prince, and the GRAIL Prince with `use_llm_prior: true`
(as in every shipped GRAIL run). Purely symbolic configs (`ig`/`random`
questions + `map` guess + `uniform`/`strategic`/`strategic3` likelihood, against
`random`/`symbolic`/`symbolic1` opponents) run on CPU with no server.

**The context window must be baked into the model, not passed per request.** The
OpenAI-compatible `/v1` endpoint silently drops Ollama generation options, so
`num_ctx` and `repeat_penalty` have to come from a Modelfile. Build a variant
with `Modelfile.template` and point `llm.model` at it. `ollama ps` shows the
context actually in force. A prompt over the window is truncated **without any
error**, so verify rather than assume.

## Running

```bash
# one config, one process
python3 main.py <config.yaml> [--print_out]

# a whole grid
python3 batch_runner.py <grid>.yaml
python3 batch_runner.py <grid>.yaml --dry-run   # list combos, run nothing
```

`main.py` **requires** the config path. It does not fall back to `Config.yaml`.
`--print_out` turns on the per-game console printout, whatever the config says.
The per-game records are written either way.

Always `--dry-run` first. `num_games` is **per combination**, so a 36-combo grid
at 100 games is 3600 games, and the combo count is the product of every sweep
list.

## Reading results

Everything lands under `logging.output_dir` (all `results/*`).

```
results/<campaign>/
    <timestamp>_<model>_<Np>/        one run = one combination
        config.yaml                  the config actually executed. Authoritative.
        summary.csv                  per-game rows
        game_0.json ...              full transcript, beliefs, LLM calls per game
                                     (not shipped for Tofu Kingdom, see "What was left out")
        memory.json                  cross-game memory runs only, with the LLM-written
                                     opponent profiles when those are enabled
    batch_results_<ts>.csv           one row per combination
    batch_summary_<ts>.yaml          same plus params, LLM/judge counters, best cell
    charts/*.png
```

Some older campaigns do not follow this naming, for example
`grid_grail(LLM variant)/player=random/`. Their `config.yaml` is still
authoritative.

`batch_summary_*.yaml` is the most useful single file. Per cell it carries
`prince_accuracy`, the faction win split, `duration_seconds`, `llm_calls`,
`llm_total_tokens`, the judge counters (`lk_judge_calls`, `lk_cache_hit_rate`,
`lk_queries`, `lk_defaulted_slots`), and the exact `params` that produced it.

### `analyze.py`: three modes, auto-detected

```bash
python3 analyze.py results/<campaign>/<run>/summary.csv        # single run
python3 analyze.py results/<campaign>/batch_results_<ts>.csv   # whole batch
python3 analyze.py --learning-curve results/grid_memory_cot [more dirs...]
```

Batch mode is detected by the presence of a `prince_accuracy` column. Charts are
written next to the CSV under `charts/`. Single mode gives
`win_distribution.png` and `accuracy_over_time.png`. Batch mode gives
`ranking.png`, `factor_main_effects.png`, `factor_impact.png` and
`accuracy_vs_cost.png`. Learning-curve mode writes `learning_curve.png` into the
first directory given, averaging over the seed series it finds and grouping by
Prince config. Pass several directories to plot arms from different campaigns on
one axis.

### `search_results.py`: find runs by config, not by folder name

Run directories are named by timestamp/model/player-count, which says nothing
about the strategy. This scans every `config.yaml` under a root, joins it with
that run's accuracy, and filters:

```bash
python3 search_results.py -q ig -g map -l strategic          # named factors
python3 search_results.py --where agents.prince.ablation.tom_eps=0.2
python3 search_results.py --where llm.reasoning_effort~=high # ~= contains
python3 search_results.py --where 'game.num_games>=50'       # >= <= > < = also work
python3 search_results.py -q ig --min-acc 0.5 --sort acc     # gates + sorting
python3 search_results.py -l bayes --json                    # machine-readable
```

Filters: `--prince --question/-q --guess/-g --likelihood/-l --tom --factor
--adapt --model --players --num-players --memory/--no-memory --where` (repeatable).
Gates: `--min-acc --max-acc --min-games`. Output: `--sort date|acc|games|dir
--limit --full --json --count`. First positional arg is the root (default
`results`). It reads Tofu Kingdom configs. The Resistance runs under `results/`
show up with blank fields.

# Part 2: The Resistance (6 players)

## Files

| Path | Purpose |
|---|---|
| `game/resistance_env.py`, `game/resistance_data_classes.py` | The game environment. |
| `agents/resistance/base.py` | The agent base class and the random player. |
| `agents/resistance/ablation_agent.py` | One agent whose propose, vote and mission strategies are each `eu` (our belief-space planner), `llm` or `random`. With `eu`, `belief_model` picks the belief: `random`, `uniform` (the default), `strategic`, `llm` or `bayes`. The LLM baseline is this agent with all three strategies set to `llm`. |
| `agents/resistance/our_agents/` | Exact belief over role assignments, the order-2 strategic model of the Symbolic players, the LLM likelihood judge and the Bayesian strategy profile with its cross-game memory. |
| `agents/resistance/grail/` | Port of GRAIL in its Graph Only form (see Third-party code). |
| `run_resistance.py` | Run one config: `python3 run_resistance.py [config.yaml]`. |
| `batch_runner_resistance.py` | Run a grid. Same grid format as Part 1. `--dry-run` lists the trials. |
| `analyze_resistance.py` | Summaries, per-agent identity accuracy, learning curves. |
| `tools/merge_final_llm.py` | Merge the six trials of the LLM likelihood run and pair it with the GRAIL run. |
| `Config_resistance.yaml` | Default single-run config, with most keys explained inline. The seat keys of the reported runs (`agents.seats`, `faction`, `agents.rotate_seats`, and `strategic3` in the Bayes run) are explained in the grids and in `run_resistance.py`. |
| `experiment_grids/` | The grids of the three reported runs. |

The Resistance code imports Tofu Kingdom code. Importing `agents.resistance`
runs `agents/__init__.py`, which loads the Tofu Kingdom agents. The GRAIL port
uses the engine in `agents/grail/pomegranate/`. `batch_runner_resistance.py`
reuses `batch_runner.py`, which imports `main.py`. So both games ship together.

## Setup

The prerequisites are those of Part 1. All three grids use
`llm.model: qwen3.5-9b-ctx12288:latest` at `http://localhost:11434/v1`. Build the
model from `Modelfile.template`, check the context, and dry-run a grid:

```bash
ollama pull qwen3.5:9b-q8_0
ollama create qwen3.5-9b-ctx12288 -f Modelfile.template
ollama ps                     # after one request, CONTEXT must read 12288
python3 batch_runner_resistance.py experiment_grids/grid_resistance_final.yaml --dry-run
```

## The three reported runs

All three runs use the same 180 game seeds: 6 trials of 30 games. The trial seeds
are 910001, 911001, ..., 915001, and game i of a trial uses seed + 1 + i.
`agents.rotate_seats` turns the line-up by one seat per trial, so the controlled
agent sits at player id k in trial k and keeps that seat for all 30 games of the
trial. The controlled agent always plays for the resistance.

The other five seats are the same in all three runs:

- one LLM baseline (`propose_strategy`, `team_vote_strategy` and
  `mission_strategy` all `llm`),
- two Symbolic players (belief-space planner `eu` with `belief_model: strategic`, order 2),
- two random players.

The three grids differ only in seat 0 and in `logging.output_dir`, so games pair
across runs by seed.

| Run | Controlled agent | Grid | Results |
|---|---|---|---|
| GRAIL | `type: grail` | `experiment_grids/grid_resistance_final.yaml` | `results/final_grail_mixed/` |
| LLM likelihood | belief-space planner (`eu`) with the LLM opponent model (`belief_model: llm`) | `experiment_grids/final_llm/t0.yaml` to `t5.yaml` | `results/final_llm_mixed/t{k}/` |
| Bayes | belief-space planner (`eu`) with the Bayesian strategy profile (`belief_model: bayes`) | `experiment_grids/grid_resistance_final_bayes.yaml` | `results/final_bayes_mixed/` |

```bash
python3 batch_runner_resistance.py experiment_grids/grid_resistance_final.yaml
for k in 0 1 2 3 4 5; do
  python3 batch_runner_resistance.py experiment_grids/final_llm/t$k.yaml
done
python3 batch_runner_resistance.py experiment_grids/grid_resistance_final_bayes.yaml
```

A rerun writes new run directories next to the shipped ones. Move the shipped
results aside first, or `tools/merge_final_llm.py` will stop on the duplicate
trials.

We ran the LLM likelihood run as six files, one per trial. Together they equal
`experiment_grids/grid_resistance_final_llm.yaml`, except that each file holds one
trial, writes it to its own directory and turns printing on. Printing does not
change the games. A trial writes `summary.csv` only after all its 30 games.

## Results layout

```
results/<run>/<timestamp>_<model>_6p/     one trial
    config.yaml        the config actually executed. Authoritative.
    summary.csv        one row per game
    beliefs.csv        one row per belief snapshot
    game_000.json ...  the full record of each game, including the LLM text
    memory.json        Bayes run only: the type posteriors learned over the trial
results/<run>/batch_resistance_<timestamp>/
    batch_summary.csv  one row per trial
    batch_beliefs.csv  the belief rows of every trial
results/final_llm_mixed/t{k}/             trial k of the LLM likelihood run
results/final_llm_mixed/merged/           written by tools/merge_final_llm.py
```

Points to know when reading these files:

- **Identity accuracy.** `identity_accuracy` in `summary.csv` pools the last
  snapshot of every resistance player that holds a belief (the random players
  hold none). In `batch_summary.csv` it is the mean of that value over the
  finished games of a trial. The accuracy of one agent comes from its rows in
  `beliefs.csv`: its last snapshot in each finished game. `analyze_resistance.py`
  and `tools/merge_final_llm.py` report it per agent.
- **Crashed games.** A game is marked `crashed` when it raises an exception, and
  `crash_reason` holds the error. In the three runs every crash was an LLM call
  that still failed after its retries (`InstructorRetryException`). Some of
  these calls came after the game was decided: the closing belief query, or, in
  three Bayes games, the scoring of the finished game for the cross-game memory. Such
  games still record a `winner`. The scripts leave every crashed game out of win
  rates and of the closing identity accuracy. The accuracy-by-missions table of
  `analyze_resistance.py` keeps their snapshots, and the LLM cost counts them.
- **Agent labels.** In the GRAIL run the two Symbolic players are labelled
  `AblationResistanceAgent(eu/eu/eu)`. In the other two runs the same agent is
  labelled `AblationResistanceAgent(eu/eu/eu,belief=strategic)`. The label
  changed in a later version of the code. The agent did not. For the same reason
  the GRAIL run's `summary.csv` has no `lk_*` columns.
- **Bayes batch files.** The Bayes run was launched twice: trials 0 to 3, then
  trials 4 and 5. Its `batch_resistance_*` files join the two launches. Each row
  matches the trial's own files.
- **Merged LLM files.** In `merged/summary.csv` and `merged/beliefs.csv`,
  `game_index` is renumbered to `trial * 30 + game_index`, so that games of
  different trials do not share an index.

## Recomputing the reported numbers

Run these from the archive root. They read only the CSV, YAML and `memory.json`
files.

```bash
# per trial: wins, crashes, Wilson intervals, accuracy per agent, LLM cost, time
python3 analyze_resistance.py results/final_grail_mixed/batch_resistance_2026-09-17_05-50-16/batch_summary.csv
python3 analyze_resistance.py results/final_bayes_mixed/batch_resistance_2026-10-05_01-38-33/batch_summary.csv

# merge the LLM likelihood trials, check them, and pair them with the GRAIL run
python3 tools/merge_final_llm.py --grail results/final_grail_mixed
python3 analyze_resistance.py results/final_llm_mixed/merged/batch_summary.csv
python3 analyze_resistance.py results/final_llm_mixed/merged/summary.csv

# learning curves over the 30 games of a trial
python3 analyze_resistance.py --learning-curve results/final_bayes_mixed results/final_llm_mixed results/final_grail_mixed

# the type posteriors of one Bayes trial
python3 analyze_resistance.py results/final_bayes_mixed/<run>/memory.json
```

`tools/merge_final_llm.py` rewrites `merged/` from the six trial directories. It
writes nothing unless every consistency check passes. With `--grail` it pairs
the two runs game by game by seed: an exact McNemar test on who won and an exact
sign test on the controlled agent's closing identity accuracy. A game that crashed
in either run is left out of both. The shipped CSV files of this run use LF line endings.
Regenerated ones use CRLF and are otherwise identical.

`tools/merge_final_llm.py` pairs only the LLM likelihood run with the GRAIL run.
No shipped script pairs the Bayes run with another run.

## Reproducibility

- **Seeds.** The role assignment and every non-LLM choice derive from the game seed. LLM
  calls pass a seed. LLM outputs can still differ across Ollama versions, model
  builds and hardware, so a rerun is not expected to replay the same games.
- **Code version.** The archive holds the final code. For each run, the code it
  executes is the same here as when it ran, apart from refactoring, input
  checks, log labels and added output columns. Agents added later are not used
  by the earlier runs.
- **Pairing.** The 180 role assignments are the same in all three runs. In every game where
  two runs take different actions, the first difference is an action of the
  controlled seat.

## Third-party code

- `agents/resistance/grail/` ports GRAIL in its Graph Only form (belief propagation without the language prior) from the
  Bayesian-Avalon repository, MIT licence (`LICENSE.bayesian-avalon`).
  `base_model.py`, `model_reduced_categories.py` and `heuristic.py` are upstream
  files, and each edit to them is marked `[Resistance port]`. `agent.py`,
  `ateam.py` and `state_vector.py` are our own. The factor weights in
  `models/v2/` are upstream's, unchanged.
- `agents/grail/pomegranate/` is the fork of pomegranate shipped with the same
  repository (`LICENSE.bayesian-avalon`). One edit, marked `[Tofu Kingdom port]`
  in `_utils.py`, makes the `apricot` import optional. pomegranate itself is by
  Jacob Schreiber under the MIT licence (`LICENSE.pomegranate`).

# What was left out of this archive

- **Tofu Kingdom per-game JSON files.** No `game_*.json` file ships for Tofu
  Kingdom, because together they exceed the size limit. Every campaign keeps its
  `config.yaml`, its `summary.csv` (one row per game), its charts, and its batch
  files and `memory.json` where present. The per-game records that the
  technical appendix cites are the rows of each run's `summary.csv`. Its columns
  carry the `game_*.json` fields the appendix names: `result.correct` is
  `correct`, `status` is `status`, `llm_stats.game_total.total_calls` is
  `llm_calls`, and `llm_stats.game_total.total_prompt_tokens` is
  `llm_prompt_tokens`. Runs of the older config generation have no `llm_*`
  columns, and their `game_*.json` had no `llm_stats` either. The per-seat LLM
  counts (`llm_stats.per_agent`), each game's seat layout (`agent_types`) and the
  full game transcripts are not in this archive.
- **Resistance console logs.** The printed logs of the runs are left out.
  Everything the analysis reads is in the CSV, YAML and JSON files.
- **Resistance charts.** The PNG charts of the three runs are left out.
  `analyze_resistance.py` draws them again.
- **Development files.** Tests, internal design notes, helper scripts other than
  `tools/merge_final_llm.py`, and the grid files of the Tofu Kingdom campaigns
  are not included. Each run's `config.yaml` records the config it ran.

The result files are otherwise as the runs wrote them, with three exceptions.
Line endings were not kept: some CSV files use LF and others CRLF. In three Tofu Kingdom `config.yaml` files, the
absolute path in `factor_function_path` is shortened to a relative path. In crash
and failure messages, the Unix time of the failed LLM response (`created=...`)
is replaced by `created=0`.

# Script index

| File | Purpose |
|---|---|
| `main.py` | Tofu Kingdom: run one config. Game loop, agent construction, logging. |
| `batch_runner.py` | Tofu Kingdom: expand a grid into combinations and run each as its own experiment. `--dry-run` only lists them. |
| `analyze.py` | Tofu Kingdom: summaries and charts for a single run, a batch, or a learning curve. |
| `search_results.py` | Tofu Kingdom: locate run directories by their recorded config; filter, sort, dump. |
| `Config.yaml` | Tofu Kingdom: default single-run config. Every grid's `base:` block follows its schema, and most parameters are explained inline. |
| `experiment_grid.yaml` | Tofu Kingdom: a worked example grid. Its sweep crosses the opponent model (`agents.prince.ablation.likelihood`) with the opponent line-up (`agents.players`): 4 combinations of 20 five-player games that need no LLM. |
| `run_resistance.py` | The Resistance: run one config. |
| `batch_runner_resistance.py` | The Resistance: run a grid. |
| `analyze_resistance.py` | The Resistance: summaries, per-agent accuracy, learning curves, type posteriors. |
| `tools/merge_final_llm.py` | The Resistance: merge and check the LLM likelihood trials; pair them with GRAIL. |
| `Config_resistance.yaml` | The Resistance: default single-run config, with most keys explained inline. |
| `Modelfile.template` | Ollama model variant with `num_ctx` baked in. |
