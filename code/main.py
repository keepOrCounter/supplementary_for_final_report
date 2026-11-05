from __future__ import annotations
import os
import sys
import yaml
import time

from game.tofu_env import TofuKingdomEnv, PRINCE_ID
from agents import BaseAgent, RandomAgent, HumanAgent, LLMAgent, LLMParseError, SymbolicGuessError, \
    BeliefUpdateError, \
    SymbolicPrinceAgent, SymbolicPlayerAgent, FirstOrderSymbolicPlayerAgent, SoftSymbolicPrinceAgent, \
    AblationPrinceAgent, GrailPrinceAgent, EMOPrinceAgent
from game.predefined_data_classes import GameState, GameResult, Faction, Role, QuestionType
from agents.our_agents.Kripke import NeuroSymbolicKripkeState
from agents.our_agents.opponent_memory import CrossGameMemory, OpponentProfiler, update_type_posteriors
from core.llm_client import create_llm_client
from core.helper_functions import SeedSource, action_description
from core.logger import create_output_dir, save_game_record, save_summary_csv, build_summary_row, \
    extract_llm_stats, extract_likelihood_stats


def load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def create_agents_from_config(config: dict, state: GameState, seed: int = None, \
    memory: CrossGameMemory = None) -> dict[int, BaseAgent]:
    """Build agents from config.

    Each agent gets an independent seed stream spawned by player_id from a single
    per-game root (see seeding.SeedSource). Streams are isolated: how many seeds
    one agent draws (e.g. an LLM agent retrying) never shifts another agent's
    stream. So across ablation configs sharing the same game seed, the deal and
    the non-treatment agents behave identically -- enabling low-variance paired
    comparison ("common random numbers").

    Return: dict of player_id to Agent
    """
    agents: dict[int, BaseAgent] = {}
    agent_config = config["agents"]

    # Single per-game root; spawn one child stream per player_id.
    root = SeedSource(seed)

    # Prince (the only agent that consumes the cross-game memory store)
    agents[PRINCE_ID] = make_agent(agent_config["prince"], player_id=PRINCE_ID, role=None, \
        config=config, seed_source=root.spawn(PRINCE_ID), memory=memory)

    # Non prince: repeat the last config when the list is shorter than the number of seats.
    player_configs = agent_config["players"]
    non_prince_ids = sorted(state.roles.keys())
    seat_configs = [player_configs[i] if i < len(player_configs) else player_configs[-1] \
        for i in range(len(non_prince_ids))]

    # Optionally shuffle which opponent config lands in which seat, per game, so a
    # given opponent type is not pinned to a fixed player id (which can bias LLM
    # agents -- especially any with cross-game memory). Drawn from root's OWN stream
    # (the per-seat streams come from root.spawn(pid), independent of root's own draw
    # counter), so it is reproducible from the game seed and perturbs no agent's
    # stream; a no-op when the opponents are homogeneous.
    if agent_config.get("shuffle_opponents", False):
        root.to_random().shuffle(seat_configs)

    for pid, player_config in zip(non_prince_ids, seat_configs):
        agents[pid] = make_agent(player_config, player_id=pid, role=state.roles[pid], \
            config=config, seed_source=root.spawn(pid))

    return agents


def make_agent(agent_config: dict, player_id: int, role: Role, config: dict, seed_source: SeedSource = None, \
    memory: CrossGameMemory = None) -> BaseAgent:
    """create single agent from config"""
    agent_type = agent_config["type"].lower()

    if agent_type == "random":
        return RandomAgent(player_id=player_id, role=role, seed_source=seed_source)

    elif agent_type == "human":
        return HumanAgent(player_id=player_id, role=role)

    elif agent_type == "llm":
        client = create_llm_client(config)
        reasoning_cfg = config.get("llm", {}).get("reasoning_history", {})
        print_out = config["game"].get("print_out", False)
        return LLMAgent(player_id=player_id, role=role, llm_client=client, \
            use_reasoning_history=reasoning_cfg.get("enabled", True), \
                print_out=print_out, seed_source=seed_source)

    elif agent_type == "symbolic":
        game_cfg = config["game"]
        num_players = game_cfg["num_players"]
        use_center_card = game_cfg.get("use_center_card", True)
        print_out = game_cfg.get("print_out", True)
        if player_id == PRINCE_ID:
            return SymbolicPrinceAgent(player_id=player_id, role=role, num_players=num_players, \
                use_center_card=use_center_card, print_out=print_out)
        else:
            return SymbolicPlayerAgent(player_id=player_id, role=role, num_players=num_players, \
                use_center_card=use_center_card, print_out=print_out)

    elif agent_type == "symbolic1":
        # 1st-order ToM opponent: best-responds to an order-0 logical Prince belief --
        # the exact opponent `likelihood: strategic` assumes (model-consistent matchup).
        if player_id == PRINCE_ID:
            raise ValueError("symbolic1 (1st-order ToM player) is only supported for non-prince players")
        game_cfg = config["game"]
        return FirstOrderSymbolicPlayerAgent(player_id=player_id, role=role, \
            num_players=game_cfg["num_players"], use_center_card=game_cfg.get("use_center_card", True), \
                print_out=game_cfg.get("print_out", True))

    elif agent_type == "soft_symbolic":
        if player_id != PRINCE_ID:
            raise ValueError("soft_symbolic agent is only supported for prince (player 0)")
        
        game_cfg = config["game"]
        num_players = game_cfg["num_players"]
        use_center_card = game_cfg.get("use_center_card", True)
        print_out = game_cfg.get("print_out", True)
        soft_cfg = agent_config.get("soft_likelihood", {})
        # The judge cache is class-level on NeuroSymbolicKripkeState, so it is shared
        # across games even though each game builds a fresh agent + llm_client.
        return SoftSymbolicPrinceAgent(player_id=player_id, role=role, num_players=num_players, \
            use_center_card=use_center_card, llm_client=create_llm_client(config), \
            top_k=soft_cfg.get("top_k", 3), weights=soft_cfg.get("answer_weight"), \
            shrink=soft_cfg.get("confidence_shrink"), print_out=print_out, seed_source=seed_source)

    elif agent_type == "ablation":
        if player_id != PRINCE_ID:
            raise ValueError("Ablation agent is only supported for prince (player 0)")
        game_cfg = config["game"]
        num_players = game_cfg["num_players"]
        use_center_card = game_cfg.get("use_center_card", True)
        print_out = game_cfg.get("print_out", True)

        # Ablation-specific params are nested under agents.prince.ablation
        abl_cfg = agent_config.get("ablation", {})
        q_strategy = abl_cfg.get("question_strategy", "ig")
        g_strategy = abl_cfg.get("guess_strategy", "map")
        likelihood = abl_cfg.get("likelihood", "uniform")
        # Strategic-likelihood knobs (only used when likelihood is strategic/strategic3);
        # None falls back to the class defaults.
        tom_beta = abl_cfg.get("tom_beta")
        tom_smoothing = abl_cfg.get("tom_smoothing", "softmax")  # softmax | eps_greedy
        tom_eps = abl_cfg.get("tom_eps")
        give_belief = abl_cfg.get("give_belief_state", True)
        give_hist = abl_cfg.get("give_history", True)
        belief_mode = abl_cfg.get("belief_mode", "distribution")
        give_memory = abl_cfg.get("give_memory", True)  # only matters when a memory store exists
        judge_reasoning_effort = abl_cfg.get("judge_reasoning_effort")
        # Thinking for the soft-likelihood judge sub-calls only
        judge_cot = abl_cfg.get("judge_cot", False)
        # judge_cot: manual chain-of-thought for the judge".
        # bayes_adapt (likelihood=bayes only): mixture | map | joint
        bayes_adapt = abl_cfg.get("bayes_adapt", "mixture")

        # Reasoning history uses same config as LLMAgent (under llm.reasoning_history)
        reasoning_cfg = config.get("llm", {}).get("reasoning_history", {})
        use_reasoning = reasoning_cfg.get("enabled", True)

        # Create LLM client if question/guess strategy needs it, or if the belief
        # update uses an LLM-judged soft likelihood (likelihood=llm / memory, or
        # bayes, whose "llm" type candidate is scored by the memoryless judge).
        needs_llm = q_strategy == "llm" or g_strategy == "llm" or likelihood in ("llm", "memory", "bayes")
        llm_client = create_llm_client(config) if needs_llm else None

        return AblationPrinceAgent(player_id=player_id, role=role, num_players=num_players, \
            use_center_card=use_center_card, question_strategy=q_strategy, guess_strategy=g_strategy, \
                likelihood=likelihood, give_belief_state=give_belief, give_history=give_hist, belief_mode=belief_mode, \
                    llm_client=llm_client, use_reasoning_history=use_reasoning, \
                        print_out=print_out, seed_source=seed_source, tom_beta=tom_beta, \
                            tom_smoothing=tom_smoothing, tom_eps=tom_eps, memory=memory, give_memory=give_memory, \
                                judge_reasoning_effort=judge_reasoning_effort, judge_cot=judge_cot, \
                                    bayes_adapt=bayes_adapt)

    elif agent_type == "grail":
        if player_id != PRINCE_ID:
            raise ValueError("grail agent is only supported for prince (player 0)")
        game_cfg = config["game"]
        num_players = game_cfg["num_players"]
        use_center_card = game_cfg.get("use_center_card", True)
        print_out = game_cfg.get("print_out", True)

        # GRAIL-specific params are nested under agents.prince.grail
        grail_cfg = agent_config.get("grail", {})
        use_llm_prior = grail_cfg.get("use_llm_prior", False)
        algorithm = grail_cfg.get("algorithm", "max")
        # Thinking effort for the 'vibes' LLM prior sub-call only; defaults to "none" like
        # the paper did.
        vibes_reasoning_effort = grail_cfg.get("vibes_reasoning_effort", "none")

        # Optional data-driven factor (analogue of GRAIL's trained NN factor):
        #   factor_function: uniform (default) | likelihood | posterior
        #   - likelihood -> learned P(answer|roles)  (grail/build_likelihood_codebook.py)
        #   - posterior  -> learned P(role|history)  (grail/posterior_factor.py, GRAIL-faithful)
        factor_function = grail_cfg.get("factor_function", "uniform")
        codebook = posterior_model = None
        if factor_function == "likelihood":
            from agents.grail.build_likelihood_codebook import load_codebook
            codebook = load_codebook(grail_cfg["factor_function_path"])
        elif factor_function == "posterior":
            from agents.grail.posterior_factor import PosteriorFactorModel
            posterior_model = PosteriorFactorModel.load(grail_cfg["factor_function_path"])

        llm_client = create_llm_client(config) if use_llm_prior else None
        return GrailPrinceAgent(player_id=player_id, role=role, num_players=num_players, \
            use_center_card=use_center_card, llm_client=llm_client, use_llm_prior=use_llm_prior, \
                algorithm=algorithm, likelihood_codebook=codebook, posterior_model=posterior_model, \
                    vibes_reasoning_effort=vibes_reasoning_effort, print_out=print_out)

    elif agent_type == "emo":
        if player_id != PRINCE_ID:
            raise ValueError("emo agent is only supported for prince (player 0)")
        game_cfg = config["game"]
        num_players = game_cfg["num_players"]
        use_center_card = game_cfg.get("use_center_card", True)
        print_out = game_cfg.get("print_out", True)

        # EMO-specific params are nested under agents.prince.emo
        emo_cfg = agent_config.get("emo", {})
        self_refine_steps = emo_cfg.get("self_refine_steps", 2)   # paper default: 2
        global_refine_steps = emo_cfg.get("global_refine_steps", 2)  # paper default: 2
        # Thinking effort for EMO. Paper runs CoT only for the final selection, so this defaults to "none";
        # the final decision uses llm.reasoning_effort.
        model_reasoning_effort = emo_cfg.get("model_reasoning_effort", "none")

        # Reasoning history: EMO carries (or not) its OWN CoT summary across turns,
        # INDEPENDENTLY of the LLM opponents. agents.prince.emo.use_reasoning_history
        # overrides the global llm.reasoning_history.enabled (which the opponents use);
        # omit it to inherit that global default.
        global_reasoning = config.get("llm", {}).get("reasoning_history", {}).get("enabled", True)
        use_reasoning = emo_cfg.get("use_reasoning_history", global_reasoning)

        return EMOPrinceAgent(player_id=player_id, role=role, num_players=num_players, \
            use_center_card=use_center_card, llm_client=create_llm_client(config), \
                self_refine_steps=self_refine_steps, global_refine_steps=global_refine_steps, \
                    use_reasoning_history=use_reasoning, model_reasoning_effort=model_reasoning_effort, \
                        print_out=print_out, seed_source=seed_source)

    else:
        raise ValueError(f"Invalid agent type: '{agent_type}'!!!!!!!")


def run_experiment(config: dict, return_stats: bool = False):
    """run all games defied in config, return list of results.

    If return_stats=True, return (results, stats) where stats aggregates this
    experiment's LLM and soft-likelihood usage (used by batch_runner)."""
    
    game_config = config["game"]
    num_players = game_config["num_players"]
    num_games = game_config["num_games"]
    seed = game_config["seed"]
    print_out = game_config["print_out"]

    # Set up logging
    logging_cfg = config.get("logging", {})
    save_results = logging_cfg.get("enabled", True)
    output_dir = None
    if save_results:
        output_dir = create_output_dir(config)
        print(f"results will be saved to: {output_dir}")

    env = TofuKingdomEnv(num_players=num_players, \
        use_center_card=game_config.get("use_center_card", True))
    results: list[GameResult] = []
    summary_rows: list[dict] = []
    failed_games: int = 0

    # Cross-game opponent memory: ONE store per experiment, accumulated over its
    # games (this run IS one learning series) and reset with the next experiment --
    # unless load_path warm-starts it from a previous run's memory.json.
    memory_cfg = config["agents"]["prince"].get("memory", {})
    memory = None
    profiler = None
    if memory_cfg.get("enabled", False):
        window = memory_cfg.get("window", 0)
        report_mode = memory_cfg.get("report", "raw")
        load_path = memory_cfg.get("load_path")
        memory = CrossGameMemory.load(load_path, window=window, report_mode=report_mode) if load_path \
            else CrossGameMemory(window=window, report_mode=report_mode)

        # Optional LLM profiler (the "summarised strategy portrait" alternative):
        # one profile-update call per opponent after every completed game.
        profile_cfg = memory_cfg.get("profile", {})
        if profile_cfg.get("enabled", False):
            profiler = OpponentProfiler(create_llm_client(config), seed_source=SeedSource(game_config["seed"]), \
                    reasoning_effort=profile_cfg.get("reasoning_effort", "none"), print_out=print_out)
        elif report_mode != "raw":
            raise ValueError(f"memory.report='{report_mode}' shows LLM profiles: enable agents.prince.memory.profile as well")
    if memory is not None and config["agents"].get("shuffle_opponents", False):
        # Memory is keyed by seat (player id).
        raise ValueError("cross-game memory requires fixed opponent seats: "
            "disable agents.shuffle_opponents or agents.prince.memory.enabled")

    # Bayesian type memory (likelihood=bayes): score every finished game against the candidate type opponent
    # models, adapt opponent models in game the prince uses. The "llm" candidate is scored by the memoryless
    # judge, so this makes LLM calls (cached by situation); skipped for every other likelihood.
    abl_cfg = config["agents"]["prince"].get("ablation", {})
    use_adaptive_likelihood = memory is not None and abl_cfg.get("likelihood") == "bayes"

    # Experiment-wide usage totals (summed over games; for return_stats / batch).
    exp_llm = {"total_calls": 0, "retry_count": 0, "total_tokens": 0, "total_prompt_tokens": 0, \
        "total_completion_tokens": 0}
    exp_lk = {"queries": 0, "judge_calls": 0, "cache_hits": 0, "defaulted_slots": 0}

    # The soft_symbolic Prince's judge cache is class-level on NeuroSymbolicKripkeState,
    # so it persists across games within this run; start each run from a clean cache.

    for game_num in range(num_games):
        seed = seed + 1
        NeuroSymbolicKripkeState.clear_cache()

        state = env.reset(seed=seed)
        agents = create_agents_from_config(config, state, seed=seed, memory=memory)

        if print_out:
            print()
            print("+++++++++++++++New Game+++++++++++++++++++++++++")
            print(f"Game {game_num + 1}/{num_games} (seed={seed})")
            print()
            # print(f"Roles:{state.roles}")
            print(f"Roles: { {pid: r.name for pid, r in state.roles.items()} }")
            if state.center_card:
                print(f"Center Card: {state.center_card.name}")
            print()

        # main loop
        action_log = []
        # game_failed = False
        fail_reason = ""
        game_start_time = time.time()

        try:
            while not env.done:
                actor_id = env.state.current_actor
                agent = agents[actor_id]
                obs = env.get_observation(actor_id)
                valid_actions = env.get_valid_actions(actor_id)
                action = agent.take_action(obs, valid_actions)

                action_log.append({
                    "actor": actor_id,
                    "agent_type": agent.__class__.__name__,
                    "action": action,
                })

                if print_out:
                    print(f"{agent}: {action_description(action)}")

                env.step(actor_id, action)

        except (LLMParseError, SymbolicGuessError, BeliefUpdateError) as e:
            duration = time.time() - game_start_time
            # game_failed = True
            fail_reason = str(e)
            failed_games += 1
            print()
            print(f"[SKIP Game {game_num + 1} ] due to exception: {e}")

            # Capture usage even though the game failed: the agents' LLM clients already
            # accumulated token counts from every attempt (including the failed/retried
            # calls that caused this exception), so recording them here avoids
            # under-counting tokens for failed games.
            llm_stats = extract_llm_stats(agents)
            likelihood_stats = extract_likelihood_stats(agents)
            for k in exp_llm:
                exp_llm[k] += llm_stats.get("game_total", {}).get(k, 0)
            for k in exp_lk:
                exp_lk[k] += likelihood_stats.get(k, 0)

            # record the failed game but do not include in results
            if save_results and output_dir:
                save_game_record(output_dir, game_num, seed, env.state, None, agents, action_log, \
                    failed=True, fail_reason=fail_reason, duration_seconds=duration)
                summary_rows.append(build_summary_row(game_num, seed, env.state, None, \
                    failed=True, fail_reason=fail_reason, duration_seconds=duration, \
                    llm_stats=llm_stats, likelihood_stats=likelihood_stats))
            continue

        duration = time.time() - game_start_time

        # if game_failed:


        result = env.get_result()
        results.append(result)

        # Post-game review: fold this game's revealed roles + answers into the
        # cross-game memory for the NEXT games, then let the optional profiler
        # update the LLM-written opponent profiles. The type scorer reuses the
        # prince's memoryless judge (same warm cache) to score the "llm" candidate.
        if memory is not None:
            memory.record_game(env.state)
            if use_adaptive_likelihood:
                update_type_posteriors(memory, env.state, llm_state=agents[PRINCE_ID].kripke.llm_state, \
                    tom_beta=abl_cfg.get("tom_beta"), \
                    tom_smoothing=abl_cfg.get("tom_smoothing", "softmax"), tom_eps=abl_cfg.get("tom_eps"))
            if profiler is not None:
                profiler.update(memory)

        if print_out:
            print()
            p = result.actual_princess
            print(f"Result: winner={result.winner_faction.value}, ", \
                    f"guessed=P{result.guessed_target}, ", \
                        f"princess={'center' if p == -1 else f'P{p}'}, ", \
                            f"duration={duration:.1f}s")
        print()
        print("QA History:")
        if env.state.qa_history:
            for qa in env.state.qa_history:
                if qa.question_type==QuestionType.Q3:
                    print(f"Round {qa.round}: Prince ask P{qa.target_id} {qa.question_type.value}", \
                            f"{f' about P{qa.question_param}' if qa.question_param and qa.question_param != -1 \
                                else 'about center'}", f" --> {qa.answer}")
                else:
                    print(f"Round {qa.round}: Prince ask P{qa.target_id} {qa.question_type.value}", \
                        f" --> {qa.answer}")
        print()

        # Per-game usage stats (computed always; cheap) and accumulated for the run.
        llm_stats = extract_llm_stats(agents)
        likelihood_stats = extract_likelihood_stats(agents) # {} unless soft_symbolic prince
        for k in exp_llm:
            exp_llm[k] += llm_stats.get("game_total", {}).get(k, 0)
        for k in exp_lk:
            exp_lk[k] += likelihood_stats.get(k, 0)

        # Save this game
        if save_results and output_dir:
            save_game_record(output_dir, game_num, seed, env.state, result, agents, action_log, \
                duration_seconds=duration)
            summary_rows.append(build_summary_row(game_num, seed, env.state, result, \
                duration_seconds=duration, llm_stats=llm_stats, likelihood_stats=likelihood_stats))

    # Save summary CSV
    if save_results and output_dir and summary_rows: # TODO move this to logger.py?
        save_summary_csv(output_dir, summary_rows)

    # Persist the accumulated cross-game memory (what the Prince "learned");
    # a later run can warm-start from this file via agents.prince.memory.load_path.
    if memory is not None and save_results and output_dir:
        memory.save(os.path.join(output_dir, "memory.json"))

    if profiler is not None:
        profiler_stats = profiler.llm_client.stats.to_dict()
        for k in exp_llm:
            exp_llm[k] += profiler_stats.get(k, 0)
        if profiler.failed_updates > 0:
            print()
            print(f"[NOTICE] {profiler.failed_updates} profile updates failed (previous profiles kept).")

    if failed_games > 0:
        print()
        print(f"[NOTICE] {failed_games}/{num_games} games failed due to agent errors.")

    # Experiment-wide soft-likelihood cost: the judge cache is class-level, so its
    # size is the number of distinct situations the LLM was ever asked to judge.
    if NeuroSymbolicKripkeState._judge_cache:
        print()
        print(f"[Soft-likelihood] distinct judged situations (LLM calls this run): ",\
            f"{len(NeuroSymbolicKripkeState._judge_cache)}")

    if return_stats:
        hit_rate = exp_lk["cache_hits"] / exp_lk["queries"] if exp_lk["queries"] else 0.0
        stats = {"llm": exp_llm, "likelihood": {**exp_lk, "cache_hit_rate": round(hit_rate, 4)}}
        return results, stats
    return results


def print_summary(results: list[GameResult], config: dict, total_games: int = None) -> None:
    """overall experiments result"""
    n = len(results)
    game_config = config["game"]
    total = total_games or game_config["num_games"]
    failed = total - n

    wins = {Faction.PRINCESS: 0, Faction.QUEEN: 0, Faction.OTHER: 0}
    for r in results:
        wins[r.winner_faction] += 1

    print()
    print("="*100)
    print(f"Experiment Summary")
    print(f"{game_config['num_players']} players, {total} games total, {n} completed, {failed} failed")
    print()

    # Agent types
    agent_config = config["agents"]
    print(f"Prince agent: {agent_config['prince']['type']}")
    player_types = [p["type"] for p in agent_config["players"]]
    print(f"Other Players' agents: {player_types}")
    print()

    # Win rates
    if n > 0:
        for faction in [Faction.PRINCESS, Faction.QUEEN, Faction.OTHER]:
            count = wins[faction]
            print(f"{faction.value}: {count} wins({count/n:.2%})")

        print()
        correct = sum(1 for r in results if r.winner_faction == Faction.PRINCESS)
        print(f"Prince found Princess: {correct}/{n} ({correct/n:.1%})")
    else:
        print("No completed games to summarize.")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config.yaml> [--print_out]")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    # print(config)

    if "--print_out" in sys.argv:
        config["game"]["print_out"] = True

    results = run_experiment(config)
    print_summary(results, config)


if __name__ == "__main__":
    main()