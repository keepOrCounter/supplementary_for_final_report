# entry point for Resistance experiments
# Mirrors main.py, which runs the Tofu Kingdom game. The two share core/.
#
#   python3 run_resistance.py [Config_resistance.yaml]
from __future__ import annotations
import os
import random
import sys
import time
import yaml

from agents.resistance import AblationResistanceAgent, ResistanceAgent, RandomResistanceAgent
from agents.resistance.ablation_agent import DEFAULT_BELIEF_MODEL, DEFAULT_STRATEGY, STRATEGY_KEYS
from agents.resistance.our_agents import BeliefResistanceAgent, NeuroSymbolicKripkeState
from agents.resistance.our_agents.bayes import strategic_knobs
from agents.resistance.our_agents.opponent_memory import TypeMemory, update_type_posteriors
from core.helper_functions import SeedSource
from core.llm_client import create_llm_client
from core.resistance_logger import (
    create_output_dir, save_summary_csv, save_game_record, build_summary_row,
    build_belief_rows, save_belief_csv, serialize_action, extract_llm_stats,
)
from game.resistance_data_classes import MissionRecord, Phase, ResistanceState, Team
from game.resistance_env import ResistanceEnv

DEFAULT_CONFIG = "Config_resistance.yaml"

AGENT_TYPES = {
    "random": RandomResistanceAgent,
    "ablation": AblationResistanceAgent, # one strategy per decision, either faction
    "belief": BeliefResistanceAgent, # our exact belief on the random policy, resistance only
}
GRAIL_TYPE = "grail" # resistance players only, imported on demand by load_agent_class

# Every llm setting this game reads, which is every argument create_llm_client
# passes on. A grid may sweep any of them.
LLM_SETTINGS = frozenset({"model", "base_url", "api_key", "max_retries", "temperature", \
    "top_p", "max_tokens", "reasoning_effort"})


def load_agent_class(agent_type: str) -> type[ResistanceAgent]:
    """the class one config entry names

    grail is imported here and not at the top of the file. Importing it loads
    torch and a belief propagation engine, which costs seconds and a few hundred
    megabytes, and a line-up without a grail player should not pay for that. It
    joins AGENT_TYPES once imported, so a run pays that cost at most once.
    """
    if agent_type == GRAIL_TYPE and GRAIL_TYPE not in AGENT_TYPES:
        from agents.resistance.grail.agent import GrailResistanceAgent
        AGENT_TYPES[GRAIL_TYPE] = GrailResistanceAgent
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {agent_type}!!!!!!!! "
            f"known: {sorted(set(AGENT_TYPES) | {GRAIL_TYPE})}")
    return AGENT_TYPES[agent_type]


def check_llm_config(config: dict) -> None:
    """raise on an llm setting this game does not read

    A grid that sweeps a setting nothing reads runs the same cell twice and
    reports it as two, which is worse than a crash because the table looks fine.
    Tofu Kingdom reads settings this game does not, so a config copied across
    fails here rather than silently doing nothing.
    """
    unknown = sorted(set(config.get("llm", {})) - LLM_SETTINGS)
    if unknown:
        raise ValueError(f"llm settings this game does not read: {unknown}!!!!!!!! known: {sorted(LLM_SETTINGS)}")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_agent(agent_config: dict, player_id: int, seed_source: SeedSource, \
    env: ResistanceEnv, config: dict, both_factions: bool = False, \
        memory: TypeMemory = None) -> ResistanceAgent:
    """build one player_id's agent from its config entry

    check_rules runs before the agent is built, so an agent that cannot play
    under this experiment's rules stops it here rather than mid game.

    both_factions: this agent is seated by player id and the deal is random, so
        it has to be able to play either side. Checked against both rather than
        against the one it happens to hold this game, or a config that cannot
        play spy would run until the first game that dealt it one.

    An agent that needs an LLM gets its own client, built from the config's
    llm block. The agent reading belief_model: bayes gets memory, the run's type
    memory.
    """
    agent_type = agent_config["type"]
    agent_class = load_agent_class(agent_type)
    factions = tuple(Team) if both_factions else (env.team_of(player_id),)
    for faction in factions:
        agent_class.check_rules(env, faction, agent_config)
    options = dict(agent_config)
    options.pop("type")
    options.pop("faction", None) # read by the deal, not by the agent
    if agent_class.needs_llm(agent_config):
        # batch_runner fills the llm block's defaults but never model, so a grid
        # that forgot the block still has one and dies on a bare KeyError inside
        # the first game instead of here.
        if not config.get("llm", {}).get("model"):
            raise ValueError(f"agent type {agent_type} needs llm.model in the config, "
                "and a grid carries its own llm block rather than reading Config_resistance.yaml")
        options["llm_client"] = create_llm_client(config)
    if agent_config.get("belief_model") == "bayes":
        options["memory"] = memory
    return agent_class(player_id=player_id, seed_source=seed_source, **options)


def pinned_spies(config: dict, seed: int, num_players: int, num_spies: int) -> frozenset[int]:
    """which players hold a Spy card, when some seats have their faction pinned

    Returns None when nothing is pinned, and then the environment deals as usual.
    """
    seats: list[dict] = config["agents"].get("seats")
    if not seats:
        return None
    if len(seats) != num_players:
        # The same check create_agents makes, and here too because the deal is
        # read first: indexing a short list would raise IndexError from inside
        # the runner instead of naming what is wrong with the config.
        raise ValueError(f"agents.seats needs one entry per player, got {len(seats)} for {num_players}")
    forced, free = [], []
    for player_id in range(num_players):
        faction = seats[player_id].get("faction")
        if faction == Team.SPIES.value:
            forced.append(player_id)
        elif faction is None:
            free.append(player_id)
        elif faction != Team.RESISTANCE.value:
            raise ValueError(f"agents.seats[{player_id}].faction must be "
                f"{Team.RESISTANCE.value} or {Team.SPIES.value}, got {faction!r}")
    if not forced and len(free) == num_players:
        return None # nothing pinned, so let the environment deal
    if len(forced) > num_spies:
        raise ValueError(f"{len(forced)} seats are pinned to {Team.SPIES.value} "
            f"and a {num_players} player game has {num_spies}")
    if len(free) < num_spies - len(forced):
        raise ValueError(f"too many seats are pinned to {Team.RESISTANCE.value}: "
            f"{num_spies - len(forced)} spy cards have {len(free)} seats to go to")
    # Its own stream, not random.Random(seed). ResistanceEnv.reset draws the
    # leader from random.Random(seed) too, and skips its own spy draw when it is
    # handed a set, so both would come off the same state: the leader would be a
    # deterministic function of the same stream that picked the spies, and the
    # two would not be independent. Measured that way, the worst cell of the
    # (lowest spy seat x leader) table ran 153% off what independence predicts.
    # spawn takes an integer key, and the agents use the player ids, so this one
    # sits past them where it cannot collide.
    dealer = SeedSource(seed).spawn(num_players).to_random()
    drawn = dealer.sample(free, num_spies - len(forced))
    return frozenset(forced + drawn)


def turned_seats(config: dict, seed: int) -> dict:
    """the config with its seat line-up turned by one seat per TRIAL

    Returns the config unchanged unless agents.rotate_seats carries the run's
    trial seeds, in order. This trial's offset is where its own seed sits in
    that list, so one sweep axis -- game.seed -- moves the trial and the seats
    together. A seed the list does not hold is an error rather than an offset
    of zero.

    PER TRIAL AND NOT PER GAME. A cross-game memory is keyed by player id and
    learns across the games INSIDE a trial, so the seat has to hold still for
    all of them. Turning it per game would leave that memory with one game per
    key, which is no memory at all.

    Turned at all because an agent's play can depend on the seat number it
    holds, and GRAIL's does: its factor graph is seat indexed all the way down,
    and rotating one record through every seat and rotating the answer back
    moves its belief by up to 0.91, where a seat symmetric model would return
    the same six numbers. One fixed seat measures that agent AT that seat.

    The whole list turns together, so the arrangement around the table is
    identical in every trial and only the seat numbers move.

    What this does NOT fix is the deal: inside any one game the pinned seat
    still never holds a spy card, so every belief still carries a prior with
    worlds the experiment has ruled out. Turning changes which seat that is.
    """
    seats = config["agents"].get("seats")
    order = config["agents"].get("rotate_seats")
    if not seats or not order:
        return config
    if seed not in order:
        raise ValueError(f"agents.rotate_seats lists the trial seeds and {seed} is not one "
            f"of them: {order}. The offset is a trial's place in that list")
    by = order.index(seed) % len(seats)
    turned = []
    for index in range(len(seats)):
        turned.append(seats[(index - by) % len(seats)])
    agents = dict(config["agents"])
    agents["seats"] = turned
    return dict(config, agents=agents)


def bayes_player(config: dict) -> int:
    """the player id of the one player reading belief_model: bayes, or None

    Its memory is keyed by player id, so the line-up has to be listed by player
    id in agents.seats. Listed by faction, a player id holds a different agent
    every game.
    """
    seats: list[dict] = config["agents"].get("seats")
    if not seats:
        for faction in Team:
            for entry in config["agents"].get(faction.value) or []:
                if entry.get("belief_model") == "bayes":
                    raise ValueError("belief_model: bayes learns the type of each player id across games, "
                        "so the line-up has to be listed by player id in agents.seats")
        return None
    players = []
    for player_id, entry in enumerate(seats):
        if entry.get("belief_model") == "bayes":
            players.append(player_id)
    if len(players) > 1:
        raise ValueError(f"players {players} all read belief_model: bayes, and a line-up takes one")
    if not players:
        return None
    check_strategic3(seats, players[0])
    return players[0]


def check_strategic3(seats: list[dict], player_id: int) -> None:
    """raise unless the strategic3 model is built like every order 2 eu player in the line-up

    strategic3 is an order 3 StrategicKripkeState, which models a player holding
    an order 2 one. So it is compared as that player: its settings with order 2.
    """
    settings = seats[player_id].get("strategic3", {})
    accepted = strategic_knobs({}) # every setting StrategicKripkeState takes
    unknown = []
    for key in settings:
        if key == "order" or key not in accepted:
            unknown.append(key)
    if unknown:
        raise ValueError(f"strategic3 does not take {sorted(unknown)}; it is an order 3 "
            "StrategicKripkeState and takes that class's settings except order")
    modelled = strategic_knobs(dict(settings, order=2))
    for other, entry in enumerate(seats):
        if entry.get("type") != "ablation" or entry.get("belief_model") != "strategic":
            continue
        strategies = []
        for key in STRATEGY_KEYS:
            strategies.append(entry.get(key, DEFAULT_STRATEGY))
        if "eu" not in strategies:
            continue # its belief is never read
        actual = strategic_knobs(entry)
        different = []
        for key in modelled:
            if modelled[key] != actual[key]:
                different.append(f"{key}: {actual[key]!r} against {modelled[key]!r}")
        if different:
            raise ValueError(f"Player {player_id}'s strategic3 does not model Player {other} "
                f"({', '.join(different)}). Write that player's settings under strategic3")


def agent_label(entry: dict) -> str:
    """a short name for one line-up entry, such as eu/eu/eu bayes"""
    if entry["type"] != "ablation":
        return entry["type"]
    strategies = []
    for key in STRATEGY_KEYS:
        strategies.append(entry.get(key, DEFAULT_STRATEGY))
    label = "/".join(strategies)
    if "eu" in strategies:
        label += f" {entry.get('belief_model', DEFAULT_BELIEF_MODEL)}"
    return label


def modelled_type(entry: dict) -> str:
    """the TYPE_MODELS entry built for this line-up entry's agent, or None

        random RandomResistanceAgent
        strategic3  an eu player reading an order 2 StrategicKripkeState
        llm a player that decides everything with the LLM
    """
    if entry["type"] == "random":
        return "random"
    if entry["type"] != "ablation":
        return None
    strategies = set()
    for key in STRATEGY_KEYS:
        strategies.add(entry.get(key, DEFAULT_STRATEGY))
    belief_model = entry.get("belief_model", DEFAULT_BELIEF_MODEL)
    if strategies == {"eu"} and belief_model == "strategic" and strategic_knobs(entry)["order"] == 2:
        return "strategic3"
    if strategies == {"llm"}:
        return "llm"
    return None


def mission_cards(action_log: list[dict], missions: list[MissionRecord]) -> list[dict[int, bool]]:
    """the card every player on each mission played, True for Success, from the action log

    Private while a game runs. Every player on a team plays one card before the
    mission resolves, so the cards come in mission order, one team at a time.
    """
    played = []
    for entry in action_log:
        if entry["phase"] == Phase.MISSION_VOTE.value:
            played.append(entry)
    cards = []
    start = 0
    for mission in missions:
        cards.append({})
        for entry in played[start:start + len(mission.team)]:
            cards[-1][entry["player_id"]] = entry["action"]["vote"]
        start += len(mission.team)
    return cards


def create_agents(config: dict, seed: int, env: ResistanceEnv, \
    memory: TypeMemory = None) -> dict[int, ResistanceAgent]:
    """one agent per player_id, seated one of two ways

    BY FACTION, the default: the config lists agents under the two faction names,
    so an agent is pinned to one side of the table. Within a faction the
    agent_configs are used in player_id order; if there are fewer agent_configs
    than players on that side, the last entry repeats. The deal is still random,
    so a faction's agents hold different player ids from game to game.

    BY SEAT, when the config carries an agents.seats list: entry i is player i,
    whatever it was dealt. The deal stays random, so each seat is a spy a third
    of the time and nothing about the faction is learnable from the seat. This is
    what cross-game memory needs, because a memory is keyed by player id and a
    faction-seated agent holds a different one every game.

    Pinning the seat AND the faction together is what must not happen: then the
    deal is a constant and any belief reads it straight off the seat numbers.

    Seed streams are keyed by the player id either way.
    """
    num_players = config["game"]["num_players"]
    root = SeedSource(seed)

    seats: list[dict] = config["agents"].get("seats")
    if seats:
        if len(seats) != num_players:
            raise ValueError(f"agents.seats needs one entry per player, got {len(seats)} for {num_players}")
        agents = {}
        for player_id in range(num_players):
            # A seat that pins its faction is checked against that one; the rest
            # take whichever side the deal gave them, so they are checked against
            # both or a config that cannot play spy would run until the first
            # game that dealt it one.
            pinned = seats[player_id].get("faction")
            agents[player_id] = make_agent(seats[player_id], player_id, \
                root.spawn(player_id), env, config, both_factions=pinned is None, memory=memory)
        return agents

    lineups = {}
    for team in Team:
        agent_configs: list[dict] = config["agents"].get(team.value)
        if not agent_configs:
            raise ValueError(f"config needs a non-empty agents.{team.value} list!!!!!!!!")
        lineups[team] = agent_configs

    filled = {Team.RESISTANCE: 0, Team.SPIES: 0}
    agents = {}
    for player_id in range(num_players):
        team = env.team_of(player_id)
        agent_configs = lineups[team]
        agents[player_id] = make_agent(agent_configs[min(filled[team], len(agent_configs) - 1)], \
            player_id, root.spawn(player_id), env, config)
        filled[team] += 1
    return agents


def describe_lineup(env: ResistanceEnv, agents: dict[int, ResistanceAgent]) -> str:
    """which agent sits for which player, grouped by faction"""
    parts = []
    for team in Team:
        players = []
        for player_id in sorted(agents):
            if env.team_of(player_id) == team:
                players.append(f"P{player_id} {agents[player_id]}")
        parts.append(f"{team.value}: " + ", ".join(players))
    return "   ".join(parts)


def describe_action(player_id: int, action: dict) -> str:
    """one console line for an action"""
    action_type = action["type"]
    if action_type == "propose_team":
        return f"P{player_id} proposes team {sorted(action['team'])}"
    if action_type == "call_vote":
        return f"P{player_id} calls the vote"
    if action_type == "team_vote":
        return f"P{player_id} votes {'approve' if action['vote'] else 'reject'}"
    if action_type == "mission_vote":
        return f"P{player_id} plays {'Success' if action['vote'] else 'Fail'}"
    return f"P{player_id} {action}"


def winner_line(state: ResistanceState, crash_reason: str) -> str:
    """who won, for the line that closes a game in the console

    With print_out on, an LLM game puts thousands of lines between its header
    and this one, so the faction is named here in capitals rather than left for
    the reader to scroll back for. A game can finish and then raise, in which
    case both the winner and the crash are printed.
    """
    if state.winner is None:
        return f"NO WINNER: {crash_reason}"
    if crash_reason:
        return f"WINNER: {state.winner.value.upper()}, {state.win_reason}, then raised: {crash_reason}"
    return f"WINNER: {state.winner.value.upper()}, {state.win_reason}"


def running_tally(rows: list[dict]) -> str:
    """the score over the games that have finished so far

    Printed on every game's closing line, so following a run does not mean
    counting the winners by hand. Crashed games are left out, the way
    print_summary leaves them out at the end.
    """
    played = 0
    resistance_wins = 0
    for row in rows:
        if row["crashed"]:
            continue
        played += 1
        if row["winner"] == Team.RESISTANCE.value:
            resistance_wins += 1
    if not played:
        return "so far: no game finished"
    return f"so far: resistance {resistance_wins}/{played} ({resistance_wins / played:.0%})"


def print_resolution(env: ResistanceEnv, phase_before, missions_before: int, \
    rejections_before: int) -> None:
    """print what the environment decided, whenever a vote just closed"""
    state = env.state

    if phase_before == Phase.TEAM_VOTE and state.phase != Phase.TEAM_VOTE:
        # The counter, not the phase: the fifth rejection ends the game, so the
        # phase after it is DONE rather than PROPOSE.
        if state.rejections > rejections_before:
            teams = "team" if state.rejections == 1 else "teams"
            print(f"    -> team rejected ({state.rejections} {teams} rejected this round)")
        else:
            print(f"    -> team approved, mission {state.mission_index + 1} begins")

    if len(state.mission_results) > missions_before:
        mission = state.missions[-1]
        outcome = "SUCCESS" if mission.success else "FAILED"
        fails = "" if mission.num_fails is None else f", {mission.num_fails} Fail card(s)"
        tally = "".join("S" if r else "F" for r in state.mission_results)
        print(f"    -> mission {mission.index + 1} {outcome}{fails}"
                f"  team={sorted(mission.team)}  results={tally}")


def add_snapshot(agent: ResistanceAgent, env: ResistanceEnv, belief_log: list[dict], \
    reported: dict[int, int], belief: dict[int, float], closing: bool = False) -> None:
    """keep one belief snapshot per agent per number of missions played

    Keeps the first belief an agent held at each mission count, and skips an
    agent that holds none.

    Args:
        reported: player_id to the mission count its belief snapshot was taken at
        closing: this is the look taken once the game is over. It always counts
            and replaces an earlier snapshot at the same mission count. A game
            the spies win on rejections ends without a mission being played, so
            the count has not moved since the last snapshot, and the closing
            answer would otherwise be asked for and then dropped.
    """
    if belief is None:
        return

    missions_seen = len(env.state.missions)
    if reported.get(agent.player_id) == missions_seen and not closing:
        return

    if closing:
        for index, record in enumerate(belief_log):
            if record["player_id"] == agent.player_id and record["missions_seen"] == missions_seen:
                del belief_log[index]
                break

    reported[agent.player_id] = missions_seen
    belief_log.append({"player_id": agent.player_id, "missions_seen": missions_seen, \
        "spy_probability": dict(belief)})


def record_belief(agent: ResistanceAgent, env: ResistanceEnv, belief_log: list[dict], \
    reported: dict[int, int]) -> None:
    """snapshot the belief this agent just acted on

    An agent only recomputes when it is asked to act, so this reads the belief
    it just used.

    Spies are skipped. A spy is shown its teammates at the deal, so it has
    nothing to infer, and the score it would get is not comparable with a
    resistance player's.
    """
    if env.team_of(agent.player_id) != Team.RESISTANCE:
        return
    add_snapshot(agent, env, belief_log, reported, agent.identity_belief(), closing=False)


def record_final_beliefs(env: ResistanceEnv, agents: dict[int, ResistanceAgent], \
    belief_log: list[dict], reported: dict[int, int]) -> None:
    """ask every resistance agent once more, now the last mission has resolved

    Nobody acts after the last mission resolves, so every snapshot above was
    taken before the result that ended the game was known. Without this the
    score would always be one mission short of the end.

    What it costs is the agent's own business: an agent that keeps a belief from
    every observation answers for free, and one that only forms a belief when it
    is asked pays for one more look. Both then see the same evidence, which is
    what makes the comparison between them fair.
    """
    for player_id in sorted(agents):
        if env.team_of(player_id) != Team.RESISTANCE:
            continue
        agent = agents[player_id]
        add_snapshot(agent, env, belief_log, reported, \
            agent.final_belief(env.final_observation_for(player_id)), closing=True)


def play_game(env: ResistanceEnv, agents: dict[int, ResistanceAgent], \
    action_log: list[dict], belief_log: list[dict] = None, print_out: bool = False) -> None:
    """drive one game to the end

    pending_actors carries the phase, so this loop never names one. In PROPOSE
    the leader is asked twice, once to assign the team and once to call the vote.

    Every action reaches action_log. The console folds a run of proposals into
    one line, because a leader may revise as often as it likes and only the
    last team is voted on.
    """
    proposals = 0 # consecutive proposals not yet printed
    belief_log = [] if belief_log is None else belief_log
    reported = {} # player_id to the mission count its last snapshot was taken at

    while not env.done:
        for player_id in sorted(env.pending_actors):
            phase_before = env.state.phase
            missions_before = len(env.state.mission_results)
            rejections_before = env.state.rejections

            action = agents[player_id].take_action(env.observation_for(player_id),env.get_valid_actions(player_id))
            action_log.append({"player_id": player_id, "phase": phase_before.value, \
                "action": serialize_action(action)})
            record_belief(agents[player_id], env, belief_log, reported)

            if print_out:
                if action["type"] == "propose_team":
                    proposals += 1
                else:
                    if proposals:
                        team = sorted(env.state.proposed_team)
                        revisions = "" if proposals == 1 else f" (after {proposals - 1} revisions)"
                        print(f"    P{env.state.leader} proposes team {team}{revisions}")
                        proposals = 0
                    print(" " + describe_action(player_id, action))

            env.step(player_id, action)
            if print_out:
                print_resolution(env, phase_before, missions_before, rejections_before)

    record_final_beliefs(env, agents, belief_log, reported)


def run_experiment(config: dict, return_stats: bool = False, belief_rows: list[dict] = None):
    """play num_games games and write them to a fresh results directory

    Args:
        belief_rows: filled with this run's belief rows, so a caller running
            several configs can merge them. The same rows go to beliefs.csv.
    """
    check_llm_config(config)
    game_config = config["game"]
    num_games = game_config["num_games"]
    seed = game_config["seed"]
    print_out = game_config.get("print_out", False)

    env = ResistanceEnv(
        num_players=game_config["num_players"],
        enforce_resistance_must_pass=game_config.get("enforce_resistance_must_pass", False),
        reveal_fail_count=game_config.get("reveal_fail_count", False),
    )
    output_dir = create_output_dir(config)

    rows = []
    belief_rows = [] if belief_rows is None else belief_rows # one row per belief snapshot
    # Every game builds fresh agents, so each one's client counts that game
    # alone and the run's usage is the sum over games.
    experiment_llm = {
        "total_calls": 0,
        "retry_count": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }
    playing = turned_seats(config, seed) # this trial's seats, held for all its games
    # One type memory for the whole run: the run is one learning series.
    owner = bayes_player(playing)
    memory = None
    if owner is not None:
        others = []
        for player_id in range(game_config["num_players"]):
            if player_id != owner:
                others.append(player_id)
        memory = TypeMemory(others)
        seats = playing["agents"]["seats"]
        for player_id in others:
            memory.agents[player_id] = {"agent": agent_label(seats[player_id]), \
                "built_for": modelled_type(seats[player_id])}
    for game_index in range(num_games):
        seed = seed + 1

        env.reset(seed=seed, spies_id=pinned_spies(playing, seed, \
            game_config["num_players"], env.preset.num_spies))
        # The judge's answers are shared by every player of one game and no further,
        # so a game does not depend on the games before it.
        NeuroSymbolicKripkeState.clear_cache()
        agents = create_agents(playing, seed, env, memory)
        action_log = []
        belief_log = []
        if print_out:
            print(f"\n=== Game {game_index + 1}/{num_games} (seed={seed}) ===")
            print(f"   first leader: P{env.state.leader}")
            print(f"   {describe_lineup(env, agents)}")
        crashed = False
        crash_reason = ""
        # The clock starts at the first action, the way main.py times a game.
        # Building an agent can load a model, which is not part of playing.
        started = time.time()
        try:
            play_game(env, agents, action_log, belief_log, print_out=print_out)
        except Exception as error:
            crashed = True
            crash_reason = f"{type(error).__name__}: {error}"
        duration = time.time() - started
        
        # Before the stats are read and the clients closed: scoring uses the Bayes
        # player's client, and this game's judge answers are still cached.
        if memory is not None and not crashed:
            # Scoring asks the judge too. A failure there marks the game crashed, as
            # a game the memory did not learn from, rather than ending the run.
            try:
                update_type_posteriors(memory, env.state, mission_cards(action_log, env.state.missions), \
                    agents[owner].belief.type_models(), game_index, seed)
            except Exception as error:
                crashed = True
                crash_reason = f"scoring the game: {type(error).__name__}: {error}"
            else:
                memory.save(os.path.join(output_dir, "memory.json"))

        game_llm = extract_llm_stats(agents)["game_total"]
        for key in experiment_llm:
            experiment_llm[key] += game_llm[key]
        save_game_record(output_dir, game_index, seed, env.state, agents, action_log, \
            belief_log, crashed=crashed, crash_reason=crash_reason, duration_seconds=duration)
        rows.append(build_summary_row(game_index, seed, env.state, agents, belief_log, \
            crashed=crashed, crash_reason=crash_reason, duration_seconds=duration))
        belief_rows.extend(build_belief_rows(game_index, seed, env.state, agents, belief_log, \
            crashed=crashed))
        # Inside the loop, and after the record is written. Writing it reads the
        # clients' stats, and the next game rebuilds the agents, so anything not
        # closed here is a client nothing can reach again.
        close_llm_clients(agents)

        if print_out:
            accuracy = rows[-1]["identity_accuracy"]
            reported = "" if accuracy is None else f", identity {accuracy:.0%}"
            print(f"    == Game {game_index + 1}/{num_games}  "\
                f"{winner_line(env.state, crash_reason)}"\
                f"   ({len(action_log)} actions{reported}, {duration:.1f}s)"\
                f"   {running_tally(rows)}")

    save_summary_csv(output_dir, rows)
    save_belief_csv(output_dir, belief_rows)
    print_summary(rows)
    print(f"\nresults written to {output_dir}")

    if return_stats:
        return rows, {"llm": experiment_llm}
    return rows


def close_llm_clients(agents: dict[int, ResistanceAgent]) -> None:
    """release the connections this game's agents hold

    Every game builds fresh agents and every LLM agent builds its own client, so
    a run that did not close them would leave one connection pool open per agent
    per game. An agent without a client is skipped.
    """
    for agent in agents.values():
        client = getattr(agent, "llm_client", None)
        if client is not None:
            client.close()


def print_summary(rows: list[dict]) -> None:
    """resistance win rate over the games that finished"""
    played = 0
    resistance_wins = 0
    crashed = 0
    accuracies = []
    missions_seen = 0 # missions the scored beliefs saw, over the games that reported one
    missions_played = 0 # missions those same games ran
    llm_calls = 0
    llm_tokens = 0
    judge_calls = 0
    for row in rows:
        # A crashed game still cost whatever LLM calls it made before it raised.
        llm_calls += row["llm_calls"]
        llm_tokens += row["llm_total_tokens"]
        judge_calls += row["lk_judge_calls"]
        if row["crashed"]:
            crashed += 1
            continue
        played += 1
        if row["winner"] == Team.RESISTANCE.value:
            resistance_wins += 1
        if row["identity_accuracy"] is not None:
            accuracies.append(row["identity_accuracy"])
            missions_seen += row["identity_missions_seen"]
            missions_played += row["missions_played"]

    print("\n" + "=" * 60)
    print(f"  games: {played} played, {crashed} crashed")
    if played:
        rate = resistance_wins / played
        print(f"  resistance wins: {resistance_wins}/{played} ({rate:.1%})")
    # The measure that does not depend on teammates, over the agents that hold a
    # belief. Empty when none of them do, the random baseline for one. The two
    # mission counts are printed with it: they are equal once every agent has
    # answered the closing question, so a gap says one of them did not.
    if accuracies:
        print(f"  identity accuracy: {sum(accuracies) / len(accuracies):.1%}"
                f"  ({len(accuracies)} games reported, beliefs saw "
                f"{missions_seen / len(accuracies):.1f} of "
                f"{missions_played / len(accuracies):.1f} missions)")
    if llm_calls:
        print(f"  llm: {llm_calls} calls, {llm_tokens} tokens"
                f"  ({llm_calls / len(rows):.1f} calls per game)")
    if judge_calls:
        # one LLM call a question, two with judge_cot; both counted in llm above
        print(f"  likelihood judge: {judge_calls} questions"
                f"  ({judge_calls / len(rows):.1f} per game)")
    print("=" * 60)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    run_experiment(load_config(path))


if __name__ == "__main__":
    main()
