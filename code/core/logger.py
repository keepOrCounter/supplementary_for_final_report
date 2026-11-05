from __future__ import annotations
import os
import json

from game.predefined_data_classes import GameResult, GameState, QAPair, Role, Faction, QuestionType
from agents import BaseAgent, LLMAgent, SymbolicPrinceAgent, SymbolicPlayerAgent, \
    SoftSymbolicPrinceAgent, AblationPrinceAgent, GrailPrinceAgent, EMOPrinceAgent
from agents.our_agents.Kripke import NeuroSymbolicKripkeState
# Re-exported: both games write the same shape of results directory, so those
# two live in core/output.py and callers of this module still find them here.
from core.output import create_output_dir, save_summary_csv


def save_game_record(output_dir: str, game_index: int, seed: int, state: GameState, result: GameResult, \
    agents: dict[int, BaseAgent], action_log: list[dict], failed: bool = False, \
        fail_reason: str = "", duration_seconds: float = None) -> None:
    """Save a single game's record as JSON.

    Args:
        output_dir: path to save
        seed: Random seed used for this game
        state: Final game state (may be incomplete if failed)
        result: GameResult (None if game failed)
        agents: Dict of player_id to agent (for extracting reasoning)
        action_log: List of action dicts recorded during the game
        failed: Whether this game was aborted due to LLM failure
        fail_reason: Error message if failed
    """
    record = {
        "game_index": game_index,
        "seed": seed,
        "num_players": state.num_players,
        "status": "failed" if failed else "completed",
        "duration_seconds": round(duration_seconds, 2) if duration_seconds is not None else None,

        # Role assignments
        "roles": {str(pid): role.name for pid, role in state.roles.items()},
        "center_card": state.center_card.name if state.center_card else None,

        "agent_types": {str(pid): agent.__class__.__name__ for pid, agent in agents.items()},

        "qa_history": [serialize_qa(qa) for qa in state.qa_history],

        # Action every step taken during the game
        "action_log": action_log,

        # LLM reasoning traces (if available)
        "reasoning_traces": extract_reasoning(agents),

        # LLM usage stats (tokens, calls, failures)
        "llm_stats": extract_llm_stats(agents),
    }

    if failed:
        record["fail_reason"] = fail_reason
        record["result"] = None
    else:
        record["result"] = {
            "winner_faction": result.winner_faction.value,
            "guessed_target": result.guessed_target,
            "actual_princess": result.actual_princess,
            "actual_queen": result.actual_queen,
            "correct": result.winner_faction == Faction.PRINCESS,
            "scores": {str(k): v for k, v in result.scores.items()},
        }

    filename = f"game_{game_index}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4, ensure_ascii=False)


def build_summary_row(game_index: int, seed: int, state: GameState, result: GameResult, \
    failed: bool = False, fail_reason: str = "", duration_seconds: float = None, \
        llm_stats: dict = None, likelihood_stats: dict = None) -> dict:
    
    """Build one row for the summary CSV."""
    row = {
        "game": game_index,
        "seed": seed,
        "num_players": state.num_players,
        "status": "failed" if failed else "completed",
        "duration_seconds": round(duration_seconds, 2) if duration_seconds is not None else None,
        "center_card": state.center_card.name if state.center_card else "none",
        "roles": ",".join(f"{pid}:{r.name}" for pid, r in sorted(state.roles.items())),
    }

    if failed:
        row.update({
            "winner_faction": None,
            "correct": None,
            "guessed_target": None,
            "actual_princess": None,
            "actual_queen": None,
            "fail_reason": fail_reason,
        })
    else:
        row.update({
            "winner_faction": result.winner_faction.value,
            "correct": result.winner_faction == Faction.PRINCESS,
            "guessed_target": result.guessed_target,
            "actual_princess": result.actual_princess,
            "actual_queen": result.actual_queen,
            "fail_reason": None,
        })

    # LLM usage stats
    if llm_stats and "game_total" in llm_stats:
        gt = llm_stats["game_total"]
        row["llm_calls"] = gt.get("total_calls", 0)
        row["llm_retries"] = gt.get("retry_count", 0)
        row["llm_total_tokens"] = gt.get("total_tokens", 0)
        row["llm_prompt_tokens"] = gt.get("total_prompt_tokens", 0)
        row["llm_completion_tokens"] = gt.get("total_completion_tokens", 0)
        row["llm_token_tracking_complete"] = gt.get("token_tracking_complete", True)
        # if False: token above are under-counted (some API responses lacked usage)
    else:
        row["llm_calls"] = 0
        row["llm_retries"] = 0
        row["llm_total_tokens"] = 0
        row["llm_prompt_tokens"] = 0
        row["llm_completion_tokens"] = 0
        row["llm_token_tracking_complete"] = False

    # Soft-likelihood (judge) stats — non-zero only for a soft_symbolic Prince.
    # Always written (with defaults) so every CSV row has the same columns.
    lk = likelihood_stats or {}
    row["lk_queries"] = lk.get("queries", 0)
    row["lk_judge_calls"] = lk.get("judge_calls", 0)
    row["lk_cache_hit_rate"] = lk.get("cache_hit_rate", 0.0)
    row["lk_defaulted_slots"] = lk.get("defaulted_slots", 0)

    return row



def serialize_qa(qa: QAPair) -> dict:
    """Convert QAPair to JSON-friendly dict."""
    return {
        "round": qa.round,
        "target_id": qa.target_id,
        "question_type": qa.question_type.value,
        "question_param": qa.question_param,
        "answer": qa.answer,
    }


def extract_reasoning(agents: dict[int, BaseAgent]) -> dict[str, dict]:
    """Extract reasoning history and summaries from LLM agents.
    
    Returns:
        Dict of player_id -> {
            "full_reasoning": [...],       # complete CoT from each turn
            "summary_iterations": [...],   # how the summary evolved each turn
            "final_summary": "..."         # last iterative summary
        }
    """
    reasoning = {}
    for pid, agent in agents.items():
        if isinstance(agent, LLMAgent):
            data = {}
            if agent.reasoning_history:
                data["full_reasoning"] = agent.reasoning_history
            if agent.summary_history:
                data["summary_iterations"] = agent.summary_history
            if agent.current_summary:
                data["final_summary"] = agent.current_summary
            if data:
                reasoning[str(pid)] = data
        elif isinstance(agent, (SymbolicPrinceAgent, SymbolicPlayerAgent)):
            # NB: SoftSymbolicPrinceAgent is a SymbolicPrinceAgent subclass.
            data = {}
            if agent.decision_log:
                data["symbolic_decisions"] = agent.decision_log
            judge = _judge_state(agent)
            if judge is not None:
                s = judge.stats
                data["likelihood_stats"] = {
                    "queries": s.queries, "judge_calls": s.judge_calls,
                    "cache_hits": s.cache_hits, "cache_hit_rate": round(s.hit_rate(), 4),
                    "defaulted_slots": s.defaulted_slots,
                }
                if judge.judge_log:
                    data["judge_log"] = judge.judge_log
            if data:
                reasoning[str(pid)] = data
        elif isinstance(agent, GrailPrinceAgent):
            # GRAIL Prince: persist the per-step belief trajectory (princess_dist +
            # full role beliefs) so its inference can be analysed like the symbolic one.
            if agent.decision_log:
                reasoning[str(pid)] = {"grail_decisions": agent.decision_log}
        elif isinstance(agent, AblationPrinceAgent):
            data = {}
            if agent.decision_log:
                data["symbolic_decisions"] = agent.decision_log
            # AblationPrinceAgent also has LLM reasoning history when strategy=llm
            if agent.reasoning_history:
                data["full_reasoning"] = agent.reasoning_history
            if agent.summary_history:
                data["summary_iterations"] = agent.summary_history
            if agent.current_summary:
                data["final_summary"] = agent.current_summary
            judge = _judge_state(agent)
            if judge is not None and judge.judge_log:
                data["judge_log"] = judge.judge_log
            if data:
                reasoning[str(pid)] = data
        elif isinstance(agent, EMOPrinceAgent):
            # EMO Prince: keep the CoT reasoning traces, the per-opponent prediction
            # trajectory (how each atomic model's role guess evolved), AND the raw
            # reasoning of every EMO sub-call (atomic models + validator).
            data = {}
            if agent.decision_log:
                data["emo_decisions"] = agent.decision_log
            opponent_models = {str(pid_j): m.prediction_trace for pid_j, m in agent._models.items() if m.prediction_trace}
            if opponent_models:
                data["opponent_models"] = opponent_models
            model_reasoning = {str(pid_j): m.reasoning_log for pid_j, m in agent._models.items() if m.reasoning_log}
            if model_reasoning:
                data["opponent_model_reasoning"] = model_reasoning
            if agent.validator.reasoning_log:
                data["validator_reasoning"] = agent.validator.reasoning_log
            if agent.reasoning_history:
                data["full_reasoning"] = agent.reasoning_history
            if agent.summary_history:
                data["summary_iterations"] = agent.summary_history
            if agent.current_summary:
                data["final_summary"] = agent.current_summary
            if data:
                reasoning[str(pid)] = data
    return reasoning


def extract_llm_stats(agents: dict[int, BaseAgent]) -> dict:
    """Extract LLM usage statistics from all agents that use an LLM client.

    Returns:
        Dict with per-agent stats and game-wide totals:
        {
            "per_agent": { "0": {...}, "1": {...} },
            "game_total": { "total_calls": N, "total_tokens": N, ... }
        }
    """
    per_agent = {}
    game_totals = {
        "total_calls": 0,
        "retry_count": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }
    token_tracking_complete = True # AND across all agents: False if any had missing usage

    for pid, agent in agents.items():
        client = None
        if isinstance(agent, LLMAgent) and agent.llm_client is not None:
            client = agent.llm_client
        elif isinstance(agent, AblationPrinceAgent) and agent.llm_client is not None:
            client = agent.llm_client
        elif isinstance(agent, SoftSymbolicPrinceAgent) and agent.llm_client is not None:
            client = agent.llm_client  # the soft-likelihood judge client (per game)
        elif isinstance(agent, GrailPrinceAgent) and agent.llm_client is not None:
            client = agent.llm_client  # the GRAIL-style 'vibes' prior client (per game)
        elif isinstance(agent, EMOPrinceAgent) and agent.llm_client is not None:
            client = agent.llm_client  # shared by every EMO model / validator / reasoning call

        if client is not None:
            stats = client.stats.to_dict()
            per_agent[str(pid)] = stats
            for key in game_totals:
                game_totals[key] += stats.get(key, 0)
            if not stats.get("token_tracking_complete", True):
                token_tracking_complete = False

    game_totals["token_tracking_complete"] = token_tracking_complete

    return {
        "per_agent": per_agent,
        "game_total": game_totals,
    }


def _judge_state(agent: BaseAgent):
    """The NeuroSymbolicKripkeState judge behind an agent, or None.

    It is either the agent's own belief state or, for a bayes Prince, the LLM
    judge it keeps in llm_state for undecided opponents.
    """
    kripke = getattr(agent, "kripke", None)
    if isinstance(kripke, NeuroSymbolicKripkeState):
        return kripke
    return getattr(kripke, "llm_state", None) if kripke is not None else None


def extract_likelihood_stats(agents: dict[int, BaseAgent]) -> dict:
    """Per-game soft-likelihood (judge) stats from a soft_symbolic Prince, if present.

    Returns {} when no NeuroSymbolicKripkeState belief state is in play (so callers
    can pass it through harmlessly for non-soft experiments).
    """
    for agent in agents.values():
        kripke = _judge_state(agent)
        if isinstance(kripke, NeuroSymbolicKripkeState):
            s = kripke.stats
            return {
                "queries": s.queries,
                "judge_calls": s.judge_calls,
                "cache_hits": s.cache_hits,
                "cache_hit_rate": round(s.hit_rate(), 4),
                "defaulted_slots": s.defaulted_slots,
            }
    return {}