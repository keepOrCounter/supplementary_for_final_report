"""Learn a data-driven answer-likelihood codebook from saved games -- condition (B),
the domain-adapted analogue of GRAIL's NN factor (GRAIL's training code is
Avalon-specific and not reusable).

Estimates a smoothed conditional frequency table P(answer | q_type, target_role,
referenced_role) from the true roles + observed answers in results/*.json. Unseen
contexts fall back to the uniform likelihood. The result is callable with the
likelihood_codebook signature TofuFactorGraph expects. Use --opponent-types to
learn only the opponent distribution you intend to test against.

Usage:
    python agents/grail/build_likelihood_codebook.py results/<run_dir> --out codebook.pkl \
        [--opponent-types SymbolicPlayerAgent LLMAgent] [--alpha 1.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # allow importing from the project root of this file (e.g. game.predefined_data_classes, agents.our_agents.Kripke)
# os.path.abspath(__file__) returns the full path of this file
# __file__ path of current file
# os.path.dirname() gets the directory of the given path(i.e. parent directory of the file/folder)
from game.predefined_data_classes import ROLES, ROLE_CONFIGS, ROLE_CONFIGS_NO_CENTER, QuestionType
from agents.our_agents.Kripke import uniform_answer_likelihood
from agents.grail.tofu_factor_graph import CENTER_ID, QA_referenced_player, answer_is_truthful


def _int_counter() -> dict:
    """Module-level factory so LearnedLikelihood stays picklable."""
    return defaultdict(int)


def role_from_log(player_id: int | None, roles: dict[str, str], center_card: str | None) -> str | None:
    """Return the role name of a player id (or center card) from a game log's roles dict.
        Returns:
            role name in str
    """
    if player_id is None:
        return None
    return center_card if player_id == CENTER_ID else roles.get(str(player_id))


class LearnedLikelihood:
    """Callable codebook: P(answer | q_type, target_role, referenced_role).

    Counts are keyed by question type so each type models the right outcome:
        - Q2: P(claimed_role | target_role)
        - Q3: P(claimed_role | target_role, X's_true_role)
        - Q1: P(pointed_player | target_role)
    Laplace-smoothed; unseen contexts fall back to the uniform likelihood.
    """

    def __init__(self, role_names: list[str], use_center_card: bool, num_players: int, alpha: float = 1.0):
        self.role_names = role_names
        self.num_roles = len(role_names)
        self.use_center_card = use_center_card
        self.num_players = num_players
        self.alpha = alpha
        # context -> {outcome: count}
        self.counts: dict[tuple, dict[str, int]] = defaultdict(_int_counter)

    def add_observation(self, q_type: QuestionType, answer: str, target_role: str, ref_role: str):
        """count one observed answer"""
        if q_type == QuestionType.Q1:
            ctx, outcome = (q_type.value, target_role), ref_role
        elif q_type == QuestionType.Q3:
            ctx, outcome = (q_type.value, target_role, ref_role), answer
        else:  # Q2
            ctx, outcome = (q_type.value, target_role), answer
        if outcome is not None:
            self.counts[ctx][outcome] += 1

    def _num_answers(self, q_type: QuestionType) -> int:
        # Q1 points at a location: every non-prince player, plus the center card if any
        if q_type == QuestionType.Q1:
            n = self.num_players - 1
            return n + 1 if self.use_center_card else n
        return self.num_roles  # Q2/Q3 answer is a role name

    def __call__(self, q_type: QuestionType, answer: str, target_role: str, ref_role: str) -> float:
        """Return the smoothed likelihood P(answer | q_type, target_role, referenced_role). 
        Unseen contexts fall back to the uniform likelihood.
        
        Essentially a smoothed conditional frequency table (count frequency of choosen answer)
        """
        if q_type == QuestionType.Q1:
            ctx, outcome = (q_type.value, target_role), ref_role
            space = self.num_roles # outcome = role of pointed player
        elif q_type == QuestionType.Q3:
            ctx, outcome = (q_type.value, target_role, ref_role), answer
            space = self.num_roles # outcome = claimed role of target player
        else: # Q2
            ctx, outcome = (q_type.value, target_role), answer
            space = self.num_roles

        table = self.counts.get(ctx)
        if not table:
            # unseen context -> fall back to the uniform likelihood
            truthful = answer_is_truthful(q_type, answer, target_role, ref_role)
            return uniform_answer_likelihood(truthful, ROLES[target_role].constraint, self._num_answers(q_type))
        total = sum(table.values())
        return (table.get(outcome, 0) + self.alpha) / (total + self.alpha * space)



def build_codebook(result_dirs: list[str], opponent_types=None, status="completed", num_players=None, \
    use_center_card=True, alpha=1.0) -> LearnedLikelihood:
    """learn answer frequency table from saved games
    Args:
        opponent_types: learn from what type of opponents (non-prince) agents. If None, learn from all games.
        use_center_card: only learn from games whose center-card setting matches this flag. Center and
            no-center games use DIFFERENT role configs (and therefore different answer spaces), so mixing
            them would corrupt the conditional frequency table -- they must be learned separately.
    """
    files = []
    # print(result_dirs)
    for result_path in result_dirs: # seek all game logs under given directories
        files += glob.glob(os.path.join(result_path, "**", "game_*.json"), recursive=True)
    if not files:
        raise SystemExit(f"No game_*.json found under {result_dirs}")
    # print(files)
    codebook = None
    num_games = num_obs = 0
    for file_path in files:
        with open(file_path, encoding="utf-8") as f: # open each game log
            game_log: dict = json.load(f)
            
        if status and game_log.get("status") != status: # filter by game status (default only use result when game completed)
            continue
        if num_players and game_log.get("num_players") != num_players: # only use games with the specified number of players
            continue
        
        roles: dict = game_log["roles"]
        center: str | None = game_log.get("center_card")
        # Center / no-center games have different role configs -> different answer
        # spaces; only keep games matching the requested center-card setting.
        if (center is not None) != use_center_card:
            continue
        agent_types: dict = game_log.get("agent_types", {})
        # learn from games whose NON-PRINCE answerers match the requested types
        if opponent_types: # check non-prince agent types
            non_prince = [t for pid, t in agent_types.items() if pid != "0"]
            if not non_prince or any(t not in opponent_types for t in non_prince):
                continue
        if codebook is None:
            role_configs = ROLE_CONFIGS if use_center_card else ROLE_CONFIGS_NO_CENTER
            num_players_log = game_log["num_players"]
            codebook = LearnedLikelihood(list(role_configs[num_players_log]), use_center_card, num_players_log, alpha=alpha)
        num_games += 1
        for qa in game_log["qa_history"]:
            if qa.get("answer") is None:
                continue
            q_type = QuestionType(qa["question_type"])
            target_role = roles.get(str(qa["target_id"]))
            if target_role is None:
                continue
            ref = QA_referenced_player(q_type, qa["answer"], qa.get("question_param"))
            ref_role = role_from_log(ref, roles, center)
            codebook.add_observation(q_type, qa["answer"], target_role, ref_role)
            num_obs += 1
    if codebook is None:
        raise SystemExit("No games matched the filters.")
    print(f"Learned from {num_games} games / {num_obs} answers; {len(codebook.counts)} contexts.")
    return codebook


def load_codebook(path) -> LearnedLikelihood:
    import agents.grail.build_likelihood_codebook as _this_module

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # codebooks pickled when the script ran as __main__ store LearnedLikelihood
            # AND its defaultdict factory (_int_counter) under '__main__'; codebooks built
            # before the agents/ re-org store them under 'grail.build_likelihood_codebook'.
            # Remap any such name to this module so unpickling works from main.py or anywhere else
            if module in ("__main__", "grail.build_likelihood_codebook") and hasattr(_this_module, name):
                return getattr(_this_module, name)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return _Unpickler(f).load()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dirs", nargs="+", help="results/<run_dir> directories")
    ap.add_argument("--out", required=True, help="output .pkl path for the codebook")
    ap.add_argument("--opponent-types", nargs="*", default=None, \
        help="only learn from games whose non-prince agents are all of these classes (e.g. SymbolicPlayerAgent LLMAgent)")
    ap.add_argument("--num-players", type=int, default=None, help="filter to this player count")
    ap.add_argument("--use-center", action="store_true", \
        help="true: learn only from games WITH a center card (default); false: only WITHOUT")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing (default 1.0)")
    args = ap.parse_args()

    codebook = build_codebook(args.result_dirs, opponent_types=args.opponent_types, num_players=args.num_players, \
        use_center_card=args.use_center, alpha=args.alpha)
    with open(args.out, "wb") as f:
        pickle.dump(codebook, f)
    print(f"Saved codebook -> {args.out}")
    print("Plug in via: TofuFactorGraph(num_players, use_center_card, likelihood_codebook=load_codebook(path))")
