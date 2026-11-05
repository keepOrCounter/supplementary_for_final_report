"""Learned 'posterior factor' -- condition (A), the faithful GRAIL analogue.

GRAIL's NN factor is a discriminative per-player posterior p(role | state). This
is the direct transliteration: a shared net mapping the QA history to a per-entity
role distribution p(role_e | history), used as a learned unary factor on each role
variable (TofuFactorGraph posterior mode).

The QA-history -> fixed feature vector encoding (encode_entity) is the one design
freedom; we use an ego-centric one that keeps the full per-entity evidence (see
encode_entity). Trained on true roles in results/*.json with an automatic train/val
split, and runs on the GPU when one is available.

Usage:
    python agents/grail/posterior_factor.py results/<run_dir> --out posterior.pt \
        --num-players 7 [--epochs 50] [--opponent-types SymbolicPlayerAgent] \
        [--val-fraction 0.2] [--device cuda] [--use-center]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from game.predefined_data_classes import ROLE_CONFIGS, ROLE_CONFIGS_NO_CENTER, QuestionType
from agents.grail.tofu_factor_graph import CENTER_ID, PRINCESS_ROLE, valid_player_ids
from core.helper_functions import player_id_from_answer


def _normalize_qa(qa) -> dict:
    """Unify QAPair objects and JSON dicts into a plain dict."""
    if isinstance(qa, dict):
        q_type = qa["question_type"]
        q_type = q_type if isinstance(q_type, QuestionType) else QuestionType(q_type)
        return {"target_id": qa["target_id"], "question_type": q_type,
                "question_param": qa.get("question_param"), "answer": qa.get("answer")}
        
    return {"target_id": qa.target_id, "question_type": qa.question_type,
            "question_param": qa.question_param, "answer": qa.answer}


def _ego_other_order(entity_id: int, player_ids: list[int]) -> list[int]:
    """The other entities, in an ego-centric, player-rotation-invariant order.
    
    remove the query player from the feature vector, then rotate the others so the ones 
    questioned right after it come first.
    
    The center card -- which never answers a question -- is always the final slot.
    
    Players are rotated so the ones questioned right after ``entity_id`` come first.
    Absolute player numbers carry no role information (roles are dealt at random), so
    feeding them directly would let the net overfit to positions; rotating into the
    ego frame is GRAIL's trick (egocentric_neuralnet.permutate_vector_for_player_order)
    that makes the shared net player-permutation-equivariant. 
    """
    players = [p for p in player_ids if p != CENTER_ID]
    if entity_id in players:
        i = players.index(entity_id)
        others = players[i + 1:] + players[:i]
    else:  # the center card has no position in the player cycle
        others = list(players)
    if CENTER_ID in player_ids and entity_id != CENTER_ID:
        others.append(CENTER_ID)
    return others


def encode_entity(qa_history: list, entity_id: int, player_ids: list[int], role_names: list[str], \
    role_index: dict[str, int]) -> np.ndarray:
    """Feature vector for ``entity_id``, built only from the PUBLIC QA history.

    This is the Prince's posterior factor, so ``entity_id`` is the hidden role variable
    being predicted -- NOT a viewpoint player. The encoding reads only what the Prince can
    observe (the questions and answers); it never uses any hidden role, so it matches the
    Prince's information state. Each entity is encoded independently and fed to the same
    shared net, so the features are written in an "ego" frame centred on ``entity_id``.

    Player numbers carry no role information (roles are dealt at random), so the other
    entities are placed in a cyclically-rotated order (see _ego_other_order). This is
    GRAIL's positional-bias fix: it removes the entity's OWN absolute index. Only the
    relative ordering among the others remains -- exactly as in the paper's ego-centric net.

    Own testimony (the entity is the answerer):
        own_q2          one-hot[num_roles] of the role it claimed for itself (Q2); all-zero
                        if it never answered Q2 (so 'not asked' needs no extra slot)
        own_q1_point    one-hot over the OTHER entities (players AND the center card) of who
                        it pointed at (Q1); all-zero only if it did not answer Q1
        own_q3          [num_others, num_roles] roles it claimed for others (Q3 it answered),
                        indexed by the ego position of the entity it spoke about
        flags           [answered_Q1, answered_Q2, was_questioned]

    Others' testimony (the entity is referenced), kept per answerer (not summed) so who made
    each claim is preserved:
        incoming        [num_others, num_roles] per other answerer, the role they claimed this
                        entity is: a Q3 'who is it?' sets the claimed role; a Q1 'where is the
                        Princess?' pointing here sets the Princess slot (same claim, same slot).
    """
    num_roles = len(role_names)
    others = _ego_other_order(entity_id, player_ids)
    ego_pos = {e: k for k, e in enumerate(others)}  # ego position of each other entity
    num_others = len(others)

    own_q2 = np.zeros(num_roles)
    own_q1_point = np.zeros(num_others)
    own_q3 = np.zeros((num_others, num_roles))
    incoming = np.zeros((num_others, num_roles))
    answered_q1 = answered_q2 = was_questioned = 0.0

    for qa in qa_history:
        qa = _normalize_qa(qa)
        answer = qa["answer"]
        if answer is None:
            continue
        target, q_type, param = qa["target_id"], qa["question_type"], qa["question_param"]

        if target == entity_id:  # the entity is the answerer -> its own testimony
            was_questioned = 1.0
            if q_type == QuestionType.Q2:
                answered_q2 = 1.0
                if answer in role_index:
                    own_q2[role_index[answer]] = 1.0
            elif q_type == QuestionType.Q1:
                answered_q1 = 1.0
                pointed = player_id_from_answer(answer)
                if pointed in ego_pos:  # who it pointed at -- a specific player OR the center card
                    own_q1_point[ego_pos[pointed]] = 1.0
            elif q_type == QuestionType.Q3 and param in ego_pos and answer in role_index:
                own_q3[ego_pos[param], role_index[answer]] = 1.0
        else:  # someone else is answering -> keep only what it says ABOUT this entity
            other_is_answerer = target in ego_pos
            answered_role_exists = answer in role_index
            answer_about_this_entity = param == entity_id
            
            if q_type == QuestionType.Q3 and answer_about_this_entity and answered_role_exists and other_is_answerer:
                incoming[ego_pos[target], role_index[answer]] = 1.0
            elif q_type == QuestionType.Q1 and player_id_from_answer(answer) == entity_id and other_is_answerer:
                # pointing here as the Princess IS a 'this entity is Princess' claim -> Princess slot
                incoming[ego_pos[target], role_index[PRINCESS_ROLE]] = 1.0

    return np.concatenate([
        own_q2, own_q1_point, own_q3.ravel(),
        [answered_q1, answered_q2, was_questioned],
        incoming.ravel(),
    ]).astype(np.float32)


class PosteriorFactorModel:
    """Shared MLP: encoded QA history -> per-entity role distribution.

    Trained per (num_players, use_center_card) config (like GRAIL trains per
    6-player setting). predict(qa_history) -> {player_id: {role: prob}}.
    """

    def __init__(self, num_players: int, use_center_card: bool, hidden_dim: int = 32, device=None):
        self.num_players = num_players
        self.use_center_card = use_center_card and num_players != 8
        role_config = ROLE_CONFIGS if use_center_card else ROLE_CONFIGS_NO_CENTER
        self.role_names = list(role_config[num_players])
        self.role_index = {role: i for i, role in enumerate(self.role_names)}
        self.player_ids = valid_player_ids(num_players, use_center_card)
        num_roles = len(self.role_names)
        # Derive the feature dimension straight from the encoder (on an empty history)
        # so it can never drift out of sync with encode_entity's layout.
        self.feature_dim = len(encode_entity([], self.player_ids[0], self.player_ids, \
            self.role_names, self.role_index))
        # Use the GPU when one is available (override with device=...). The model is
        # tiny, but training scans every saved game, so GPU still helps large corpora.
        self.device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim), nn.ReLU(), \
            nn.Linear(hidden_dim, num_roles)).to(self.device)

    def _encode(self, qa_history) -> np.ndarray:
        return np.stack([encode_entity(qa_history, entity_id, self.player_ids, self.role_names, \
            self.role_index) for entity_id in self.player_ids])

    def predict(self, qa_history) -> dict[int, dict[str, float]]:
        """
        Returns:
            dict: {player_id: {role_name: belief}}
        """
        self.net.eval()
        # ALL entities are encoded as one batch, so the net output (and probs) is 2-D:
        # [num_entities, num_roles] -- row i is entity i's role distribution.
        features = torch.from_numpy(self._encode(qa_history)).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.net(features), dim=1).cpu().numpy()

        result = {}
        for i, entity_id in enumerate(self.player_ids):
            result[entity_id] = {role: float(probs[i][j]) for j, role in enumerate(self.role_names)}
        return result

    def fit(self, features: np.ndarray, labels: np.ndarray, groups: np.ndarray = None, epochs: int = 50, \
        lr: float = 1e-2, batch_size: int = 256, val_fraction: float = 0.2, seed: int = 0):
        """Train the shared net, holding out a validation split to monitor fit.

        The validation split is GAME-LEVEL when ``groups`` (a game id per sample) is
        given: whole games are held out, so no game has samples in both splits. This
        matters because an entity's features depend on the entire game history -- a
        sample-level split would leak, since sibling entities from the same game share
        that history. ``val_fraction`` and the fixed ``seed`` make the split reproducible.
        Per-epoch train/val loss and val accuracy are printed and the parameters with the
        best val loss are restored at the end, so the saved model is the best-generalising
        one rather than the last (possibly overfit) epoch.
        """
        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)

        # Slice the split indices on CPU, then move the subsets to the (possibly GPU) device.
        train_idx, val_idx = self._split_train_val(len(features), groups, val_fraction, seed)
        n_val = len(val_idx)
        x_train, y_train = features[train_idx].to(self.device), labels[train_idx].to(self.device)
        x_val, y_val = features[val_idx].to(self.device), labels[val_idx].to(self.device)
        split = "game-level" if groups is not None else "sample-level"
        print(f"Training on {len(x_train)} samples, validating on {n_val} ({split} split, device={self.device}).")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val = float("inf")
        best_state = None

        for epoch in range(epochs):
            self.net.train()
            order = torch.randperm(len(x_train), device=self.device)
            total_loss = 0.0
            for start in range(0, len(x_train), batch_size):
                batch_idx = order[start:start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.net(x_train[batch_idx]), y_train[batch_idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)
            train_loss = total_loss / len(x_train)

            val_loss, val_acc = self._evaluate(x_val, y_val, loss_fn)
            if val_loss < best_val:  # keep the best-generalising weights
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

            if (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"  epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}"
                if n_val:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
                print(msg)

        if best_state is not None:
            self.net.load_state_dict(best_state)
            print(f"Restored best validation weights (val_loss={best_val:.4f}).")

    @staticmethod
    def _split_train_val(num_samples: int, groups, val_fraction: float, seed: int):
        """(train_idx, val_idx) for the train / validation split.

        GAME-LEVEL when ``groups`` is given: whole games are sent to one side (so no game
        straddles the split), otherwise sample-level. Returns torch index tensors.
        """
        rng = np.random.default_rng(seed)
        if val_fraction <= 0:
            return torch.arange(num_samples), torch.arange(0)
        if groups is not None:
            games = np.unique(groups)
            rng.shuffle(games)
            n_val_games = max(1, int(len(games) * val_fraction))
            val_games = set(games[:n_val_games].tolist())
            is_val = np.fromiter((g in val_games for g in groups), dtype=bool, count=len(groups))
            train_idx, val_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
        else: # just random split if no game ids are given
            perm = rng.permutation(num_samples)
            n_val = int(num_samples * val_fraction)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        return torch.from_numpy(train_idx), torch.from_numpy(val_idx)

    def _evaluate(self, x_val, y_val, loss_fn) -> tuple[float, float]:
        """Validation loss and accuracy (no-op if the val split is empty)."""
        if len(x_val) == 0:
            return float("nan"), float("nan")
        self.net.eval()
        with torch.no_grad():
            logits = self.net(x_val)
            loss = loss_fn(logits, y_val).item()
            acc = (logits.argmax(dim=1) == y_val).float().mean().item()
        return loss, acc

    def save(self, path):
        torch.save({"state_dict": self.net.state_dict(), "num_players": self.num_players, \
            "use_center": self.use_center_card, "in_dim": self.feature_dim, "role_names": self.role_names}, path)

    @classmethod
    def load(cls, path, device=None) -> "PosteriorFactorModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(checkpoint["num_players"], checkpoint["use_center"], device=device)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.to(model.device)
        return model


def build_dataset(result_dirs, num_players, use_center_card, opponent_types=None, status="completed"):
    role_config = ROLE_CONFIGS if use_center_card else ROLE_CONFIGS_NO_CENTER
    role_names = list(role_config[num_players])
    role_index = {role: i for i, role in enumerate(role_names)}
    entity_ids = valid_player_ids(num_players, use_center_card)
    features, labels, groups = [], [], []  # groups[i] = game id of sample i (for a game-level val split)
    game_files = []
    
    for result_dir in result_dirs:
        game_files += glob.glob(os.path.join(result_dir, "**", "game_*.json"), recursive=True)
    
    n_games = 0
    for fp in game_files:
        with open(fp, encoding="utf-8") as f:
            game_log = json.load(f)
        if status and game_log.get("status") != status:
            continue
        if game_log.get("num_players") != num_players:
            continue
        if (game_log.get("center_card") is not None) != use_center_card:
            continue
        
        if opponent_types:
            non_prince = [t for pid, t in game_log.get("agent_types", {}).items() if pid != "0"]
            if not non_prince or any(t not in opponent_types for t in non_prince):
                continue
            
        roles, center = game_log["roles"], game_log.get("center_card")
        qa_history = game_log["qa_history"]
        n_games += 1
        
        for entity_id in entity_ids:
            true_role = center if entity_id == CENTER_ID else roles.get(str(entity_id))
            if true_role not in role_index:
                continue
            features.append(encode_entity(qa_history, entity_id, entity_ids, role_names, role_index))
            labels.append(role_index[true_role])
            groups.append(n_games)  # all entities of this game share one game id
    if not features:
        raise SystemExit("No training samples matched the filters.")
    print(f"Dataset: {n_games} games -> {len(features)} (entity, history) samples.")
    return np.stack(features), np.array(labels, dtype=np.int64), np.array(groups, dtype=np.int64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-players", type=int, required=True)
    parser.add_argument("--use-center", action="store_true", \
        help="true: learn only from games WITH a center card (default); false: only WITHOUT")
    parser.add_argument("--opponent-types", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.2, \
        help="fraction of samples held out for validation (0 disables validation)")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda / cpu (default: auto)")
    args = parser.parse_args()

    features, labels, groups = build_dataset(args.result_dirs, args.num_players, args.use_center, \
        opponent_types=args.opponent_types)
    model = PosteriorFactorModel(args.num_players, args.use_center, device=args.device)
    model.fit(features, labels, groups=groups, epochs=args.epochs, val_fraction=args.val_fraction)
    model.save(args.out)
    print(f"Saved posterior-factor model -> {args.out}")
    print("Plug in via: TofuFactorGraph(num_players, use_center_card, posterior_model=PosteriorFactorModel.load(path))")