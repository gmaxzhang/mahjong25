# mahjong_rl/env_wrapper.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np

# Import your simulator & scoring
from sim_and_train import (
    Env, SUITS, HONORS, is_flower, _tile_rank_suit, _is_suit_tile,
    RandomPolicy, WinProbPolicy, PayoutOptPolicy, AggroPolicy, HybridPolicy,
)
from custom_mahjong_rules import score_hand, HandState

# --- Tile indexing (34 non-flower kinds)
TILES_34: List[str] = [f"{r}{s}" for s in SUITS for r in map(str, range(1,10))] + HONORS
T2I = {t:i for i,t in enumerate(TILES_34)}

def tile_list_to_counts_34(ts: List[str]) -> np.ndarray:
    v = np.zeros(34, dtype=np.float32)
    for t in ts:
        if t in T2I: v[T2I[t]] += 1.0
    return v

def encode_melds_cheap(melds: List[Any]) -> np.ndarray:
    """Counts of open/closed pungs/kongs and chows, + declared total."""
    open_p, open_k, open_c = 0,0,0
    closed_p, closed_k, closed_c = 0,0,0
    for m in melds:
        k = (getattr(m, "type", None) or getattr(m, "kind", None) or "").lower()
        if k in ("chi","chow"):
            if getattr(m, "open", True): open_c += 1
            else: closed_c += 1
        elif k in ("pong","pung"):
            if getattr(m, "open", True): open_p += 1
            else: closed_p += 1
        elif k in ("kan","kong"):
            if getattr(m, "open", True): open_k += 1
            else: closed_k += 1
    return np.array([open_c,closed_c,open_p,closed_p,open_k,closed_k,
                     open_c+closed_c+open_p+closed_p+open_k+closed_k], dtype=np.float32)

def global_table_features(env: Env, seat: int) -> np.ndarray:
    """Very light global context."""
    # Turn index, discards length, seat index (one-hot 4), wall size fraction.
    seat_oh = np.eye(4, dtype=np.float32)[seat]
    discards = sum(len(d) for d in env.discards)
    wall_frac = float(len(env.wall)) / 144.0
    return np.concatenate([
        np.array([float(env.turn), float(discards), wall_frac], dtype=np.float32),
        seat_oh
    ])

def build_observation(env: Env, seat: int) -> np.ndarray:
    """Concatenate per-seat features into a single float vector."""
    p = env.players[seat]
    hand_nf = [t for t in p.concealed if not is_flower(t)]
    hand_cnt = tile_list_to_counts_34(hand_nf)
    disc_cnt = tile_list_to_counts_34([t for row in env.discards for t in row])
    flowers = np.array([len(p.flowers)], dtype=np.float32)
    meld_vec = encode_melds_cheap(p.melds)
    global_vec = global_table_features(env, seat)
    return np.concatenate([hand_cnt, disc_cnt, flowers, meld_vec, global_vec], axis=0)

def compute_rl_reward(terminal: Dict[str,Any], seat: int, rules: Dict) -> int:
    """Map the terminal dict to the RL reward for `seat` (your scoring/payouts)."""
    pay_disc = rules.get("payouts", {}).get("on_discard", {})
    pay_self = rules.get("payouts", {}).get("on_self_draw", {})

    # Build deltas like summarize_jsonl
    delta = [0,0,0,0]
    src = terminal.get("source")
    if src == "drawn_game":
        # no points, might have side events
        pass
    elif src == "discard":
        wg = int(pay_disc.get("winner_gain", 1))
        ll = int(pay_disc.get("loser_loss", -1))
        loser = terminal.get("ron_loser")
        winners = terminal.get("winners")
        if winners:  # multi-ron
            total_loser_loss = 0
            for item in winners:
                w = int(item.get("seat"))
                pts = int(item.get("points", 0))
                delta[w] += wg * pts
                total_loser_loss += ll * pts
            if isinstance(loser, int):
                delta[loser] += total_loser_loss
        else:
            w = int(terminal.get("winner"))
            pts = int(terminal.get("points", 0))
            delta[w] += wg * pts
            if isinstance(loser, int):
                delta[loser] += ll * pts
    elif src == "self_draw":
        wg = int(pay_self.get("winner_gain", 3))
        ol = int(pay_self.get("others_loss_each", -1))
        w = int(terminal.get("winner"))
        pts = int(terminal.get("points", 0))
        delta[w] += wg * pts
        for s in range(4):
            if s != w: delta[s] += ol * pts

    # Add side events if any
    if isinstance(terminal.get("side_delta"), list) and len(terminal["side_delta"]) == 4:
        for i in range(4): delta[i] += int(terminal["side_delta"][i])
    return int(delta[seat])

def make_lineup_with_rl(rl_policy, rules: Dict, others: List[str]) -> List[Any]:
    """Seat 0 is RL; others are from {random, wp, payout, hybrid, aggro}."""
    name2cls = {
        "random": RandomPolicy,
        "wp": WinProbPolicy,
        "payout": PayoutOptPolicy,
        "hybrid": HybridPolicy,
        "aggro": AggroPolicy,
    }
    assert len(others) == 3
    lineup = [rl_policy]
    for i, tag in enumerate(others, start=1):
        lineup.append(name2cls[tag](i, rules, tuner=None))
    return lineup
