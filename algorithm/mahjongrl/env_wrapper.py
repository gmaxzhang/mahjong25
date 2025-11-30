# algorithm/mahjongrl/env_wrapper.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import random

# Import your simulator & helpers (must exist in your tree)
from algorithm.sim_and_train import (
    Env, SUITS, HONORS, is_flower,
    RandomPolicy, WinProbPolicy, PayoutOptPolicy, AggroPolicy, HybridPolicy, HybridAggroPolicy, FlexibleAggroPolicy, FlexibleAggroPolicyD
)

# ===========================
# Tile indexing (34 non-flower kinds)
# ===========================
TILES_34: List[str] = [f"{r}{s}" for s in SUITS for r in map(str, range(1, 10))] + HONORS
T2I = {t: i for i, t in enumerate(TILES_34)}

def _tile_code(x: Any) -> Optional[str]:
    """Best-effort to convert env tile objects into canonical strings like '5b','E','C'."""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("code") or x.get("name") or None
    # common object attrs
    for a in ("code", "name", "label"):
        if hasattr(x, a):
            v = getattr(x, a)
            if isinstance(v, str):
                return v
    # fallback
    s = str(x)
    return s if s else None

def tile_list_to_counts_34(ts: List[Any]) -> np.ndarray:
    v = np.zeros(34, dtype=np.float32)
    for t in ts:
        c = _tile_code(t)
        if c in T2I:
            v[T2I[c]] += 1.0
    return v

def encode_melds_cheap(melds: List[Any]) -> np.ndarray:
    """Counts of open/closed pungs/kongs and chows, + declared total."""
    open_p, open_k, open_c = 0, 0, 0
    closed_p, closed_k, closed_c = 0, 0, 0
    for m in melds:
        k = (getattr(m, "type", None) or getattr(m, "kind", None) or "").lower()
        is_open = bool(getattr(m, "open", True))
        if k in ("chi", "chow"):
            if is_open: open_c += 1
            else:       closed_c += 1
        elif k in ("pong", "pung"):
            if is_open: open_p += 1
            else:       closed_p += 1
        elif k in ("kan", "kong"):
            if is_open: open_k += 1
            else:       closed_k += 1
    return np.array(
        [open_c, closed_c, open_p, closed_p, open_k, closed_k,
         open_c + closed_c + open_p + closed_p + open_k + closed_k],
        dtype=np.float32
    )

# ===========================
# Recent action history encoding (H most recent)
# ===========================
ACTION_TYPES = ["discard", "draw", "chow", "pung", "kong", "ron", "tsumo", "pass", "flower"]
A2I = {a: i for i, a in enumerate(ACTION_TYPES)}

def _encode_one_action(seat: int, kind: str, tile: Optional[Any]) -> np.ndarray:
    """one-hot seat(4) + one-hot action(lenA) + one-hot tile(34)."""
    v = np.zeros(4 + len(ACTION_TYPES) + 34, dtype=np.float32)
    if 0 <= seat < 4:
        v[seat] = 1.0
    k = (kind or "").lower()
    if k in A2I:
        v[4 + A2I[k]] = 1.0
    tc = _tile_code(tile)
    if tc in T2I:
        v[4 + len(ACTION_TYPES) + T2I[tc]] = 1.0
    return v

def _recent_action_vector(env: Env, H: int = 8) -> np.ndarray:
    """Try env.action_log[-H:], else approximate from last discards."""
    feat_len = 4 + len(ACTION_TYPES) + 34
    out = np.zeros(H * feat_len, dtype=np.float32)

    hist: List[Tuple[int, str, Optional[Any]]] = []
    if hasattr(env, "action_log") and isinstance(env.action_log, list):
        for rec in env.action_log[-H:]:
            if isinstance(rec, dict):
                seat = rec.get("seat")
                try:
                    seat = int(seat)
                except Exception:
                    seat = -1
                kind = str(rec.get("kind", ""))
                tile = rec.get("tile")
                hist.append((seat, kind, tile))
    else:
        # Fallback: last discard from each seat
        try:
            for s in range(4):
                row = env.discards[s] if hasattr(env, "discards") else []
                if row:
                    hist.append((s, "discard", row[-1]))
        except Exception:
            pass

    hist = hist[-H:]
    start = max(0, H - len(hist))
    for i, (seat, kind, tile) in enumerate(hist, start=start):
        out[i * feat_len:(i + 1) * feat_len] = _encode_one_action(seat, kind, tile)
    return out

# ===========================
# Global + AI-parameter features
# ===========================
def _rules_ai_vector(rules: Dict[str, Any]) -> np.ndarray:
    """
    Encode a small, stable vector from rules['ai'] to help the model 'remember'
    opponent style. Keys not present default to 0.
    """
    ai = rules.get("ai", {}) if isinstance(rules, dict) else {}
    keys = [
        "tsumo_ev_multiplier", "ron_bias", "feed_risk_coeff",
        "discard_keep_weight", "discard_risk_weight",
        "claim_aggressiveness", "pung_margin", "kong_margin", "chow_bias"
    ]
    return np.array([float(ai.get(k, 0.0)) for k in keys], dtype=np.float32)

def global_table_features(env: Env, seat: int, rules: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Turn, total discards, wall fraction, seat one-hot, + optional AI params."""
    turn = float(getattr(env, "turn", 0))
    discards = 0
    if hasattr(env, "discards"):
        try:
            discards = sum(len(d) for d in env.discards)
        except Exception:
            discards = 0
    wall_len = float(len(getattr(env, "wall", [])))
    # 144 covers 108 numerals + 16 honors + 8 flowers + 12 seasons/animals if present; ratio is just a proxy.
    wall_frac = wall_len / 144.0 if 144.0 > 0 else 0.0
    seat_oh = np.eye(4, dtype=np.float32)[seat]
    ai_vec = _rules_ai_vector(rules or {})
    return np.concatenate([np.array([turn, float(discards), wall_frac], dtype=np.float32), seat_oh, ai_vec])

# ===========================
# Observation builder (with history)
# ===========================
def build_observation(env: Env, seat: int) -> np.ndarray:
    """
    Features:
      - my concealed (non-flower) 34-counts
      - aggregate discards 34-counts (non-flower)
      - #flowers I have
      - my meld summary
      - global features (+ AI params)
      - recent action history (H=8)
    """
    p = env.players[seat]
    hand_nf = [t for t in getattr(p, "concealed", []) if not is_flower(_tile_code(t))]
    hand_cnt = tile_list_to_counts_34(hand_nf)

    all_discards: List[Any] = []
    if hasattr(env, "discards"):
        for row in env.discards:
            all_discards.extend(row)
    disc_nf = [t for t in all_discards if not is_flower(_tile_code(t))]
    disc_cnt = tile_list_to_counts_34(disc_nf)

    flowers = np.array([len(getattr(p, "flowers", []))], dtype=np.float32)
    meld_vec = encode_melds_cheap(getattr(p, "melds", []))
    global_vec = global_table_features(env, seat, getattr(env, "rules", None))
    hist_vec = _recent_action_vector(env, H=8)

    return np.concatenate([hand_cnt, disc_cnt, flowers, meld_vec, global_vec, hist_vec], axis=0)

# ===========================
# Reward as ACTUAL point transfers
# ===========================


def _terminal_as_dict(term: Any) -> Dict[str, Any]:
    """
    Make a best-effort dict view of the terminal object.
    Works if term is already a dict or some small dataclass-like object.
    """
    if term is None:
        return {}
    if isinstance(term, dict):
        return term
    d: Dict[str, Any] = {}
    for name in dir(term):
        if name.startswith("_"):
            continue
        try:
            d[name] = getattr(term, name)
        except Exception:
            pass
    return d


def compute_rl_reward(
    term: Any,
    seat: int,
    rules: Dict[str, Any],
    *,
    score_scale: float = 8000.0,
    win_bonus: float = 0.3,
    dealin_penalty: float = 0.2,
) -> float:
    """
    Map Env.terminal into a scalar reward for RL.

    Design:
      - Base term is normalized score delta for this seat, with soft clipping.
      - Add a small +/- win_bonus to emphasise win vs loss.
      - Add a small extra penalty if we dealt into ron.
      - If we cannot recover score deltas, we still use the win/loss signal.

    This keeps the reward magnitude modest (no huge spikes),
    while still nudging the agent toward winning more often.
    """
    if term is None:
        return 0.0

    d = _terminal_as_dict(term)
    src = d.get("source", None) or d.get("reason", None)

    # -------------------- 1) Score delta --------------------
    delta = 0.0

    # (a) Try explicit delta arrays/scalars
    for key in ("score_delta", "score_deltas", "delta_scores",
                "points_delta", "points_deltas"):
        v = d.get(key, None)
        if isinstance(v, (list, tuple)) and len(v) > seat:
            delta = float(v[seat])
            break
        if isinstance(v, (int, float)):
            delta = float(v)
            break

    # (b) If not found, try final - initial scores
    if delta == 0.0:
        after = d.get("scores", None) or d.get("final_scores", None)
        before = d.get("scores_before", None) or d.get("start_scores", None)
        if isinstance(after, (list, tuple)) and isinstance(before, (list, tuple)):
            try:
                delta = float(after[seat] - before[seat])
            except Exception:
                pass

    # Normalize & softly cap so we don't blow up the scale.
    base = 0.0
    if score_scale > 0.0 and delta != 0.0:
        x = delta / score_scale
        # hard clip; you can replace with tanh if you prefer
        base = max(-3.0, min(3.0, x))

    # -------------------- 2) Win / loss shaping --------------------
    win_flag = 0.0
    lose_flag = 0.0

    w = d.get("winner", None)
    if isinstance(w, int):
        if w == seat:
            win_flag = 1.0
        else:
            lose_flag = 1.0

    winners = d.get("winners", None)
    if isinstance(winners, (list, tuple)) and winners:
        if seat in winners:
            win_flag = 1.0
        elif win_flag == 0.0:  # only treat as loss if we aren't already a winner
            lose_flag = 1.0

    loser = d.get("loser", None)
    if isinstance(loser, int) and loser == seat:
        lose_flag = 1.0

    winloss = win_bonus * (win_flag - lose_flag)

    # -------------------- 3) Extra penalty for dealing in --------------------
    extra = 0.0
    if src == "ron" and d.get("loser", None) == seat:
        extra -= dealin_penalty

    return base + winloss + extra

# def compute_rl_reward(terminal: Dict[str, Any], seat: int, rules: Dict) -> int:
#     """
#     Winner/loser deltas in points with Ron/Tsumo multipliers from rules['payouts'].
#     Supports single-ron and multi-ron; adds side events if `side_delta` present.
#     """
#     def _extract_pts(obj: Dict[str, Any]) -> int:
#         for k in ("points", "score", "hand_points", "base_points", "value", "total_points"):
#             v = obj.get(k, None)
#             if isinstance(v, (int, float)):
#                 return int(v)
#         sub = obj.get("hand") or obj.get("scoring")
#         if isinstance(sub, dict):
#             for kk in ("points", "score", "value", "total"):
#                 v = sub.get(kk, None)
#                 if isinstance(v, (int, float)):
#                     return int(v)
#         return 0

#     pay_disc = rules.get("payouts", {}).get("on_discard", {})
#     pay_self = rules.get("payouts", {}).get("on_self_draw", {})

#     ron_winner_gain   = int(pay_disc.get("winner_gain", 1))
#     ron_loser_loss    = int(pay_disc.get("loser_loss", -1))
#     tsumo_winner_gain = int(pay_self.get("winner_gain", 3))
#     tsumo_others_loss = int(pay_self.get("others_loss_each", -1))

#     delta = [0, 0, 0, 0]
#     src = terminal.get("source")

#     if src == "drawn_game" or src is None:
#         pass

#     elif src == "discard":
#         loser = terminal.get("ron_loser")
#         winners = terminal.get("winners")
#         if winners and isinstance(winners, list):
#             total_loser = 0
#             for wrec in winners:
#                 w = int(wrec.get("seat"))
#                 pts = _extract_pts(wrec)
#                 delta[w] += ron_winner_gain * pts
#                 total_loser += ron_loser_loss * pts
#             if isinstance(loser, int):
#                 delta[int(loser)] += total_loser
#         else:
#             w = int(terminal.get("winner"))
#             pts = _extract_pts(terminal)
#             delta[w] += ron_winner_gain * pts
#             if isinstance(loser, int):
#                 delta[int(loser)] += ron_loser_loss * pts

#     elif src == "self_draw":
#         w = int(terminal.get("winner"))
#         pts = _extract_pts(terminal)
#         delta[w] += tsumo_winner_gain * pts
#         for s in range(4):
#             if s != w:
#                 delta[s] += tsumo_others_loss * pts

#     sd = terminal.get("side_delta")
#     if isinstance(sd, list) and len(sd) == 4:
#         delta = [int(a) + int(b) for a, b in zip(delta, sd)]

#     return int(delta[seat])

# ===========================
# Flower resolution utilities
# ===========================
FlowerKeepDecider = Callable[[Any, str, Dict[str, Any]], bool]

def _default_keep_decider(player, tile_code: str, rules: Dict[str, Any]) -> bool:
    """
    Default: DO NOT keep flowers. If pursuing 'all apart', wrap this in train.py
    to return True for that player.
    """
    return False

def resolve_flowers_for_player(
    env: Env,
    seat: int,
    rules: Dict[str, Any],
    keep_decider: FlowerKeepDecider = _default_keep_decider
) -> None:
    """
    Expose any flowers in the player's concealed tiles, drawing replacements from the
    TAIL of the wall. If replacement is also a flower and keep_decider returns False,
    repeat. If keep_decider returns True for a flower, keep it in-hand (exception).
    """
    p = env.players[seat]
    while True:
        idx = None
        for i, t in enumerate(getattr(p, "concealed", [])):
            code = _tile_code(t)
            if is_flower(code) and not keep_decider(p, code, rules):
                idx = i
                break
        if idx is None:
            break

        flower_tile = p.concealed.pop(idx)
        if not hasattr(p, "flowers"):
            p.flowers = []
        p.flowers.append(flower_tile)

        if not env.wall:
            break
        replacement = env.wall.pop()  # draw from tail
        p.concealed.append(replacement)
        # loop continues until no (unkept) flowers remain

def resolve_flowers_in_all_hands(
    env: Env,
    rules: Dict[str, Any],
    keep_decider: FlowerKeepDecider = _default_keep_decider
) -> None:
    for s in range(4):
        resolve_flowers_for_player(env, s, rules, keep_decider)

def shuffle_unknown_slots_and_resolve_flowers(
    env: Env,
    seat_view: int,
    rules: Dict[str, Any],
    keep_decider: FlowerKeepDecider = _default_keep_decider,
) -> None:
    """
    Shuffle all UNKNOWN slots together (opponents’ concealed + the wall),
    re-deal them, then resolve flowers for those opponents by exposing and drawing
    replacements from the tail (unless keep_decider says to keep).
    """
    # gather pool (opponents + wall)
    slots = []  # ("opps", seat, idx) or ("wall", None, idx)
    pool = []

    # Opponents
    for s in range(4):
        if s == seat_view:
            continue
        opp = env.players[s]
        for i in range(len(getattr(opp, "concealed", []))):
            slots.append(("opps", s, i))
            pool.append(opp.concealed[i])

    # Wall
    for i in range(len(getattr(env, "wall", []))):
        slots.append(("wall", None, i))
        pool.append(env.wall[i])

    random.shuffle(pool)

    it = iter(pool)
    for kind, s, i in slots:
        t = next(it)
        if kind == "opps":
            env.players[s].concealed[i] = t
        else:
            env.wall[i] = t

    # Resolve flowers for opponents only (we don't touch seat_view's hand here)
    for s in range(4):
        if s != seat_view:
            resolve_flowers_for_player(env, s, rules, keep_decider)

# ===========================
# Opponent lineup helper
# ===========================
def make_lineup_with_rl(rl_policy, rules: Dict, others: List[str]) -> List[Any]:
    """Seat 0 is RL; others are from {random, wp, payout, hybrid, aggro}."""
    name2cls = {
        "random": RandomPolicy,
        "wp": WinProbPolicy,
        "payout": PayoutOptPolicy,
        "hybrid": HybridPolicy,
        "aggro": AggroPolicy,
        "hyaggro": HybridAggroPolicy,
        "flexaggro": FlexibleAggroPolicy,
        "flexaggrod": FlexibleAggroPolicyD,
    }
    assert len(others) == 3, "lineup must provide exactly 3 opponent tags"
    lineup = [rl_policy]
    for i, tag in enumerate(others, start=1):
        if tag not in name2cls:
            raise ValueError(f"unknown opponent tag: {tag!r}")
        lineup.append(name2cls[tag](i, rules, tuner=None))
    return lineup
