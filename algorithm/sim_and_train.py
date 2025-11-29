# -*- coding: utf-8 -*-
"""
sim_and_train.py — simulator with:
  - strict win-shape by COMPLETE hand (declared melds fixed)
  - correct flower detection (F1..F8 only; 'F' is GREEN DRAGON)
  - chow/pung/OPEN-KONG (via add-kan) and CLOSED-KONG
  - back-wall flower and kong replacement
  - 4-same-discard side payment
  - menqing gate only for 10/11-point totals (blocked only if a discard was claimed)
  - zero-sum summary
  - optional win snapshots
  - claims priority: Ron > Pung > Chow (pung over chow)
  - multi-ron on same discard: all eligible ron claims win; discarder pays each
  - Opening draw snapshots per seat (before & after flower settlement)

Adaptive behavior (episode-by-episode):
  * Seven-pairs tendency is learned (and now strongly bounded & context-gated).
  * Claim biases (chow/pung/open/closed-kong), keep/risk weights, and margins
    are all dynamically adjusted using outcome EMAs and action-level attributions.
  * Add-kan is policy-gated and tracked.

Scoring rule:
  * If a special hand exists, its points REPLACE base.
  * If multiple specials apply, their points SUM and still replace base.
  * Bonuses (e.g., flowers/other extras) add on top of that replacement.

Notes on kongs:
  * Closed-kong from concealed (your turn).
  * Add-kan (upgrade exposed pung to open kong) may be attempted on your turn
    and immediately after a pung claim (with replacement draw & tsumo check).
"""

from __future__ import annotations
import json, random
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from algorithm.rules_io import load_rules
from algorithm.gameplay0 import HandState, Meld, score_hand  # scoring = source of truth

# ---------------------------- Tiles & Wall ----------------------------

SUITS = ["b","w","t"]  # bamboos, characters, circles
RANKS = [str(r) for r in range(1, 10)]
HONORS = ["E","S","W","N","C","F","B"]  # 'F' is Green Dragon (NOT a flower)

# ---------- Tile class mapping (0..33) for forced discards ----------
# 0..8: 1b..9b, 9..17: 1w..9w, 18..26: 1t..9t, 27..33: E,S,W,N,C,F,B
_SUITS_ORDER = ["b","w","t"]
_HONOR_ORDER = ["E","S","W","N","C","F","B"]  # 'F' here is the Green Dragon (NOT flower)
_SUITS_IDX = {s:i for i,s in enumerate(_SUITS_ORDER)}
_HONOR_IDX = {h:i for i,h in enumerate(_HONOR_ORDER)}

def tile_to_class(t: str) -> Optional[int]:
    if is_flower(t):  # F1..F8 → flowers are not part of 0..33 action space
        return None
    if _is_suit_tile(t):
        r, s = _tile_rank_suit(t)
        return _SUITS_IDX[s] * 9 + (r - 1)  # 0..26
    # honors (single char among HONORS)
    if t in _HONOR_IDX:
        return 27 + _HONOR_IDX[t]
    return None  # unknown / malformed

# class (0..33) -> canonical tile string
def class_to_tile(cls: int) -> Optional[str]:
    if 0 <= cls <= 26:
        suit = _SUITS_ORDER[cls // 9]
        rank = 1 + (cls % 9)
        return f"{rank}{suit}"
    if 27 <= cls <= 33:
        return _HONOR_ORDER[cls - 27]   # "E","S","W","N","C","F","B" (F = Green Dragon)
    return None

# choose an actual tile from this player's hand that matches the class
def pick_tile_from_class_in_hand(hand: List[str], cls: int) -> Optional[str]:
    # flowers are never in the 0..33 action space
    cand = class_to_tile(cls)
    if cand is None:
        return None
    # If the canonical tile exists, use it
    if cand in hand:
        return cand
    # If not present (e.g., stale force), give up
    return None

def is_flower(x: str) -> bool:
    return len(x) >= 2 and x[0] == "F" and x[1:].isdigit()

def _is_suit_tile(t: str) -> bool:
    return len(t) == 2 and t[0].isdigit() and t[1] in SUITS

def _tile_rank_suit(t: str) -> Tuple[int,str]:
    return (int(t[0]), t[1])

def all_tiles(include_flowers: bool=True) -> List[str]:
    tiles = [f"{r}{s}" for s in SUITS for r in RANKS] + HONORS
    tiles = [t for t in tiles for _ in range(4)]
    if include_flowers:
        tiles += [f"F{i}" for i in range(1,9)]  # 8 flowers
    return tiles

class Wall:
    """
    Two-layer wall made of pairs. Drawing from front/back always takes the TOP tile
    of the first/last pair, then the bottom, then removes the pair.
    """
    def __init__(self, include_flowers: bool, seed: Optional[int]):
        rng = random.Random(seed)
        arr = all_tiles(include_flowers)
        rng.shuffle(arr)
        # Build pairs [top, bottom]
        self.pairs: List[List[Optional[str]]] = []
        for i in range(0, len(arr), 2):
            self.pairs.append([arr[i], arr[i+1]])

    def _draw_from_index(self, idx: int) -> Optional[str]:
        if not self.pairs:
            return None
        pair = self.pairs[idx]
        # top first
        if pair[0] is not None:
            t = pair[0]
            pair[0] = None
            return t
        # then bottom, remove the pair
        t = pair[1]
        # pop left or right based on idx
        if idx == 0:
            self.pairs.pop(0)
        else:
            self.pairs.pop()
        return t

    def draw_front(self) -> Optional[str]:
        return self._draw_from_index(0) if self.pairs else None

    def draw_back(self) -> Optional[str]:
        return self._draw_from_index(-1) if self.pairs else None

    def __len__(self) -> int:
        # remaining tiles = 2*#pairs - #tops_already_taken
        rem = 0
        for top, bot in self.pairs:
            rem += (1 if top is not None else 0) + (1 if bot is not None else 0)
        return rem

    def __bool__(self) -> bool:
        return len(self) > 0

def make_wall(include_flowers: bool, seed: Optional[int]) -> Wall:
    return Wall(include_flowers, seed)

# ---------------------------- Player State ----------------------------

@dataclass
class PlayerView:
    concealed: List[str]
    melds: List[Meld]
    flowers: List[str]
    used_discard_claim: bool = False  # chow/pung/open-kong came from a discard
    needs_discard: bool = False       # True immediately after chow/pung (and post add-kan replacement)

# ---------------------------- Hand validators (shape gate) ----------------------------

def _is_seven_pairs(non_flower_14: List[str]) -> bool:
    if len(non_flower_14) != 14: return False
    cnt = Counter(non_flower_14)
    pairs = 0
    for v in cnt.values():
        if v == 2: pairs += 1
        elif v == 4: pairs += 2
        else: return False
    return pairs == 7

def _is_all_apart(tiles14_with_flowers: List[str]) -> bool:
    tiles = tiles14_with_flowers[:]
    if len(tiles) != 14: return False
    if sum(1 for t in tiles if is_flower(t)) > 1: return False
    suited = [x for x in tiles if _is_suit_tile(x)]
    if len(suited) != 9: return False
    others = [x for x in tiles if not _is_suit_tile(x)]
    if len(others) != 5 or len(set(others)) != 5: return False
    groups = [{1,4,7},{2,5,8},{3,6,9}]
    def ok_suit(s: str) -> bool:
        ranks = sorted(int(x[0]) for x in suited if x[1]==s)
        return any(sorted(g)==ranks for g in groups)
    return ok_suit("b") and ok_suit("w") and ok_suit("t")

def _can_form_standard_4m1p(non_flower_14: List[str]) -> bool:
    tiles = sorted(non_flower_14)
    if len(tiles) != 14: return False
    cnt = Counter(tiles)
    def remove_triplet_or_seq(c: Counter) -> bool:
        if sum(c.values()) == 0: return True
        t = min([k for k,v in c.items() if v>0])
        if c[t] >= 3:
            c[t] -= 3
            if remove_triplet_or_seq(c): return True
            c[t] += 3
        if _is_suit_tile(t):
            r,s = _tile_rank_suit(t)
            a,b = f"{r+1}{s}", f"{r+2}{s}"
            if c[a] > 0 and c[b] > 0:
                c[t] -= 1; c[a] -= 1; c[b] -= 1
                if remove_triplet_or_seq(c): return True
                c[t] += 1; c[a] += 1; c[b] += 1
        return False
    for k,v in list(cnt.items()):
        if v >= 2:
            cnt[k] -= 2
            if remove_triplet_or_seq(cnt.copy()):
                return True
            cnt[k] += 2
    return False

def _legal_win_shape(
    concealed: List[str],
    melds: List[Meld],
    winning_tile: str,
    source: str
) -> Tuple[bool, str]:
    """
    Declared melds are FIXED. Validate by:
      1) Build full non-flower multiset = declared_tiles + concealed + winning_tile.
      2) Remove tiles occupied by declared sets from the multiset (they are locked).
      3) Solve residual must be (4 - #declared) melds + 1 pair.
      4) Seven Pairs / All Apart only if no declared sets.
    """
    declared = [
        m for m in (melds or [])
        if _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None))) in {"chow", "pung", "kong"}
    ]
    m_fixed = len(declared)

    declared_tiles: List[str] = []
    for m in declared:
        declared_tiles.extend(list(getattr(m, "tiles", [])))

    full_non_fl = [t for t in (declared_tiles + concealed + [winning_tile]) if not is_flower(t)]
    if len(full_non_fl) < 2:
        return False, "too_small"

    if m_fixed == 0:
        full14_view = concealed[:] + [winning_tile] + declared_tiles
        if _is_all_apart(full14_view):
            return True, "all_apart"
        if len(full_non_fl) == 14 and _is_seven_pairs(full_non_fl):
            return True, "seven_pairs"

    cnt = Counter(full_non_fl)
    for m in declared:
        tiles = list(getattr(m, "tiles", []))
        for t in tiles:
            if cnt[t] <= 0:
                return False, "declared_overlaps_missing"
            cnt[t] -= 1

    target_m = 4 - m_fixed
    if target_m < 0:
        return False, "too_many_declared"

    residual = [k for k, v in cnt.items() for _ in range(v)]

    if target_m == 0:
        ok_pair = (len(residual) == 2 and residual[0] == residual[1])
        return (ok_pair, "standard" if ok_pair else "wrong_residual_size")

    need_len = target_m * 3 + 2
    if len(residual) != need_len:
        return False, "wrong_residual_size"

    def _can_form_m_melds_plus_pair(non_flower_tiles: List[str], m_needed: int) -> bool:
        tiles = sorted(non_flower_tiles)
        if len(tiles) != m_needed * 3 + 2:
            return False
        c = Counter(tiles)

        def remove_melds_then_pair(cc: Counter, m_left: int) -> bool:
            if m_left == 0:
                if sum(cc.values()) == 2:
                    a = next(k for k, v in cc.items() if v)
                    return cc[a] == 2
                return False
            t = min(k for k, v in cc.items() if v > 0)
            if cc[t] >= 3:
                cc[t] -= 3
                if remove_melds_then_pair(cc, m_left - 1): return True
                cc[t] += 3
            if _is_suit_tile(t):
                r, s = _tile_rank_suit(t)
                a, b = f"{r+1}{s}", f"{r+2}{s}"
                if cc[a] > 0 and cc[b] > 0:
                    cc[t] -= 1; cc[a] -= 1; cc[b] -= 1
                    if remove_melds_then_pair(cc, m_left - 1): return True
                    cc[t] += 1; cc[a] += 1; cc[b] += 1
            return False

        for k, v in list(c.items()):
            if v >= 2:
                c2 = c.copy()
                c2[k] -= 2
                if remove_melds_then_pair(c2, m_needed):
                    return True
        return False

    return (True, "standard") if _can_form_m_melds_plus_pair(residual, target_m) else (False, "no_shape")

# ---------------------------- Utility: hand snapshots ----------------------------

def _normalize_meld_type(t: Optional[str]) -> str:
    if not t: return "?"
    t = t.lower()
    if t in ("chi", "chow"): return "chow"
    if t in ("pong", "pung"): return "pung"
    if t in ("kan", "kong"):  return "kong"
    return t

def _serialize_meld(m) -> Dict[str,Any]:
    try:
        name = _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None)))
        tiles = list(getattr(m, "tiles"))
        opened = bool(getattr(m, "open", getattr(m, "opened", True)))
        formed = bool(getattr(m, "formed_by_claim", False))
        from_disc = bool(getattr(m, "from_discard", False))
        return {"kind": name, "tiles": tiles, "open": opened, "claimed": formed, "from_discard": from_disc}
    except Exception:
        return {"repr": repr(m)}

def _snapshot_winning_hand(env, seat: int, hs: HandState, points: int, source: str, extra: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    p = env.players[seat]
    concealed_before = list(hs.concealed) if source == "self_draw" else list(p.concealed)
    if source == "self_draw":
        concealed_after = sorted([t for t in p.concealed])
    else:
        concealed_after = sorted([t for t in (p.concealed + [hs.winning_tile])])

    snap = {
        "seat": seat,
        "source": source,
        "winning_tile": hs.winning_tile,
        "points": points,
        "flowers": sorted(list(p.flowers)),
        "melds": [_serialize_meld(m) for m in p.melds],
        "concealed_before_win": concealed_before,
        "concealed_after_win": concealed_after,
        "used_discard_claim": bool(getattr(p, "used_discard_claim", False)),
        "claim_log": list(getattr(env, "claim_log", [])),
    }
    if extra:
        snap.update(extra)
    return snap

# ---------------------------- Rough evaluators ----------------------------

def rough_ukeire(hand: List[str]) -> int:
    cnt = Counter(x for x in hand if not is_flower(x))
    eff: set[str] = set()
    for t,c in cnt.items():
        if _is_suit_tile(t):
            eff.add(t)
            r,s = _tile_rank_suit(t)
            for dr in (-2,-1,1,2):
                rr = r + dr
                if 1 <= rr <= 9: eff.add(f"{rr}{s}")
        else:
            if c in (1,2): eff.add(t)
    return len(eff)

def rough_shanten_like(hand: List[str]) -> int:
    tiles = [t for t in hand if not is_flower(t)]
    cnt = Counter(tiles)
    meldish = 0
    pairish = 0
    for t in list(cnt):
        while cnt[t] >= 3:
            meldish += 1
            cnt[t] -= 3
    for s in SUITS:
        for r in range(1,8):
            a,b,c = f"{r}{s}", f"{r+1}{s}", f"{r+2}{s}"
            while cnt[a] > 0 and cnt[b] > 0 and cnt[c] > 0:
                meldish += 1
                cnt[a] -= 1; cnt[b] -= 1; cnt[c] -= 1
    for t in list(cnt):
        while cnt[t] >= 2 and pairish < 1:
            pairish += 1
            cnt[t] -= 2
    missing_melds = max(0, 4 - meldish)
    missing_pair  = max(0, 1 - pairish)
    return missing_melds*2 + missing_pair

# --- Seven-pairs distance + Composite metric (weight is learned + context gated) ---

def chiitoi_shanten_like(hand: List[str]) -> float:
    tiles = [t for t in hand if not is_flower(t)]
    cnt = Counter(tiles)
    pair_types = min(sum(1 for c in cnt.values() if c >= 2), 7)
    need_pairs = 7 - pair_types
    overfull = sum(max(0, c - 2) for c in cnt.values())
    return need_pairs + 0.5 * overfull

def composite_shape_metric(hand: List[str], chiitoi_weight: float) -> float:
    std = rough_shanten_like(hand) - 0.02 * rough_ukeire(hand)
    ctt = chiitoi_shanten_like(hand)
    w = max(0.0, min(1.0, chiitoi_weight))
    return (1.0 - w) * std + w * ctt

# ---------------------------- Meld creation helper ----------------------------

def _make_meld(kind: str, tiles: Tuple[str, ...], open: bool, formed_by_claim: bool, from_discard: bool=False) -> Meld:
    m = None
    try:
        m = Meld(kind, tiles, open)
    except TypeError:
        try:
            m = Meld(kind, tiles)
            try: setattr(m, "open", open)
            except Exception: pass
        except Exception:
            m = Meld.__new__(Meld)
            for attr, val in (("type", kind), ("tiles", list(tiles)), ("open", open)):
                try: setattr(m, attr, val)
                except Exception: pass
    for attr, val in (("type", kind), ("tiles", list(tiles)), ("open", open)):
        try:
            if not hasattr(m, attr):
                setattr(m, attr, val)
        except Exception:
            pass
    for attr, val in (("formed_by_claim", formed_by_claim), ("from_discard", from_discard)):
        try: setattr(m, attr, val)
        except Exception: pass
    return m

# ---------------------------- Scoring normalizer ----------------------------

# def _normalize_points(sb, shape_tag: Optional[str] = None, hs: Optional[HandState] = None, rules: Optional[Dict] = None) -> int:
#     """
#     Normalize points from the scoring object:
#       • Prefer explicit 'specials' replacing base.
#       • Add any explicit bonus/flower fields if present.
#       • If no explicit flower bonus found, fall back to (#flowers * per-flower-from-rules, default 1).
#       • House rule: if hand is standard and all 4 declared melds were made from discards,
#         upgrade base from 10 to 20 (unless a special replaced base).
#     """
#     def _get(obj, name, default=None):
#         try:
#             return getattr(obj, name)
#         except Exception:
#             pass
#         try:
#             return obj[name]  # type: ignore[index]
#         except Exception:
#             return default

#     # --- 1) Specials replace base (sum if multiple)
#     specials = None
#     for field in ("specials", "special_hands", "special_components"):
#         v = _get(sb, field)
#         if v is not None:
#             specials = v
#             break

#     def _sum_specials(x):
#         try:
#             if x is None:
#                 return None
#             if isinstance(x, (int, float)):
#                 return int(x)
#             if isinstance(x, list):
#                 if x and hasattr(x[0], "get"):
#                     return sum(int(d.get("points", 0)) for d in x)
#                 return sum(int(v) for v in x)
#         except Exception:
#             return None
#         return None

#     specials_total = _sum_specials(specials)

#     # --- 2) Base (if not replaced)
#     base = None
#     for field in ("base_points", "base", "base_score"):
#         v = _get(sb, field)
#         if v is not None:
#             try:
#                 base = int(v)
#             except Exception:
#                 base = None
#             break

#     # --- 2.5) House rule: if standard and all 4 declared melds are from discards, base := max(base, 20)
#     if specials_total is None and base is not None and hs is not None and shape_tag == "standard":
#         def _kind(m):
#             return _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None)))
#         declared = [m for m in getattr(hs, "melds", []) if _kind(m) in {"chow","pung","kong"}]
#         if len(declared) == 4 and all(getattr(m, "from_discard", False) for m in declared):
#             base = max(base, 20)

#     # --- 3) Bonuses / flowers: collect everything the scorer might expose
#     bonuses = 0
#     flower_bonus_found = False

#     # direct numeric fields
#     for field in ("bonus_points", "extras_points", "flowers_points", "flower_points", "flower_score", "flower_bonus"):
#         v = _get(sb, field)
#         if isinstance(v, (int, float)):
#             bonuses += int(v)
#             if "flower" in field:
#                 flower_bonus_found = True

#     # nested breakdowns (dicts or lists)
#     for field in ("extras", "bonus_breakdown", "details", "components", "extra_components"):
#         ex = _get(sb, field)
#         if isinstance(ex, dict):
#             for k, v in ex.items():
#                 kl = str(k).lower()
#                 if isinstance(v, (int, float)):
#                     if "flower" in kl:
#                         bonuses += int(v); flower_bonus_found = True
#                 elif isinstance(v, dict):
#                     p = v.get("points") if hasattr(v, "get") else None
#                     if isinstance(p, (int, float)):
#                         bonuses += int(p)
#                         if "flower" in kl:
#                             flower_bonus_found = True
#         if isinstance(ex, list):
#             for item in ex:
#                 if hasattr(item, "get"):
#                     name = str(item.get("name", "")).lower()
#                     pts = item.get("points", 0)
#                     if isinstance(pts, (int, float)):
#                         bonuses += int(pts)
#                         if "flower" in name:
#                             flower_bonus_found = True

#     # --- 4) If scorer didn’t give a flower bonus explicitly, fall back to counting
#     if not flower_bonus_found and hs is not None:
#         r = rules or {}
#         per = (
#             r.get("bonuses", {}).get("flower_points_each")
#             or r.get("bonuses", {}).get("flower_point")
#             or r.get("points", {}).get("flower")
#             or 1
#         )
#         fl_cnt = sum(1 for t in getattr(hs, "flowers", []) if is_flower(t))
#         bonuses += int(per) * int(fl_cnt)

#     # --- 5) Specials replace base; bonuses always add on top
#     if specials_total is not None:
#         total = int(specials_total) + int(bonuses)
#     elif shape_tag == "seven_pairs":
#         total = 40 + int(bonuses)
#     elif base is not None:
#         total = int(base) + int(bonuses)
#     else:
#         v = _get(sb, "total_points", 0)
#         try:
#             total = int(v)
#         except Exception:
#             total = int(bonuses)  # at least include flowers
#     return total

def _normalize_points_verbose(sb, shape_tag: Optional[str] = None,
                              hs: Optional[HandState] = None,
                              rules: Optional[Dict] = None) -> int:
    """
    Replacement scoring normalizer with full special-hand detection.
    Keeps base scorer output but logs all detected specials and bonuses.
    """

    def _get(obj, name, default=None):
        try:
            return getattr(obj, name)
        except Exception:
            pass
        try:
            return obj[name]
        except Exception:
            return default

    # Base and specials from scorer
    base = _get(sb, "base_points", 10)
    bonus = 0
    if hasattr(sb, "bonus_points"): bonus += int(getattr(sb, "bonus_points", 0))
    if hasattr(sb, "flowers_points"): bonus += int(getattr(sb, "flowers_points", 0))
    flowers = getattr(hs, "flowers", []) if hs else []
    melds = getattr(hs, "melds", []) if hs else []
    concealed = getattr(hs, "concealed", []) if hs else []
    all_tiles = concealed + [t for m in melds for t in getattr(m, "tiles", [])]
    suits = {t[1] for t in all_tiles if len(t) == 2 and t[0].isdigit()}
    has_honor = any(t in HONORS for t in all_tiles)

    specials = []
    base_special = 0

    # --- Pung-Pung Hu
    meld_types = [getattr(m, "kind", getattr(m, "type", "")) for m in melds]
    if meld_types and all(k.lower() in ("pung", "kong") for k in meld_types):
        specials.append("peng_peng_hu")
        base_special += 20

    # --- Eating hand (all 4 declared melds claimed from discards)
    if len(melds) == 4 and all(getattr(m, "from_discard", False) for m in melds):
        specials.append("eating_hand")
        base_special += 20

    # --- One-color / mixed one-color
    if len(suits) == 1:
        if has_honor:
            specials.append("mixed_one_color")
            base_special += 20
        else:
            specials.append("one_color")
            base_special += 40

    # --- All-apart (9-suit + 5 distinct honors)
    tiles14 = concealed + [getattr(hs, "winning_tile", "")]
    if _is_all_apart(tiles14):
        specials.append("all_apart")
        base_special += 20

    # --- Seven-pairs
    cnt = Counter([t for t in all_tiles if not is_flower(t)])
    pairs = sum(1 for v in cnt.values() if v == 2) + 2 * sum(1 for v in cnt.values() if v == 4)
    if pairs == 7:
        specials.append("seven_pairs")
        base_special += 40

    # --- Flower / honor / dragon bonuses
    fl_bonus = sum(1 for f in flowers if is_flower(f))
    bonus += fl_bonus
    for t in all_tiles:
        if t in ("E", "S", "W", "N"): bonus += 1
        if t in ("C", "F", "B"): bonus += 2  # dragons slightly higher

    total_special = base_special if specials else base
    total = total_special + bonus

    # if specials:
    #     print(f"[scoring] specials={specials} base={base_special} + bonus={bonus} → total={total}")
    return int(total)


# ---------------------------- Adaptive tuner ----------------------------

class AdaptiveTuner:
    """
    Meta tuner for all knobs:
      - Claim biases (pung/chow/open/closed-kong)
      - Keep/risk weights (keep_shape_weight, danger_weight)
      - Acceptance margins (pung/chow + decay w/ #melds)
      - Seven-pairs weight (bounded, with strong downward pressure)
    Signals (EMA):
      - open/closed win split, tsumo share, feed rate
      - win shape mix & avg points (seven_pairs vs standard)
      - action attributions: uses, used-when-winning, feed-when-used, win points
    """
    def __init__(self,
                 init: Dict[str, float] = None,
                 step: float = 0.06,
                 ema: float = 0.15):
        self.bias = {
            # ↑ Aggressive defaults to encourage discard melds/openings
            "pung_bias": 0.90,
            "chow_bias": 0.85,
            "open_kong_bias": 0.65,
            "closed_kong_bias": 0.35,
            "danger_weight": 0.02,
            "keep_shape_weight": 0.80,
            "meld_target_bonus": 1.0,
            # Seven-pairs mixture (global base; further gated in-hand):
            "chiitoi_weight": 0.03,
            # Learned margins (higher => more willing to claim even if slightly worse)
            "pung_margin_base": 1.15,
            "chow_margin_base": 0.75,
            "margin_decay_per_meld": 0.15,
        }
        if init:
            self.bias.update(init)
        self.ema = ema
        self.step = step

        # Per-seat EMAs
        self.stats = [{
            "open_used": 0.0, "open_wins": 0.0, "closed_wins": 0.0,
            "open_events": 0.0, "ron_losses": 0.0, "ron_points_lost": 0.0,
            "tsumo_wins": 0.0, "hands": 0.0
        } for _ in range(4)]

        # Global win-shape EMAs
        self.shape = {"wins": 0.0, "sp_wins": 0.0, "std_wins": 0.0, "sp_pts": 0.0, "std_pts": 0.0}

        # Action-level attributions (EMA)
        self.act = {
            "chow": {"uses":0.0,"win_uses":0.0,"win_pts":0.0,"feed_uses":0.0},
            "pung": {"uses":0.0,"win_uses":0.0,"win_pts":0.0,"feed_uses":0.0},
            "kong_open": {"uses":0.0,"win_uses":0.0,"win_pts":0.0,"feed_uses":0.0},
            "kong_closed": {"uses":0.0,"win_uses":0.0,"win_pts":0.0,"feed_uses":0.0},
        }

    def get(self, k: str) -> float:
        return float(self.bias[k])

    # ---- helpers ----
    def _ema_add(self, x_old: float, inc: float) -> float:
        return (1 - self.ema) * x_old + self.ema * inc

    def _clamp(self):
        # keep standard biases in [0.05, 0.98]
        for k in ("pung_bias","chow_bias","open_kong_bias","closed_kong_bias"):
            self.bias[k] = max(0.05, min(0.98, self.bias[k]))
        # strongly bound chiitoi
        self.bias["chiitoi_weight"] = max(0.00, min(0.25, self.bias["chiitoi_weight"]))
        # reasonable margins
        self.bias["pung_margin_base"] = max(0.30, min(1.80, self.bias["pung_margin_base"]))
        self.bias["chow_margin_base"] = max(0.20, min(1.40, self.bias["chow_margin_base"]))
        self.bias["margin_decay_per_meld"] = max(0.05, min(0.40, self.bias["margin_decay_per_meld"]))
        # weights
        self.bias["danger_weight"] = max(0.00, min(0.12, self.bias["danger_weight"]))
        self.bias["keep_shape_weight"] = max(0.60, min(1.20, self.bias["keep_shape_weight"]))
        self.bias["meld_target_bonus"] = max(0.50, min(1.50, self.bias["meld_target_bonus"]))

    def record_episode(self, env_terminal: Dict[str, Any], env_claim_log: List[Dict[str, Any]]):
        # Collect which actions each seat used this hand
        used_actions = {s: set() for s in range(4)}
        for ev in env_claim_log:
            t = ev.get("type","")
            who = ev.get("who")
            if isinstance(who, int):
                if t in ("pung","chow"): used_actions[who].add(t)
                elif t.startswith("kong(open"):
                    used_actions[who].add("kong_open")
                elif t.startswith("kong(closed"):
                    used_actions[who].add("kong_closed")

        # Winner seats + points
        winners = []
        if env_terminal.get("source") == "discard" and env_terminal.get("winners"):
            for w in env_terminal["winners"]:
                winners.append((int(w.get("seat", -1)), float(w.get("points", 0))))
        elif isinstance(env_terminal.get("winner"), int):
            winners.append((int(env_terminal.get("winner")), float(env_terminal.get("points",0))))

        # Per-seat EMAs + action EMAs
        ron_loser = env_terminal.get("ron_loser")
        for s in range(4):
            st = self.stats[s]
            st["hands"] = self._ema_add(st["hands"], 1.0)
            st["open_events"] = self._ema_add(st["open_events"], 1.0 if used_actions[s] else 0.0)
            won = any(ws == s for (ws, _) in winners)
            st["open_wins"] = self._ema_add(st["open_wins"], 1.0 if (won and used_actions[s]) else 0.0)
            st["closed_wins"] = self._ema_add(st["closed_wins"], 1.0 if (won and not used_actions[s]) else 0.0)
            st["tsumo_wins"] = self._ema_add(st["tsumo_wins"], 1.0 if (env_terminal.get("source")=="self_draw" and won) else 0.0)
            fed = (isinstance(ron_loser, int) and ron_loser == s)
            st["ron_losses"] = self._ema_add(st["ron_losses"], 1.0 if fed else 0.0)
            st["ron_points_lost"] = self._ema_add(st["ron_points_lost"], float(env_terminal.get("points",0)) if fed else 0.0)

        # Global shape EMAs + action attributions
        def _acc_win(item: Dict[str, Any]):
            pts = float(item.get("points", 0))
            snap = item.get("winning_hand", {}) or {}
            shape_tag = snap.get("shape_tag", "standard")
            self.shape["wins"] = self._ema_add(self.shape["wins"], 1.0)
            if shape_tag == "seven_pairs":
                self.shape["sp_wins"] = self._ema_add(self.shape["sp_wins"], 1.0)
                self.shape["sp_pts"]  = self._ema_add(self.shape["sp_pts"], pts)
            else:
                self.shape["std_wins"] = self._ema_add(self.shape["std_wins"], 1.0)
                self.shape["std_pts"]  = self._ema_add(self.shape["std_pts"], pts)

        if env_terminal.get("source") == "discard" and env_terminal.get("winners"):
            for w in env_terminal["winners"]:
                _acc_win(w)
        elif isinstance(env_terminal.get("winner"), int):
            _acc_win(env_terminal)

        # Action-EMAs: uses, wins, feeds
        winners_set = {ws for (ws, _) in winners}
        points_by_winner = {ws: pts for (ws, pts) in winners}
        for s in range(4):
            for a in used_actions[s]:
                self.act[a]["uses"] = self._ema_add(self.act[a]["uses"], 1.0)
                if s in winners_set:
                    self.act[a]["win_uses"] = self._ema_add(self.act[a]["win_uses"], 1.0)
                    self.act[a]["win_pts"]  = self._ema_add(self.act[a]["win_pts"], points_by_winner.get(s,0.0))
                if isinstance(ron_loser, int) and ron_loser == s:
                    self.act[a]["feed_uses"] = self._ema_add(self.act[a]["feed_uses"], 1.0)

        self._nudge()

    def _nudge(self):
        lr = self.step

        # Aggregates
        hands = sum(st["hands"] for st in self.stats) + 1e-9
        open_rate = (sum(st["open_events"] for st in self.stats) / hands)
        open_win = (sum(st["open_wins"] for st in self.stats) / (sum(st["open_events"] for st in self.stats) + 1e-9))
        closed_win = (sum(st["closed_wins"] for st in self.stats) / (hands - sum(st["open_events"] for st in self.stats) + 1e-9))
        tsumo_share = (sum(st["tsumo_wins"] for st in self.stats) / hands)
        feed_rate = (sum(st["ron_losses"] for st in self.stats) / hands)

        wins_ema = max(self.shape["wins"], 1e-9)
        sp_share = self.shape["sp_wins"] / wins_ema
        avg_sp  = self.shape["sp_pts"]  / max(self.shape["sp_wins"], 1e-9)
        avg_std = self.shape["std_pts"] / max(self.shape["std_wins"], 1e-9)
        avg_all = (self.shape["sp_pts"] + self.shape["std_pts"]) / wins_ema

        # Targets — nudge toward more openings, but keep feeds in check
        target_open_rate   = 0.45
        target_feed_rate   = 0.06
        target_tsumo_share = 0.42
        target_sp_share    = 0.10

        err_open  = open_rate - target_open_rate
        err_feed  = feed_rate - target_feed_rate
        err_tsumo = tsumo_share - target_tsumo_share

        if open_win > closed_win + 0.02 and err_feed <= 0:
            self.bias["pung_bias"]      += lr * 0.7
            self.bias["chow_bias"]      += lr * 0.7
            self.bias["open_kong_bias"] += lr * 0.3
            self.bias["pung_margin_base"] += lr * 0.25
            self.bias["chow_margin_base"] += lr * 0.25
        else:
            damp = lr * (0.6 + max(0.0, err_open) + max(0.0, err_feed))
            self.bias["pung_bias"]      -= damp
            self.bias["chow_bias"]      -= damp
            self.bias["open_kong_bias"] -= lr * (0.3 + max(0.0, err_feed))

        if err_tsumo < 0:
            self.bias["closed_kong_bias"] += lr * 0.30
            self.bias["keep_shape_weight"] += lr * 0.15
        if err_feed > 0:
            self.bias["closed_kong_bias"] -= lr * 0.20
            self.bias["danger_weight"] += lr * 0.50
            self.bias["keep_shape_weight"] += lr * 0.10

        # Seven-pairs mix controller
        sp_overused = (sp_share > target_sp_share + 0.04)
        sp_weak     = (avg_sp + 1 < avg_std)
        sp_under    = (sp_share < target_sp_share - 0.05) and (avg_sp >= avg_std - 1)

        if sp_overused or sp_weak:
            self.bias["chiitoi_weight"]  -= lr * (0.50 if sp_overused and sp_weak else 0.30)
            self.bias["pung_margin_base"] += lr * 0.30
            self.bias["chow_margin_base"] += lr * 0.30
            self.bias["pung_bias"]        += lr * 0.20
            self.bias["chow_bias"]        += lr * 0.20
        elif sp_under:
            self.bias["chiitoi_weight"]  += lr * 0.20
            self.bias["pung_margin_base"] -= lr * 0.15
            self.bias["chow_margin_base"] -= lr * 0.15

        # Action-level attributions → fine-grained nudges
        for name, key in (("pung","pung_bias"), ("chow","chow_bias"), ("kong_open","open_kong_bias"), ("kong_closed","closed_kong_bias")):
            a = self.act[name]
            uses = a["uses"] + 1e-9
            win_rate_adj = (a["win_uses"] / uses) - 0.33
            feed_adj     = -(a["feed_uses"] / uses)
            pts_adv = (a["win_pts"] / max(a["win_uses"],1e-9)) - avg_all
            self.bias[key] += lr * (0.6*win_rate_adj + 0.4*feed_adj + 0.1*pts_adv)

        self._clamp()

# Global toggles
ALWAYS_CLAIM_DEBUG = False
FORCE_CLAIMS       = False

# ---------------------------- Environment ----------------------------

class Env:
    """
    East starts. Turn loop:
      1) Draw from front (or from back only as flower/kong replacement).
      2) Auto-play flowers (draw from back until no flowers).
      3) Optionally declare CLOSED KONG(s) (draw replacement each time).
      4) Optionally upgrade exposed pungs to OPEN KONG (add-kan) with replacement.
      5) Check TSUMO on latest draw via strict shape + score_hand.
      6) Discard a non-flower.
      7) Claims on discard in order: Ron > Pung > Chow.  (Pung over Chow)
         * Multi-Ron: all claimants win on same discard; discarder pays each.
    """
    def __init__(self, rules: Dict, seed: Optional[int]=None):
        self.rules = rules
        self.rng = random.Random(seed)
        self.wall = make_wall(include_flowers=bool(rules.get("tileset",{}).get("use_flowers", True)), seed=seed)
        self.players = [PlayerView([],[],[]) for _ in range(4)]
        self.turn = 0
        self.discards: List[List[str]] = [[] for _ in range(4)]
        self.discard_history: List[Tuple[int,str]] = []
        self.last_discard: Optional[Tuple[int,str]] = None
        self.side_delta = [0,0,0,0]
        self.side_events: List[Dict[str,Any]] = []
        self.terminal: Optional[Dict[str,Any]] = None
        self.stats = {"chow":0, "pung":0, "kong_open":0, "kong_closed":0}
        self.claim_log: List[Dict[str, Any]] = []

        # Opening snapshots
        for _ in range(13):
            for p in self.players: self._draw_into(p, back=False)
        self._draw_into(self.players[0], back=False)
        self.opening_before_flowers: List[List[str]] = [list(p.concealed) for p in self.players]
        for i in range(4): self._settle_flowers(i)
        self.opening_after_flowers: List[List[str]] = [list(p.concealed) for p in self.players]
        self.opening_flowers: List[List[str]] = [list(p.flowers) for p in self.players]
        self._forced: Dict[int, Dict[str, Any]] = {}  # seat -> {"kind": Optional[str], "idx": int}

    def _draw_into(self, p: PlayerView, back: bool):
        if not self.wall: return
        t = self.wall.draw_back() if back else self.wall.draw_front()
        if t is None: return
        p.concealed.append(t)

    def _settle_flowers(self, seat: int):
        p = self.players[seat]
        changed = True
        while changed:
            changed = False
            for t in list(p.concealed):
                if is_flower(t):
                    p.concealed.remove(t)
                    p.flowers.append(t)
                    self._draw_into(p, back=True)
                    changed = True

    def legal_discards(self, seat: int) -> List[str]:
        return [t for t in self.players[seat].concealed if not is_flower(t)]

    def hand_state_for(self, seat: int, winning_tile: str, source: str) -> HandState:
        p = self.players[seat]
        concealed = p.concealed[:]
        if source == "self_draw" and winning_tile in concealed:
            concealed.remove(winning_tile)
        return HandState(
            flowers=p.flowers[:],
            concealed=concealed,
            melds=p.melds[:],
            winning_tile=winning_tile,
            win_source="self_draw" if source=="self_draw" else "discard",
            back_wall_bonus=False,
            used_any_claim_for_sets=p.used_discard_claim
        )

    # ---- side-payment: 4 same discards in order
    def _maybe_apply_four_same_discard_penalty(self):
        cfg = self.rules.get("special_events", {}).get("four_same_consecutive_discard", {})
        if not cfg.get("enabled", True): return
        amt = int(cfg.get("amount_per_opponent", 1))
        if len(self.discard_history) < 4: return
        s0,t0 = self.discard_history[-4]
        s1,t1 = self.discard_history[-3]
        s2,t2 = self.discard_history[-2]
        s3,t3 = self.discard_history[-1]
        if t0==t1==t2==t3 and s1==(s0+1)%4 and s2==(s1+1)%4 and s3==(s2+1)%4:
            self.side_delta[s0] -= 3*amt
            self.side_delta[s1] += amt
            self.side_delta[s2] += amt
            self.side_delta[s3] += amt
            self.side_events.append({"type":"four_same_consecutive_discard","tile":t0,"first":s0,"amount_per_opponent":amt})

    # ---- win acceptor with strict shape + menqing 10/11 gate (discard-claim only)
    def _try_accept_win(self, seat: int, source: str, winning_tile: str) -> Optional[Tuple[int, str]]:
        p = self.players[seat]
        concealed_for_shape = p.concealed[:]
        if source == "self_draw" and winning_tile in concealed_for_shape:
            concealed_for_shape.remove(winning_tile)
        ok, shape_tag = _legal_win_shape(concealed_for_shape, p.melds, winning_tile, source)
        if not ok: return None
        hs = self.hand_state_for(seat, winning_tile, source)
        try:
            sb = score_hand(hs, self.rules)
        except Exception:
            return None
        pts_norm = _normalize_points_verbose(sb, shape_tag, hs=hs, rules=self.rules)
        # Only block RON (discard win) at 10/11 if the player has opened.
        if (source == "discard") and p.used_discard_claim and pts_norm in (10, 11):
            return None
        self.terminal = {
            "winner": seat,
            "source": source,
            "points": pts_norm,
            "winning_hand": _snapshot_winning_hand(self, seat, hs, pts_norm, source, extra={"shape_tag": shape_tag}),
            "side_delta": self.side_delta[:],
            "side_events": self.side_events[:],
        }
        if source == "discard":
            self.terminal["ron_loser"] = self.last_discard[0] if self.last_discard else None
        return pts_norm, shape_tag

    # ---- declaring/claim helpers

    def _form_pung(self, seat: int, tile: str):
        p = self.players[seat]
        removed = 0
        for t in list(p.concealed):
            if t == tile and removed < 2:
                p.concealed.remove(t); removed += 1
        m = _make_meld("pung", (tile, tile, tile), open=True, formed_by_claim=True, from_discard=True)
        p.melds.append(m)
        p.used_discard_claim = True
        self.stats["pung"] += 1

    def _form_chow(self, seat: int, a: str, m: str, b: str):
        p = self.players[seat]
        p.concealed.remove(a); p.concealed.remove(b)
        tiles = sorted([a, m, b], key=lambda x:(x[1], int(x[0]) if _is_suit_tile(x) else 0))
        meld = _make_meld("chow", tuple(tiles), open=True, formed_by_claim=True, from_discard=True)
        p.melds.append(meld)
        p.used_discard_claim = True
        self.stats["chow"] += 1

    def _form_closed_kong(self, seat: int, tile: str):
        p = self.players[seat]
        removed = 0
        for t in list(p.concealed):
            if t == tile and removed < 4:
                p.concealed.remove(t); removed += 1
        m = _make_meld("kong", (tile, tile, tile, tile), open=False, formed_by_claim=False, from_discard=False)
        p.melds.append(m)
        self.stats["kong_closed"] += 1

    def _after_kong_draw_and_maybe_tsumo(self, seat: int, policies) -> bool:
        self._draw_into(self.players[seat], back=True)
        self._settle_flowers(seat)
        drawn = self.players[seat].concealed[-1] if self.players[seat].concealed else None
        res = self._try_accept_win(seat, "self_draw", drawn) if drawn else None
        return res is not None

    # ---- add-kan (upgrade exposed pung to open kong)

    def _maybe_added_kongs(self, seat: int, policies) -> bool:
        p = self.players[seat]
        changed = False
        while True:
            upgrade = None
            for idx, m in enumerate(p.melds):
                kind = _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None)))
                opened = bool(getattr(m, "open", True))
                tiles = list(getattr(m, "tiles", []))
                if kind == "pung" and opened and len(tiles) == 3 and tiles[0] == tiles[1] == tiles[2]:
                    tile = tiles[0]
                    if p.concealed.count(tile) >= 1:
                        forced = self._pop_forced(seat, "kong") or self._pop_forced(seat, None)
                        ok = False
                        if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
                            ok = True
                        elif forced is not None:
                            # For add-kan we treat any index (0/1) as "yes"; if you prefer multi-option indexing, enumerate upgradable pungs first.
                            ok = bool(forced["idx"])
                        else:
                            decider = getattr(policies[seat], "decide_add_kong", None)
                            ok = (decider is None) or decider(self, seat, tile)

                        if ok:
                            upgrade = (idx, tile); break
            if upgrade is None:
                return changed
            idx, tile = upgrade
            p.concealed.remove(tile)
            p.melds[idx] = _make_meld("kong", (tile, tile, tile, tile), open=True, formed_by_claim=True, from_discard=False)
            self.stats["kong_open"] += 1
            self.claim_log.append({"who": seat, "type": "kong(open,added)", "on": tile})
            if self._after_kong_draw_and_maybe_tsumo(seat, policies):
                return True
            changed = True

    def _maybe_upgrade_recent_pung_to_kong(self, seat: int, tile: str, policies) -> bool:
        p = self.players[seat]
        for idx, m in enumerate(p.melds):
            kind = _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None)))
            tiles = list(getattr(m, "tiles", []))
            if kind == "pung" and len(tiles) == 3 and tiles[0] == tiles[1] == tiles[2] == tile and p.concealed.count(tile) >= 1:
                decider = getattr(policies[seat], "decide_add_kong", None)
                if decider is None or decider(self, seat, tile):
                    p.concealed.remove(tile)
                    p.melds[idx] = _make_meld("kong", (tile, tile, tile, tile), open=True, formed_by_claim=True, from_discard=False)
                    self.stats["kong_open"] += 1
                    self.claim_log.append({"who": seat, "type": "kong(open,added)", "on": tile})
                    if self._after_kong_draw_and_maybe_tsumo(seat, policies):
                        return True
        return False

    # ---- claims on discard

    def _ron_window(self, discarder: int, tile: str, policies) -> bool:
        winners: List[Tuple[int,int,str]] = []
        for k in (1,2,3):
            s = (discarder + k) % 4
            res = self._try_accept_win(s, "discard", tile)
            if res is None: continue
            pts, shape_tag = res
            forced_bin = self._pop_forced(s, "binary") or self._pop_forced(s, None)
            if forced_bin is not None:
                take_ron = bool(forced_bin["idx"])  # 1=yes, 0=no
            else:
                take_ron = True
                if hasattr(policies[s], "decide_ron"):
                    take_ron = policies[s].decide_ron(self, tile, pts, discarder)
            if not take_ron:
                self.terminal = None
                continue

            winners.append((s, pts, shape_tag))
            self.terminal = None  # aggregate at end

        if not winners: return False

        winners_sorted = sorted(winners, key=lambda x: x[0])
        self.terminal = {
            "source": "discard",
            "ron_loser": discarder,
            "side_delta": self.side_delta[:],
            "side_events": self.side_events[:],
        }

        if len(winners_sorted) == 1:
            w, pts, shape_tag = winners_sorted[0]
            hs = self.hand_state_for(w, tile, "discard")
            self.terminal.update({
                "winner": w,
                "points": pts,
                "winning_hand": _snapshot_winning_hand(self, w, hs, pts, "discard", extra={"shape_tag": shape_tag}),
            })
        else:
            self.terminal.update({
                "winner": None,
                "winners": [
                    {
                        "seat": w,
                        "points": pts,
                        "winning_hand": _snapshot_winning_hand(
                            self, w, self.hand_state_for(w, tile, "discard"), pts, "discard",
                            extra={"shape_tag": shape_tag}
                        ),
                    }
                    for (w, pts, shape_tag) in winners_sorted
                ],
            })
        return True

    def _pung_claims(self, discarder: int, tile: str, policies) -> Optional[int]:
        for k in (1,2,3):
            s = (discarder + k) % 4
            if _meld_count(self.players[s]) >= 4: continue
            cnt = Counter(self.players[s].concealed)
            if cnt[tile] >= 2:
                take = False
                if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
                    take = True
                else:
                    forced_bin = self._pop_forced(s, "binary") or self._pop_forced(s, None)
                    if forced_bin is not None:
                        take = bool(forced_bin["idx"])  # 1=yes, 0=no
                    elif hasattr(policies[s], "decide_pung"):
                        take = policies[s].decide_pung(self, s, tile)
                if not take: 
                    continue
                self._form_pung(s, tile)
                if self._maybe_upgrade_recent_pung_to_kong(s, tile, policies): return s
                self.players[s].needs_discard = True
                self.claim_log.append({"who": s, "type": "pung", "on": tile, "from": discarder})
                self.turn = s
                return s
        return None

    def _chow_claim(self, discarder: int, tile: str, policies) -> Optional[int]:
        s = (discarder + 1) % 4
        if _meld_count(self.players[s]) >= 4: return None
        if not _is_suit_tile(tile): return None
        r, suit = _tile_rank_suit(tile)
        sets = []
        for a,b in [(r-2,r-1),(r-1,r+1),(r+1,r+2)]:
            if 1 <= a <= 9 and 1 <= b <= 9:
                A,B = f"{a}{suit}", f"{b}{suit}"
                if A in self.players[s].concealed and B in self.players[s].concealed:
                    sets.append((A,B))
        if not sets: return None
        pick = None
        forced = self._pop_forced(s, "chow") or self._pop_forced(s, None)
        if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
            pick = sets[0]
        elif forced is not None:
            idx = int(forced["idx"])
            if 0 <= idx < len(sets):
                pick = sets[idx]
        else:
            if hasattr(policies[s], "choose_chow"):
                pick = policies[s].choose_chow(self, s, tile, sets)
        if pick is None:
            return None
        self._form_chow(s, pick[0], tile, pick[1])
        self.players[s].needs_discard = True
        self.claim_log.append({"who": s, "type": "chow", "on": tile, "with": list(pick), "from": discarder})
        self.turn = s
        return s

    # ---- CLOSED KONGS (on your turn, before tsumo check)

    def _maybe_closed_kongs(self, seat: int, policies) -> bool:
        p = self.players[seat]
        while True:
            cnt = Counter([t for t in p.concealed if not is_flower(t)])
            candidates = [t for t,c in cnt.items() if c >= 4]
            if not candidates or _meld_count(p) >= 4: return False
            tile = None
            forced = self._pop_forced(seat, "kong") or self._pop_forced(seat, None)
            if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
                tile = candidates[0]
            elif forced is not None:
                idx = int(forced["idx"])
                if 0 <= idx < len(candidates):
                    tile = candidates[idx]
            else:
                if hasattr(policies[seat], "decide_closed_kong"):
                    tile = policies[seat].decide_closed_kong(self, seat, candidates)
            if tile is None:
                return False
            self._form_closed_kong(seat, tile)
            self.claim_log.append({"who": seat, "type": "kong(closed)", "on": tile})
            if self._after_kong_draw_and_maybe_tsumo(seat, policies): return True

    # ---- main step

    def step_turn(self, policies) -> None:
        if self.terminal: return
        seat = self.turn

        # If we just claimed, we must discard immediately.
        if self.players[seat].needs_discard:
            opts = self.legal_discards(seat)
            tile = None

            # Forced discard (maps 0..33 class → actual tile in hand)
            f = self._pop_forced(seat, "discard") or self._pop_forced(seat, None)
            if f is not None:
                cls = int(f["idx"])
                for cand in opts:
                    c = tile_to_class(cand)  # 34-class mapping; flowers are not in opts
                    if c is not None and c == cls:
                        tile = cand
                        break

            # Fallback to policy if not forced or no matching tile
            if tile is None:
                tile = policies[seat].pick_discard(self) if opts else (
                    self.players[seat].concealed[0] if self.players[seat].concealed else None
                )

            if tile is None:
                return

            self.players[seat].concealed.remove(tile)
            self.discards[seat].append(tile)
            self.discard_history.append((seat, tile))
            self.last_discard = (seat, tile)
            self.players[seat].needs_discard = False

            self._maybe_apply_four_same_discard_penalty()

            if self._ron_window(seat, tile, policies): return
            if self._pung_claims(seat, tile, policies) is not None: return
            if self._chow_claim(seat, tile, policies) is not None: return

            self.turn = (self.turn + 1) % 4
            self._draw_into(self.players[self.turn], back=False)
            self._settle_flowers(self.turn)
            return

        # Normal start-of-turn flow
        if self.players[seat].concealed:
            if self._maybe_closed_kongs(seat, policies): return
            if self._maybe_added_kongs(seat, policies): return
            drawn = self.players[seat].concealed[-1]
            if self._try_accept_win(seat, "self_draw", drawn) is not None: return

        # Discard
        # Discard
        opts = self.legal_discards(seat)
        tile = None

        # Forced discard (0..33 → tile in hand)
        f = self._pop_forced(seat, "discard") or self._pop_forced(seat, None)
        if f is not None:
            cls = int(f["idx"])
            for cand in opts:
                c = tile_to_class(cand)
                if c is not None and c == cls:
                    tile = cand
                    break

        # Fallback to policy
        if tile is None:
            tile = policies[seat].pick_discard(self) if opts else (
                self.players[seat].concealed[0] if self.players[seat].concealed else None
            )

        # Safety guard (keep this exactly as-is)
        if tile not in self.players[seat].concealed and opts:
            tile = random.choice(opts)
        if tile is None:
            return

        self.players[seat].concealed.remove(tile)
        self.discards[seat].append(tile)
        self.discard_history.append((seat, tile))
        self.last_discard = (seat, tile)

        self._maybe_apply_four_same_discard_penalty()

        if self._ron_window(seat, tile, policies): return
        if self._pung_claims(seat, tile, policies) is not None: return
        if self._chow_claim(seat, tile, policies) is not None: return

        self.turn = (self.turn + 1) % 4
        self._draw_into(self.players[self.turn], back=False)
        self._settle_flowers(self.turn)
    
    # ---------- Rollout helpers ----------
    # in sim_and_train.Env
    # in sim_and_train.Env
    def snapshot(self):
        return {
            "rng_state": self.rng.getstate(),
            "wall_pairs": [pair[:] for pair in self.wall.pairs],
            "players": [
                {
                    "concealed": p.concealed[:],
                    "melds": list(p.melds),      # shallow or deep depending on Meld
                    "flowers": p.flowers[:],
                    "used_discard_claim": p.used_discard_claim,
                    "needs_discard": p.needs_discard,
                }
                for p in self.players
            ],
            "turn": self.turn,
            "discards": [d[:] for d in self.discards],
            "discard_history": self.discard_history[:],
            "last_discard": self.last_discard,
            "side_delta": self.side_delta[:],
            "side_events": list(self.side_events),
            "terminal": None,
            "stats": dict(self.stats),
            "claim_log": list(self.claim_log),
            "_forced": dict(self._forced),
        }

    def restore(self, snap):
        self.rng.setstate(snap["rng_state"])
        self.wall.pairs = [pair[:] for pair in snap["wall_pairs"]]
        for p, s in zip(self.players, snap["players"]):
            p.concealed = s["concealed"][:]
            p.melds = list(s["melds"])
            p.flowers = s["flowers"][:]
            p.used_discard_claim = s["used_discard_claim"]
            p.needs_discard = s["needs_discard"]
        self.turn = snap["turn"]
        self.discards = [d[:] for d in snap["discards"]]
        self.discard_history = snap["discard_history"][:]
        self.last_discard = snap["last_discard"]
        self.side_delta = snap["side_delta"][:]
        self.side_events = list(snap["side_events"])
        self.terminal = snap["terminal"]
        self.stats = dict(snap["stats"])
        self.claim_log = list(snap["claim_log"])
        self._forced = dict(snap["_forced"])


    
    def fork(self) -> "Env":
        """Deep-copy the environment (RNG + wall + hands) for a rollout."""
        import copy
        return copy.deepcopy(self)

    def force_action(self, *args):
        """
        Queue a one-shot forced action:
        - force_action(action_idx) → assumes current turn seat; kind unspecified (first decision)
        - force_action(seat, action_idx) → kind unspecified
        - force_action(seat, kind, action_idx) → explicit kind in {"discard","binary","chow","kong"}
        """
        if len(args) == 1:
            seat = self.turn
            kind = None
            idx = int(args[0])
        elif len(args) == 2:
            seat, idx = int(args[0]), int(args[1])
            kind = None
        elif len(args) == 3:
            seat, kind, idx = int(args[0]), str(args[1]).lower(), int(args[2])
            if kind not in (None, "discard", "binary", "chow", "kong"):
                raise ValueError(f"Unknown forced kind: {kind}")
        else:
            raise ValueError("force_action expects 1, 2, or 3 args")

        if seat < 0 or seat > 3:
            raise ValueError("seat must be 0..3")
        self._forced[seat] = {"kind": kind, "idx": idx}

    def _peek_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
        f = self._forced.get(seat)
        if not f:
            return None
        return f if (f["kind"] is None or kind is None or f["kind"] == kind) else None

    def _pop_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
        f = self._peek_forced(seat, kind)
        if f:
            self._forced.pop(seat, None)
            return f
        return None


# ---------------------------- Policies ----------------------------

def _meld_count(p) -> int:
    return sum(1 for m in p.melds if _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None))) in {"chow","pung","kong"})

def _tile_is_isolate(hand_cnt: Counter, t: str) -> bool:
    if not _is_suit_tile(t):  # honors: isolate if singleton
        return hand_cnt[t] == 1
    r,s = _tile_rank_suit(t)
    neighbors = []
    for dr in (-2,-1,1,2):
        rr = r + dr
        if 1 <= rr <= 9:
            neighbors.append(f"{rr}{s}")
    return hand_cnt[t] == 1 and not any(hand_cnt[n] > 0 for n in neighbors)

def _pairs_in(hand: List[str]) -> int:
    cnt = Counter([t for t in hand if not is_flower(t)])
    return min(sum(1 for c in cnt.values() if c >= 2), 7)

class BasePolicy:
    def __init__(self, seat: int, rules: Dict, tuner: Optional[AdaptiveTuner]):
        self.seat, self.rules, self.tuner = seat, rules, tuner or AdaptiveTuner()
    def _b(self, k: str) -> float: return self.tuner.get(k)
    def _ol(self) -> float:
        try:
            return float(self.rules.get("policy", {}).get("open_lean", 0.0))
        except Exception:
            return 0.0

    def _effective_w(self, env, hand: List[str]) -> float:
        """Context-gate the seven-pairs weight to avoid overemphasis in-play."""
        base = self._b("chiitoi_weight")
        declared = _meld_count(env.players[self.seat])
        pairs = _pairs_in(hand)
        stage = max(0.3, min(1.0, len(env.discard_history)/42.0))
        declared_factor = 0.6 if declared > 0 else 1.0
        pair_factor = min(1.0, 0.12 * pairs + 0.10)
        return base * stage * declared_factor * pair_factor

class RandomPolicy(BasePolicy):
    def pick_discard(self, env: Env) -> str:
        opts = env.legal_discards(self.seat)
        if not opts: return env.players[self.seat].concealed[0]
        p = env.players[self.seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        cnt = Counter(hand)
        isolates = [t for t in opts if _tile_is_isolate(cnt, t)]
        return random.choice(isolates or opts)
    def decide_ron(self, env, tile, points, loser): return True
    def decide_open_kong(self, env, seat, tile): return ALWAYS_CLAIM_DEBUG or (random.random() < self._b("open_kong_bias"))
    def decide_add_kong(self, env, seat, tile): return ALWAYS_CLAIM_DEBUG or (random.random() < self._b("open_kong_bias"))
    def decide_pung(self, env, seat, tile): return ALWAYS_CLAIM_DEBUG or (random.random() < self._b("pung_bias"))
    def choose_chow(self, env, seat, tile, chow_sets):
        if not chow_sets: return None
        return chow_sets[0] if (ALWAYS_CLAIM_DEBUG or random.random() < self._b("chow_bias")) else None
    def decide_closed_kong(self, env, seat, candidates):
        return (candidates and candidates[0]) if (ALWAYS_CLAIM_DEBUG or (candidates and random.random()<self._b("closed_kong_bias"))) else None

class WinProbPolicy(BasePolicy):
    """
    Meld-friendly, guided by tuner. Uses *learned + context-gated* chiitoi weight,
    adaptive acceptance margins, and a soft-margin override to claim more often.
    """
    def _ev_danger(self, env: Env, tile: str) -> float:
        loss = 0.0
        for k in (1,2,3):
            opp = (self.seat + k) % 4
            hs = env.hand_state_for(opp, tile, source="discard")
            try:
                sb = score_hand(hs, env.rules)
                loss += _normalize_points_verbose(sb, hs=hs, rules=env.rules)
            except Exception:
                pass
        return self._b("danger_weight") * loss

    def decide_ron(self, env, tile, points, loser): return True

    def decide_open_kong(self, env: Env, seat: int, tile: str) -> bool:
        if ALWAYS_CLAIM_DEBUG: return True
        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        if hand.count(tile) < 3: return False
        declared = _meld_count(p)
        base = self._b("open_kong_bias") + (0.2 if declared >= 1 else 0.0) + 0.25*self._ol()
        return random.random() < min(0.98, base)

    def decide_add_kong(self, env: Env, seat: int, tile: str) -> bool:
        if ALWAYS_CLAIM_DEBUG: return True
        base = self._b("open_kong_bias") + 0.25*self._ol()
        return random.random() < min(0.98, base)

    def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
        if ALWAYS_CLAIM_DEBUG: return True
        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        if hand.count(tile) < 2: return False

        w = self._effective_w(env, hand)
        before = composite_shape_metric(hand, w)

        h2 = hand[:]; c = 0
        for t2 in list(h2):
            if t2 == tile and c < 2:
                h2.remove(t2); c += 1

        after = composite_shape_metric(h2, w)
        declared = _meld_count(p)
        margin = self._b("pung_margin_base") - min(declared, 3) * self._b("margin_decay_per_meld")
        margin += 0.8 * self._ol()  # ← open-lean widens willingness

        if after <= before + margin:
            return True

        # Soft-margin override: allow near-neutral pungs occasionally
        soft_extra = self._ol() * (0.6 + 0.4*self._b("pung_bias"))
        if after <= before + margin + soft_extra:
            prob = min(0.98, self._b("pung_bias") * (0.75 + 0.25*self._b("meld_target_bonus")))
            return random.random() < prob
        return False

    def choose_chow(self, env: Env, seat: int, tile: str, chow_sets: List[Tuple[str,str]]) -> Optional[Tuple[str,str]]:
        if not chow_sets: return None
        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        w = self._effective_w(env, hand)
        before = composite_shape_metric(hand, w)

        best = None; best_score = 1e9
        for a,b in chow_sets:
            h2 = hand[:]
            if a in h2: h2.remove(a)
            if b in h2: h2.remove(b)
            score = composite_shape_metric(h2, w)
            if score < best_score: best, best_score = (a,b), score

        declared = _meld_count(p)
        margin = self._b("chow_margin_base") - min(declared, 3) * self._b("margin_decay_per_meld")
        margin += 1.1 * self._ol()

        if best_score <= before + margin:
            return best

        # Soft-margin override for chow
        soft_extra = self._ol() * (0.5 + 0.5*self._b("chow_bias"))
        if best_score <= before + margin + soft_extra:
            accept_prob = min(0.98, self._b("chow_bias") * (0.85 + 0.15*self._b("meld_target_bonus")) * (1.0 + 0.35*self._ol()))
            if random.random() < accept_prob:
                return best
        return None

    def decide_closed_kong(self, env: Env, seat: int, candidates: List[str]) -> Optional[str]:
        if not candidates: return None
        if ALWAYS_CLAIM_DEBUG: return candidates[0]
        base = self._b("closed_kong_bias")
        return random.choice(candidates) if (random.random() < base) else None

    def pick_discard(self, env: Env) -> str:
        p = env.players[self.seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        cnt = Counter(hand)
        opts = env.legal_discards(self.seat)
        if not opts: return p.concealed[0]
        w = self._effective_w(env, hand)
        best = None; best_val = 1e9
        for t in opts:
            h2 = hand[:]
            if t in h2: h2.remove(t)
            keep_val = self._b("keep_shape_weight") * composite_shape_metric(h2, w)
            danger = self._ev_danger(env, t)
            isolate_bonus = -0.5 if _tile_is_isolate(cnt, t) else 0.0
            val = keep_val + danger + isolate_bonus
            if val < best_val: best, best_val = t, val
        return best

class PayoutOptPolicy(WinProbPolicy):
    def _expected_feed_loss(self, env: Env, tile: str) -> float:
        loss = 0.0
        for k in (1,2,3):
            opp = (self.seat + k) % 4
            hs = env.hand_state_for(opp, tile, source="discard")
            try:
                sb = score_hand(hs, env.rules)
                loss += _normalize_points_verbose(sb, hs=hs, rules=env.rules)
            except Exception:
                pass
        return 0.05 * loss

    def pick_discard(self, env: Env) -> str:
        p = env.players[self.seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        cnt = Counter(hand)
        opts = env.legal_discards(self.seat)
        if not opts: return p.concealed[0]
        w = self._effective_w(env, hand)
        best = None; best_val = 1e9
        for t in opts:
            h2 = hand[:]
            if t in h2: h2.remove(t)
            keep = self._b("keep_shape_weight") * composite_shape_metric(h2, w)
            feed = self._expected_feed_loss(env, t)
            isolate_bonus = -0.5 if _tile_is_isolate(cnt, t) else 0.0
            val = keep + feed + isolate_bonus
            if val < best_val: best, best_val = t, val
        return best

class AggroPolicy(WinProbPolicy):
    def decide_open_kong(self, env, seat, tile): return True
    def decide_add_kong(self, env, seat, tile): return True
    def decide_pung(self, env, seat, tile):
        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        return hand.count(tile) >= 2 and (ALWAYS_CLAIM_DEBUG or random.random() < 0.97)
    def decide_closed_kong(self, env, seat, candidates):
        return candidates[0] if (candidates and (ALWAYS_CLAIM_DEBUG or random.random() < max(0.25, self._b("closed_kong_bias")))) else None
    def choose_chow(self, env, seat, tile, chow_sets):
        return chow_sets[0] if chow_sets else None
    
class HybridAggroPolicy(BasePolicy):
    """
    Hybrid core + aggro fallbacks:
      • pick_discard: PayoutOpt (risk-aware EV).
      • claims (pung/chow/open/closed-kong): try WinProb; if it declines, use an
        aggro override with higher acceptance probability, influenced by tuner biases
        and open_lean from rules.
    """
    def __init__(self, seat: int, rules: Dict, tuner: Optional[AdaptiveTuner]):
        super().__init__(seat, rules, tuner)
        self.wp = WinProbPolicy(seat, rules, tuner)
        self.po = PayoutOptPolicy(seat, rules, tuner)

    # ---- helpers
    def _ol(self) -> float:
        # reuse BasePolicy._ol for readability
        return super()._ol()

    # ---- decisions
    def decide_ron(self, env, tile, points, loser):
        # Always take Ron (same as WP).
        return True

    def decide_open_kong(self, env: Env, seat: int, tile: str) -> bool:
        # First: WP decision (context-aware). If it accepts, we're done.
        if self.wp.decide_open_kong(env, seat, tile):
            return True
        # Aggro fallback: slightly-boosted probability.
        base = self.tuner.get("open_kong_bias")
        declared = _meld_count(env.players[seat])
        bump = 0.10 + 0.07 * declared + 0.25 * self._ol()
        p = min(0.98, max(0.35, base) + bump)
        return random.random() < p

    def decide_add_kong(self, env: Env, seat: int, tile: str) -> bool:
        if self.wp.decide_add_kong(env, seat, tile):
            return True
        base = self.tuner.get("open_kong_bias")
        p = min(0.98, max(0.35, base) + 0.20 + 0.25 * self._ol())
        return random.random() < p

    def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
        # Try WP (margin-based).
        if self.wp.decide_pung(env, seat, tile):
            return True
        # Aggro fallback only if we truly can pung.
        pview = env.players[seat]
        hand = [t for t in pview.concealed if not is_flower(t)]
        if hand.count(tile) < 2:
            return False
        declared = _meld_count(pview)
        base = self.tuner.get("pung_bias")
        # Stronger bump as we open more; incorporate open_lean.
        p = min(0.98, max(0.55, base) + 0.15 + 0.10 * declared + 0.30 * self._ol())
        return random.random() < p

    def choose_chow(self, env: Env, seat: int, tile: str, chow_sets: List[Tuple[str,str]]) -> Optional[Tuple[str,str]]:
        # Ask WP to evaluate best chow by composite metric.
        best = self.wp.choose_chow(env, seat, tile, chow_sets)
        if best is not None:
            return best
        if not chow_sets:
            return None
        # Aggro fallback: still take a chow sometimes to push openings.
        base = self.tuner.get("chow_bias")
        declared = _meld_count(env.players[seat])
        p = min(0.98, max(0.45, base) + 0.12 + 0.08 * declared + 0.35 * self._ol())
        return chow_sets[0] if (random.random() < p) else None

    def decide_closed_kong(self, env: Env, seat: int, candidates: List[str]) -> Optional[str]:
        pick = self.wp.decide_closed_kong(env, seat, candidates)
        if pick:
            return pick
        if not candidates:
            return None
        base = self.tuner.get("closed_kong_bias")
        p = min(0.98, max(0.25, base) + 0.15 + 0.20 * self._ol())
        return random.choice(candidates) if (random.random() < p) else None

    def pick_discard(self, env: Env) -> str:
        # Keep PayoutOpt’s risk-aware discard (tends to perform best in EV).
        return self.po.pick_discard(env)


POLICY_MAP = {
    "random": RandomPolicy,
    "wp": WinProbPolicy,
    "payout": PayoutOptPolicy,
    "hybrid": None,   # filled below
    "aggro": AggroPolicy,
}

class HybridPolicy(BasePolicy):
    def __init__(self, seat: int, rules: Dict, tuner: Optional[AdaptiveTuner]):
        super().__init__(seat, rules, tuner)
        self.wp = WinProbPolicy(seat, rules, tuner)
        self.po = PayoutOptPolicy(seat, rules, tuner)
    def decide_ron(self, env, tile, points, loser): return self.po.decide_ron(env, tile, points, loser)
    def decide_open_kong(self, env, seat, tile): return self.wp.decide_open_kong(env, seat, tile)
    def decide_add_kong(self, env, seat, tile): return self.wp.decide_add_kong(env, seat, tile)
    def decide_pung(self, env, seat, tile): return self.wp.decide_pung(env, seat, tile)
    def choose_chow(self, env, seat, tile, chow_sets): return self.wp.choose_chow(env, seat, tile, chow_sets)
    def decide_closed_kong(self, env, seat, candidates): return self.wp.decide_closed_kong(env, seat, candidates)
    def pick_discard(self, env): return self.po.pick_discard(env)

POLICY_MAP["hybrid"] = HybridPolicy
POLICY_MAP["hyaggro"] = HybridAggroPolicy

def build_policies(lineup: List[str], rules: Dict, tuner: Optional[AdaptiveTuner]):
    if len(lineup) != 4: raise ValueError("lineup must have 4 entries")
    out = []
    for i,k in enumerate(lineup):
        cls = POLICY_MAP[k.lower()]
        if cls is None:
            raise ValueError(f"Unknown policy '{k}'")
        out.append(cls(i, rules, tuner))
    return out

# ---------------------------- Episode + I/O ----------------------------

def run_episode(rules: Dict, lineup: List[str], tuner: Optional[AdaptiveTuner], seed: Optional[int]=None, max_draws: int=600) -> Dict[str,Any]:
    env = Env(rules, seed=seed)
    policies = build_policies(lineup, rules, tuner)
    draws = 0
    while draws < max_draws and env.wall and not env.terminal:
        env.step_turn(policies)
        draws += 1
    if not env.terminal:
        env.terminal = {"winner": None, "source": "drawn_game", "points": 0,
                        "side_delta": env.side_delta[:], "side_events": env.side_events[:]}
    if tuner is not None:
        tuner.record_episode(env.terminal, env.claim_log)

    openings = {
        "before_flowers": [list(x) for x in env.opening_before_flowers],
        "after_flowers":  [list(x) for x in env.opening_after_flowers],
        "flowers":        [list(x) for x in env.opening_flowers],
    }

    return {
        "terminal": env.terminal,
        "stats": env.stats,
        "claim_log": env.claim_log,
        "openings": openings
    }

def generate_jsonl(path: str, n: int, rules: Dict, lineup: List[str], seed: Optional[int]=None, adaptive: bool=True):
    rng = random.Random(seed)
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    tuner = AdaptiveTuner() if adaptive else None
    with out.open("w", encoding="utf-8") as f:
        for _ in range(n):
            ep = run_episode(rules, lineup, tuner, seed=rng.randint(1,10**9))
            f.write(json.dumps(ep) + "\n")

# ---------------------------- Pretty-print helpers ----------------------------

def _split_4m1p_for_print(non_flowers: List[str]) -> Optional[Tuple[List[List[str]], List[str]]]:
    tiles = sorted(non_flowers)
    if len(tiles) != 14: return None
    cnt = Counter(tiles)
    def try_melds(c: Counter, taken: List[List[str]]) -> Optional[List[List[str]]]:
        if sum(c.values()) == 0: return taken
        if len(taken) == 4:
            left = [k for k,v in c.items() for _ in range(v)]
            return taken if len(left)==2 and left[0]==left[1] else None
        t = min(k for k,v in c.items() if v>0)
        if c[t] >= 3:
            c[t] -= 3
            r = try_melds(c, taken + [[t,t,t]])
            if r: return r
            c[t] += 3
        if _is_suit_tile(t):
            r,s = _tile_rank_suit(t)
            a,b = f"{r+1}{s}", f"{r+2}{s}"
            if c[a] > 0 and c[b] > 0:
                c[t] -= 1; c[a] -= 1; c[b] -= 1
                r2 = try_melds(c, taken + [[t,a,b]])
                if r2: return r2
                c[t] += 1; c[a] += 1; c[b] += 1
        return None
    for k,v in list(cnt.items()):
        if v >= 2:
            c2 = cnt.copy()
            c2[k] -= 2
            melds = try_melds(c2, [])
            if melds is not None:
                return (melds, [k,k])
    return None

def _pretty_hand(snap: Dict[str,Any]) -> str:
    before = " ".join(snap.get("concealed_before_win", []))
    after  = " ".join(snap.get("concealed_after_win", []))
    all_melds = snap.get("melds", []) or []

    def _lab(m: Dict[str,Any]) -> str:
        kind = _normalize_meld_type(m.get("kind") or m.get("type") or "?")
        tiles = ",".join(m.get("tiles", []))
        opened = m.get("open", True)
        from_disc = bool(m.get("from_discard", False))
        claimed = bool(m.get("claimed", False))
        # Origin label
        if from_disc:
            origin = "discard"
        elif claimed and opened and kind == "kong":
            origin = "add-kan"
        elif not opened:
            origin = "closed"
        else:
            origin = "self"
        return f"{kind}:{tiles} ({'open' if opened else 'closed'}; {origin})"

    melds_all_str = " | ".join(_lab(m) for m in all_melds) or "-"
    melds_claimed_disc = [m for m in all_melds if m.get("from_discard")]
    melds_claimed_disc_str = " | ".join(_lab(m) for m in melds_claimed_disc) or "-"

    # Count summary
    def _is_declared(m):
        return _normalize_meld_type(m.get("kind") or m.get("type") or "?") in {"chow","pung","kong"}
    n_declared = sum(1 for m in all_melds if _is_declared(m))
    n_from_disc = sum(1 for m in all_melds if _is_declared(m) and m.get("from_discard"))

    # Optional 4m+pair reconstruction for standard hands
    non_fl = [t for t in snap.get("concealed_after_win", []) if not is_flower(t)]
    recon = _split_4m1p_for_print(non_fl)
    pretty_struct = ""
    if recon:
        sets, eyes = recon
        sets_str = " | ".join([" ".join(s) for s in sets])
        eyes_str = " ".join(eyes)
        pretty_struct = f"\n structure: {sets_str}  +  pair: {eyes_str}"

    shape_tag = snap.get("shape_tag", "?")
    return "\n".join([
        f" seat={snap.get('seat')} source={snap.get('source')} points={snap.get('points')} tile={snap.get('winning_tile')} shape={shape_tag}",
        f" concealed(before): {before}",
        f" concealed(after) : {after}",
        f" declared melds   : {n_declared} (from discard: {n_from_disc})",
        f" melds (all)      : {melds_all_str}",
        f" melds (discard)  : {melds_claimed_disc_str}",
        f" flowers          : {' '.join(snap.get('flowers', [])) or '-'}",
        pretty_struct
    ])


# ---------------------------- Summary ----------------------------

def summarize_jsonl(path: str, rules: Dict, enforce_zero_sum: bool = True, print_wins: int = 0, print_openings: int = 0) -> None:
    total = wins = tsumo = ron = draws = 0
    seat_points = [0,0,0,0]
    seat_tsumo = [0,0,0,0]
    seat_ron   = [0,0,0,0]
    side_events_count = 0
    drift_hands = 0
    drift_total_abs = 0
    pay_disc = rules.get("payouts", {}).get("on_discard", {})
    pay_self = rules.get("payouts", {}).get("on_self_draw", {})
    printed = 0
    printed_openings = 0
    total_stats = {"chow":0, "pung":0, "kong_open":0, "kong_closed":0}

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip(): continue
            ep = json.loads(line)
            st = ep.get("stats", {})
            for k in total_stats: total_stats[k] += int(st.get(k,0))
            term = ep.get("terminal", {}) or {}
            total += 1
            delta = [0,0,0,0]

            if printed_openings < print_openings:
                opens = ep.get("openings", {})
                bf = opens.get("before_flowers", [])
                af = opens.get("after_flowers", [])
                fl = opens.get("flowers", [])
                print(f"\n=== Opening hands (episode {idx}) ===")
                for seat in range(4):
                    b = " ".join(bf[seat]) if seat < len(bf) else "-"
                    a = " ".join(af[seat]) if seat < len(af) else "-"
                    fs = " ".join(sorted(fl[seat])) if seat < len(fl) else "-"
                    print(f"  Seat {seat}: before: {b}")
                    print(f"           after(flowers replaced): {a}")
                    print(f"           flowers: {fs}")
                printed_openings += 1

            w = term.get("winner")
            winners_list = term.get("winners")

            if w is None and not winners_list:
                draws += 1
            else:
                src = term.get("source")
                if winners_list:
                    ron += len(winners_list)
                    loser = term.get("ron_loser")
                    wg = pay_disc.get("winner_gain", 1)
                    ll = pay_disc.get("loser_loss", -1)
                    if enforce_zero_sum and (wg + ll) != 0:
                        wg, ll = 1, -1
                    total_loser_loss = 0
                    for item in winners_list:
                        seat_w = int(item.get("seat"))
                        pts = int(item.get("points", 0))
                        delta[seat_w] += wg * pts
                        seat_ron[seat_w] += 1
                        wins += 1
                        total_loser_loss += ll * pts
                        if print_wins and printed < print_wins:
                            snap = item.get("winning_hand", {})
                            print("\n=== Winning hand ===")
                            print(_pretty_hand(snap) if snap else "(no snapshot)")
                            printed += 1
                    if isinstance(loser, int):
                        delta[loser] += total_loser_loss
                else:
                    wins += 1
                    pts = int(term.get("points", 0))
                    if src == "self_draw":
                        tsumo += 1
                        wg = pay_self.get("winner_gain", 1)
                        ol = pay_self.get("others_loss_each", -1)
                        if enforce_zero_sum and (wg + 3*ol) != 0:
                            wg, ol = 3, -1
                        delta[w] += wg * pts
                        for s in range(4):
                            if s != w: delta[s] += ol * pts
                        seat_tsumo[w] += 1
                    elif src == "discard":
                        ron += 1
                        loser = term.get("ron_loser")
                        wg = pay_disc.get("winner_gain", 1)
                        ll = pay_disc.get("loser_loss", -1)
                        if enforce_zero_sum and (wg + ll) != 0:
                            wg, ll = 1, -1
                        delta[w] += wg * pts
                        if isinstance(loser, int): delta[loser] += ll * pts
                        seat_ron[w] += 1

                    if print_wins and printed < print_wins:
                        snap = term.get("winning_hand", {})
                        print("\n=== Winning hand ===")
                        print(_pretty_hand(snap) if snap else "(no snapshot)")
                        printed += 1

            sd = term.get("side_delta")
            if isinstance(sd, list) and len(sd)==4:
                for i in range(4): delta[i] += int(sd[i])
            ev = term.get("side_events", [])
            if isinstance(ev, list): side_events_count += len(ev)

            net = sum(delta)
            if net != 0:
                drift_hands += 1
                drift_total_abs += abs(net)

            for i in range(4): seat_points[i] += delta[i]

    print(
        f"Episodes: {total} | Wins: {wins} (tsumo={tsumo}, ron={ron}) | "
        f"Draws: {draws} | Win rate: {wins/total:.2%} | Side events: {side_events_count}"
    )
    print(f"Claims: chow={total_stats['chow']} pung={total_stats['pung']} "
          f"open_kong={total_stats['kong_open']} closed_kong={total_stats['kong_closed']}")
    if drift_hands:
        print(f"[WARN] Non-zero-sum drift on {drift_hands} hand(s); total |drift|={drift_total_abs}")

    print("\nPer-seat breakdown:")
    for i in range(4):
        print(f"  Seat {i}: points={seat_points[i]} | wins=tsumo:{seat_tsumo[i]} ron:{seat_ron[i]}")

# ---------------------------- CLI ----------------------------

def _parse_lineup(s: str) -> List[str]:
    lineup = [x.strip().lower() for x in s.split(",")]
    allowed = {"random","wp","payout","hybrid","aggro","hyaggro"}
    if len(lineup) != 4 or any(x not in allowed for x in lineup):
        raise SystemExit("Invalid --lineup; choose 4 from {random,wp,payout,hybrid,aggro,hyaggro}")
    return lineup

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="Path to rules JSON")
    ap.add_argument("--out", default="episodes.jsonl", help="Output JSONL path")
    ap.add_argument("--episodes", type=int, default=500, help="Number of episodes (hands) to simulate")
    ap.add_argument("--lineup", default="aggro,wp,hybrid,payout",
                    help="Four agents from {random,wp,payout,hybrid,aggro}, comma-separated")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--print-wins", type=int, default=0, help="Print first N winning hand snapshots in summary")
    ap.add_argument("--print-openings", type=int, default=0, help="Print opening draws (before/after flowers) for first N episodes")
    ap.add_argument("--no-adaptive", action="store_true", help="Disable adaptive tuning (incl. 7P mix learning)")
    ap.add_argument("--open-lean", type=float, default=0.0,
                    help="Softly widens meld acceptance margins & boosts open claim probabilities (0..1).")
    args = ap.parse_args()

    rules = load_rules(args.rules)
    rules.setdefault("win_requirements", {}).setdefault("allow_pass_ron", True)
    rules.setdefault("special_events", {}).setdefault(
        "four_same_consecutive_discard", {"enabled": True, "amount_per_opponent": 1}
    )
    rules.setdefault("tileset", {}).setdefault("use_flowers", True)
    # pass policy knob via rules to policies
    rules.setdefault("policy", {})["open_lean"] = max(0.0, min(1.0, args.open_lean))

    lineup = _parse_lineup(args.lineup)
    rng = random.Random(args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    tuner = None if args.no_adaptive else AdaptiveTuner()
    with out.open("w", encoding="utf-8") as f:
        for _ in range(args.episodes):
            ep = run_episode(rules, lineup, tuner, seed=rng.randint(1,10**9))
            f.write(json.dumps(ep) + "\n")

    summarize_jsonl(args.out, rules, enforce_zero_sum=True, print_wins=args.print_wins, print_openings=args.print_openings)
