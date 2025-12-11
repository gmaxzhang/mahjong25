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
import collections
import json, random
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from algorithm.rules_io import load_rules
from algorithm.gameplay0 import HandState, Meld, score_hand  # scoring = source of truth

from collections import defaultdict

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

TILE_ORDER = (
    [f"{r}b" for r in range(1,10)] +
    [f"{r}w" for r in range(1,10)] +
    [f"{r}t" for r in range(1,10)] +
    ["E","S","W","N","C","F","B"]   # Winds + Dragons
)
TILE_TO_IDX = {t: i for i, t in enumerate(TILE_ORDER)}  # 0..33

def sanity_check_tile_encoding() -> None:
    """
    Assert TILE_ORDER indices and tile_to_class agree, and that
    flowers are excluded from 0..33.
    """
    # 1) Every tile in TILE_ORDER has the matching class index
    for i, t in enumerate(TILE_ORDER):
        c = tile_to_class(t)
        if c != i:
            raise RuntimeError(
                f"[SANITY] tile encoding mismatch: {t!r} "
                f"TILE_ORDER index={i}, tile_to_class={c}"
            )

    # 2) Flowers never map into 0..33
    for f in [f"F{i}" for i in range(1, 9)]:
        c = tile_to_class(f)
        if c is not None:
            raise RuntimeError(
                f"[SANITY] flower {f!r} unexpectedly has class index {c}"
            )

    print("[SANITY] TILE_ORDER and tile_to_class encoding are consistent.")


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

def _snapshot_winning_hand(
    env,
    seat: int,
    hs: HandState,
    points: int,
    source: str,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    p = env.players[seat]

    # What was "concealed" immediately before the win?
    if source == "self_draw":
        # hs.concealed is already "before win" (we removed winning_tile for shape)
        concealed_before = list(hs.concealed)
        # After win, your hand contains the drawn tile as well
        concealed_after = sorted(list(p.concealed))
    else:
        # discard win: winning_tile is not in concealed yet
        concealed_before = list(p.concealed)
        concealed_after = sorted(list(p.concealed) + [hs.winning_tile])

    # Pass through *real* meld flags instead of defaulting claimed/from_discard.
    meld_dicts: List[Dict[str, Any]] = []
    for m in p.melds:
        kind = getattr(m, "type", getattr(m, "kind", None))
        tiles = list(getattr(m, "tiles", []))

        is_open      = bool(getattr(m, "open", True))
        formed_claim = bool(getattr(m, "formed_by_claim",
                                    getattr(m, "claimed", False)))
        from_discard = bool(getattr(m, "from_discard", False))

        meld_dicts.append({
            "kind": kind,
            "tiles": tiles,
            "open": is_open,
            "claimed": formed_claim,
            "from_discard": from_discard,
        })

    snap = {
        "seat": seat,
        "source": source,
        "winning_tile": hs.winning_tile,
        "points": points,
        "flowers": sorted(list(p.flowers)),
        "melds": meld_dicts,
        "concealed_before_win": concealed_before,
        "concealed_after_win": concealed_after,
        "used_discard_claim": bool(getattr(p, "used_discard_claim", False)),
        "claim_log": list(getattr(env, "claim_log", [])),
    }

    if extra:
        snap.update(extra)

    # Optional: keep your virtual-meld debug if you still want it
    virt_melds, _ = find_virtual_melds_from_concealed(hs.concealed)
    # print("[DEBUG-WIN]",
    #       f"declared={len(hs.melds)}",
    #       f"virt={len(virt_melds)}",
    #       f"virt_melds={virt_melds}")

    return snap


# def _snapshot_winning_hand(env, seat: int, hs: HandState, points: int, source: str, extra: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
#     p = env.players[seat]
#     concealed_before = list(hs.concealed) if source == "self_draw" else list(p.concealed)
#     if source == "self_draw":
#         concealed_after = sorted([t for t in p.concealed])
#     else:
#         concealed_after = sorted([t for t in (p.concealed + [hs.winning_tile])])

#     snap = {
#         "seat": seat,
#         "source": source,
#         "winning_tile": hs.winning_tile,
#         "points": points,
#         "flowers": sorted(list(p.flowers)),
#         "melds": [_serialize_meld(m) for m in p.melds],
#         "concealed_before_win": concealed_before,
#         "concealed_after_win": concealed_after,
#         "used_discard_claim": bool(getattr(p, "used_discard_claim", False)),
#         "claim_log": list(getattr(env, "claim_log", [])),
#     }
#     if extra:
#         snap.update(extra)
#     virt_melds, _ = find_virtual_melds_from_concealed(hs.concealed)
#     # print("[DEBUG-WIN]",
#     #     f"declared={len(hs.melds)}",
#     #     f"virt={len(virt_melds)}",
#     #     f"virt_melds={virt_melds}")

#     return snap

# ---------------------------- Rough evaluators ----------------------------

ALL_TILE_CLASSES: List[str] = [
    *(f"{r}{s}" for s in SUITS for r in RANKS),
    *HONORS,
]


def _visible_tile_counts(env: "Env") -> Dict[str, int]:
    """
    Count how many copies of each tile *are already visible*:
    - all players' concealed tiles
    - all players' meld tiles
    - all players' flowers
    - all discards
    This lets us approximate "copies left" for ukeire as 4 - visible_count.
    """
    vis: Dict[str, int] = Counter()

    # Players: concealed, melds, flowers
    for seat in range(4):
        p = env.players[seat]

        # concealed
        for t in p.concealed:
            vis[t] += 1

        # flowers
        for t in p.flowers:
            vis[t] += 1

        # melds
        for m in p.melds:
            tiles = getattr(m, "tiles", [])
            for t in tiles:
                vis[t] += 1

    # Discards
    for seat in range(4):
        for t in env.discards[seat]:
            vis[t] += 1

    return vis

def rough_ukeire(
    env: "Env",
    seat: int,
    hand: Optional[List[str]] = None,
    declared_melds: int = 0,
) -> int:
    """
    Shanten-aware ukeire with *copies-left*:
      - Starting from the current hand + declared meld count,
      - For each tile class t with copies_left(t) > 0:
          simulate drawing t (add 1 copy to hand),
          recompute rough_shanten_like,
          if shanten strictly decreases, add copies_left(t) to "outs".

    copies_left(t) ≈ 4 - (#visible copies of t in all hands / melds / flowers / discards).

    NOTE: this is still approximate (we don't model wall depletion exactly),
    but it's a big step up from hand-only type-based ukeire.
    """
    # Default hand = this seat's concealed (no flowers)
    if hand is None:
        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]
    else:
        # strip flowers if caller passed a raw hand
        hand = [t for t in hand if not is_flower(t)]

    base = rough_shanten_like(hand, declared_melds)
    vis = _visible_tile_counts(env)

    outs = 0
    for t in ALL_TILE_CLASSES:
        # How many physical copies of t are still potentially left?
        copies_left = max(0, 4 - vis.get(t, 0))
        if copies_left <= 0:
            continue

        # Simulate drawing one copy of t
        new_hand = hand + [t]
        new_sh = rough_shanten_like(new_hand, declared_melds)

        # If this draw improves shanten, all remaining copies are "outs"
        if new_sh < base - 1e-9:
            outs += copies_left

    # if len(hand) <= 14:  # only log on normal hand sizes, not weird evals
    #     print(
    #         f"[DEBUG-UKEIRE] seat={seat} declared={declared_melds} "
    #         f"std_core={base:.2f} outs={outs}"
    #     )
    return outs

# def rough_ukeire(hand: List[str]) -> int:
#     cnt = Counter(x for x in hand if not is_flower(x))
#     eff: set[str] = set()
#     for t,c in cnt.items():
#         if _is_suit_tile(t):
#             eff.add(t)
#             r,s = _tile_rank_suit(t)
#             for dr in (-2,-1,1,2):
#                 rr = r + dr
#                 if 1 <= rr <= 9: eff.add(f"{rr}{s}")
#         else:
#             if c in (1,2): eff.add(t)
#     return len(eff)

# def rough_shanten_like(hand: List[str], declared_melds: int = 0) -> float:
#     """
#     Very rough shanten proxy for the *standard 4m+1p path*.

#     `declared_melds` counts already-locked chows/pungs/kongs.
#     We reduce the number of missing melds by this amount so that
#     calling a pung/chow actually shows up as progress.
#     """
#     tiles = [t for t in hand if not is_flower(t)]
#     cnt = Counter(tiles)

#     meldish = 0
#     pairish = 0

#     # First take triplets
#     for t in list(cnt):
#         while cnt[t] >= 3:
#             meldish += 1
#             cnt[t] -= 3

#     # Then take sequences
#     for s in SUITS:
#         for r in range(1, 8):
#             a, b, c = f"{r}{s}", f"{r+1}{s}", f"{r+2}{s}"
#             while cnt[a] > 0 and cnt[b] > 0 and cnt[c] > 0:
#                 meldish += 1
#                 cnt[a] -= 1
#                 cnt[b] -= 1
#                 cnt[c] -= 1

#     # Finally a single pair
#     for t in list(cnt):
#         while cnt[t] >= 2 and pairish < 1:
#             pairish += 1
#             cnt[t] -= 2

#     # Now account for already-declared melds.
#     total_melds = meldish + max(0, declared_melds)
#     missing_melds = max(0, 4 - total_melds)
#     missing_pair  = max(0, 1 - pairish)

#     val = missing_melds * 2 + missing_pair

#     # Strongly separate true tenpai / win shapes
#     if missing_melds == 0 and missing_pair == 0:
#         return -2.0   # complete hand
#     if missing_melds == 0 and missing_pair == 1:
#         return 0.0    # tenpai-ish

#     return float(val)


def rough_shanten_like(hand: List[str], declared_melds: int = 0) -> float:
    """
    Rough shanten-like metric for the standard 4 melds + 1 pair structure.

    Improvements:
      • Prefers early pair formation (3m+1p > 4m+0p).
      • Rewards multiple pairs as future pung potential.
      • Penalizes over-melded shapes with no pair.
      • Integrates concealed 'virtual melds' but limits total melds ≤ 4.
    """

    # --- Strip flowers ---
    tiles = [t for t in hand if not is_flower(t)]

    # --- Clamp declared meld count ---
    effective_declared = max(0, min(4, declared_melds))

    # --- Virtual melds from concealed tiles ---
    virt_melds, virt_used = [], []
    if effective_declared < 4:
        all_melds, _ = find_virtual_melds_from_concealed(tiles)
        avail_slots = 4 - effective_declared
        virt_melds = all_melds[:avail_slots]
        for meld in virt_melds:
            virt_used.extend(meld)

    virt_count = len(virt_melds)

    # --- Remove virtual meld tiles before counting residual shapes ---
    residual = tiles[:]
    for t in virt_used:
        if t in residual:
            residual.remove(t)
    cnt = Counter(residual)

    meldish = pairish = 0

    # Triplets first
    for t in list(cnt):
        while cnt[t] >= 3:
            meldish += 1
            cnt[t] -= 3

    # Sequences
    for s in SUITS:
        for r in range(1, 8):
            a, b, c = f"{r}{s}", f"{r+1}{s}", f"{r+2}{s}"
            while cnt[a] > 0 and cnt[b] > 0 and cnt[c] > 0:
                meldish += 1
                cnt[a] -= 1; cnt[b] -= 1; cnt[c] -= 1

    # Pairs
    for t in list(cnt):
        while cnt[t] >= 2:
            pairish += 1
            cnt[t] -= 2

    base_pairish = 1 if pairish >= 1 else 0

    # --- Combine declared + virtual + concealed melds ---
    total_melds = min(4, max(0, effective_declared + virt_count + meldish))
    missing_melds = max(0, 4 - total_melds)
    missing_pair = max(0, 1 - base_pairish)

    # --- Base shanten-like score ---
    base = float(missing_melds + missing_pair)

    # --- Custom shaping for realistic preference ---
    if missing_melds == 0 and missing_pair == 0:
        # 4m + 1p complete
        base = -2.0
    elif missing_melds == 1 and missing_pair == 0:
        # 3m + 1p preferred
        base = -0.5
    elif missing_melds == 0 and missing_pair == 1:
        # 4m + 0p penalized
        base = 1.5

    # --- Extra pair bonuses / over-meld penalties ---
    extra_pairs = max(0, pairish - 1)
    if extra_pairs > 0:
        base -= 0.5 * extra_pairs

    if total_melds <= 2 and base_pairish == 1:
        base -= 0.3

    if total_melds >= 3 and base_pairish == 0 and len(residual) >= 3:
        base += 1.0

    return float(base)



# def rough_shanten_like(hand: List[str], declared_melds: int = 0) -> float:
#     """
#     Very rough shanten proxy for the *standard 4m+1p path*.

#     - `declared_melds` = already-locked open melds (p.melds: chow/pung/kong).
#     - We ALSO treat fully-formed *closed* melds in the concealed hand as
#       'virtual' melds, but only up to the remaining meld slots in a 4-meld
#       structure (so declared + virtual <= 4).
#     """

#     # Strip flowers
#     tiles = [t for t in hand if not is_flower(t)]

#     # Clamp declared melds into a sane range
#     effective_declared = max(0, min(4, declared_melds))

#     # ---- Virtual melds from concealed structure ----
#     virt_melds: List[Tuple[str, ...]] = []
#     virt_used: List[str] = []

#     if effective_declared < 4:
#         # Find all candidate virtual melds from the concealed tiles
#         all_melds, _ = find_virtual_melds_from_concealed(tiles)

#         # We can only "fit" up to (4 - effective_declared) more melds total
#         avail_slots = 4 - effective_declared
#         virt_melds = all_melds[:avail_slots]

#         # Rebuild which tiles are actually consumed by the chosen virtual melds
#         for meld in virt_melds:
#             for t in meld:
#                 virt_used.append(t)

#     virt_count = len(virt_melds)

#     # Remove virtual-meld tiles from the hand so we don't double-count them
#     residual = tiles[:]
#     for t in virt_used:
#         if t in residual:
#             residual.remove(t)

#     cnt = Counter(residual)

#     meldish = 0
#     pairish = 0

#     # First take triplets (from residual)
#     for t in list(cnt):
#         while cnt[t] >= 3:
#             meldish += 1
#             cnt[t] -= 3

#     # Then take sequences (from residual)
#     for s in SUITS:
#         for r in range(1, 8):
#             a, b, c = f"{r}{s}", f"{r+1}{s}", f"{r+2}{s}"
#             while cnt[a] > 0 and cnt[b] > 0 and cnt[c] > 0:
#                 meldish += 1
#                 cnt[a] -= 1
#                 cnt[b] -= 1
#                 cnt[c] -= 1

#     # Finally, a single pair (from residual)
#     for t in list(cnt):
#         while cnt[t] >= 2 and pairish < 1:
#             pairish += 1
#             cnt[t] -= 2

#     # ---- Account for BOTH declared and virtual melds (never > 4) ----
#     total_melds = min(4, effective_declared + virt_count + meldish)
#     missing_melds = max(0, 4 - total_melds)
#     missing_pair  = max(0, 1 - pairish)

#     # Base shanten-like value
#     base = float(missing_melds + missing_pair)

#     # Special handling for very good shapes
#     if missing_melds == 0 and missing_pair == 0:
#         # Fully complete 4m+1p structure
#         base = -2.0
#     elif missing_melds == 0 and missing_pair == 1:
#         # Classic 1-away: 4 melds, no pair yet
#         base = 0.25
#     elif missing_melds == 1 and missing_pair == 0:
#         # 3 melds + 1 pair with extras → strong 1-shanten
#         base = 0.0

#     # if len(hand) <= 14:
#     #     print(
#     #         "[DEBUG-SHANTEN]",
#     #         f"declared_raw={declared_melds}",
#     #         f"declared_eff={effective_declared}",
#     #         f"virt_count={virt_count}",
#     #         f"virt_melds={virt_melds}",
#     #         f"residual_len={len(residual)}",
#     #         f"total_melds={total_melds}",
#     #         f"missing_melds={missing_melds}",
#     #         f"missing_pair={missing_pair}",
#     #         f"pairish={pairish}",
#     #         f"base={base:.2f}",
#     #     )

#     return base


# def rough_shanten_like(hand: List[str], declared_melds: int = 0) -> float:
#     """
#     Very rough shanten proxy for the *standard 4m+1p path*.

#     `declared_melds` = already-locked open melds (p.melds: chow/pung/kong).
#     We ALSO treat fully-formed *closed* melds in the concealed hand as
#     'virtual' melds: we carve them out first, bump the effective meld count,
#     and only then look for additional melds/pairs in the residual tiles.
#     """
#     # Strip flowers
#     tiles = [t for t in hand if not is_flower(t)]

#     # ---- extract virtual melds from concealed structure ----
#     virt_melds, virt_used = find_virtual_melds_from_concealed(tiles)
#     virt_count = len(virt_melds)

#     # Remove virtual-meld tiles from the hand so we don't double-count them
#     residual = tiles[:]
#     for t in virt_used:
#         if t in residual:
#             residual.remove(t)

#     cnt = Counter(residual)

#     meldish = 0
#     pairish = 0

#     # First take triplets (from residual)
#     for t in list(cnt):
#         while cnt[t] >= 3:
#             meldish += 1
#             cnt[t] -= 3

#     # Then take sequences (from residual)
#     for s in SUITS:
#         for r in range(1, 8):
#             a, b, c = f"{r}{s}", f"{r+1}{s}", f"{r+2}{s}"
#             while cnt[a] > 0 and cnt[b] > 0 and cnt[c] > 0:
#                 meldish += 1
#                 cnt[a] -= 1
#                 cnt[b] -= 1
#                 cnt[c] -= 1

#     # Finally, a single pair (from residual)
#     for t in list(cnt):
#         while cnt[t] >= 2 and pairish < 1:
#             pairish += 1
#             cnt[t] -= 2

#     # ---- Account for BOTH declared and virtual melds ----
#     total_melds = meldish + max(0, declared_melds) + virt_count
#     missing_melds = max(0, 4 - total_melds)
#     missing_pair  = max(0, 1 - pairish)

#     val = missing_melds * 2 + missing_pair

#     # Strongly separate true tenpai / win shapes
#     if missing_melds == 0 and missing_pair == 0:
#         # fully complete: treat as clearly better than any tenpai
#         base = -2.0
#     elif (
#         (missing_melds == 0 and missing_pair == 1) or
#         (missing_melds == 1 and missing_pair == 0)
#     ):
#         # "one block away" either as "need only the pair" OR "need only the last meld"
#         # Treat both as tenpai-ish in this coarse proxy.
#         base = 0.0
#     else:
#         base = float(val)

#     if len(hand) <= 14:  # only print on real hands, not full-wall evals
#         print("[DEBUG-SHANTEN]",
#               f"declared={declared_melds}",
#               f"virt_count={virt_count}",
#               f"virt_melds={virt_melds}",
#               f"residual_len={len(residual)}",
#               f"missing_melds={missing_melds}",
#               f"missing_pair={missing_pair}",
#               f"pairish={pairish}",
#               f"base={base}")
#     return base


# --- Seven-pairs distance + Composite metric (weight is learned + context gated) ---

# def chiitoi_shanten_like(hand: List[str], declared_melds: int = 0) -> float:
#     """
#     Rough seven-pairs shanten.
#     If we've already declared any melds, chiitoi is basically dead,
#     so return a value that is always worse than any standard path.
#     """
#     if declared_melds > 0:
#         # Max realistic chiitoi shanten is 6; 7.0 keeps it always dominated.
#         return 7.0

#     tiles = [t for t in hand if not is_flower(t)]
#     cnt = Counter(tiles)

#     pair_types   = sum(1 for c in cnt.values() if c >= 2)
#     single_types = sum(1 for c in cnt.values() if c == 1)

#     pair_types = min(pair_types, 7)
#     useful_singles = min(single_types, 7 - pair_types)

#     # Need 7 distinct pairs total
#     need_pairs = 7 - pair_types
#     # If we don't even have 7 distinct values (pairs + singles),
#     # we will need extra tiles beyond that; charge them at 0.5 each.
#     need_extra_tiles = max(0, 7 - (pair_types + useful_singles))

#     return float(need_pairs + 0.5 * need_extra_tiles)

def chiitoi_shanten_like(hand: List[str], declared_melds: int = 0) -> float:
    """
    Rough seven-pairs shanten.
    If we've already declared any melds, chiitoi is basically dead,
    so return a value that is always worse than any standard path.
    """
    if declared_melds > 0:
        # Max realistic chiitoi shanten is 6; 7.0 keeps it always dominated.
        return 7.0

    tiles = [t for t in hand if not is_flower(t)]
    cnt = Counter(tiles)

    pair_types   = sum(1 for c in cnt.values() if c >= 2)
    single_types = sum(1 for c in cnt.values() if c == 1)

    pair_types = min(pair_types, 7)
    useful_singles = min(single_types, 7 - pair_types)

    # Need 7 distinct pairs total
    need_pairs = 7 - pair_types
    # If we don't even have 7 distinct values (pairs + singles),
    # we will need extra tiles beyond that; charge them at 0.5 each.
    need_extra_tiles = max(0, 7 - (pair_types + useful_singles))

    return float(need_pairs + 0.5 * need_extra_tiles)

# def composite_shape_metric(
#     hand: List[str],
#     chiitoi_weight: float,
#     declared_melds: int,
#     env: "Env",
#     seat: int,
# ) -> float:
#     std_core = rough_shanten_like(hand, declared_melds)
#     uke = rough_ukeire(env, seat, hand, declared_melds)
#     # DEBUG: see how outs behave relative to declared + virtual melds
#     if len(hand) <= 14:  # avoid crazy spam for weird states
#         print(f"[DEBUG-UKEIRE] declared={declared_melds} "
#               f"std_core={std_core:.2f} outs={uke}")

#     std = std_core - 0.04 * float(uke)
#     ctt = chiitoi_shanten_like(hand, declared_melds)
#     w = max(0.0, min(1.0, chiitoi_weight))
#     return (1.0 - w) * std + w * ctt

def composite_shape_metric(
    hand: List[str],
    chiitoi_weight: float,
    declared_melds: int,
    env: "Env",
    seat: int,
) -> float:
    std_core = rough_shanten_like(hand, declared_melds)
    uke = rough_ukeire(env, seat, hand, declared_melds)

    # if len(hand) <= 14:
    #     print(f"[DEBUG-UKEIRE] seat={seat} declared={declared_melds} "
    #           f"std_core={std_core:.2f} outs={uke}")

    std = std_core - 0.02 * float(uke)
    ctt = chiitoi_shanten_like(hand, declared_melds)
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
    if len(melds) == 4 and all(getattr(m, "open", False) for m in melds):
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

    # --- Flower / honor / dragon bonuses (rules-aware) ---
    fl_bonus = 0
    if rules and "flower_points" in rules:
        fp = rules["flower_points"]
        per_flower = fp.get("per_flower_tile", 0)
        fl_bonus += per_flower * sum(1 for f in flowers if is_flower(f))

        set_pts = fp.get("set_points", {})
        dragon_pts = set_pts.get("dragon", {})
        wind_pts   = set_pts.get("wind", {})
        suit_pts   = set_pts.get("suit", {})

        for m in melds:
            k = getattr(m, "kind", getattr(m, "type", "")).lower()
            tiles = getattr(m, "tiles", [])
            if not tiles:
                continue

            first = tiles[0]
            if first in ("C", "F", "B"):   # dragons
                bonus += dragon_pts.get(k, 0)
            elif first in ("E", "S", "W", "N"):  # winds
                bonus += wind_pts.get(k, 0)
            else:
                bonus += suit_pts.get(k, 0)
    else:
        # fallback if rules not provided
        fl_bonus = sum(1 for f in flowers if is_flower(f))
        #bonus += fl_bonus
        for t in all_tiles:
            if t in ("E", "S", "W", "N"): bonus += 1
            if t in ("C", "F", "B"): bonus += 2
    bonus += fl_bonus


    total_special = base_special if specials else base
    total = total_special + bonus

    # if specials:
    #     print(f"[scoring] specials={specials} base={base_special} + bonus={bonus} → total={total}")

    # Attach verbose scoring info for downstream consumers (episodes.jsonl, stats, etc.)
    payload = {
        "specials": specials,          # e.g. ["peng_peng_hu", "one_color"]
        "base_if_special": total_special,
        "raw_base": base,
        "bonus": bonus,
        "total": int(total),
        "shape_tag": shape_tag,        # "standard" or "seven_pairs" or whatever you set upstream
    }

    # Store on hs (HandState) so it gets serialized into env_terminal["winning_hand"]
    if hs is not None:
        try:
            setattr(hs, "score_verbose", payload)
        except Exception:
            pass

    # Also store on sb in case some paths look there
    try:
        setattr(sb, "score_verbose", payload)
    except Exception:
        pass

    # if specials:
    #     print(f"[scoring] specials={specials} base={base_special} + bonus={bonus} → total={total}")
    return int(total)

from collections import Counter
from typing import List, Tuple


def find_virtual_melds_from_concealed(concealed: List[str]) -> Tuple[List[Tuple[str, ...]], List[str]]:
    """
    From a concealed hand, greedily extract non-overlapping *closed* melds
    (pungs & chows) to treat as 'virtual melds' for shanten/shape evaluation.

    Returns:
      virt_melds: list of tile-tuples, e.g. [("3b","4b","5b"), ("7w","7w","7w")]
      used_tiles: flat list of tiles that were consumed into those virtual melds
                  (so you can subtract them from the hand when computing shanten).

    NOTE: This is purely evaluative; it does NOT touch Env or p.melds.
    """
    tiles = [t for t in concealed if not is_flower(t)]
    cnt = Counter(tiles)

    virt_melds: List[Tuple[str, ...]] = []
    used_tiles: List[str] = []

    # ---- 1) Take pungs first ----
    for t, c in list(cnt.items()):
        while c >= 3:
            virt_melds.append((t, t, t))
            used_tiles.extend([t, t, t])
            c -= 3
            cnt[t] -= 3

    # ---- 2) Then take chows greedily per suit ----
    for suit in SUITS:  # e.g. SUITS = ("b", "w", "t")
        # Build rank multiset from remaining tiles of that suit
        ranks_cnt = Counter()
        for t, c in cnt.items():
            if _is_suit_tile(t):
                r, s = _tile_rank_suit(t)
                if s == suit and c > 0:
                    ranks_cnt[r] += c

        # Extract sequences 123..789 greedily
        while True:
            made_any = False
            for r in range(1, 8):
                if ranks_cnt[r] > 0 and ranks_cnt[r+1] > 0 and ranks_cnt[r+2] > 0:
                    t1 = f"{r}{suit}"
                    t2 = f"{r+1}{suit}"
                    t3 = f"{r+2}{suit}"
                    virt_melds.append((t1, t2, t3))
                    used_tiles.extend([t1, t2, t3])

                    for rr in (r, r+1, r+2):
                        ranks_cnt[rr] -= 1
                        cnt[f"{rr}{suit}"] -= 1
                    made_any = True
            if not made_any:
                break

    return virt_melds, used_tiles



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
                if t in ("pung","chow"):
                    used_actions[who].add(t)
                elif t.startswith("kong(open"):
                    used_actions[who].add("kong_open")
                elif t.startswith("kong(closed"):
                    used_actions[who].add("kong_closed")

        # Winner seats + points (env_terminal still carries old 'points'; we may override with score_verbose)
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
            st["tsumo_wins"] = self._ema_add(
                st["tsumo_wins"],
                1.0 if (env_terminal.get("source")=="self_draw" and won) else 0.0
            )
            fed = (isinstance(ron_loser, int) and ron_loser == s)
            st["ron_losses"] = self._ema_add(st["ron_losses"], 1.0 if fed else 0.0)
            st["ron_points_lost"] = self._ema_add(
                st["ron_points_lost"],
                float(env_terminal.get("points",0)) if fed else 0.0
            )

        # Small helper so we can safely add new shape keys on the fly
        def _shape_ema(key: str, value: float):
            if key not in self.shape:
                self.shape[key] = 0.0
            self.shape[key] = self._ema_add(self.shape[key], value)

        # Global shape EMAs + action attributions
        def _acc_win(item: Dict[str, Any]):
            """
            Use verbose scoring payload if available; otherwise fall back to
            legacy points + shape_tag.
            """
            snap = item.get("winning_hand", {}) or {}

            # Prefer verbose scoring attached by _normalize_points_verbose
            sv = snap.get("score_verbose") or item.get("score_verbose")

            if sv is not None:
                pts = float(sv.get("total", item.get("points", 0)))
                shape_tag = sv.get("shape_tag", snap.get("shape_tag", "standard"))
                specials = sv.get("specials", [])
            else:
                pts = float(item.get("points", 0))
                shape_tag = snap.get("shape_tag", "standard")
                specials = []

            # Overall win count
            _shape_ema("wins", 1.0)

            # Keep existing std vs seven_pairs counters, but also respect specials
            if shape_tag == "seven_pairs" or "seven_pairs" in specials:
                _shape_ema("sp_wins", 1.0)
                _shape_ema("sp_pts", pts)
            else:
                _shape_ema("std_wins", 1.0)
                _shape_ema("std_pts", pts)

            # Optional: track each special type separately
            for sp in specials:
                _shape_ema(f"{sp}_wins", 1.0)
                _shape_ema(f"{sp}_pts", pts)

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
                    self.act[a]["win_pts"]  = self._ema_add(
                        self.act[a]["win_pts"],
                        points_by_winner.get(s,0.0)
                    )
                if isinstance(ron_loser, int) and ron_loser == s:
                    self.act[a]["feed_uses"] = self._ema_add(self.act[a]["feed_uses"], 1.0)

        self._nudge()

    # def record_episode(self, env_terminal: Dict[str, Any], env_claim_log: List[Dict[str, Any]]):
    #     # Collect which actions each seat used this hand
    #     used_actions = {s: set() for s in range(4)}
    #     for ev in env_claim_log:
    #         t = ev.get("type","")
    #         who = ev.get("who")
    #         if isinstance(who, int):
    #             if t in ("pung","chow"): used_actions[who].add(t)
    #             elif t.startswith("kong(open"):
    #                 used_actions[who].add("kong_open")
    #             elif t.startswith("kong(closed"):
    #                 used_actions[who].add("kong_closed")

    #     # Winner seats + points
    #     winners = []
    #     if env_terminal.get("source") == "discard" and env_terminal.get("winners"):
    #         for w in env_terminal["winners"]:
    #             winners.append((int(w.get("seat", -1)), float(w.get("points", 0))))
    #     elif isinstance(env_terminal.get("winner"), int):
    #         winners.append((int(env_terminal.get("winner")), float(env_terminal.get("points",0))))

    #     # Per-seat EMAs + action EMAs
    #     ron_loser = env_terminal.get("ron_loser")
    #     for s in range(4):
    #         st = self.stats[s]
    #         st["hands"] = self._ema_add(st["hands"], 1.0)
    #         st["open_events"] = self._ema_add(st["open_events"], 1.0 if used_actions[s] else 0.0)
    #         won = any(ws == s for (ws, _) in winners)
    #         st["open_wins"] = self._ema_add(st["open_wins"], 1.0 if (won and used_actions[s]) else 0.0)
    #         st["closed_wins"] = self._ema_add(st["closed_wins"], 1.0 if (won and not used_actions[s]) else 0.0)
    #         st["tsumo_wins"] = self._ema_add(st["tsumo_wins"], 1.0 if (env_terminal.get("source")=="self_draw" and won) else 0.0)
    #         fed = (isinstance(ron_loser, int) and ron_loser == s)
    #         st["ron_losses"] = self._ema_add(st["ron_losses"], 1.0 if fed else 0.0)
    #         st["ron_points_lost"] = self._ema_add(st["ron_points_lost"], float(env_terminal.get("points",0)) if fed else 0.0)

    #     # Global shape EMAs + action attributions
    #     def _acc_win(item: Dict[str, Any]):
    #         pts = float(item.get("points", 0))
    #         snap = item.get("winning_hand", {}) or {}
    #         shape_tag = snap.get("shape_tag", "standard")
    #         self.shape["wins"] = self._ema_add(self.shape["wins"], 1.0)
    #         if shape_tag == "seven_pairs":
    #             self.shape["sp_wins"] = self._ema_add(self.shape["sp_wins"], 1.0)
    #             self.shape["sp_pts"]  = self._ema_add(self.shape["sp_pts"], pts)
    #         else:
    #             self.shape["std_wins"] = self._ema_add(self.shape["std_wins"], 1.0)
    #             self.shape["std_pts"]  = self._ema_add(self.shape["std_pts"], pts)

    #     if env_terminal.get("source") == "discard" and env_terminal.get("winners"):
    #         for w in env_terminal["winners"]:
    #             _acc_win(w)
    #     elif isinstance(env_terminal.get("winner"), int):
    #         _acc_win(env_terminal)

    #     # Action-EMAs: uses, wins, feeds
    #     winners_set = {ws for (ws, _) in winners}
    #     points_by_winner = {ws: pts for (ws, pts) in winners}
    #     for s in range(4):
    #         for a in used_actions[s]:
    #             self.act[a]["uses"] = self._ema_add(self.act[a]["uses"], 1.0)
    #             if s in winners_set:
    #                 self.act[a]["win_uses"] = self._ema_add(self.act[a]["win_uses"], 1.0)
    #                 self.act[a]["win_pts"]  = self._ema_add(self.act[a]["win_pts"], points_by_winner.get(s,0.0))
    #             if isinstance(ron_loser, int) and ron_loser == s:
    #                 self.act[a]["feed_uses"] = self._ema_add(self.act[a]["feed_uses"], 1.0)

    #     self._nudge()

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
        self.turn = random.randint(0, 3)
        self.discards: List[List[str]] = [[] for _ in range(4)]
        self.discard_history: List[Tuple[int,str]] = []
        self.last_discard: Optional[Tuple[int,str]] = None
        self.side_delta = [0,0,0,0]
        self.side_events: List[Dict[str,Any]] = []
        self.terminal: Optional[Dict[str,Any]] = None
        self.stats = {"chow":0, "pung":0, "kong_open":0, "kong_closed":0}
        self.claim_log: List[Dict[str, Any]] = []
        # For RL legal-head extraction (multi-way chow/kong heads)
        self._pending_chow_sets: Dict[int, List[Tuple[str, str]]] = {}
        self._pending_kong_candidates: Dict[int, List[str]] = {}

        # Opening snapshots
        for _ in range(13):
            for p in self.players: self._draw_into(p, back=False)
        self._draw_into(self.players[self.turn], back=False)
        for s, p in enumerate(self.players):
            expected = 14 if s == self.turn else 13
            #print(f"[init-check] seat={s} has {len(p.concealed)} tiles (expected {expected})")
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
        #print(f"[flower-check] seat={seat} entering with concealed={p.concealed}")
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

    def debug_player_state(self, seat: int, tag: str = "") -> None:
        p = self.players[seat]
        concealed = list(getattr(p, "concealed", []))
        melds = list(getattr(p, "melds", []))
        flowers = list(getattr(p, "flowers", []))

        C = len(concealed)
        M = len(melds)
        meld_tiles = []
        for m in melds:
            tiles = getattr(m, "tiles", None)
            if tiles is None:
                # fall back to treating meld itself as a list
                if isinstance(m, (list, tuple)):
                    tiles = m
                else:
                    continue
            meld_tiles.extend(tiles)

        F = len(flowers)
        total = C + len(meld_tiles) + F

        # print(
        #     f"[debug-hand] tag={tag} seat={seat} "
        #     f"C={C} M={M} meld_tiles={len(meld_tiles)} flowers={F} total={total} "
        #     f"concealed={concealed} "
        #     f"melds={[getattr(m, 'tiles', m) for m in melds]}"
        # )

    # ---- win acceptor with strict shape + menqing 10/11 gate (discard-claim only)
    def _try_accept_win(self, seat: int, source: str, winning_tile: str) -> Optional[Tuple[int, str]]:
        p = self.players[seat]
        concealed_for_shape = p.concealed[:]
        if source == "self_draw" and winning_tile in concealed_for_shape:
            concealed_for_shape.remove(winning_tile)
        ok, shape_tag = _legal_win_shape(concealed_for_shape, p.melds, winning_tile, source)
        if not ok: return None
        hs = self.hand_state_for(seat, winning_tile, source)
        # for m in hs.melds:
        #     print(f"[DEBUG] hs.meld: {m.type} open={getattr(m,'open',None)} from_discard={getattr(m,'from_discard',None)}")
        try:
            sb = score_hand(hs, self.rules)
        except Exception:
            return None
        pts_norm = _normalize_points_verbose(sb, shape_tag, hs=hs, rules=self.rules)
        # if seat == 0:  # or drop this guard if you want all four seats
            # snap = _snapshot_winning_hand(self, seat, hs, pts_norm, source, extra={"shape_tag": shape_tag})
            # print(
            #     "[WIN-DEBUG]",
            #     f"seat={seat}",
            #     f"source={source}",
            #     f"tile={winning_tile}",
            #     f"pts={pts_norm}",
            #     f"shape={shape_tag}",
            #     f"used_discard_claim={p.used_discard_claim}",
            # )
            # print("concealed:", hs.concealed)
            # print("winning_tile:", hs.winning_tile)
            # print("flowers:", hs.flowers)
            # for idx, m in enumerate(hs.melds):
            #     print(f"meld[{idx}]: type={m.type}, open={getattr(m, 'open', None)}, from_discard={getattr(m, 'from_discard', None)}")


            # # Compute open/closed counts
            # meld_infos = []
            # for m in hs.melds:
            #     mtype = _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None)))
            #     tiles = list(getattr(m, "tiles", []))
            #     is_open = bool(getattr(m, "open", True))
            #     meld_infos.append((mtype, tiles, is_open))

            # num_open = sum(1 for _, _, is_open in meld_infos if is_open)
            # num_closed = len(meld_infos) - num_open

            # print("    melds:")
            # for idx, (mtype, tiles, is_open) in enumerate(meld_infos):
            #     print(
            #         f"      meld[{idx}]:",
            #         f"type={mtype}",
            #         f"tiles={tiles}",
            #         f"open={is_open}",
            #     )

            # print(f"    meld_summary: open={num_open}, closed={num_closed}")

            # Optional: sanity check for eating-hand bonus
            # Replace 'eating_hand' with whatever flag/field you actually use
            # has_eating_bonus = getattr(pts_norm, "eating_hand", False)
            # if num_open == 4 and not has_eating_bonus:
            #     print("    [WARN] Hand has 4 open melds but eating-hand bonus is NOT present in pts_norm!")

    

        
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
        else:
            pass
            # if seat == 0:
            #     print(f"[win-debug] seat0 TSUMO pts={pts_norm}")
        ok, shape_tag = _legal_win_shape(concealed_for_shape, p.melds, winning_tile, source)
        if seat == 0 and ok:
            virt_melds, _ = find_virtual_melds_from_concealed(concealed_for_shape)
            # print(
            #     f"[shape-debug] seat={seat} src={source} shape={shape_tag} "
            #     f"declared={len(p.melds)} virt={len(virt_melds)} "
            #     f"concealed={sorted(concealed_for_shape)}"
            # )

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
        #print(f"[confirm-kong] CLOSED-KONG formed seat={seat} tile={tile} melds={self.players[seat].melds}")

    def _after_kong_draw_and_maybe_tsumo(self, seat: int, policies) -> bool:
        self._draw_into(self.players[seat], back=True)
        self._settle_flowers(seat)
        drawn = self.players[seat].concealed[-1] if self.players[seat].concealed else None
        res = self._try_accept_win(seat, "self_draw", drawn) if drawn else None
        return res is not None

    # ---- add-kan (upgrade exposed pung to open kong)
    def _maybe_added_kongs(self, seat: int, policies) -> bool:
        """
        Upgrade exposed pungs to open kongs (add-kan).

        We treat the 'kong' head as a binary yes/no for a single candidate tile:
        - idx == 0 → do not upgrade
        - idx == 1 → upgrade this pung to a kong
        """
        p = self.players[seat]
        changed = False

        while True:
            upgrade = None  # (meld_idx, tile) to upgrade

            for meld_idx, m in enumerate(p.melds):
                kind = _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None)))
                opened = bool(getattr(m, "open", True))
                tiles = list(getattr(m, "tiles", []))

                # Only upgrade open pungs of the form (X,X,X)
                if kind == "pung" and opened and len(tiles) == 3 and tiles[0] == tiles[1] == tiles[2]:
                    tile = tiles[0]
                    if p.concealed.count(tile) < 1:
                        continue

                    # ---- expose THIS tile as the only kong candidate for RL/teacher ----
                    # This is the key step from "Step 3"
                    self._pending_kong_candidates[seat] = [tile]

                    ok = False
                    try:
                        # See if RL/teacher forced a kong decision
                        forced = self._pop_forced(seat, "kong") or self._pop_forced(seat, None)

                        if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
                            ok = True

                        elif forced is not None:
                            forced_idx = int(forced["idx"])
                            # In the add-kong context: 0 = no, 1 = yes
                            ok = bool(forced_idx)

                        else:
                            decider = getattr(policies[seat], "decide_add_kong", None)
                            # This call will go through RLPolicy.decide_add_kong → _decide_index("kong")
                            ok = (decider is None) or decider(self, seat, tile)

                    finally:
                        # ---- close the kong decision window for this seat ----
                        # so later steps don't see stale candidates
                        self._pending_kong_candidates.pop(seat, None)

                    if ok:
                        upgrade = (meld_idx, tile)
                        break  # stop scanning p.melds

            if upgrade is None:
                return changed

            meld_idx, tile = upgrade

            # Safety: guard in case of weird index
            if not (0 <= meld_idx < len(p.melds)):
                print(f"[add-kong-error] seat={seat} bad meld_idx={meld_idx} len={len(p.melds)}")
                return changed

            # Perform the actual upgrade
            p.concealed.remove(tile)
            p.melds[meld_idx] = _make_meld(
                "kong", (tile, tile, tile, tile),
                open=True, formed_by_claim=True, from_discard=False
            )
            self.stats["kong_open"] += 1
            self.claim_log.append({"who": seat, "type": "kong(open,added)", "on": tile})
            self.turn = seat

            if self._after_kong_draw_and_maybe_tsumo(seat, policies):
                return True

            changed = True


    # def _maybe_added_kongs(self, seat: int, policies) -> bool:
    #     p = self.players[seat]
    #     changed = False
    #     while True:
    #         upgrade = None
    #         for idx, m in enumerate(p.melds):
    #             kind = _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None)))
    #             opened = bool(getattr(m, "open", True))
    #             tiles = list(getattr(m, "tiles", []))
    #             if kind == "pung" and opened and len(tiles) == 3 and tiles[0] == tiles[1] == tiles[2]:
    #                 tile = tiles[0]
    #                 if p.concealed.count(tile) >= 1:
    #                     forced = self._pop_forced(seat, "kong") or self._pop_forced(seat, None)
    #                     ok = False
    #                     if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
    #                         ok = True
    #                     elif forced is not None:
    #                         idx = int(forced["idx"])
    #                         #print(f"[debug-claim] add-kong | seat={seat} forced_idx={idx} pung_candidates={[m.tiles for m in p.melds if getattr(m, 'type', '')=='pung']}")
    #                         # For add-kan we treat any index (0/1) as "yes"; if you prefer multi-option indexing, enumerate upgradable pungs first.
    #                         ok = bool(forced["idx"])
    #                     else:
    #                         decider = getattr(policies[seat], "decide_add_kong", None)
    #                         ok = (decider is None) or decider(self, seat, tile)

    #                     if ok:
    #                         upgrade = (idx, tile); break
    #         #print(f"[debug-claim] add-kong | upgrade={upgrade} changed={changed}")
    #         if upgrade is None:
    #             return changed
    #         idx, tile = upgrade
    #         p.concealed.remove(tile)
    #         p.melds[idx] = _make_meld("kong", (tile, tile, tile, tile), open=True, formed_by_claim=True, from_discard=False)
    #         self.stats["kong_open"] += 1
    #         self.claim_log.append({"who": seat, "type": "kong(open,added)", "on": tile})
    #         if self._after_kong_draw_and_maybe_tsumo(seat, policies):
    #             return True
    #         changed = True
            #print(f"[confirm-kong] ADD-KONG upgrade seat={seat} tile={tile} melds={self.players[seat].melds}")

    def _maybe_upgrade_recent_pung_to_kong(self, seat: int, tile: str, policies) -> bool:
        """
        Immediately after forming a pung from a discard, optionally upgrade it
        to an open kong (add-kan).

        We expose THIS tile as the only candidate in _pending_kong_candidates[seat],
        so the RL 'kong' head sees legal indices [0, 1]:

        0 → pass
        1 → upgrade this pung to a kong
        """
        p = self.players[seat]

        for idx, m in enumerate(p.melds):
            kind  = _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None)))
            tiles = list(getattr(m, "tiles", []))

            # Only consider the pung we just formed: (tile,tile,tile)
            if not (kind == "pung"
                    and len(tiles) == 3
                    and tiles[0] == tiles[1] == tiles[2] == tile):
                continue

            # Need one extra copy in concealed to actually make a kong
            if p.concealed.count(tile) < 1:
                continue

            ok = False

            # ---- open kong-decision window for this seat ----
            self._pending_kong_candidates[seat] = [tile]
            try:
                if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
                    ok = True
                else:
                    decider = getattr(policies[seat], "decide_add_kong", None)
                    # RLPolicy.decide_add_kong will call _decide_index("kong"),
                    # which now sees legal=[0,1] from get_legal_kong_indices.
                    ok = (decider is None) or decider(self, seat, tile)
            finally:
                # ---- always close the window so we don't leak candidates ----
                self._pending_kong_candidates.pop(seat, None)

            if not ok:
                # Player declined the upgrade; try other melds (in practice there
                # should only be this one "recent" pung, but this is safe).
                continue

            # Actually perform the upgrade
            p.concealed.remove(tile)
            p.melds[idx] = _make_meld(
                "kong",
                (tile, tile, tile, tile),
                open=True,
                formed_by_claim=True,
                from_discard=False,
            )
            self.stats["kong_open"] += 1
            self.claim_log.append({"who": seat, "type": "kong(open,added)", "on": tile})
            self.turn = seat

            if self._after_kong_draw_and_maybe_tsumo(seat, policies):
                return True

            return True

        return False


    # ---- claims on discard

    def check_hand_invariants(self, tag: str = "") -> None:
        """
        Sanity-check hand sizes across all seats.

        Supports multiple kongs and add-kongs correctly.

        For each seat:
        C          = # non-flower concealed tiles
        meld_tiles = total # tiles stored inside melds
        extra_kong = sum(max(len(tiles) - 3, 0) for each meld)

        Then:
        non_flower_raw  = C + meld_tiles
        non_flower_norm = non_flower_raw - extra_kong

        Expected invariant (normalized):
        - Every seat should have 13 or 14 effective tiles.
        - 14 is allowed for the current draw seat and for any seat that has just
            performed one or more kong replacements (since turn may lag).
        """
        #print("[hand-ids]", [id(p.concealed) for p in self.players])
        for s, p in enumerate(self.players):
            concealed_nf = [
                t for t in getattr(p, "concealed", [])
                if not is_flower(t)
            ]
            C = len(concealed_nf)

            meld_tiles = 0
            extra_kong = 0
            num_kongs = 0
            for m in getattr(p, "melds", []):
                tiles = getattr(m, "tiles", None)
                if tiles is None:
                    tiles = m
                L = len(tiles)
                meld_tiles += L
                if L > 3:
                    extra_kong += (L - 3)
                    num_kongs += 1

            flowers = len(getattr(p, "flowers", []))
            non_flower_raw = C + meld_tiles
            non_flower_norm = non_flower_raw - extra_kong
            draw_seat = getattr(self, "turn", None)

            def _dump_bug(reason: str) -> None:
                concealed_all = list(getattr(p, "concealed", []))
                print(
                    f"[tile-bug] {reason} tag={tag} seat={s} "
                    f"raw={non_flower_raw} norm={non_flower_norm} "
                    f"C={C} meld_tiles={meld_tiles} extra_kong={extra_kong} "
                    f"num_kongs={num_kongs} flowers={flowers} draw_seat={draw_seat}"
                )
                print("  concealed =", concealed_all)
                print("  melds = [", end="")
                for mm in getattr(p, "melds", []):
                    tiles = getattr(mm, "tiles", None)
                    if tiles is None:
                        tiles = mm
                    print(f"{list(tiles)}", end=", ")
                print("]")

            # --- adaptive invariant ---
            # Allow 13–14 normalized tiles normally,
            # and up to (13 + num_kongs) if multiple kongs were just formed.
            upper_bound = 14 if num_kongs == 0 else 14 + (num_kongs - 1)

            if non_flower_norm < 13 or non_flower_norm > upper_bound:
                _dump_bug("hard-invalid")
                # import traceback
                # traceback.print_stack()
                # print("⚠️ [INVARIANT WARNING] continuing despite violation")
                # return
                raise AssertionError(
                    f"hand size invariant broken (norm={non_flower_norm}, "
                    f"allowed up to {upper_bound})"
                )

            # Optional soft diagnostic: non-turn seat with 14 normalized tiles.
            if draw_seat is not None and s != draw_seat and non_flower_norm > 13:
                _dump_bug("soft-14-nonturn")






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
        winners = []
        if self.terminal.get("winner") is not None:
            winners = [self.terminal["winner"]]
        elif "winners" in self.terminal:
            winners = [w["seat"] for w in self.terminal["winners"]]

        # if 0 in winners:
        #     print(f"[win-debug] seat0 RON pts={pts} (multi={len(winners) > 1})")
        return True

    def _pung_claims(self, discarder: int, tile: str, policies) -> Optional[int]:
        forced_bin = None  # for trailing debug if nothing fires
        last_candidate_seat = None
        #print(f"[pre-claim-check] turn={self.turn} discarder={discarder} concealed_len={[len(p.concealed) for p in self.players]}")
        #self.check_hand_invariants(tag="before-pung-claims")
        for k in (1, 2, 3):
            s = (discarder + k) % 4
            last_candidate_seat = s

            if _meld_count(self.players[s]) >= 4:
                continue

            cnt = Counter(self.players[s].concealed)
            if cnt[tile] < 2:
                continue

            # We know seat s *could* pung this tile
            take = False
            src = "none"

            if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
                take = True
                src = "FORCED_ALWAYS"
            else:
                # First, check for a forced decision from RL/teacher
                forced_bin = self._pop_forced(s, "pung") or self._pop_forced(s, None)
                if forced_bin is not None:
                    idx = int(forced_bin["idx"])
                    take = bool(idx)  # 0 = no, 1 = yes
                    src = f"forced(idx={idx})"
                else:
                    policy = policies[s]
                    if hasattr(policy, "decide_pung"):
                        # --- DEBUG: which policy is deciding this pung? ---
                        # print(
                        #     f"[DEBUG-CLAIM] pung | seat={s} "
                        #     f"policy={type(policy).__name__} tile={tile}"
                        # )
                        take = policy.decide_pung(self, s, tile)
                        src = f"policy({type(policy).__name__})"
                    else:
                        take = False
                        src = "default_false"

            # Optional: debug the final decision
            # print(f"[DEBUG-CLAIM] pung | seat={s} src={src} take={take}")

            if not take:
                continue
            # Invariant just before mutating for a pung
            #self.check_hand_invariants(tag="before-form-pung")

            self._form_pung(s, tile)
            if self._maybe_upgrade_recent_pung_to_kong(s, tile, policies):
                # Invariant after possible upgrade to kong
                #self.check_hand_invariants(tag="after-pung->kong")
                return s
            # --- DEBUG: actually taking the pung ---
            # print(
            #     f"[DEBUG-CLAIM] pung TAKE | seat={s} tile={tile} "
            #     f"policy={src}"
            # )
            self.players[s].needs_discard = True
            self.claim_log.append({"who": s, "type": "pung", "on": tile, "from": discarder})
            self.turn = s

            #self.check_hand_invariants(tag="after-form-pung")
            return s

        # If we got here, nobody took the pung; last seat for debugging
        if last_candidate_seat is not None:
            s = last_candidate_seat
            cand_tiles = [t for t, c in Counter(self.players[s].concealed).items() if c >= 2]
            # print(f"[DEBUG-CLAIM] pung | seat={s} candidates={cand_tiles} tile={tile}")

        return None

    # def _pung_claims(self, discarder: int, tile: str, policies) -> Optional[int]:
    #     forced_bin = None  # for trailing debug if nothing fires
    #     last_candidate_seat = None

    #     for k in (1, 2, 3):
    #         s = (discarder + k) % 4
    #         last_candidate_seat = s

    #         # --- NEW: use virtual meld context, not just raw _meld_count ---
    #         real_declared, eff_declared, virt_tiles = _get_virtual_meld_context(self, s)

    #         # Do not exceed 4 meld “slots” counting virtuals as well.
    #         if eff_declared >= 4:
    #             # print(f"[DEBUG-CLAIM] pung | seat={s} skip: eff_declared={eff_declared}")
    #             continue

    #         # If this tile is already part of a virtual meld, avoid opening it.
    #         # (This is the “don’t declare what’s already virtual” rule.)
    #         if tile in virt_tiles:
    #             # print(f"[DEBUG-CLAIM] pung | seat={s} skip: tile {tile} in virt_tiles")
    #             continue

    #         cnt = Counter(self.players[s].concealed)
    #         if cnt[tile] < 2:
    #             continue

    #         # We know seat s *could* pung this tile
    #         take = False
    #         src = "none"

    #         if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
    #             take = True
    #             src = "FORCED_ALWAYS"
    #         else:
    #             # First, check for a forced decision from RL/teacher
    #             forced_bin = self._pop_forced(s, "pung") or self._pop_forced(s, None)
    #             if forced_bin is not None:
    #                 idx = int(forced_bin["idx"])
    #                 take = bool(idx)  # 0 = no, 1 = yes
    #                 src = f"forced(idx={idx})"
    #             else:
    #                 policy = policies[s]
    #                 if hasattr(policy, "decide_pung"):
    #                     print(
    #                         f"[DEBUG-CLAIM] pung | seat={s} "
    #                         f"policy={type(policy).__name__} tile={tile} "
    #                         f"real_decl={real_declared} eff_decl={eff_declared}"
    #                     )
    #                     take = policy.decide_pung(self, s, tile)
    #                     src = f"policy({type(policy).__name__})"
    #                 else:
    #                     take = False
    #                     src = "default_false"

    #         if not take:
    #             continue

    #         print(
    #             f"[DEBUG-CLAIM] pung TAKE | seat={s} tile={tile} "
    #             f"policy={src} real_decl={real_declared} eff_decl={eff_declared}"
    #         )

    #         self._form_pung(s, tile)
    #         if self._maybe_upgrade_recent_pung_to_kong(s, tile, policies):
    #             return s
    #         self.players[s].needs_discard = True
    #         self.claim_log.append({"who": s, "type": "pung", "on": tile, "from": discarder})
    #         self.turn = s
    #         return s

    #     if last_candidate_seat is not None:
    #         s = last_candidate_seat
    #         cand_tiles = [t for t, c in Counter(self.players[s].concealed).items() if c >= 2]
    #         # print(f"[DEBUG-CLAIM] pung | seat={s} candidates={cand_tiles}")
    #     return None

        # ---- RL helper: expose legal indices for chow/kong heads ----
    def get_legal_chow_indices(self, seat: int) -> List[int]:
        """
        For the current chow window for this seat, return legal class indices
        for the 4-way 'chow' head.

        Convention (C = 4):
          - index 0      → pass
          - index i>0    → take chow_sets[i-1], where chow_sets = self._pending_chow_sets[seat].
        """
        sets = self._pending_chow_sets.get(seat) or []
        if not sets:
            # no chow possible → only 'pass'
            return [0]

        k = min(len(sets), 3)  # at most 3 concrete chows; head dim is 4
        return list(range(0, k + 1))  # [0, 1, ..., k]

    def get_legal_kong_indices(self, seat: int) -> List[int]:
        """
        Legal indices for the 5-way 'kong' head.

        Convention (C = 5):
        - 0        → pass
        - i > 0    → take candidate[i-1],
                    where candidate list = self._pending_kong_candidates[seat].
        """
        candidates = self._pending_kong_candidates.get(seat) or []
        if not candidates:
            return [0]  # only pass

        k = min(len(candidates), 4)  # at most 4 tiles into 5-way head
        return list(range(0, k + 1))  # [0, 1, ..., k]



    # def _chow_claim(self, discarder: int, tile: str, policies) -> Optional[int]:
    #     s = (discarder + 1) % 4
    #     if _meld_count(self.players[s]) >= 4:
    #         return None
    #     if not _is_suit_tile(tile):
    #         return None

    #     r, suit = _tile_rank_suit(tile)
    #     sets = []
    #     for a, b in [(r-2, r-1), (r-1, r+1), (r+1, r+2)]:
    #         if 1 <= a <= 9 and 1 <= b <= 9:
    #             A, B = f"{a}{suit}", f"{b}{suit}"
    #             if A in self.players[s].concealed and B in self.players[s].concealed:
    #                 sets.append((A, B))
    #     if not sets:
    #         return None

    #     pick = None
    #     forced = self._pop_forced(s, "chow") or self._pop_forced(s, None)

    #     if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
    #         pick = sets[0]

    #     elif forced is not None:
    #         idx = int(forced["idx"])
    #         #print(f"[debug-claim] chow | seat={s} forced_idx={idx} sets={sets}")

    #         if idx <= 0 or idx > len(sets):
    #             # 0 or out-of-range -> pass
    #             #print(f"[debug-claim] chow | seat={s} forced_idx={idx} -> PASS")
    #             pick = None
    #         else:
    #             pick = sets[idx - 1]

    #     else:
    #         if hasattr(policies[s], "choose_chow"):
    #             pick = policies[s].choose_chow(self, s, tile, sets)

    #     if pick is None:
    #         return None

    #     self._form_chow(s, pick[0], tile, pick[1])
    #     self.players[s].needs_discard = True
    #     self.claim_log.append({
    #         "who": s, "type": "chow", "on": tile,
    #         "with": list(pick), "from": discarder
    #     })
    #     self.turn = s
    #     #print(f"[debug-claim] chow | sets={sets} pick={pick}")
    #     return s


    def _chow_claim(self, discarder: int, tile: str, policies) -> Optional[int]:
        s = (discarder + 1) % 4
        
        # Clear any stale chow context
        self._pending_chow_sets.pop(s, None)

        #self.check_hand_invariants(tag="before-chow-claim")

        if _meld_count(self.players[s]) >= 4:
            return None
        if not _is_suit_tile(tile):
            return None

        r, suit = _tile_rank_suit(tile)
        sets: List[Tuple[str, str]] = []
        for a, b in [(r-2, r-1), (r-1, r+1), (r+1, r+2)]:
            if 1 <= a <= 9 and 1 <= b <= 9:
                A, B = f"{a}{suit}", f"{b}{suit}"
                if A in self.players[s].concealed and B in self.players[s].concealed:
                    sets.append((A, B))

        #print(f"[chow-env] tile={tile} sets={sets} concealed={self.players[s].concealed}")
        
        if not sets:
            return None

        # Expose these for RLPolicy via get_legal_chow_indices
        self._pending_chow_sets[s] = sets
        # print(
        #     f"[chow-env] env={id(self)%10000} seat={s} discarder={discarder} "
        #     f"tile={tile} sets={sets} concealed_len={len(self.players[s].concealed)} "
        #     f"concealed={self.players[s].concealed}"
        # )
        self.debug_player_state(s, tag="chow-offer")
        #self.check_hand_invariants(tag="after-chow-offer")
        pick = None
        forced = self._pop_forced(s, "chow") or self._pop_forced(s, None)

        try:
            if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
                pick = sets[0]

            elif forced is not None:
                idx = int(forced["idx"])
                # idx = 0 → pass; idx in 1..len(sets) → take that chow
                if idx <= 0 or idx > len(sets):
                    pick = None
                else:
                    pick = sets[idx - 1]

            else:
                if hasattr(policies[s], "choose_chow"):
                    pick = policies[s].choose_chow(self, s, tile, sets)
        finally:
            # claim window is over; clear pending sets either way
            self._pending_chow_sets.pop(s, None)

        if pick is None:
            # No chow taken, but good to check we didn’t mutate anything weird
            #self.check_hand_invariants(tag="after-chow-none")
            return None

        # Just before we mutate
        #self.check_hand_invariants(tag="before-form-chow")
        #print(f"[chow-apply] seat={s} tile={tile} pick={pick} sets={sets}")
        self._form_chow(s, pick[0], tile, pick[1])
        self.players[s].needs_discard = True
        self.claim_log.append({
            "who": s, "type": "chow", "on": tile,
            "with": list(pick), "from": discarder
        })
        self.turn = s
        #self.check_hand_invariants(tag="after-form-chow")
        return s



    # ---- CLOSED KONGS (on your turn, before tsumo check)

    def _maybe_closed_kongs(self, seat: int, policies) -> bool:
        p = self.players[seat]
        while True:
            cnt = Counter([t for t in p.concealed if not is_flower(t)])
            candidates = [t for t, c in cnt.items() if c >= 4]

            # Clear any stale kong context for this seat
            self._pending_kong_candidates.pop(seat, None)

            if not candidates or _meld_count(p) >= 4:
                # No kong actually formed in this call
                #self.check_hand_invariants(tag="after-closed-kongs-none")
                return False

            # Expose current candidates to RL via get_legal_kong_indices
            self._pending_kong_candidates[seat] = candidates

            tile = None
            forced = self._pop_forced(seat, "kong") or self._pop_forced(seat, None)

            try:
                if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
                    tile = candidates[0]

                elif forced is not None:
                    idx = int(forced["idx"])
                    # idx = 0 → pass, idx in 1..len(candidates) → choose that tile
                    if idx == 0:
                        #self.check_hand_invariants(tag="after-closed-kongs-forced-pass")
                        return False
                    if 0 <= (idx - 1) < len(candidates):
                        tile = candidates[idx - 1]

                else:
                    if hasattr(policies[seat], "decide_closed_kong"):
                        tile = policies[seat].decide_closed_kong(self, seat, candidates)
            finally:
                # Kong decision window is done
                self._pending_kong_candidates.pop(seat, None)

            if tile is None:
                #self.check_hand_invariants(tag="after-closed-kongs-decide-none")
                return False

            self._form_closed_kong(seat, tile)
            self.claim_log.append({"who": seat, "type": "kong(closed)", "on": tile})
            self.turn = seat
            if self._after_kong_draw_and_maybe_tsumo(seat, policies):
                # Check after the kong flow including kong-draw + possible tsumo
                #self.check_hand_invariants(tag="after-closed-kong+draw")
                return True

            # After forming kong but no tsumo; loop may continue
            #self.check_hand_invariants(tag="after-form-closed-kong-loop")


    # def _maybe_closed_kongs(self, seat: int, policies) -> bool:
    #     p = self.players[seat]
    #     while True:
    #         cnt = Counter([t for t in p.concealed if not is_flower(t)])
    #         candidates = [t for t,c in cnt.items() if c >= 4]
            
    #         if not candidates or _meld_count(p) >= 4: return False
    #         tile = None
    #         forced = self._pop_forced(seat, "kong") or self._pop_forced(seat, None)
    #         if FORCE_CLAIMS or ALWAYS_CLAIM_DEBUG:
    #             tile = candidates[0]
    #         elif forced is not None:
    #             idx = int(forced["idx"])
    #             #print(f"[debug-claim] closed-kong | seat={seat} forced_idx={idx} candidates={candidates}")
    #             if idx == 0:
    #                 #print(f"[debug-claim] closed-kong | seat={seat} forced_idx={idx} -> PASS")
    #                 return False
    #             if 0 <= (idx-1) < len(candidates):
    #                 tile = candidates[idx-1]
    #         else:
    #             if hasattr(policies[seat], "decide_closed_kong"):
    #                 tile = policies[seat].decide_closed_kong(self, seat, candidates)
    #         if tile is None:
    #             return False
    #         self._form_closed_kong(seat, tile)
    #         self.claim_log.append({"who": seat, "type": "kong(closed)", "on": tile})
    #         #print(f"[debug-claim] closed-kong | candidates={candidates} tile={tile}")
    #         if self._after_kong_draw_and_maybe_tsumo(seat, policies): 
    #             return True
    
    def _infer_concealed_melds(self, seat: int):
        """
        Pure helper: infer 'virtual' melds that live entirely in concealed tiles
        but have NOT been declared as open melds.

        Returns a list of dicts:
          [{"type": "pung", "tiles": ("5w","5w","5w")},
           {"type": "chow", "tiles": ("3b","4b","5b")},
           ...]
        We do a simple greedy pass; it's fine if this is approximate,
        since it's just for heuristics/policy, not for rules.
        """
        p = self.players[seat]
        concealed = list(p.concealed)

        # ---- 1) Pungs in concealed ----
        cnt = Counter(concealed)
        virtual = []

        for tile, c in cnt.items():
            if c >= 3:
                # We don't mutate concealed here; this is just informational
                virtual.append({
                    "type": "pung",
                    "tiles": (tile, tile, tile),
                })

        # ---- 2) Chows in concealed (greedy by suit / rank) ----
        # Group suit tiles by suit, then treat ranks as small integers.
        suits = defaultdict(list)  # suit -> list of ranks (ints)
        for t in concealed:
            if _is_suit_tile(t):
                r, suit = _tile_rank_suit(t)  # you already have this helper
                suits[suit].append(r)

        for suit, ranks in suits.items():
            ranks.sort()
            # greedy: walk 1..9 and see what triples (r, r+1, r+2) we can form
            # using *at least* one of each rank in concealed
            used = Counter()
            for r in range(1, 8):  # r, r+1, r+2 inside 1..9
                a, b, c = r, r+1, r+2
                # check availability: count(rank) - used(rank) >= 1
                if (ranks.count(a) - used[a] >= 1 and
                    ranks.count(b) - used[b] >= 1 and
                    ranks.count(c) - used[c] >= 1):
                    # record one chow
                    tiles = (f"{a}{suit}", f"{b}{suit}", f"{c}{suit}")
                    virtual.append({
                        "type": "chow",
                        "tiles": tuple(sorted(tiles, key=lambda x:(x[1], int(x[0])))),
                    })
                    used[a] += 1
                    used[b] += 1
                    used[c] += 1

        return virtual
    
    def meld_summary(self, seat: int, include_virtual: bool = False):
        """
        Utility for policies / debugging.

        Returns dict like:
          {
            "real_melds": [...],
            "real_open": k1,
            "real_closed": k2,
            "virtual_melds": [...],   # only if include_virtual=True
          }
        """
        p = self.players[seat]
        real_melds = list(p.melds)

        real_open = 0
        for m in real_melds:
            is_open = bool(getattr(m, "open", True))
            if is_open:
                real_open += 1
        real_closed = len(real_melds) - real_open

        summary = {
            "real_melds": real_melds,
            "real_open": real_open,
            "real_closed": real_closed,
        }

        if include_virtual:
            summary["virtual_melds"] = self._infer_concealed_melds(seat)

        return summary

    
    # ---- main step

    def step_turn(self, policies) -> None:
        if self.terminal: return
        seat = self.turn
        #self.check_hand_invariants(tag="start-step-turn")
        # If we just claimed, we must discard immediately.
        if self.players[seat].needs_discard:
            opts = self.legal_discards(seat)
            tile = None

            # Forced discard (maps 0..33 class → actual tile in hand)
            f = self._pop_forced(seat, "discard") or self._pop_forced(seat, None)
            if f is not None:
                cls = int(f["idx"])
                print(
                    f"[env-discard] forced seat={seat} cls={cls} "
                    f"opts={[(c, tile_to_class(c)) for c in opts]}"
                )
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
            
            # print(
            #     f"[env-discard-final] seat={seat} tile={tile} "
            #     f"cls={tile_to_class(tile) if tile else None}"
            # )

            if tile is None:
                return
            #print(f"[discard-intent] seat={seat} tile={tile} concealed_before={self.players[seat].concealed}")
            #self.check_hand_invariants(tag="before-discard-remove")
            try:
                self.players[seat].concealed.remove(tile)
            except ValueError:
                #print(f"[discard-error] seat={seat} tried to remove tile={tile} but not in concealed={self.players[seat].concealed}")
                raise
            #print(f"[discard-after] seat={seat} concealed_after={self.players[seat].concealed}")
            #self.check_hand_invariants(tag="after-discard-remove")
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
        #self.check_hand_invariants(tag="after-kongs-and-tsumo-check")
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
        
        #print(f"[discard-intent] seat={seat} tile={tile} concealed_before={self.players[seat].concealed}")
        #self.check_hand_invariants(tag="before-discard-remove")
        try:
            self.players[seat].concealed.remove(tile)
        except ValueError:
            #print(f"[discard-error] seat={seat} tried to remove tile={tile} but not in concealed={self.players[seat].concealed}")
            raise
        #print(f"[discard-after] seat={seat} concealed_after={self.players[seat].concealed}")
        #self.check_hand_invariants(tag="after-discard-remove")
        self.discards[seat].append(tile)
        self.discard_history.append((seat, tile))
        self.last_discard = (seat, tile)

        self._maybe_apply_four_same_discard_penalty()
        #self.check_hand_invariants(tag="after-four-same-penalty")

        if self._ron_window(seat, tile, policies): return
        #self.check_hand_invariants(tag="after-ron-window")
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
            if kind not in (None, "discard", "binary", "chow", "kong","pung"):
                raise ValueError(f"Unknown forced kind: {kind}")
        else:
            raise ValueError("force_action expects 1, 2, or 3 args")

        if seat < 0 or seat > 3:
            raise ValueError("seat must be 0..3")
        self._forced[seat] = {"kind": kind, "idx": idx}

    # def _peek_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
    #     f = self._forced.get(seat)
    #     if not f:
    #         return None
    #     return f if (f["kind"] is None or kind is None or f["kind"] == kind) else None
    def _ensure_forced_dict(self):
        """
        Ensure self._forced is a dict. If it's missing or some weird value,
        reinitialize it (this also fixes older runs where it might have been 0, etc.).
        """
        if not hasattr(self, "_forced") or not isinstance(self._forced, dict):
            self._forced = {}
            
    def _peek_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
        self._ensure_forced_dict()
        f = self._forced.get(seat)
        if not f:
            return None

        f_kind = f.get("kind")

        # If this is a *generic* forced action (kind=None in the entry),
        # allow it to satisfy any request.
        if f_kind is None:
            return f

        # If the caller is asking for a generic action (kind=None),
        # do NOT steal a typed forced action like "chow"/"pung"/"kong"/"binary".
        if kind is None:
            return None

        # Otherwise, require exact match of kind.
        return f if f_kind == kind else None


    # def _peek_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
    #     self._ensure_forced_dict()
    #     f = self._forced.get(seat)
    #     if not f:
    #         return None
    #     f_kind = f.get("kind")
    #     return f if (f_kind is None or kind is None or f_kind == kind) else None

    # def _pop_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
    #     self._ensure_forced_dict()
    #     f = self._peek_forced(seat, kind)
    #     if f is not None:
    #         # f is a dict, so even if idx == 0 this is still truthy and we will pop.
    #         self._forced.pop(seat, None)
    #     return f
    
    def _pop_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
        self._ensure_forced_dict()
        f = self._peek_forced(seat, kind)
        if f is not None:
            print(f"[forced-pop] seat={seat} kind={kind} got={f}")
            self._forced.pop(seat, None)
        return f


    # def _peek_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
    #     f = self._forced.get(seat)
    #     if not f:
    #         return None

    #     f_kind = f.get("kind", None)

    #     # Only match if:
    #     #   - kinds are exactly equal, OR
    #     #   - both are None (generic forced action)
    #     if f_kind == kind or (f_kind is None and kind is None):
    #         return f
    #     return None


    # # def _pop_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
    # #     f = self._peek_forced(seat, kind)
    # #     if f:
    # #         self._forced.pop(seat, None)
    # #         return f
    # #     return None

    # def _pop_forced(self, seat: int, kind: Optional[str]) -> Optional[Dict[str, Any]]:
    #     f = self._peek_forced(seat, kind)
    #     if f is None:
    #         return None
    #     # remove the entry for this seat
    #     self._forced.pop(seat, None)
    #     return f



# ---------------------------- Policies ----------------------------

def _declared_meld_count(p) -> int:
    return sum(
        1
        for m in p.melds
        if _normalize_meld_type(
            getattr(m, "type", getattr(m, "kind", None))
        ) in {"chow", "pung", "kong"}
    )

def _meld_count(
    p,
    env: "Env" = None,
    seat: int = None,
    include_virtual: bool = False,
) -> int:
    """
    Count melds for a player.

    - Default: counts only *declared* melds from p.melds (same as old behavior).
    - If include_virtual=True and env/seat are provided:
        uses effective_declared = real_declared + #virtual_melds (clipped at 4)
        from _get_virtual_meld_context.
    """
    real = _declared_meld_count(p)

    if include_virtual and env is not None and seat is not None:
        real_declared, effective_declared, _ = _get_virtual_meld_context(env, seat)

        # Sanity: if for some reason effective_declared comes back smaller,
        # fall back to the real count.
        if effective_declared < real_declared:
            return real

        return effective_declared

    return real


# def _meld_count(p) -> int:
#     return sum(1 for m in p.melds if _normalize_meld_type(getattr(m, "type", getattr(m, "kind", None))) in {"chow","pung","kong"})

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

def _get_virtual_meld_context(env: Env, seat: int):
    """
    Convenience helper for policies.

    Returns:
      real_declared      = number of *actual* melds in p.melds
      effective_declared = real_declared + (#virtual melds), clipped at 4
      virt_tiles         = set of tiles that participate in any virtual meld
    """
    summary = env.meld_summary(seat, include_virtual=True)
    real_declared = len(summary["real_melds"])
    virtual_melds = summary.get("virtual_melds", [])

    virt_tiles = set()
    for vm in virtual_melds:
        virt_tiles.update(vm["tiles"])

    effective_declared = min(4, real_declared + len(virtual_melds))
    return real_declared, effective_declared, virt_tiles


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
        declared = _meld_count(env.players[self.seat], env, self.seat, include_virtual=True)
        pairs = _pairs_in(hand)
        stage = max(0.3, min(1.0, len(env.discard_history) / 42.0))

        # Once we have any declared meld, chiitoi path should be dead.
        declared_factor = 0.0 if declared > 0 else 1.0
        pair_factor = min(1.0, 0.12 * pairs + 0.10)

        return base * stage * declared_factor * pair_factor


class RandomPolicy(BasePolicy):
    def pick_discard(self, env: Env) -> str:
        opts = env.legal_discards(self.seat)
        if not opts:
            return env.players[self.seat].concealed[0]

        p = env.players[self.seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        cnt = Counter(hand)

        # NEW: don't treat tiles that sit inside virtual melds as isolates
        _, _, virt_tiles = _get_virtual_meld_context(env, self.seat)

        isolates = [
            t for t in opts
            if _tile_is_isolate(cnt, t) and t not in virt_tiles
        ]
        return random.choice(isolates or opts)
    # def pick_discard(self, env: Env) -> str:
    #     opts = env.legal_discards(self.seat)
    #     if not opts: return env.players[self.seat].concealed[0]
    #     p = env.players[self.seat]
    #     hand = [t for t in p.concealed if not is_flower(t)]
    #     cnt = Counter(hand)
    #     isolates = [t for t in opts if _tile_is_isolate(cnt, t)]
    #     return random.choice(isolates or opts)
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
        for k in (1, 2, 3):
            opp = (self.seat + k) % 4
            hs = env.hand_state_for(opp, tile, source="discard")
            try:
                sb = score_hand(hs, env.rules)
                loss += _normalize_points_verbose(sb, hs=hs, rules=env.rules)
            except Exception:
                pass
        return self._b("danger_weight") * loss

    def decide_ron(self, env, tile, points, loser):
        return True

    def decide_open_kong(self, env: Env, seat: int, tile: str) -> bool:
        if ALWAYS_CLAIM_DEBUG:
            return True
        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        if hand.count(tile) < 3:
            return False
        declared = _meld_count(p, env, self.seat, include_virtual=True)
        base = self._b("open_kong_bias") + (0.2 if declared >= 1 else 0.0) + 0.25 * self._ol()
        return random.random() < min(0.98, base)

    def decide_add_kong(self, env: Env, seat: int, tile: str) -> bool:
        if ALWAYS_CLAIM_DEBUG:
            return True
        base = self._b("open_kong_bias") + 0.25 * self._ol()
        return random.random() < min(0.98, base)
    
    def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
        if ALWAYS_CLAIM_DEBUG:
            return True

        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        if hand.count(tile) < 2:
            return False

        # NEW: incorporate virtual melds into the "declared_melds" notion
        real_declared, declared_eff, _ = _get_virtual_meld_context(env, seat)

        w = self._effective_w(env, hand)

        # BEFORE pung: we already have declared_eff melds' worth of structure
        before = composite_shape_metric(hand, w, declared_melds=declared_eff,env=env, seat=seat)

        # AFTER pung: remove two tiles and bump effective declared meld count by 1
        h2 = hand[:]
        c = 0
        for t2 in list(h2):
            if t2 == tile and c < 2:
                h2.remove(t2)
                c += 1
        after = composite_shape_metric(
            h2, w, declared_melds=min(4, declared_eff + 1), env=env, seat=self.seat
        )

        margin = self._b("pung_margin_base") - min(declared_eff, 3) * self._b("margin_decay_per_meld")
        margin += 0.8 * self._ol()  # widen with open_lean

        if after <= before + margin:
            return True

        # Soft-margin override: allow near-neutral pungs occasionally
        soft_extra = self._ol() * (0.6 + 0.4 * self._b("pung_bias"))
        if after <= before + margin + soft_extra:
            prob = min(
                0.98,
                self._b("pung_bias") * (0.75 + 0.25 * self._b("meld_target_bonus"))
            )
            return random.random() < prob
        return False

    # def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
    #     if ALWAYS_CLAIM_DEBUG:
    #         return True
    #     p = env.players[seat]
    #     hand = [t for t in p.concealed if not is_flower(t)]
    #     if hand.count(tile) < 2:
    #         return False

    #     declared = _meld_count(p)
    #     w = self._effective_w(env, hand)

    #     # BEFORE pung: current declared melds
    #     before = composite_shape_metric(hand, w, declared_melds=declared)

    #     # AFTER pung: those 2 tiles leave hand, and we gain 1 declared meld
    #     h2 = hand[:]
    #     c = 0
    #     for t2 in list(h2):
    #         if t2 == tile and c < 2:
    #             h2.remove(t2)
    #             c += 1
    #     after = composite_shape_metric(h2, w, declared_melds=declared + 1)

    #     margin = self._b("pung_margin_base") - min(declared, 3) * self._b("margin_decay_per_meld")
    #     margin += 0.8 * self._ol()  # widen with open_lean

    #     if after <= before + margin:
    #         return True

    #     # Soft-margin override: allow near-neutral pungs occasionally
    #     soft_extra = self._ol() * (0.6 + 0.4 * self._b("pung_bias"))
    #     if after <= before + margin + soft_extra:
    #         prob = min(0.98, self._b("pung_bias") * (0.75 + 0.25 * self._b("meld_target_bonus")))
    #         return random.random() < prob
    #     return False

    def choose_chow(
        self,
        env: Env,
        seat: int,
        tile: str,
        chow_sets: List[Tuple[str, str]]
    ) -> Optional[Tuple[str, str]]:
        if not chow_sets:
            return None
        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]

        # NEW: effective declared meld count (real + virtual)
        real_declared, declared_eff, _ = _get_virtual_meld_context(env, seat)

        w = self._effective_w(env, hand)

        # BEFORE chow
        before = composite_shape_metric(hand, w, declared_melds=declared_eff,env=env, seat=seat)

        # AFTER chow: remove two tiles, +1 meld
        best = None
        best_score = 1e9
        for a, b in chow_sets:
            h2 = hand[:]
            if a in h2:
                h2.remove(a)
            if b in h2:
                h2.remove(b)
            score = composite_shape_metric(
                h2, w, declared_melds=min(4, declared_eff + 1),env=env, seat=self.seat
            )
            if score < best_score:
                best, best_score = (a, b), score

        margin = self._b("chow_margin_base") - min(declared_eff, 3) * self._b("margin_decay_per_meld")
        margin += 1.1 * self._ol()

        if best_score <= before + margin:
            return best

        # Soft-margin override for chow
        soft_extra = self._ol() * (0.5 + 0.5 * self._b("chow_bias"))
        if best_score <= before + margin + soft_extra:
            accept_prob = min(
                0.98,
                self._b("chow_bias")
                * (0.85 + 0.15 * self._b("meld_target_bonus"))
                * (1.0 + 0.35 * self._ol())
            )
            if random.random() < accept_prob:
                return best
        return None


    # def choose_chow(
    #     self,
    #     env: Env,
    #     seat: int,
    #     tile: str,
    #     chow_sets: List[Tuple[str, str]]
    # ) -> Optional[Tuple[str, str]]:
    #     if not chow_sets:
    #         return None
    #     p = env.players[seat]
    #     hand = [t for t in p.concealed if not is_flower(t)]
    #     declared = _meld_count(p)
    #     w = self._effective_w(env, hand)

    #     # BEFORE chow
    #     before = composite_shape_metric(hand, w, declared_melds=declared)

    #     # AFTER chow: 2 tiles leave hand, +1 declared meld
    #     best = None
    #     best_score = 1e9
    #     for a, b in chow_sets:
    #         h2 = hand[:]
    #         if a in h2:
    #             h2.remove(a)
    #         if b in h2:
    #             h2.remove(b)
    #         score = composite_shape_metric(h2, w, declared_melds=declared + 1)
    #         if score < best_score:
    #             best, best_score = (a, b), score

    #     margin = self._b("chow_margin_base") - min(declared, 3) * self._b("margin_decay_per_meld")
    #     margin += 1.1 * self._ol()

    #     if best_score <= before + margin:
    #         return best

    #     # Soft-margin override for chow
    #     soft_extra = self._ol() * (0.5 + 0.5 * self._b("chow_bias"))
    #     if best_score <= before + margin + soft_extra:
    #         accept_prob = min(
    #             0.98,
    #             self._b("chow_bias")
    #             * (0.85 + 0.15 * self._b("meld_target_bonus"))
    #             * (1.0 + 0.35 * self._ol())
    #         )
    #         if random.random() < accept_prob:
    #             return best
    #     return None

    def decide_closed_kong(self, env: Env, seat: int, candidates: List[str]) -> Optional[str]:
        if not candidates:
            return None
        if ALWAYS_CLAIM_DEBUG:
            return candidates[0]
        base = self._b("closed_kong_bias")
        return random.choice(candidates) if (random.random() < base) else None
    
    def pick_discard(self, env: Env) -> str:
        p = env.players[self.seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        cnt = Counter(hand)
        opts = env.legal_discards(self.seat)
        if not opts:
            return p.concealed[0]

        # NEW: effective declared & virtual tiles
        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, self.seat)

        w = self._effective_w(env, hand)

        best = None
        best_val = 1e9
        for t in opts:
            h2 = hand[:]
            if t in h2:
                h2.remove(t)

            keep_val = self._b("keep_shape_weight") * composite_shape_metric(
                h2, w, declared_melds=declared_eff,env=env, seat=self.seat
            )
            danger = self._ev_danger(env, t)

            # NEW: don't penalize tiles that sit inside virtual melds as "isolates"
            isolate_bonus = -0.5 if (_tile_is_isolate(cnt, t) and t not in virt_tiles) else 0.0

            val = keep_val + danger + isolate_bonus
            if val < best_val:
                best, best_val = t, val
        return best


    # def pick_discard(self, env: Env) -> str:
    #     p = env.players[self.seat]
    #     hand = [t for t in p.concealed if not is_flower(t)]
    #     cnt = Counter(hand)
    #     opts = env.legal_discards(self.seat)
    #     if not opts:
    #         return p.concealed[0]
    #     declared = _meld_count(p)
    #     w = self._effective_w(env, hand)

    #     best = None
    #     best_val = 1e9
    #     for t in opts:
    #         h2 = hand[:]
    #         if t in h2:
    #             h2.remove(t)

    #         keep_val = self._b("keep_shape_weight") * composite_shape_metric(
    #             h2, w, declared_melds=declared
    #         )
    #         danger = self._ev_danger(env, t)
    #         isolate_bonus = -0.5 if _tile_is_isolate(cnt, t) else 0.0
    #         val = keep_val + danger + isolate_bonus
    #         if val < best_val:
    #             best, best_val = t, val
    #     return best

# class WinProbPolicy(BasePolicy):
#     """
#     Meld-friendly, guided by tuner. Uses *learned + context-gated* chiitoi weight,
#     adaptive acceptance margins, and a soft-margin override to claim more often.
#     """
#     def _ev_danger(self, env: Env, tile: str) -> float:
#         loss = 0.0
#         for k in (1,2,3):
#             opp = (self.seat + k) % 4
#             hs = env.hand_state_for(opp, tile, source="discard")
#             try:
#                 sb = score_hand(hs, env.rules)
#                 loss += _normalize_points_verbose(sb, hs=hs, rules=env.rules)
#             except Exception:
#                 pass
#         return self._b("danger_weight") * loss

#     def decide_ron(self, env, tile, points, loser): return True

#     def decide_open_kong(self, env: Env, seat: int, tile: str) -> bool:
#         if ALWAYS_CLAIM_DEBUG: return True
#         p = env.players[seat]
#         hand = [t for t in p.concealed if not is_flower(t)]
#         if hand.count(tile) < 3: return False
#         declared = _meld_count(p)
#         base = self._b("open_kong_bias") + (0.2 if declared >= 1 else 0.0) + 0.25*self._ol()
#         return random.random() < min(0.98, base)

#     def decide_add_kong(self, env: Env, seat: int, tile: str) -> bool:
#         if ALWAYS_CLAIM_DEBUG: return True
#         base = self._b("open_kong_bias") + 0.25*self._ol()
#         return random.random() < min(0.98, base)

#     def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
#         if ALWAYS_CLAIM_DEBUG: return True
#         p = env.players[seat]
#         hand = [t for t in p.concealed if not is_flower(t)]
#         if hand.count(tile) < 2: return False

#         w = self._effective_w(env, hand)
#         before = composite_shape_metric(hand, w)

#         h2 = hand[:]; c = 0
#         for t2 in list(h2):
#             if t2 == tile and c < 2:
#                 h2.remove(t2); c += 1

#         after = composite_shape_metric(h2, w)
#         declared = _meld_count(p)
#         margin = self._b("pung_margin_base") - min(declared, 3) * self._b("margin_decay_per_meld")
#         margin += 0.8 * self._ol()  # ← open-lean widens willingness

#         if after <= before + margin:
#             return True

#         # Soft-margin override: allow near-neutral pungs occasionally
#         soft_extra = self._ol() * (0.6 + 0.4*self._b("pung_bias"))
#         if after <= before + margin + soft_extra:
#             prob = min(0.98, self._b("pung_bias") * (0.75 + 0.25*self._b("meld_target_bonus")))
#             return random.random() < prob
#         return False

#     def choose_chow(self, env: Env, seat: int, tile: str, chow_sets: List[Tuple[str,str]]) -> Optional[Tuple[str,str]]:
#         if not chow_sets: return None
#         p = env.players[seat]
#         hand = [t for t in p.concealed if not is_flower(t)]
#         w = self._effective_w(env, hand)
#         before = composite_shape_metric(hand, w)

#         best = None; best_score = 1e9
#         for a,b in chow_sets:
#             h2 = hand[:]
#             if a in h2: h2.remove(a)
#             if b in h2: h2.remove(b)
#             score = composite_shape_metric(h2, w)
#             if score < best_score: best, best_score = (a,b), score

#         declared = _meld_count(p)
#         margin = self._b("chow_margin_base") - min(declared, 3) * self._b("margin_decay_per_meld")
#         margin += 1.1 * self._ol()

#         if best_score <= before + margin:
#             return best

#         # Soft-margin override for chow
#         soft_extra = self._ol() * (0.5 + 0.5*self._b("chow_bias"))
#         if best_score <= before + margin + soft_extra:
#             accept_prob = min(0.98, self._b("chow_bias") * (0.85 + 0.15*self._b("meld_target_bonus")) * (1.0 + 0.35*self._ol()))
#             if random.random() < accept_prob:
#                 return best
#         return None

#     def decide_closed_kong(self, env: Env, seat: int, candidates: List[str]) -> Optional[str]:
#         if not candidates: return None
#         if ALWAYS_CLAIM_DEBUG: return candidates[0]
#         base = self._b("closed_kong_bias")
#         return random.choice(candidates) if (random.random() < base) else None

#     def pick_discard(self, env: Env) -> str:
#         p = env.players[self.seat]
#         hand = [t for t in p.concealed if not is_flower(t)]
#         cnt = Counter(hand)
#         opts = env.legal_discards(self.seat)
#         if not opts: return p.concealed[0]
#         w = self._effective_w(env, hand)
#         best = None; best_val = 1e9
#         for t in opts:
#             h2 = hand[:]
#             if t in h2: h2.remove(t)
#             keep_val = self._b("keep_shape_weight") * composite_shape_metric(h2, w)
#             danger = self._ev_danger(env, t)
#             isolate_bonus = -0.5 if _tile_is_isolate(cnt, t) else 0.0
#             val = keep_val + danger + isolate_bonus
#             if val < best_val: best, best_val = t, val
#         return best

class PayoutOptPolicy(WinProbPolicy):
    def _expected_feed_loss(self, env: Env, tile: str) -> float:
        loss = 0.0
        for k in (1, 2, 3):
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
        if not opts:
            return p.concealed[0]

        declared = _meld_count(p, env, self.seat, include_virtual=True)
        w = self._effective_w(env, hand)

        best = None
        best_val = 1e9
        for t in opts:
            h2 = hand[:]
            if t in h2:
                h2.remove(t)

            keep = self._b("keep_shape_weight") * composite_shape_metric(
                h2, w, declared_melds=declared,env=env, seat=self.seat
            )
            feed = self._expected_feed_loss(env, t)
            isolate_bonus = -0.5 if _tile_is_isolate(cnt, t) else 0.0
            val = keep + feed + isolate_bonus
            if val < best_val:
                best, best_val = t, val
        return best


# class PayoutOptPolicy(WinProbPolicy):
#     def _expected_feed_loss(self, env: Env, tile: str) -> float:
#         loss = 0.0
#         for k in (1,2,3):
#             opp = (self.seat + k) % 4
#             hs = env.hand_state_for(opp, tile, source="discard")
#             try:
#                 sb = score_hand(hs, env.rules)
#                 loss += _normalize_points_verbose(sb, hs=hs, rules=env.rules)
#             except Exception:
#                 pass
#         return 0.05 * loss

#     def pick_discard(self, env: Env) -> str:
#         p = env.players[self.seat]
#         hand = [t for t in p.concealed if not is_flower(t)]
#         cnt = Counter(hand)
#         opts = env.legal_discards(self.seat)
#         if not opts: return p.concealed[0]
#         w = self._effective_w(env, hand)
#         best = None; best_val = 1e9
#         for t in opts:
#             h2 = hand[:]
#             if t in h2: h2.remove(t)
#             keep = self._b("keep_shape_weight") * composite_shape_metric(h2, w)
#             feed = self._expected_feed_loss(env, t)
#             isolate_bonus = -0.5 if _tile_is_isolate(cnt, t) else 0.0
#             val = keep + feed + isolate_bonus
#             if val < best_val: best, best_val = t, val
#         return best
    
class AggroPolicy(WinProbPolicy):
    """
    Very aggressive policy:
      - Inherits all the virtual-meld-aware discard / EV logic from WinProbPolicy.
      - For claims (pung/chow/kongs), first uses WinProbPolicy's shape-based
        decision (which sees virtual melds), then falls back to a high-probability
        aggressive rule if WinProb says no.
    """

    def decide_open_kong(self, env: Env, seat: int, tile: str) -> bool:
        # Pure aggro: always upgrade to open kong if the Env offers it.
        return True

    def decide_add_kong(self, env: Env, seat: int, tile: str) -> bool:
        # Pure aggro: always add-kan if possible.
        return True

    def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
        # 1) Let WinProbPolicy (virtual-meld aware) decide first.
        if super().decide_pung(env, seat, tile):
            return True

        # 2) Aggro fallback: if we *can* pung, still often take it.
        p = env.players[seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        if hand.count(tile) < 2:
            return False

        if ALWAYS_CLAIM_DEBUG:
            return True

        # Old behavior: ~97% chance to take any available pung.
        return random.random() < 0.97

    def decide_closed_kong(self, env: Env, seat: int, candidates: List[str]) -> Optional[str]:
        # 1) Ask WinProbPolicy first (uses tuner + virtual meld context).
        pick = super().decide_closed_kong(env, seat, candidates)
        if pick:
            return pick

        if not candidates:
            return None

        if ALWAYS_CLAIM_DEBUG:
            return candidates[0]

        # 2) Aggro fallback: bump probability to take *some* closed kong.
        base = self._b("closed_kong_bias")
        prob = min(0.98, max(0.25, base) + 0.25)
        return random.choice(candidates) if (random.random() < prob) else None

    def choose_chow(
        self,
        env: Env,
        seat: int,
        tile: str,
        chow_sets: List[Tuple[str, str]]
    ) -> Optional[Tuple[str, str]]:
        if not chow_sets:
            return None

        # 1) Let WinProbPolicy pick the best chow using composite_shape_metric
        #    (now with declared_melds = real + virtual).
        best = super().choose_chow(env, seat, tile, chow_sets)
        if best is not None:
            return best

        # 2) Aggro fallback: still sometimes chow even if WinProb said no.
        if ALWAYS_CLAIM_DEBUG:
            return chow_sets[0]

        # High but not insane probability to take a chow anyway.
        prob = 0.85
        return chow_sets[0] if (random.random() < prob) else None


# class AggroPolicy(WinProbPolicy):
#     def decide_open_kong(self, env, seat, tile): return True
#     def decide_add_kong(self, env, seat, tile): return True
#     def decide_pung(self, env, seat, tile):
#         p = env.players[seat]
#         hand = [t for t in p.concealed if not is_flower(t)]
#         return hand.count(tile) >= 2 and (ALWAYS_CLAIM_DEBUG or random.random() < 0.97)
#     def decide_closed_kong(self, env, seat, candidates):
#         return candidates[0] if (candidates and (ALWAYS_CLAIM_DEBUG or random.random() < max(0.25, self._b("closed_kong_bias")))) else None
#     def choose_chow(self, env, seat, tile, chow_sets):
#         return chow_sets[0] if chow_sets else None
    
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
        declared = _meld_count(env.players[seat], env, self.seat, include_virtual=True)
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
        # First: WinProbPolicy decision (margin-based, context-aware).
        if self.wp.decide_pung(env, seat, tile):
            return True

        pview = env.players[seat]
        hand = [t for t in pview.concealed if not is_flower(t)]

        # Must actually be able to pung.
        if hand.count(tile) < 2:
            return False

        # Look at declared + virtual meld context.
        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, seat)

        # If this tile is already part of a virtual meld, strongly discourage opening it.
        # We want to *keep* strong concealed structure closed most of the time.
        if tile in virt_tiles:
            if random.random() < 0.85:
                return False

        base = self.tuner.get("pung_bias")
        p = max(0.55, base) + 0.15 + 0.10 * declared_eff + 0.30 * self._ol()

        # If we already have many melds (real + virtual), reduce eagerness to open more.
        if declared_eff >= 3:
            p *= 0.5

        p = min(0.98, max(0.0, p))
        return random.random() < p

    # def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
    #     # Try WP (margin-based).
    #     if self.wp.decide_pung(env, seat, tile):
    #         return True
    #     # Aggro fallback only if we truly can pung.
    #     pview = env.players[seat]
    #     hand = [t for t in pview.concealed if not is_flower(t)]
    #     if hand.count(tile) < 2:
    #         return False
    #     declared = _meld_count(pview, env, self.seat, include_virtual=True)
    #     base = self.tuner.get("pung_bias")
    #     # Stronger bump as we open more; incorporate open_lean.
    #     p = min(0.98, max(0.55, base) + 0.15 + 0.10 * declared + 0.30 * self._ol())
    #     return random.random() < p

    def choose_chow(
        self,
        env: Env,
        seat: int,
        tile: str,
        chow_sets: List[Tuple[str, str]]
    ) -> Optional[Tuple[str, str]]:
        # Ask WinProbPolicy to pick by its composite metric first.
        best = self.wp.choose_chow(env, seat, tile, chow_sets)
        if best is not None:
            return best
        if not chow_sets:
            return None

        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, seat)
        base = self.tuner.get("chow_bias")

        # Filter chow candidates that would break virtual melds.
        filtered: List[Tuple[str, str]] = []
        for cs in chow_sets:
            # If any tile in this chow is part of a virtual meld, usually skip it.
            if any(t in virt_tiles for t in cs):
                if random.random() < 0.85:
                    continue
            filtered.append(cs)

        if not filtered:
            return None

        p = max(0.45, base) + 0.12 + 0.08 * declared_eff + 0.35 * self._ol()

        # Again, if we already effectively have 3+ melds, be less eager to open.
        if declared_eff >= 3:
            p *= 0.5

        p = min(0.98, max(0.0, p))
        return filtered[0] if (random.random() < p) else None

    # def choose_chow(self, env: Env, seat: int, tile: str, chow_sets: List[Tuple[str,str]]) -> Optional[Tuple[str,str]]:
    #     # Ask WP to evaluate best chow by composite metric.
    #     best = self.wp.choose_chow(env, seat, tile, chow_sets)
    #     if best is not None:
    #         return best
    #     if not chow_sets:
    #         return None
    #     # Aggro fallback: still take a chow sometimes to push openings.
    #     base = self.tuner.get("chow_bias")
    #     declared = _meld_count(env.players[seat], env, self.seat, include_virtual=True)
    #     p = min(0.98, max(0.45, base) + 0.12 + 0.08 * declared + 0.35 * self._ol())
    #     return chow_sets[0] if (random.random() < p) else None

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

HONOR_WINDS   = {"E", "S", "W", "N"}
HONOR_DRAGONS = {"C", "F", "B"}
ALL_HONORS    = HONOR_WINDS | HONOR_DRAGONS

def _discard_priority_flex(tile: str, cnt: Counter) -> float:
    """
    Larger = worse tile to keep (higher discard priority).
    Winds (E,S,W,N) are worst, Dragons (C,F,B) somewhat better.
    Suited tiles: terminals 1/9 worst if isolated, 3–7 best.
    Synergy (pairs, neighbors, inside shapes) reduces the priority.
    """
    # Honors: winds > dragons
    if tile in ALL_HONORS:
        if tile in HONOR_WINDS:
            base = 5  # dump these first
        else:  # C,F,B
            base = 4   # a bit more valuable; dump after winds

        n = cnt.get(tile, 0)
        # Pairs/triples of honors are good, so reduce priority
        if n >= 2:
            base -= 3.5
        if n >= 3:
            base -= 1.0
        return base

    # Non-honor non-suit weirdness (shouldn’t really happen, but be safe)
    if not _is_suit_tile(tile):
        return 2.0

    # Suited tiles
    r, s = _tile_rank_suit(tile)

    # Base penalty by rank: terminals worst if isolated
    if r in (1, 9):
        base = 2.5
    elif r in (2, 8):
        base = 2.0
    else:  # 3–7
        base = 1.5

    # Synergy: duplicates & neighbors
    dup = max(0, cnt.get(tile, 0) - 1)
    neighbors = 0.0
    for d, w in [(-1, 1.0), (1, 1.0), (-2, 0.5), (2, 0.5)]:
        rr = r + d
        if 1 <= rr <= 9:
            nt = f"{rr}{s}"
            if cnt.get(nt, 0) > 0:
                neighbors += w

    inside_bonus = 0.0
    left = f"{r-1}{s}" if r - 1 >= 1 else None
    right = f"{r+1}{s}" if r + 1 <= 9 else None
    if left and right and cnt.get(left, 0) > 0 and cnt.get(right, 0) > 0:
        # e.g., tile is the "4" in 3-4-5
        inside_bonus += 1.0

    synergy = 0.8 * dup + 0.6 * neighbors + inside_bonus

    # High base, high synergy -> lower final priority (we want to keep it)
    return base - synergy

class FlexibleAggroPolicy(BasePolicy):
    """
    Aggressive but flexible:
      - Discard logic uses a flexible meld/pair heuristic (no hard 4-meld-first rule).
      - Claim logic is at least as aggressive as HybridAggroPolicy, with extra boosts.
      - Melds only end up open if they are formed from someone else's discard
        (as enforced by Env, not by this policy).
    """
    def __init__(self, seat: int, rules: Dict, tuner: Optional[AdaptiveTuner]):
        super().__init__(seat, rules, tuner)
        self.hy = HybridAggroPolicy(seat, rules, tuner)
        self.rng = random.Random()

        self.aggressiveness = 0.90
        self.risk_weight = 0.03
        self.keep_weight = 1.0

    # ---------- Discard logic (flexible meld/pair logic) ----------

    def pick_discard(self, env: Env) -> str:
        p = env.players[self.seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        if not hand:
            return p.concealed[0]

        opts = env.legal_discards(self.seat)
        if not opts:
            return p.concealed[0]

        cnt = Counter(hand)
        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, self.seat)
        w = self._effective_w(env, hand)

        best_tile, best_val = None, float("inf")
        for t in opts:
            h2 = hand[:]
            if t in h2:
                h2.remove(t)

            meld_value = self._meld_potential_flexible(h2)
            shape_val = self.keep_weight * composite_shape_metric(
                h2, w, declared_melds=declared_eff,env=env, seat=self.seat
            )
            isolate_bonus = -0.4 if (_tile_is_isolate(cnt, t) and t not in virt_tiles) else 0.0
            safety_val = self._estimate_safety(t)
            danger_val = self.risk_weight * safety_val

            total_val = shape_val - 0.35 * meld_value + danger_val + isolate_bonus
            if total_val < best_val:
                best_val = total_val
                best_tile = t

        return best_tile or random.choice(opts)


    # def pick_discard(self, env: Env) -> str:
    #     p = env.players[self.seat]
    #     hand = [t for t in p.concealed if not is_flower(t)]
    #     if not hand:
    #         return p.concealed[0]

    #     opts = env.legal_discards(self.seat)
    #     if not opts:
    #         return p.concealed[0]

    #     cnt = Counter(hand)
    #     declared = _meld_count(p)
    #     w = self._effective_w(env, hand)

    #     best_tile, best_val = None, float("inf")
    #     for t in opts:
    #         h2 = hand[:]
    #         if t in h2:
    #             h2.remove(t)

    #         meld_value = self._meld_potential_flexible(h2)

    #         shape_val = self.keep_weight * composite_shape_metric(
    #             h2, w, declared_melds=declared
    #         )

    #         isolate_bonus = -0.4 if _tile_is_isolate(cnt, t) else 0.0
    #         safety_val = self._estimate_safety(t)
    #         danger_val = self.risk_weight * safety_val

    #         total_val = shape_val - 0.35 * meld_value + danger_val + isolate_bonus
    #         if total_val < best_val:
    #             best_val = total_val
    #             best_tile = t

    #     return best_tile or random.choice(opts)

    def _meld_potential_flexible(self, hand: List[str]) -> float:
        cnt = Counter(hand)
        score = 0.0
        for t, c in cnt.items():
            if c >= 3:
                score += 2.5
            elif c == 2:
                score += 1.5
            if _is_suit_tile(t):
                r, s = _tile_rank_suit(t)
                for dr in (-2, -1, 1, 2):
                    rr = r + dr
                    if 1 <= rr <= 9 and f"{rr}{s}" in cnt:
                        score += 0.4
        return score

    def _estimate_safety(self, tile: str) -> float:
        if tile in ("E", "S", "W", "N", "C", "F", "B"):
            return 0.8
        if not _is_suit_tile(tile):
            return 0.6
        r, s = _tile_rank_suit(tile)
        if r in (1, 9):
            return 0.5
        return 0.2

    # ---------- Claim logic (boosted over HybridAggro) ----------

    def decide_ron(self, env, tile, points, loser):
        return True
    
    def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
        # First, let HybridAggroPolicy decide (now includes virtual-meld guard).
        if self.hy.decide_pung(env, seat, tile):
            return True

        pview = env.players[seat]
        hand = [t for t in pview.concealed if not is_flower(t)]
        if hand.count(tile) < 2:
            return False

        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, seat)

        # Same "don't break virtual meld" rule for the fallback.
        if tile in virt_tiles:
            if self.rng.random() < 0.85:
                return False

        early = len(env.discard_history) < 24
        base_p = 0.55 + 0.10 * declared_eff
        if early:
            base_p += 0.20

        if declared_eff >= 3:
            base_p *= 0.5

        prob = min(0.98, max(0.0, base_p) * self.aggressiveness)
        return self.rng.random() < prob

    # def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
    #     if self.hy.decide_pung(env, seat, tile):
    #         return True

    #     p = env.players[seat]
    #     hand = [t for t in p.concealed if not is_flower(t)]
    #     if hand.count(tile) < 2:
    #         return False

    #     declared = _meld_count(p, env, self.seat, include_virtual=True)
    #     early = len(env.discard_history) < 24
    #     base_p = 0.55 + 0.10 * declared
    #     if early:
    #         base_p += 0.20
    #     prob = min(0.98, base_p * self.aggressiveness)
    #     return self.rng.random() < prob

    def choose_chow(
        self,
        env: Env,
        seat: int,
        tile: str,
        chow_sets: List[Tuple[str, str]]
    ) -> Optional[Tuple[str, str]]:
        # First, try HybridAggroPolicy (which already respects virtual melds).
        best = self.hy.choose_chow(env, seat, tile, chow_sets)
        if best is not None:
            return best
        if not chow_sets:
            return None

        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, seat)

        # Avoid breaking virtual melds in the fallback too.
        filtered: List[Tuple[str, str]] = []
        for cs in chow_sets:
            if any(t in virt_tiles for t in cs):
                if self.rng.random() < 0.85:
                    continue
            filtered.append(cs)

        if not filtered:
            return None

        early = len(env.discard_history) < 24
        base_p = 0.45 + 0.08 * declared_eff
        if early:
            base_p += 0.15

        if declared_eff >= 3:
            base_p *= 0.5

        prob = min(0.95, max(0.0, base_p) * self.aggressiveness)
        return filtered[0] if (self.rng.random() < prob) else None

    # def choose_chow(
    #     self,
    #     env: Env,
    #     seat: int,
    #     tile: str,
    #     chow_sets: List[Tuple[str, str]]
    # ) -> Optional[Tuple[str, str]]:
    #     best = self.hy.choose_chow(env, seat, tile, chow_sets)
    #     if best is not None:
    #         return best
    #     if not chow_sets:
    #         return None

    #     declared = _meld_count(env.players[seat], env, self.seat, include_virtual=True)
    #     early = len(env.discard_history) < 24
    #     base_p = 0.45 + 0.08 * declared
    #     if early:
    #         base_p += 0.15
    #     prob = min(0.95, base_p * self.aggressiveness)
    #     return chow_sets[0] if (self.rng.random() < prob) else None

    def decide_open_kong(self, env: Env, seat: int, tile: str) -> bool:
        if self.hy.decide_open_kong(env, seat, tile):
            return True
        base = self._b("open_kong_bias")
        prob = min(0.98, 0.6 + 0.3 * base)
        return self.rng.random() < prob

    def decide_add_kong(self, env: Env, seat: int, tile: str) -> bool:
        if self.hy.decide_add_kong(env, seat, tile):
            return True
        base = self._b("open_kong_bias")
        prob = min(0.98, 0.55 + 0.35 * base)
        return self.rng.random() < prob

    def decide_closed_kong(
        self,
        env: Env,
        seat: int,
        candidates: List[str]
    ) -> Optional[str]:
        pick = self.hy.decide_closed_kong(env, seat, candidates)
        if pick:
            return pick
        if not candidates:
            return None

        base = self._b("closed_kong_bias")
        prob = min(0.9, 0.4 + 0.3 * base)
        return random.choice(candidates) if (self.rng.random() < prob) else None

    
class FlexibleAggroPolicyD(BasePolicy):
    """
    Aggressive but flexible (defensive flavor):
      - Same discard logic as FlexibleAggroPolicyD in your original, but now
        declared_melds is threaded into the shape metric.
    """
    def __init__(self, seat: int, rules: Dict, tuner: Optional[AdaptiveTuner]):
        super().__init__(seat, rules, tuner)
        self.hy = HybridAggroPolicy(seat, rules, tuner)
        self.rng = random.Random()
        self.aggressiveness = 0.90
        self.risk_weight = 0.03
        self.keep_weight = 1.0

    def pick_discard(self, env: Env) -> str:
        p = env.players[self.seat]
        hand = [t for t in p.concealed if not is_flower(t)]
        if not hand:
            return p.concealed[0]

        opts = env.legal_discards(self.seat)
        if not opts:
            return p.concealed[0]

        cnt = Counter(hand)
        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, self.seat)
        w = self._effective_w(env, hand)

        best_tile, best_val = None, float("inf")
        for t in opts:
            h2 = hand[:]
            if t in h2:
                h2.remove(t)

            meld_value = self._meld_potential_flexible(h2)
            shape_val = self.keep_weight * composite_shape_metric(
                h2, w, declared_melds=declared_eff,env=env, seat=self.seat
            )
            isolate_bonus = -0.4 if (_tile_is_isolate(cnt, t) and t not in virt_tiles) else 0.0
            safety_val = self._estimate_safety(t)
            danger_val = self.risk_weight * safety_val

            total_val = shape_val - 0.35 * meld_value + danger_val + isolate_bonus
            if total_val < best_val:
                best_val = total_val
                best_tile = t

        return best_tile or random.choice(opts)


    # def pick_discard(self, env: Env) -> str:
    #     p = env.players[self.seat]
    #     hand = [t for t in p.concealed if not is_flower(t)]
    #     if not hand:
    #         return p.concealed[0]

    #     opts = env.legal_discards(self.seat)
    #     if not opts:
    #         return p.concealed[0]

    #     cnt = Counter(hand)
    #     declared = _meld_count(p)
    #     w = self._effective_w(env, hand)

    #     best_tile, best_val = None, float("inf")
    #     for t in opts:
    #         h2 = hand[:]
    #         if t in h2:
    #             h2.remove(t)

    #         meld_value = self._meld_potential_flexible(h2)
    #         shape_val = self.keep_weight * composite_shape_metric(
    #             h2, w, declared_melds=declared
    #         )
    #         isolate_bonus = -0.4 if _tile_is_isolate(cnt, t) else 0.0
    #         safety_val = self._estimate_safety(t)
    #         danger_val = self.risk_weight * safety_val

    #         total_val = shape_val - 0.35 * meld_value + danger_val + isolate_bonus
    #         if total_val < best_val:
    #             best_val = total_val
    #             best_tile = t

    #return best_tile or random.choice(opts)

    def _meld_potential_flexible(self, hand: List[str]) -> float:
        cnt = Counter(hand)
        score = 0.0
        for t, c in cnt.items():
            if c >= 3:
                score += 2.5
            elif c == 2:
                score += 1.5
            if _is_suit_tile(t):
                r, s = _tile_rank_suit(t)
                for dr in (-2, -1, 1, 2):
                    rr = r + dr
                    if 1 <= rr <= 9 and f"{rr}{s}" in cnt:
                        score += 0.4
        return score

    def _estimate_safety(self, tile: str) -> float:
        if tile in ("E", "S", "W", "N", "C", "F", "B"):
            return 0.8
        if not _is_suit_tile(tile):
            return 0.6
        r, s = _tile_rank_suit(tile)
        if r in (1, 9):
            return 0.5
        return 0.2

    # Claim logic – same as your original FlexibleAggroPolicyD,
    # no composite_shape_metric involved here:

    def decide_ron(self, env, tile, points, loser):
        return True

    def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
        if self.hy.decide_pung(env, seat, tile):
            return True

        pview = env.players[seat]
        hand = [t for t in pview.concealed if not is_flower(t)]
        if hand.count(tile) < 2:
            return False

        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, seat)

        if tile in virt_tiles:
            if self.rng.random() < 0.85:
                return False

        early = len(env.discard_history) < 24
        base_p = 0.55 + 0.10 * declared_eff
        if early:
            base_p += 0.20

        if declared_eff >= 3:
            base_p *= 0.5

        prob = min(0.98, max(0.0, base_p) * self.aggressiveness)
        return self.rng.random() < prob
    
    # def decide_pung(self, env: Env, seat: int, tile: str) -> bool:
    #     if self.hy.decide_pung(env, seat, tile):
    #         return True
    #     p = env.players[seat]
    #     hand = [t for t in p.concealed if not is_flower(t)]
    #     if hand.count(tile) < 2:
    #         return False
    #     declared = _meld_count(p, env, self.seat, include_virtual=True)
    #     early = len(env.discard_history) < 24
    #     base_p = 0.55 + 0.10 * declared
    #     if early:
    #         base_p += 0.20
    #     prob = min(0.98, base_p * self.aggressiveness)
    #     return self.rng.random() < prob

    def choose_chow(
        self,
        env: Env,
        seat: int,
        tile: str,
        chow_sets: List[Tuple[str, str]]
    ) -> Optional[Tuple[str, str]]:
        best = self.hy.choose_chow(env, seat, tile, chow_sets)
        if best is not None:
            return best
        if not chow_sets:
            return None

        real_declared, declared_eff, virt_tiles = _get_virtual_meld_context(env, seat)

        filtered: List[Tuple[str, str]] = []
        for cs in chow_sets:
            if any(t in virt_tiles for t in cs):
                if self.rng.random() < 0.85:
                    continue
            filtered.append(cs)

        if not filtered:
            return None

        early = len(env.discard_history) < 24
        base_p = 0.45 + 0.08 * declared_eff
        if early:
            base_p += 0.15

        if declared_eff >= 3:
            base_p *= 0.5

        prob = min(0.95, max(0.0, base_p) * self.aggressiveness)
        return filtered[0] if (self.rng.random() < prob) else None
    
    # def choose_chow(
    #     self,
    #     env: Env,
    #     seat: int,
    #     tile: str,
    #     chow_sets: List[Tuple[str, str]]
    # ) -> Optional[Tuple[str, str]]:
    #     best = self.hy.choose_chow(env, seat, tile, chow_sets)
    #     if best is not None:
    #         return best
    #     if not chow_sets:
    #         return None

    #     declared = _meld_count(env.players[seat], env, self.seat, include_virtual=True)
    #     early = len(env.discard_history) < 24
    #     base_p = 0.45 + 0.08 * declared
    #     if early:
    #         base_p += 0.15
    #     prob = min(0.95, base_p * self.aggressiveness)
    #     return chow_sets[0] if (self.rng.random() < prob) else None

    def decide_open_kong(self, env: Env, seat: int, tile: str) -> bool:
        if self.hy.decide_open_kong(env, seat, tile):
            return True
        base = self._b("open_kong_bias")
        prob = min(0.98, 0.6 + 0.3 * base)
        return self.rng.random() < prob

    def decide_add_kong(self, env: Env, seat: int, tile: str) -> bool:
        if self.hy.decide_add_kong(env, seat, tile):
            return True
        base = self._b("open_kong_bias")
        prob = min(0.98, 0.55 + 0.35 * base)
        return self.rng.random() < prob

    def decide_closed_kong(
        self,
        env: Env,
        seat: int,
        candidates: List[str]
    ) -> Optional[str]:
        pick = self.hy.decide_closed_kong(env, seat, candidates)
        if pick:
            return pick
        if not candidates:
            return None

        base = self._b("closed_kong_bias")
        prob = min(0.9, 0.4 + 0.3 * base)
        return random.choice(candidates) if (self.rng.random() < prob) else None


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
POLICY_MAP["flexaggro"] = FlexibleAggroPolicy
POLICY_MAP["flexaggrod"] = FlexibleAggroPolicyD

# from algorithm.mahjongrl.agent import RLPolicy
# POLICY_MAP["rl"] = RLPolicy

def build_policies(lineup, rules, tuner):
    policies = []
    for seat, k in enumerate(lineup):
        # NEW: allow factory callables
        if callable(k):
            p = k(seat, rules, tuner)
        else:
            cls = POLICY_MAP[k.lower()]
            p = cls(seat, rules, tuner)
        policies.append(p)
    return policies


# ---------------------------- Episode + I/O ----------------------------

from typing import Dict, Any, List, Optional
# make sure you already import these somewhere in the file:
# from algorithm.mahjong.tiles import is_flower, ALL_TILE_CLASSES
# from algorithm.mahjongrl.shanten import rough_shanten_like, chiitoi_shanten_like, rough_ukeire
# (or whatever the actual import paths are in your repo)

def _compute_opening_metrics(env, chiitoi_weight: float = 0.3) -> List[Dict[str, float]]:
    """
    Compute shanten / chiitoi shanten / env-aware ukeire for each seat
    at *opening* (after flowers, before any play).

    Returns a list of 4 dicts, one per seat:
      [
        {"std_core": ..., "chiitoi": ..., "ukeire": ..., "composite": ...},
        ...
      ]
    """
    metrics: List[Dict[str, float]] = []

    for seat in range(4):
        # opening_after_flowers is set in Env.__init__
        tiles = [t for t in env.opening_after_flowers[seat] if not is_flower(t)]

        std_core = float(rough_shanten_like(tiles, declared_melds=0))
        chiitoi  = float(chiitoi_shanten_like(tiles, declared_melds=0))
        uke      = float(rough_ukeire(env, seat, hand=tiles, declared_melds=0))

        w = max(0.0, min(1.0, chiitoi_weight))

        # Your composite: shanten minus a small multiple of outs, blended with chiitoi
        std_adj   = std_core - 0.02 * uke
        composite = (1.0 - w) * std_adj + w * chiitoi

        metrics.append(
            {
                "std_core": std_core,
                "chiitoi": chiitoi,
                "ukeire": uke,
                "composite": composite,
            }
        )

    return metrics


def run_episode(
    rules: Dict,
    lineup: List[str],
    tuner: Optional[AdaptiveTuner],
    seed: Optional[int] = None,
    max_draws: int = 600,
    return_env: bool = False,
    chiitoi_weight: float = 0.3,  # <- new optional arg if you want
) -> Dict[str, Any]:
    env = Env(rules, seed=seed)

    # *** NEW: compute env-aware opening metrics right after dealing ***
    opening_metrics = _compute_opening_metrics(env, chiitoi_weight=chiitoi_weight)

    policies = build_policies(lineup, rules, tuner)
    draws = 0
    while draws < max_draws and env.wall and not env.terminal:
        env.step_turn(policies)
        draws += 1

    if not env.terminal:
        env.terminal = {
            "winner": None,
            "source": "drawn_game",
            "points": 0,
            "side_delta": env.side_delta[:],
            "side_events": env.side_events[:],
        }

    if tuner is not None:
        tuner.record_episode(env.terminal, env.claim_log)

    openings = {
        "before_flowers": [list(x) for x in env.opening_before_flowers],
        "after_flowers":  [list(x) for x in env.opening_after_flowers],
        "flowers":        [list(x) for x in env.opening_flowers],
    }

    result: Dict[str, Any] = {
        "terminal": env.terminal,
        "stats": env.stats,
        "claim_log": env.claim_log,
        "openings": openings,
        # *** NEW: stash metrics per seat ***
        "opening_metrics": opening_metrics,
    }

    if return_env:
        result["env"] = env

    return result



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
    allowed = {"random","wp","payout","hybrid","aggro","hyaggro","flexaggro","flexaggrod"}
    if len(lineup) != 4 or any(x not in allowed for x in lineup):
        raise SystemExit("Invalid --lineup; choose 4 from {random,wp,payout,hybrid,aggro,hyaggro,flexaggro,flexaggrod}")
    return lineup

if __name__ == "__main__":
    import argparse, json, random
    from pathlib import Path
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

    print(f"[episodes] generating {args.episodes} episodes → {out}")
    n_written = 0

    with out.open("w", encoding="utf-8") as f:
        for epi in range(args.episodes):
            ep = run_episode(rules, lineup, tuner, seed=rng.randint(1,10**9))

            # --- Safety normalization ---
            if not isinstance(ep, dict):
                ep = {"terminal": {"source": "unknown"}, "stats": {}, "claim_log": [], "openings": {}}
            else:
                # Ensure the fields exist
                ep.setdefault("terminal", {})
                ep.setdefault("stats", {})
                ep.setdefault("claim_log", [])
                ep.setdefault("openings", {})

            f.write(json.dumps(ep, ensure_ascii=False))
            f.write("\n")
            n_written += 1

            if (epi + 1) % 50 == 0:
                print(f"[episodes] {epi+1}/{args.episodes} done")

    print(f"[episodes] DONE: wrote {n_written} lines to {out.resolve()}")

    # Summarize results (unchanged)
    summarize_jsonl(
        args.out,
        rules,
        enforce_zero_sum=True,
        print_wins=args.print_wins,
        print_openings=args.print_openings
    )

# hand = ['5b','6b','7b','7t','8t','9t','3w','3w','E','E','S','S','9w','9w']
# print("No declared melds:")
# print(rough_shanten_like(hand, declared_melds=0))
# print("With one declared meld:")
# print(rough_shanten_like(hand, declared_melds=1))
