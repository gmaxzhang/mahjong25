# -*- coding: utf-8 -*-
"""
Custom Mahjong rules + scoring (rules-driven)
============================================

Tile format:
  - Suited: "1b","9t","5w"  (b=sticks/bamboo, w=characters, t=dots)
  - Honors: "E","S","W","N","C","F","B" (winds, dragons; 'F' here is GREEN DRAGON)
  - Flowers: "F1".."F8" only

This module exposes:
  - class Meld
  - class HandState
  - class ScoreBreakdown
  - function score_hand(state, rules=None)

Key behaviors:
  - Declared melds (chi/chow, pung, kong) are **fixed**; their tiles are not reused by the
    shape solver. Undeclared 4-of-a-kind in the concealed tiles can still count as two pairs
    for Seven Pairs (only when there are no declared melds).
  - "Eating Hand" = all four melds are claimed chows.
  - Min-flower rule on RON when player has used a claimed set is enforced via rules:
        win_requirements.ron_min_flower_points_if_any_claimed_set
        win_requirements.exempt_if_menzen_before_ron
        win_requirements.tsumo_requires_min_flowers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable, Sequence, Literal
from collections import Counter

# ----------------------- Tile encoding -----------------------

def is_flower(x: str) -> bool:
    # Flowers are strictly F1..F8
    return len(x) >= 2 and x[0] == "F" and x[1:].isdigit()

def is_honor(x: str) -> bool:
    return x in {"E","S","W","N","C","F","B"}  # 'F' here is green dragon, NOT a flower

def is_wind(x: str) -> bool:
    return x in {"E","S","W","N"}

def is_dragon(x: str) -> bool:
    return x in {"C","F","B"}

def is_suited(x: str) -> bool:
    return len(x) == 2 and x[0].isdigit() and x[1] in "bwt"

def suit_of(x: str) -> Optional[str]:
    return x[1] if is_suited(x) else None

def rank_of(x: str) -> Optional[int]:
    return int(x[0]) if is_suited(x) else None

def is_terminal(x: str) -> bool:
    r = rank_of(x)
    return is_suited(x) and r in {1,9}

# ----------------------- Meld representation -----------------------

# Accept both "chi" and "chow" (simulator uses "chi")
MeldType = Literal["chi","chow","pung","kong","pair"]

@dataclass(frozen=True)
class Meld:
    type: MeldType
    tiles: Tuple[str, ...]             # 3 or 4 for sets, 2 for pair
    open: bool = False
    formed_by_claim: bool = False      # if the set was claimed from another player's discard

    def normalized_type(self) -> str:
        return "chow" if self.type == "chi" else self.type

    def key(self) -> Tuple:
        return (self.normalized_type(), tuple(sorted(self.tiles)), self.open)

# ----------------------- Hand state -----------------------

@dataclass
class HandState:
    flowers: List[str]
    concealed: List[str]
    melds: List[Meld]
    winning_tile: str
    win_source: Literal["self_draw","discard"]
    back_wall_bonus: bool = False
    used_any_claim_for_sets: bool = False

# ----------------------- Pattern helpers -----------------------

def is_chow_tiles(tiles: Sequence[str]) -> bool:
    if len(tiles) != 3: return False
    a,b,c = sorted(tiles, key=lambda t: (suit_of(t), rank_of(t)))
    if not (is_suited(a) and is_suited(b) and is_suited(c)): return False
    if suit_of(a) != suit_of(b) or suit_of(b) != suit_of(c): return False
    ra,rb,rc = rank_of(a), rank_of(b), rank_of(c)
    return rb == ra+1 and rc == rb+1

def is_pung_tiles(tiles: Sequence[str]) -> bool:
    return len(tiles)==3 and len(set(tiles))==1

def is_kong_tiles(tiles: Sequence[str]) -> bool:
    return len(tiles)==4 and len(set(tiles))==1

def is_pair_tiles(tiles: Sequence[str]) -> bool:
    return len(tiles)==2 and len(set(tiles))==1

def tiles_one_color(tiles: Iterable[str]) -> bool:
    suits = {suit_of(t) for t in tiles if is_suited(t)}
    honors = [t for t in tiles if is_honor(t)]
    return len(suits) == 1 and len(honors) == 0 and len(suits) == 1

def tiles_mixed_one_color(tiles: Iterable[str], eyes: Sequence[str]) -> bool:
    suits = {suit_of(t) for t in tiles if is_suited(t)}
    honors_in_eyes = (len(eyes)==2 and eyes[0]==eyes[1] and is_honor(eyes[0]))
    return len(suits) == 1 and honors_in_eyes

def is_peng_peng_hu(melds: Sequence[Meld]) -> bool:
    """All sets are pungs/kongs (no chow)."""
    types = {m.normalized_type() for m in melds if m.normalized_type() != "pair"}
    return "chow" not in types and len([m for m in melds if m.normalized_type() in {"pung","kong"}]) == 4

def _flatten_declared_set_tiles(melds: Sequence[Meld]) -> List[str]:
    out: List[str] = []
    for m in melds:
        if m.normalized_type() in {"chow","pung","kong"}:
            out.extend(m.tiles)
    return out

# ----------------------- Simple decomposition (shape) -----------------------

def split_into_sets_and_pair(all_tiles_no_flowers: List[str]) -> Optional[Tuple[List[Meld], Tuple[str,str]]]:
    """
    Pure structural 4m1p decomposition on 14 NON-FLOWER tiles.
    This does not respect declared/fixed sets; it just answers "is there some 4m1p".
    Returns (list_of_Melds, pair_eyes) or None if impossible.
    """
    tiles = sorted([t for t in all_tiles_no_flowers if not is_flower(t)])
    if len(tiles) != 14:
        return None
    cnt = Counter(tiles)

    def backtrack(current_sets: List[Meld]) -> Optional[Tuple[List[Meld], Tuple[str,str]]]:
        taken = Counter()
        for m in current_sets:
            for x in m.tiles: taken[x]+=1
        remains: List[str] = []
        for x,c in cnt.items():
            r = c - taken[x]
            remains += [x]*r
        if len(current_sets)==4:
            if len(remains)==2 and remains[0]==remains[1]:
                return current_sets, (remains[0], remains[1])
            return None
        if not remains: return None
        x = remains[0]
        # pung
        if remains.count(x) >= 3:
            m = Meld("pung", (x,x,x), open=False)
            res = backtrack(current_sets+[m])
            if res: return res
        # chow
        if is_suited(x):
            s = suit_of(x); r = rank_of(x)
            if r is not None and r <= 7:
                y = f"{r+1}{s}"
                z = f"{r+2}{s}"
                if y in remains and z in remains:
                    m = Meld("chow", (x,y,z), open=False)
                    res = backtrack(current_sets+[m])
                    if res: return res
        # kong (treat as a set option; rarely needed for 4m1p check)
        if remains.count(x) == 4:
            m = Meld("kong", (x,x,x,x), open=False)
            res = backtrack(current_sets+[m])
            if res: return res
        return None

    return backtrack([])

# ----------------------- Flower points from sets -----------------------

def flower_points_from_sets(melds: Sequence[Meld], rules: Dict) -> int:
    set_pts = rules["flower_points"]["set_points"]
    pts = 0
    for m in melds:
        mtype = m.normalized_type()
        if mtype == "pair":
            continue
        base = m.tiles[0]
        is_open = bool(getattr(m, "open", True))
        closed_kong = (mtype=="kong" and not is_open)
        open_kong   = (mtype=="kong" and is_open)
        is_pung_set = (mtype=="pung")

        key = "suit"
        if is_dragon(base): key = "dragon"
        elif is_wind(base): key = "wind"

        if is_pung_set:
            pts += set_pts[key].get("pung", 0)
        if open_kong:
            pts += set_pts[key].get("kong_open", 0)
        if closed_kong:
            pts += set_pts[key].get("kong_closed", 0)
    return pts

# ----------------------- Special hands -----------------------

def all_apart(tiles14_with_flowers: List[str]) -> bool:
    """
    All Apart / 13 non-neighbors:
      - Exactly one of {1,4,7} or {2,5,8} or {3,6,9} in each suit (so 9 suited tiles total).
      - Remaining 5 are honors/flowers, all distinct, with at most 1 flower among the 14.
      - Only sensible on a pure 14-tile hand (i.e., no declared melds).
    """
    tiles = tiles14_with_flowers[:]
    if len(tiles) != 14:
        return False
    flowers = [x for x in tiles if is_flower(x)]
    if len(flowers) > 1:
        return False

    suited = [x for x in tiles if is_suited(x)]
    others = [x for x in tiles if not is_suited(x)]
    if len(suited) != 9 or len(others) != 5:
        return False

    groups = [{1,4,7},{2,5,8},{3,6,9}]
    def suit_ok(s: str) -> bool:
        ranks = sorted(rank_of(x) for x in suited if suit_of(x)==s)
        return any(sorted(g) == ranks for g in groups)

    if not (suit_ok('b') and suit_ok('w') and suit_ok('t')):
        return False

    non_suited = [x for x in tiles if not is_suited(x)]
    # all distinct among honors/flowers
    if len(set(non_suited)) != len(non_suited):
        return False
    return True

# ----------------------- Scoring -----------------------

@dataclass
class ScoreBreakdown:
    base_points: int
    flower_points: int
    specials: Dict[str,int] = field(default_factory=dict)
    back_wall_bonus: int = 0
    total_points: int = 0

DEFAULT_RULES = {
  "base_scoring": {"base_default": 10, "back_wall_bonus": 5, "specials_stack": True},
  "flower_points": {
    "per_flower_tile": 1,
    "set_points": {
      "dragon": {"pung": 2, "kong_open": 3, "kong_closed": 4},
      "wind":   {"pung": 1, "kong_open": 2, "kong_closed": 3},
      "suit":   {"pung": 0, "kong_open": 1, "kong_closed": 2}
    }
  },
  "win_requirements": {
    # On RON, if you used any claimed set, require >= this many flower points unless exempted.
    "ron_min_flower_points_if_any_claimed_set": 2,
    "exempt_if_menzen_before_ron": True,
    # For TSUMO (self-draw), require min flowers too? (usually False by house rules)
    "tsumo_requires_min_flowers": False
  },
  "special_hands": {
    "one_color": 40,
    "mixed_one_color": 20,
    "all_apart": 20,
    "seven_pairs": 40,
    "peng_peng_hu": 20,
    "eating_hand": 20
  }
}

def _collect_complete_tiles_for_scoring(state: HandState) -> Tuple[List[str], List[str]]:
    """
    Returns a tuple:
      tiles_no_flowers: the 14 NON-FLOWER tiles representing the complete hand
      tiles_with_flowers: the 14 tiles with flowers included (used only for All Apart check)
    Notes:
      - For TSUMO, the winning tile is already in 'concealed'; for RON we must add it.
      - Declared meld tiles are included to reach 14.
    """
    # concealed + maybe winning tile
    concealed_all = list(state.concealed)
    if state.win_source == "discard":
        concealed_all = concealed_all + [state.winning_tile]

    # include declared meld tiles
    meld_tiles = _flatten_declared_set_tiles(state.melds)

    full14_with_flowers = concealed_all + meld_tiles

    # --- Defensive normalization: if TSUMO and winning tile somehow not present, add it
    if state.win_source == "self_draw" and state.winning_tile not in concealed_all:
        full14_with_flowers.append(state.winning_tile)

    # Remove flowers for non-flower view
    tiles_no_flowers = [t for t in full14_with_flowers if not is_flower(t)]

    return tiles_no_flowers, full14_with_flowers

def score_hand(state: HandState, rules: Optional[Dict] = None) -> ScoreBreakdown:
    rules = rules or DEFAULT_RULES
    base_default = int(rules["base_scoring"]["base_default"])
    back_bonus_val = int(rules["base_scoring"]["back_wall_bonus"])

    # Build complete hand tiles (with/without flowers)
    tiles_no_flowers, full14_with_flowers = _collect_complete_tiles_for_scoring(state)

    # Flower points: flowers played + from sets per rules
    flower_pts = len([f for f in state.flowers if is_flower(f)]) * int(rules["flower_points"]["per_flower_tile"]) \
                 + flower_points_from_sets(state.melds, rules)

    # Minimum flower rule
    if state.win_source == "discard":
        need = int(rules["win_requirements"]["ron_min_flower_points_if_any_claimed_set"])
        menzen = not bool(getattr(state, "used_any_claim_for_sets", False))
        exempt = bool(rules["win_requirements"]["exempt_if_menzen_before_ron"]) and menzen
        if (not exempt) and (flower_pts < need):
            raise ValueError(
                f"Invalid win by discard: requires at least {need} flower points when claimed sets were used."
            )
    else:  # self_draw
        if bool(rules["win_requirements"]["tsumo_requires_min_flowers"]):
            need = int(rules["win_requirements"]["ron_min_flower_points_if_any_claimed_set"])
            if flower_pts < need:
                raise ValueError(f"Invalid self-draw: requires at least {need} flower points per rules.")

    # Base = base_default + flower points
    base = base_default + flower_pts

    # Specials
    specials: Dict[str,int] = {}
    sh = rules["special_hands"]

    # Structural 4m1p decomposition (only defined if exactly 14 non-flower tiles)
    struct = split_into_sets_and_pair(tiles_no_flowers) if len(tiles_no_flowers) == 14 else None

    # Count declared sets for constraints (Seven Pairs / All Apart only with NO declared sets)
    declared_sets = [m for m in state.melds if m.normalized_type() in {"chow","pung","kong"}]
    num_declared = len(declared_sets)

    # Seven Pairs (no declared sets; 4-of-a-kind counts as two pairs)
    if num_declared == 0 and len(tiles_no_flowers) == 14:
        cnt = Counter(tiles_no_flowers)
        pairs = 0
        valid = True
        for v in cnt.values():
            if v == 2: pairs += 1
            elif v == 4: pairs += 2
            else: valid = False; break
        if valid and pairs == 7:
            specials["Seven Pairs"] = int(sh["seven_pairs"]) - base_default

    # All Apart (no declared sets; check on 14 with flowers to allow up to 1 flower)
    if num_declared == 0 and len(full14_with_flowers) == 14 and all_apart(full14_with_flowers):
        specials["All Apart"] = int(sh["all_apart"]) - base_default

    # One color / Mixed one color (over all non-flower tiles)
    if struct:
        sets_s, eyes = struct
        if tiles_one_color(tiles_no_flowers):
            specials["One Color"] = int(sh["one_color"]) - base_default
        elif tiles_mixed_one_color(tiles_no_flowers, eyes):
            specials["Mixed One Color"] = int(sh["mixed_one_color"]) - base_default

        # Peng-peng-hu (all pungs/kongs) — structural view so concealed pungs count
        if is_peng_peng_hu(sets_s):
            specials["Peng-peng-hu"] = int(sh["peng_peng_hu"])

    # Eating Hand: all four melds must be claimed chows in declared sets
    # (Treat "formed_by_claim"==True OR open==True as 'claimed' to be robust to upstream data.)
    sets_only = [m for m in state.melds if m.normalized_type() in {"chow","pung","kong"}]
    if len(sets_only) == 4 and all(
        (m.normalized_type() == "chow") and (getattr(m, "formed_by_claim", False) or getattr(m, "open", False))
        for m in sets_only
    ):
        specials["Eating Hand"] = int(sh["eating_hand"])

    # Back-wall bonus
    back_bonus = back_bonus_val if bool(getattr(state, "back_wall_bonus", False)) else 0

    total = base + sum(specials.values()) + back_bonus

    return ScoreBreakdown(
        base_points=base_default,
        flower_points=flower_pts,
        specials=specials,
        back_wall_bonus=back_bonus,
        total_points=total
    )
