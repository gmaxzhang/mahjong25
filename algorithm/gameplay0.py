from __future__ import annotations
# Create a Python scaffolding that encodes tiles, hands, and the family scoring rules,
# plus stubs for simulation + RL/non-ML policy hooks. Comments are in English.
from pathlib import Path

#code=r'''
"""
Custom Mahjong project scaffolding
==================================

This module implements:
1) Tile encoding & parsing (bamboos: b, characters: w, circles: t, winds & dragons: E/W/N/S/C/F/B).
2) Hand/meld representation with open/closed info.
3) Scoring according to the family rules described by the user.
4) Payout aggregation (self-draw vs. win-on-discard).
5) Stubs for simulation, RL environment, and a non-ML EV maximizer.

Assumptions noted where rules were ambiguous (see comments below).
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable, Sequence, Literal
from collections import Counter, defaultdict

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

# ----------------------- NEW: concealed split given declared sets -----------------------

def split_concealed_given_declared(
    concealed_no_fl: List[str],
    declared_melds: List[Meld]
) -> Optional[Tuple[List[List[str]], List[str]]]:
    """
    Reconstruct the concealed structure for a winning hand that already has declared sets.
    Returns (concealed_sets, pair) where concealed_sets are 3-tile melds, pair is [x,x].
    Each kong in declared_melds counts as ONE declared meld.
    """
    def _kind(m: Meld) -> str:
        return m.normalized_type()

    declared = [m for m in (declared_melds or []) if _kind(m) in {"chow","pung","kong"}]
    m_fixed = len(declared)
    m_needed = 4 - m_fixed
    if m_needed < 0:
        return None  # too many declared sets

    tiles = sorted([t for t in concealed_no_fl if not is_flower(t)])
    need_len = m_needed * 3 + 2
    if len(tiles) != need_len:
        return None

    cnt = Counter(tiles)

    def try_with_pair(pair_tile: str) -> Optional[Tuple[List[List[str]], List[str]]]:
        if cnt[pair_tile] < 2:
            return None
        cnt[pair_tile] -= 2

        def take_melds(c: Counter, left: int) -> Optional[List[List[str]]]:
            if left == 0:
                return [] if sum(c.values()) == 0 else None
            t = None
            for k, v in c.items():
                if v > 0:
                    t = k; break
            if t is None:
                return None
            # pung
            if c[t] >= 3:
                c[t] -= 3
                r = take_melds(c, left-1)
                if r is not None:
                    return [[t,t,t]] + r
                c[t] += 3
            # chow
            if is_suited(t):
                rnk, sut = rank_of(t), suit_of(t)
                a, b = f"{rnk+1}{sut}", f"{rnk+2}{sut}"
                if c[a] > 0 and c[b] > 0:
                    c[t] -= 1; c[a] -= 1; c[b] -= 1
                    r = take_melds(c, left-1)
                    if r is not None:
                        return [[t,a,b]] + r
                    c[t] += 1; c[a] += 1; c[b] += 1
            return None

        res = take_melds(cnt, m_needed)
        cnt[pair_tile] += 2
        if res is None:
            return None
        return (res, [pair_tile, pair_tile])

    for k, v in list(cnt.items()):
        if v >= 2:
            got = try_with_pair(k)
            if got:
                return got
    return None

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
    "ron_min_flower_points_if_any_claimed_set": 2,
    "exempt_if_menzen_before_ron": True,
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
      tiles_no_flowers: NON-FLOWER tiles used for hand-wide color checks (may exceed 14 if kongs declared)
      tiles_with_flowers: tiles including flowers (used for All Apart check on pure 14-tile hands)

    Notes:
      - For TSUMO, the winning tile is already in 'concealed'; for RON we add it to concealed.
      - Declared meld tiles (including kongs with 4 tiles) are appended; therefore tiles_no_flowers
        may be > 14 when there are declared kongs.
    """
    concealed_all = list(state.concealed)
    if state.win_source == "discard":
        concealed_all = concealed_all + [state.winning_tile]

    meld_tiles = _flatten_declared_set_tiles(state.melds)

    full_with_flowers = concealed_all + meld_tiles

    # Defensive: if TSUMO and winning tile somehow not present, add it
    if state.win_source == "self_draw" and state.winning_tile not in concealed_all:
        full_with_flowers.append(state.winning_tile)

    tiles_no_flowers = [t for t in full_with_flowers if not is_flower(t)]
    return tiles_no_flowers, full_with_flowers

def score_hand(state: HandState, rules: Optional[Dict] = None) -> ScoreBreakdown:
    rules = rules or DEFAULT_RULES
    base_default = int(rules["base_scoring"]["base_default"])
    back_bonus_val = int(rules["base_scoring"]["back_wall_bonus"])

    # Build hand tiles (kong-safe)
    tiles_no_fl, tiles_with_fl = _collect_complete_tiles_for_scoring(state)

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

    # Base points before specials replacement
    base = base_default + flower_pts

    # Specials (absolute points; stack-and-replace semantics)
    specials: Dict[str,int] = {}
    sh = rules["special_hands"]

    # Declared sets (true sets only)
    declared_sets = [m for m in state.melds if m.normalized_type() in {"chow","pung","kong"}]
    num_declared = len(declared_sets)

    # Structural decomposition for no-declared case (exact 14 non-flower)
    struct_no_declared = None
    if num_declared == 0 and len([t for t in tiles_no_fl if not is_flower(t)]) == 14:
        struct_no_declared = split_into_sets_and_pair(tiles_no_fl)

    # Reconstruct concealed structure in declared case (kong-safe)
    # Concealed NON-FLOWERS for structure:
    concealed_no_fl = [t for t in state.concealed if not is_flower(t)]
    if state.win_source == "discard":
        if not is_flower(state.winning_tile):
            concealed_no_fl = concealed_no_fl + [state.winning_tile]

    concealed_struct = None
    if num_declared > 0:
        concealed_struct = split_concealed_given_declared(concealed_no_fl, declared_sets)

    # --- Seven Pairs (only with NO declared sets)
    if num_declared == 0 and len([t for t in tiles_no_fl if not is_flower(t)]) == 14:
        cnt = Counter([t for t in tiles_no_fl if not is_flower(t)])
        pairs = 0
        valid = True
        for v in cnt.values():
            if v == 2: pairs += 1
            elif v == 4: pairs += 2
            else: valid = False; break
        if valid and pairs == 7:
            specials["Seven Pairs"] = int(sh["seven_pairs"])

    # --- All Apart (only with NO declared sets; check on 14 including up to 1 flower)
    if num_declared == 0 and len(tiles_with_fl) == 14 and all_apart(tiles_with_fl):
        specials["All Apart"] = int(sh["all_apart"])

    # --- One Color (works regardless of declared sets/kongs)
    if tiles_one_color(tiles_no_fl):
        specials["One Color"] = int(sh["one_color"])

    # --- Mixed One Color (need eyes to be an honor pair)
    eyes = None
    if struct_no_declared:
        _, eyes = struct_no_declared
    elif concealed_struct:
        _, eyes = concealed_struct  # pair always resides in the concealed remainder

    if eyes and tiles_mixed_one_color(tiles_no_fl, eyes):
        specials["Mixed One Color"] = int(sh["mixed_one_color"])

    # --- Peng-peng-hu (all pungs/kongs) — works with declared sets
    # Build full set list for structure check:
    def _meld_from_3tiles(ts: List[str]) -> Meld:
        if is_chow_tiles(ts):
            return Meld("chow", tuple(sorted(ts)), open=False)
        elif is_pung_tiles(ts):
            return Meld("pung", tuple(ts), open=False)
        else:
            # should not happen if reconstruction was valid; treat as pung fallback
            return Meld("pung", tuple(ts), open=False)

    full_sets_for_check: Optional[List[Meld]] = None
    if struct_no_declared:
        sets_s, _ = struct_no_declared
        full_sets_for_check = sets_s
    elif concealed_struct:
        concealed_sets, _ = concealed_struct
        full_sets_for_check = list(declared_sets) + [_meld_from_3tiles(s) for s in concealed_sets]

    if full_sets_for_check and is_peng_peng_hu(full_sets_for_check):
        specials["Peng-peng-hu"] = int(sh["peng_peng_hu"])

    # --- Eating Hand: all four declared sets are claimed chows
    sets_only = [m for m in state.melds if m.normalized_type() in {"chow","pung","kong"}]
    if len(sets_only) == 4 and all(
        (m.normalized_type() == "chow") and (getattr(m, "formed_by_claim", False) or getattr(m, "open", False))
        for m in sets_only
    ):
        specials["Eating Hand"] = int(sh["eating_hand"])

    # Back-wall bonus
    back_bonus = back_bonus_val if bool(getattr(state, "back_wall_bonus", False)) else 0

    # Total with "specials replace base" semantics (and specials stack if multiple)
    if specials and bool(rules["base_scoring"].get("specials_stack", True)):
        total = sum(specials.values()) + flower_pts + back_bonus
    elif specials:
        # if stack==False, pick the max special
        total = max(specials.values()) + flower_pts + back_bonus
    else:
        total = base_default + flower_pts + back_bonus

    return ScoreBreakdown(
        base_points=base_default,
        flower_points=flower_pts,
        specials=specials,
        back_wall_bonus=back_bonus,
        total_points=total
    )

# ----------------------- Payout aggregation -----------------------

def payout_for_table(hand_points: int, winner: int, loser: Optional[int], num_players: int = 4,
                     win_source: Literal["self_draw","discard"] = "self_draw") -> List[int]:
    """
    Returns a list of score deltas for each seat [0..num_players-1].
    - On discard: winner gains P, discarder loses P, others 0.
    - On self_draw: winner gains 3P, others each lose P.
    """
    deltas = [0]*num_players
    if win_source == "discard":
        assert loser is not None, "loser seat must be provided for discard wins"
        deltas[winner] += hand_points
        deltas[loser] -= hand_points
    else:
        deltas[winner] += 3*hand_points
        for i in range(num_players):
            if i != winner:
                deltas[i] -= hand_points
    return deltas

# ----------------------- Simulation + RL stubs -----------------------

class MahjongEnv:
    """
    Minimal environment skeleton for RL / simulation.
    - state(): returns observation dict (features for policy).
    - step(action): applies an action; returns (obs, reward, done, info).
    - legal_actions(): enumerates legal moves.
    This env must incorporate the custom scoring via score_hand() at terminal.
    """
    def __init__(self, rules=None):
        self.rules = rules or {}
        # TODO: initialize wall, hands, dealer, etc.
    def reset(self, seed: Optional[int]=None):
        # TODO: deal tiles, reset round state
        pass
    def state(self, seat: int):
        # TODO: build features (counts, meld indicators, flowers, shanten-like distance, etc.)
        return {}
    def legal_actions(self, seat: int):
        # TODO: enumerate legal moves for seat
        return []
    def step(self, seat: int, action):
        # TODO: apply action, handle claims, kongs/flowers, draw from back if needed,
        # and return reward when the hand ends using payout_for_table+score_hand.
        return {}, 0.0, False, {}

# ----------------------- Non-ML EV maximizer -----------------------

def expectimax_policy(env: MahjongEnv, seat: int, rollouts: int = 256) -> object:
    """
    Non-ML policy hook: choose action that maximizes expected value.
    Implement via Monte Carlo rollouts or depth-limited expectimax/UCT.
    (Stub for now.)
    """
    # TODO: simulate forward under simple opponent models and pick argmax EV.
    legal = env.legal_actions(seat)
    return legal[0] if legal else None

# ----------------------- Example usage -----------------------
if __name__ == "__main__":
    # Simple scoring example (fabricated):
    # Hand: pung of C (closed), pung of 5b (open), chow 2b3b4b (open), pong 9t (closed), pair EE
    hs = HandState(
        flowers=["F1","F2"],
        concealed=["E","E","C","C","C","9t","9t","9t"],
        melds=[
            Meld("pung", ("C","C","C"), open=False, formed_by_claim=False),
            Meld("pung", ("5b","5b","5b"), open=True, formed_by_claim=True),
            Meld("chow", ("2b","3b","4b"), open=True, formed_by_claim=True),
        ],
        winning_tile="9t",
        win_source="discard",
        back_wall_bonus=False,
        used_any_claim_for_sets=True
    )
    sb = score_hand(hs)
    print("Base:", sb.base_points, "FlowerPts:", sb.flower_points, "Specials:", sb.specials, "Back+:", sb.back_wall_bonus, "Total:", sb.total_points)
    print("Payout self-draw example:", payout_for_table(sb.total_points, winner=0, loser=None, win_source="self_draw"))
#'''

