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

# Suits: bamboo=b, character=w, circle=t
# Honors: winds E,S,W,N; dragons C (zhong/red), F (fa/green), B (bai/white)
#
# We'll store tiles as simple strings, e.g. "1b","9t","5w","E","C".
# A "flower" is any token "F1","F2",... (we only need count).

def is_flower(x: str) -> bool:
    return x.startswith("F")

def is_honor(x: str) -> bool:
    return x in {"E","S","W","N","C","F","B"}

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
    return is_suited(x) and rank_of(x) in {1,9}

# ----------------------- Meld representation -----------------------

MeldType = Literal["chow","pung","kong","pair"]

@dataclass(frozen=True)
class Meld:
    type: MeldType
    tiles: Tuple[str, ...]             # 3 or 4 for sets, 2 for pair, 3 in sequence for chow
    open: bool = False                 # open (claimed) or closed
    formed_by_claim: bool = False      # whether this exact meld involved claiming an opponent's discard

    def key(self) -> Tuple:
        # normalized key for identity
        return (self.type, tuple(sorted(self.tiles)), self.open)

# ----------------------- Hand state -----------------------

@dataclass
class HandState:
    flowers: List[str]                         # list of flower tiles (played)
    concealed: List[str]                       # concealed tiles not formed into open melds yet
    melds: List[Meld]                          # current melds (some may be closed kongs/pungs/chows)
    winning_tile: str                          # the tile that completed the hand
    win_source: Literal["self_draw","discard"] # how the hand was won
    back_wall_bonus: bool = False              # +5 if drawn from back after flower or kong (per rules)
    # Book-keeping: whether the player ever claimed a discard to build ANY meld (not just the last one)
    used_any_claim_for_sets: bool = False

# ----------------------- Pattern helpers -----------------------

def is_chow(tiles: Sequence[str]) -> bool:
    if len(tiles) != 3: return False
    a,b,c = sorted(tiles, key=lambda t: (suit_of(t), rank_of(t)))
    if not (is_suited(a) and is_suited(b) and is_suited(c)): return False
    if suit_of(a) != suit_of(b) or suit_of(b) != suit_of(c): return False
    ra,rb,rc = rank_of(a), rank_of(b), rank_of(c)
    return rb == ra+1 and rc == rb+1

def is_pung(tiles: Sequence[str]) -> bool:
    return len(tiles)==3 and len(set(tiles))==1

def is_kong(tiles: Sequence[str]) -> bool:
    return len(tiles)==4 and len(set(tiles))==1

def is_pair(tiles: Sequence[str]) -> bool:
    return len(tiles)==2 and len(set(tiles))==1

def tiles_one_color(tiles: Iterable[str]) -> bool:
    suits = {suit_of(t) for t in tiles if is_suited(t)}
    honors = [t for t in tiles if is_honor(t)]
    return len(suits) == 1 and len(honors) == 0  # pure suit, no honors

def tiles_mixed_one_color(tiles: Iterable[str], eyes: Sequence[str]) -> bool:
    suits = {suit_of(t) for t in tiles if is_suited(t)}
    honors_in_eyes = all(is_honor(t) for t in eyes) and len(eyes)==2 and eyes[0]==eyes[1]
    return len(suits) == 1 and honors_in_eyes

def is_peng_peng_hu(melds: Sequence[Meld]) -> bool:
    # all sets are triplets/quads (no chows)
    types = {m.type for m in melds if m.type != "pair"}
    return "chow" not in types

def is_eating_hand(melds: Sequence[Meld]) -> bool:
    # all four sets were formed by taking discards
    sets_only = [m for m in melds if m.type in {"chow","pung","kong"}]
    return len(sets_only)==4 and all(m.formed_by_claim for m in sets_only)

def is_seven_pairs(all_tiles: List[str]) -> bool:
    # ignores flowers
    tiles = [t for t in all_tiles if not is_flower(t)]
    cnt = Counter(tiles)
    return sum(v==2 for v in cnt.values())==7 and all(v in (2,4) for v in cnt.values())

def split_into_sets_and_pair(all_tiles: List[str]) -> Optional[Tuple[List[Meld], Tuple[str,str]]]:
    """
    Try to split 14 (non-flower) tiles into 4 sets (chow/pung/kong) + 1 pair.
    This is a simple backtracking; good enough for scoring logic.
    """
    tiles = sorted([t for t in all_tiles if not is_flower(t)])
    if len(tiles) != 14:
        return None

    cnt = Counter(tiles)

    def backtrack(current_sets: List[Meld]) -> Optional[Tuple[List[Meld], Tuple[str,str]]]:
        # if 4 sets chosen, what's left should be the pair
        taken = Counter()
        for m in current_sets:
            for x in m.tiles: taken[x]+=1
        remains = []
        for x,c in cnt.items():
            r = c - taken[x]
            remains += [x]*r
        if len(current_sets)==4:
            if len(remains)==2 and remains[0]==remains[1]:
                return current_sets, (remains[0], remains[1])
            return None

        # pick smallest remaining tile
        if not remains: return None
        x = remains[0]

        # try pung
        if remains.count(x) >= 3:
            m = Meld("pung", (x,x,x), open=False)
            res = backtrack(current_sets+[m])
            if res: return res

        # try chow
        if is_suited(x):
            s = suit_of(x); r = rank_of(x)
            y = f"{r+1}{s}" if r and r<=7 else None
            z = f"{r+2}{s}" if r and r<=7 else None
            if y in remains and z in remains:
                m = Meld("chow", (x,y,z), open=False)
                res = backtrack(current_sets+[m])
                if res: return res

        # try kong (treat as pung for base structure)
        if remains.count(x) == 4:
            m = Meld("kong", (x,x,x,x), open=False)
            res = backtrack(current_sets+[m])
            if res: return res

        return None

    return backtrack([])

# ----------------------- Flower points -----------------------

def flower_points_from_sets(melds: Sequence[Meld]) -> int:
    """
    Family rule (interpreting an apparent typo):
    - Dragons:  pung=2, open kong=3, closed kong=4
    - Winds:    pung=1, open kong=2, closed kong=3   <-- Assumption: the second "Dragons" was meant "Winds"
    - Suits:    pung=0, open kong=1, closed kong=2
    Pairs do not give flower points.
    """
    pts = 0
    for m in melds:
        if m.type == "pair": 
            continue
        tiles = m.tiles
        base = tiles[0]
        is_open = m.open
        closed_kong = (m.type=="kong" and not is_open)
        open_kong   = (m.type=="kong" and is_open)
        is_pung_set = (m.type=="pung")
        # category:
        if is_dragon(base):
            if is_pung_set: pts += 2
            if open_kong:   pts += 3
            if closed_kong: pts += 4
        elif is_wind(base):
            if is_pung_set: pts += 1
            if open_kong:   pts += 2
            if closed_kong: pts += 3
        elif is_suited(base):
            if is_pung_set: pts += 0
            if open_kong:   pts += 1
            if closed_kong: pts += 2
    return pts

# ----------------------- Special hands -----------------------

def all_apart(tiles14: List[str]) -> bool:
    """
    All Apart (custom family hand):
    - For each suit (b,w,t), you have exactly one KNITTED group among {147},{258},{369}
      (not necessarily a chow; they're singletons by rank class).
    - Remaining 5 tiles are unique honors and/or at most 1 flower.
    - This hand does not follow 4-sets+pair.
    Implementation detail: we require exactly 9 suited tiles forming one of the three "knitted" groups
    in each suit (3*3) and the other 5 tiles are distinct honors (and <=1 flower across the whole hand).
    """
    tiles = tiles14[:]
    flowers = [x for x in tiles if is_flower(x)]
    if len(flowers) > 1: 
        return False
    suited = [x for x in tiles if is_suited(x)]
    honors = [x for x in tiles if is_honor(x)]
    if len(suited) != 9 or len(honors) + len(flowers) != 5:
        return False
    def fits_suit(group: set[int], s: str) -> bool:
        ranks = sorted(rank_of(x) for x in suited if suit_of(x)==s)
        return sorted(group) == ranks
    groups = [{1,4,7},{2,5,8},{3,6,9}]
    ok = any(fits_suit(g, 'b') for g in groups) \
         and any(fits_suit(g, 'w') for g in groups) \
         and any(fits_suit(g, 't') for g in groups)
    if not ok: 
        return False
    # Honors must be unique
    if len(set(honors)) != len(honors):
        return False
    return True

# ----------------------- Scoring -----------------------

@dataclass
class ScoreBreakdown:
    base_points: int
    flower_points: int
    specials: Dict[str,int] = field(default_factory=dict)  # name -> points
    back_wall_bonus: int = 0
    total_points: int = 0

def score_hand(state: HandState) -> ScoreBreakdown:
    """Compute the hand's point value per family rules."""
    # Gather all 14 tiles in the winning hand (exclude played flowers from meld logic).
    all14 = [t for t in state.concealed if not is_flower(t)] + \
            [x for m in state.melds for x in m.tiles if m.type!="pair"] + \
            [state.winning_tile]
    # Note: pairs might already be in concealed if 4-sets+pair was accomplished via backtracking.

    # Base score starts at 10; each "flower point" adds +1
    # First compute flower points: from flowers + from set categories
    flower_pts = len(state.flowers) + flower_points_from_sets(state.melds)

    # If winning via discard: need 2 flower points unless no sets/runs were ever claimed.
    if state.win_source == "discard":
        if state.used_any_claim_for_sets and flower_pts < 2:
            # Not enough to claim a discard win
            raise ValueError("Invalid win by discard: requires at least 2 flower points when claims were used.")

    base = 10 + flower_pts

    specials: Dict[str,int] = {}

    # Special hands (these add to base, they stack):
    # One Color (pure suit, no honors) = +30 net vs base 10 → we model as +30 so total becomes 40
    tiles_no_flowers = [t for t in all14 if not is_flower(t)]
    # Try to get sets/pair structure (for checking some specials)
    struct = split_into_sets_and_pair(tiles_no_flowers)

    # Seven Pairs (40 base instead of 10)
    if is_seven_pairs(tiles_no_flowers):
        specials["Seven Pairs"] = 40 - 10  # add-on above base

    # All Apart (20 base)
    if all_apart(tiles_no_flowers + state.flowers):
        specials["All Apart"] = 20 - 10

    if struct:
        sets, eyes = struct
        # One Color (all b or w or t, no honors)
        if tiles_one_color(tiles_no_flowers):
            specials["One Color"] = 40 - 10
        # Mixed One Color (4 sets pure suit, eyes are honors)
        elif tiles_mixed_one_color(tiles_no_flowers, eyes):
            specials["Mixed One Color"] = 20 - 10
        # Peng-peng-hu (all pungs/kongs)
        if is_peng_peng_hu(sets):
            specials["Peng-peng-hu"] = 20
        # Eating hand (all 4 sets formed by claiming discards)
        if is_eating_hand(sets):
            specials["Eating Hand"] = 20

    # Back-wall / replacement draw bonus (+5)
    back_bonus = 5 if state.back_wall_bonus else 0

    total = base + sum(specials.values()) + back_bonus

    return ScoreBreakdown(
        base_points=10,
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

