# We'll overwrite /mnt/data/custom_mahjong_rules.py to add JSON-driven rules support,
# and create two new files:
#  - /mnt/data/rules_io.py : loader/validator for the JSON rules
#  - /mnt/data/sim_and_train.py : simple simulator, data generation, and ML training stubs
from pathlib import Path

custom_code = r'''# -*- coding: utf-8 -*-
"""
Custom Mahjong project scaffolding (Rules-driven)
=================================================

This version makes `score_hand` rules-driven. You can pass a `rules: dict` loaded
from JSON (see rules_io.load_rules). If None, defaults are used.

Tile format: "1b","9t","5w", honors "E,S,W,N,C,F,B", flowers "F1"...
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable, Sequence, Literal
from collections import Counter

# ----------------------- Tile encoding -----------------------

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
    tiles: Tuple[str, ...]             # 3 or 4 for sets, 2 for pair
    open: bool = False
    formed_by_claim: bool = False

    def key(self) -> Tuple:
        return (self.type, tuple(sorted(self.tiles)), self.open)

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
    return len(suits) == 1 and len(honors) == 0

def tiles_mixed_one_color(tiles: Iterable[str], eyes: Sequence[str]) -> bool:
    suits = {suit_of(t) for t in tiles if is_suited(t)}
    honors_in_eyes = all(is_honor(t) for t in eyes) and len(eyes)==2 and eyes[0]==eyes[1]
    return len(suits) == 1 and honors_in_eyes

def is_peng_peng_hu(melds: Sequence[Meld]) -> bool:
    types = {m.type for m in melds if m.type != "pair"}
    return "chow" not in types and len(types) > 0

def is_eating_hand(melds: Sequence[Meld]) -> bool:
    sets_only = [m for m in melds if m.type in {"chow","pung","kong"}]
    return len(sets_only)==4 and all(m.formed_by_claim for m in sets_only)

def is_seven_pairs(all_tiles: List[str]) -> bool:
    tiles = [t for t in all_tiles if not is_flower(t)]
    cnt = Counter(tiles)
    return sum(v==2 for v in cnt.values())==7 and all(v in (2,4) for v in cnt.values())

def split_into_sets_and_pair(all_tiles: List[str]) -> Optional[Tuple[List[Meld], Tuple[str,str]]]:
    tiles = sorted([t for t in all_tiles if not is_flower(t)])
    if len(tiles) != 14:
        return None
    cnt = Counter(tiles)

    def backtrack(current_sets: List[Meld]) -> Optional[Tuple[List[Meld], Tuple[str,str]]]:
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
            y = f"{r+1}{s}" if r and r<=7 else None
            z = f"{r+2}{s}" if r and r<=7 else None
            if y in remains and z in remains:
                m = Meld("chow", (x,y,z), open=False)
                res = backtrack(current_sets+[m])
                if res: return res
        # kong (treated as a set for structure)
        if remains.count(x) == 4:
            m = Meld("kong", (x,x,x,x), open=False)
            res = backtrack(current_sets+[m])
            if res: return res
        return None

    return backtrack([])

# ----------------------- Flower points -----------------------

def flower_points_from_sets(melds: Sequence[Meld], rules: Dict) -> int:
    set_pts = rules["flower_points"]["set_points"]
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

        key = "suit"
        if is_dragon(base): key = "dragon"
        elif is_wind(base): key = "wind"

        if is_pung_set:
            pts += set_pts[key]["pung"]
        if open_kong:
            pts += set_pts[key]["kong_open"]
        if closed_kong:
            pts += set_pts[key]["kong_closed"]
    return pts

# ----------------------- Special hands -----------------------

def all_apart(tiles14_with_flowers: List[str]) -> bool:
    tiles = tiles14_with_flowers[:]
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
    if len(set(honors)) != len(honors):
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

def score_hand(state: HandState, rules: Optional[Dict] = None) -> ScoreBreakdown:
    rules = rules or DEFAULT_RULES
    base_default = rules["base_scoring"]["base_default"]
    back_bonus_val = rules["base_scoring"]["back_wall_bonus"]

    # Gather all 14 tiles in winning hand (exclude played flowers in meld logic).
    all14 = [t for t in state.concealed if not is_flower(t)] + \
            [x for m in state.melds for x in m.tiles if m.type!="pair"] + \
            [state.winning_tile]

    # Flower points: flowers played + from sets per rules
    flower_pts = len(state.flowers) * rules["flower_points"]["per_flower_tile"] + \
                 flower_points_from_sets(state.melds, rules)

    # Minimum flower rule
    if state.win_source == "discard":
        need = rules["win_requirements"]["ron_min_flower_points_if_any_claimed_set"]
        exempt = rules["win_requirements"]["exempt_if_menzen_before_ron"] and (not state.used_any_claim_for_sets)
        if (not exempt) and (flower_pts < need):
            raise ValueError("Invalid win by discard: requires at least "
                             f"{need} flower points when claimed sets were used.")
    elif state.win_source == "self_draw":
        if rules["win_requirements"]["tsumo_requires_min_flowers"]:
            need = rules["win_requirements"]["ron_min_flower_points_if_any_claimed_set"]
            if flower_pts < need:
                raise ValueError("Invalid self-draw: requires min flower points per rules.")

    base = base_default + flower_pts
    specials: Dict[str,int] = {}

    tiles_no_flowers = [t for t in all14 if not is_flower(t)]
    struct = split_into_sets_and_pair(tiles_no_flowers)

    # Specials (stacking add-ons calculated as (target_base - base_default))
    sh = rules["special_hands"]

    if is_seven_pairs(tiles_no_flowers):
        specials["Seven Pairs"] = sh["seven_pairs"] - base_default

    if all_apart(tiles_no_flowers + state.flowers):
        specials["All Apart"] = sh["all_apart"] - base_default

    if struct:
        sets, eyes = struct
        if tiles_one_color(tiles_no_flowers):
            specials["One Color"] = sh["one_color"] - base_default
        elif tiles_mixed_one_color(tiles_no_flowers, eyes):
            specials["Mixed One Color"] = sh["mixed_one_color"] - base_default
        if is_peng_peng_hu(sets):
            specials["Peng-peng-hu"] = sh["peng_peng_hu"]
        if is_eating_hand(sets):
            specials["Eating Hand"] = sh["eating_hand"]

    back_bonus = back_bonus_val if state.back_wall_bonus else 0
    total = base + sum(specials.values()) + back_bonus

    return ScoreBreakdown(
        base_points=base_default,
        flower_points=flower_pts,
        specials=specials,
        back_wall_bonus=back_bonus,
        total_points=total
    )

# ----------------------- Payouts -----------------------

def payout_for_table(hand_points: int, winner: int, loser: Optional[int], num_players: int = 4,
                     win_source: Literal["self_draw","discard"] = "self_draw") -> List[int]:
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
'''

rules_io_code = r'''# -*- coding: utf-8 -*-
"""
rules_io.py
-----------
Utilities to load and validate rule JSON files for the custom Mahjong scorer.
"""

from __future__ import annotations
import json
from typing import Dict

_SCHEMA_KEYS = {
    "base_scoring": {"base_default", "back_wall_bonus", "specials_stack"},
    "flower_points": {"per_flower_tile", "set_points"},
    "win_requirements": {"ron_min_flower_points_if_any_claimed_set", "exempt_if_menzen_before_ron", "tsumo_requires_min_flowers"},
    "special_hands": {"one_color","mixed_one_color","all_apart","seven_pairs","peng_peng_hu","eating_hand"}
}

def _validate_rules(rules: Dict) -> None:
    for k in _SCHEMA_KEYS:
        if k not in rules:
            raise ValueError(f"Missing top-level key: {k}")
        missing = _SCHEMA_KEYS[k] - set(rules[k].keys())
        if missing:
            raise ValueError(f"Missing keys under '{k}': {sorted(missing)}")
    sp = rules["flower_points"]["set_points"]
    for group in ("dragon","wind","suit"):
        if group not in sp:
            raise ValueError(f"Missing set_points group: {group}")
        for sub in ("pung","kong_open","kong_closed"):
            if sub not in sp[group]:
                raise ValueError(f"Missing set_points[{group}].{sub}")

def load_rules(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    _validate_rules(rules)
    return rules
'''

sim_and_train_code = r'''# -*- coding: utf-8 -*-
"""
sim_and_train.py
----------------
Simple simulation harness, data generation, and (optional) ML training stubs.

- Generates self-play episodes under your rules and writes JSONL trajectories.
- Provides a minimal random/heuristic agent.
- If PyTorch is available, contains a tiny policy/value net and a training loop stub.
"""

from __future__ import annotations
import os, json, random, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter

from rules_io import load_rules
from custom_mahjong_rules import HandState, Meld, score_hand

# ---------------------------- Utilities ----------------------------

ALL_TILES = [f"{r}{s}" for s in "bwt" for r in range(1,10)] + ["E","S","W","N","C","F","B"]
# For simplicity, no duplicate flowers here; gameplay will add flowers when drawn.
FOUR_OF_EACH = [t for t in ALL_TILES for _ in range(4)]

def make_wall(include_flowers: bool = True, seed: Optional[int]=None) -> List[str]:
    rng = random.Random(seed)
    wall = FOUR_OF_EACH[:]
    if include_flowers:
        wall += [f"F{i}" for i in range(1,9)]  # 8 flowers as example
    rng.shuffle(wall)
    return wall

# ---------------------------- Env ----------------------------

@dataclass
class PlayerView:
    concealed: List[str]
    melds: List[Meld]
    flowers: List[str]
    used_claim: bool = False

class SimpleEnv:
    """
    Super-minimal turn-based simulator sufficient for data generation.
    - Deals 13 tiles to each player (east gets 14 and discards once).
    - On draw: auto-play flowers (replace from back).
    - Random agents discard a random non-flower tile.
    - Win is detected only when a player calls 'win' externally
      (we provide helper to check if a 14-tile hand can be split into sets+pair, or seven pairs).
    This is NOT a complete rules engine, but enough to construct trajectories and train a starter model.
    """

    def __init__(self, rules: Dict, seed: Optional[int]=None):
        self.rules = rules
        self.rng = random.Random(seed)
        self.reset(seed)

    def reset(self, seed: Optional[int]=None):
        self.wall = deque(make_wall(include_flowers=True, seed=seed))
        self.back = deque()  # not modeling separate back; we will pop from right as back
        while self.wall:
            self.back.append(self.wall.pop())
        self.players = [PlayerView([], [], []) for _ in range(4)]
        # deal
        for _ in range(13):
            for p in self.players:
                self._draw_into(p, from_back=False)
        # dealer extra draw & discard
        self.turn = 0
        self.discards: List[List[str]] = [[] for _ in range(4)]
        self.last_discard: Optional[Tuple[int,str]] = None
        self._draw_into(self.players[0], from_back=False)
        # Auto-play flowers for all players
        for i in range(4):
            self._settle_flowers(i)
        return self._obs()

    def _draw_into(self, p: PlayerView, from_back: bool):
        src = self.back if from_back else self.wall
        if not src: return
        t = src.pop() if from_back else src.popleft()
        p.concealed.append(t)

    def _settle_flowers(self, seat: int):
        p = self.players[seat]
        changed = True
        while changed:
            changed = False
            for t in list(p.concealed):
                if t.startswith("F"):
                    p.concealed.remove(t)
                    p.flowers.append(t)
                    # replacement from back
                    self._draw_into(p, from_back=True)
                    changed = True

    def legal_discards(self, seat: int) -> List[str]:
        # any non-flower in concealed
        return [t for t in self.players[seat].concealed if not t.startswith("F")]

    def step_discard(self, seat: int, tile: str):
        p = self.players[seat]
        assert tile in p.concealed
        p.concealed.remove(tile)
        self.discards[seat].append(tile)
        self.last_discard = (seat, tile)
        # Next player's turn
        self.turn = (self.turn + 1) % 4
        # Draw for next
        self._draw_into(self.players[self.turn], from_back=False)
        self._settle_flowers(self.turn)

    def hand_tiles_for_scoring(self, seat: int, winning_tile: str, win_source: str) -> HandState:
        p = self.players[seat]
        return HandState(
            flowers=p.flowers[:],
            concealed=[x for x in p.concealed if x != winning_tile],
            melds=p.melds[:],
            winning_tile=winning_tile,
            win_source=win_source,  # "self_draw" or "discard"
            back_wall_bonus=False,
            used_any_claim_for_sets=p.used_claim
        )

    def _obs(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "handsizes": [len(p.concealed) for p in self.players],
            "wall_front": len(self.wall),
            "wall_back": len(self.back),
            "discards": [d[:] for d in self.discards]
        }

# ---------------------------- Random agent & rollout ----------------------------

def can_win_now(state: HandState, rules: Dict) -> bool:
    # Quick legality gate: try scoring; if invalid discard-win, it will throw
    try:
        from custom_mahjong_rules import split_into_sets_and_pair, is_seven_pairs
        tiles = [t for t in state.concealed if not t.startswith("F")] + [state.winning_tile]
        return (split_into_sets_and_pair(tiles) is not None) or is_seven_pairs(tiles)
    except Exception:
        return False

def random_episode(rules: Dict, seed: Optional[int]=None, max_steps: int = 200) -> Dict[str, Any]:
    env = SimpleEnv(rules, seed=seed)
    traj = {"steps": [], "terminal": None}
    steps = 0
    while steps < max_steps and env.wall:
        seat = env.turn
        # naive: try to win on draw if possible
        drawn = env.players[seat].concealed[-1] if env.players[seat].concealed else None
        if drawn:
            hs = env.hand_tiles_for_scoring(seat, drawn, "self_draw")
            if can_win_now(hs, rules):
                sb = score_hand(hs, rules)
                traj["terminal"] = {"winner": seat, "source": "self_draw", "points": sb.total_points}
                return traj
        # otherwise discard randomly
        options = env.legal_discards(seat)
        if not options: break
        choice = random.choice(options)
        env.step_discard(seat, choice)
        steps += 1
    traj["terminal"] = {"winner": None, "source": "drawn_game", "points": 0}
    return traj

def generate_jsonl(path: str, n: int, rules: Dict, seed: Optional[int]=None):
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            ep = random_episode(rules, seed=rng.randint(1,10**9))
            f.write(json.dumps(ep)+"\n")

# ---------------------------- ML training stub ----------------------------
# If PyTorch is installed, you can extend this with a tiny policy/value network.
# Here we only include a placeholder function.

def train_from_jsonl(data_path: str):
    """
    Placeholder: read trajectories and print a tiny summary.
    Extend with torch to implement BC/actor-critic.
    """
    total = 0
    wins = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            total += 1
            term = ep["terminal"]
            if term and term.get("winner") is not None:
                wins += 1
    print(f"Read {total} episodes. Non-draw wins: {wins}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", type=str, required=True)
    ap.add_argument("--out", type=str, default="episodes.jsonl")
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()
    rules = load_rules(args.rules)
    generate_jsonl(args.out, args.episodes, rules)
    train_from_jsonl(args.out)
'''

root = Path("algorithm")

(root / "custom_mahjong_rules.py").write_text(custom_code, encoding="utf-8")
(root / "rules_io.py").write_text(rules_io_code, encoding="utf-8")
(root / "sim_and_train.py").write_text(sim_and_train_code, encoding="utf-8")
print("Wrote files:")
print(" - mahjong25/algorithm/custom_mahjong_rules.py")
print(" - mahjong25/algorithm/rules_io.py")
print(" - mahjong25/algorithm/sim_and_train.py")
