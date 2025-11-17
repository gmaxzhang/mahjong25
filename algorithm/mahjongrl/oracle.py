# mahjong_rl/oracle.py
from __future__ import annotations
from typing import Optional, Tuple, List
from collections import Counter

from sim_and_train import (
    is_flower, _is_suit_tile, _tile_rank_suit,
    composite_shape_metric, rough_shanten_like, rough_ukeire, chiitoi_shanten_like
)

def oracle_discard(env, seat: int, legal_discards: List[str]) -> Optional[int]:
    """Choose discard minimizing shape + light danger (mirrors WinProb/Payout)."""
    p = env.players[seat]
    hand = [t for t in p.concealed if not is_flower(t)]
    cnt = Counter(hand)
    best_i, best_val = None, 1e9
    for i, t in enumerate(legal_discards):
        h2 = hand[:]
        if t in h2: h2.remove(t)
        # reuse the same composite metric the sim uses
        w = 0.08  # small chiitoi emphasis as a default
        comp = composite_shape_metric(h2, w)
        # cheap isolate bonus
        iso = -0.5 if (cnt[t] == 1 and _is_suit_tile(t) and all(
            t2 not in cnt or cnt[t2] == 0
            for dr in (-2,-1,1,2)
            for t2 in [f"{_tile_rank_suit(t)[0]+dr}{_tile_rank_suit(t)[1]}"]
            if 1 <= _tile_rank_suit(t)[0]+dr <= 9
        )) else 0.0
        val = comp + iso
        if val < best_val:
            best_val, best_i = val, i
    return best_i

def oracle_yesno(default_prob_yes: float = 0.8) -> int:
    """Bias toward accepting useful claims."""
    # 1 = yes, 0 = no
    import random
    return 1 if random.random() < default_prob_yes else 0

def oracle_chow(chow_sets: List[Tuple[str,str]], env, seat) -> Optional[int]:
    """Pick the chow pair that best improves composite."""
    if not chow_sets: return None
    p = env.players[seat]
    hand = [t for t in p.concealed if not is_flower(t)]
    w = 0.08
    before = composite_shape_metric(hand, w)
    best_idx, best_val = None, 1e9
    for i,(a,b) in enumerate(chow_sets):
        h2 = hand[:]
        if a in h2: h2.remove(a)
        if b in h2: h2.remove(b)
        val = composite_shape_metric(h2, w)
        if val < best_val:
            best_val, best_idx = val, i
    return best_idx
