# -*- coding: utf-8 -*-
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
