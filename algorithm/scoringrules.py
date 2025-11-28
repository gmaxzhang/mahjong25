# Write a JSON rules file that matches your family's rules and exposes the "min flower" requirement.
import json, pathlib, textwrap

rules = {
    "name": "family_rules_v1",
    "tileset": {
      "use_flowers": True,
      "honors": ["E", "S", "W", "N", "C", "F", "B"],
      "suits": ["b", "w", "t"]
    },
    "base_scoring": {
      "base_default": 10,
      "back_wall_bonus": 5,
      "specials_stack": True
    },
    "flower_points": {
      "per_flower_tile": 1,
      "set_points": {
        "dragon": { "pung": 2, "kong_open": 3, "kong_closed": 4 },
        "wind":   { "pung": 1, "kong_open": 2, "kong_closed": 3 },
        "suit":   { "pung": 0, "kong_open": 1, "kong_closed": 2 }
      }
    },
    "win_requirements": {
      "allow_pass_ron": True,
      "ron_min_flower_points_if_any_claimed_set": 0,
      "exempt_if_menzen_before_ron": True,
      "tsumo_requires_min_flowers": False,
      "menqing_required_for_10_11": True
    },
    "special_hands": {
      "one_color": 40,
      "mixed_one_color": 20,
      "all_apart": 20,
      "seven_pairs": 40,
      "peng_peng_hu": 20,
      "eating_hand": 20
    },
    "payouts": {
      "on_discard":   { "winner_gain": 1, "loser_loss": -1, "others": 0 },
      "on_self_draw": { "winner_gain": 3, "others_loss_each": -1 }
    },
    "special_events": {
      "four_same_consecutive_discard": {
        "enabled": True,
        "amount_per_opponent": 1
      }
    },
    "ai": {
      "tsumo_ev_multiplier": 1.05,      
      "ron_bias": 0.80,                 
      "feed_risk_coeff": 0.012,          
      "discard_keep_weight": 1.0,
      "discard_risk_weight": 0.6,      
  
      "claim_aggressiveness": 0.85,     
      "pung_margin": 0.20,              
      "kong_margin": 0.25,               
      "chow_bias": 0.85                  
    },
    "notes": "Ron has no flower minimum even after claims; menqing only required for 10/11-point hands; tsumo has no flower minimum."
  }

path = pathlib.Path("/algorithm/familyrules.json")
path.write_text(json.dumps(rules, indent=2))
print("Wrote", path)
