# Write a JSON rules file that matches your family's rules and exposes the "min flower" requirement.
import json, pathlib, textwrap

rules = {
  "name": "family_rules_v1",
  "tileset": {
    "use_flowers": True,
    "honors": ["E","S","W","N","C","F","B"],
    "suits": ["b","w","t"]
  },
  "base_scoring": {
    "base_default": 10,
    "back_wall_bonus": 5,
    "specials_stack": True  # special-hand bases add together
  },
  "flower_points": {
    "per_flower_tile": 1,
    "set_points": {
      "dragon": {"pung": 2, "kong_open": 3, "kong_closed": 4},
      "wind":   {"pung": 1, "kong_open": 2, "kong_closed": 3},
      "suit":   {"pung": 0, "kong_open": 1, "kong_closed": 2},
      "pair_counts": False  # pairs never add flower points
    }
  },
  "win_requirements": {
    # My interpretation of your rule:
    # For a RON (win on opponent discard), you need at least this many "flower points"
    # IF you previously used any opponent discard to form a set/run (chi/pung/kong).
    "ron_min_flower_points_if_any_claimed_set": 2,
    # If you were MENZEN (no claimed tiles used to form sets/runs) before the final ron tile,
    # the minimum does NOT apply—even though the final tile is a discard.
    "exempt_if_menzen_before_ron": True,
    # No minimum for self-draw wins.
    "tsumo_requires_min_flowers": False
  },
  "special_hands": {
    # Each value is the total base the hand is worth; the scorer adds (value - base_default).
    "one_color": 40,
    "mixed_one_color": 20,
    "all_apart": 20,
    "seven_pairs": 40,
    "peng_peng_hu": 20,
    "eating_hand": 20
  },
  "payouts": {
    "on_discard": {"winner_gain": 1, "loser_loss": -1, "others": 0},
    "on_self_draw": {"winner_gain": 3, "others_loss_each": -1}
  },
  "notes": textwrap.dedent("""\
    - "Flower points" = number of flower tiles you played + set-based bonuses from the table above.
    - Pairs do not contribute flower points.
    - The ron minimum (2 flower points) applies only if you had claimed any tile earlier to form a set/run.
      If you were closed (menzen) right up until the winning discard, you are exempt.
    - Back-wall bonus (+5) is added when drawing from the back after a flower or kong replacement.
  """)
}

path = pathlib.Path("/mnt/data/family_rules_v1.json")
path.write_text(json.dumps(rules, indent=2))
print("Wrote", path)
