# algorithm/bc_dataset_step0a.py
from __future__ import annotations
import numpy as np, torch, random, argparse
from pathlib import Path
from tqdm import trange

from algorithm.sim_and_train import Env, FlexibleAggroPolicyD, HybridAggroPolicy, AggroPolicy, FlexibleAggroPolicy
from algorithm.rules_io import load_rules
from algorithm.mahjongrl.env_wrapper import build_observation, _tile_code
from algorithm.mahjongrl.train import TILE_TO_IDX, is_flower

# Map decision kind to numeric head
HEAD_MAP = {"discard":0, "binary":1, "pung":2, "chow":3, "kong":4}

# Pool of strong teacher tags to randomize from
POLICY_TAGS = ["aggro", "hyaggro", "flexaggro", "flexaggrod"]

# --- factory to build a bot for a given tag ---
def make_bot_for_tag(tag: str, seat: int, rules):
    """Return a bot object given a string tag."""
    tag = tag.lower()
    if tag == "hyaggro":
        return HybridAggroPolicy(seat, rules, None)
    elif tag == "aggro":
        return AggroPolicy(seat, rules, None)
    elif tag == "flexaggro":
        return FlexibleAggroPolicy(seat, rules, None)
    else:
        # If you later add AggroPolicy or FlexibleAggroPolicy, map them here
        return FlexibleAggroPolicyD(seat, rules, None)

# --- helper to save dataset chunks ---
def save_chunk(obs, heads, acts, tags, outdir, idx):
    if not obs:
        return
    arr_obs   = np.stack(obs)
    arr_heads = np.array(heads, np.int8)
    arr_acts  = np.array(acts,  np.int16)
    arr_tags  = np.array(tags,  np.str_)
    np.savez_compressed(
        f"{outdir}/chunk_{idx:05d}.npz",
        obs=arr_obs,
        heads=arr_heads,
        acts=arr_acts,
        tags=arr_tags,
    )

# --- main simulation loop ---
def simulate_dataset(rules, n_hands:int, outdir:str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    all_obs, all_heads, all_actions, all_tags = [], [], [], []

    for i in trange(n_hands, desc="Simulating BC data"):
        env = Env(rules, seed=random.randint(0, 1_000_000))

        # Randomize lineup across strong teacher bots
        lineup_tags = [random.choice(POLICY_TAGS) for _ in range(4)]
        bots = [make_bot_for_tag(tag, s, rules) for s, tag in enumerate(lineup_tags)]

        # Simulate a full hand
        while env.wall and not env.terminal:
            seat = env.turn
            if seat != 0:
                env.step_turn(bots)
                continue

            # Seat 0 (the teacher) — record before acting
            obs = build_observation(env, seat=0)
            kind = getattr(env, "pending_kind", "discard")
            head = HEAD_MAP.get(kind, 0)
            bot = bots[0]

            if kind == "discard":
                tile = bot.pick_discard(env)
                idx = TILE_TO_IDX.get(tile, 0)
            elif kind == "binary":
                yes = bot.decide_ron(
                    env,
                    getattr(env, "last_discard_tile", None),
                    getattr(env, "points", None),
                    getattr(env, "ron_loser", None),
                )
                idx = 1 if yes else 0
            elif kind == "pung":
                last = getattr(env, "last_discard", None)
                tile = last[1] if last else None
                idx = 1 if (tile and bot.decide_pung(env, seat, tile)) else 0
            elif kind == "chow":
                last = getattr(env, "last_discard", None)
                discarder, tile = last if last else (None, None)
                chow_sets = []
                if tile and _tile_code(tile) not in ("E","S","W","N","C","F","B"):
                    r, suit = int(tile[0]), tile[1]
                    for a,b in [(r-2,r-1),(r-1,r+1),(r+1,r+2)]:
                        if 1 <= a <= 9 and 1 <= b <= 9:
                            A, B = f"{a}{suit}", f"{b}{suit}"
                            if A in bots[0].concealed and B in bots[0].concealed:
                                chow_sets.append((A,B))
                idx = 1 if (chow_sets and bot.choose_chow(env, seat, tile, chow_sets)) else 0
            else:
                idx = 0

            # Store the observation + label
            all_obs.append(obs)
            all_heads.append(head)
            all_actions.append(idx)
            all_tags.append(lineup_tags[0])  # which teacher produced it

            env.step_turn(bots)

        if i % 200 == 0 and i > 0:
            save_chunk(all_obs, all_heads, all_actions, all_tags, outdir, i)
            all_obs, all_heads, all_actions, all_tags = [], [], [], []

    save_chunk(all_obs, all_heads, all_actions, all_tags, outdir, n_hands)
    print(f"[done] saved {n_hands} simulated teacher hands → {outdir}")

# --- CLI entrypoint ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate randomized BC dataset from strong teacher bots.")
    ap.add_argument("--rules", required=True, help="Path to rules JSON")
    ap.add_argument("--hands", type=int, default=5000)
    ap.add_argument("--outdir", default="data/bc_randomized")
    args = ap.parse_args()

    simulate_dataset(load_rules(args.rules), args.hands, args.outdir)
