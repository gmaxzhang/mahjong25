# algorithm/bc_dataset_returns_step0b.py
from __future__ import annotations
import argparse, random
from pathlib import Path

import numpy as np
from tqdm import trange

from algorithm.sim_and_train import Env, FlexibleAggroPolicyD, HybridAggroPolicy, AggroPolicy, FlexibleAggroPolicy
from algorithm.rules_io import load_rules
from algorithm.mahjongrl.env_wrapper import build_observation, compute_rl_reward

# Pool of strong teacher tags to randomize from (same as step0a)
POLICY_TAGS = ["aggro", "hyaggro", "flexaggro", "flexaggrod"]


def make_bot_for_tag(tag: str, seat: int, rules):
    """
    Same teacher-factory as in bc_dataset_step0a, but we don't
    need heads/acts here – we just let Env.step_turn(bots) drive.
    """
    tag = tag.lower()
    if tag == "hyaggro":
        return HybridAggroPolicy(seat, rules, None)
    elif tag == "aggro":
        return AggroPolicy(seat, rules, None)
    elif tag == "flexaggro":
        return FlexibleAggroPolicy(seat, rules, None)
    else:
        # default / flexaggrod
        return FlexibleAggroPolicyD(seat, rules, None)


def save_chunk(obs_list, G_list, outdir: str, idx: int):
    """
    Save a chunk of (obs, G) pairs for value pretraining.

    obs_list: list of np.array shape (D,)
    G_list:   list of float returns (same length as obs_list)
    """
    if not obs_list:
        return
    arr_obs = np.stack(obs_list).astype(np.float32)
    arr_G   = np.asarray(G_list, dtype=np.float32)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    out_path = Path(outdir) / f"chunk_{idx:05d}.npz"
    np.savez_compressed(out_path, obs=arr_obs, G=arr_G)


def simulate_value_dataset(
    rules,
    n_hands: int,
    outdir: str,
    gamma: float,
    base_seed: int,
) -> None:
    """
    Simulate hands with strong teacher bots and build a dataset
    of (observation, MC return) pairs for SEAT 0.

    For each episode:
      - run Env(rules, ...) with a randomized strong lineup
      - record every observation for seat 0 at its decision times
      - at terminal, compute RL reward for seat 0 via compute_rl_reward
      - assign a discounted MC return G_t = gamma^(T-1-t) * R_hand
        to each recorded state
    """
    random.seed(base_seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    all_obs: list[np.ndarray] = []
    all_G:   list[float]      = []

    for ep_idx in trange(n_hands, desc="Simulating value data"):
        # Fresh env + randomized strong lineup
        env = Env(rules, seed=random.randint(0, 1_000_000))
        lineup_tags = [random.choice(POLICY_TAGS) for _ in range(4)]
        bots = [make_bot_for_tag(tag, s, rules) for s, tag in enumerate(lineup_tags)]

        episode_obs: list[np.ndarray] = []

        # Play the hand
        while env.wall and not env.terminal:
            seat = env.turn
            if seat != 0:
                # Non-teacher seats: just let env drive with bots
                env.step_turn(bots)
                continue

            # Seat 0: record the state *before* the action
            obs = build_observation(env, seat=0)
            episode_obs.append(obs)

            # Let env/bot actually act and advance the game
            env.step_turn(bots)

        # Hand finished; compute RL reward for seat 0
        term = env.terminal or {}
        # Defensive fallback: if term is empty, mark as drawn_game
        term_for_reward = term if term else {"source": "drawn_game"}

        R0 = compute_rl_reward(term_for_reward, seat=0, rules=rules)

        T = len(episode_obs)
        if T == 0:
            # No decisions by seat 0 (edge case) – nothing to label
            continue

        # Monte Carlo returns for each recorded state.
        # If rewards are only terminal, this is gamma^(T-1-t) * R0.
        for t, obs in enumerate(episode_obs):
            G_t = (gamma ** (T - 1 - t)) * float(R0)
            all_obs.append(obs)
            all_G.append(G_t)

        # Periodic chunk save by episode index (like step0a)
        if ep_idx > 0 and ep_idx % 200 == 0:
            save_chunk(all_obs, all_G, outdir, ep_idx)
            all_obs, all_G = [], []

    # Final flush
    save_chunk(all_obs, all_G, outdir, n_hands)
    print(f"[done] simulated {n_hands} episodes for value pretraining → {outdir}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate (obs, return) dataset for value-head supervised pretraining."
    )
    ap.add_argument("--rules", required=True, help="Path to rules JSON")
    ap.add_argument("--hands", type=int, default=5000, help="Number of hands to simulate")
    ap.add_argument("--outdir", default="data/value_bc", help="Output folder for .npz chunks")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor (should match train.py --gamma)")
    ap.add_argument("--seed", type=int, default=123456, help="Base random seed")
    args = ap.parse_args()

    rules = load_rules(args.rules)
    simulate_value_dataset(
        rules=rules,
        n_hands=args.hands,
        outdir=args.outdir,
        gamma=args.gamma,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
