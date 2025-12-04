# algorithm/mahjongrl/sim_compare.py
import argparse, random, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Dict, Any, List

from algorithm.rules_io import load_rules
from algorithm.sim_and_train import run_episode
import torch
from algorithm.mahjongrl.model import LSTMActorCritic, ACConfig
from algorithm.mahjongrl.agent import RLPolicy


# Step 1: define the configuration (must match training)
cfg = ACConfig(obs_dim=468, hidden=256, lstm=256)

# Step 2: create the model
model = LSTMActorCritic(cfg)

ckpt = torch.load("runs/mahjongbaserun/checkpoints/last.pt", map_location="cpu")

# The actual weights are stored under "model_state"
state_dict = ckpt["model_state"]   # or ckpt.get("model_state")

model.load_state_dict(state_dict)
model.eval()


VALID_POLICIES = [
    "random",
    "wp",
    "payout",
    "hybrid",
    "aggro",
    "hyaggro",
    "flexaggro",
    "flexaggrod"
    #"rl"
]

# ----------------------------------------------------------------------
# --- Exact per-seat scoring (matches summarize_jsonl in sim_and_train) ---
# ----------------------------------------------------------------------

def _episode_delta_for_seats(term: Dict[str, Any], rules: Dict[str, Any]) -> List[int]:
    delta = [0, 0, 0, 0]

    pay_disc = rules.get("payouts", {}).get("on_discard", {})
    pay_self = rules.get("payouts", {}).get("on_self_draw", {})

    w = term.get("winner")
    winners_list = term.get("winners")

    if w is None and not winners_list:
        pass
    elif winners_list:
        # Multi-ron case
        loser = term.get("ron_loser")
        wg = pay_disc.get("winner_gain", 1)
        ll = pay_disc.get("loser_loss", -1)
        if (wg + ll) != 0:
            wg, ll = 1, -1

        total_loser_loss = 0
        for item in winners_list:
            seat_w = int(item.get("seat"))
            pts = int(item.get("points", 0))
            delta[seat_w] += wg * pts
            total_loser_loss += ll * pts
        if isinstance(loser, int):
            delta[loser] += total_loser_loss

    else:
        pts = int(term.get("points", 0))
        src = term.get("source")
        if src == "self_draw":
            wg = pay_self.get("winner_gain", 1)
            ol = pay_self.get("others_loss_each", -1)
            if (wg + 3 * ol) != 0:
                wg, ol = 3, -1
            delta[w] += wg * pts
            for s in range(4):
                if s != w:
                    delta[s] += ol * pts
        elif src == "discard":
            loser = term.get("ron_loser")
            wg = pay_disc.get("winner_gain", 1)
            ll = pay_disc.get("loser_loss", -1)
            if (wg + ll) != 0:
                wg, ll = 1, -1
            delta[w] += wg * pts
            if isinstance(loser, int):
                delta[loser] += ll * pts

    sd = term.get("side_delta")
    if isinstance(sd, list) and len(sd) == 4:
        for i in range(4):
            delta[i] += int(sd[i])

    return delta

# ----------------------------------------------------------------------
# --- Pairwise evaluation: target vs 3×opponent ---
# ----------------------------------------------------------------------

def evaluate_pair(
    rules: Dict[str, Any],
    target: str,
    opponent: str,
    episodes: int = 500,
    seed: int | None = None,
) -> Dict[str, Any]:
    assert target in VALID_POLICIES
    assert opponent in VALID_POLICIES

    rng = random.Random(seed)

    # --- build factories for lineup entries ---
    def rl_factory(seat, rules_, tuner=None):
        # seat- and rules-aware RL policy using the *same* model
        return RLPolicy(seat, rules_, model, tuner)

    if target == "rl":
        target_spec = rl_factory
    else:
        target_spec = target          # a string like "wp", "aggro", etc.

    if opponent == "rl":
        opp_spec = rl_factory
    else:
        opp_spec = opponent

    # target in seat 0, others are opponents
    lineup = [target_spec, opp_spec, opp_spec, opp_spec]

    target_total_points = 0
    target_wins = 0
    draws = 0

    for _ in range(episodes):
        ep = run_episode(rules, lineup, tuner=None, seed=rng.randint(1, 10**9))
        term = ep.get("terminal", {}) or {}

        delta = _episode_delta_for_seats(term, rules)
        target_total_points += delta[0]

        w = term.get("winner")
        winners_list = term.get("winners")

        if w is None and not winners_list:
            draws += 1
        elif winners_list:
            if any(int(item.get("seat", -1)) == 0 for item in winners_list):
                target_wins += 1
        elif w == 0:
            target_wins += 1

    win_rate = target_wins / float(episodes)
    avg_points = target_total_points / float(episodes)

    return {
        "win_rate": win_rate,
        "total_points": int(round(target_total_points)),
        "avg_points": float(avg_points),
        "episodes": episodes,
        "draws": draws,
    }

# ----------------------------------------------------------------------
# --- Plotting utilities ---
# ----------------------------------------------------------------------

def _build_matrix(stats: Dict[str, Dict[str, Any]], metric: str):
    labels = sorted(stats.keys())
    n = len(labels)
    mat = np.zeros((n, n))
    for i, t in enumerate(labels):
        for j, o in enumerate(labels):
            mat[i, j] = stats[t][o].get(metric, 0.0)
    return labels, mat

def plot_metric_matrix(stats, metric, fname_prefix, title, cmap="viridis"):
    labels, mat = _build_matrix(stats, metric)
    plt.figure(figsize=(8, 6))
    fmt = ".0f" if metric == "total_points" else ".3f"
    sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 8},
        cbar_kws={"format": "%d"} if metric == "total_points" else None
    )
    #if metric == "total_points":
       # plt.ticklabel_format(style="plain", axis="both")
    plt.title(title)
    plt.xlabel("Opponent (×3)")
    plt.ylabel("Target")
    plt.tight_layout()
    Path("resultsf").mkdir(exist_ok=True)
    plt.savefig(f"resultsf/{fname_prefix}.png", dpi=200)
    plt.close()

def make_all_plots(stats):
    plot_metric_matrix(
        stats,
        metric="win_rate",
        fname_prefix="policy_matrix_winrate",
        title="Win rate (rows = target, cols = 3×opponent)",
    )
    plot_metric_matrix(
        stats,
        metric="avg_points",
        fname_prefix="policy_matrix_avg_points",
        title="Average points per hand (rows = target, cols = 3×opponent)",
        cmap="magma",
    )
    plot_metric_matrix(
        stats,
        metric="total_points",
        fname_prefix="policy_matrix_total_points",
        title="Total points (rows = target, cols = 3×opponent)",
        cmap="magma",
    )

# ----------------------------------------------------------------------
# --- Main runner ---
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="Path to rules JSON")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    rules = load_rules(args.rules)
    stats = {t: {} for t in VALID_POLICIES}

    for t in VALID_POLICIES:
        for o in VALID_POLICIES:
            print(f"[eval] target={t} vs opponent={o}×3 …")
            res = evaluate_pair(rules, t, o, episodes=args.episodes, seed=args.seed)
            stats[t][o] = res
            print(f" → win_rate={res['win_rate']:.3f}, avg_points={res['avg_points']:.3f}")

    results_dir = Path("resultsf")
    results_dir.mkdir(parents=True, exist_ok=True)

    stats_path = results_dir / "sim_compare_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    make_all_plots(stats)
    print(f"Finished all evaluations. Results saved under {results_dir}/")

if __name__ == "__main__":
    main()
