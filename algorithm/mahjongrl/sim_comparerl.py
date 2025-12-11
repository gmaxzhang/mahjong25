# algorithm/mahjongrl/sim_compare_rl_only.py
import argparse, random, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from scipy.stats import ttest_1samp

from algorithm.rules_io import load_rules
from algorithm.sim_and_train import run_episode
from algorithm.mahjongrl.model import LSTMActorCritic, ACConfig
from algorithm.mahjongrl.agent import RLPolicy
from multiprocessing import Pool, cpu_count

# --------------------- Model setup ---------------------

# Must match your training config
cfg = ACConfig(obs_dim=468, hidden=256, lstm=256)
model = LSTMActorCritic(cfg)

# Load trained checkpoint (adjust path if needed)
ckpt = torch.load("runs/draft50super5/checkpoints/last.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

TARGET_POLICY = "rl"

OPPONENT_POLICIES = [
    "random", "wp", "payout",
    "hybrid", "aggro", "hyaggro",
    "flexaggro", "flexaggrod",
]

# --------------------- Scoring helpers ---------------------

def _episode_delta_for_seats(term: Dict[str, Any], rules: Dict[str, Any]) -> List[int]:
    """
    Compute per-seat point delta for a single episode, using the same
    payout logic as summarize_jsonl in sim_and_train.py.
    """
    delta = [0, 0, 0, 0]

    pay_disc = rules.get("payouts", {}).get("on_discard", {})
    pay_self = rules.get("payouts", {}).get("on_self_draw", {})

    w = term.get("winner")
    winners_list = term.get("winners")

    # No winner (drawn hand)
    if w is None and not winners_list:
        pass

    # Multi-Ron (several winners on one discard)
    elif winners_list:
        loser = term.get("ron_loser")
        wg = pay_disc.get("winner_gain", 1)
        ll = pay_disc.get("loser_loss", -1)
        # Normalize to +1 / -1 if rules are non-zero-sum
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

    # Single winner
    else:
        pts = int(term.get("points", 0))
        src = term.get("source")

        # Self-draw (tsumo)
        if src == "self_draw":
            wg = pay_self.get("winner_gain", 1)
            ol = pay_self.get("others_loss_each", -1)
            if (wg + 3 * ol) != 0:
                wg, ol = 3, -1
            delta[w] += wg * pts
            for s in range(4):
                if s != w:
                    delta[s] += ol * pts

        # Ron on discard
        elif src == "discard":
            loser = term.get("ron_loser")
            wg = pay_disc.get("winner_gain", 1)
            ll = pay_disc.get("loser_loss", -1)
            if (wg + ll) != 0:
                wg, ll = 1, -1
            delta[w] += wg * pts
            if isinstance(loser, int):
                delta[loser] += ll * pts

    # Side bets / flowers etc.
    sd = term.get("side_delta")
    if isinstance(sd, list) and len(sd) == 4:
        for i in range(4):
            delta[i] += int(sd[i])

    return delta

# --------------------- Evaluation ---------------------

def evaluate_vs_opponent(
    rules: Dict[str, Any],
    opponent: str,
    episodes: int = 300,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Evaluate RL policy in seat 0 vs 3 copies of `opponent`.
    Returns mean stats and stores all per-episode seat-0 deltas.
    """
    assert opponent in OPPONENT_POLICIES

    rng = random.Random(seed)

    def rl_factory(seat, rules_, tuner=None):
        # seat- and rules-aware RL policy using the *same* model
        return RLPolicy(seat, rules_, model, tuner)

    target_spec = rl_factory  # RL in seat 0
    opp_spec    = opponent    # baseline tag

    lineup = [target_spec, opp_spec, opp_spec, opp_spec]

    deltas: List[float] = []
    wins = 0
    losses = 0
    draws = 0

    for _ in range(episodes):
        ep = run_episode(rules, lineup, tuner=None, seed=rng.randint(1, 10**9))
        term = ep.get("terminal", {}) or {}

        delta = _episode_delta_for_seats(term, rules)
        d0 = delta[0]
        deltas.append(d0)

        w = term.get("winner")
        winners_list = term.get("winners")

        if w is None and not winners_list:
            draws += 1
        elif winners_list:
            if any(int(item.get("seat", -1)) == 0 for item in winners_list):
                wins += 1
            else:
                losses += 1
        elif w == 0:
            wins += 1
        else:
            losses += 1

    deltas_np = np.array(deltas, dtype=float)

    win_mask  = deltas_np > 0
    loss_mask = deltas_np < 0

    avg_points      = float(deltas_np.mean())
    win_rate        = wins / float(episodes)
    draw_rate       = draws / float(episodes)
    avg_win_points  = float(deltas_np[win_mask].mean()) if win_mask.any() else 0.0
    avg_loss_points = float(abs(deltas_np[loss_mask].mean())) if loss_mask.any() else 0.0
    exp_point_diff  = avg_points  # same thing here

    return {
        "win_rate":        win_rate,
        "draw_rate":       draw_rate,
        "avg_points":      avg_points,
        "avg_win_points":  avg_win_points,
        "avg_loss_points": avg_loss_points,
        "expected_diff":   exp_point_diff,
        "episodes":        episodes,
        "total_points":    float(deltas_np.sum()),
        "all_points":      deltas_np.tolist(),
    }

def _eval_job(job):
    opp, rules, episodes, seed = job
    res = evaluate_vs_opponent(rules, opp, episodes=episodes, seed=seed)
    return opp, res

# --------------------- Plot helpers ---------------------

def bootstrap_ci(scores: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return (0.0, 0.0)
    means = [
        np.mean(np.random.choice(scores, scores.shape[0], replace=True))
        for _ in range(n_boot)
    ]
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def plot_box_and_bar_rl(stats: Dict[str, Dict[str, Any]], results_dir: Path):
    """
    Boxplot of per-episode point differential for RL vs each opponent,
    plus bar plot of mean ±95% CI for RL vs each opponent.
    """
    # ---------- Long-format DF of per-episode points ----------
    rows = []
    for opp, res in stats.items():
        for pt in res.get("all_points", []):
            rows.append({"opponent": opp, "points": pt})
    df = pd.DataFrame(rows)

    # ---------- Boxplot ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        x="opponent",
        y="points",
        data=df,
        ax=ax,
        showfliers=True,
    )
    ax.set_title("RL seat-0 point differential vs each opponent")
    ax.set_xlabel("Opponent (×3)")
    ax.set_ylabel("Seat-0 point differential")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(results_dir / "rl_boxplot_per_episode_points.png", dpi=200)
    plt.close(fig)

    # ---------- Bar plot with CIs ----------
    bars, cis, labels = [], [], []
    for opp in sorted(stats.keys()):
        vals = np.asarray(stats[opp].get("all_points", []), dtype=float)
        if vals.size == 0:
            continue
        ci = bootstrap_ci(vals)
        mean_val = float(vals.mean())
        bars.append(mean_val)
        cis.append([mean_val - ci[0], ci[1] - mean_val])
        labels.append(opp)

    if bars:
        plt.figure(figsize=(8, 5))
        yerr = np.array(cis).T
        plt.bar(range(len(bars)), bars, yerr=yerr, capsize=5)
        plt.xticks(range(len(labels)), labels, rotation=30)
        plt.ylabel("Average seat-0 points ±95% CI")
        plt.title("RL vs each opponent (average point differential)")
        plt.tight_layout()
        plt.savefig(results_dir / "rl_bar_avgpoints_ci.png", dpi=200)
        plt.close()

def plot_significance_rl(stats: Dict[str, Dict[str, Any]],
                         results_dir: Path,
                         alpha: float = 0.05):
    """
    For each opponent O:
      - test whether RL's per-episode point differential vs O has mean != 0
        using a two-sided one-sample t-test.
      - store mean, p-value, CI; plot as a 1×N heatmap with p-value annotations.
    """
    labels = sorted(stats.keys())
    means = []
    pvals = []
    ci_lows = []
    ci_highs = []

    for opp in labels:
        vals = np.asarray(stats[opp].get("all_points", []), dtype=float)
        if vals.size < 5:
            m = 0.0
            p = 1.0
            ci = (0.0, 0.0)
        else:
            m = float(vals.mean())
            _, p = ttest_1samp(vals, popmean=0.0)
            ci = bootstrap_ci(vals)
        means.append(m)
        pvals.append(p)
        ci_lows.append(ci[0])
        ci_highs.append(ci[1])

    # Save numeric table
    df_sig = pd.DataFrame({
        "opponent": labels,
        "mean_points": means,
        "ci_low": ci_lows,
        "ci_high": ci_highs,
        "pvalue_vs_zero": pvals,
    })
    df_sig.to_csv(results_dir / "rl_vs_opponents_significance.csv", index=False)

    # Build 1×N matrix for heatmap
    dom = np.array([means])  # shape (1, N)
    annotations = np.array([[
        f"{p:.3f}" if p < alpha else ""
        for p in pvals
    ]])

    plt.figure(figsize=(max(6, len(labels) * 0.6), 2.8))
    sns.heatmap(
        dom,
        annot=annotations,
        fmt="",
        cmap="coolwarm",
        center=0.0,
        xticklabels=labels,
        yticklabels=["RL"],
        cbar_kws={"label": "Mean seat-0 point diff"},
    )
    plt.title(f"RL vs opponents: mean point diff (p < {alpha} annotated)")
    plt.xlabel("Opponent (×3)")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(results_dir / "rl_significance_heatmap.png", dpi=200)
    plt.close()

def plot_avgpoints_heatmap_rl(stats: Dict[str, Dict[str, Any]],
                              results_dir: Path):
    """
    1×N heatmap of RL's average seat-0 point differential vs each opponent.
    """
    labels = sorted(stats.keys())
    means = [float(stats[opp]["avg_points"]) for opp in labels]

    mat = np.array([means], dtype=float)  # shape (1, N)

    plt.figure(figsize=(max(6, len(labels) * 0.6), 2.8))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="magma",
        center=0.0,
        xticklabels=labels,
        yticklabels=["RL"],
        cbar_kws={"label": "Mean seat-0 point diff"},
    )
    plt.title("RL vs opponents: average point differential")
    plt.xlabel("Opponent (×3)")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(results_dir / "rl_avgpoints_heatmap.png", dpi=200)
    plt.close()

def plot_winrate_rl(stats: Dict[str, Dict[str, Any]], results_dir: Path):
    """
    Bar plot of RL win rate vs each opponent (seat 0 RL vs 3×opponent).
    """
    labels = sorted(stats.keys())
    win_rates = [stats[opp]["win_rate"] for opp in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(labels)), win_rates)
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.ylabel("Win rate (fraction of episodes)")
    plt.ylim(0.0, 1.0)
    plt.title("RL win rate vs each opponent (seat 0 vs 3×opponent)")
    plt.tight_layout()
    plt.savefig(results_dir / "rl_bar_winrate.png", dpi=200)
    plt.close()



# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="Path to rules JSON")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rules = load_rules(args.rules)
    results_dir = Path("plot50super5")
    results_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, Any]] = {}

    # Build jobs: RL vs each opponent
    jobs = []
    base_seed = args.seed
    for i, opp in enumerate(OPPONENT_POLICIES):
        job_seed = base_seed + i * 1000
        jobs.append((opp, rules, args.episodes, job_seed))

    # Parallel execution over opponents
    n_proc = min(cpu_count(), len(jobs))
    with Pool(processes=n_proc) as pool:
        for opp, res in pool.imap_unordered(_eval_job, jobs):
            print(
                f"[eval] RL (seat 0) vs {opp}×3 → "
                f"win_rate={res['win_rate']:.3f}, avg_points={res['avg_points']:.3f}"
            )
            stats[opp] = res

    # Save raw stats
    with (results_dir / "sim_compare_rl_vs_opponents.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Plots & significance
    plot_box_and_bar_rl(stats, results_dir)
    plot_significance_rl(stats, results_dir, alpha=0.05)
    plot_avgpoints_heatmap_rl(stats, results_dir)
    plot_winrate_rl(stats, results_dir)

    print("✅ Finished RL-only evaluations. See plot50super5/ for plots and stats.")

if __name__ == "__main__":
    main()
