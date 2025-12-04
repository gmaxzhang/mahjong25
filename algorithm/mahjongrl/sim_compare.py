# algorithm/mahjongrl/sim_compare.py
import argparse, random, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import torch
from scipy.stats import ttest_rel, ttest_ind

from algorithm.rules_io import load_rules
from algorithm.sim_and_train import run_episode
from algorithm.mahjongrl.model import LSTMActorCritic, ACConfig
from algorithm.mahjongrl.agent import RLPolicy

# --------------------- Model setup ---------------------

# Must match your training config
cfg = ACConfig(obs_dim=468, hidden=256, lstm=256)
model = LSTMActorCritic(cfg)

# Load trained checkpoint
ckpt = torch.load("runs/drafttrain1/checkpoints/last.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

VALID_POLICIES = [
    "random", "wp", "payout",
    "hybrid", "aggro", "hyaggro",
    "flexaggro", "flexaggrod",
    "rl",  # uncomment if you want to include the RL policy as well
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

def evaluate_pair(
    rules: Dict[str, Any],
    target: str,
    opponent: str,
    episodes: int = 300,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Evaluate 'target' policy in seat 0 vs 3 copies of 'opponent'.
    Returns mean stats and stores all per-episode seat-0 deltas.
    """
    assert target in VALID_POLICIES
    assert opponent in VALID_POLICIES

    rng = random.Random(seed)

    def rl_factory(seat, rules_, tuner=None):
        # seat- and rules-aware RL policy using the *same* model
        return RLPolicy(seat, rules_, model, tuner)

    target_spec = rl_factory if target == "rl" else target
    opp_spec    = rl_factory if opponent == "rl" else opponent

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
        "win_rate":       win_rate,
        "draw_rate":      draw_rate,
        "avg_points":     avg_points,
        "avg_win_points": avg_win_points,
        "avg_loss_points": avg_loss_points,
        "expected_diff":  exp_point_diff,
        "episodes":       episodes,
        "total_points":    float(deltas_np.sum()),
        "all_points":     deltas_np.tolist(),
    }

# --------------------- Matrix helpers ---------------------

def _build_matrix(stats: Dict[str, Dict[str, Any]], metric: str):
    labels = sorted(stats.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)
    for i, t in enumerate(labels):
        for j, o in enumerate(labels):
            mat[i, j] = stats[t][o].get(metric, 0.0)
    return labels, mat

def bootstrap_ci(scores: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return (0.0, 0.0)
    means = [
        np.mean(np.random.choice(scores, scores.shape[0], replace=True))
        for _ in range(n_boot)
    ]
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def compute_significance_matrix(
    stats: Dict[str, Dict[str, Any]],
    paired: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Build matrix of p-values comparing policy i vs policy j
    using per-episode deltas (all_points).

    If paired=False: Welch's t-test (independent samples).
    """
    labels = sorted(stats.keys())
    n = len(labels)
    mat_p = np.ones((n, n), dtype=float)

    for i, ti in enumerate(labels):
        for j, tj in enumerate(labels):
            if i == j:
                continue
            A = np.array(stats[ti][tj].get("all_points", []), dtype=float)
            B = np.array(stats[tj][ti].get("all_points", []), dtype=float)
            if len(A) > 5 and len(B) > 5:
                m = min(len(A), len(B))
                if paired:
                    _, p = ttest_rel(A[:m], B[:m])
                else:
                    _, p = ttest_ind(A[:m], B[:m], equal_var=False)
                mat_p[i, j] = float(p)
    return labels, mat_p

# --------------------- Visualization ---------------------

def plot_metric_matrix(
    stats: Dict[str, Dict[str, Any]],
    metric: str,
    title: str,
    fname: str,
    cmap: str = "magma",
):
    labels, mat = _build_matrix(stats, metric)
    plt.figure(figsize=(8, 6))
    fmt = ".3f"
    if metric == "total_points":  # not currently used, but kept for future
        fmt = ".0f"
    sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Opponent (×3)")
    plt.ylabel("Target")
    plt.tight_layout()
    Path("resultsdraft").mkdir(exist_ok=True)
    plt.savefig(f"resultsdraft/{fname}.png", dpi=200)
    plt.close()

def plot_box_and_bar(stats):
    # ---------- build long-format dataframe of per-episode points ----------
    rows = []
    for t, row in stats.items():
        for o, res in row.items():
            for pt in res.get("all_points", []):
                rows.append({
                    "target": t,
                    "opponent": o,
                    "points": pt,
                })
    df = pd.DataFrame(rows)

    # ---------- Boxplot: legend moved outside ----------
    fig, ax = plt.subplots(figsize=(14, 5))

    sns.boxplot(
        x="opponent",
        y="points",
        hue="target",
        data=df,
        ax=ax,
        showfliers=True,
    )

    ax.set_title("Per-episode point differential by target vs each opponent")
    ax.set_xlabel("Opponent (×3)")
    ax.set_ylabel("Seat-0 point differential")
    plt.xticks(rotation=45)

    # Move the *existing* legend outside the axes
    # (this guarantees we actually have handles/labels)
    legend = ax.legend(
        title="target",
        bbox_to_anchor=(1.02, 0.5),   # to the right of the axes
        loc="center left",
        borderaxespad=0.0,
    )

    # Make room on the right for the legend
    plt.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])

    outdir = Path("resultsdraft")
    outdir.mkdir(exist_ok=True)
    fig.savefig(outdir / "boxplot_per_episode_points.png", dpi=200)
    plt.close(fig)

    # ---------- Bar plot with CIs (same idea as before) ----------
    bars, cis, labels = [], [], []
    for t in sorted(stats.keys()):
        vals = []
        for o in stats[t].keys():
            vals.extend(stats[t][o].get("all_points", []))
        if not vals:
            continue
        ci = bootstrap_ci(vals)
        mean_val = float(np.mean(vals))
        bars.append(mean_val)
        cis.append([mean_val - ci[0], ci[1] - mean_val])
        labels.append(t)

    if bars:
        plt.figure(figsize=(8, 5))
        yerr = np.array(cis).T
        plt.bar(range(len(bars)), bars, yerr=yerr, capsize=5)
        plt.xticks(range(len(labels)), labels, rotation=30)
        plt.ylabel("Average points ±95% CI")
        plt.title("Average point differential per policy (all opponents)")
        plt.tight_layout()
        plt.savefig(outdir / "bar_avgpoints_ci.png", dpi=200)
        plt.close()



def build_significance_matrices(
    stats: Dict[str, Dict[str, Any]],
    alpha: float = 0.05,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Returns:
      labels: list of policy names in consistent order
      diff_sig: matrix where entry (i,j) is the mean point diff
                (i vs j) - (j vs i) if significant at level alpha,
                otherwise NaN
      pvals:   full matrix of p-values for i vs j
    """
    labels, pvals = compute_significance_matrix(stats, paired=False)
    n = len(labels)
    diff = np.zeros((n, n), dtype=float)

    for i, ti in enumerate(labels):
        for j, tj in enumerate(labels):
            if i == j:
                continue
            mu_ij = stats[ti][tj]["avg_points"]
            mu_ji = stats[tj][ti]["avg_points"]
            diff[i, j] = mu_ij - mu_ji

    sig_mask = pvals < alpha
    diff_sig = np.where(sig_mask, diff, np.nan)
    return labels, diff_sig, pvals

def plot_significance_heatmap(
    stats: Dict[str, Dict[str, Any]],
    alpha: float = 0.05,
):
    """
    Heatmap where cell (i,j) is the significant mean point
    advantage of policy i over policy j, i.e.
        E[points(i vs j×3)] - E[points(j vs i×3)]
    Only cells with p < alpha are shown; others are NaN.
    """
    labels, diff_sig, pvals = build_significance_matrices(stats, alpha=alpha)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        diff_sig,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Significant mean point diff (row − column), α={alpha}")
    plt.xlabel("Opponent")
    plt.ylabel("Target")
    plt.tight_layout()
    Path("resultsdraft").mkdir(exist_ok=True)
    plt.savefig("resultsdraft/significance_heatmap.png", dpi=200)
    plt.close()

    # Also dump numeric tables as CSVs
    diff_df = pd.DataFrame(diff_sig, index=labels, columns=labels)
    pval_df = pd.DataFrame(pvals,    index=labels, columns=labels)
    diff_df.to_csv("resultsdraft/significance_diff_matrix.csv")
    pval_df.to_csv("resultsdraft/significance_pvalues_matrix.csv")

def plot_dominance_network(stats: Dict[str, Dict[str, Any]], alpha: float = 0.05):
    """
    Directed graph where an edge i -> j means policy i significantly
    outperforms policy j at level alpha (positive mean diff).
    """
    labels, diff_sig, pvals = build_significance_matrices(stats, alpha=alpha)
    G = nx.DiGraph()

    # Add edges for significant advantages
    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            if i == j:
                continue
            if pvals[i, j] < alpha:
                diff = diff_sig[i, j]
                if not np.isnan(diff) and diff > 0:
                    # weight ~ effect size; use abs(diff)
                    G.add_edge(src, dst, weight=abs(diff))

    if len(G.edges) == 0:
        print("No significant dominance relationships found at this alpha.")
        return

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(7, 6))
    edge_widths = [d["weight"] for _, _, d in G.edges(data=True)]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1800,
        node_color="lightblue",
        font_size=9,
        arrowsize=18,
        width=edge_widths,
        ax=ax,
    )
    ax.set_title(f"Policy Dominance Graph (edges: p < {alpha})")
    Path("resultsdraft").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("resultsdraft/policy_dominance_network.png", dpi=200)
    plt.close(fig)

def compute_vs_common_opponent_tables(stats: Dict[str, Dict[str, Any]],
                                      alpha: float = 0.05):
    """
    For each fixed opponent O, compare every pair of target policies (A,B)
    on their per-episode point diffs vs O using a *paired* t-test.

    Returns:
        tables[O] = {
            "labels": [...policy names...],
            "diff":   2D array of significant mean differences (A - B, NaN if not sig),
            "pval":   2D array of p-values (1.0 on diag)
        }
    """
    from scipy.stats import ttest_rel

    labels = sorted(stats.keys())
    opponents = sorted(next(iter(stats.values())).keys())

    tables: Dict[str, Dict[str, Any]] = {}
    for opp in opponents:
        n_pols = len(labels)
        diff_mat = np.full((n_pols, n_pols), np.nan, dtype=float)
        pval_mat = np.ones((n_pols, n_pols), dtype=float)

        for i, pi in enumerate(labels):
            for j, pj in enumerate(labels):
                if i == j:
                    continue

                A = np.array(stats[pi][opp].get("all_points", []), dtype=float)
                B = np.array(stats[pj][opp].get("all_points", []), dtype=float)
                n = min(len(A), len(B))
                if n < 6:
                    continue  # not enough data

                # paired t-test: episode k of A vs episode k of B, same env seed
                t, p = ttest_rel(A[:n], B[:n])
                pval_mat[i, j] = p
                if p < alpha:
                    diff_mat[i, j] = A.mean() - B.mean()

        tables[opp] = {"labels": labels, "diff": diff_mat, "pval": pval_mat}

    return tables


def save_vs_common_opponent_tables(stats: Dict[str, Dict[str, Any]],
                                   results_dir: Path,
                                   alpha: float = 0.05):
    """
    Saves, for each opponent O:
      - significance_vs_<O>_diff.csv    (significant mean differences, row - col)
      - significance_vs_<O>_pvalues.csv (raw p-values)
      - significance_vs_<O>_heatmap.png (optional visualization)
    """
    tables = compute_vs_common_opponent_tables(stats, alpha=alpha)

    for opp, data in tables.items():
        labels = data["labels"]
        diff_mat = data["diff"]
        pval_mat = data["pval"]

        # CSV: significant mean differences
        diff_df = pd.DataFrame(diff_mat, index=labels, columns=labels)
        diff_df.to_csv(results_dir / f"significance_vs_{opp}_diff.csv")

        # CSV: p-values
        pval_df = pd.DataFrame(pval_mat, index=labels, columns=labels)
        pval_df.to_csv(results_dir / f"significance_vs_{opp}_pvalues.csv")

        # Heatmap (optional)
        plt.figure(figsize=(8, 6))
        mask = np.isnan(diff_mat)
        sns.heatmap(
            diff_mat,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0.0,
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"Significant mean diff vs common opponent '{opp}'\n(row − col, paired t-test, p < {alpha})")
        plt.xlabel("Policy B")
        plt.ylabel("Policy A")
        plt.tight_layout()
        plt.savefig(results_dir / f"significance_vs_{opp}_heatmap.png", dpi=200)
        plt.close()

def plot_pvalue_dominance_heatmap(stats, results_dir: Path, alpha: float = 0.05):
    """
    Creates a p-value heatmap with dominance coloring (red=target stronger, blue=target weaker).
    Each annotated cell shows the p-value if significant (p < alpha).
    """
    labels, pvals = compute_significance_matrix(stats)
    n = len(labels)
    dom_color = np.full((n, n), np.nan)
    annotations = np.full((n, n), "", dtype=object)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A = np.array(stats[labels[i]][labels[j]].get("all_points", []))
            B = np.array(stats[labels[j]][labels[i]].get("all_points", []))
            if len(A) < 5 or len(B) < 5:
                continue
            _, p = ttest_ind(A, B, equal_var=False)
            diff = A.mean() - B.mean()
            if p < alpha:
                dom_color[i, j] = diff
                annotations[i, j] = f"{p:.3f}"
            else:
                annotations[i, j] = ""

    plt.figure(figsize=(8, 6))
    mask = np.isnan(dom_color)
    sns.heatmap(
        dom_color,
        mask=mask,
        cmap="coolwarm",
        center=0.0,
        annot=annotations,
        fmt="",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Row stronger (+) / weaker (−)"},
    )

    plt.title(f"P-value heatmap (α = {alpha}), red = row stronger, blue = weaker")
    plt.xlabel("Opponent (Policy B)")
    plt.ylabel("Target (Policy A)")
    plt.tight_layout()
    plt.savefig(results_dir / "pvalue_dominance_heatmap.png", dpi=200)
    plt.close()


# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="Path to rules JSON")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rules = load_rules(args.rules)
    results_dir = Path("resultsdraft")
    results_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, Any]] = {t: {} for t in VALID_POLICIES}

    for t in VALID_POLICIES:
        for o in VALID_POLICIES:
            print(f"[eval] target={t} vs {o}×3 …")
            res = evaluate_pair(rules, t, o, episodes=args.episodes, seed=args.seed)
            stats[t][o] = res
            print(f" → win_rate={res['win_rate']:.3f}, avg_points={res['avg_points']:.3f}")

    # Save raw stats
    with (results_dir / "sim_compare_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Metric matrices
    plot_metric_matrix(stats, "win_rate",
                       "Win rate matrix (rows = target, cols = 3×opponent)",
                       "matrix_winrate",
                       cmap="viridis")
    plot_metric_matrix(stats, "total_points",
                       "Total point differential (rows = target, cols = 3×opponent)",
                       "matrix_total_points",
                       cmap="YlOrRd")
    plot_metric_matrix(stats, "avg_points",
                       "Average point differential (rows = target, cols = 3×opponent)",
                       "matrix_avgpoints",
                       cmap="magma")
    plot_metric_matrix(stats, "draw_rate",
                       "Draw rate matrix (rows = target, cols = 3×opponent)",
                       "matrix_drawrate",
                       cmap="coolwarm")

    # Distributional and aggregate plots
    plot_box_and_bar(stats)

    # Significance table + heatmap + CSVs
    plot_significance_heatmap(stats, alpha=0.05)

    # Dominance network based on same significance tests
    plot_dominance_network(stats, alpha=0.05)

    plot_pvalue_dominance_heatmap(stats, results_dir, alpha= 0.05)

    save_vs_common_opponent_tables(stats, results_dir, alpha=0.05)

    print("✅ Finished all evaluations. See resultsdraft/ for plots, stats, and significance tables.")

if __name__ == "__main__":
    main()

# # algorithm/mahjongrl/sim_compare.py
# import argparse, random, json
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import networkx as nx
# from pathlib import Path
# from typing import Dict, Any, List
# from scipy.stats import ttest_rel, ttest_ind

# import torch
# from algorithm.rules_io import load_rules
# from algorithm.sim_and_train import run_episode
# from algorithm.mahjongrl.model import LSTMActorCritic, ACConfig
# from algorithm.mahjongrl.agent import RLPolicy

# # --------------------- Model setup ---------------------
# cfg = ACConfig(obs_dim=468, hidden=256, lstm=256)
# model = LSTMActorCritic(cfg)
# ckpt = torch.load("runs/mahjongR2/checkpoints/last.pt", map_location="cpu")
# model.load_state_dict(ckpt["model_state"])
# model.eval()

# VALID_POLICIES = [
#     "random", "wp", "payout",
#     "hybrid", "aggro", "hyaggro",
#     "flexaggro", "flexaggrod"
# ]

# # --------------------- Scoring helpers ---------------------
# def _episode_delta_for_seats(term: Dict[str, Any], rules: Dict[str, Any]) -> List[int]:
#     delta = [0, 0, 0, 0]
#     pay_disc = rules.get("payouts", {}).get("on_discard", {})
#     pay_self = rules.get("payouts", {}).get("on_self_draw", {})

#     w = term.get("winner")
#     winners_list = term.get("winners")

#     if w is None and not winners_list:
#         pass
#     elif winners_list:
#         loser = term.get("ron_loser")
#         wg = pay_disc.get("winner_gain", 1)
#         ll = pay_disc.get("loser_loss", -1)
#         if (wg + ll) != 0:
#             wg, ll = 1, -1
#         total_loser_loss = 0
#         for item in winners_list:
#             seat_w = int(item.get("seat"))
#             pts = int(item.get("points", 0))
#             delta[seat_w] += wg * pts
#             total_loser_loss += ll * pts
#         if isinstance(loser, int):
#             delta[loser] += total_loser_loss
#     else:
#         pts = int(term.get("points", 0))
#         src = term.get("source")
#         if src == "self_draw":
#             wg = pay_self.get("winner_gain", 1)
#             ol = pay_self.get("others_loss_each", -1)
#             if (wg + 3 * ol) != 0:
#                 wg, ol = 3, -1
#             delta[w] += wg * pts
#             for s in range(4):
#                 if s != w:
#                     delta[s] += ol * pts
#         elif src == "discard":
#             loser = term.get("ron_loser")
#             wg = pay_disc.get("winner_gain", 1)
#             ll = pay_disc.get("loser_loss", -1)
#             if (wg + ll) != 0:
#                 wg, ll = 1, -1
#             delta[w] += wg * pts
#             if isinstance(loser, int):
#                 delta[loser] += ll * pts

#     sd = term.get("side_delta")
#     if isinstance(sd, list) and len(sd) == 4:
#         for i in range(4):
#             delta[i] += int(sd[i])
#     return delta

# # --------------------- Evaluation ---------------------
# def evaluate_pair(rules, target, opponent, episodes=300, seed=None):
#     assert target in VALID_POLICIES
#     assert opponent in VALID_POLICIES
#     rng = random.Random(seed)

#     def rl_factory(seat, rules_, tuner=None):
#         return RLPolicy(seat, rules_, model, tuner)

#     target_spec = rl_factory if target == "rl" else target
#     opp_spec = rl_factory if opponent == "rl" else opponent
#     lineup = [target_spec, opp_spec, opp_spec, opp_spec]

#     deltas, wins, losses, draws = [], 0, 0, 0

#     for _ in range(episodes):
#         ep = run_episode(rules, lineup, tuner=None, seed=rng.randint(1, 10**9))
#         term = ep.get("terminal", {}) or {}
#         delta = _episode_delta_for_seats(term, rules)
#         d0 = delta[0]
#         deltas.append(d0)

#         w = term.get("winner")
#         winners_list = term.get("winners")
#         if w is None and not winners_list:
#             draws += 1
#         elif winners_list:
#             if any(int(item.get("seat", -1)) == 0 for item in winners_list):
#                 wins += 1
#             else:
#                 losses += 1
#         elif w == 0:
#             wins += 1
#         else:
#             losses += 1

#     deltas = np.array(deltas)
#     win_mask, loss_mask = deltas > 0, deltas < 0
#     avg_points = deltas.mean()
#     win_rate = wins / episodes
#     draw_rate = draws / episodes
#     avg_win_points = deltas[win_mask].mean() if win_mask.any() else 0
#     avg_loss_points = abs(deltas[loss_mask].mean()) if loss_mask.any() else 0
#     exp_point_diff = avg_points

#     return {
#         "win_rate": win_rate,
#         "draw_rate": draw_rate,
#         "avg_points": float(avg_points),
#         "avg_win_points": float(avg_win_points),
#         "avg_loss_points": float(avg_loss_points),
#         "expected_diff": float(exp_point_diff),
#         "total_points": float(deltas.sum()),
#         "episodes": episodes,
#         "all_points": deltas.tolist(),
#     }

# # --------------------- Stats & visualization ---------------------
# def _build_matrix(stats, metric):
#     labels = sorted(stats.keys())
#     mat = np.zeros((len(labels), len(labels)))
#     for i, t in enumerate(labels):
#         for j, o in enumerate(labels):
#             mat[i, j] = stats[t][o].get(metric, 0)
#     return labels, mat

# def bootstrap_ci(scores, n_boot=1000, alpha=0.05):
#     means = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n_boot)]
#     return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])

# def compute_significance_matrix(stats, metric="avg_points", paired=True):
#     labels = sorted(stats.keys())
#     n = len(labels)
#     mat_p = np.ones((n, n))
#     for i, ti in enumerate(labels):
#         for j, tj in enumerate(labels):
#             if i == j: 
#                 continue
#             A = np.array(stats[ti][tj].get("all_points", []))
#             B = np.array(stats[tj][ti].get("all_points", []))
#             if len(A) > 5 and len(B) > 5:
#                 if paired:
#                     _, p = ttest_rel(A[:min(len(A), len(B))], B[:min(len(A), len(B))])
#                 else:
#                     _, p = ttest_ind(A[:min(len(A), len(B))], B[:min(len(A), len(B))], equal_var=False)
#                 mat_p[i, j] = p
#     return labels, mat_p

# def plot_metric_matrix(stats, metric, title, fname, cmap="magma", fmt=".3f"):
#     labels, mat = _build_matrix(stats, metric)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap,
#                 xticklabels=labels, yticklabels=labels)
#     plt.title(title)
#     plt.xlabel("Opponent (×3)")
#     plt.ylabel("Target")
#     plt.tight_layout()
#     Path("resultsdraft").mkdir(exist_ok=True)
#     plt.savefig(f"resultsdraft/{fname}.png", dpi=200)
#     plt.close()

# def plot_box_and_bar(stats):
#     df_rows = []
#     for t, row in stats.items():
#         for o, res in row.items():
#             df_rows.append({
#                 "target": t, "opponent": o,
#                 "points": res["avg_points"],
#                 "win_rate": res["win_rate"],
#                 "draw_rate": res["draw_rate"]
#             })
#     df = pd.DataFrame(df_rows)

#     plt.figure(figsize=(10,6))
#     sns.boxplot(x="opponent", y="points", hue="target", data=df)
#     plt.title("Distribution of Average Points by Opponent")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("resultsdraft/boxplot_avg_points.png", dpi=200)
#     plt.close()

#     bars, cis, labels = [], [], []
#     for t in df["target"].unique():
#         vals = []
#         for o in df["opponent"].unique():
#             vals.extend(stats[t][o].get("all_points", []))
#         if len(vals) < 2:
#             continue
#         ci = bootstrap_ci(vals)
#         bars.append(np.mean(vals))
#         cis.append([np.mean(vals)-ci[0], ci[1]-np.mean(vals)])
#         labels.append(t)

#     plt.figure(figsize=(9,5))
#     plt.bar(range(len(bars)), bars, yerr=np.array(cis).T, capsize=5, color="skyblue", edgecolor="black")
#     plt.xticks(range(len(labels)), labels, rotation=30)
#     plt.ylabel("Average points ±95% CI")
#     plt.title("Average Points with 95% Confidence Interval (Across All Opponents)")
#     plt.tight_layout()
#     plt.savefig("resultsdraft/bar_avgpoints_ci.png", dpi=200)
#     plt.close()

# def plot_dominance_network(stats):
#     labels, pvals = compute_significance_matrix(stats)
#     G = nx.DiGraph()
#     for i, a in enumerate(labels):
#         for j, b in enumerate(labels):
#             if i != j and pvals[i][j] < 0.05:
#                 G.add_edge(a, b, weight=1.0 / max(1e-6, pvals[i][j]))
#     pos = nx.spring_layout(G, seed=42)
#     fig, ax = plt.subplots(figsize=(7,6))
#     nx.draw(G, pos, with_labels=True, node_size=1800,
#             node_color="lightblue", font_size=9, arrowsize=18,
#             width=[d["weight"]*0.1 for _,_,d in G.edges(data=True)], ax=ax)
#     plt.title("Policy Dominance Graph (p < 0.05)")
#     Path("resultsdraft").mkdir(exist_ok=True)
#     plt.tight_layout()
#     plt.savefig("resultsdraft/policy_dominance_network.png", dpi=200)
#     plt.close(fig)

# # --------------------- Main ---------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--rules", required=True)
#     ap.add_argument("--episodes", type=int, default=300)
#     ap.add_argument("--seed", type=int, default=0)
#     args = ap.parse_args()

#     rules = load_rules(args.rules)
#     stats = {t: {} for t in VALID_POLICIES}

#     for t in VALID_POLICIES:
#         for o in VALID_POLICIES:
#             print(f"[eval] target={t} vs {o}×3 …")
#             res = evaluate_pair(rules, t, o, episodes=args.episodes, seed=args.seed)
#             stats[t][o] = res
#             print(f" → win_rate={res['win_rate']:.3f}, avg_points={res['avg_points']:.3f}")

#     results_dir = Path("resultsdraft")
#     results_dir.mkdir(exist_ok=True)
#     json.dump(stats, open(results_dir/"sim_compare_stats.json", "w"), indent=2)

#     plot_metric_matrix(stats, "total_points", "Total Points (Integer)", "matrix_total_points", fmt=".0f", cmap="YlOrRd")
#     plot_metric_matrix(stats, "avg_points", "Average Points", "matrix_avgpoints", cmap="magma")
#     plot_metric_matrix(stats, "win_rate", "Win Rate Matrix", "matrix_winrate", cmap="viridis")
#     plot_metric_matrix(stats, "draw_rate", "Draw Rate Matrix", "matrix_drawrate", cmap="coolwarm")

#     plot_box_and_bar(stats)
#     plot_dominance_network(stats)

#     print("✅ Finished all evaluations. See resultsdraft/ for plots and stats.")

# if __name__ == "__main__":
#     main()


# # algorithm/mahjongrl/sim_compare.py
# import argparse, random, json
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import networkx as nx
# import scipy
# from scipy import stats
# from pathlib import Path
# from typing import Dict, Any, List
# from scipy.stats import ttest_rel, ttest_ind

# import torch
# from algorithm.rules_io import load_rules
# from algorithm.sim_and_train import run_episode
# from algorithm.mahjongrl.model import LSTMActorCritic, ACConfig
# from algorithm.mahjongrl.agent import RLPolicy

# # --------------------- Model setup ---------------------
# cfg = ACConfig(obs_dim=468, hidden=256, lstm=256)
# model = LSTMActorCritic(cfg)
# ckpt = torch.load("runs/mahjongR2/checkpoints/last.pt", map_location="cpu")
# model.load_state_dict(ckpt["model_state"])
# model.eval()

# VALID_POLICIES = [
#     "random", "wp", "payout",
#     "hybrid", "aggro", "hyaggro",
#     "flexaggro", "flexaggrod"#, "rl"
# ]

# # --------------------- Scoring helpers ---------------------
# def _episode_delta_for_seats(term: Dict[str, Any], rules: Dict[str, Any]) -> List[int]:
#     delta = [0, 0, 0, 0]
#     pay_disc = rules.get("payouts", {}).get("on_discard", {})
#     pay_self = rules.get("payouts", {}).get("on_self_draw", {})

#     w = term.get("winner")
#     winners_list = term.get("winners")

#     if w is None and not winners_list:
#         pass
#     elif winners_list:
#         loser = term.get("ron_loser")
#         wg = pay_disc.get("winner_gain", 1)
#         ll = pay_disc.get("loser_loss", -1)
#         if (wg + ll) != 0:
#             wg, ll = 1, -1
#         total_loser_loss = 0
#         for item in winners_list:
#             seat_w = int(item.get("seat"))
#             pts = int(item.get("points", 0))
#             delta[seat_w] += wg * pts
#             total_loser_loss += ll * pts
#         if isinstance(loser, int):
#             delta[loser] += total_loser_loss
#     else:
#         pts = int(term.get("points", 0))
#         src = term.get("source")
#         if src == "self_draw":
#             wg = pay_self.get("winner_gain", 1)
#             ol = pay_self.get("others_loss_each", -1)
#             if (wg + 3 * ol) != 0:
#                 wg, ol = 3, -1
#             delta[w] += wg * pts
#             for s in range(4):
#                 if s != w:
#                     delta[s] += ol * pts
#         elif src == "discard":
#             loser = term.get("ron_loser")
#             wg = pay_disc.get("winner_gain", 1)
#             ll = pay_disc.get("loser_loss", -1)
#             if (wg + ll) != 0:
#                 wg, ll = 1, -1
#             delta[w] += wg * pts
#             if isinstance(loser, int):
#                 delta[loser] += ll * pts

#     sd = term.get("side_delta")
#     if isinstance(sd, list) and len(sd) == 4:
#         for i in range(4):
#             delta[i] += int(sd[i])
#     return delta

# # --------------------- Evaluation ---------------------
# def evaluate_pair(rules, target, opponent, episodes=300, seed=None):
#     assert target in VALID_POLICIES
#     assert opponent in VALID_POLICIES
#     rng = random.Random(seed)

#     def rl_factory(seat, rules_, tuner=None):
#         return RLPolicy(seat, rules_, model, tuner)

#     target_spec = rl_factory if target == "rl" else target
#     opp_spec = rl_factory if opponent == "rl" else opponent
#     lineup = [target_spec, opp_spec, opp_spec, opp_spec]

#     deltas, wins, losses, draws = [], 0, 0, 0

#     for _ in range(episodes):
#         ep = run_episode(rules, lineup, tuner=None, seed=rng.randint(1, 10**9))
#         term = ep.get("terminal", {}) or {}
#         delta = _episode_delta_for_seats(term, rules)
#         d0 = delta[0]
#         deltas.append(d0)

#         w = term.get("winner")
#         winners_list = term.get("winners")
#         if w is None and not winners_list:
#             draws += 1
#         elif winners_list:
#             if any(int(item.get("seat", -1)) == 0 for item in winners_list):
#                 wins += 1
#             else:
#                 losses += 1
#         elif w == 0:
#             wins += 1
#         else:
#             losses += 1

#     deltas = np.array(deltas)
#     win_mask, loss_mask = deltas > 0, deltas < 0
#     avg_points = deltas.mean()
#     win_rate = wins / episodes
#     draw_rate = draws / episodes
#     avg_win_points = deltas[win_mask].mean() if win_mask.any() else 0
#     avg_loss_points = abs(deltas[loss_mask].mean()) if loss_mask.any() else 0
#     exp_point_diff = avg_points

#     return {
#         "win_rate": win_rate,
#         "draw_rate": draw_rate,
#         "avg_points": float(avg_points),
#         "avg_win_points": float(avg_win_points),
#         "avg_loss_points": float(avg_loss_points),
#         "expected_diff": float(exp_point_diff),
#         "episodes": episodes,
#         "all_points": deltas.tolist(),
#     }

# # --------------------- Stats & visualization ---------------------
# def _build_matrix(stats, metric):
#     labels = sorted(stats.keys())
#     mat = np.zeros((len(labels), len(labels)))
#     for i, t in enumerate(labels):
#         for j, o in enumerate(labels):
#             mat[i, j] = stats[t][o].get(metric, 0)
#     return labels, mat

# def bootstrap_ci(scores, n_boot=1000, alpha=0.05):
#     means = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n_boot)]
#     return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])

# def compute_significance_matrix(stats, metric="avg_points", paired=True):
#     labels = sorted(stats.keys())
#     n = len(labels)
#     mat_p = np.ones((n, n))
#     for i, ti in enumerate(labels):
#         for j, tj in enumerate(labels):
#             if i == j: continue
#             A = np.array(stats[ti][tj].get("all_points", []))
#             B = np.array(stats[tj][ti].get("all_points", []))
#             if len(A) > 5 and len(B) > 5:
#                 if paired:
#                     _, p = ttest_rel(A[:min(len(A), len(B))], B[:min(len(A), len(B))])
#                 else:
#                     _, p = ttest_ind(A[:min(len(A), len(B))], B[:min(len(A), len(B))], equal_var=False)
#                 mat_p[i, j] = p
#     return labels, mat_p

# def plot_metric_matrix(stats, metric, title, fname, cmap="magma"):
#     labels, mat = _build_matrix(stats, metric)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(mat, annot=True, fmt=".3f", cmap=cmap,
#                 xticklabels=labels, yticklabels=labels)
#     plt.title(title)
#     plt.xlabel("Opponent (×3)")
#     plt.ylabel("Target")
#     plt.tight_layout()
#     Path("resultsdraft").mkdir(exist_ok=True)
#     plt.savefig(f"resultsdraft/{fname}.png", dpi=200)
#     plt.close()

# def plot_box_and_bar(stats):
#     df_rows = []
#     for t, row in stats.items():
#         for o, res in row.items():
#             df_rows.append({
#                 "target": t, "opponent": o,
#                 "points": res["avg_points"],
#                 "win_rate": res["win_rate"],
#                 "draw_rate": res["draw_rate"]
#             })
#     df = pd.DataFrame(df_rows)

#     sns.boxplot(x="opponent", y="points", hue="target", data=df)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("resultsdraft/boxplot_avg_points.png", dpi=200)
#     plt.close()

#     bars, cis, labels = [], [], []
#     for t in df["target"].unique():
#         vals = []
#         for o in df["opponent"].unique():
#             vals.extend(stats[t][o].get("all_points", []))
#         ci = bootstrap_ci(vals)
#         bars.append(np.mean(vals))
#         cis.append([np.mean(vals)-ci[0], ci[1]-np.mean(vals)])
#         labels.append(t)

#     plt.bar(range(len(bars)), bars, yerr=np.array(cis).T, capsize=5)
#     plt.xticks(range(len(labels)), labels, rotation=30)
#     plt.ylabel("Average points ±95% CI")
#     plt.tight_layout()
#     plt.savefig("resultsdraft/bar_avgpoints_ci.png", dpi=200)
#     plt.close()

# def plot_dominance_network(stats):
#     import networkx as nx
#     import matplotlib.pyplot as plt
#     from pathlib import Path

#     labels, pvals = compute_significance_matrix(stats)
#     G = nx.DiGraph()

#     # build edges
#     for i, a in enumerate(labels):
#         for j, b in enumerate(labels):
#             if i != j and pvals[i][j] < 0.05:
#                 G.add_edge(a, b, weight=1.0 / max(1e-6, pvals[i][j]))

#     # use spring layout or circular
#     pos = nx.spring_layout(G, seed=42)

#     fig, ax = plt.subplots(figsize=(7, 6))   # <-- create fig/ax cleanly
#     nx.draw(
#         G,
#         pos,
#         with_labels=True,
#         node_size=1800,
#         node_color="lightblue",
#         font_size=9,
#         arrowsize=18,
#         width=[d["weight"] * 0.1 for _, _, d in G.edges(data=True)],
#         ax=ax                               # <-- draw on explicit axis
#     )

#     plt.title("Policy Dominance Graph (p < 0.05)")
#     Path("resultsf").mkdir(exist_ok=True)
#     plt.tight_layout()
#     plt.savefig("resultsdraft/policy_dominance_network.png", dpi=200)
#     plt.close(fig)


# # --------------------- Main ---------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--rules", required=True)
#     ap.add_argument("--episodes", type=int, default=300)
#     ap.add_argument("--seed", type=int, default=0)
#     args = ap.parse_args()

#     rules = load_rules(args.rules)
#     stats = {t: {} for t in VALID_POLICIES}

#     for t in VALID_POLICIES:
#         for o in VALID_POLICIES:
#             print(f"[eval] target={t} vs {o}×3 …")
#             res = evaluate_pair(rules, t, o, episodes=args.episodes, seed=args.seed)
#             stats[t][o] = res
#             print(f" → win_rate={res['win_rate']:.3f}, avg_points={res['avg_points']:.3f}")

#     results_dir = Path("resultsdraft")
#     results_dir.mkdir(exist_ok=True)
#     json.dump(stats, open(results_dir/"sim_compare_stats.json", "w"), indent=2)

#     # Plots
#     plot_metric_matrix(stats, "win_rate", "Win rate matrix", "matrix_winrate", cmap="viridis")
#     plot_metric_matrix(stats, "avg_points", "Average points", "matrix_avgpoints")
#     plot_metric_matrix(stats, "draw_rate", "Draw rate", "matrix_drawrate", cmap="coolwarm")
#     plot_box_and_bar(stats)
#     plot_dominance_network(stats)
#     print("✅ Finished all evaluations. See resultsdraft/ for plots and stats.")

# if __name__ == "__main__":
#     main()


# # algorithm/mahjongrl/sim_compare.py
# from __future__ import annotations

# import json
# import random
# from pathlib import Path
# from collections import defaultdict
# from typing import Dict, Any, List

# import numpy as np
# import matplotlib.pyplot as plt

# from algorithm.rules_io import load_rules
# from algorithm.sim_and_train import run_episode, _parse_lineup  # adjust import if needed


# # ---------------- Canonical policy tags ----------------

# _CANONICAL_TAGS = {
#     # flexible → flex
#     "flexibleaggro": "flexaggro",
#     "flexibleaggrod": "flexaggrod",
#     # already accepted tags
#     "flexaggro": "flexaggro",
#     "flexaggrod": "flexaggrod",
#     "random": "random",
#     "wp": "wp",
#     "payout": "payout",
#     "hybrid": "hybrid",
#     "aggro": "aggro",
#     "hyaggro": "hyaggro",
# }


# from typing import Dict, Any, List

# def _seat_deltas_from_terminal(term: Dict[str, Any],
#                                rules: Dict[str, Any],
#                                enforce_zero_sum: bool = True) -> List[int]:
#     """
#     Reproduce the per-seat point delta logic from sim_and_train.summarize_jsonl
#     for a *single hand*.

#     Returns a list of length 4: net points for seats 0..3 for this episode,
#     including side payments.
#     """
#     delta = [0, 0, 0, 0]

#     pay_disc = rules.get("payouts", {}).get("on_discard", {})
#     pay_self = rules.get("payouts", {}).get("on_self_draw", {})

#     w = term.get("winner")
#     winners_list = term.get("winners")

#     # Drawn game: no base payouts, just side_delta (added later)
#     if w is None and not winners_list:
#         pass
#     else:
#         src = term.get("source")

#         # Multi-ron on a discard
#         if winners_list:
#             loser = term.get("ron_loser")
#             wg = int(pay_disc.get("winner_gain", 1))
#             ll = int(pay_disc.get("loser_loss", -1))
#             if enforce_zero_sum and (wg + ll) != 0:
#                 wg, ll = 1, -1

#             total_loser_loss = 0
#             for item in winners_list:
#                 seat_w = int(item.get("seat", -1))
#                 pts = int(item.get("points", 0))
#                 if 0 <= seat_w < 4:
#                     delta[seat_w] += wg * pts
#                 total_loser_loss += ll * pts

#             if isinstance(loser, int) and 0 <= loser < 4:
#                 delta[loser] += total_loser_loss

#         # Single winner (tsumo or ron)
#         else:
#             pts = int(term.get("points", 0))

#             if not isinstance(w, int) or not (0 <= w < 4):
#                 # malformed, just bail to side_delta
#                 pts = int(term.get("points", 0))
#             else:
#                 if src == "self_draw":
#                     wg = int(pay_self.get("winner_gain", 1))
#                     ol = int(pay_self.get("others_loss_each", -1))
#                     if enforce_zero_sum and (wg + 3 * ol) != 0:
#                         wg, ol = 3, -1

#                     delta[w] += wg * pts
#                     for s in range(4):
#                         if s != w:
#                             delta[s] += ol * pts

#                 elif src == "discard":
#                     loser = term.get("ron_loser")
#                     wg = int(pay_disc.get("winner_gain", 1))
#                     ll = int(pay_disc.get("loser_loss", -1))
#                     if enforce_zero_sum and (wg + ll) != 0:
#                         wg, ll = 1, -1

#                     delta[w] += wg * pts
#                     if isinstance(loser, int) and 0 <= loser < 4:
#                         delta[loser] += ll * pts

#                 # If source is "drawn_game" or unknown, we do nothing more here.

#     # Add side payments (4-same-discard, etc.)
#     sd = term.get("side_delta")
#     if isinstance(sd, list) and len(sd) == 4:
#         for i in range(4):
#             try:
#                 delta[i] += int(sd[i])
#             except Exception:
#                 pass

#     return delta



# def canonicalize_tag(name: str) -> str:
#     return name.lower()


# # ---------------- Evaluation core ----------------

# def _extract_per_seat_vector(term: Dict[str, Any], keys: List[str], default_len: int = 4):
#     """
#     Try multiple keys on `term` and return a per-seat list if found.
#     Handles:
#       - list/tuple length >= default_len
#       - dict seat->value
#     Returns None if nothing usable is found.
#     """
#     for k in keys:
#         if k not in term:
#             continue
#         v = term[k]
#         if isinstance(v, dict):
#             return [float(v.get(i, 0.0)) for i in range(default_len)]
#         if isinstance(v, (list, tuple)) and len(v) >= default_len:
#             return [float(x) for x in v[:default_len]]
#     return None


# def evaluate_pair(
#     rules_path: str,
#     target: str,
#     opponent: str,
#     episodes: int = 1000,
#     seed: int | None = None,
#     enforce_zero_sum: bool = True,
# ):
#     """
#     Run `target` vs 3×`opponent` for a number of episodes.

#     Metrics:
#         - win_rate: fraction of episodes in which seat 0 (target) wins
#           (including multi-ron where seat 0 is one of the winners).
#         - total_points: sum over episodes of seat 0's net points
#           (same zero-sum logic as summarize_jsonl).
#         - avg_points: average per-episode net points for seat 0.
#     """
#     rng = random.Random(seed)
#     rules = load_rules(rules_path)

#     t_tag = canonicalize_tag(target)
#     o_tag = canonicalize_tag(opponent)

#     lineup = [t_tag, o_tag, o_tag, o_tag]

#     target_total_points = 0.0
#     target_wins = 0
#     draws = 0

#     for _ in range(episodes):
#         ep = run_episode(rules, lineup, tuner=None, seed=rng.randint(1, 10**9))
#         term = ep.get("terminal", {}) or {}

#         # Count wins for seat 0, following the same structure
#         w = term.get("winner")
#         winners_list = term.get("winners")

#         if w is None and not winners_list:
#             draws += 1
#         else:
#             if winners_list:
#                 # multi-ron case
#                 for item in winners_list:
#                     seat_w = int(item.get("seat", -1))
#                     if seat_w == 0:
#                         target_wins += 1
#             else:
#                 if isinstance(w, int) and w == 0:
#                     target_wins += 1

#         # Get per-seat deltas exactly like summarize_jsonl
#         deltas = _seat_deltas_from_terminal(term, rules, enforce_zero_sum=enforce_zero_sum)
#         target_total_points += deltas[0]

#     win_rate = target_wins / float(episodes) if episodes > 0 else 0.0
#     avg_points = target_total_points / float(episodes) if episodes > 0 else 0.0

#     return {
#         "win_rate": win_rate,
#         "avg_points": avg_points,       # per-episode net points
#         "total_points": target_total_points,  # what you were calling "total handscore"
#         "episodes": episodes,
#         "draws": draws,
#     }



# def summarize_grid(
#     rules_path: str,
#     policies: list[str],
#     episodes: int = 1000,
#     seed: int | None = None,
# ):
#     """
#     Build a matrix of stats for all policy pairs.

#     stats[target][opponent] = {
#         "win_rate": float,
#         "avg_points": float,
#         "total_handscore": float,
#         "episodes": int,
#     }
#     """
#     rng = random.Random(seed)
#     stats: Dict[str, Dict[str, Dict[str, float]]] = {}

#     for target in policies:
#         row: Dict[str, Dict[str, float]] = {}
#         for opponent in policies:
#             print(f"[eval] target={target} vs opponent={opponent}×3 …")
#             s = evaluate_pair(
#                 rules_path=rules_path,
#                 target=target,
#                 opponent=opponent,
#                 episodes=episodes,
#                 seed=rng.randint(1, 10**9),
#             )
#             row[opponent] = s
#         stats[target] = row

#     return stats


# # ---------------- Plot helpers ----------------

# def _build_matrix(stats: dict, metric: str):
#     """
#     Turn stats dict into (labels, matrix) for a given metric.
#     labels: list of policy names (rows and cols in same order)
#     matrix[i,j] = stats[labels[i]][labels[j]][metric]
#     """
#     labels = list(stats.keys())
#     n = len(labels)
#     mat = np.zeros((n, n), dtype=float)

#     for i, t in enumerate(labels):
#         for j, o in enumerate(labels):
#             mat[i, j] = float(stats[t][o][metric])

#     return labels, mat


# def plot_metric_matrix(
#     stats: dict,
#     metric: str,
#     fname_prefix: str,
#     cmap: str = "viridis",
#     vmin: float | None = None,
#     vmax: float | None = None,
#     center_zero: bool = False,
#     title: str | None = None,
# ):
#     """
#     Save a heatmap (PDF + PNG) for a given metric in `stats`.
#     """
#     labels, mat = _build_matrix(stats, metric)

#     if center_zero:
#         # symmetric color scale around 0
#         abs_max = float(np.max(np.abs(mat))) or 1.0
#         vmin = -abs_max
#         vmax = abs_max

#     fig, ax = plt.subplots(figsize=(6.0, 5.0))
#     im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)

#     # Axes labels / ticks
#     ax.set_xticks(range(len(labels)))
#     ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
#     ax.set_yticks(range(len(labels)))
#     ax.set_yticklabels(labels, fontsize=9)

#     if title is None:
#         title = metric
#     ax.set_title(title, fontsize=12)

#     # Annotate cells
#     for i in range(len(labels)):
#         for j in range(len(labels)):
#             val = mat[i, j]
#             text = f"{val:.2f}"
#             ax.text(
#                 j,
#                 i,
#                 text,
#                 ha="center",
#                 va="center",
#                 fontsize=7,
#             )

#     cbar = fig.colorbar(im, ax=ax)
#     cbar.ax.tick_params(labelsize=8)

#     plt.tight_layout()

#     pdf_path = f"{fname_prefix}.pdf"
#     png_path = f"{fname_prefix}.png"
#     fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
#     fig.savefig(png_path, bbox_inches="tight", dpi=300)
#     print(f"[plot] saved {pdf_path} and {png_path}")

#     plt.close(fig)


# def make_all_plots(stats: dict, out_json: str = "compare_results.json"):
#     """
#     Save stats JSON and produce:
#       - policy_matrix_winrate.(pdf|png)
#       - policy_matrix_total_handscore.(pdf|png)
#     """
#     Path(out_json).write_text(json.dumps(stats, indent=2))
#     print(f"[stats] wrote {out_json}")

#     # Win rate: between 0 and 1
#     plot_metric_matrix(
#         stats,
#         metric="win_rate",
#         fname_prefix="policy_matrix_winrate",
#         cmap="Blues",
#         vmin=0.0,
#         vmax=1.0,
#         title="Win rate (rows = target, cols = 3×opponent)",
#     )

#     # Total hand score for the target policy
#     plot_metric_matrix(
#         stats,
#         metric="total_points",
#         fname_prefix="policy_matrix_total_handscore",
#         cmap="magma",
#         center_zero=False,
#         title="Total hand score (rows = target, cols = 3×opponent)",
#     )


# # ---------------- CLI ----------------

# if __name__ == "__main__":
#     import argparse

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--rules", required=True)
#     ap.add_argument("--episodes", type=int, default=1000)
#     ap.add_argument(
#         "--policies",
#         type=str,
#         # You can still use flexibleaggro/flexibleaggrod here; they get canonicalized.
#         default="random,wp,payout,aggro,hyaggro,flexibleaggro,flexibleaggrod",
#     )
#     ap.add_argument("--seed", type=int, default=12345)
#     args = ap.parse_args()

#     raw_policies = [p.strip() for p in args.policies.split(",") if p.strip()]
#     policies = [canonicalize_tag(p) for p in raw_policies]

#     stats = summarize_grid(
#         rules_path=args.rules,
#         policies=policies,
#         episodes=args.episodes,
#         seed=args.seed,
#     )
#     make_all_plots(stats)
