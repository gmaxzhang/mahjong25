# algorithm/mahjongrl/sim_compare.py
from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from algorithm.rules_io import load_rules
from algorithm.sim_and_train import run_episode, _parse_lineup  # adjust import if needed


# ---------------- Canonical policy tags ----------------

_CANONICAL_TAGS = {
    # flexible → flex
    "flexibleaggro": "flexaggro",
    "flexibleaggrod": "flexaggrod",
    # already accepted tags
    "flexaggro": "flexaggro",
    "flexaggrod": "flexaggrod",
    "random": "random",
    "wp": "wp",
    "payout": "payout",
    "hybrid": "hybrid",
    "aggro": "aggro",
    "hyaggro": "hyaggro",
}


from typing import Dict, Any, List

def _seat_deltas_from_terminal(term: Dict[str, Any],
                               rules: Dict[str, Any],
                               enforce_zero_sum: bool = True) -> List[int]:
    """
    Reproduce the per-seat point delta logic from sim_and_train.summarize_jsonl
    for a *single hand*.

    Returns a list of length 4: net points for seats 0..3 for this episode,
    including side payments.
    """
    delta = [0, 0, 0, 0]

    pay_disc = rules.get("payouts", {}).get("on_discard", {})
    pay_self = rules.get("payouts", {}).get("on_self_draw", {})

    w = term.get("winner")
    winners_list = term.get("winners")

    # Drawn game: no base payouts, just side_delta (added later)
    if w is None and not winners_list:
        pass
    else:
        src = term.get("source")

        # Multi-ron on a discard
        if winners_list:
            loser = term.get("ron_loser")
            wg = int(pay_disc.get("winner_gain", 1))
            ll = int(pay_disc.get("loser_loss", -1))
            if enforce_zero_sum and (wg + ll) != 0:
                wg, ll = 1, -1

            total_loser_loss = 0
            for item in winners_list:
                seat_w = int(item.get("seat", -1))
                pts = int(item.get("points", 0))
                if 0 <= seat_w < 4:
                    delta[seat_w] += wg * pts
                total_loser_loss += ll * pts

            if isinstance(loser, int) and 0 <= loser < 4:
                delta[loser] += total_loser_loss

        # Single winner (tsumo or ron)
        else:
            pts = int(term.get("points", 0))

            if not isinstance(w, int) or not (0 <= w < 4):
                # malformed, just bail to side_delta
                pts = int(term.get("points", 0))
            else:
                if src == "self_draw":
                    wg = int(pay_self.get("winner_gain", 1))
                    ol = int(pay_self.get("others_loss_each", -1))
                    if enforce_zero_sum and (wg + 3 * ol) != 0:
                        wg, ol = 3, -1

                    delta[w] += wg * pts
                    for s in range(4):
                        if s != w:
                            delta[s] += ol * pts

                elif src == "discard":
                    loser = term.get("ron_loser")
                    wg = int(pay_disc.get("winner_gain", 1))
                    ll = int(pay_disc.get("loser_loss", -1))
                    if enforce_zero_sum and (wg + ll) != 0:
                        wg, ll = 1, -1

                    delta[w] += wg * pts
                    if isinstance(loser, int) and 0 <= loser < 4:
                        delta[loser] += ll * pts

                # If source is "drawn_game" or unknown, we do nothing more here.

    # Add side payments (4-same-discard, etc.)
    sd = term.get("side_delta")
    if isinstance(sd, list) and len(sd) == 4:
        for i in range(4):
            try:
                delta[i] += int(sd[i])
            except Exception:
                pass

    return delta



def canonicalize_tag(name: str) -> str:
    return name.lower()


# ---------------- Evaluation core ----------------

def _extract_per_seat_vector(term: Dict[str, Any], keys: List[str], default_len: int = 4):
    """
    Try multiple keys on `term` and return a per-seat list if found.
    Handles:
      - list/tuple length >= default_len
      - dict seat->value
    Returns None if nothing usable is found.
    """
    for k in keys:
        if k not in term:
            continue
        v = term[k]
        if isinstance(v, dict):
            return [float(v.get(i, 0.0)) for i in range(default_len)]
        if isinstance(v, (list, tuple)) and len(v) >= default_len:
            return [float(x) for x in v[:default_len]]
    return None


def evaluate_pair(
    rules_path: str,
    target: str,
    opponent: str,
    episodes: int = 1000,
    seed: int | None = None,
    enforce_zero_sum: bool = True,
):
    """
    Run `target` vs 3×`opponent` for a number of episodes.

    Metrics:
        - win_rate: fraction of episodes in which seat 0 (target) wins
          (including multi-ron where seat 0 is one of the winners).
        - total_points: sum over episodes of seat 0's net points
          (same zero-sum logic as summarize_jsonl).
        - avg_points: average per-episode net points for seat 0.
    """
    rng = random.Random(seed)
    rules = load_rules(rules_path)

    t_tag = canonicalize_tag(target)
    o_tag = canonicalize_tag(opponent)

    lineup = [t_tag, o_tag, o_tag, o_tag]

    target_total_points = 0.0
    target_wins = 0
    draws = 0

    for _ in range(episodes):
        ep = run_episode(rules, lineup, tuner=None, seed=rng.randint(1, 10**9))
        term = ep.get("terminal", {}) or {}

        # Count wins for seat 0, following the same structure
        w = term.get("winner")
        winners_list = term.get("winners")

        if w is None and not winners_list:
            draws += 1
        else:
            if winners_list:
                # multi-ron case
                for item in winners_list:
                    seat_w = int(item.get("seat", -1))
                    if seat_w == 0:
                        target_wins += 1
            else:
                if isinstance(w, int) and w == 0:
                    target_wins += 1

        # Get per-seat deltas exactly like summarize_jsonl
        deltas = _seat_deltas_from_terminal(term, rules, enforce_zero_sum=enforce_zero_sum)
        target_total_points += deltas[0]

    win_rate = target_wins / float(episodes) if episodes > 0 else 0.0
    avg_points = target_total_points / float(episodes) if episodes > 0 else 0.0

    return {
        "win_rate": win_rate,
        "avg_points": avg_points,       # per-episode net points
        "total_points": target_total_points,  # what you were calling "total handscore"
        "episodes": episodes,
        "draws": draws,
    }



def summarize_grid(
    rules_path: str,
    policies: list[str],
    episodes: int = 1000,
    seed: int | None = None,
):
    """
    Build a matrix of stats for all policy pairs.

    stats[target][opponent] = {
        "win_rate": float,
        "avg_points": float,
        "total_handscore": float,
        "episodes": int,
    }
    """
    rng = random.Random(seed)
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for target in policies:
        row: Dict[str, Dict[str, float]] = {}
        for opponent in policies:
            print(f"[eval] target={target} vs opponent={opponent}×3 …")
            s = evaluate_pair(
                rules_path=rules_path,
                target=target,
                opponent=opponent,
                episodes=episodes,
                seed=rng.randint(1, 10**9),
            )
            row[opponent] = s
        stats[target] = row

    return stats


# ---------------- Plot helpers ----------------

def _build_matrix(stats: dict, metric: str):
    """
    Turn stats dict into (labels, matrix) for a given metric.
    labels: list of policy names (rows and cols in same order)
    matrix[i,j] = stats[labels[i]][labels[j]][metric]
    """
    labels = list(stats.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)

    for i, t in enumerate(labels):
        for j, o in enumerate(labels):
            mat[i, j] = float(stats[t][o][metric])

    return labels, mat


def plot_metric_matrix(
    stats: dict,
    metric: str,
    fname_prefix: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    center_zero: bool = False,
    title: str | None = None,
):
    """
    Save a heatmap (PDF + PNG) for a given metric in `stats`.
    """
    labels, mat = _build_matrix(stats, metric)

    if center_zero:
        # symmetric color scale around 0
        abs_max = float(np.max(np.abs(mat))) or 1.0
        vmin = -abs_max
        vmax = abs_max

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)

    # Axes labels / ticks
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    if title is None:
        title = metric
    ax.set_title(title, fontsize=12)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            text = f"{val:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    pdf_path = f"{fname_prefix}.pdf"
    png_path = f"{fname_prefix}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"[plot] saved {pdf_path} and {png_path}")

    plt.close(fig)


def make_all_plots(stats: dict, out_json: str = "compare_results.json"):
    """
    Save stats JSON and produce:
      - policy_matrix_winrate.(pdf|png)
      - policy_matrix_total_handscore.(pdf|png)
    """
    Path(out_json).write_text(json.dumps(stats, indent=2))
    print(f"[stats] wrote {out_json}")

    # Win rate: between 0 and 1
    plot_metric_matrix(
        stats,
        metric="win_rate",
        fname_prefix="policy_matrix_winrate",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        title="Win rate (rows = target, cols = 3×opponent)",
    )

    # Total hand score for the target policy
    plot_metric_matrix(
        stats,
        metric="total_points",
        fname_prefix="policy_matrix_total_handscore",
        cmap="magma",
        center_zero=False,
        title="Total hand score (rows = target, cols = 3×opponent)",
    )


# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument(
        "--policies",
        type=str,
        # You can still use flexibleaggro/flexibleaggrod here; they get canonicalized.
        default="random,wp,payout,aggro,hyaggro,flexibleaggro,flexibleaggrod",
    )
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    raw_policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    policies = [canonicalize_tag(p) for p in raw_policies]

    stats = summarize_grid(
        rules_path=args.rules,
        policies=policies,
        episodes=args.episodes,
        seed=args.seed,
    )
    make_all_plots(stats)
