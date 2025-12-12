# algorithm/mahjongrl/shanten_victory_with_firstmove.py

from __future__ import annotations
import argparse, random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import collections
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from algorithm.rules_io import load_rules
from algorithm.sim_and_train import (
    run_episode,
    rough_shanten_like,
    chiitoi_shanten_like,
    rough_ukeire,
    is_flower
)

def offline_ukeire_from_hand(
    tiles: List[str],
    declared_melds: int = 0,
) -> int:
    """
    Approximate ukeire using only the hand (no Env):

      - Base shanten = rough_shanten_like(hand, declared_melds)
      - Assume at most 4 copies of each tile in the game.
      - 'Visible' = copies in this hand only (we can't see discards / others offline).
      - For each tile class t:
          copies_left = max(0, 4 - count_in_hand(t))
          simulate drawing t; if shanten improves, all copies_left are outs.

    This is weaker than the full env-based rough_ukeire, but it's well-defined
    in this offline analysis script.
    """
    # Strip flowers; they don't matter for shanten
    hand = [t for t in tiles if not is_flower(t)]

    base = rough_shanten_like(hand, declared_melds=declared_melds)

    vis = collections.Counter(hand)  # only this hand is "visible" offline

    outs = 0
    for t in ALL_TILE_CLASSES:
        copies_left = max(0, 4 - vis.get(t, 0))
        if copies_left <= 0:
            continue

        new_hand = hand + [t]
        new_sh = rough_shanten_like(new_hand, declared_melds=declared_melds)

        if new_sh < base - 1e-9:
            outs += copies_left

    return outs


# -------------------------------------------------------
#   Composite shape metric (online version)
# -------------------------------------------------------
def composite_shape_metric_online(
    env,
    seat: int,
    chiitoi_weight: float = 0.3,
    declared_melds: int = 0,
) -> float:
    tiles = [t for t in env.players[seat].concealed if not is_flower(t)]
    std_core = rough_shanten_like(tiles, declared_melds=declared_melds)
    uke = rough_ukeire(env, seat, tiles, declared_melds=declared_melds)
    ctt = chiitoi_shanten_like(tiles, declared_melds=declared_melds)
    w = max(0.0, min(1.0, chiitoi_weight))
    return (1.0 - w) * (std_core - 0.02 * uke) + w * ctt


# -------------------------------------------------------
#   Episode helpers
# -------------------------------------------------------
def extract_win_for_seat(term: Dict[str, Any], seat: int) -> int:
    winner = term.get("winner")
    winners_list = term.get("winners")
    if winner is None and not winners_list:
        return 0
    if isinstance(winners_list, list):
        for item in winners_list:
            try:
                if int(item.get("seat")) == seat:
                    return 1
            except Exception:
                pass
    if isinstance(winner, int) and winner == seat:
        return 1
    return 0


def extract_points_for_seat(term: Dict[str, Any], seat: int) -> float:
    """
    Extract per-seat point delta if available, else 0.
    """
    sd = term.get("side_delta")
    if isinstance(sd, list) and len(sd) == 4:
        return float(sd[seat])
    pts = term.get("points")
    if isinstance(pts, (int, float)) and term.get("winner") == seat:
        return float(pts)
    return 0.0


from typing import Dict, Any, Optional

def get_start_seat(ep: Dict[str, Any]) -> Optional[int]:
    """
    Try to infer who started / dealer.

    Priority:
      1) Explicit metadata fields if present.
      2) Otherwise: infer from opening hands using Env's dealing logic:
         exactly one seat has 14 tiles at opening (after flowers), others have 13.
    """
    # 1) Look for explicit metadata if it exists.
    term = ep.get("terminal", {}) or {}
    openings = ep.get("openings", {}) or {}

    for d in (ep, openings, term):
        if isinstance(d, dict):
            for k in (
                "start_seat",
                "dealer",
                "start_player",
                "first_player",
                "start_turn",
            ):
                v = d.get(k)
                if isinstance(v, int) and 0 <= v <= 3:
                    return v

    # 2) Infer from opening hands: after_flowers preferred, else before_flowers.
    for key in ("after_flowers", "before_flowers"):
        hands = openings.get(key)
        if not isinstance(hands, list) or len(hands) != 4:
            continue

        lengths = [len(h) for h in hands]
        try:
            max_len = max(lengths)
            min_len = min(lengths)
        except ValueError:
            continue

        # In this Env, we expect something like [13, 13, 14, 13]
        if max_len > min_len:
            candidates = [i for i, L in enumerate(lengths) if L == max_len]
            if len(candidates) == 1:
                return candidates[0]

    # 3) Fallback: unknown
    return None



def policy_tag_for_spec(spec) -> str:
    return spec if isinstance(spec, str) else "rl"

# -------------------------------------------------------
#   Simulation
# -------------------------------------------------------
def simulate_and_build_dataset(
    rules: Dict[str, Any],
    lineup_specs: List[str],
    episodes: int,
    base_seed: int,
    chiitoi_weight: float,
) -> pd.DataFrame:
    rng = random.Random(base_seed)
    rows: List[Dict[str, Any]] = []

    for ep_idx in range(episodes):
        seed = rng.randint(1, 10**9)

        # IMPORTANT: pass chiitoi_weight through if you want
        ep = run_episode(
            rules=rules,
            lineup=lineup_specs,
            tuner=None,
            seed=seed,
            max_draws=600,
            chiitoi_weight=chiitoi_weight,
        )

        term     = ep.get("terminal", {}) or {}
        openings = ep.get("openings", {}) or {}
        metrics  = ep.get("opening_metrics", None)

        hands = openings.get("after_flowers") or openings.get("before_flowers")
        if not hands or len(hands) != 4:
            continue

        start_seat = get_start_seat(ep)

        for seat in range(4):
            # --- get composite shape using live-env ukeire ---
            if metrics is not None and 0 <= seat < len(metrics):
                m = metrics[seat]
                # you can either:
                opening_shape = float(m["composite"])
                # or recombine std_core/ukeire/chiitoi here if you want to change weights
                # std_core = m["std_core"]
                # uke     = m["ukeire"]
                # ctt     = m["chiitoi"]
                # w       = chiitoi_weight
                # opening_shape = (1 - w) * (std_core - 0.02 * uke) + w * ctt
            else:
                # Fallback: offline only if metrics missing (shouldn't happen once patched)
                tiles = hands[seat]
                std_core = rough_shanten_like(tiles, declared_melds=0)
                ctt = chiitoi_shanten_like(tiles, declared_melds=0)
                # no env here, so no ukeire in fallback
                opening_shape = (1.0 - chiitoi_weight) * std_core + chiitoi_weight * ctt

            win    = extract_win_for_seat(term, seat)
            reward = extract_points_for_seat(term, seat)
            policy_tag = policy_tag_for_spec(lineup_specs[seat])
            is_first   = int(start_seat is not None and seat == start_seat)

            rows.append(
                {
                    "episode_id": ep_idx,
                    "seat": seat,
                    "policy_tag": policy_tag,
                    "opening_shape": opening_shape,
                    "win": win,
                    "reward": reward,
                    "start_seat": start_seat if start_seat is not None else -1,
                    "is_first": is_first,
                }
            )

        if (ep_idx + 1) % max(1, episodes // 10) == 0:
            print(f"[sim] {ep_idx + 1}/{episodes} episodes")

    return pd.DataFrame(rows)

# -------------------------------------------------------
#   Regression analyses
# -------------------------------------------------------

def run_regressions(df: pd.DataFrame):
    """
    Expected–reward decomposition:

      1) Logit for win probability:
           win ~ opening_shape (+ C(policy_tag) + is_first)

      2) OLS for win magnitude (conditional on reward > 0):
           reward ~ opening_shape + C(policy_tag) + is_first

      3) OLS for loss magnitude (conditional on reward < 0):
           loss_mag ~ opening_shape + C(policy_tag) + is_first,
           where loss_mag = -reward > 0 on losing hands.

      4) Combine into an approximate expected reward:
           E_hat[R | X] = p_hat_win * mu_hat_win
                         - (1 - p_hat_win) * mu_hat_loss

    And then report correlations / a small summary.
    """
    df = df.copy()
    df["policy_tag"] = df["policy_tag"].astype("category")

    # --------------------------------------------------
    # 1) Win probability models (same idea as before)
    # --------------------------------------------------
    print("\n========== Model 1 (logit): win ~ opening_shape ==========")
    m1 = smf.logit("win ~ opening_shape", data=df).fit(disp=True)
    print(m1.summary())
    corr = df[["opening_shape", "win"]].corr().iloc[0, 1]
    print(f"\n[Model 1] Pearson corr(opening_shape, win) = {corr:.4f}")

    # Degenerate policy_tag handling
    n_policies = df["policy_tag"].nunique()
    if n_policies > 1:
        formula2 = "win ~ opening_shape + C(policy_tag) + is_first"
        print("\n========== Model 2 (logit): win ~ opening_shape + C(policy_tag) + is_first ==========")
    else:
        formula2 = "win ~ opening_shape + is_first"
        print("\n[info] Only one policy_tag present; dropping C(policy_tag) to avoid collinearity.")
        print("========== Model 2 (logit): win ~ opening_shape + is_first ==========")

    m2 = smf.logit(formula2, data=df).fit(disp=True)
    print(m2.summary())

    # --------------------------------------------------
    # 2) Split reward into wins and losses
    # --------------------------------------------------
    win_mask = df["reward"] > 0
    loss_mask = df["reward"] < 0

    n_total = len(df)
    n_win   = int(win_mask.sum())
    n_loss  = int(loss_mask.sum())
    n_zero  = n_total - n_win - n_loss
    
    print("\n[decomp] reward sign counts:")
    print(f"    total rows : {n_total}")
    print(f"    reward > 0 : {n_win}")
    print(f"    reward < 0 : {n_loss}")
    print(f"    reward = 0 : {n_zero} (draws / neutral hands)")

    # --------------------------------------------------
    # 3) Conditional magnitude models
    # --------------------------------------------------
    # 3a) Win magnitude: reward | reward > 0
    if n_win > 10:  # crude sanity check
        print("\n========== Model W (OLS): reward | reward>0 ==========")
        m_win = smf.ols(
            "reward ~ opening_shape + C(policy_tag) + is_first",
            data=df[win_mask]
        ).fit()
        print(m_win.summary())
    else:
        print("\n[warn] Too few winning hands to fit win-magnitude model reliably.")
        m_win = None

    # 3b) Loss magnitude: loss_mag = -reward on losing hands
    df["loss_mag"] = -df["reward"]
    if n_loss > 10:
        print("\n========== Model L (OLS): loss_mag | reward<0 ==========")
        m_loss = smf.ols(
            "loss_mag ~ opening_shape + C(policy_tag) + is_first",
            data=df[loss_mask]
        ).fit()
        print(m_loss.summary())
    else:
        print("\n[warn] Too few losing hands to fit loss-magnitude model reliably.")
        m_loss = None

    # --------------------------------------------------
    # 4) Expected reward decomposition
    # --------------------------------------------------
    # p_hat_win(X) from logistic model 2
    try:
        df["p_win_hat"] = m2.predict(df)
    except Exception as e:
        print(f"\n[warn] failed to compute p_win_hat from m2: {e}")
        df["p_win_hat"] = 0.0

    # Clip to [0,1] just in case of numerical oddities
    df["p_win_hat"] = df["p_win_hat"].clip(0.0, 1.0)

    # mu_hat_win(X): predicted win magnitude for *all* X
    if m_win is not None:
        df["mu_win_hat"] = m_win.predict(df)
    else:
        df["mu_win_hat"] = df["reward"].where(win_mask, df["reward"][win_mask].mean() if n_win > 0 else 0.0)

    # mu_hat_loss(X): predicted loss magnitude (positive) for *all* X
    if m_loss is not None:
        df["mu_loss_hat"] = m_loss.predict(df)
    else:
        mean_loss_mag = df.loc[loss_mask, "loss_mag"].mean() if n_loss > 0 else 0.0
        df["mu_loss_hat"] = mean_loss_mag

    # Approximate p_loss(X) as 1 - p_win(X) (draws get 0 reward anyway)
    df["p_loss_hat"] = 1.0 - df["p_win_hat"]

    # Expected reward:
    #   E_hat[R|X] = p_win * mu_win  -  p_loss * mu_loss
    df["E_reward_hat"] = (
        df["p_win_hat"]  * df["mu_win_hat"]
        - df["p_loss_hat"] * df["mu_loss_hat"]
    )

    overall_E_hat = df["E_reward_hat"].mean()
    overall_R_emp = df["reward"].mean()

    print("\n========== Expected reward decomposition summary ==========")
    print(f"  Empirical mean reward        : {overall_R_emp:.4f}")
    print(f"  Decomposed E_hat[reward | X] : {overall_E_hat:.4f}")

    corr_E_shape = df[["opening_shape", "E_reward_hat"]].corr().iloc[0, 1]
    print(f"  Corr(opening_shape, E_hat[R]): {corr_E_shape:.4f}")

    # You can also return the models & augmented df for further plotting if you want
    return m1, m2, m_win, m_loss, df


# def run_regressions(df: pd.DataFrame):
#     df = df.copy()
#     df["policy_tag"] = df["policy_tag"].astype("category")

#     print("\n========== Model 1 (logit): win ~ opening_shape ==========")
#     m1 = smf.logit("win ~ opening_shape", data=df).fit(disp=True)
#     print(m1.summary())
#     corr = df[["opening_shape", "win"]].corr().iloc[0, 1]
#     print(f"\n[Model 1] Pearson corr(opening_shape, win) = {corr:.4f}")

#     # --- Model 2: handle degenerate policy_tag gracefully ---
#     n_policies = df["policy_tag"].nunique()
#     if n_policies > 1:
#         formula2 = "win ~ opening_shape + C(policy_tag) + is_first"
#         print("\n========== Model 2 (logit): win ~ opening_shape + C(policy_tag) + is_first ==========")
#     else:
#         formula2 = "win ~ opening_shape + is_first"
#         print("\n[info] Only one policy_tag present; dropping C(policy_tag) to avoid collinearity.")
#         print("========== Model 2 (logit): win ~ opening_shape + is_first ==========")

#     m2 = smf.logit(formula2, data=df).fit(disp=True)
#     print(m2.summary())
#         # --- Continuous: reward ---
#     print("\n========== Model 3 (OLS): reward ~ opening_shape ==========")
#     m3 = smf.ols("reward ~ opening_shape", data=df).fit()
#     print(m3.summary())
#     corr_rew = df[["opening_shape", "reward"]].corr().iloc[0, 1]
#     print(f"\n[Model 5] Pearson corr(opening_shape, reward) = {corr_rew:.4f}")

#     print("\n========== Model 4 (OLS): reward ~ opening_shape + C(policy_tag) + is_first ==========")
#     m4 = smf.ols("reward ~ opening_shape + C(policy_tag) + is_first", data=df).fit()
#     print(m4.summary())

#     return m1, m2, m3, m4


# -------------------------------------------------------
#   CLI
# -------------------------------------------------------
def parse_lineup_arg(lineup_str: str) -> List[str]:
    parts = [x.strip() for x in lineup_str.split(",") if x.strip()]
    if not parts:
        raise ValueError("Lineup string must have at least one policy tag.")
    if len(parts) < 4:
        parts += [parts[-1]] * (4 - len(parts))
    elif len(parts) > 4:
        parts = parts[:4]
    return parts

def run_simple_reward_regression(df: pd.DataFrame):
    """
    Simple fixed-effects reward model (no game-level random effect):

        reward_{g,s} = μ + α_{policy(g,s)} + β * opening_shape_{g,s}
                       + γ * is_first_{g,s} + ε_{g,s}.

    We also cluster standard errors by episode_id to respect
    the within-game correlation between the 4 seats.
    """
    df = df.copy()
    df["policy_tag"] = df["policy_tag"].astype("category")

    formula = "reward ~ opening_shape + C(policy_tag) + is_first"
    print("\n========== Simple reward model ==========")
    print(f"Formula: {formula}")
    print("SEs clustered by episode_id\n")

    # Cluster-robust SEs by game (episode)
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["episode_id"]}
    )

    print(model.summary())
    return model

def main():
    ap = argparse.ArgumentParser(description="Analyze correlation between opening shape, win, and dealer advantage.")
    ap.add_argument("--rules", required=True)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lineup", type=str, default="flexaggrod,flexaggrod,flexaggrod,flexaggrod")
    ap.add_argument("--chiitoi-weight", type=float, default=0.3)
    ap.add_argument("--output-csv", type=str, default="shanten_victory_with_firstmove.csv")
    args = ap.parse_args()

    rules = load_rules(args.rules)
    lineup_specs = parse_lineup_arg(args.lineup)
    print(f"[config] lineup={lineup_specs}, episodes={args.episodes}")

    df = simulate_and_build_dataset(
        rules, lineup_specs, args.episodes, args.seed, args.chiitoi_weight
    )
    print(f"[sim] built dataset with {len(df)} rows")

    out_csv = Path(args.output_csv)
    df.to_csv(out_csv, index=False)
    print(f"[save] wrote {out_csv}")

    run_regressions(df)
    run_simple_reward_regression(df)  # the clean R_{g,s} ~ shape + policy + dealer model

    print("\n✅ Done — models completed with dealer/first-move factor.")




if __name__ == "__main__":
    main()


# # algorithm/mahjongrl/shanten_victory_sim_and_analysis.py

# from __future__ import annotations

# import argparse
# import json
# import random
# from pathlib import Path
# from typing import Dict, Any, List, Tuple

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import matplotlib.pyplot as plt

# from algorithm.rules_io import load_rules
# from algorithm.sim_and_train import (
#     run_episode,
#     rough_shanten_like,
#     chiitoi_shanten_like,
# )
# from algorithm.mahjongrl.env_wrapper import compute_rl_reward


# # ---------- Composite opening shape metric (offline) ----------

# def composite_shape_metric_offline(
#     tiles: List[str],
#     chiitoi_weight: float = 0.3,
#     declared_melds: int = 0,
# ) -> float:
#     """
#     Composite "shape" metric for an opening hand, using the same shanten
#     primitives as the training code:

#         std_core = rough_shanten_like(...)
#         ctt      = chiitoi_shanten_like(...)

#     We do NOT use ukeire/env here so this can be computed from the
#     saved hand alone. Lower = better.

#     chiitoi_weight in [0,1] trades off standard vs chiitoi path.
#     """
#     std_core = rough_shanten_like(tiles, declared_melds=declared_melds)
#     ctt = chiitoi_shanten_like(tiles, declared_melds=declared_melds)
#     w = max(0.0, min(1.0, chiitoi_weight))
#     return (1.0 - w) * float(std_core) + w * float(ctt)


# # ---------- Helpers to extract info from a run_episode result ----------

# def extract_win_for_seat(term: Dict[str, Any], seat: int) -> int:
#     """
#     Binary label:
#         win = 1 if this seat is a winner (single or multi-ron),
#         win = 0 otherwise.
#     """
#     winner = term.get("winner", None)
#     winners_list = term.get("winners", None)

#     if winner is None and not winners_list:
#         # Drawn hand, no winner
#         return 0

#     # Multi-ron case
#     if isinstance(winners_list, list):
#         for item in winners_list:
#             try:
#                 s = int(item.get("seat"))
#                 if s == seat:
#                     return 1
#             except Exception:
#                 pass

#     # Single winner case
#     if isinstance(winner, int) and winner == seat:
#         return 1

#     return 0

# def extract_points_for_seat(term: Dict[str, Any], seat: int) -> float:
#     """
#     Return this seat's point outcome for the hand.
#     Tries a few plausible layouts:
#       - term["points"] is a list length 4
#       - term["points"] is a dict keyed by seat
#     Falls back to 0.0 if not present.
#     """
#     pts = term.get("points", None) or term.get("scores", None) or term.get("deltas", None)
#     if pts is None:
#         return 0.0

#     # list/tuple case
#     if isinstance(pts, (list, tuple)) and len(pts) >= 4:
#         try:
#             return float(pts[seat])
#         except Exception:
#             return 0.0

#     # dict case: keys might be ints or strings
#     if isinstance(pts, dict):
#         if seat in pts:
#             return float(pts[seat])
#         if str(seat) in pts:
#             return float(pts[str(seat)])
#         return 0.0

#     # Fallback: unknown shape
#     return 0.0

# # def extract_points_for_seat(term: Dict[str, Any], seat: int) -> float:
# #     """
# #     Try to pull a per-seat point delta from the terminal dict.

# #     This is intentionally defensive: it checks several likely key names.
# #     If nothing is found, returns 0.0.
# #     """
# #     if not isinstance(term, dict):
# #         return 0.0

# #     # Try a few common patterns; adjust if your run_episode uses different keys.
# #     for key in ("points_delta", "delta_points", "score_delta", "delta_score", "points"):
# #         val = term.get(key)
# #         if isinstance(val, (list, tuple)) and len(val) > seat:
# #             try:
# #                 return float(val[seat])
# #             except Exception:
# #                 pass
# #         if isinstance(val, dict):
# #             # e.g. {0: x0, 1: x1, ...}
# #             try:
# #                 if seat in val:
# #                     return float(val[seat])
# #             except Exception:
# #                 pass

# #     return 0.0


# def policy_tag_for_spec(spec) -> str:
#     """
#     Convert a lineup spec entry into a string tag for regression.
#     For now:
#       - if it's a string, use it;
#       - otherwise label it generically as 'rl' (for RLPolicy factories).
#     """
#     if isinstance(spec, str):
#         return spec
#     return "rl"


# # ---------- Simulate episodes + build dataset ----------

# def simulate_and_build_dataset(
#     rules: Dict[str, Any],
#     lineup_specs: List,
#     episodes: int,
#     base_seed: int,
#     chiitoi_weight: float,
# ) -> pd.DataFrame:
#     """
#     For each episode, run run_episode(rules, lineup, ...),
#     then for each seat s in {0,1,2,3}:

#       - compute opening_shape from openings["after_flowers"][s]
#       - compute win label from terminal
#       - compute points (per-hand point delta)
#       - compute reward via compute_rl_reward
#       - record policy_tag from lineup_specs[s]

#     Returns a pandas DataFrame with columns:
#         episode_id, policy_tag, win, opening_shape, points, reward
#     """
#     rng = random.Random(base_seed)
#     rows: List[Dict[str, Any]] = []

#     for ep_idx in range(episodes):
#         seed = rng.randint(1, 10**9)

#         ep = run_episode(
#             rules=rules,
#             lineup=lineup_specs,
#             tuner=None,
#             seed=seed,
#         )

#         term = ep.get("terminal", {}) or {}
#         openings = ep.get("openings", {}) or {}

#         # Prefer after_flowers (actual starting hands)
#         hands = openings.get("after_flowers") or openings.get("before_flowers")
#         if not hands or len(hands) != 4:
#             # Skip if malformed
#             continue

#         for seat in range(4):
#             tiles = hands[seat]

#             opening_shape = composite_shape_metric_offline(
#                 tiles,
#                 chiitoi_weight=chiitoi_weight,
#                 declared_melds=0,  # at opening, no declared melds yet
#             )

#             win = extract_win_for_seat(term, seat)
#             policy_tag = policy_tag_for_spec(lineup_specs[seat])

#             # Continuous outcomes
#             pts = extract_points_for_seat(term, seat)
#             rew = compute_rl_reward(term or {"source": "drawn_game"}, seat=seat, rules=rules)

#             rows.append(
#                 {
#                     "episode_id": ep_idx,
#                     "policy_tag": policy_tag,
#                     "win": int(win),
#                     "opening_shape": float(opening_shape),
#                     "points": float(pts),
#                     "reward": float(rew),
#                 }
#             )

#         if (ep_idx + 1) % max(1, episodes // 10) == 0:
#             print(f"[sim] completed {ep_idx + 1}/{episodes} episodes")

#     if not rows:
#         raise RuntimeError("No dataset rows produced; check lineup and run_episode outputs.")

#     df = pd.DataFrame(rows)
#     return df



# # ---------- Logistic + linear regression analysis ----------

# def run_regressions(df: pd.DataFrame, out_dir: Path | None = None):
#     """
#     Run:
#       (1) Logit: win ~ opening_shape
#       (2) Logit: win ~ opening_shape + C(policy_tag)
#       (3) OLS:   points ~ opening_shape
#       (4) OLS:   points ~ opening_shape + C(policy_tag)
#       (5) OLS:   reward ~ opening_shape
#       (6) OLS:   reward ~ opening_shape + C(policy_tag)
#     """
#     df["policy_tag"] = df["policy_tag"].astype("category")

#     # --- Binary (win) models ---
#     print("\n========== Model 1: win ~ opening_shape ==========")
#     m1 = smf.logit("win ~ opening_shape", data=df).fit(disp=True)
#     print(m1.summary())
#     corr_win = df[["opening_shape", "win"]].corr().iloc[0, 1]
#     print(f"\n[Model 1] Pearson corr(opening_shape, win) = {corr_win:.4f}")

#     print("\n========== Model 2: win ~ opening_shape + C(policy_tag) ==========")
#     m2 = smf.logit("win ~ opening_shape + C(policy_tag)", data=df).fit(disp=True)
#     print(m2.summary())

#     # --- Continuous: points ---
#     print("\n========== Model 3 (OLS): points ~ opening_shape ==========")
#     m3 = smf.ols("points ~ opening_shape", data=df).fit()
#     print(m3.summary())
#     corr_pts = df[["opening_shape", "points"]].corr().iloc[0, 1]
#     print(f"\n[Model 3] Pearson corr(opening_shape, points) = {corr_pts:.4f}")

#     print("\n========== Model 4 (OLS): points ~ opening_shape + C(policy_tag) ==========")
#     m4 = smf.ols("points ~ opening_shape + C(policy_tag)", data=df).fit()
#     print(m4.summary())

#     # --- Continuous: reward ---
#     print("\n========== Model 5 (OLS): reward ~ opening_shape ==========")
#     m5 = smf.ols("reward ~ opening_shape", data=df).fit()
#     print(m5.summary())
#     corr_rew = df[["opening_shape", "reward"]].corr().iloc[0, 1]
#     print(f"\n[Model 5] Pearson corr(opening_shape, reward) = {corr_rew:.4f}")

#     print("\n========== Model 6 (OLS): reward ~ opening_shape + C(policy_tag) ==========")
#     m6 = smf.ols("reward ~ opening_shape + C(policy_tag)", data=df).fit()
#     print(m6.summary())

#     # Optional plotting
#     if out_dir is not None:
#         make_plots(df, out_dir)

#     return m1, m2, m3, m4, m5, m6


# # ---------- Plotting ----------

# def make_plots(df: pd.DataFrame, out_dir: Path):
#     """
#     Produce a few simple plots:
#       - scatter: opening_shape vs points
#       - scatter: opening_shape vs reward
#       - binned mean points vs opening_shape
#       - binned mean reward vs opening_shape
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Scatter: opening_shape vs points
#     plt.figure()
#     plt.scatter(df["opening_shape"], df["points"], alpha=0.3, s=8)
#     plt.xlabel("Opening shape (lower = better)")
#     plt.ylabel("Points (seat-level delta)")
#     plt.title("Points vs opening shape")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_dir / "points_vs_opening_shape_scatter.png")
#     plt.close()

#     # Scatter: opening_shape vs reward
#     plt.figure()
#     plt.scatter(df["opening_shape"], df["reward"], alpha=0.3, s=8)
#     plt.xlabel("Opening shape (lower = better)")
#     plt.ylabel("RL reward (seat-level)")
#     plt.title("Reward vs opening shape")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_dir / "reward_vs_opening_shape_scatter.png")
#     plt.close()

#     # Binned means for points
#     try:
#         df["shape_bin"] = pd.qcut(df["opening_shape"], q=10, duplicates="drop")
#     except Exception:
#         df["shape_bin"] = pd.cut(df["opening_shape"], bins=10)

#     grp_pts = df.groupby("shape_bin", observed=True)["points"].mean()
#     x = np.arange(len(grp_pts))

#     plt.figure()
#     plt.bar(x, grp_pts.values)
#     plt.xticks(x, grp_pts.index.astype(str), rotation=45, ha="right")
#     plt.xlabel("Opening shape decile (lower bins = better)")
#     plt.ylabel("Mean points")
#     plt.title("Mean points by opening-shape bin")
#     plt.tight_layout()
#     plt.savefig(out_dir / "points_by_opening_shape_bin.png")
#     plt.close()

#     # Binned means for reward
#     grp_rew = df.groupby("shape_bin", observed=True)["reward"].mean()
#     x = np.arange(len(grp_rew))

#     plt.figure()
#     plt.bar(x, grp_rew.values)
#     plt.xticks(x, grp_rew.index.astype(str), rotation=45, ha="right")
#     plt.xlabel("Opening shape decile (lower bins = better)")
#     plt.ylabel("Mean reward")
#     plt.title("Mean reward by opening-shape bin")
#     plt.tight_layout()
#     plt.savefig(out_dir / "reward_by_opening_shape_bin.png")
#     plt.close()


# # ---------- CLI / main ----------

# def parse_lineup_arg(lineup_str: str) -> List[str]:
#     """
#     Parse a lineup spec string like:

#         'flexaggro,flexaggro,flexaggro,flexaggro'
#         'flexaggro,hyaggro,wp,payout'

#     into a list of 4 policy tags.

#     If fewer than 4 tags are given, we repeat the last one.
#     If more than 4, we truncate to 4.
#     """
#     parts = [x.strip() for x in lineup_str.split(",") if x.strip()]
#     if not parts:
#         raise ValueError("Lineup string must contain at least one policy tag.")

#     if len(parts) < 4:
#         parts = parts + [parts[-1]] * (4 - len(parts))
#     elif len(parts) > 4:
#         parts = parts[:4]
#     return parts


# def main():
#     ap = argparse.ArgumentParser(
#         description="Simulate Mahjong episodes and analyze correlation between opening shanten-like metric and winning / points / reward."
#     )
#     ap.add_argument("--rules", required=True, help="Path to rules JSON (same as used in training).")
#     ap.add_argument(
#         "--episodes",
#         type=int,
#         default=5000,
#         help="Number of episodes to simulate (default: 5000).",
#     )
#     ap.add_argument(
#         "--seed",
#         type=int,
#         default=0,
#         help="Base random seed for simulation (default: 0).",
#     )
#     ap.add_argument(
#         "--lineup",
#         type=str,
#         default="flexaggrod,flexaggrod,flexaggrod,flexaggrod",
#         help=(
#             "Comma-separated lineup of policy tags for seats 0-3. "
#             "If fewer than 4 provided, last one repeats. "
#             "Example: 'flexaggro,flexaggro,flexaggro,flexaggro'."
#         ),
#     )
#     ap.add_argument(
#         "--chiitoi-weight",
#         type=float,
#         default=0.3,
#         help="Weight in [0,1] for chiitoi path in composite shape metric (default: 0.3).",
#     )
#     ap.add_argument(
#         "--output-csv",
#         type=str,
#         default="shanten_victory_sim_dataset.csv",
#         help="Where to save the per-seat dataset CSV (default: shanten_victory_sim_dataset.csv).",
#     )

#     args = ap.parse_args()

#     rules_path = Path(args.rules)
#     rules = load_rules(str(rules_path))

#     lineup_specs = parse_lineup_arg(args.lineup)
#     print(f"[config] lineup specs (seat 0..3) = {lineup_specs}")
#     print(f"[config] episodes = {args.episodes}, base_seed = {args.seed}")
#     print(f"[config] chiitoi_weight = {args.chiitoi_weight}")

#     print("[sim] starting simulation …")
#     df = simulate_and_build_dataset(
#         rules=rules,
#         lineup_specs=lineup_specs,
#         episodes=args.episodes,
#         base_seed=args.seed,
#         chiitoi_weight=args.chiitoi_weight,
#     )
#     print(f"[sim] dataset built with {len(df)} rows")

#     out_csv = Path(args.output_csv)
#     df.to_csv(out_csv, index=False)
#     print(f"[save] wrote dataset to {out_csv}")

#     # Run regressions + plots (plots go next to the CSV)
#     run_regressions(df, out_dir=out_csv.parent)

#     print("\n✅ Done. You now have:")
#     print(f"    • CSV dataset: {out_csv}")
#     print(f"    • Plots + regression summaries in: {out_csv.parent}")


# if __name__ == "__main__":
#     main()
