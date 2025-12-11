# algorithm/mahjongrl/train.py
from __future__ import annotations
import argparse, random, sys, traceback, os, time, json, pickle, hashlib
from typing import List, Optional, Callable, Dict, Tuple, Any
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # safe for headless training
import matplotlib.pyplot as plt
from copy import deepcopy
from contextlib import contextmanager
import collections
from collections import deque
import multiprocessing as mp
import types
from collections import Counter
from algorithm.rules_io import load_rules
from algorithm.sim_and_train import Env
from algorithm.sim_and_train import _is_suit_tile, _tile_rank_suit, is_flower#, _pung_claims, _chow_claim, _ron_window, _maybe_closed_kongs, _maybe_added_kongs
from algorithm.mahjongrl.env_wrapper import make_lineup_with_rl, build_observation, compute_rl_reward, compute_tenpai_flags, _tile_code
from algorithm.mahjongrl.model import ACConfig, LSTMActorCritic

from algorithm.sim_and_train import FlexibleAggroPolicyD, HybridAggroPolicy 

from algorithm.mahjongrl.env_wrapper import compute_rl_reward

import types

# # ---------- Simple global win stats (per process) ----------
# win_counts = [0, 0, 0, 0]
# hand_count = 0

# def _record_win_stats(env, print_every: int = 100):
#     """
#     Increment per-seat win counters whenever a hand ends.
#     This is cheap and only for debugging / monitoring.
#     """
#     global win_counts, hand_count

#     term = getattr(env, "terminal", None)
#     hand_count += 1

#     if term is not None:
#         # Single winner case
#         if term.get("winner") is not None:
#             w = term["winner"]
#             if 0 <= w < 4:
#                 win_counts[w] += 1

#         # Multi-ron case
#         elif term.get("winners") is not None:
#             for winfo in term["winners"]:
#                 s = winfo.get("seat", None)
#                 if s is not None and 0 <= s < 4:
#                     win_counts[s] += 1

#     if hand_count % print_every == 0:
#         seat0_rate = win_counts[0] / max(hand_count, 1)
#         print(f"[stats] hands={hand_count} wins={win_counts} "
#               f"seat0_win_rate={seat0_rate:.3f}")


# def patched_step_turn(self, policies):
#     if getattr(self, "terminal", False):
#         return
#     seat = self.turn

#     # last_discard is either None or (discarder, tile_str)
#     last = getattr(self, "last_discard", None)

#     try:
#         # Only check ron/pung/chow if there’s an actual discard tile
#         if last is not None:
#             discarder, tile = last  # tile is now the string, e.g. "7w"

#             if hasattr(self, "_ron_window"):
#                 self._ron_window(discarder, tile, policies)
#             if hasattr(self, "_pung_claims"):
#                 self._pung_claims(discarder, tile, policies)
#             if hasattr(self, "_chow_claim"):
#                 self._chow_claim(discarder, tile, policies)

#         # Kongs can occur even without a discard
#         if hasattr(self, "_maybe_closed_kongs"):
#             self._maybe_closed_kongs(seat, policies)
#         if hasattr(self, "_maybe_added_kongs"):
#             self._maybe_added_kongs(seat, policies)

#     except Exception as e:
#         import traceback
#         print(f"[patch-step_turn-error] seat={seat} turn={self.turn} last_discard={last}")
#         print(f"[patch-step_turn-error] exception={e}")
#         traceback.print_exc()

#     # Always fall back to normal step logic
#     return Env._orig_step_turn(self, policies)

# # Apply patch
# if not hasattr(Env, "_orig_step_turn"):
#     Env._orig_step_turn = Env.step_turn
# Env.step_turn = patched_step_turn


# def patched_step_turn(self, policies):
#     if getattr(self, "terminal", False):
#         return
#     seat = self.turn
#     tile = getattr(self, "last_discard", None)
#     try:
#         # Only check ron/pung/chow if there’s an actual discard tile
#         if tile is not None:
#             if hasattr(self, "_ron_window"):
#                 self._ron_window(seat, tile, policies)
#             if hasattr(self, "_pung_claims"):
#                 self._pung_claims(seat, tile, policies)
#             if hasattr(self, "_chow_claim"):
#                 self._chow_claim(seat, tile, policies)

#         # Kongs can occur even without a discard
#         if hasattr(self, "_maybe_closed_kongs"):
#             self._maybe_closed_kongs(seat, policies)
#         if hasattr(self, "_maybe_added_kongs"):
#             self._maybe_added_kongs(seat, policies)

#     except Exception as e:
#         import traceback
#         print(f"[patch-step_turn-error] seat={seat} turn={self.turn} tile={tile}")
#         print(f"[patch-step_turn-error] exception={e}")
#         traceback.print_exc()

#     # Always fall back to normal step logic
#     return Env._orig_step_turn(self, policies)

# # Apply patch
# if not hasattr(Env, "_orig_step_turn"):
#     Env._orig_step_turn = Env.step_turn
# Env.step_turn = patched_step_turn
# #print("[patch] Env.step_turn patched safely (tile guard added)")




# ---------------------------- Repro ----------------------------
def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

# ---------------------------- Action masking helpers ----------------------------
TILE_ORDER = (
    [f"{r}b" for r in range(1,10)] +
    [f"{r}w" for r in range(1,10)] +
    [f"{r}t" for r in range(1,10)] +
    ["E","S","W","N","C","F","B"]   # Winds + Dragons
)
TILE_TO_IDX = {t: i for i, t in enumerate(TILE_ORDER)}  # 0..33

def apply_action_mask(logits: torch.Tensor, valid_indices: List[int]):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    C = logits.size(-1)
    if not valid_indices:
        return logits
    idx = torch.as_tensor(valid_indices, device=logits.device, dtype=torch.long)
    mask = torch.full((C,), float("-inf"), device=logits.device, dtype=logits.dtype)
    mask[idx] = 0.0
    return logits + mask.unsqueeze(0)

def _extract_valid_indices(s, C: int) -> List[int]:
    for name in ("legal_idx", "valid_idx", "legal_indices", "valid_indices"):
        if hasattr(s, name):
            li = getattr(s, name)
            if isinstance(li, (list, tuple)) and len(li) > 0:
                return [int(i) for i in li if 0 <= int(i) < C]
    for name in ("mask", "legal_mask", "action_mask"):
        if hasattr(s, name):
            m = getattr(s, name)
            try:
                arr = np.asarray(m, dtype=float).reshape(-1)
                if arr.shape[0] == C:
                    return [int(i) for i, v in enumerate(arr) if v > 0.5]
            except Exception:
                pass
    return list(range(C))

def _choice_index_for_head(s, C: int):
    ch = getattr(s, "choice", None)
    if ch is None:
        return None
    if isinstance(ch, str) and C == 34:
        return TILE_TO_IDX.get(ch, None)
    try:
        idx = int(ch)
    except Exception:
        return None
    return idx if 0 <= idx < C else None

# ---------------------------- Safe wall access helpers ----------------------------
def _wall_as_list(wall):
    if isinstance(wall, list):
        return list(wall)
    for attr in ("tiles", "_tiles", "data"):
        if hasattr(wall, attr):
            try:
                return list(getattr(wall, attr))
            except Exception:
                pass
    try:
        return list(wall)
    except Exception:
        return []

def _wall_set_from_list(wall, tiles):
    tiles = list(tiles)
    if isinstance(wall, list):
        wall[:] = tiles
        return
    for setter in ("set_tiles", "reset", "from_list"):
        if hasattr(wall, setter) and callable(getattr(wall, setter)):
            try:
                getattr(wall, setter)(tiles)
                return
            except Exception:
                pass
    if hasattr(wall, "clear") and hasattr(wall, "extend"):
        try:
            wall.clear()
            wall.extend(tiles)
            return
        except Exception:
            pass
    if hasattr(wall, "tiles"):
        try:
            setattr(wall, "tiles", tiles)
            return
        except Exception:
            pass
    try:
        while True:
            if len(wall) <= 0:
                break
            wall.pop()
    except Exception:
        pass
    try:
        for t in tiles:
            wall.append(t)
    except Exception:
        pass

# ---------------------------- Oracle plumbing ----------------------------
def _try_force_action(sim: Env, seat: int, action_idx: int) -> bool:
    for name in ("force_action", "apply_forced_action", "apply_action", "step_with_forced_action"):
        if hasattr(sim, name):
            fn = getattr(sim, name)
            try:
                if fn.__code__.co_argcount >= 3:
                    fn(seat, action_idx)
                else:
                    fn(action_idx)
                return True
            except Exception:
                return False
    return False

def _has_force(sim: Env) -> bool:
    return any(hasattr(sim, n) for n in (
        "force_action", "apply_forced_action", "apply_action", "step_with_forced_action"
    ))

@contextmanager
def _oracle_temporarily_disabled(rl):
    had = hasattr(rl, "oracle_picker")
    old = getattr(rl, "oracle_picker", None) if had else None
    try:
        if had:
            rl.oracle_picker = None
        yield
    finally:
        if had:
            rl.oracle_picker = old

class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

# ---------------------------- Peek mask / re-deal ----------------------------
def _make_peek_mask(env: Env, seat0_eval: int, p: float) -> Dict:
    mask = {"p": float(p), "opps": {}, "wall": []}
    for s in range(4):
        if s == seat0_eval:
            continue
        n = len(env.players[s].concealed)
        mask["opps"][s] = [random.random() < p for _ in range(n)]
    wall_len = len(_wall_as_list(env.wall))
    mask["wall"] = [random.random() < p for _ in range(wall_len)]
    return mask

def _apply_peek_belief_with_mask(sim: Env, seat0_eval: int, mask: Dict):
    p = float(mask.get("p", 0.0))

    for s in range(4):
        if s == seat0_eval:
            continue
        concealed = sim.players[s].concealed
        n_cur = len(concealed)
        m = mask["opps"].get(s, [])
        if len(m) < n_cur:
            m += [random.random() < p for _ in range(n_cur - len(m))]
            mask["opps"][s] = m
        elif len(m) > n_cur:
            mask["opps"][s] = m[:n_cur]

    wall_list = _wall_as_list(sim.wall)
    W = len(wall_list)
    m_wall = mask.get("wall", [])
    if len(m_wall) < W:
        m_wall += [random.random() < p for _ in range(W - len(m_wall))]
        mask["wall"] = m_wall
    elif len(m_wall) > W:
        mask["wall"] = m_wall[:W]

    unknown_slots = []
    wall_unknown_idx = []
    for s in range(4):
        if s == seat0_eval:
            continue
        for i, peek in enumerate(mask["opps"][s]):
            if not peek:
                unknown_slots.append(("opps", s, i))
    for i, peek in enumerate(mask["wall"]):
        if not peek:
            wall_unknown_idx.append(i)

    pool = []
    for _, s, i in unknown_slots:
        pool.append(sim.players[s].concealed[i])
    for i in wall_unknown_idx:
        if 0 <= i < len(wall_list):
            pool.append(wall_list[i])

    if not pool:
        return

    random.shuffle(pool)

    k = 0
    for _, s, i in unknown_slots:
        sim.players[s].concealed[i] = pool[k]
        k += 1

    for idx in wall_unknown_idx:
        if 0 <= idx < len(wall_list) and k < len(pool):
            wall_list[idx] = pool[k]
            k += 1

    _wall_set_from_list(sim.wall, wall_list)

# ---------------------------- Teacher rollouts ----------------------------
def _peeked_rollout_once(env: Env,
                         seat: int,
                         action_idx: int,
                         lineup_after_pick,
                         rules: dict,
                         horizon: int,
                         rl_guard,
                         peek_mask: Optional[Dict],
                         seat0_eval: int = 0) -> Optional[float]:
    """
    Single rollout using env.snapshot()/restore() when available, otherwise deepcopy.
    """
    try:
        snap = env.snapshot()
        sim = env
        sim.restore(snap)
    except Exception:
        sim = deepcopy(env)

    if not _try_force_action(sim, seat, action_idx):
        return None
    if peek_mask is not None:
        _apply_peek_belief_with_mask(sim, seat0_eval=seat0_eval, mask=peek_mask)

    steps = 0
    ctx = _oracle_temporarily_disabled(rl_guard) if rl_guard is not None else nullcontext()
    with ctx:
        while sim.wall and not sim.terminal and steps < horizon:
            sim.step_turn(lineup_after_pick)
            steps += 1

    term = sim.terminal or {"source": "drawn_game"}
    return compute_rl_reward(term, seat=seat0_eval, rules=rules)

def pick_oracle_action(env: Env,
                       seat: int,
                       candidate_indices: List[int],
                       lineup_after_pick,
                       rules: dict,
                       rollouts_per_action: int,
                       rollout_horizon: int,
                       rl_guard,
                       peek_mask: Optional[Dict],
                       seat0_eval: int = 0) -> Optional[int]:
    """
    Evaluate each candidate action by performing K rollouts per action.
    Uses env.snapshot()/env.restore() instead of deepcopy for speed when possible.
    """
    if not candidate_indices:
        return None

    try:
        base_snap = env.snapshot()
    except Exception as e:
        base_snap = None
        sys.stderr.write(f"[oracle] WARNING: snapshot failed ({e}), falling back to deepcopy\n")

    best = None

    for a in candidate_indices:
        vals: List[float] = []

        for _ in range(rollouts_per_action):
            if base_snap is not None:
                env.restore(base_snap)
                sim = env
            else:
                sim = deepcopy(env)

            if not _try_force_action(sim, seat, a):
                continue

            if peek_mask is not None:
                _apply_peek_belief_with_mask(sim, seat0_eval=seat0_eval, mask=peek_mask)

            steps = 0
            ctx = _oracle_temporarily_disabled(rl_guard) if rl_guard is not None else nullcontext()
            with ctx:
                while sim.wall and not sim.terminal and steps < rollout_horizon:
                    sim.step_turn(lineup_after_pick)
                    steps += 1

            term = sim.terminal or {"source": "drawn_game"}
            v = compute_rl_reward(term, seat=seat0_eval, rules=rules)
            if v is not None:
                vals.append(v)

        if vals:
            m = float(np.mean(vals))
            if best is None or m > best[0]:
                best = (m, a)

    return None if best is None else best[1]

def _attach_oracle(rl, picker: Callable):
    attached_paths = []
    for name in ("enable_oracle", "set_oracle", "set_teacher"):
        if hasattr(rl, name) and callable(getattr(rl, name)):
            try:
                getattr(rl, name)(picker)
                attached_paths.append(name)
            except Exception:
                pass
    for name in ("oracle_picker", "teacher_picker", "oracle", "teacher", "advice_fn", "picker"):
        try:
            setattr(rl, name, picker)
            attached_paths.append(f"attr:{name}")
        except Exception:
            pass
    for flag in ("use_oracle", "use_teacher", "oracle_enabled", "teacher_enabled"):
        try:
            setattr(rl, flag, True)
            attached_paths.append(f"flag:{flag}=True")
        except Exception:
            pass
    #print("[attach_oracle]", ", ".join(attached_paths) or "no attach points")
    # if attached_paths:
    #     sys.stderr.write("[attach_oracle] " + ", ".join(attached_paths) + "\n")
    # else:
    #     sys.stderr.write("[attach_oracle] no known attach points on RLPolicy\n")

# ---------------------------- A2C forward (with imitation loss hooks) ----------------------------
def a2c_forward(buffers: List[List], model: LSTMActorCritic, device: str):
    cfg = model.cfg
    model.train()
    logprobs, entropies, used_idx, vpred_list = [], [], [], []
    bc_terms = []
    bc_indices = []   # NEW: track which step each BC term is from
    global_pred_list = []
    step_offset = 0

    for buf in buffers:
        if not buf:
            continue
        obs = torch.from_numpy(np.stack([s.obs for s in buf], axis=0)).float().to(device)

        hx = (torch.zeros(1, 1, cfg.lstm, device=device),
              torch.zeros(1, 1, cfg.lstm, device=device))
        y, _ = model(obs[None, :, :], hx)
        heads = model.step_logits_value(y.squeeze(0))  # includes "global_reward" if model supports it

        # Value predictions
        v = heads["value"].squeeze(-1) if "value" in heads else torch.zeros(len(buf), device=device)
        vpred_list.append(v)

        # Global reward predictions (auxiliary)
        g_head = heads.get("global_reward", None)
        if g_head is not None:
            g_vec = g_head.squeeze(-1)
            if g_vec.dim() == 0:
                g_vec = g_vec.view(1)
        else:
            g_vec = torch.zeros(len(buf), device=device)
        global_pred_list.append(g_vec)

        for t, s in enumerate(buf):
            kind = getattr(s, "kind", "discard")
            head = {
                "discard": "discard",
                "ron":     "binary",
                "pung":    "pung",
                "binary":  "binary",
                "chow":    "chow",
                "kong":    "kong"
            }.get(kind, "discard")

            if head not in heads or heads[head].ndim == 0:
                continue

            logits = heads[head][t:t+1, :]
            C = int(logits.size(-1))
            legal_idx = _extract_valid_indices(s, C) or list(range(C))
            masked = apply_action_mask(logits, legal_idx)

            # --- RL policy term (log π(a_t | s_t)) ---
            act_idx = _choice_index_for_head(s, C)
            if act_idx is not None and act_idx in legal_idx:
                dist = torch.distributions.Categorical(logits=masked)
                logprobs.append(
                    dist.log_prob(torch.tensor([act_idx], device=device)).squeeze(0)
                )
                entropies.append(dist.entropy().mean())
                used_idx.append(step_offset + t)

            # --- Teacher / BC term ---
            t_idx = getattr(s, "teacher_idx", None)
            if t_idx is None:
                tch = getattr(s, "teacher_choice", None)
                if tch is not None:
                    if isinstance(tch, str) and C == 34:
                        t_idx = TILE_TO_IDX.get(tch, None)
                    else:
                        try:
                            t_idx = int(tch)
                        except Exception:
                            t_idx = None

            if (
                t_idx is not None
                and 0 <= int(t_idx) < C
                and (not legal_idx or int(t_idx) in legal_idx)
            ):
                logp_at_teacher = torch.log_softmax(masked, dim=-1)[0, int(t_idx)]
                bc_terms.append(-logp_at_teacher)
                bc_indices.append(step_offset + t)   # NEW: record which step this is

        step_offset += len(buf)

    v_pred = torch.cat(vpred_list) if vpred_list else torch.empty(0, device=device)
    g_pred = torch.cat(global_pred_list) if global_pred_list else torch.empty(0, device=device)

    logprobs_t  = torch.stack(logprobs) if logprobs else torch.tensor([0.0], device=device)
    entropies_t = torch.stack(entropies) if entropies else torch.tensor([0.0], device=device)
    used_idx_t  = torch.tensor(used_idx, dtype=torch.long, device=device) if used_idx else torch.empty(0, dtype=torch.long, device=device)

    bc_terms_t = torch.stack(bc_terms) if bc_terms else torch.empty(0, device=device)
    bc_idx_t   = torch.tensor(bc_indices, dtype=torch.long, device=device) if bc_indices else torch.empty(0, dtype=torch.long, device=device)

    return logprobs_t, entropies_t, v_pred, used_idx_t, bc_terms_t, bc_idx_t, g_pred


# ---------------------------- GAE(λ) ----------------------------
def compute_returns_and_advantages(
    buffers: List[List],
    rewards_final: List[float],
    vals_concat: torch.Tensor,
    gamma: float, lam: float,
    shaping_coef: float,
    device: str
):
    rews_all: List[float] = []
    for buf, R in zip(buffers, rewards_final):
        T = len(buf)
        for j, s in enumerate(buf):
            r_shape = 0.0
            if shaping_coef > 0.0:
                if hasattr(s, "shape_delta"):
                    r_shape = float(getattr(s, "shape_delta"))
                elif hasattr(s, "comp_improve"):
                    r_shape = float(getattr(s, "comp_improve"))
            r_final = R if j == T - 1 else 0.0
            rews_all.append(shaping_coef * r_shape + r_final)

    r = torch.tensor(rews_all, dtype=torch.float32, device=device)
    v = vals_concat
    assert r.shape[0] == v.shape[0], f"reward/values length mismatch: {r.shape} vs {v.shape}"

    adv = torch.zeros_like(v)
    ret = torch.zeros_like(v)
    idx = 0
    for buf in buffers:
        T = len(buf)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            rt = r[idx + t]
            vt = v[idx + t]
            delta = rt + gamma * (next_value if t != T - 1 else 0.0) - vt
            gae = float(delta) + gamma * lam * gae
            adv[idx + t] = gae
            ret[idx + t] = adv[idx + t] + vt
            next_value = vt
        idx += T

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return ret, adv

# ---------------------------- Utilities ----------------------------
def _randomize_start_player(env) -> int:
    k = random.randint(0, 3)
    for attr in ("dealer", "turn", "current_player", "start_seat"):
        if hasattr(env, attr):
            try:
                setattr(env, attr, k)
            except Exception:
                pass
    return k

def _resample_others_and_wall_keep_seat0(sim: Env):
    opp_slots = []
    opp_tiles = []
    for s in range(1, 4):
        c = sim.players[s].concealed
        for i in range(len(c)):
            opp_slots.append((s, i))
            opp_tiles.append(c[i])

    wall_tiles = _wall_as_list(sim.wall)

    pool = opp_tiles + wall_tiles
    random.shuffle(pool)

    j = 0
    for (s, i) in opp_slots:
        pass  # kept in case of future logic changes
    for (s, i) in opp_slots:
        sim.players[s].concealed[i] = pool[j]
        j += 1

    new_wall = pool[j:]
    _wall_set_from_list(sim.wall, new_wall)

# ---------------------------- Checkpoint helpers ----------------------------
def _now_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _sha_short(d: Dict[str, Any]) -> str:
    try:
        s = json.dumps(d, sort_keys=True).encode("utf-8")
        return hashlib.sha1(s).hexdigest()[:8]
    except Exception:
        return "na"

def save_ckpt(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

import warnings  # put this near the top of the file with the other imports

def load_ckpt(path: str) -> Dict[str, Any]:
    try:
        # PyTorch 2.6+ supports weights_only
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch (no weights_only argument)
        return torch.load(path, map_location="cpu")


@contextmanager
def _temp_args(args, **overrides):
    backup = {}
    for k, v in overrides.items():
        backup[k] = getattr(args, k)
        setattr(args, k, v)
    try:
        yield
    finally:
        for k, v in backup.items():
            setattr(args, k, v)

def _attach_tenpai_flags_if_draw(env: Env) -> None:
    """
    If env.terminal is a drawn game, compute per-seat tenpai flags and
    attach them to the terminal so compute_rl_reward can use them.
    """
    term = env.terminal
    if term is None:
        return

    # Extract `source` / `reason` robustly from dict or object
    if isinstance(term, dict):
        src = term.get("source") or term.get("reason")
    else:
        src = getattr(term, "source", None) or getattr(term, "reason", None)

    if src != "drawn_game":
        return

    try:
        flags = compute_tenpai_flags(env)  # returns List[bool] length 4
    except Exception:
        return

    if isinstance(term, dict):
        # don't overwrite if something upstream already set it
        term.setdefault("tenpai_flags", flags)
    else:
        try:
            if not hasattr(term, "tenpai_flags"):
                setattr(term, "tenpai_flags", flags)
        except Exception:
            pass


# DEBUG_LIMIT = 50
# debug_counter = {"n": 0}
win_counts = [0, 0, 0, 0]
hand_count = 0
# ---------------------------- Episode runner (top-level) ----------------------------
def run_episode_core(
    args,
    rules: dict,
    model: LSTMActorCritic,
    device: str,
    lineup_tags: List[str],
    seed: int,
    selfplay: bool,
    peek_mask_for_episode: Optional[Dict],
    compute_scale: float,
    behavior_prob_use: float,
    env: Optional[Env]=None
) -> Tuple[List, float, Optional[int]]:
    if env is None:
        env = Env(rules, seed=seed)
    # if getattr(args, "randomize_dealer", True):
    #     _randomize_start_player(env)

    from algorithm.mahjongrl.agent import RLPolicy  # avoid circular at import time
    rl = RLPolicy(seat=0, rules=rules, model=model, device=device)

    teacher_bot = FlexibleAggroPolicyD(seat=0, rules=rules, tuner=None)

    advice_q: List[Tuple[str, Optional[int]]] = []  # (head, idx)

    start_seat = None    # may be None if env doesn't expose it
    for attr in ("turn", "dealer", "current_player", "start_seat"):
        if hasattr(env, attr):
            try:
                start_seat = int(getattr(env, attr))
                break
            except Exception:
                pass

    if selfplay:
        lineup = [
            rl,
            RLPolicy(1, rules, model, device),
            RLPolicy(2, rules, model, device),
            RLPolicy(3, rules, model, device),
        ]
    else:
        lineup = make_lineup_with_rl(rl, rules, lineup_tags)

        from collections import Counter  # already imported at top, but safe here

        def _choose_yes_no(legal_idx: List[int], yes: bool) -> int:
            """Map yes/no to an actual index in legal_idx."""
            if not legal_idx:
                return 0
            if not yes:
                return legal_idx[0]      # usually 0 = pass/no
            nonzero = [i for i in legal_idx if i != 0]
            return nonzero[0] if nonzero else legal_idx[0]

        def teacher_picker(
            env_now: Env,
            seat: int,
            legal_idx: List[int],
            head: str = "discard",
        ) -> Optional[int]:
            """
            Teacher for seat 0 using FlexibleAggroPolicyD on all heads.

            - For each head, we compute a *label* index (label_idx) in the local action
            space, and record it into advice_q for behavior cloning.
            - Separately, we decide whether to actually EXECUTE the teacher suggestion;
            for claim heads, we ONLY execute when the teacher wants to CLAIM
            (label_idx > 0), never when it wants to pass.
            """
            # Only teach the RL agent, and only if there is at least one legal action.
            if seat != 0 or not legal_idx:
                advice_q.append((head, None))
                return None

            label_idx: Optional[int] = None  # teacher's choice in legal_idx-space

            try:
                # ---------- DISCARD (multi-class over tiles) ----------
                if head == "discard":
                    tile = teacher_bot.pick_discard(env_now)
                    raw_idx = TILE_TO_IDX.get(tile)

                    if raw_idx is not None and raw_idx in legal_idx:
                        label_idx = raw_idx
                    else:
                        label_idx = None  # suggested tile not legal for this head

                    # Optional debug:
                    # print(f"[teacher-discard] tile={tile} raw_idx={raw_idx} legal={legal_idx}")

                # ---------- PUNG (binary yes/no) ----------
                elif head == "pung":
                    last = getattr(env_now, "last_discard", None)
                    if last is not None:
                        discarder, tile = last
                        yes = teacher_bot.decide_pung(env_now, seat, tile)
                    else:
                        yes = False

                    # We assume 0 = pass, others = claim choices.
                    if yes:
                        claim_indices = [i for i in legal_idx if i != 0]
                        label_idx = claim_indices[0] if claim_indices else legal_idx[0]
                    else:
                        label_idx = 0 if 0 in legal_idx else legal_idx[0]

                # ---------- CHOW (multi-way: pass + 1..k chow options) ----------
                elif head == "chow":
                    last = getattr(env_now, "last_discard", None)
                    if not last:
                        # No actual chow opportunity; no label.
                        advice_q.append((head, None))
                        return None

                    discarder, tile = last
                    reacting_seat = (discarder + 1) % 4  # your rules: only next seat can chow
                    if seat != reacting_seat:
                        # Env shouldn't be asking us, but be safe.
                        advice_q.append((head, None))
                        return None

                    if hasattr(env_now, "debug_player_state"):
                        env_now.debug_player_state(seat, tag="teacher-chow")

                    # Build chow_sets in the SAME order as env._chow_claim:
                    p = env_now.players[seat]
                    hand = [t for t in p.concealed if not is_flower(_tile_code(t))]
                    chow_sets: List[Tuple[str, str]] = []

                    if tile is not None and _is_suit_tile(tile):
                        r, suit = _tile_rank_suit(tile)
                        # (r-2, r-1), (r-1, r+1), (r+1, r+2) – same triple you saw in the logs
                        for a, b in [(r - 2, r - 1), (r - 1, r + 1), (r + 1, r + 2)]:
                            if 1 <= a <= 9 and 1 <= b <= 9:
                                A, B = f"{a}{suit}", f"{b}{suit}"
                                if A in hand and B in hand:
                                    chow_sets.append((A, B))

                    if not chow_sets:
                        # There is a chow head but we don't truly have a sequence – label = pass.
                        label_idx = 0 if 0 in legal_idx else legal_idx[0]
                    else:
                        chosen = teacher_bot.choose_chow(env_now, seat, tile, chow_sets)
                        if chosen is None:
                            # Teacher wants to pass.
                            label_idx = 0 if 0 in legal_idx else legal_idx[0]
                        else:
                            # Teacher picked a specific set, e.g. ('2w','3w') vs ('5w','6w').
                            try:
                                j = chow_sets.index(chosen)  # 0-based in chow_sets
                                # In env, index j+1 corresponds to chow_sets[j].
                                candidate = j + 1
                                # Ensure candidate is in legal_idx (it should be [0..k]).
                                label_idx = candidate if candidate in legal_idx else None
                            except ValueError:
                                # chosen not actually in chow_sets – treat as no advice
                                label_idx = None

                    # Optional:
                    # print(f"[teacher-chow] tile={tile} chow_sets={chow_sets} chosen={chosen} "
                    #       f"label_idx={label_idx} legal={legal_idx}")

                # ---------- KONG (binary yes/no combined) ----------
                elif head == "kong":
                    last = getattr(env_now, "last_discard", None)
                    tile_from_discard = last[1] if last is not None else None

                    open_yes = teacher_bot.decide_open_kong(env_now, seat, tile_from_discard)
                    # We don't need the exact closed candidate here for labeling; just any_kong.
                    closed_pick = teacher_bot.decide_closed_kong(env_now, seat, [])
                    add_yes = teacher_bot.decide_add_kong(env_now, seat, tile_from_discard)

                    any_kong = open_yes or bool(closed_pick) or add_yes

                    if any_kong:
                        claim_indices = [i for i in legal_idx if i != 0]
                        label_idx = claim_indices[0] if claim_indices else legal_idx[0]
                    else:
                        label_idx = 0 if 0 in legal_idx else legal_idx[0]

                # ---------- BINARY (RON yes/no) ----------
                elif head == "binary":
                    tile = getattr(env_now, "last_discard_tile", None)
                    yes = teacher_bot.decide_ron(
                        env_now,
                        tile,
                        getattr(env_now, "points", None),
                        getattr(env_now, "ron_loser", None),
                    )

                    if yes:
                        claim_indices = [i for i in legal_idx if i != 0]
                        label_idx = claim_indices[0] if claim_indices else legal_idx[0]
                    else:
                        label_idx = 0 if 0 in legal_idx else legal_idx[0]

                # Other heads -> no advice
                else:
                    label_idx = None

                # Final safety: teacher label must be in legal space or it's ignored.
                if label_idx is not None and label_idx not in legal_idx:
                    label_idx = None

            except Exception:
                import traceback
                traceback.print_exc()
                label_idx = None

            # 1) Record LABEL for behavior cloning (even if we don't execute it).
            advice_q.append((head, label_idx))

            # 2) Decide whether to EXECUTE teacher instead of π.
            eff_p = max(0.0, min(1.0, behavior_prob_use * args.oracle_exec_prob))

            if label_idx is None:
                exec_now = False
            elif head in ("pung", "chow", "kong", "binary"):
                # For claim heads, do NOT force teacher when it wants to PASS (label_idx == 0).
                exec_now = (label_idx != 0 and random.random() < eff_p)
            else:
                # Discards etc. – you can let teacher override more freely, or keep this probabilistic:
                exec_now = (random.random() < eff_p)

            # If we don't execute teacher, return None so RL uses π's action.
            return label_idx if exec_now else None


        # def teacher_picker(
        #     env_now: Env,
        #     seat: int,
        #     legal_idx: List[int],
        #     head: str = "discard",
        # ) -> Optional[int]:
        #     idx: Optional[int] = None

        #     # Only ever teach our RL agent
        #     if seat != 0 or not legal_idx:
        #         advice_q.append((head, None))
        #         return None

        #     try:
        #         # ---------- DISCARD (DAgger) ----------
        #         if head == "discard":
        #             tile = teacher_bot.pick_discard(env_now)
        #             raw_idx = TILE_TO_IDX.get(tile)
        #             if raw_idx is not None and raw_idx in legal_idx:
        #                 idx = raw_idx
        #             else:
        #                 idx = None

        #         # ---------- PUNG (yes / no) ----------
        #         elif head == "pung":
        #             last = getattr(env_now, "last_discard", None)
        #             tile = last[1] if last else None
        #             yes = bool(tile) and teacher_bot.decide_pung(env_now, seat, tile)
        #             idx = _choose_yes_no(legal_idx, yes)

        #         # ---------- CHOW (yes / no, ignore which set) ----------
        #         elif head == "chow":
        #             last = getattr(env_now, "last_discard", None)
        #             if not last:
        #                 idx = legal_idx[0]
        #             else:
        #                 discarder, tile = last
        #                 reacting_seat = (discarder + 1) % 4
        #                 if reacting_seat != seat or tile is None:
        #                     idx = legal_idx[0]
        #                 else:
        #                     chow_sets = []
        #                     getter = getattr(env_now, "legal_chow_sets", None)
        #                     if callable(getter):
        #                         try:
        #                             chow_sets = getter(seat)
        #                         except Exception:
        #                             chow_sets = []

        #                     yes = False
        #                     if chow_sets:
        #                         try:
        #                             chosen = teacher_bot.choose_chow(env_now, seat, tile, chow_sets)
        #                             yes = chosen is not None
        #                         except Exception:
        #                             yes = False

        #                     idx = _choose_yes_no(legal_idx, yes)

        #         # ---------- KONG (any kong vs no kong) ----------
        #         elif head == "kong":
        #             seat_state = env_now.players[seat]
        #             cnt = Counter(getattr(seat_state, "concealed", []))
        #             closed_candidates = [t for t, c in cnt.items() if c >= 4]

        #             add_candidates = [
        #                 m.tiles[0]
        #                 for m in getattr(seat_state, "melds", [])
        #                 if (getattr(m, "type", getattr(m, "kind", "")).lower() in ("pung", "pong"))
        #                 and all(t == m.tiles[0] for t in m.tiles)
        #                 and seat_state.concealed.count(m.tiles[0]) >= 1
        #             ]

        #             last = getattr(env_now, "last_discard", None)
        #             open_tile = last[1] if last else None

        #             wants = False
        #             try:
        #                 if closed_candidates and hasattr(teacher_bot, "decide_closed_kong"):
        #                     wants = wants or bool(
        #                         teacher_bot.decide_closed_kong(env_now, seat, closed_candidates)
        #                     )
        #                 if add_candidates and hasattr(teacher_bot, "decide_add_kong"):
        #                     wants = wants or bool(
        #                         teacher_bot.decide_add_kong(env_now, seat, add_candidates)
        #                     )
        #                 if open_tile and hasattr(teacher_bot, "decide_open_kong"):
        #                     wants = wants or bool(
        #                         teacher_bot.decide_open_kong(env_now, seat, open_tile)
        #                     )
        #             except Exception:
        #                 wants = False

        #             idx = _choose_yes_no(legal_idx, wants)

        #         # ---------- RON / BINARY (yes / no) ----------
        #         elif head == "binary":
        #             last = getattr(env_now, "last_discard", None)
        #             tile = last[1] if last else None
        #             try:
        #                 yes = teacher_bot.decide_ron(
        #                     env_now,
        #                     tile,
        #                     getattr(env_now, "points", None),
        #                     getattr(env_now, "ron_loser", None),
        #                 )
        #             except Exception:
        #                 yes = False
        #             idx = _choose_yes_no(legal_idx, yes)

        #     except Exception:
        #         idx = None

        #     # Log teacher advice (for BC later)
        #     advice_q.append((head, idx))

        #     # DAgger schedule for discards, but **always** follow teacher on claims
        #     base_p = max(0.0, min(1.0, behavior_prob_use * args.oracle_exec_prob))
        #     if head in ("pung", "chow", "kong", "binary"):
        #         exec_p = 1.0    # always execute teacher on claim heads
        #     else:
        #         exec_p = base_p

        #     return idx if (idx is not None and random.random() < exec_p) else None

        _attach_oracle(rl, teacher_picker)
        setattr(rl, "teacher_picker", teacher_picker)




    # Simulate episode
        # Simulate episode
    draws = 0
    try:
        while env.wall and not env.terminal and draws < args.max_draws:
            #if hasattr(env, "legal_idx"):
                #print("[debug-legal] generic legal_idx:", getattr(env, "legal_idx"))
            for name in ["legal_pung_indices", "legal_chow_indices", "legal_kong_indices"]:
                if hasattr(env, name):
                    try:
                        v = getattr(env, name)
                        val = v(0) if callable(v) else v
                        #print(f"[debug-legal] {name} ->", val)
                    except Exception as e:
                        pass
                        #print(f"[debug-legal] {name} failed:", e)           
            env.step_turn(lineup)
            draws += 1
    except Exception as e:
        msg = str(e)
        if args.skip_bad_episodes and (
            "list.remove" in msg
            or "list index out of range" in msg
            or isinstance(e, IndexError)
        ):
            sys.stderr.write(f"[warn] skipping corrupted episode (IndexError): {e}\n")
            # Return empty buffer so caller just ignores this episode
            return [], 0.0, start_seat
        else:
            traceback.print_exc()
            raise

    _attach_tenpai_flags_if_draw(env)

    # Attach advice by head
    if rl.buffer:
        head_map = {
            "discard": "discard",
            "ron":     "binary",
            "pung":    "pung",
            "binary":  "binary",
            "chow":    "chow",
            "kong":    "kong",
        }

        q_by_head: Dict[str, deque] = {h: deque() for h in head_map.values()}
        for h, idx in advice_q:
            if h in q_by_head:
                q_by_head[h].append(idx)

        attach_cnt = collections.Counter()
        total_nonnull = sum(1 for _, v in advice_q if v is not None)

        for step in rl.buffer:
            step_head = head_map.get(getattr(step, "kind", "discard"))
            if step_head not in q_by_head:
                continue
            if not q_by_head[step_head]:
                continue
            idx = q_by_head[step_head].popleft()
            if idx is not None:
                try:
                    setattr(step, "teacher_idx", int(idx))
                    attach_cnt[step_head] += 1
                except Exception:
                    pass

        if total_nonnull == 0:
            # Fallback #1: offline masked-argmax (needs step.obs)
            offline_cnt = collections.Counter()
            for step in rl.buffer:
                kind = getattr(step, "kind", "discard")
                head = head_map.get(kind, None)
                if head is None or not hasattr(step, "obs"):
                    continue
                try:
                    obs_np = np.asarray(step.obs, dtype=np.float32)
                    obs = torch.from_numpy(obs_np).float().to(device)[None, None, :]
                    with torch.no_grad():
                        y, _ = model(obs)
                        heads = model.step_logits_value(y.squeeze(0))
                        logits = heads.get(head, None)
                        if logits is None:
                            continue
                        C = int(logits.size(-1))
                        legal_idx = _extract_valid_indices(step, C) or list(range(C))
                        masked = apply_action_mask(logits[0:1, :], legal_idx)
                        idx = int(torch.argmax(masked, dim=-1).item())
                        setattr(step, "teacher_idx", idx)
                        offline_cnt[head] += 1
                except Exception:
                    pass

            if sum(offline_cnt.values()) == 0:
                # Fallback #2: executed-action imitation (uses step.choice only)
                exec_cnt = collections.Counter()
                for step in rl.buffer:
                    kind = getattr(step, "kind", "discard")
                    head = head_map.get(kind, None)
                    if head is None:
                        continue
                    if hasattr(step, "choice"):
                        try:
                            setattr(step, "teacher_choice", getattr(step, "choice"))
                            exec_cnt[head] += 1
                        except Exception:
                            pass
    #print("[debug-buffer]", collections.Counter(([getattr(step, "kind", None) for step in rl.buffer])))

    reward = compute_rl_reward(env.terminal or {"source":"drawn_game"}, seat=0, rules=rules)
    #print("[src-counts]", getattr(rl, "_src_counts", {}))
    #_record_win_stats(env)
    return rl.buffer, reward, start_seat

def schedule_coef(epoch, total_epochs, final_value, start_frac=0.1):
    """
    Linear ramp-up schedule:
    - epoch < total_epochs * start_frac → 0
    - epoch >= total_epochs → final_value
    """
    if epoch < total_epochs * start_frac:
        return 0.0
    progress = (epoch - total_epochs * start_frac) / (total_epochs * (1 - start_frac))
    progress = max(0.0, min(1.0, progress))
    return final_value * progress

# ---------------------------- Multiprocessing worker context ----------------------------
_WORKER_MODEL: Optional[LSTMActorCritic] = None
_WORKER_RULES: Optional[dict] = None
_WORKER_ARGS = None
_WORKER_DEVICE: str = "cpu"

def _mp_init_worker(model_state: Dict[str, Any],
                    cfg_dict: Dict[str, Any],
                    rules: Dict[str, Any],
                    args_dict: Dict[str, Any]):
    """
    Initializer for each worker process.
    Builds a copy of the model and stores rules/args globals.
    Env stays on CPU; model uses GPU if available and not args.cpu.
    """
    global _WORKER_MODEL, _WORKER_RULES, _WORKER_ARGS, _WORKER_DEVICE
    use_cuda = torch.cuda.is_available() and (not args_dict.get("cpu", False))
    _WORKER_DEVICE = "cuda" if use_cuda else "cpu"
    cfg = ACConfig(**cfg_dict)
    m = LSTMActorCritic(cfg).to(_WORKER_DEVICE)
    m.load_state_dict({k: v.to(_WORKER_DEVICE) for k, v in model_state.items()})
    _WORKER_MODEL = m
    _WORKER_RULES = rules
    _WORKER_ARGS = types.SimpleNamespace(**args_dict)

def _mp_run_episode(job):
    """
    job = (seed, epoch_idx, tile_p, beh_p, vs_bots_flag, lineup_tags)
    Returns (buffer, reward, coverage).
    """
    global _WORKER_MODEL, _WORKER_RULES, _WORKER_ARGS, _WORKER_DEVICE
    seed, epoch_idx, tile_p, beh_p, vs_bots_flag, lineup_tags = job

    setattr(_WORKER_ARGS, "epoch", epoch_idx)

    env0 = Env(_WORKER_RULES, seed=seed)
    peek_mask = _make_peek_mask(env0, seat0_eval=0, p=tile_p)
    total_slots = sum(len(v) for v in peek_mask["opps"].values()) + len(peek_mask["wall"])
    if total_slots > 0:
        cov = (sum(sum(1 for b in v if b) for v in peek_mask["opps"].values())
               + sum(1 for b in peek_mask["wall"] if b))
        coverage = cov / total_slots
    else:
        coverage = 0.0

    from algorithm.mahjongrl.agent import RLPolicy

    if vs_bots_flag:
        buf, rew, _ = run_episode_core(
            _WORKER_ARGS, _WORKER_RULES, _WORKER_MODEL, _WORKER_DEVICE,
            lineup_tags=lineup_tags, seed=seed, selfplay=False,
            peek_mask_for_episode=peek_mask, compute_scale=tile_p,
            behavior_prob_use=beh_p,env=env0
        )
    else:
        buf, rew, _ = run_episode_core(
            _WORKER_ARGS, _WORKER_RULES, _WORKER_MODEL, _WORKER_DEVICE,
            lineup_tags=[], seed=seed, selfplay=True,
            peek_mask_for_episode=peek_mask, compute_scale=tile_p,
            behavior_prob_use=beh_p,env=env0
        )

    return buf, float(rew), float(coverage)

# ---------------------------- Opponent curriculum ----------------------------
def sample_lineup_tags(epoch_idx: int, args, vs_bots_epochs: int) -> List[str]:
    """
    Sample 3 opponent tags for seats 1,2,3 based on curriculum phase.
    We avoid random/wp/payout because they don't complete hands often enough.

    Phases (fraction of vs_bots_epochs):
      - [0.0, 0.3):   only aggro-ish bots (aggro, hyaggro)
      - [0.3, 0.7):   mix in flexaggro
      - [0.7, 1.0]:   mostly strong policies (hyaggro, flexaggro, flexaggrod)

    If vs_bots_epochs <= 0, fall back to args.lineup.
    """
    if vs_bots_epochs <= 0:
        # Fallback: use whatever was specified on the CLI
        return args.lineup.split(",")

    phase = min(1.0, max(0.0, epoch_idx / float(vs_bots_epochs)))

    pool = ["aggro", "hyaggro", "flexaggro", "flexaggrod"]

    # Sample 3 opponents i.i.d. from pool
    return [random.choice(pool) for _ in range(3)]


# ---------------------------- Training ----------------------------
def train(args):
    # Directories / run metadata
    run_id = args.run_id or _now_run_id()
    run_dir = Path(args.outdir) / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last.pt"
    meta = {"argv": " ".join(os.sys.argv), "args": vars(args)}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[run] dir={run_dir}")
    vsbots_epochs_history: List[int] = []
    vsbots_avg_reward_history: List[float] = []
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[device] using {device}")

    rules = load_rules(args.rules)
    set_global_seeds(args.seed)

    dummy = Env(rules, seed=123)
    print("[env-claim-helpers]", 
          [m for m in dir(dummy) 
           if ("chow" in m or "pung" in m or "kong" in m)])
    obs0 = build_observation(dummy, seat=0)
    print(f"[init] detected obs_dim = {obs0.shape[0]}")
    cfg = ACConfig(obs_dim=obs0.shape[0], hidden=args.hidden, lstm=args.lstm)
    model = LSTMActorCritic(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    print(f"[model] params = {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    # ---- Resume (explicit or auto from last.pt) ----
    start_epoch = 0
    ckpt_path: Optional[str] = None

    # 1) Explicit resume has priority
    if args.resume:
        ckpt_path = args.resume
    else:
        # 2) Auto-resume: if last.pt exists for this run_id, load it
        auto_ckpt = ckpt_dir / "last.pt"
        if auto_ckpt.exists():
            ckpt_path = str(auto_ckpt)

    if ckpt_path is not None:
        print(f"[ckpt] loading {ckpt_path}")
        state = load_ckpt(ckpt_path)

        msd = state.get("model_state") or state.get("model")
        if msd:
            try:
                model.load_state_dict(msd)
            except Exception as e:
                print(f"[ckpt] model load warning: {e}")

        osd = state.get("opt_state") or state.get("optimizer")
        if osd:
            try:
                opt.load_state_dict(osd)
            except Exception as e:
                print(f"[ckpt] optimizer load warning: {e}")

        # RNG
        if state.get("py_rng_state"):
            random.setstate(state["py_rng_state"])
        if state.get("np_rng_state"):
            np.random.set_state(state["np_rng_state"])  # type: ignore
        if state.get("torch_rng_state") is not None:
            try:
                torch.random.set_rng_state(state["torch_rng_state"])  # type: ignore
            except Exception:
                pass

        start_epoch = int(state.get("epoch", -1)) + 1
        print(f"[ckpt] resume at epoch {start_epoch}")
    else:
        print("[ckpt] no checkpoint found; starting from scratch")

    # ---- Eval-only shortcut ----
    if args.eval_only:
        assert args.resume, "--eval-only requires --resume"
        lineup = args.eval_lineup.split(",") if args.eval_lineup else args.lineup.split(",")
        episodes = max(1, int(args.eval_episodes))
        avg_r = 0.0
        with _temp_args(args,
                        oracle_rollouts=0,
                        oracle_exec_prob=0.0,
                        peek_prob0=0.0,
                        peek_prob_target=0.0):
            for _ in range(episodes):
                seed = random.randint(1, 10**9)
                env0 = Env(rules, seed=seed)
                peek_mask = _make_peek_mask(env0, seat0_eval=0, p=0.0)
                buf, rew, _ = run_episode_core(
                    args, rules, model, device,
                    lineup_tags=lineup, seed=seed, selfplay=False,
                    peek_mask_for_episode=peek_mask, compute_scale=0.0,
                    behavior_prob_use=0.0,
                )
                avg_r += float(rew)
        avg_r /= episodes
        print(f"[eval] lineup={','.join(lineup)} episodes={episodes} avg_reward={avg_r:.3f}")
        return

    # ---- Schedules ----
    def schedule_val(a0, aT, epoch_idx: int, kind: str, warmup: int, total_epochs: int) -> float:
        if epoch_idx < warmup:
            return a0
        k = epoch_idx - warmup
        E = max(1, total_epochs - warmup)
        if kind == "exp":
            d = (aT / max(1e-12, a0)) ** (1.0 / max(1, E - 1))
            return max(aT, a0 * (d ** k))
        t = k / max(1, E - 1)
        return max(aT, a0 + (aT - a0) * t)

    def peek_probability(epoch_idx: int) -> float:
        return schedule_val(args.peek_prob0, args.peek_prob_target, epoch_idx,
                            args.peek_prob_schedule, args.peek_prob_warmup, args.epochs)

    def behavior_prob(epoch_idx: int) -> float:
        return schedule_val(args.oracle_behavior_prob, args.oracle_behavior_target,
                            epoch_idx, args.oracle_behavior_schedule, args.oracle_behavior_schedule_warmup, args.epochs)

    def bc_weight(epoch_idx: int) -> float:
        """
        BC weight schedule with two phases:
          - [0, bc_pretrain_epochs): pure BC; keep weight = bc_weight_start
          - [bc_pretrain_epochs, ...): RL+BC with annealed weight
        """
        if epoch_idx < args.bc_pretrain_epochs:
            return args.bc_weight_start

        # Shift schedule so annealing starts *after* pretrain
        eff_epoch = epoch_idx - args.bc_pretrain_epochs
        total_after = max(1, args.epochs - args.bc_pretrain_epochs)

        return schedule_val(
            args.bc_weight_start,
            args.bc_weight_final,
            eff_epoch,
            args.bc_weight_schedule,
            args.bc_weight_warmup,
            total_after,
        )


    def entropy_weight(epoch_idx: int) -> float:
        hi, lo = max(args.entropy_coef, 0.03), args.entropy_coef
        T = max(1, args.vs_bots_epochs)
        t = min(1.0, epoch_idx / T)
        return lo + (hi - lo) * (1.0 - t)

    # ---- Checkpoint saver ----
    def _maybe_save(epoch: int, extra: Optional[Dict[str, Any]] = None):
        if (epoch % max(1, args.save_every)) != 0:
            return
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "args": vars(args),
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "py_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "timestamp": time.time(),
        }
        if extra:
            payload.update(extra)
        tag = f"epoch{epoch:04d}"
        save_ckpt(str((ckpt_dir / f"{tag}.pt").resolve()), payload)
        save_ckpt(str((ckpt_dir / "last.pt").resolve()), payload)
        print(f"[ckpt] saved {tag}")

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        args.epoch = epoch
        bc_only = (epoch < args.bc_pretrain_epochs)
        ent_w = entropy_weight(epoch)
        tile_p = float(peek_probability(epoch))
        beh_p = float(behavior_prob(epoch))
        bc_w  = float(bc_weight(epoch))

        buffers: List[List] = []
        rewards_final: List[float] = []
        coverages: List[float] = []

        entropy_coef = schedule_coef(epoch, args.epochs, final_value=0.04, start_frac=0.01)
        value_coef   = schedule_coef(epoch, args.epochs, final_value=0.5, start_frac=0.5)
        shaping_coef = schedule_coef(epoch, args.epochs, final_value=0.05,  start_frac=0.2)

        # overwrite args or pass into loss directly
        args.entropy_coef = entropy_coef
        args.value_coef   = value_coef
        args.shaping_coef = shaping_coef

        # Standard episodes (possibly parallelized)
        if args.num_workers > 0:
            ctx = mp.get_context("spawn")
            model_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            cfg_dict = {
                "obs_dim": cfg.obs_dim,
                "hidden": cfg.hidden,
                "lstm": cfg.lstm,
            }
            args_dict = dict(vars(args))
            vs_bots_flag = (epoch < args.vs_bots_epochs)

            jobs = []
            for _ in range(args.episodes_per_epoch):
                seed = random.randint(1, 10**9)
                if vs_bots_flag:
                    # Curriculum-based opponent sampling
                    lineup_tags = sample_lineup_tags(epoch, args, args.vs_bots_epochs)
                else:
                    # Selfplay: lineup_tags ignored (run_episode_core uses selfplay=True)
                    lineup_tags = []
                jobs.append((seed, epoch, tile_p, beh_p, vs_bots_flag, lineup_tags))


            with ctx.Pool(
                processes=args.num_workers,
                initializer=_mp_init_worker,
                initargs=(model_state_cpu, cfg_dict, rules, args_dict),
            ) as pool:
                results = pool.map(_mp_run_episode, jobs)

            for buf, rew, cov in results:
                # if rew is not None:
                #     print(f"[debug-reward] epoch={epoch} seed={seed} rew={rew:.3f}")
                if buf:
                    buffers.append(buf)
                    rewards_final.append(rew)
                    coverages.append(cov)
        else:
            for _ in range(args.episodes_per_epoch):
                seed = random.randint(1, 10**9)

                env0 = Env(rules, seed=seed)
                
                peek_mask = _make_peek_mask(env0, seat0_eval=0, p=tile_p)
                total_slots = sum(len(v) for v in peek_mask["opps"].values()) + len(peek_mask["wall"])
                cov = (sum(sum(1 for b in v if b) for v in peek_mask["opps"].values())
                       + sum(1 for b in peek_mask["wall"] if b))
                coverages.append((cov / max(1, total_slots)) if total_slots else 0.0)

                if epoch < args.vs_bots_epochs:
                    # Curriculum-based bots (aggro → hyaggro → flexaggro → flexaggrod)
                    lineup_tags = sample_lineup_tags(epoch, args, args.vs_bots_epochs)
                    buf, rew, _ = run_episode_core(
                        args, rules, model, device,
                        lineup_tags=lineup_tags, seed=seed, selfplay=False,
                        peek_mask_for_episode=peek_mask,
                        compute_scale=tile_p,
                        behavior_prob_use=beh_p,
                    )
                else:
                    # Self-play phase
                    buf, rew, _ = run_episode_core(
                        args, rules, model, device,
                        lineup_tags=[], seed=seed, selfplay=True,
                        peek_mask_for_episode=peek_mask,
                        compute_scale=tile_p,
                        behavior_prob_use=beh_p,
                    )
                if buf:
                    buffers.append(buf)
                    rewards_final.append(rew)


        # Anchored repeats
        if args.anchored_batches > 0 and args.anchored_K > 0:
            from algorithm.mahjongrl.agent import RLPolicy
            for _ in range(args.anchored_batches):
                base_env = Env(rules, seed=random.randint(1, 10**9))
                #_ = _randomize_start_player(base_env)
                for _k in range(args.anchored_K):
                    sim = deepcopy(base_env)
                    _resample_others_and_wall_keep_seat0(sim)

                    rl0 = RLPolicy(0, rules, model, device)
                    lineup = [rl0, RLPolicy(1, rules, model, device),
                                   RLPolicy(2, rules, model, device),
                                   RLPolicy(3, rules, model, device)]
                    _attach_oracle(rl0, lambda *a, **kw: None)

                    draws = 0
                    try:
                        while sim.wall and not sim.terminal and draws < args.max_draws:
                            sim.step_turn(lineup)
                            draws += 1
                        # sim.terminal now set or wall empty
                    except Exception:
                        break

                    if rl0.buffer:
                        buffers.append(rl0.buffer)
                        reward = compute_rl_reward(sim.terminal or {"source":"drawn_game"}, seat=0, rules=rules)
                        rewards_final.append(reward)

                # Skip update if nothing collected
                # Skip update if nothing collected
        if not buffers:
            print(f"[epoch {epoch+1}/{args.epochs}] no steps collected; skipping update")
            _maybe_save(epoch)
            continue

        # ---- Optimization over collected batch (possibly multiple passes) ----
                # ---- Optimization over collected batch (multi-pass, episode mini-batches) ----
        last_losses = {}
        num_eps = len(buffers)
        ep_indices = list(range(num_eps))
        episodes_per_batch = max(1, getattr(args, "minibatch_episodes", 16))

        for opt_iter in range(max(1, getattr(args, "opt_iters_per_epoch", 1))):
            random.shuffle(ep_indices)

            for start_idx in range(0, num_eps, episodes_per_batch):
                mb_ids = ep_indices[start_idx:start_idx + episodes_per_batch]
                mb_buffers = [buffers[i] for i in mb_ids]
                mb_rewards = [rewards_final[i] for i in mb_ids]

                # Forward pass on this mini-batch of episodes
                logprobs, ent_terms, vals, used_idx, bc_terms, bc_idx, g_pred = a2c_forward(
                    mb_buffers, model, device
                )

                # One-time BC sanity print
                if epoch == 0 and opt_iter == 0 and start_idx == 0:
                    cnt = int(bc_terms.numel()) if torch.is_tensor(bc_terms) else 0
                    mean_bc = (bc_terms.mean().item() if torch.is_tensor(bc_terms) and cnt > 0 else float('nan'))
                    print(f"[bc-sanity] bc_terms_count={cnt} mean={mean_bc:.4f}")

                # Returns + advantages for this mini-batch
                ret, adv = compute_returns_and_advantages(
                    buffers=mb_buffers,
                    rewards_final=mb_rewards,
                    vals_concat=vals,
                    gamma=args.gamma,
                    lam=args.gae_lambda,
                    shaping_coef=args.shaping_coef,
                    device=device,
                )

                if bc_only:
                    # -------- PURE BEHAVIOR CLONING PHASE --------
                    if torch.is_tensor(bc_terms) and bc_terms.numel() > 0:
                        bc_loss = bc_w * bc_terms.mean()
                    else:
                        bc_loss = torch.tensor(0.0, device=device)

                    pol_loss    = torch.tensor(0.0, device=device)
                    ent_loss    = torch.tensor(0.0, device=device)
                    val_loss    = torch.tensor(0.0, device=device)
                    global_loss = torch.tensor(0.0, device=device)

                else:
                    # -------- NORMAL RL + ADVANTAGE-GATED BC --------
                    if used_idx.numel() > 0:
                        pol_loss = -(logprobs * adv[used_idx]).mean()
                        ent_loss = - ent_w * ent_terms.mean()
                    else:
                        pol_loss = torch.tensor(0.0, device=device)
                        ent_loss = torch.tensor(0.0, device=device)

                    val_loss = args.value_coef * 0.5 * (ret - vals).pow(2).mean()

                    # Advantage-gated BC (only imitate when advantage not too negative)
                    if (
                        torch.is_tensor(bc_terms)
                        and bc_terms.numel() > 0
                        and bc_idx.numel() > 0
                    ):
                        adv_for_bc = adv[bc_idx]
                        good_mask = adv_for_bc > -0.5
                        if good_mask.any():
                            bc_loss = bc_w * bc_terms[good_mask].mean()
                        else:
                            bc_loss = torch.tensor(0.0, device=device)
                    else:
                        bc_loss = torch.tensor(0.0, device=device)

                    # Global reward prediction loss (auxiliary)
                    if g_pred.numel() > 0 and args.global_reward_coef > 0:
                        global_targets_list: List[float] = []
                        for buf, R in zip(mb_buffers, mb_rewards):
                            T = len(buf)
                            global_targets_list.extend([float(R)] * T)

                        if len(global_targets_list) == int(g_pred.shape[0]):
                            global_targets = torch.tensor(
                                global_targets_list,
                                dtype=torch.float32,
                                device=device,
                            )
                            global_loss = args.global_reward_coef * 0.5 * (g_pred - global_targets).pow(2).mean()
                        else:
                            global_loss = torch.tensor(0.0, device=device)
                    else:
                        global_loss = torch.tensor(0.0, device=device)

                loss = pol_loss + val_loss + ent_loss + bc_loss + global_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                # Track the last mini-batch losses for logging
                last_losses = {
                    "loss": loss.item(),
                    "pol": pol_loss.item(),
                    "val": val_loss.item(),
                    "ent": (-ent_loss).item(),
                    "bc": bc_loss.item(),
                    "glob": global_loss.item(),
                }

        # last_losses = {}

        # for opt_iter in range(max(1, getattr(args, "opt_iters_per_epoch", 1))):
        #     # Forward pass on the current model over all collected steps
        #     logprobs, ent_terms, vals, used_idx, bc_terms, bc_idx, g_pred = a2c_forward(buffers, model, device)

        #     # One-time BC sanity print
        #     if epoch == 0 and opt_iter == 0:
        #         cnt = int(bc_terms.numel()) if torch.is_tensor(bc_terms) else 0
        #         mean_bc = (bc_terms.mean().item() if torch.is_tensor(bc_terms) and cnt > 0 else float('nan'))
        #         print(f"[bc-sanity] bc_terms_count={cnt} mean={mean_bc:.4f}")

        #     # Returns + advantages for this forward pass
        #     ret, adv = compute_returns_and_advantages(
        #         buffers=buffers,
        #         rewards_final=rewards_final,
        #         vals_concat=vals,
        #         gamma=args.gamma,
        #         lam=args.gae_lambda,
        #         shaping_coef=args.shaping_coef,
        #         device=device,
        #     )

        #     # Policy / entropy / value losses (same as before)
        #     if used_idx.numel() > 0:
        #         pol_loss = -(logprobs * adv[used_idx]).mean()
        #         ent_loss = - ent_w * ent_terms.mean()
        #     else:
        #         pol_loss = torch.tensor(0.0, device=device)
        #         ent_loss = torch.tensor(0.0, device=device)

        #     val_loss = args.value_coef * 0.5 * (ret - vals).pow(2).mean()

        #     # Advantage-gated BC: only imitate when advantage not too negative
        #     if (
        #         torch.is_tensor(bc_terms)
        #         and bc_terms.numel() > 0
        #         and bc_idx.numel() > 0
        #     ):
        #         adv_for_bc = adv[bc_idx]
        #         good_mask = adv_for_bc > -0.5
        #         if good_mask.any():
        #             bc_loss = bc_w * bc_terms[good_mask].mean()
        #         else:
        #             bc_loss = torch.tensor(0.0, device=device)
        #     else:
        #         bc_loss = torch.tensor(0.0, device=device)

        #     # Global reward prediction loss (auxiliary)
        #     global_targets_list: List[float] = []
        #     for buf, R in zip(buffers, rewards_final):
        #         T = len(buf)
        #         global_targets_list.extend([float(R)] * T)

        #     if g_pred.numel() > 0 and len(global_targets_list) == int(g_pred.shape[0]):
        #         global_targets = torch.tensor(global_targets_list, dtype=torch.float32, device=device)
        #         global_loss = args.global_reward_coef * 0.5 * (g_pred - global_targets).pow(2).mean()
        #     else:
        #         global_loss = torch.tensor(0.0, device=device)

        #     loss = pol_loss + val_loss + ent_loss + bc_loss + global_loss

        #     opt.zero_grad()
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #     opt.step()

        #     # Keep track of the last pass's losses for logging
        #     last_losses = {
        #         "loss": loss.item(),
        #         "pol": pol_loss.item(),
        #         "val": val_loss.item(),
        #         "ent": (-ent_loss).item(),
        #         "bc": bc_loss.item(),
        #         "glob": global_loss.item(),
        #     }

        # ---- Logging for this epoch (using last optimization pass) ----
        avg_r = float(np.mean(rewards_final)) if rewards_final else 0.0
        cov_rate = float(np.mean(coverages)) if coverages else 0.0

        if epoch < args.vs_bots_epochs:
            vsbots_epochs_history.append(epoch + 1)        # 1-based
            vsbots_avg_reward_history.append(avg_r)

        print(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"opt_iters={getattr(args, 'opt_iters_per_epoch', 1)} "
            f"eps={len(rewards_final)} avg_reward={avg_r:.2f} "
            f"loss={last_losses['loss']:.3f} pol={last_losses['pol']:.3f} "
            f"val={last_losses['val']:.3f} ent={last_losses['ent']:.3f} "
            f"bc={last_losses['bc']:.3f} glob={last_losses['glob']:.3f} "
            f"tile_p={tile_p:.3f} peek_cov={cov_rate:.2f} "
            f"beh_p={beh_p:.2f} bc_w={bc_w:.3f}"
        )

        _maybe_save(epoch)



        # # Skip update if nothing collected
        # if not buffers:
        #     print(f"[epoch {epoch+1}/{args.epochs}] no steps collected; skipping update")
        #     _maybe_save(epoch)
        #     continue

        # logprobs, ent_terms, vals, used_idx, bc_terms, bc_idx, g_pred = a2c_forward(buffers, model, device)

        # if epoch == 0:
        #     cnt = int(bc_terms.numel()) if torch.is_tensor(bc_terms) else 0
        #     mean_bc = (bc_terms.mean().item() if torch.is_tensor(bc_terms) and cnt > 0 else float('nan'))
        #     print(f"[bc-sanity] bc_terms_count={cnt} mean={mean_bc:.4f}")

        # if bc_only:
        #     # -------- PURE BEHAVIOR CLONING PHASE --------
        #     if torch.is_tensor(bc_terms) and bc_terms.numel() > 0:
        #         bc_loss = bc_w * bc_terms.mean()
        #     else:
        #         bc_loss = torch.tensor(0.0, device=device)

        #     pol_loss   = torch.tensor(0.0, device=device)
        #     ent_loss   = torch.tensor(0.0, device=device)
        #     val_loss   = torch.tensor(0.0, device=device)
        #     global_loss = torch.tensor(0.0, device=device)

        # else:
        #     # -------- NORMAL RL + ADVANTAGE-GATED BC --------
        #     ret, adv = compute_returns_and_advantages(
        #         buffers=buffers,
        #         rewards_final=rewards_final,
        #         vals_concat=vals,
        #         gamma=args.gamma,
        #         lam=args.gae_lambda,
        #         shaping_coef=args.shaping_coef,
        #         device=device,
        #     )

        #     # Policy / entropy / value
        #     if used_idx.numel() > 0:
        #         pol_loss = -(logprobs * adv[used_idx]).mean()
        #         ent_loss = - ent_w * ent_terms.mean()
        #     else:
        #         pol_loss = torch.tensor(0.0, device=device)
        #         ent_loss = torch.tensor(0.0, device=device)

        #     val_loss = args.value_coef * 0.5 * (ret - vals).pow(2).mean()

        #     # Advantage-gated BC
        #     if (
        #         torch.is_tensor(bc_terms)
        #         and bc_terms.numel() > 0
        #         and bc_idx.numel() > 0
        #     ):
        #         adv_for_bc = adv[bc_idx]
        #         good_mask = adv_for_bc > -0.5  # gate out obviously-bad teacher actions

        #         if good_mask.any():
        #             bc_loss = bc_w * bc_terms[good_mask].mean()
        #         else:
        #             bc_loss = torch.tensor(0.0, device=device)
        #     else:
        #         bc_loss = torch.tensor(0.0, device=device)

        #     # Global reward prediction loss (auxiliary)
        #     global_targets_list: List[float] = []
        #     for buf, R in zip(buffers, rewards_final):
        #         T = len(buf)
        #         global_targets_list.extend([float(R)] * T)

        #     if g_pred.numel() > 0 and len(global_targets_list) == int(g_pred.shape[0]):
        #         global_targets = torch.tensor(global_targets_list, dtype=torch.float32, device=device)
        #         global_loss = args.global_reward_coef * 0.5 * (g_pred - global_targets).pow(2).mean()
        #     else:
        #         global_loss = torch.tensor(0.0, device=device)

        # # Shared: backprop this total loss
        # loss = pol_loss + val_loss + ent_loss + bc_loss + global_loss


        # opt.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # opt.step()

        # avg_r = float(np.mean(rewards_final)) if rewards_final else 0.0
        # cov_rate = float(np.mean(coverages)) if coverages else 0.0

        # if epoch < args.vs_bots_epochs:
        #     vsbots_epochs_history.append(epoch + 1)        # 1-based epochs for plotting
        #     vsbots_avg_reward_history.append(avg_r)

        # print(f"[epoch {epoch+1}/{args.epochs}] eps={len(rewards_final)} "
        #       f"avg_reward={avg_r:.2f} loss={loss.item():.3f} pol={pol_loss.item():.3f} "
        #       f"val={val_loss.item():.3f} ent={(-ent_loss).item():.3f} bc={bc_loss.item():.3f} "
        #       f"glob={global_loss.item():.3f} "
        #       f"tile_p={tile_p:.3f} peek_cov={cov_rate:.2f} beh_p={beh_p:.2f} bc_w={bc_w:.3f}")

        # _maybe_save(epoch)
        # --- After training: save avg reward curve over vs-bots epochs ---
    if vsbots_avg_reward_history:
        try:
            plt.figure()
            plt.plot(vsbots_epochs_history, vsbots_avg_reward_history, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Average reward (seat 0)")
            plt.title("Vs-bots average reward over epochs")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = run_dir / "avg_reward_vs_bots.png"
            plt.savefig(out_path)
            plt.close()
            print(f"[plot] saved vs-bots avg reward curve to {out_path}")
        except Exception as e:
            print(f"[plot] WARNING: failed to save avg reward plot: {e}")


# ---------------------------- CLI ----------------------------
from algorithm.sim_and_train import sanity_check_tile_encoding
if __name__ == "__main__":
    sanity_check_tile_encoding()
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--lineup", default="aggro,aggro,hyaggro")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--episodes-per-epoch", type=int, default=384)
    ap.add_argument("--vs-bots-epochs", type=int, default=250)
    ap.add_argument("--max-draws", type=int, default=250)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lstm", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.015)
    ap.add_argument("--value-coef", type=float, default=0.25)
    ap.add_argument("--shaping-coef", type=float, default=0.1)
    ap.add_argument("--global-reward-coef", type=float, default=0,
                    help="Weight for global reward prediction auxiliary loss (0 = disable).")
    ap.add_argument("--seed", type=int, default=123456)

    # Teacher
    ap.add_argument("--oracle-rollouts", type=int, default=0)
    ap.add_argument("--oracle-horizon", type=int, default=48)
    ap.add_argument("--oracle-topN", type=int, default=16)
    # Oracle gating
    og = ap.add_mutually_exclusive_group()
    og.add_argument("--oracle-only-vsbots", dest="oracle_only_vsbots", action="store_true",
                    help="Use oracle only during vs-bots epochs.")
    og.add_argument("--no-oracle-only-vsbots", dest="oracle_only_vsbots", action="store_false",
                    help="Allow oracle in all epochs.")
    ap.set_defaults(oracle_only_vsbots=False)

    ap.add_argument("--oracle-exec-prob", type=float, default=1.0)

    # Peek prob schedule
    ap.add_argument("--peek-prob0", type=float, default=0.0)
    ap.add_argument("--peek-prob-target", type=float, default=0.0)
    ap.add_argument("--peek-prob-warmup", type=int, default=5)
    ap.add_argument("--peek-prob-schedule", choices=["exp","linear"], default="exp")

    # DAgger μ schedule
    ap.add_argument("--oracle-behavior-prob", type=float, default=1.0)
    ap.add_argument("--oracle-behavior-target", type=float, default=0.07)
    ap.add_argument("--oracle-behavior-schedule", choices=["exp","linear"], default="linear")
    ap.add_argument("--oracle-behavior-schedule-warmup", type=int, default=20)

    # BC weight schedule
    ap.add_argument("--bc-weight-start", type=float, default=0.3)
    ap.add_argument("--bc-weight-final", type=float, default=0.03)
    ap.add_argument("--bc-weight-warmup", type=int, default=20)
    ap.add_argument("--bc-weight-schedule", choices=["exp","linear"], default="linear")
    ap.add_argument(
        "--bc-pretrain-epochs",
        type=int,
        default=0,
        help="Number of initial epochs to run pure behavior cloning (no policy/value loss).",
    )

    # Anchored repeats
    ap.add_argument("--anchored-batches", type=int, default=0)
    ap.add_argument("--anchored-K", type=int, default=0)

    # New: persistence / eval controls
    ap.add_argument("--outdir", default="runs", help="Root folder for checkpoints/logs")
    ap.add_argument("--run-id", default=None, help="Run name (subfolder under --outdir). Default: timestamp")
    ap.add_argument("--save-every", type=int, default=10, help="Save a checkpoint every N epochs")
    ap.add_argument("--resume", default=None,
    help="Optional path to checkpoint (.pt) to resume from. "
         "If omitted, train.py automatically resumes from runs/<run_id>/checkpoints/last.pt if present.")
    ap.add_argument("--eval-only", action="store_true", help="Load --resume and run evaluation only")
    ap.add_argument("--eval-lineup", default="flexaggro,hyaggro,aggro", help="Opponent lineup for eval")
    ap.add_argument("--eval-episodes", type=int, default=64, help="Evaluation episodes")

    # Number of CPU env workers (0 = no parallelism).
    ap.add_argument("--num-workers", type=int, default=8,
                    help="Number of CPU workers for parallel env rollouts (0 = no parallelism)")

    ap.add_argument("--randomize-dealer", action="store_true", default=False)
    ap.add_argument("--skip-bad-episodes", action="store_true", default=True)

    ap.add_argument(
        "--opt-iters-per-epoch",
        type=int,
        default=2,
        help="How many optimization passes to run over each epoch's batch (e.g. 3–5).",
    )
    ap.add_argument(
        "--minibatch-episodes",
        type=int,
        default=50,
        help="Number of episodes per optimization mini-batch.",
    )


    # Force training on CPU (overrides GPU even if available)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    train(args)
