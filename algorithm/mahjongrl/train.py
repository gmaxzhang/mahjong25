# algorithm/mahjongrl/train.py
from __future__ import annotations
import argparse, random, sys, traceback, os, time, json, pickle, hashlib
from typing import List, Optional, Callable, Dict, Tuple, Any
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
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
from algorithm.mahjongrl.env_wrapper import make_lineup_with_rl, build_observation, compute_rl_reward, compute_tenpai_flags
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


def patched_step_turn(self, policies):
    if getattr(self, "terminal", False):
        return
    seat = self.turn

    # last_discard is either None or (discarder, tile_str)
    last = getattr(self, "last_discard", None)

    try:
        # Only check ron/pung/chow if there’s an actual discard tile
        if last is not None:
            discarder, tile = last  # tile is now the string, e.g. "7w"

            if hasattr(self, "_ron_window"):
                self._ron_window(discarder, tile, policies)
            if hasattr(self, "_pung_claims"):
                self._pung_claims(discarder, tile, policies)
            if hasattr(self, "_chow_claim"):
                self._chow_claim(discarder, tile, policies)

        # Kongs can occur even without a discard
        if hasattr(self, "_maybe_closed_kongs"):
            self._maybe_closed_kongs(seat, policies)
        if hasattr(self, "_maybe_added_kongs"):
            self._maybe_added_kongs(seat, policies)

    except Exception as e:
        import traceback
        print(f"[patch-step_turn-error] seat={seat} turn={self.turn} last_discard={last}")
        print(f"[patch-step_turn-error] exception={e}")
        traceback.print_exc()

    # Always fall back to normal step logic
    return Env._orig_step_turn(self, policies)

# Apply patch
if not hasattr(Env, "_orig_step_turn"):
    Env._orig_step_turn = Env.step_turn
Env.step_turn = patched_step_turn


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
    global_pred_list = []
    step_offset = 0

    for buf in buffers:
        if not buf:
            continue
        obs = torch.from_numpy(np.stack([s.obs for s in buf], axis=0)).float().to(device)

        hx = (torch.zeros(1,1,cfg.lstm, device=device),
              torch.zeros(1,1,cfg.lstm, device=device))
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
            head = {"discard":"discard","ron":"binary","pung":"binary","binary":"binary",
                    "chow":"chow","kong":"kong"}.get(kind, "discard")
            if head not in heads or heads[head].ndim == 0:
                continue

            logits = heads[head][t:t+1, :]
            C = int(logits.size(-1))
            legal_idx = _extract_valid_indices(s, C) or list(range(C))
            masked = apply_action_mask(logits, legal_idx)

            # policy term
            act_idx = _choice_index_for_head(s, C)
            if act_idx is not None and act_idx in legal_idx:
                dist = torch.distributions.Categorical(logits=masked)
                logprobs.append(dist.log_prob(torch.tensor([act_idx], device=device)).squeeze(0))
                entropies.append(dist.entropy().mean())
                used_idx.append(step_offset + t)

            # teacher term
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

            if t_idx is not None and 0 <= int(t_idx) < C and (not legal_idx or int(t_idx) in legal_idx):
                logp_at_teacher = torch.log_softmax(masked, dim=-1)[0, int(t_idx)]
                bc_terms.append(-logp_at_teacher)

        step_offset += len(buf)

    v_pred = torch.cat(vpred_list) if vpred_list else torch.empty(0, device=device)
    g_pred = torch.cat(global_pred_list) if global_pred_list else torch.empty(0, device=device)
    logprobs_t = torch.stack(logprobs) if logprobs else torch.tensor([0.0], device=device)
    entropies_t = torch.stack(entropies) if entropies else torch.tensor([0.0], device=device)
    used_idx_t = torch.tensor(used_idx, dtype=torch.long, device=device) if used_idx else torch.empty(0, dtype=torch.long, device=device)
    bc_terms_t = torch.stack(bc_terms) if bc_terms else torch.empty(0, device=device)
    return logprobs_t, entropies_t, v_pred, used_idx_t, bc_terms_t, g_pred

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
) -> Tuple[List, float, Optional[int]]:
    env = Env(rules, seed=seed)
    if getattr(args, "randomize_dealer", True):
        _randomize_start_player(env)

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
        def teacher_picker(env_now: Env, seat: int, legal_idx: List[int], head: str = "discard") -> Optional[int]:
            """
            Teacher: imitate FlexibleAggroPolicyD on all heads.
            We translate its decisions into the env's action indices (legal_idx).
            """
            idx: Optional[int] = None

            # Only teach our RL agent (seat 0) and only if there is at least one legal action.
            if seat != 0 or not legal_idx:
                advice_q.append((head, None))
                return None

            try:
                # ---------- DISCARD ----------
                if head == "discard":
                    tile = teacher_bot.pick_discard(env_now)
                    raw_idx = TILE_TO_IDX.get(tile)
                    if raw_idx is not None and raw_idx in legal_idx:
                        idx = raw_idx
                    else:
                        idx = None  # don't force an illegal discard


                    # # Determine actual seat that _pung_claims() will check
                    # # (one of the 3 seats after the discarder)
                    # if last is not None:
                    #     discarder, _ = last
                    #     for k in (1, 2, 3):
                    #         s_claim = (discarder + k) % 4
                    #         if s_claim == seat:   # our agent is the claimant
                    #             env_now._forced[s_claim] = {"kind": "pung", "idx": idx}
                    #             print(f"[teacher-picker] pung | discarder={discarder} reacting_seat={s_claim} "
                    #                 f"tile={tile} decide_pung={yes} -> idx={idx} legal={legal_idx}")
                    #             break
                    #     else:
                    #         # not an eligible claimant
                    #         print(f"[teacher-picker] pung | tile={tile} seat={seat} not eligible claimant "
                    #             f"decide_pung={yes} skipped.")
                    # else:
                    #     print(f"[teacher-picker] pung | tile=None no last_discard available")


                # elif head == "pung":
                #     tile = getattr(env_now, "last_discard_tile", None)
                #     yes = teacher_bot.decide_pung(env_now, seat, tile)

                #     if yes:
                #         # Choose some non-zero index if available (treat 0 as "pass").
                #         claim_indices = [i for i in legal_idx if i != 0]
                #         idx = claim_indices[0] if claim_indices else legal_idx[0]
                #     else:
                #         # Explicitly pass.
                #         idx = legal_idx[0]
                #     print(f"[teacher-picker] decide_pung -> {yes}")

                
                # elif head == "chow":
                #     # In your env, last_discard = (seat, tile)
                #     tile = env_now.last_discard[1] if env_now.last_discard else None
                #     if tile is None or not _is_suit_tile(tile):
                #         idx = 0
                #     else:
                #         # Determine which chow sets are actually possible.
                #         discarder = env_now.last_discard[0]
                #         s = (discarder + 1) % 4
                #         p = env_now.players[s]

                #         r, suit = _tile_rank_suit(tile)
                #         chow_sets = []
                #         for a, b in [(r - 2, r - 1), (r - 1, r + 1), (r + 1, r + 2)]:
                #             if 1 <= a <= 9 and 1 <= b <= 9:
                #                 A, B = f"{a}{suit}", f"{b}{suit}"
                #                 if A in p.concealed and B in p.concealed:
                #                     chow_sets.append((A, B))

                #         if not chow_sets:
                #             idx = 0
                #         else:
                #             chosen = teacher_bot.choose_chow(env_now, s, tile, chow_sets)
                #             if chosen in chow_sets:
                #                 idx = chow_sets.index(chosen) + 1
                #             else:
                #                 idx = 0
                        

                #     print(f"[teacher-picker] chow | tile={tile} -> idx={idx}")


                # elif head == "chow":
                #     tile = getattr(env_now, "last_discard_tile", None)
                #     chow_sets = getattr(env_now, "legal_chow_sets", lambda s: [])(seat)
                #     chosen = teacher_bot.choose_chow(env_now, seat, tile, chow_sets)

                #     C = max(legal_idx) + 1 if legal_idx else 4

                #     if not chow_sets:
                #         # No chow available according to env; just pass.
                #         idx = legal_idx[0]

                #     else:
                #         # Try to get a parallel list of indices for the chow sets
                #         idx_list = None
                #         for m in ("legal_chow_indices", "get_legal_chow_indices", "chow_legal_idx", "legal_chows"):
                #             if hasattr(env_now, m):
                #                 v = getattr(env_now, m)
                #                 li = v(seat) if callable(v) else v
                #                 li = [int(x) for x in li]
                #                 idx_list = [x for x in li if x in legal_idx]
                #                 break

                #         if chosen is None:
                #             # Teacher wants to pass.
                #             if idx_list is not None:
                #                 # Prefer an index that is *not* one of the chow indices.
                #                 pass_candidates = [i for i in legal_idx if i not in idx_list]
                #                 idx = pass_candidates[0] if pass_candidates else legal_idx[0]
                #             else:
                #                 # Fallback: assume 0 is "no chow" if present.
                #                 idx = 0 if 0 in legal_idx else legal_idx[0]
                #         else:
                #             # Teacher chose a specific set; map it to the corresponding index.
                #             if idx_list is not None:
                #                 try:
                #                     j = chow_sets.index(chosen)
                #                     if 0 <= j < len(idx_list):
                #                         idx = idx_list[j]
                #                     else:
                #                         idx = None
                #                 except ValueError:
                #                     idx = None
                #             else:
                #                 # Fallback: assume layout [0=pass, 1..k = chow_sets in order].
                #                 try:
                #                     j = chow_sets.index(chosen)
                #                     candidate = 1 + j
                #                     idx = candidate if candidate in legal_idx else None
                #                 except ValueError:
                #                     idx = None

                # ---------- KONG (multi-way; open/add/closed) ----------
                    
                elif head == "kong":
                    seat_state = env_now.players[seat]
                    wants_kong = False

                    #print(f"[teacher-debug] entering kong | seat={seat}")

                    # --- Closed kongs (4 identical concealed) ---
                    # If you imported is_flower, you can use the filtered version.
                    # cnt = Counter([t for t in seat_state.concealed if not is_flower(t)])
                    # For now, keep it simple:
                    cnt = Counter(seat_state.concealed)
                    closed_candidates = [t for t, c in cnt.items() if c >= 4]

                    # --- Add-kong (upgrade an existing pung) ---
                    add_candidates = [
                        m.tiles[0]
                        for m in seat_state.melds
                        if getattr(m, "type", getattr(m, "kind", "")) == "pung"
                        and all(t == m.tiles[0] for t in m.tiles)
                        and seat_state.concealed.count(m.tiles[0]) >= 1
                    ]

                    # --- Open-kong (claim discard for a 4th identical) ---
                    open_candidates = []
                    last = getattr(env_now, "last_discard", None)
                    if last is not None:
                        discarder, tile = last
                        cnt2 = Counter(seat_state.concealed)
                        if cnt2[tile] >= 3:
                            open_candidates.append(tile)

                    # Query teacher bot
                    chosen_tile = None
                    if hasattr(teacher_bot, "decide_closed_kong"):
                        chosen_tile = teacher_bot.decide_closed_kong(env_now, seat, closed_candidates)
                        if chosen_tile:
                            wants_kong = True

                    if hasattr(teacher_bot, "decide_add_kong"):
                        add_ok = teacher_bot.decide_add_kong(env_now, seat, add_candidates)
                        wants_kong = wants_kong or add_ok

                    if hasattr(teacher_bot, "decide_open_kong"):
                        open_ok = teacher_bot.decide_open_kong(env_now, seat, open_candidates)
                        wants_kong = wants_kong or open_ok

                    idx = 1 if wants_kong else 0

                    # print(
                    #     f"[teacher-debug] kong | seat={seat} wants_kong={wants_kong} "
                    #     f"closed={closed_candidates} add={add_candidates} open={open_candidates} "
                    #     f"-> proposed idx={idx}, legal={legal_idx}"
                    # )

                    # Ensure idx is legal
                    if idx not in legal_idx:
                        legal_idx.append(idx)

                    # Force into env
                    if hasattr(env_now, "_forced"):
                        env_now._forced[seat] = {"kind": "kong", "idx": idx}
                        # print(
                        #     f"[teacher-picker] kong | seat={seat} -> forced idx={idx} "
                        #     f"(legal after patch={legal_idx})"
                        # )
                    else:
                        pass
                        #print("[teacher-picker-warning] env_now has no _forced dict!")


                # elif head == "kong":
                #     seat_state = env_now.players[seat]
                #     wants_kong = False

                #     # Candidates for closed kongs (4 identical concealed)
                #     cnt = Counter([t for t in seat_state.concealed if not is_flower(t)])
                #     closed_candidates = [t for t, c in cnt.items() if c >= 4]

                #     # Candidates for add-kong (upgrade pung)
                #     add_candidates = [
                #         m.tiles[0]
                #         for m in seat_state.melds
                #         if getattr(m, "type", getattr(m, "kind", "")) == "pung"
                #         and all(t == m.tiles[0] for t in m.tiles)
                #         and seat_state.concealed.count(m.tiles[0]) >= 1
                #     ]

                #     chosen_tile = None
                #     if hasattr(teacher_bot, "decide_closed_kong"):
                #         chosen_tile = teacher_bot.decide_closed_kong(env_now, seat, closed_candidates)
                #         if chosen_tile:
                #             wants_kong = True
                #     if hasattr(teacher_bot, "decide_add_kong"):
                #         add_ok = teacher_bot.decide_add_kong(env_now, seat, None)
                #         wants_kong = wants_kong or add_ok

                #     idx = 1 if wants_kong else 0
                #     print(f"[teacher-picker] kong | seat={seat} wants_kong={wants_kong} closed={closed_candidates} add={add_candidates} -> idx={idx}")

                # ---------- PUNG (binary yes/no, but env might encode as [0] or [0,1,...]) ----------
                elif head == "pung":
                    # Env stores (discarder, tile)
                    last = getattr(env_now, "last_discard", None)
                    if last is None:
                        tile = None
                        yes = False
                        discarder = None
                    else:
                        discarder, tile = last
                        # Determine if our agent (seat) is eligible to claim
                        # Claimant seats are (discarder + 1) % 4, (discarder + 2) % 4, (discarder + 3) % 4
                        claimable_seats = [(discarder + k) % 4 for k in (1, 2, 3)]
                        is_claimant = seat in claimable_seats
                        yes = teacher_bot.decide_pung(env_now, seat, tile) if (tile and is_claimant) else False

                    # Convention: 0 = no, 1 = yes
                    proposed_idx = 1 if yes else 0

                    # If env only exposes [0] but teacher wants to pung, extend with pseudo-1 for logging
                    if yes and legal_idx == [0]:
                        legal_idx.append(1)

                    # Choose final idx consistent with legal space
                    idx = proposed_idx if proposed_idx in legal_idx else max(legal_idx)
                    if hasattr(env_now, "_forced"):
                        env_now._forced[seat] = {"kind": "pung", "idx": idx}
                        #print(f"[teacher-picker] pung | seat={seat} tile={tile} -> forced idx={idx} in active env")
                    else:
                        pass
                        #print("[teacher-picker-warning] env_now has no _forced dict!")

                # ---------- CHOW (multi-way: pass + 1–3 chow options) ----------
                    
                elif head == "chow":
                    # Use the last discard info
                    last = getattr(env_now, "last_discard", None)
                    if not last:
                        return None

                    tile = last[1]
                    discarder = last[0]
                    reacting_seat = (discarder + 1) % 4  # only next seat can chow

                    if tile is None or not _is_suit_tile(tile):
                        idx = 0
                    else:
                        # Determine which chow sets are actually possible for reacting seat.
                        p = env_now.players[reacting_seat]
                        r, suit = _tile_rank_suit(tile)
                        chow_sets = []
                        for a, b in [(r - 2, r - 1), (r - 1, r + 1), (r + 1, r + 2)]:
                            if 1 <= a <= 9 and 1 <= b <= 9:
                                A, B = f"{a}{suit}", f"{b}{suit}"
                                if A in p.concealed and B in p.concealed:
                                    chow_sets.append((A, B))

                        if not chow_sets:
                            idx = 0
                        else:
                            chosen = teacher_bot.choose_chow(env_now, reacting_seat, tile, chow_sets)
                            if chosen in chow_sets:
                                idx = chow_sets.index(chosen) + 1
                            else:
                                idx = 0

                    #print(f"[teacher-picker] chow | tile={tile} seat={reacting_seat} sets={chow_sets} -> idx={idx}")
                    env_now._forced[reacting_seat] = {"kind": "chow", "idx": idx}

                # ---------- BINARY (ron) ----------
                elif head == "binary":
                    tile = getattr(env_now, "last_discard_tile", None)
                    yes = teacher_bot.decide_ron(
                        env_now,
                        tile,
                        getattr(env_now, "points", None),
                        getattr(env_now, "ron_loser", None),
                    )

                    if len(legal_idx) == 1:
                        # Only one choice; env isn't offering us a real decision.
                        idx = legal_idx[0]
                    else:
                        if yes:
                            claim_indices = [i for i in legal_idx if i != 0]
                            idx = claim_indices[0] if claim_indices else legal_idx[0]
                        else:
                            idx = legal_idx[0]

                # Final safety: never return an illegal action.
                if idx is not None:
                    # For big multi-class heads we still enforce membership
                    if head not in ("pung", "binary"):
                        if idx not in legal_idx:
                            idx = None
            

            except Exception as e:
                #print(f"[teacher-global-error] head={head} seat={seat} exception={e}")
                import traceback
                traceback.print_exc()
                idx = None


            advice_q.append((head, idx))

            # Effective probability of *executing* teacher’s suggestion.
            eff_p = max(0.0, min(1.0, behavior_prob_use * args.oracle_exec_prob))

            # Make sure args.epoch is kept in sync with training loop.
            cur_epoch = getattr(args, "epoch", 0)
            exec_now = (idx is not None and random.random() < eff_p)

            # if seat == 0 and idx is not None:
            #     print(f"[teacher-debug] epoch={args.epoch} head={head} idx={idx} legal={legal_idx}")
            # print(f"[teacher-out] epoch={args.epoch} head={head} idx={idx}")
            # --- BEGIN PATCH: force teacher action into env._forced so env can execute it ---
            # --- BEGIN PATCH ---
            if exec_now and idx is not None and head in ("pung", "chow", "kong", "binary"):
                try:
                    if not isinstance(getattr(env_now, "_forced", None), dict):
                        env_now._forced = {}
                    env_now._forced[seat] = {"kind": head, "idx": int(idx)}
                    #print(f"[forced-debug] pushed seat={seat} head={head} idx={idx} legal={legal_idx}")
                except Exception as e:
                    pass
                    #print(f"[forced-debug-error] seat={seat} head={head} err={repr(e)}")
            # --- END PATCH ---

            # --- BEGIN PATCH (robust _forced wiring) ---
            # if exec_now and idx is not None:
            #     try:
            #         forced = getattr(env_now, "_forced", None)
            #         if not isinstance(forced, dict):
            #             # Reinitialize if missing or some weird legacy value (like 0)
            #             forced = {}
            #             setattr(env_now, "_forced", forced)

            #         forced[seat] = {"kind": head, "idx": int(idx)}
            #         print(f"[forced-debug] pushed seat={seat} head={head} idx={idx} legal={legal_idx}")
            #     except Exception as e:
            #         # Show full exception repr so we can see what's actually happening if something breaks
            #         print(f"[forced-debug-error] seat={seat} head={head} err={repr(e)}")
            # --- END PATCH ---

            # if exec_now and idx is not None and hasattr(env_now, "_forced"):
            #     try:
            #         #env_now._forced[seat] = {"kind": head, "idx": idx}
            #         if not hasattr(env_now, "_forced"):
            #             env_now._forced = [{} for _ in range(4)]
            #         if isinstance(env_now._forced[seat], dict):
            #             env_now._forced[seat][head] = {"idx": idx}
            #         print(f"[forced-debug] pushed seat={seat} head={head} idx={idx}")
            #     except Exception as e:
            #         print(f"[forced-debug-error] seat={seat} head={head} err={e}")
            # --- END PATCH ---

            return idx if exec_now else None

        # Single teacher that uses FlexAggroD for seat 0 on all heads.
        # def teacher_picker(env_now: Env, seat: int, legal_idx: List[int], head: str = "discard") -> Optional[int]:
        #     idx: Optional[int] = None
        #     # Only teach our RL agent (seat 0)
        #     if seat != 0 or not legal_idx:
        #         advice_q.append((head, None))
        #         return None

        #     try:
        #         if head == "discard":
        #             tile = teacher_bot.pick_discard(env_now)
        #             idx = TILE_TO_IDX.get(tile)

        #         elif head == "pung":
        #             # use last discard for the candidate tile
        #             tile = getattr(env_now, "last_discard_tile", None)
        #             yes = teacher_bot.decide_pung(env_now, seat, tile)
        #             idx = 1 if yes else 0

        #         elif head == "chow":
        #             tile = getattr(env_now, "last_discard_tile", None)
        #             chow_sets = getattr(env_now, "legal_chow_sets", lambda s: [])(seat)
        #             chosen = teacher_bot.choose_chow(env_now, seat, tile, chow_sets)
        #             idx = 0 if chosen is None else 1

        #         elif head == "kong":
        #             open_yes  = teacher_bot.decide_open_kong(env_now, seat, getattr(env_now, "last_discard_tile", None))
        #             closed    = teacher_bot.decide_closed_kong(env_now, seat, getattr(env_now, "legal_kong_candidates", lambda s: [])(seat))
        #             add_yes   = teacher_bot.decide_add_kong(env_now, seat, getattr(env_now, "last_discard_tile", None))
        #             any_kong  = (open_yes or closed or add_yes)

        #             # map to first legal kong candidate if yes
        #             if any_kong and len(legal_idx) > 1:
        #                 idx = legal_idx[1]      # pick first actual kong option
        #             else:
        #                 idx = legal_idx[0]      # 0 = no kong

        #         elif head == "binary":
        #             tile = getattr(env_now, "last_discard_tile", None)
        #             yes = teacher_bot.decide_ron(env_now, tile, getattr(env_now, "points", None), getattr(env_now, "ron_loser", None))
        #             idx = 1 if yes else 0

        #         if head in ("pung", "chow", "kong", "binary"):
        #             if len(legal_idx) == 2:  # remap 0=no, 1=yes
        #                 idx = legal_idx[1 if (idx == 1) else 0]
        #                         # --- BEGIN PATCH: safe fallback for claim heads ---
        #         if head in ("pung", "chow", "kong", "binary") and (not legal_idx or len(legal_idx) <= 1):
        #             # When env legality is incomplete, fall back to teacher_bot logic
        #             if head == "pung":
        #                 tile = getattr(env_now, "last_discard_tile", None)
        #                 yes = teacher_bot.decide_pung(env_now, seat, tile)
        #                 idx = 1 if yes else 0
        #             elif head == "chow":
        #                 tile = getattr(env_now, "last_discard_tile", None)
        #                 chow_sets = getattr(env_now, "legal_chow_sets", lambda s: [])(seat)
        #                 chosen = teacher_bot.choose_chow(env_now, seat, tile, chow_sets)
        #                 idx = 0 if chosen is None else 1
        #             elif head == "kong":
        #                 open_yes  = teacher_bot.decide_open_kong(env_now, seat, getattr(env_now, "last_discard_tile", None))
        #                 closed    = teacher_bot.decide_closed_kong(env_now, seat, getattr(env_now, "legal_kong_candidates", lambda s: [])(seat))
        #                 add_yes   = teacher_bot.decide_add_kong(env_now, seat, getattr(env_now, "last_discard_tile", None))
        #                 any_kong  = (open_yes or closed or add_yes)
        #                 idx = 1 if any_kong else 0
        #         # --- END PATCH ---



        #         if idx is not None and idx not in legal_idx:
        #             idx = None
                

        #     except Exception as e:
        #         # print(f"[teacher-picker-error] head={head} err={e}")
        #         idx = None

        #     advice_q.append((head, idx))
        #     eff_p = max(0.0, min(1.0, behavior_prob_use * args.oracle_exec_prob))

        #     # Force teacher for first epochs or testing
        #     if getattr(args, "epoch", 0) < 50:
        #         exec_now = True
        #     else:
        #         exec_now = (idx is not None and random.random() < eff_p)
        #     if seat == 0 and idx is not None:
        #         print(f"[teacher-debug] epoch={args.epoch} head={head} idx={idx} legal={legal_idx}")

        #     #print(f"[teacher-final] seat={seat} head={head} idx={idx} exec={exec_now}")
        #     return idx if exec_now else None

        

        _attach_oracle(rl, teacher_picker)
        #print(f"[debug] epoch={args.epoch} oracle_exec_prob={args.oracle_exec_prob} beh_p={behavior_prob_use}")
        setattr(rl, "teacher_picker", teacher_picker)  # prevent override in later selfplay/oracle code

        #print("[debug] attached_oracle:", getattr(rl, "oracle_picker", None),
            #getattr(rl, "teacher_picker", None))
        # can_rollout = _has_force(env)
        # if (args.oracle_rollouts > 0 or args.oracle_exec_prob > 0) and (not selfplay or not args.oracle_only_vsbots):
        #     rollout_tags = lineup_tags if lineup_tags else args.lineup.split(",")
        #     rollout_lineup = make_lineup_with_rl(rl, rules, rollout_tags)

        #     def scale(x, floor_frac=0.25):
        #         return max(1, int(round(x * max(floor_frac, compute_scale))))

        #     k_eff    = scale(args.oracle_rollouts)
        #     H_eff    = max(8, int(round(args.oracle_horizon * (0.5 + 0.5 * compute_scale))))
        #     topN_eff = scale(args.oracle_topN) if args.oracle_topN > 0 else 0

        #     def teacher_picker(env_now: Env, seat: int, legal_idx: List[int], head: str = "discard") -> Optional[int]:
        #         idx = None
        #         if can_rollout:
        #             cands = legal_idx[:topN_eff] if (topN_eff and topN_eff > 0) else list(legal_idx)
        #             idx = pick_oracle_action(
        #                 env_now, seat, cands, rollout_lineup, rules,
        #                 rollouts_per_action=k_eff, rollout_horizon=H_eff,
        #                 rl_guard=rl, peek_mask=peek_mask_for_episode, seat0_eval=0
        #             )
        #         if idx is None:
        #             try:
        #                 obs_np = build_observation(env_now, seat)
        #                 obs = torch.from_numpy(obs_np).float().to(device)[None, None, :]
        #                 with torch.no_grad():
        #                     y, _ = model(obs)  # hx is optional in updated model
        #                     heads = model.step_logits_value(y.squeeze(0))
        #                     logits = heads.get(head)
        #                     if logits is not None:
        #                         masked = apply_action_mask(logits[0:1, :], legal_idx or list(range(logits.size(-1))))
        #                         idx = int(torch.argmax(masked, dim=-1).item())
        #             except Exception:
        #                 pass
        #         if idx is None and legal_idx:
        #             idx = int(legal_idx[0])

        #         advice_q.append((head, idx))
        #         eff_p = max(0.0, min(1.0, behavior_prob_use * args.oracle_exec_prob))
        #         return idx if (idx is not None and random.random() < eff_p) else None

        #     _attach_oracle(rl, teacher_picker)
        # else:
        #     def teacher_picker_noop(
        #         env_now: Env,
        #         seat: int,
        #         legal_idx: List[int],
        #         head: str = "discard",
        #     ) -> Optional[int]:
        #         idx: Optional[int] = None

        #         # Use flexaggrod as a *teacher* for seat 0 discards
        #         if seat == 0 and head == "discard" and legal_idx:
        #             try:
        #                 # Ask the heuristic bot which tile it would discard
        #                 tile = teacher_bot.pick_discard(env_now)
        #                 idx = TILE_TO_IDX.get(tile)
        #                 # If the suggested tile is somehow illegal in this head, drop it
        #                 if idx is None or idx not in legal_idx:
        #                     idx = None
        #             except Exception:
        #                 idx = None

        #         # Record advice (may be None for non-discard / non-seat0 calls)
        #         advice_q.append((head, idx))

        #         eff_p = max(0.0, min(1.0, behavior_prob_use * args.oracle_exec_prob))
        #         return idx if (idx is not None and random.random() < eff_p) else None

        #     _attach_oracle(rl, teacher_picker_noop)


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
            behavior_prob_use=beh_p,
        )
    else:
        buf, rew, _ = run_episode_core(
            _WORKER_ARGS, _WORKER_RULES, _WORKER_MODEL, _WORKER_DEVICE,
            lineup_tags=[], seed=seed, selfplay=True,
            peek_mask_for_episode=peek_mask, compute_scale=tile_p,
            behavior_prob_use=beh_p,
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

    if phase < 0.3:
        # Early: only aggro-ish bots that actually finish hands
        pool = ["aggro", "hyaggro"]
    elif phase < 0.7:
        # Middle: start mixing flexaggro, still see aggro/hyaggro a lot
        pool = ["aggro", "hyaggro", "flexaggro", "flexaggrod"]
    else:
        # Late vs-bots: mostly strong policies
        pool = ["flexaggro", "flexaggrod"]

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
        return schedule_val(args.bc_weight_start, args.bc_weight_final, epoch_idx,
                            args.bc_weight_schedule, args.bc_weight_warmup, args.epochs)

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
        ent_w = entropy_weight(epoch)
        tile_p = float(peek_probability(epoch))
        beh_p = float(behavior_prob(epoch))
        bc_w  = float(bc_weight(epoch))

        buffers: List[List] = []
        rewards_final: List[float] = []
        coverages: List[float] = []

        entropy_coef = schedule_coef(epoch, args.epochs, final_value=0.015, start_frac=0.1)
        value_coef   = schedule_coef(epoch, args.epochs, final_value=0.25, start_frac=0.1)
        shaping_coef = schedule_coef(epoch, args.epochs, final_value=0.1,  start_frac=0.1)

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
                _ = _randomize_start_player(base_env)
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
        if not buffers:
            print(f"[epoch {epoch+1}/{args.epochs}] no steps collected; skipping update")
            _maybe_save(epoch)
            continue

        logprobs, ent_terms, vals, used_idx, bc_terms, g_pred = a2c_forward(buffers, model, device)

        if epoch == 0:
            cnt = int(bc_terms.numel()) if torch.is_tensor(bc_terms) else 0
            mean_bc = (bc_terms.mean().item() if torch.is_tensor(bc_terms) and cnt > 0 else float('nan'))
            print(f"[bc-sanity] bc_terms_count={cnt} mean={mean_bc:.4f}")

        ret, adv = compute_returns_and_advantages(
            buffers=buffers,
            rewards_final=rewards_final,
            vals_concat=vals,
            gamma=args.gamma,
            lam=args.gae_lambda,
            shaping_coef=args.shaping_coef,
            device=device,
        )

        # Policy / entropy / value / BC losses
        if used_idx.numel() > 0:
            pol_loss = -(logprobs * adv[used_idx]).mean()
            ent_loss = - ent_w * ent_terms.mean()
        else:
            pol_loss = torch.tensor(0.0, device=device)
            ent_loss = torch.tensor(0.0, device=device)
        val_loss = args.value_coef * 0.5 * (ret - vals).pow(2).mean()

        if torch.is_tensor(bc_terms) and bc_terms.numel() > 0:
            bc_loss = bc_w * bc_terms.mean()
        else:
            bc_loss = torch.tensor(0.0, device=device)

        # Global reward prediction loss (auxiliary)
        global_targets_list: List[float] = []
        for buf, R in zip(buffers, rewards_final):
            T = len(buf)
            global_targets_list.extend([float(R)] * T)

        if g_pred.numel() > 0 and len(global_targets_list) == int(g_pred.shape[0]):
            global_targets = torch.tensor(global_targets_list, dtype=torch.float32, device=device)
            global_loss = args.global_reward_coef * 0.5 * (g_pred - global_targets).pow(2).mean()
        else:
            global_loss = torch.tensor(0.0, device=device)

        loss = pol_loss + val_loss + ent_loss + bc_loss + global_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        avg_r = float(np.mean(rewards_final)) if rewards_final else 0.0
        cov_rate = float(np.mean(coverages)) if coverages else 0.0

        print(f"[epoch {epoch+1}/{args.epochs}] eps={len(rewards_final)} "
              f"avg_reward={avg_r:.2f} loss={loss.item():.3f} pol={pol_loss.item():.3f} "
              f"val={val_loss.item():.3f} ent={(-ent_loss).item():.3f} bc={bc_loss.item():.3f} "
              f"glob={global_loss.item():.3f} "
              f"tile_p={tile_p:.3f} peek_cov={cov_rate:.2f} beh_p={beh_p:.2f} bc_w={bc_w:.3f}")

        _maybe_save(epoch)

# ---------------------------- CLI ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--lineup", default="aggro,aggro,hyaggro")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--episodes-per-epoch", type=int, default=384)
    ap.add_argument("--vs-bots-epochs", type=int, default=500)
    ap.add_argument("--max-draws", type=int, default=700)
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
    ap.add_argument("--oracle-behavior-target", type=float, default=0.2)
    ap.add_argument("--oracle-behavior-schedule", choices=["exp","linear"], default="exp")
    ap.add_argument("--oracle-behavior-schedule-warmup", type=int, default=40)

    # BC weight schedule
    ap.add_argument("--bc-weight-start", type=float, default=1.0)
    ap.add_argument("--bc-weight-final", type=float, default=0.25)
    ap.add_argument("--bc-weight-warmup", type=int, default=40)
    ap.add_argument("--bc-weight-schedule", choices=["exp","linear"], default="exp")

    # Anchored repeats
    ap.add_argument("--anchored-batches", type=int, default=0)
    ap.add_argument("--anchored-K", type=int, default=0)

    # New: persistence / eval controls
    ap.add_argument("--outdir", default="runs", help="Root folder for checkpoints/logs")
    ap.add_argument("--run-id", default=None, help="Run name (subfolder under --outdir). Default: timestamp")
    ap.add_argument("--save-every", type=int, default=4, help="Save a checkpoint every N epochs")
    ap.add_argument("--resume", default=None,
    help="Optional path to checkpoint (.pt) to resume from. "
         "If omitted, train.py automatically resumes from runs/<run_id>/checkpoints/last.pt if present.")
    ap.add_argument("--eval-only", action="store_true", help="Load --resume and run evaluation only")
    ap.add_argument("--eval-lineup", default="flexaggro,hyaggro,aggro", help="Opponent lineup for eval")
    ap.add_argument("--eval-episodes", type=int, default=64, help="Evaluation episodes")

    # Number of CPU env workers (0 = no parallelism).
    ap.add_argument("--num-workers", type=int, default=8,
                    help="Number of CPU workers for parallel env rollouts (0 = no parallelism)")

    ap.add_argument("--randomize-dealer", action="store_true", default=True)
    ap.add_argument("--skip-bad-episodes", action="store_true", default=True)

    # Force training on CPU (overrides GPU even if available)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    train(args)
