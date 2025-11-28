# # algorithm/mahjongrl/train.py (stateful checkpoints + resume + eval)
# from __future__ import annotations
# import argparse, random, sys, traceback, os, time, json, pickle, hashlib
# from typing import List, Optional, Callable, Dict, Tuple, Any
# from pathlib import Path
# import numpy as np
# import torch
# import torch.optim as optim
# from copy import deepcopy
# from contextlib import contextmanager
# from collections import deque, Counter

# from algorithm.rules_io import load_rules
# from algorithm.sim_and_train import Env
# from algorithm.mahjongrl.env_wrapper import make_lineup_with_rl, build_observation, compute_rl_reward
# from algorithm.mahjongrl.model import ACConfig, LSTMActorCritic
# from algorithm.mahjongrl.agent import RLPolicy

# # ---------------------------- Repro ----------------------------
# def set_global_seeds(seed: int):
#     random.seed(seed)
#     np.random.seed(seed % (2**32 - 1))
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     try:
#         torch.use_deterministic_algorithms(False)
#     except Exception:
#         pass

# # ---------------------------- Action masking helpers ----------------------------
# TILE_ORDER = (
#     [f"{r}b" for r in range(1,10)] +
#     [f"{r}w" for r in range(1,10)] +
#     [f"{r}t" for r in range(1,10)] +
#     ["E","S","W","N","C","F","B"]   # Winds + Dragons
# )
# TILE_TO_IDX = {t: i for i, t in enumerate(TILE_ORDER)}  # 0..33

# def apply_action_mask(logits: torch.Tensor, valid_indices: List[int]):
#     if logits.dim() == 1:
#         logits = logits.unsqueeze(0)
#     C = logits.size(-1)
#     if not valid_indices:
#         return logits
#     idx = torch.as_tensor(valid_indices, device=logits.device, dtype=torch.long)
#     mask = torch.full((C,), float("-inf"), device=logits.device, dtype=logits.dtype)
#     mask[idx] = 0.0
#     return logits + mask.unsqueeze(0)

# def _extract_valid_indices(s, C: int) -> List[int]:
#     for name in ("legal_idx", "valid_idx", "legal_indices", "valid_indices"):
#         if hasattr(s, name):
#             li = getattr(s, name)
#             if isinstance(li, (list, tuple)) and len(li) > 0:
#                 return [int(i) for i in li if 0 <= int(i) < C]
#     for name in ("mask", "legal_mask", "action_mask"):
#         if hasattr(s, name):
#             m = getattr(s, name)
#             try:
#                 arr = np.asarray(m, dtype=float).reshape(-1)
#                 if arr.shape[0] == C:
#                     return [int(i) for i, v in enumerate(arr) if v > 0.5]
#             except Exception:
#                 pass
#     return list(range(C))

# def _choice_index_for_head(s, C: int):
#     ch = getattr(s, "choice", None)
#     if ch is None:
#         return None
#     if isinstance(ch, str) and C == 34:
#         return TILE_TO_IDX.get(ch, None)
#     try:
#         idx = int(ch)
#     except Exception:
#         return None
#     return idx if 0 <= idx < C else None

# # ---------------------------- Safe wall access helpers ----------------------------
# def _wall_as_list(wall):
#     if isinstance(wall, list):
#         return list(wall)
#     for attr in ("tiles", "_tiles", "data"):
#         if hasattr(wall, attr):
#             try:
#                 return list(getattr(wall, attr))
#             except Exception:
#                 pass
#     try:
#         return list(wall)
#     except Exception:
#         return []

# def _wall_set_from_list(wall, tiles):
#     tiles = list(tiles)
#     if isinstance(wall, list):
#         wall[:] = tiles
#         return
#     for setter in ("set_tiles", "reset", "from_list"):
#         if hasattr(wall, setter) and callable(getattr(wall, setter)):
#             try:
#                 getattr(wall, setter)(tiles)
#                 return
#             except Exception:
#                 pass
#     if hasattr(wall, "clear") and hasattr(wall, "extend"):
#         try:
#             wall.clear()
#             wall.extend(tiles)
#             return
#         except Exception:
#             pass
#     if hasattr(wall, "tiles"):
#         try:
#             setattr(wall, "tiles", tiles)
#             return
#         except Exception:
#             pass
#     try:
#         while True:
#             if len(wall) <= 0:
#                 break
#             wall.pop()
#     except Exception:
#         pass
#     try:
#         for t in tiles:
#             wall.append(t)
#     except Exception:
#         pass

# # ---------------------------- Oracle plumbing ----------------------------
# def _try_force_action(sim: Env, seat: int, action_idx: int) -> bool:
#     for name in ("force_action", "apply_forced_action", "apply_action", "step_with_forced_action"):
#         if hasattr(sim, name):
#             fn = getattr(sim, name)
#             try:
#                 if fn.__code__.co_argcount >= 3:
#                     fn(seat, action_idx)
#                 else:
#                     fn(action_idx)
#                 return True
#             except Exception:
#                 return False
#     return False

# def _has_force(sim: Env) -> bool:
#     return any(hasattr(sim, n) for n in (
#         "force_action", "apply_forced_action", "apply_action", "step_with_forced_action"
#     ))

# @contextmanager
# def _oracle_temporarily_disabled(rl: RLPolicy):
#     had = hasattr(rl, "oracle_picker")
#     old = getattr(rl, "oracle_picker", None) if had else None
#     try:
#         if had:
#             rl.oracle_picker = None
#         yield
#     finally:
#         if had:
#             rl.oracle_picker = old

# class nullcontext:
#     def __enter__(self): return None
#     def __exit__(self, *exc): return False

# # ---------------------------- Peek mask / re-deal ----------------------------
# def _make_peek_mask(env: Env, seat0_eval: int, p: float) -> Dict:
#     mask = {"p": float(p), "opps": {}, "wall": []}
#     for s in range(4):
#         if s == seat0_eval:
#             continue
#         n = len(env.players[s].concealed)
#         mask["opps"][s] = [random.random() < p for _ in range(n)]
#     wall_len = len(_wall_as_list(env.wall))
#     mask["wall"] = [random.random() < p for _ in range(wall_len)]
#     return mask

# def _apply_peek_belief_with_mask(sim: Env, seat0_eval: int, mask: Dict):
#     p = float(mask.get("p", 0.0))

#     for s in range(4):
#         if s == seat0_eval:
#             continue
#         concealed = sim.players[s].concealed
#         n_cur = len(concealed)
#         m = mask["opps"].get(s, [])
#         if len(m) < n_cur:
#             m += [random.random() < p for _ in range(n_cur - len(m))]
#             mask["opps"][s] = m
#         elif len(m) > n_cur:
#             mask["opps"][s] = m[:n_cur]

#     wall_list = _wall_as_list(sim.wall)
#     W = len(wall_list)
#     m_wall = mask.get("wall", [])
#     if len(m_wall) < W:
#         m_wall += [random.random() < p for _ in range(W - len(m_wall))]
#         mask["wall"] = m_wall
#     elif len(m_wall) > W:
#         mask["wall"] = m_wall[:W]

#     unknown_slots = []
#     for s in range(4):
#         if s == seat0_eval:
#             continue
#         for i, peek in enumerate(mask["opps"][s]):
#             if not peek:
#                 unknown_slots.append(("opps", s, i))
#     wall_unknown_idx = [i for i, peek in enumerate(mask["wall"]) if not peek]

#     pool = []
#     for _, s, i in unknown_slots:
#         pool.append(sim.players[s].concealed[i])
#     for i in wall_unknown_idx:
#         if 0 <= i < len(wall_list):
#             pool.append(wall_list[i])

#     if not pool:
#         return

#     random.shuffle(pool)

#     k = 0
#     for _, s, i in unknown_slots:
#         sim.players[s].concealed[i] = pool[k]
#         k += 1

#     for idx in wall_unknown_idx:
#         if 0 <= idx < len(wall_list) and k < len(pool):
#             wall_list[idx] = pool[k]
#             k += 1

#     _wall_set_from_list(sim.wall, wall_list)

# # ---------------------------- Teacher rollouts ----------------------------
# def _peeked_rollout_once(env: Env,
#                          seat: int,
#                          action_idx: int,
#                          lineup_after_pick,
#                          rules: dict,
#                          horizon: int,
#                          rl_guard: Optional[RLPolicy],
#                          peek_mask: Optional[Dict],
#                          seat0_eval: int = 0) -> Optional[float]:
#     snap = env.snapshot()

#     sim = env  # reuse the same Env object
#     sim.restore(snap)
#     if not _try_force_action(sim, seat, action_idx):
#         return None
#     if peek_mask is not None:
#         _apply_peek_belief_with_mask(sim, seat0_eval=seat0_eval, mask=peek_mask)

#     steps = 0
#     ctx = _oracle_temporarily_disabled(rl_guard) if rl_guard is not None else nullcontext()
#     with ctx:
#         while sim.wall and not sim.terminal and steps < horizon:
#             sim.step_turn(lineup_after_pick)
#             steps += 1

#     term = sim.terminal or {"source": "drawn_game"}
#     return compute_rl_reward(term, seat=seat0_eval, rules=rules)
#     # sim = deepcopy(env)
#     # if not _try_force_action(sim, seat, action_idx):
#     #     return None
#     # if peek_mask is not None:
#     #     _apply_peek_belief_with_mask(sim, seat0_eval=seat0_eval, mask=peek_mask)

#     # steps = 0
#     # ctx = _oracle_temporarily_disabled(rl_guard) if rl_guard is not None else nullcontext()
#     # with ctx:
#     #     while sim.wall and not sim.terminal and steps < horizon:
#     #         sim.step_turn(lineup_after_pick)
#     #         steps += 1

#     # term = sim.terminal or {"source": "drawn_game"}
#     # return compute_rl_reward(term, seat=seat0_eval, rules=rules)

# def pick_oracle_action(env: Env,
#                        seat: int,
#                        candidate_indices: List[int],
#                        lineup_after_pick,
#                        rules: dict,
#                        rollouts_per_action: int,
#                        rollout_horizon: int,
#                        rl_guard: Optional[RLPolicy],
#                        peek_mask: Optional[Dict],
#                        seat0_eval: int = 0) -> Optional[int]:
#     """
#     Evaluate each candidate action by performing K rollouts per action.
#     Uses env.snapshot()/env.restore() instead of deepcopy for speed.
#     Semantics identical to the original version.
#     """
#     if not candidate_indices:
#         return None

#     # -------- snapshot once --------
#     try:
#         base_snap = env.snapshot()
#     except Exception as e:
#         # fallback: if snapshot not available, keep old behavior
#         import copy
#         base_snap = None
#         sys.stderr.write(f"[oracle] WARNING: snapshot failed ({e}), falling back to deepcopy\n")

#     best = None

#     for a in candidate_indices:
#         vals: List[float] = []

#         for _ in range(rollouts_per_action):

#             if base_snap is not None:
#                 # ---- restore original state ----
#                 env.restore(base_snap)
#                 sim = env
#             else:
#                 # fallback: old behavior
#                 import copy
#                 sim = copy.deepcopy(env)

#             # ---- force the candidate action ----
#             if not _try_force_action(sim, seat, a):
#                 continue

#             # ---- apply partial peek belief if needed ----
#             if peek_mask is not None:
#                 _apply_peek_belief_with_mask(sim, seat0_eval=seat0_eval, mask=peek_mask)

#             # ---- rollout for horizon ----
#             steps = 0
#             ctx = _oracle_temporarily_disabled(rl_guard) if rl_guard is not None else nullcontext()
#             with ctx:
#                 while sim.wall and not sim.terminal and steps < rollout_horizon:
#                     sim.step_turn(lineup_after_pick)
#                     steps += 1

#             term = sim.terminal or {"source": "drawn_game"}
#             v = compute_rl_reward(term, seat=seat0_eval, rules=rules)

#             if v is not None:
#                 vals.append(v)

#         # ---- aggregate results ----
#         if vals:
#             m = float(np.mean(vals))
#             if best is None or m > best[0]:
#                 best = (m, a)

#     return None if best is None else best[1]


# # def pick_oracle_action(env: Env,
# #                        seat: int,
# #                        candidate_indices: List[int],
# #                        lineup_after_pick,
# #                        rules: dict,
# #                        rollouts_per_action: int,
# #                        rollout_horizon: int,
# #                        rl_guard: Optional[RLPolicy],
# #                        peek_mask: Optional[Dict],
# #                        seat0_eval: int = 0) -> Optional[int]:
# #     if not candidate_indices:
# #         return None
# #     best = None
# #     for a in candidate_indices:
# #         vals: List[float] = []
# #         for _ in range(rollouts_per_action):
# #             v = _peeked_rollout_once(env, seat, a, lineup_after_pick, rules,
# #                                      rollout_horizon, rl_guard, peek_mask, seat0_eval)
# #             if v is not None:
# #                 vals.append(v)
# #         if vals:
# #             m = float(np.mean(vals))
# #             if best is None or m > best[0]:
# #                 best = (m, a)
# #     return None if best is None else best[1]

# def _attach_oracle(rl: RLPolicy, picker: Callable):
#     attached_paths = []
#     for name in ("enable_oracle", "set_oracle", "set_teacher"):
#         if hasattr(rl, name) and callable(getattr(rl, name)):
#             try:
#                 getattr(rl, name)(picker)
#                 attached_paths.append(name)
#             except Exception:
#                 pass
#     for name in ("oracle_picker", "teacher_picker", "oracle", "teacher", "advice_fn", "picker"):
#         try:
#             setattr(rl, name, picker)
#             attached_paths.append(f"attr:{name}")
#         except Exception:
#             pass
#     for flag in ("use_oracle", "use_teacher", "oracle_enabled", "teacher_enabled"):
#         try:
#             setattr(rl, flag, True)
#             attached_paths.append(f"flag:{flag}=True")
#         except Exception:
#             pass
#     if attached_paths:
#         sys.stderr.write("[attach_oracle] " + ", ".join(attached_paths) + "\n")
#     else:
#         sys.stderr.write("[attach_oracle] no known attach points on RLPolicy\n")

# # ---------------------------- A2C forward (with imitation loss hooks) ----------------------------
# def a2c_forward(buffers: List[List], model: LSTMActorCritic, device: str):
#     cfg = model.cfg
#     model.train()
#     logprobs, entropies, used_idx, vpred_list = [], [], [], []
#     bc_terms = []
#     step_offset = 0

#     for buf in buffers:
#         if not buf:
#             continue
#         obs = torch.from_numpy(np.stack([s.obs for s in buf], axis=0)).float().to(device)

#         hx = (torch.zeros(1,1,cfg.lstm, device=device),
#               torch.zeros(1,1,cfg.lstm, device=device))
#         y, _ = model(obs[None, :, :], hx)
#         heads = model.step_logits_value(y.squeeze(0))

#         v = heads["value"].squeeze(-1) if "value" in heads else torch.zeros(len(buf), device=device)
#         vpred_list.append(v)

#         for t, s in enumerate(buf):
#             kind = getattr(s, "kind", "discard")
#             head = {"discard":"discard","ron":"binary","pung":"binary","binary":"binary",
#                     "chow":"chow","kong":"kong"}.get(kind, "discard")
#             if head not in heads or heads[head].ndim == 0:
#                 continue

#             logits = heads[head][t:t+1, :]
#             C = int(logits.size(-1))
#             legal_idx = _extract_valid_indices(s, C) or list(range(C))
#             masked = apply_action_mask(logits, legal_idx)

#             # policy term
#             act_idx = _choice_index_for_head(s, C)
#             if act_idx is not None and act_idx in legal_idx:
#                 dist = torch.distributions.Categorical(logits=masked)
#                 logprobs.append(dist.log_prob(torch.tensor([act_idx], device=device)).squeeze(0))
#                 entropies.append(dist.entropy().mean())
#                 used_idx.append(step_offset + t)

#             # teacher term
#             t_idx = getattr(s, "teacher_idx", None)
#             if t_idx is None:
#                 tch = getattr(s, "teacher_choice", None)
#                 if tch is not None:
#                     if isinstance(tch, str) and C == 34:
#                         t_idx = TILE_TO_IDX.get(tch, None)
#                     else:
#                         try:
#                             t_idx = int(tch)
#                         except Exception:
#                             t_idx = None

#             if t_idx is not None and 0 <= int(t_idx) < C and (not legal_idx or int(t_idx) in legal_idx):
#                 logp_at_teacher = torch.log_softmax(masked, dim=-1)[0, int(t_idx)]
#                 bc_terms.append(-logp_at_teacher)

#         step_offset += len(buf)

#     v_pred = torch.cat(vpred_list) if vpred_list else torch.empty(0, device=device)
#     logprobs_t = torch.stack(logprobs) if logprobs else torch.tensor([0.0], device=device)
#     entropies_t = torch.stack(entropies) if entropies else torch.tensor([0.0], device=device)
#     used_idx_t = torch.tensor(used_idx, dtype=torch.long, device=device) if used_idx else torch.empty(0, dtype=torch.long, device=device)
#     bc_terms_t = torch.stack(bc_terms) if bc_terms else torch.empty(0, device=device)
#     return logprobs_t, entropies_t, v_pred, used_idx_t, bc_terms_t

# # ---------------------------- GAE(λ) ----------------------------
# def compute_returns_and_advantages(
#     buffers: List[List],
#     rewards_final: List[float],
#     vals_concat: torch.Tensor,
#     gamma: float, lam: float,
#     shaping_coef: float,
#     device: str
# ):
#     rews_all: List[float] = []
#     for buf, R in zip(buffers, rewards_final):
#         T = len(buf)
#         for j, s in enumerate(buf):
#             r_shape = 0.0
#             if shaping_coef > 0.0:
#                 if hasattr(s, "shape_delta"):
#                     r_shape = float(getattr(s, "shape_delta"))
#                 elif hasattr(s, "comp_improve"):
#                     r_shape = float(getattr(s, "comp_improve"))
#             r_final = R if j == T - 1 else 0.0
#             rews_all.append(shaping_coef * r_shape + r_final)

#     r = torch.tensor(rews_all, dtype=torch.float32, device=device)
#     v = vals_concat
#     assert r.shape[0] == v.shape[0], f"reward/values length mismatch: {r.shape} vs {v.shape}"

#     adv = torch.zeros_like(v)
#     ret = torch.zeros_like(v)
#     idx = 0
#     for buf in buffers:
#         T = len(buf)
#         gae = 0.0
#         next_value = 0.0
#         for t in reversed(range(T)):
#             rt = r[idx + t]
#             vt = v[idx + t]
#             delta = rt + gamma * (next_value if t != T - 1 else 0.0) - vt
#             gae = float(delta) + gamma * lam * gae
#             adv[idx + t] = gae
#             ret[idx + t] = adv[idx + t] + vt
#             next_value = vt
#         idx += T

#     adv = (adv - adv.mean()) / (adv.std() + 1e-8)
#     return ret, adv

# # ---------------------------- Utilities ----------------------------
# def _randomize_start_player(env) -> int:
#     k = random.randint(0, 3)
#     for attr in ("dealer", "turn", "current_player", "start_seat"):
#         if hasattr(env, attr):
#             try:
#                 setattr(env, attr, k)
#             except Exception:
#                 pass
#     return k

# def _resample_others_and_wall_keep_seat0(sim: Env):
#     opp_slots = []
#     opp_tiles = []
#     for s in range(1, 4):
#         c = sim.players[s].concealed
#         for i in range(len(c)):
#             opp_slots.append((s, i))
#             opp_tiles.append(c[i])

#     wall_tiles = _wall_as_list(sim.wall)

#     pool = opp_tiles + wall_tiles
#     random.shuffle(pool)

#     j = 0
#     for (s, i) in opp_slots:
#         pass  # (kept in case of future logic changes)
#     for (s, i) in opp_slots:
#         sim.players[s].concealed[i] = pool[j]
#         j += 1

#     new_wall = pool[j:]
#     _wall_set_from_list(sim.wall, new_wall)

# # ---------------------------- Checkpoint helpers ----------------------------
# def _now_run_id() -> str:
#     return time.strftime("%Y%m%d-%H%M%S")

# def _sha_short(d: Dict[str, Any]) -> str:
#     try:
#         s = json.dumps(d, sort_keys=True).encode("utf-8")
#         return hashlib.sha1(s).hexdigest()[:8]
#     except Exception:
#         return "na"

# def save_ckpt(path: str, payload: Dict[str, Any]) -> None:
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save(payload, path)

# def load_ckpt(path: str) -> Dict[str, Any]:
#     return torch.load(path, map_location="cpu")

# @contextmanager
# def _temp_args(args, **overrides):
#     backup = {}
#     for k, v in overrides.items():
#         backup[k] = getattr(args, k)
#         setattr(args, k, v)
#     try:
#         yield
#     finally:
#         for k, v in backup.items():
#             setattr(args, k, v)

# # ---------------------------- Training ----------------------------
# def train(args):
#     # Directories / run metadata
#     run_id = args.run_id or _now_run_id()
#     run_dir = Path(args.outdir) / run_id
#     ckpt_dir = run_dir / "checkpoints"
#     ckpt_dir.mkdir(parents=True, exist_ok=True)
#     meta = {"argv": " ".join(os.sys.argv), "args": vars(args)}
#     (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
#     print(f"[run] dir={run_dir}")

#     device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
#     rules = load_rules(args.rules)
#     set_global_seeds(args.seed)

#     dummy = Env(rules, seed=123)
#     obs0 = build_observation(dummy, seat=0)
#     cfg = ACConfig(obs_dim=obs0.shape[0], hidden=args.hidden, lstm=args.lstm)
#     model = LSTMActorCritic(cfg).to(device)
#     opt = optim.Adam(model.parameters(), lr=args.lr)

#     # ---- Resume (if provided) ----
#     start_epoch = 0
#     if args.resume:
#         print(f"[ckpt] loading {args.resume}")
#         state = load_ckpt(args.resume)
#         msd = state.get("model_state") or state.get("model")
#         if msd:
#             try: model.load_state_dict(msd)
#             except Exception as e: print(f"[ckpt] model load warning: {e}")
#         osd = state.get("opt_state") or state.get("optimizer")
#         if osd:
#             try: opt.load_state_dict(osd)
#             except Exception as e: print(f"[ckpt] optimizer load warning: {e}")
#         # RNG
#         if state.get("py_rng_state"): random.setstate(state["py_rng_state"])
#         if state.get("np_rng_state"): np.random.set_state(state["np_rng_state"])  # type: ignore
#         if state.get("torch_rng_state") is not None:
#             try: torch.random.set_rng_state(state["torch_rng_state"])  # type: ignore
#             except Exception: pass
#         start_epoch = int(state.get("epoch", -1)) + 1
#         print(f"[ckpt] resume at epoch {start_epoch}")

#     # ---- Eval-only shortcut ----
#     if args.eval_only:
#         assert args.resume, "--eval-only requires --resume"
#         lineup = args.eval_lineup.split(",") if args.eval_lineup else args.lineup.split(",")
#         episodes = max(1, int(args.eval_episodes))
#         avg_r = 0.0
#         with _temp_args(args,
#                         oracle_rollouts=0,
#                         oracle_exec_prob=0.0,
#                         peek_prob0=0.0,
#                         peek_prob_target=0.0):
#             for _ in range(episodes):
#                 seed = random.randint(1, 10**9)
#                 env0 = Env(rules, seed=seed)
#                 peek_mask = _make_peek_mask(env0, seat0_eval=0, p=0.0)
#                 buf, rew, _ = run_episode_core(
#                     args, rules, model, device,
#                     lineup_tags=lineup, seed=seed, selfplay=False,
#                     peek_mask_for_episode=peek_mask, compute_scale=0.0, behavior_prob_use=0.0
#                 )
#                 avg_r += float(rew)
#         avg_r /= episodes
#         print(f"[eval] lineup={','.join(lineup)} episodes={episodes} avg_reward={avg_r:.3f}")
#         return

#     # ---- Episode runner captured as inner so it can be reused by eval ----
#     def run_episode_core(args, rules, model, device,
#                          lineup_tags: List[str], seed: int, selfplay: bool,
#                          peek_mask_for_episode: Optional[Dict],
#                          compute_scale: float, behavior_prob_use: float) -> Tuple[List, float, Optional[int]]:
#         env = Env(rules, seed=seed)
#         if getattr(args, "randomize_dealer", True):
#             _randomize_start_player(env)

#         rl = RLPolicy(seat=0, rules=rules, model=model, device=device)

#         start_seat = None
#         for attr in ("turn", "dealer", "current_player", "start_seat"):
#             if hasattr(env, attr):
#                 try:
#                     start_seat = int(getattr(env, attr))
#                     break
#                 except Exception:
#                     pass

#         if selfplay:
#             lineup = [rl, RLPolicy(1, rules, model, device),
#                          RLPolicy(2, rules, model, device),
#                          RLPolicy(3, rules, model, device)]
#         else:
#             lineup = make_lineup_with_rl(rl, rules, lineup_tags)

#         advice_q: List[Tuple[str, Optional[int]]] = []  # (head, idx)
#         can_rollout = _has_force(env)
#         if args.oracle_rollouts > 0 and (not selfplay or not args.oracle_only_vsbots):
#             rollout_tags = lineup_tags if lineup_tags else args.lineup.split(",")
#             rollout_lineup = make_lineup_with_rl(rl, rules, rollout_tags)

#             def scale(x, floor_frac=0.25):
#                 return max(1, int(round(x * max(floor_frac, compute_scale))))

#             k_eff    = scale(args.oracle_rollouts)
#             H_eff    = max(8, int(round(args.oracle_horizon * (0.5 + 0.5 * compute_scale))))
#             topN_eff = scale(args.oracle_topN) if args.oracle_topN > 0 else 0

#             def teacher_picker(env_now: Env, seat: int, legal_idx: List[int], head: str = "discard") -> Optional[int]:
#                 idx = None
#                 if can_rollout:
#                     cands = legal_idx[:topN_eff] if (topN_eff and topN_eff > 0) else list(legal_idx)
#                     idx = pick_oracle_action(env_now, seat, cands, rollout_lineup, rules,
#                                              rollouts_per_action=k_eff, rollout_horizon=H_eff,
#                                              rl_guard=rl, peek_mask=peek_mask_for_episode, seat0_eval=0)
#                 if idx is None:
#                     try:
#                         obs_np = build_observation(env_now, seat)
#                         obs = torch.from_numpy(obs_np).float().to(device)[None, None, :]
#                         with torch.no_grad():
#                             y, _ = model(obs)
#                             heads = model.step_logits_value(y.squeeze(0))
#                             logits = heads.get(head)
#                             if logits is not None:
#                                 masked = apply_action_mask(logits[0:1, :], legal_idx or list(range(logits.size(-1))))
#                                 idx = int(torch.argmax(masked, dim=-1).item())
#                     except Exception:
#                         pass
#                 if idx is None and legal_idx:
#                     idx = int(legal_idx[0])

#                 advice_q.append((head, idx))
#                 eff_p = max(0.0, min(1.0, behavior_prob_use * args.oracle_exec_prob))
#                 return idx if (idx is not None and random.random() < eff_p) else None

#             _attach_oracle(rl, teacher_picker)
#         else:
#             def teacher_picker_noop(env_now: Env, seat: int, legal_idx: List[int], head: str = "discard") -> Optional[int]:
#                 advice_q.append((head, None))
#                 return None
#             _attach_oracle(rl, teacher_picker_noop)

#         # Simulate episode
#         draws = 0
#         try:
#             while env.wall and not env.terminal and draws < args.max_draws:
#                 env.step_turn(lineup)
#                 draws += 1
#         except Exception as e:
#             if args.skip_bad_episodes and "list.remove" in str(e):
#                 sys.stderr.write(f"[warn] skipping corrupted episode (remove error): {e}\n")
#                 return [], 0.0, start_seat
#             else:
#                 traceback.print_exc()
#                 raise

#         # Attach advice by head
#         if rl.buffer:
#             head_map = {
#                 "discard": "discard",
#                 "ron":     "binary",
#                 "pung":    "binary",
#                 "binary":  "binary",
#                 "chow":    "chow",
#                 "kong":    "kong",
#             }

#             q_by_head: Dict[str, deque] = {h: deque() for h in ("discard","binary","chow","kong")}
#             for h, idx in advice_q:
#                 if h in q_by_head:
#                     q_by_head[h].append(idx)

#             attach_cnt = Counter()
#             total_nonnull = sum(1 for _, v in advice_q if v is not None)

#             for step in rl.buffer:
#                 step_head = head_map.get(getattr(step, "kind", "discard"))
#                 if step_head not in q_by_head:
#                     continue
#                 if not q_by_head[step_head]:
#                     continue
#                 idx = q_by_head[step_head].popleft()
#                 if idx is not None:
#                     try:
#                         setattr(step, "teacher_idx", int(idx))
#                         attach_cnt[step_head] += 1
#                     except Exception:
#                         pass

#             if len(advice_q) == 0:
#                 # Fallback #1: offline masked-argmax (needs step.obs)
#                 offline_cnt = Counter()
#                 for step in rl.buffer:
#                     kind = getattr(step, "kind", "discard")
#                     head = head_map.get(kind, None)
#                     if head is None or not hasattr(step, "obs"):
#                         continue
#                     try:
#                         obs_np = np.asarray(step.obs, dtype=np.float32)
#                         obs = torch.from_numpy(obs_np).float().to(device)[None, None, :]
#                         with torch.no_grad():
#                             y, _ = model(obs)
#                             heads = model.step_logits_value(y.squeeze(0))
#                             logits = heads.get(head, None)
#                             if logits is None:
#                                 continue
#                             C = int(logits.size(-1))
#                             legal_idx = _extract_valid_indices(step, C) or list(range(C))
#                             masked = apply_action_mask(logits[0:1, :], legal_idx)
#                             idx = int(torch.argmax(masked, dim=-1).item())
#                             setattr(step, "teacher_idx", idx)
#                             offline_cnt[head] += 1
#                     except Exception:
#                         pass

#                 if sum(offline_cnt.values()) > 0:
#                     pass
#                 else:
#                     # Fallback #2: executed-action imitation (uses step.choice only)
#                     exec_cnt = Counter()
#                     for step in rl.buffer:
#                         kind = getattr(step, "kind", "discard")
#                         head = head_map.get(kind, None)
#                         if head is None:
#                             continue
#                         if hasattr(step, "choice"):
#                             try:
#                                 setattr(step, "teacher_choice", getattr(step, "choice"))
#                                 exec_cnt[head] += 1
#                             except Exception:
#                                 pass
#             else:
#                 pass

#         reward = compute_rl_reward(env.terminal or {"source":"drawn_game"}, seat=0, rules=rules)
#         return rl.buffer, reward, start_seat

#     # expose inner runner to outer scope (for eval-only path)
#     globals()["run_episode_core"] = run_episode_core

#     # ---- Schedules ----
#     def schedule_val(a0, aT, epoch_idx: int, kind: str, warmup: int, total_epochs: int) -> float:
#         if epoch_idx < warmup:
#             return a0
#         k = epoch_idx - warmup
#         E = max(1, total_epochs - warmup)
#         if kind == "exp":
#             d = (aT / max(1e-12, a0)) ** (1.0 / max(1, E - 1))
#             return max(aT, a0 * (d ** k))
#         t = k / max(1, E - 1)
#         return max(aT, a0 + (aT - a0) * t)

#     def peek_probability(epoch_idx: int) -> float:
#         return schedule_val(args.peek_prob0, args.peek_prob_target, epoch_idx,
#                             args.peek_prob_schedule, args.peek_prob_warmup, args.epochs)

#     def behavior_prob(epoch_idx: int) -> float:
#         return schedule_val(args.oracle_behavior_prob, args.oracle_behavior_target,
#                             epoch_idx, args.oracle_behavior_schedule, args.oracle_behavior_schedule_warmup, args.epochs)

#     def bc_weight(epoch_idx: int) -> float:
#         return schedule_val(args.bc_weight_start, args.bc_weight_final, epoch_idx,
#                             args.bc_weight_schedule, args.bc_weight_warmup, args.epochs)

#     def entropy_weight(epoch_idx: int) -> float:
#         hi, lo = max(args.entropy_coef, 0.03), args.entropy_coef
#         T = max(1, args.vs_bots_epochs)
#         t = min(1.0, epoch_idx / T)
#         return lo + (hi - lo) * (1.0 - t)

#     # ---- Checkpoint saver ----
#     def _maybe_save(epoch: int, extra: Optional[Dict[str, Any]] = None):
#         if (epoch % max(1, args.save_every)) != 0:
#             return
#         payload: Dict[str, Any] = {
#             "epoch": epoch,
#             "args": vars(args),
#             "model_state": model.state_dict(),
#             "opt_state": opt.state_dict(),
#             "py_rng_state": random.getstate(),
#             "np_rng_state": np.random.get_state(),
#             "torch_rng_state": torch.random.get_rng_state(),
#             "timestamp": time.time(),
#         }
#         if extra:
#             payload.update(extra)
#         tag = f"epoch{epoch:04d}"
#         save_ckpt(str((ckpt_dir / f"{tag}.pt").resolve()), payload)
#         save_ckpt(str((ckpt_dir / "last.pt").resolve()), payload)
#         print(f"[ckpt] saved {tag}")

#     # ---- Training loop ----
#     for epoch in range(start_epoch, args.epochs):
#         ent_w = entropy_weight(epoch)
#         tile_p = float(peek_probability(epoch))
#         beh_p = float(behavior_prob(epoch))
#         bc_w  = float(bc_weight(epoch))

#         buffers: List[List] = []
#         rewards_final: List[float] = []
#         coverages: List[float] = []

#         # Standard episodes
#         for _ in range(args.episodes_per_epoch):
#             seed = random.randint(1, 10**9)

#             env0 = Env(rules, seed=seed)
#             peek_mask = _make_peek_mask(env0, seat0_eval=0, p=tile_p)
#             total_slots = sum(len(v) for v in peek_mask["opps"].values()) + len(peek_mask["wall"])
#             cov = (sum(sum(1 for b in v if b) for v in peek_mask["opps"].values())
#                    + sum(1 for b in peek_mask["wall"] if b))
#             coverages.append((cov / max(1, total_slots)) if total_slots else 0.0)

#             if epoch < args.vs_bots_epochs:
#                 lineup = args.lineup.split(",")
#                 buf, rew, _ = run_episode_core(
#                     args, rules, model, device,
#                     lineup_tags=lineup, seed=seed, selfplay=False,
#                     peek_mask_for_episode=peek_mask,
#                     compute_scale=tile_p,
#                     behavior_prob_use=beh_p,
#                 )
#             else:
#                 buf, rew, _ = run_episode_core(
#                     args, rules, model, device,
#                     lineup_tags=[], seed=seed, selfplay=True,
#                     peek_mask_for_episode=peek_mask,
#                     compute_scale=tile_p,
#                     behavior_prob_use=beh_p,
#                 )
#             if buf:
#                 buffers.append(buf)
#                 rewards_final.append(rew)

#         # Anchored repeats
#         if args.anchored_batches > 0 and args.anchored_K > 0:
#             for _ in range(args.anchored_batches):
#                 base_env = Env(rules, seed=random.randint(1, 10**9))
#                 _ = _randomize_start_player(base_env)
#                 for _k in range(args.anchored_K):
#                     sim = deepcopy(base_env)
#                     _resample_others_and_wall_keep_seat0(sim)

#                     rl0 = RLPolicy(0, rules, model, device)
#                     lineup = [rl0, RLPolicy(1, rules, model, device),
#                                    RLPolicy(2, rules, model, device),
#                                    RLPolicy(3, rules, model, device)]
#                     _attach_oracle(rl0, lambda *a, **kw: None)

#                     draws = 0
#                     try:
#                         while sim.wall and not sim.terminal and draws < args.max_draws:
#                             sim.step_turn(lineup)
#                             draws += 1
#                     except Exception:
#                         break

#                     if rl0.buffer:
#                         buffers.append(rl0.buffer)
#                         reward = compute_rl_reward(sim.terminal or {"source":"drawn_game"}, seat=0, rules=rules)
#                         rewards_final.append(reward)

#         # Skip update if nothing collected
#         if not buffers:
#             print(f"[epoch {epoch+1}/{args.epochs}] no steps collected; skipping update")
#             _maybe_save(epoch)
#             continue

#         logprobs, ent_terms, vals, used_idx, bc_terms = a2c_forward(buffers, model, device)

#         if epoch == 0:
#             cnt = int(bc_terms.numel()) if torch.is_tensor(bc_terms) else 0
#             mean_bc = (bc_terms.mean().item() if torch.is_tensor(bc_terms) and cnt > 0 else float('nan'))
#             print(f"[bc-sanity] bc_terms_count={cnt} mean={mean_bc:.4f}")

#         ret, adv = compute_returns_and_advantages(
#             buffers=buffers,
#             rewards_final=rewards_final,
#             vals_concat=vals,
#             gamma=args.gamma,
#             lam=args.gae_lambda,
#             shaping_coef=args.shaping_coef,
#             device=device,
#         )

#         if used_idx.numel() > 0:
#             pol_loss = -(logprobs * adv[used_idx]).mean()
#             ent_loss = - ent_w * ent_terms.mean()
#         else:
#             pol_loss = torch.tensor(0.0, device=device)
#             ent_loss = torch.tensor(0.0, device=device)
#         val_loss = args.value_coef * 0.5 * (ret - vals).pow(2).mean()

#         if torch.is_tensor(bc_terms) and bc_terms.numel() > 0:
#             bc_loss = bc_w * bc_terms.mean()
#         else:
#             bc_loss = torch.tensor(0.0, device=device)

#         loss = pol_loss + val_loss + ent_loss + bc_loss

#         opt.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         opt.step()

#         avg_r = float(np.mean(rewards_final)) if rewards_final else 0.0
#         cov_rate = float(np.mean(coverages)) if coverages else 0.0

#         print(f"[epoch {epoch+1}/{args.epochs}] eps={len(rewards_final)} "
#               f"avg_reward={avg_r:.2f} loss={loss.item():.3f} pol={pol_loss.item():.3f} "
#               f"val={val_loss.item():.3f} ent={(-ent_loss).item():.3f} bc={bc_loss.item():.3f} "
#               f"tile_p={tile_p:.3f} peek_cov={cov_rate:.2f} beh_p={beh_p:.2f} bc_w={bc_w:.3f}")

#         _maybe_save(epoch)

# # ---------------------------- CLI ----------------------------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--rules", required=True)
#     ap.add_argument("--lineup", default="hybrid,hybrid,hybrid")
#     ap.add_argument("--epochs", type=int, default=120)
#     ap.add_argument("--episodes-per-epoch", type=int, default=384)
#     ap.add_argument("--vs-bots-epochs", type=int, default=60)
#     ap.add_argument("--max-draws", type=int, default=700)
#     ap.add_argument("--hidden", type=int, default=256)
#     ap.add_argument("--lstm", type=int, default=256)
#     ap.add_argument("--lr", type=float, default=3e-4)
#     ap.add_argument("--gamma", type=float, default=0.99)
#     ap.add_argument("--gae-lambda", type=float, default=0.95)
#     ap.add_argument("--entropy-coef", type=float, default=0.01)
#     ap.add_argument("--value-coef", type=float, default=0.25)
#     ap.add_argument("--shaping-coef", type=float, default=0.0)
#     ap.add_argument("--seed", type=int, default=1234567)

#     # Teacher
#     ap.add_argument("--oracle-rollouts", type=int, default=64)
#     ap.add_argument("--oracle-horizon", type=int, default=48)
#     ap.add_argument("--oracle-topN", type=int, default=16)
#     # Oracle gating
#     og = ap.add_mutually_exclusive_group()
#     og.add_argument("--oracle-only-vsbots", dest="oracle_only_vsbots", action="store_true",
#                     help="Use oracle only during vs-bots epochs.")
#     og.add_argument("--no-oracle-only-vsbots", dest="oracle_only_vsbots", action="store_false",
#                     help="Allow oracle in all epochs.")
#     ap.set_defaults(oracle_only_vsbots=False)

#     ap.add_argument("--oracle-exec-prob", type=float, default=0.5)

#     # Peek prob schedule
#     ap.add_argument("--peek-prob0", type=float, default=1.0)
#     ap.add_argument("--peek-prob-target", type=float, default=0.05)
#     ap.add_argument("--peek-prob-warmup", type=int, default=5)
#     ap.add_argument("--peek-prob-schedule", choices=["exp","linear"], default="exp")

#     # DAgger μ schedule
#     ap.add_argument("--oracle-behavior-prob", type=float, default=0.8)
#     ap.add_argument("--oracle-behavior-target", type=float, default=0.2)
#     ap.add_argument("--oracle-behavior-schedule", choices=["exp","linear"], default="exp")
#     ap.add_argument("--oracle-behavior-schedule-warmup", type=int, default=0)

#     # BC weight schedule
#     ap.add_argument("--bc-weight-start", type=float, default=0.5)
#     ap.add_argument("--bc-weight-final", type=float, default=0.05)
#     ap.add_argument("--bc-weight-warmup", type=int, default=0)
#     ap.add_argument("--bc-weight-schedule", choices=["exp","linear"], default="exp")

#     # Anchored repeats
#     ap.add_argument("--anchored-batches", type=int, default=4)
#     ap.add_argument("--anchored-K", type=int, default=32)

#     # New: persistence / eval controls
#     ap.add_argument("--outdir", default="runs", help="Root folder for checkpoints/logs")
#     ap.add_argument("--run-id", default=None, help="Run name (subfolder under --outdir). Default: timestamp")
#     ap.add_argument("--save-every", type=int, default=1, help="Save a checkpoint every N epochs")
#     ap.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume from")
#     ap.add_argument("--eval-only", action="store_true", help="Load --resume and run evaluation only")
#     ap.add_argument("--eval-lineup", default="hybrid,hybrid,hybrid", help="Opponent lineup for eval")
#     ap.add_argument("--eval-episodes", type=int, default=64, help="Evaluation episodes")

#     ap.add_argument("--randomize-dealer", action="store_true", default=True)
#     ap.add_argument("--skip-bad-episodes", action="store_true", default=True)
#     ap.add_argument("--cpu", action="store_true")
#     args = ap.parse_args()
#     train(args)

# algorithm/mahjongrl/train.py (stateful checkpoints + resume + eval)
from __future__ import annotations
import argparse, random, sys, traceback, os, time, json, pickle, hashlib
from typing import List, Optional, Callable, Dict, Tuple, Any
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from contextlib import contextmanager
from collections import deque, Counter
import multiprocessing as mp
import types

from algorithm.rules_io import load_rules
from algorithm.sim_and_train import Env
from algorithm.mahjongrl.env_wrapper import make_lineup_with_rl, build_observation, compute_rl_reward
from algorithm.mahjongrl.model import ACConfig, LSTMActorCritic
from algorithm.mahjongrl.agent import RLPolicy

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
def _oracle_temporarily_disabled(rl: RLPolicy):
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
    for s in range(4):
        if s == seat0_eval:
            continue
        for i, peek in enumerate(mask["opps"][s]):
            if not peek:
                unknown_slots.append(("opps", s, i))
    wall_unknown_idx = [i for i, peek in enumerate(mask["wall"]) if not peek]

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
                         rl_guard: Optional[RLPolicy],
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
                       rl_guard: Optional[RLPolicy],
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

def _attach_oracle(rl: RLPolicy, picker: Callable):
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
    if attached_paths:
        sys.stderr.write("[attach_oracle] " + ", ".join(attached_paths) + "\n")
    else:
        sys.stderr.write("[attach_oracle] no known attach points on RLPolicy\n")

# ---------------------------- A2C forward (with imitation loss hooks) ----------------------------
def a2c_forward(buffers: List[List], model: LSTMActorCritic, device: str):
    cfg = model.cfg
    model.train()
    logprobs, entropies, used_idx, vpred_list = [], [], [], []
    bc_terms = []
    step_offset = 0

    for buf in buffers:
        if not buf:
            continue
        obs = torch.from_numpy(np.stack([s.obs for s in buf], axis=0)).float().to(device)

        hx = (torch.zeros(1,1,cfg.lstm, device=device),
              torch.zeros(1,1,cfg.lstm, device=device))
        y, _ = model(obs[None, :, :], hx)
        heads = model.step_logits_value(y.squeeze(0))

        v = heads["value"].squeeze(-1) if "value" in heads else torch.zeros(len(buf), device=device)
        vpred_list.append(v)

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
    logprobs_t = torch.stack(logprobs) if logprobs else torch.tensor([0.0], device=device)
    entropies_t = torch.stack(entropies) if entropies else torch.tensor([0.0], device=device)
    used_idx_t = torch.tensor(used_idx, dtype=torch.long, device=device) if used_idx else torch.empty(0, dtype=torch.long, device=device)
    bc_terms_t = torch.stack(bc_terms) if bc_terms else torch.empty(0, device=device)
    return logprobs_t, entropies_t, v_pred, used_idx_t, bc_terms_t

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
        pass  # (kept in case of future logic changes)
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

def load_ckpt(path: str) -> Dict[str, Any]:
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

    rl = RLPolicy(seat=0, rules=rules, model=model, device=device)

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

    advice_q: List[Tuple[str, Optional[int]]] = []  # (head, idx)
    can_rollout = _has_force(env)
    if args.oracle_rollouts > 0 and (not selfplay or not args.oracle_only_vsbots):
        rollout_tags = lineup_tags if lineup_tags else args.lineup.split(",")
        rollout_lineup = make_lineup_with_rl(rl, rules, rollout_tags)

        def scale(x, floor_frac=0.25):
            return max(1, int(round(x * max(floor_frac, compute_scale))))

        k_eff    = scale(args.oracle_rollouts)
        H_eff    = max(8, int(round(args.oracle_horizon * (0.5 + 0.5 * compute_scale))))
        topN_eff = scale(args.oracle_topN) if args.oracle_topN > 0 else 0

        def teacher_picker(env_now: Env, seat: int, legal_idx: List[int], head: str = "discard") -> Optional[int]:
            idx = None
            if can_rollout:
                cands = legal_idx[:topN_eff] if (topN_eff and topN_eff > 0) else list(legal_idx)
                idx = pick_oracle_action(
                    env_now, seat, cands, rollout_lineup, rules,
                    rollouts_per_action=k_eff, rollout_horizon=H_eff,
                    rl_guard=rl, peek_mask=peek_mask_for_episode, seat0_eval=0
                )
            if idx is None:
                try:
                    obs_np = build_observation(env_now, seat)
                    obs = torch.from_numpy(obs_np).float().to(device)[None, None, :]
                    with torch.no_grad():
                        y, _ = model(obs)
                        heads = model.step_logits_value(y.squeeze(0))
                        logits = heads.get(head)
                        if logits is not None:
                            masked = apply_action_mask(logits[0:1, :], legal_idx or list(range(logits.size(-1))))
                            idx = int(torch.argmax(masked, dim=-1).item())
                except Exception:
                    pass
            if idx is None and legal_idx:
                idx = int(legal_idx[0])

            advice_q.append((head, idx))
            eff_p = max(0.0, min(1.0, behavior_prob_use * args.oracle_exec_prob))
            return idx if (idx is not None and random.random() < eff_p) else None

        _attach_oracle(rl, teacher_picker)
    else:
        def teacher_picker_noop(env_now: Env, seat: int, legal_idx: List[int], head: str = "discard") -> Optional[int]:
            advice_q.append((head, None))
            return None
        _attach_oracle(rl, teacher_picker_noop)

    # Simulate episode
    draws = 0
    try:
        while env.wall and not env.terminal and draws < args.max_draws:
            env.step_turn(lineup)
            draws += 1
    except Exception as e:
        if args.skip_bad_episodes and "list.remove" in str(e):
            sys.stderr.write(f"[warn] skipping corrupted episode (remove error): {e}\n")
            return [], 0.0, start_seat
        else:
            traceback.print_exc()
            raise

    # Attach advice by head
    if rl.buffer:
        head_map = {
            "discard": "discard",
            "ron":     "binary",
            "pung":    "binary",
            "binary":  "binary",
            "chow":    "chow",
            "kong":    "kong",
        }

        q_by_head: Dict[str, deque] = {h: deque() for h in ("discard","binary","chow","kong")}
        for h, idx in advice_q:
            if h in q_by_head:
                q_by_head[h].append(idx)

        attach_cnt = Counter()
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

        if len(advice_q) == 0:
            # Fallback #1: offline masked-argmax (needs step.obs)
            offline_cnt = Counter()
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
                exec_cnt = Counter()
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

    reward = compute_rl_reward(env.terminal or {"source":"drawn_game"}, seat=0, rules=rules)
    return rl.buffer, reward, start_seat

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
    Builds a CPU copy of the model and stores rules/args globals.
    """
    global _WORKER_MODEL, _WORKER_RULES, _WORKER_ARGS, _WORKER_DEVICE
    _WORKER_DEVICE = "cpu"
    cfg = ACConfig(**cfg_dict)
    m = LSTMActorCritic(cfg).to(_WORKER_DEVICE)
    m.load_state_dict({k: v.cpu() for k, v in model_state.items()})
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

    env0 = Env(_WORKER_RULES, seed=seed)
    peek_mask = _make_peek_mask(env0, seat0_eval=0, p=tile_p)
    total_slots = sum(len(v) for v in peek_mask["opps"].values()) + len(peek_mask["wall"])
    if total_slots > 0:
        cov = (sum(sum(1 for b in v if b) for v in peek_mask["opps"].values())
               + sum(1 for b in peek_mask["wall"] if b))
        coverage = cov / total_slots
    else:
        coverage = 0.0

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

# ---------------------------- Training ----------------------------
def train(args):
    # Directories / run metadata
    run_id = args.run_id or _now_run_id()
    run_dir = Path(args.outdir) / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {"argv": " ".join(os.sys.argv), "args": vars(args)}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[run] dir={run_dir}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    rules = load_rules(args.rules)
    set_global_seeds(args.seed)

    dummy = Env(rules, seed=123)
    obs0 = build_observation(dummy, seat=0)
    cfg = ACConfig(obs_dim=obs0.shape[0], hidden=args.hidden, lstm=args.lstm)
    model = LSTMActorCritic(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Resume (if provided) ----
    start_epoch = 0
    if args.resume:
        print(f"[ckpt] loading {args.resume}")
        state = load_ckpt(args.resume)
        msd = state.get("model_state") or state.get("model")
        if msd:
            try: model.load_state_dict(msd)
            except Exception as e: print(f"[ckpt] model load warning: {e}")
        osd = state.get("opt_state") or state.get("optimizer")
        if osd:
            try: opt.load_state_dict(osd)
            except Exception as e: print(f"[ckpt] optimizer load warning: {e}")
        # RNG
        if state.get("py_rng_state"): random.setstate(state["py_rng_state"])
        if state.get("np_rng_state"): np.random.set_state(state["np_rng_state"])  # type: ignore
        if state.get("torch_rng_state") is not None:
            try: torch.random.set_rng_state(state["torch_rng_state"])  # type: ignore
            except Exception: pass
        start_epoch = int(state.get("epoch", -1)) + 1
        print(f"[ckpt] resume at epoch {start_epoch}")

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
        ent_w = entropy_weight(epoch)
        tile_p = float(peek_probability(epoch))
        beh_p = float(behavior_prob(epoch))
        bc_w  = float(bc_weight(epoch))

        buffers: List[List] = []
        rewards_final: List[float] = []
        coverages: List[float] = []

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
            lineup_tags = args.lineup.split(",")

            jobs = []
            for _ in range(args.episodes_per_epoch):
                seed = random.randint(1, 10**9)
                jobs.append((seed, epoch, tile_p, beh_p, vs_bots_flag, lineup_tags))

            with ctx.Pool(
                processes=args.num_workers,
                initializer=_mp_init_worker,
                initargs=(model_state_cpu, cfg_dict, rules, args_dict),
            ) as pool:
                results = pool.map(_mp_run_episode, jobs)

            for buf, rew, cov in results:
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
                    lineup = args.lineup.split(",")
                    buf, rew, _ = run_episode_core(
                        args, rules, model, device,
                        lineup_tags=lineup, seed=seed, selfplay=False,
                        peek_mask_for_episode=peek_mask,
                        compute_scale=tile_p,
                        behavior_prob_use=beh_p,
                    )
                else:
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

        logprobs, ent_terms, vals, used_idx, bc_terms = a2c_forward(buffers, model, device)

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

        loss = pol_loss + val_loss + ent_loss + bc_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        avg_r = float(np.mean(rewards_final)) if rewards_final else 0.0
        cov_rate = float(np.mean(coverages)) if coverages else 0.0

        print(f"[epoch {epoch+1}/{args.epochs}] eps={len(rewards_final)} "
              f"avg_reward={avg_r:.2f} loss={loss.item():.3f} pol={pol_loss.item():.3f} "
              f"val={val_loss.item():.3f} ent={(-ent_loss).item():.3f} bc={bc_loss.item():.3f} "
              f"tile_p={tile_p:.3f} peek_cov={cov_rate:.2f} beh_p={beh_p:.2f} bc_w={bc_w:.3f}")

        _maybe_save(epoch)

# ---------------------------- CLI ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--lineup", default="hybrid,hybrid,hybrid")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--episodes-per-epoch", type=int, default=384)
    ap.add_argument("--vs-bots-epochs", type=int, default=60)
    ap.add_argument("--max-draws", type=int, default=700)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lstm", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.25)
    ap.add_argument("--shaping-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234567)

    # Teacher
    ap.add_argument("--oracle-rollouts", type=int, default=64)
    ap.add_argument("--oracle-horizon", type=int, default=48)
    ap.add_argument("--oracle-topN", type=int, default=16)
    # Oracle gating
    og = ap.add_mutually_exclusive_group()
    og.add_argument("--oracle-only-vsbots", dest="oracle_only_vsbots", action="store_true",
                    help="Use oracle only during vs-bots epochs.")
    og.add_argument("--no-oracle-only-vsbots", dest="oracle_only_vsbots", action="store_false",
                    help="Allow oracle in all epochs.")
    ap.set_defaults(oracle_only_vsbots=False)

    ap.add_argument("--oracle-exec-prob", type=float, default=0.5)

    # Peek prob schedule
    ap.add_argument("--peek-prob0", type=float, default=1.0)
    ap.add_argument("--peek-prob-target", type=float, default=0.05)
    ap.add_argument("--peek-prob-warmup", type=int, default=5)
    ap.add_argument("--peek-prob-schedule", choices=["exp","linear"], default="exp")

    # DAgger μ schedule
    ap.add_argument("--oracle-behavior-prob", type=float, default=0.8)
    ap.add_argument("--oracle-behavior-target", type=float, default=0.2)
    ap.add_argument("--oracle-behavior-schedule", choices=["exp","linear"], default="exp")
    ap.add_argument("--oracle-behavior-schedule-warmup", type=int, default=0)

    # BC weight schedule
    ap.add_argument("--bc-weight-start", type=float, default=0.5)
    ap.add_argument("--bc-weight-final", type=float, default=0.05)
    ap.add_argument("--bc-weight-warmup", type=int, default=0)
    ap.add_argument("--bc-weight-schedule", choices=["exp","linear"], default="exp")

    # Anchored repeats
    ap.add_argument("--anchored-batches", type=int, default=4)
    ap.add_argument("--anchored-K", type=int, default=32)

    # New: persistence / eval controls
    ap.add_argument("--outdir", default="runs", help="Root folder for checkpoints/logs")
    ap.add_argument("--run-id", default=None, help="Run name (subfolder under --outdir). Default: timestamp")
    ap.add_argument("--save-every", type=int, default=1, help="Save a checkpoint every N epochs")
    ap.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume from")
    ap.add_argument("--eval-only", action="store_true", help="Load --resume and run evaluation only")
    ap.add_argument("--eval-lineup", default="hybrid,hybrid,hybrid", help="Opponent lineup for eval")
    ap.add_argument("--eval-episodes", type=int, default=64, help="Evaluation episodes")

    # New: number of CPU workers for env rollouts (0 = single-process)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Number of CPU workers for parallel env rollouts (0 = no parallelism)")

    ap.add_argument("--randomize-dealer", action="store_true", default=True)
    ap.add_argument("--skip-bad-episodes", action="store_true", default=True)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    train(args)

