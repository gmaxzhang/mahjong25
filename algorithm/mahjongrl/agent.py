# algorithm/mahjongrl/agent.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch

from algorithm.mahjongrl.env_wrapper import build_observation
from algorithm.sim_and_train import _tile_rank_suit, _is_suit_tile, is_flower

# ----- 34-way tile classes (1..9 of b/w/t, then Winds+Dragons) -----
_TILE_ORDER = (
    [f"{r}b" for r in range(1,10)] +
    [f"{r}w" for r in range(1,10)] +
    [f"{r}t" for r in range(1,10)] +
    ["E","S","W","N","C","F","B"]
)
_TILE_TO_IDX = {t:i for i,t in enumerate(_TILE_ORDER)}

# Map env "kind" labels to model heads
_HEAD_MAP = {
    "discard": "discard",
    "chow":    "chow",
    "pung":    "pung",
    "ron":     "binary",
    "kong":    "kong",
    "binary":  "binary",
}

@dataclass
class StepRec:
    obs: np.ndarray
    kind: str
    choice: Optional[int] = None
    legal_idx: Optional[List[int]] = None
    mask: Optional[np.ndarray] = None
    teacher_idx: Optional[int] = None
    teacher_choice: Optional[Any] = None

# ---------------- Utilities ----------------
def _apply_action_mask(logits: torch.Tensor, legal_idx: List[int]) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    C = logits.size(-1)
    if not legal_idx:
        return logits
    mask = torch.full((C,), float("-inf"), device=logits.device, dtype=logits.dtype)
    idx = torch.as_tensor(legal_idx, device=logits.device, dtype=torch.long)
    mask[idx] = 0.0
    return logits + mask.unsqueeze(0)

def _tile_to_idx(tile: Any) -> Optional[int]:
    if tile is None:
        return None
    if isinstance(tile, int):
        if 0 <= tile < 34:
            return tile
        if 0 <= tile < 136:
            return tile % 34
    if isinstance(tile, str):
        return _TILE_TO_IDX.get(tile)
    for attr in ("short", "code", "name", "label", "s"):
        if hasattr(tile, attr):
            v = getattr(tile, attr)
            if isinstance(v, str):
                k = _TILE_TO_IDX.get(v)
                if k is not None:
                    return k
            if isinstance(v, int) and 0 <= v < 34:
                return v
    if hasattr(tile, "rank") and hasattr(tile, "suit"):
        try:
            r = int(getattr(tile, "rank"))
            s = str(getattr(tile, "suit"))[:1].lower()
            return _TILE_TO_IDX.get(f"{r}{s}")
        except Exception:
            pass
    return None

def _hand_class_counts(env, seat: int) -> Tuple[List[int], Dict[int, List[int]]]:
    hand = getattr(env.players[seat], "concealed", [])
    pos_by_cls: Dict[int, List[int]] = {}
    for pos, tile in enumerate(hand):
        k = _tile_to_idx(tile)
        if k is None:
            continue
        pos_by_cls.setdefault(k, []).append(pos)
    legal = sorted(pos_by_cls.keys())
    return legal, pos_by_cls

# def _extract_legal_from_env(env, head: str, C: int, seat: int) -> List[int]:
#     # 1) Generic fields the env might expose
#     for name in ("legal_idx", "valid_idx", "legal_indices", "valid_indices"):
#         if hasattr(env, name):
#             try:
#                 li = [int(x) for x in list(getattr(env, name))]
#                 li = [x for x in li if 0 <= x < C]
#                 if li:
#                     return li
#             except Exception:
#                 pass
#     # 2) Head-specific fields
#     for name in (f"legal_idx_{head}", f"{head}_legal_idx", f"legal_{head}_idx"):
#         if hasattr(env, name):
#             try:
#                 li = [int(x) for x in list(getattr(env, name))]
#                 li = [x for x in li if 0 <= x < C]
#                 if li:
#                     return li
#             except Exception:
#                 pass
#     # 3) Common helpers by head
#     if head == "binary":
#         return [0, 1][:C]
#     if head == "pung":
#         return [0, 1][:C]
#     if head == "discard":
#         li, _ = _hand_class_counts(env, seat)
#         if li:
#             return li
#     # Try a few env-specific helpers (best-effort)
#     try:
#         if head == "chow":
#             for m in ("legal_chow_indices","get_legal_chow_indices","chow_legal_idx","legal_chows"):
#                 if hasattr(env, m):
#                     v = getattr(env, m)
#                     li = v(seat) if callable(v) else v
#                     li = [int(x) for x in li if 0 <= int(x) < C]
#                     if li:
#                         return li
#         if head == "kong":
#             for m in ("legal_kong_indices","get_legal_kong_indices","kong_legal_idx","legal_kongs"):
#                 if hasattr(env, m):
#                     v = getattr(env, m)
#                     li = v(seat) if callable(v) else v
#                     li = [int(x) for x in li if 0 <= int(x) < C]
#                     if li:
#                         return li
#         if head == "pung":
#         # Try env-specific helper first
#             for m in ("legal_pung_indices", "get_legal_pung_indices", "pung_legal_idx", "legal_pungs"):
#                 if hasattr(env, m):
#                     v = getattr(env, m)
#                     li = v(seat) if callable(v) else v
#                     li = [int(x) for x in li if 0 <= int(x) < C]
#                     if li:
#                         return li
#             #return [0, 1][:C]

#     except Exception:
#         pass
#     # 4) Fallback: allow all
#     return list(range(C))

def _extract_legal_from_env(env, head: str, C: int, seat: int) -> List[int]:
    """
    Map env state -> legal logits indices for each head.

    Heads:
      - 'discard': 34-class discard over non-flower tiles in concealed hand.
      - 'binary': generic yes/no: [0,1][:C]
      - 'pung':   yes/no claim on a discard: [0,1][:C]
      - 'chow':   0 = PASS; 1..k = which chow set (depends on last_discard & concealed)
      - 'kong':   closed/add kongs; 0 = PASS; 1..k = which candidate (for closed kongs);
                  for add-kong we only care about 0 (no) vs nonzero (yes).
    """
    # ---- 0) generic env-wide fields (keep this, harmless) ----
    for name in ("legal_idx", "valid_idx", "legal_indices", "valid_indices"):
        if hasattr(env, name):
            try:
                li = [int(x) for x in list(getattr(env, name))]
                li = [x for x in li if 0 <= x < C]
                if li:
                    return li
            except Exception:
                pass

    # ---- 1) head-specific env fields (future-proofing) ----
    for name in (f"legal_idx_{head}", f"{head}_legal_idx", f"legal_{head}_idx"):
        if hasattr(env, name):
            try:
                li = [int(x) for x in list(getattr(env, name))]
                li = [x for x in li if 0 <= x < C]
                if li:
                    return li
            except Exception:
                pass

    # ---- 2) heads we know how to derive from this Env ----

    # Binary (RON yes/no, etc.)
    if head == "binary":
        return [0, 1][:C]

    # PUNG is modeled as binary 0=no, 1=yes in RLPolicy.decide_pung
    if head == "pung":
        return [0, 1][:C]

    # DISCARD: use the same 34-class mapping as elsewhere
    if head == "discard":
        li, _ = _hand_class_counts(env, seat)
        if li:
            return li

    # CHOW: mirror Env._chow_claim logic
    if head == "chow":
        try:
            last = getattr(env, "last_discard", None)
            if not last:
                # No discard on table -> can't chow, only pass
                return [0]

            discarder, tile = last

            # Only the player immediately after discarder can chow
            if (discarder + 1) % 4 != seat:
                return [0]

            if not _is_suit_tile(tile):
                return [0]

            r, suit = _tile_rank_suit(tile)
            p = env.players[seat]

            sets = []
            for a, b in [(r - 2, r - 1), (r - 1, r + 1), (r + 1, r + 2)]:
                if 1 <= a <= 9 and 1 <= b <= 9:
                    A, B = f"{a}{suit}", f"{b}{suit}"
                    if A in p.concealed and B in p.concealed:
                        sets.append((A, B))

            # Debug if you want:
            # print(f"[chow-legal] seat={seat} last={last} sets={sets}")

            if not sets:
                return [0]

            # Match Env._chow_claim semantics:
            # idx = 0        -> pass
            # idx in [1..k]  -> choose sets[idx-1]
            max_idx = min(len(sets), C - 1)
            return list(range(max_idx + 1))  # [0, 1, ..., max_idx]
        except Exception as e:
            # If something goes wrong, don't crash the env; just be permissive.
            # print(f"[chow-legal-error] {e}")
            return list(range(C))

    # KONG: mirror closed-kong candidate logic; add-kong treats any nonzero as "yes"
    if head == "kong":
        try:
            p = env.players[seat]
            # Same filter as Env._maybe_closed_kongs
            cnt = Counter([t for t in p.concealed if not is_flower(t)])
            candidates = [t for t, c in cnt.items() if c >= 4]

            # Debug if you want:
            # print(f"[kong-legal] seat={seat} candidates={candidates}")

            if not candidates:
                # No closed-kong candidate; still allow "no kong"
                return [0]

            # For closed kongs:
            #   idx = 0         -> pass (Env._maybe_closed_kongs treats it that way)
            #   idx in [1..k]   -> candidates[idx-1]
            max_idx = min(len(candidates), C - 1)
            return list(range(max_idx + 1))
        except Exception as e:
            # print(f"[kong-legal-error] {e}")
            # Fallback: binary "no/yes"
            return [0, 1][:C]

    # ---- 3) as a last resort, look for env-specific helpers (future envs) ----
    try:
        if head == "chow":
            for m in ("legal_chow_indices", "get_legal_chow_indices", "chow_legal_idx", "legal_chows"):
                if hasattr(env, m):
                    v = getattr(env, m)
                    li = v(seat) if callable(v) else v
                    li = [int(x) for x in li if 0 <= int(x) < C]
                    if li:
                        return li

        if head == "kong":
            for m in ("legal_kong_indices", "get_legal_kong_indices", "kong_legal_idx", "legal_kongs"):
                if hasattr(env, m):
                    v = getattr(env, m)
                    li = v(seat) if callable(v) else v
                    li = [int(x) for x in li if 0 <= int(x) < C]
                    if li:
                        return li

        if head == "pung":
            for m in ("legal_pung_indices", "get_legal_pung_indices", "pung_legal_idx", "legal_pungs"):
                if hasattr(env, m):
                    v = getattr(env, m)
                    li = v(seat) if callable(v) else v
                    li = [int(x) for x in li if 0 <= int(x) < C]
                    if li:
                        return li
    except Exception:
        pass

    # ---- 4) final fallback: allow all indices for this head ----
    return list(range(C))



# ---------------- Policy ----------------
class RLPolicy:
    """
    RL-driven Mahjong policy that:
      • keeps its own LSTM hidden state,
      • calls an attached teacher/oracle picker(env, seat, legal_idx, head) → idx|None,
      • guarantees returning a legal action,
      • records each step for BC/A2C training.
    """
    def __init__(self, seat: int, rules: Dict, model, device: str = "cpu"):
        self.seat = int(seat)
        self.rules = rules
        self.model = model
        self.device = device

        self.oracle_picker: Optional[Callable[[Any,int,List[int],str], Optional[int]]] = None
        self.use_oracle: bool = True
        self.buffer: List[StepRec] = []

        # LSTM hidden state (initialized lazily/zeroed)
        self._hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._init_hidden()

        # short debug
        self._dbg_calls_left = 12
        self._src_counts = {"T": 0, "pi": 0}


    # ---- oracle/teacher plumbing ----
    def enable_oracle(self, picker: Callable):
        self.oracle_picker = picker
        self.use_oracle = True
    def set_oracle(self, picker: Callable):
        self.enable_oracle(picker)
    def set_teacher(self, picker: Callable):
        self.enable_oracle(picker)
    def disable_oracle(self):
        self.use_oracle = False

    # Aliases some frameworks/trainers might set
    @property
    def teacher_picker(self): return self.oracle_picker
    @teacher_picker.setter
    def teacher_picker(self, picker): self.oracle_picker = picker
    @property
    def oracle(self): return self.oracle_picker
    @oracle.setter
    def oracle(self, picker): self.oracle_picker = picker
    @property
    def teacher(self): return self.oracle_picker
    @teacher.setter
    def teacher(self, picker): self.oracle_picker = picker
    @property
    def advice_fn(self): return self.oracle_picker
    @advice_fn.setter
    def advice_fn(self, picker): self.oracle_picker = picker
    @property
    def picker(self): return self.oracle_picker
    @picker.setter
    def picker(self, picker): self.oracle_picker = picker

    # ---- LSTM state ----
    def _init_hidden(self):
        H = getattr(getattr(self.model, "cfg", None), "lstm", 256)
        self._hx = (torch.zeros(1,1,H, device=self.device),
                    torch.zeros(1,1,H, device=self.device))
    def reset_hidden(self):
        self._init_hidden()

    # helper for idx
        
        # ---- Extract legal indices for each action head ----
    def _extract_legal_from_env(self, env, head: str, C: int) -> List[int]:
        """
        Map env state → list of legal action indices for this head.
        Guarantees non-empty output so teacher and model both have valid choices.
        """
        seat = self.seat

        # DISCARD: tiles in hand (0–33 mapping)
        if head == "discard":
            legal = set()
            hand = getattr(env.players[seat], "concealed", [])
            for t in hand:
                try:
                    idx = _tile_to_idx(t)
                except Exception:
                    idx = None
                if idx is not None and 0 <= idx < C:
                    legal.add(idx)
            return sorted(legal) if legal else list(range(min(C, 34)))

        # PUNG: yes/no (binary)
        if head == "pung":
            for name in ("legal_pung_indices", "get_legal_pung_indices", "pung_legal_idx"):
                if hasattr(env, name):
                    v = getattr(env, name)
                    li = v(seat) if callable(v) else v
                    try:
                        li = [int(i) for i in li if 0 <= int(i) < C]
                    except Exception:
                        li = []
                    if li:
                        return li
            return [0, 1][:C]

        # CHOW: 0 = pass, 1..k = options
        if head == "chow":
            for name in ("legal_chow_indices", "get_legal_chow_indices", "chow_legal_idx", "legal_chows"):
                if hasattr(env, name):
                    v = getattr(env, name)
                    li = v(seat) if callable(v) else v
                    try:
                        li = [int(i) for i in li if 0 <= int(i) < C]
                    except Exception:
                        li = []
                    if li:
                        return li
            return [0]

        # KONG: 0 = pass, 1..k = options
        if head == "kong":
            for name in ("legal_kong_indices", "get_legal_kong_indices", "kong_legal_idx", "legal_kongs"):
                if hasattr(env, name):
                    v = getattr(env, name)
                    li = v(seat) if callable(v) else v
                    try:
                        li = [int(i) for i in li if 0 <= int(i) < C]
                    except Exception:
                        li = []
                    if li:
                        return li
            return [0]

        # BINARY: ron/tsumo (yes/no)
        if head == "binary":
            return [0, 1][:C]

        # Default: assume all legal
        return list(range(C))


    # ---- core decision path ----
    # def _decide_index(self, env, head: str) -> Tuple[int, List[int], Optional[int], np.ndarray]:
    #     # 1) Build observation
    #     obs_np = build_observation(env, self.seat)
    #     obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(self.device)[None, None, :]

    #     # 2) Forward with LSTM state (fallback if model ignores hx)
    #     with torch.no_grad():
    #         try:
    #             y, self._hx = self.model(obs, self._hx)   # preferred
    #         except TypeError:
    #             y, _ = self.model(obs)                     # stateless fallback
    #         heads = self.model.step_logits_value(y.squeeze(0))

    #     # 3) Pick head logits (graceful fallback to 1-class zero logits)
    #     logits = heads.get(head, None)
    #     if logits is None:
    #         logits = torch.zeros(1, 1, device=self.device)
    #     C = int(logits.size(-1))

    #     # 4) Legal set + mask
    #     legal_idx = self._extract_legal_from_env(env, head, C)
    #     masked = _apply_action_mask(logits[0:1, :], legal_idx)

    #             # --- Sanity check / debug ---
    #     # if self._dbg_calls_left > 0:
    #     #     print(f"[legal-debug] seat={self.seat} head={head} | C={C} | legal={legal_idx[:10]} (len={len(legal_idx)})")

    #     # 5) Teacher/oracle (if available)
    #     t_idx: Optional[int] = None
    #     # print(f"[rlpolicy->teacher] use_oracle={self.use_oracle}, "
    #     #     f"callable={callable(self.oracle_picker)}, head={head}, "
    #     #     f"len(legal_idx)={len(legal_idx)}")
    #     if self.use_oracle and callable(self.oracle_picker):
    #         try:
    #             t_idx = self.oracle_picker(env, self.seat, list(legal_idx), head)
    #         except Exception:
    #             t_idx = None


    #     # 6) Choose index (teacher first, else masked argmax), force legal
    #     idx = int(torch.argmax(masked, dim=-1).item()) if t_idx is None else int(t_idx)
    #     if legal_idx and idx not in legal_idx:
    #         idx = int(legal_idx[0])

    #     # 7) Debug line (limited)
    #         # 7) Debug line (limited)
    #     if self._dbg_calls_left > 0:
    #         self._dbg_calls_left -= 1
    #         src = "T" if t_idx is not None else "π"  # teacher vs policy

    #         # Human-readable interpretation of idx
    #         extra = ""
    #         if head == "discard":
    #             tile = self._idx_to_hand_tile(env, idx)
    #             extra = f"tile={tile}"
    #         elif head in ("binary", "pung", "kong"):
    #             yn = "YES" if idx == 1 else "NO"
    #             extra = f"decision={yn}"

    #             # If this is a kong decision, also show candidates if available
    #             if head == "kong" and hasattr(env, "_pending_kong_candidates"):
    #                 try:
    #                     cand = getattr(env, "_pending_kong_candidates", {}).get(self.seat, None)
    #                 except Exception:
    #                     cand = None
    #                 if cand:
    #                     extra += f" candidates={cand}"

    #         # print(
    #         #     f"[act-debug] seat={self.seat} head={head:<7} C={C:<2} "
    #         #     f"legal={legal_idx} idx={idx} {extra} src={src}"
    #         # )


    #     # 8) Log for training
    #     try:
    #         self.buffer.append(
    #             StepRec(
    #                 obs=obs_np,
    #                 kind=head,
    #                 choice=idx,
    #                 legal_idx=list(legal_idx) if legal_idx else None,
    #                 mask=masked.detach().cpu().numpy().reshape(-1),
    #                 teacher_idx=t_idx
    #             )
    #         )
    #     except Exception:
    #         pass

    #     pi_idx = int(torch.argmax(masked, dim=-1).item())

    #     if self.use_oracle and callable(self.oracle_picker):
    #         try:
    #             t_idx = self.oracle_picker(env, self.seat, list(legal_idx), head)
    #         except Exception:
    #             t_idx = None

    #     if t_idx is not None:
    #         idx = int(t_idx)
    #         src = "T"
    #     else:
    #         idx = pi_idx
    #         src = "pi"

    #     if legal_idx and idx not in legal_idx:
    #         idx = int(legal_idx[0])

    #     self._src_counts[src] += 1

    #     if self._dbg_calls_left > 0:
    #         self._dbg_calls_left -= 1
    #         print(f"[rlpolicy] head={head:<7} idx={idx} t_idx={t_idx} pi_idx={pi_idx} "
    #             f"legal={legal_idx} src={src}")

    #     return idx, legal_idx, t_idx, masked.detach().cpu().numpy().reshape(-1)

    def _decide_index(self, env, head: str) -> Tuple[int, List[int], Optional[int], np.ndarray]:
        """
        Core decision helper.

        - Builds observation and runs model with internal LSTM state.
        - Computes masked logits for the requested head.
        - Optionally consults teacher/oracle exactly once.
        - Returns a legal index and logs what was actually used.
        """
        # 1) Build observation
        obs_np = build_observation(env, self.seat)
        obs = torch.from_numpy(
            np.asarray(obs_np, dtype=np.float32)
        ).to(self.device)[None, None, :]   # [1,1,D]

        # 2) Forward through model with LSTM state
        with torch.no_grad():
            try:
                y, self._hx = self.model(obs, self._hx)   # [1,1,H]
            except TypeError:
                y, _ = self.model(obs)
            # use last time step; shape [1,H]
            y_t = y[:, -1, :]
            heads = self.model.step_logits_value(y_t)

        # 3) Pick head logits; graceful fallback if missing
        logits = heads.get(head, None)
        if logits is None:
            logits = torch.zeros(1, 1, device=self.device)  # [1,1]
        C = int(logits.size(-1))

        # 4) Legal indices + masked logits
        legal_idx = self._extract_legal_from_env(env, head, C)
        if not legal_idx:
            # ultra-safety: if something went wrong, fall back to all classes
            legal_idx = list(range(C))

        masked_logits = _apply_action_mask(logits, legal_idx)  # [1,C]

        # 5) Policy argmax (π)
        pi_idx = int(torch.argmax(masked_logits, dim=-1).item())

        # 6) Optional teacher/oracle (called ONCE)
        t_idx: Optional[int] = None
        if self.use_oracle and callable(self.oracle_picker):
            try:
                t_idx = self.oracle_picker(env, self.seat, list(legal_idx), head)
            except Exception:
                t_idx = None

        # 7) Final choice: teacher if provided, else π
        if t_idx is not None:
            idx = int(t_idx)
            src = "T"
        else:
            idx = pi_idx
            src = "pi"

        # 8) Force final idx to be legal
        if legal_idx and idx not in legal_idx:
            idx = int(legal_idx[0])

        # 9) Log for training with the *final* action
        try:
            self.buffer.append(
                StepRec(
                    obs=obs_np,
                    kind=head,
                    choice=idx,
                    legal_idx=list(legal_idx) if legal_idx else None,
                    # NOTE: this is masked *logits*, not a boolean mask
                    mask=masked_logits.detach().cpu().numpy().reshape(-1),
                    teacher_idx=t_idx,
                )
            )
        except Exception:
            pass

        # 10) Debug and stats
        self._src_counts[src] += 1
        if self._dbg_calls_left > 0:
            self._dbg_calls_left -= 1
            # print(
            #     f"[rlpolicy] head={head:<7} idx={idx} t_idx={t_idx} "
            #     f"pi_idx={pi_idx} legal={legal_idx} src={src}"
            # )

        return idx, legal_idx, t_idx, masked_logits.detach().cpu().numpy().reshape(-1)


    # ---- helpers to map class -> actual tile for discards ----
    def _idx_to_hand_tile(self, env, cls_idx: int):
        hand = getattr(env.players[self.seat], "concealed", [])
        for tile in hand:
            if _tile_to_idx(tile) == cls_idx:
                return tile
        return hand[0] if hand else None
    
        # ---- Claim decisions (used by Env) ----
    def decide_pung(self, env, seat, tile):
        idx, legal, _, _ = self._decide_index(env, "pung")
        # binary heads: 1 = yes, 0 = no
        return int(idx == 1)

    # def choose_chow(self, env, seat, tile, chow_sets):
    #     idx, legal, _, _ = self._decide_index(env, "chow")
    #     # Return the chosen chow set if available
    #     if not chow_sets:
    #         return None
    #     if idx is not None and 0 <= idx < len(chow_sets):
    #         return chow_sets[idx]
    #     # fallback
    #     return chow_sets[0] if chow_sets else None

    def choose_chow(self, env, seat, tile, chow_sets):
        if not chow_sets:
            return None

        idx, legal, _, _ = self._decide_index(env, "chow")

        # print(
        #     f"[chow-rlpolicy] seat={self.seat} tile={tile} "
        #     f"sets={chow_sets} idx={idx} legal={legal}"
        # )

        # 0 = pass
        if idx == 0:
            return None

        choice = idx - 1
        if 0 <= choice < len(chow_sets):
            return chow_sets[choice]

        # safety: if something weird happens, just pass
        return None

    def decide_closed_kong(self, env, seat, candidates):
        if not candidates:
            return None

        idx, legal, _, _ = self._decide_index(env, "kong")
        # 0 = pass
        if idx == 0:
            return None

        choice = idx - 1
        if 0 <= choice < len(candidates):
            return candidates[choice]
        return None


    def decide_open_kong(self, env, seat, tile):
        idx, legal, _, _ = self._decide_index(env, "kong")
        return int(idx == 1)

    def decide_add_kong(self, env, seat, tile):
        idx, legal, _, _ = self._decide_index(env, "kong")
        return int(idx == 1)

    # def decide_closed_kong(self, env, seat, candidates):
    #     idx, legal, _, _ = self._decide_index(env, "kong")
    #     if candidates and 0 <= idx < len(candidates):
    #         return candidates[idx]
    #     return None

    def decide_ron(self, env, tile, points, loser):
        idx, legal, _, _ = self._decide_index(env, "binary")
        return int(idx == 1)

    # # ---------------- env-facing methods ----------------
    def pick_discard(self, env) -> Any:
        idx, _, _, _ = self._decide_index(env, "discard")
        #print(f"[rlpolicy-call] seat={self.seat}")
        return self._idx_to_hand_tile(env, idx)

    # def pick_chow(self, env) -> int:
    #     idx, _, _, _ = self._decide_index(env, "chow")
    #     return int(idx)

    # def pick_kong(self, env) -> int:
    #     idx, _, _, _ = self._decide_index(env, "kong")
    #     return int(idx)

    # def pick_binary(self, env) -> int:
    #     idx, legal, _, _ = self._decide_index(env, "binary")
    #     if not legal:
    #         return int(idx % 2)
    #     return int(1 if 1 in legal and idx == 1 else 0)

    # # Aliases some envs may call
    # def pick_ron(self, env) -> int:  # yes/no
    #     return self.pick_binary(env)
    # def pick_pung(self, env) -> int: # yes/no
    #     return self.pick_binary(env)

    # Optional generic aliases
    def act(self, env, **kw):           return self.pick_discard(env)
    def step(self, env, **kw):          return self.pick_discard(env)
    def select_action(self, env, **kw): return self.pick_discard(env)
    def policy(self, env, **kw):        return self.pick_discard(env)


