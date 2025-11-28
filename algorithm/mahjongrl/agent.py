# algorithm/mahjongrl/agent.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch

from algorithm.mahjongrl.env_wrapper import build_observation

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
    "pung":    "binary",
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

def _extract_legal_from_env(env, head: str, C: int, seat: int) -> List[int]:
    # 1) Generic fields the env might expose
    for name in ("legal_idx", "valid_idx", "legal_indices", "valid_indices"):
        if hasattr(env, name):
            try:
                li = [int(x) for x in list(getattr(env, name))]
                li = [x for x in li if 0 <= x < C]
                if li:
                    return li
            except Exception:
                pass
    # 2) Head-specific fields
    for name in (f"legal_idx_{head}", f"{head}_legal_idx", f"legal_{head}_idx"):
        if hasattr(env, name):
            try:
                li = [int(x) for x in list(getattr(env, name))]
                li = [x for x in li if 0 <= x < C]
                if li:
                    return li
            except Exception:
                pass
    # 3) Common helpers by head
    if head == "binary":
        return [0, 1][:C]
    if head == "discard":
        li, _ = _hand_class_counts(env, seat)
        if li:
            return li
    # Try a few env-specific helpers (best-effort)
    try:
        if head == "chow":
            for m in ("legal_chow_indices","get_legal_chow_indices","chow_legal_idx","legal_chows"):
                if hasattr(env, m):
                    v = getattr(env, m)
                    li = v(seat) if callable(v) else v
                    li = [int(x) for x in li if 0 <= int(x) < C]
                    if li:
                        return li
        if head == "kong":
            for m in ("legal_kong_indices","get_legal_kong_indices","kong_legal_idx","legal_kongs"):
                if hasattr(env, m):
                    v = getattr(env, m)
                    li = v(seat) if callable(v) else v
                    li = [int(x) for x in li if 0 <= int(x) < C]
                    if li:
                        return li
    except Exception:
        pass
    # 4) Fallback: allow all
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

    # ---- core decision path ----
    def _decide_index(self, env, head: str) -> Tuple[int, List[int], Optional[int], np.ndarray]:
        # 1) Build observation
        obs_np = build_observation(env, self.seat)
        obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(self.device)[None, None, :]

        # 2) Forward with LSTM state (fallback if model ignores hx)
        with torch.no_grad():
            try:
                y, self._hx = self.model(obs, self._hx)   # preferred
            except TypeError:
                y, _ = self.model(obs)                     # stateless fallback
            heads = self.model.step_logits_value(y.squeeze(0))

        # 3) Pick head logits (graceful fallback to 1-class zero logits)
        logits = heads.get(head, None)
        if logits is None:
            logits = torch.zeros(1, 1, device=self.device)
        C = int(logits.size(-1))

        # 4) Legal set + mask
        legal_idx = _extract_legal_from_env(env, head, C, self.seat)
        masked = _apply_action_mask(logits[0:1, :], legal_idx)

        # 5) Teacher/oracle (if available)
        t_idx: Optional[int] = None
        if self.use_oracle and callable(self.oracle_picker):
            try:
                t_idx = self.oracle_picker(env, self.seat, list(legal_idx), head)
            except Exception:
                t_idx = None

        # 6) Choose index (teacher first, else masked argmax), force legal
        idx = int(torch.argmax(masked, dim=-1).item()) if t_idx is None else int(t_idx)
        if legal_idx and idx not in legal_idx:
            idx = int(legal_idx[0])

        # 7) Debug line (limited)
        if self._dbg_calls_left > 0:
            self._dbg_calls_left -= 1
            src = "T" if t_idx is not None else "π"
            #print(f"[rlpolicy] head={head:<7} C={C:<2} legal={len(legal_idx):<2} idx={idx} src={src}")

        # 8) Log for training
        try:
            self.buffer.append(
                StepRec(
                    obs=obs_np,
                    kind=head,
                    choice=idx,
                    legal_idx=list(legal_idx) if legal_idx else None,
                    mask=masked.detach().cpu().numpy().reshape(-1),
                    teacher_idx=t_idx
                )
            )
        except Exception:
            pass

        return idx, legal_idx, t_idx, masked.detach().cpu().numpy().reshape(-1)

    # ---- helpers to map class -> actual tile for discards ----
    def _idx_to_hand_tile(self, env, cls_idx: int):
        hand = getattr(env.players[self.seat], "concealed", [])
        for tile in hand:
            if _tile_to_idx(tile) == cls_idx:
                return tile
        return hand[0] if hand else None

    # ---------------- env-facing methods ----------------
    def pick_discard(self, env) -> Any:
        idx, _, _, _ = self._decide_index(env, "discard")
        return self._idx_to_hand_tile(env, idx)

    def pick_chow(self, env) -> int:
        idx, _, _, _ = self._decide_index(env, "chow")
        return int(idx)

    def pick_kong(self, env) -> int:
        idx, _, _, _ = self._decide_index(env, "kong")
        return int(idx)

    def pick_binary(self, env) -> int:
        idx, legal, _, _ = self._decide_index(env, "binary")
        if not legal:
            return int(idx % 2)
        return int(1 if 1 in legal and idx == 1 else 0)

    # Aliases some envs may call
    def pick_ron(self, env) -> int:  # yes/no
        return self.pick_binary(env)
    def pick_pung(self, env) -> int: # yes/no
        return self.pick_binary(env)

    # Optional generic aliases
    def act(self, env, **kw):           return self.pick_discard(env)
    def step(self, env, **kw):          return self.pick_discard(env)
    def select_action(self, env, **kw): return self.pick_discard(env)
    def policy(self, env, **kw):        return self.pick_discard(env)
