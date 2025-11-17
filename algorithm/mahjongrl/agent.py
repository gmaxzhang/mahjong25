# mahjong_rl/agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from mahjongrl.model import LSTMActorCritic, ACConfig, masked_categorical
from mahjongrl.env_wrapper import build_observation, T2I
from mahjongrl import oracle as OR

@dataclass
class StepRecord:
    obs: np.ndarray
    kind: str                 # 'discard' | 'ron' | 'pung' | 'chow' | 'kong_open'|'kong_add'|'kong_closed'
    mask: np.ndarray
    choice: int               # chosen index within mask
    logp: float
    value: float
    oracle_choice: Optional[int]  # for imitation loss

class RLPolicy:
    """Implements the policy interface expected by Env, while internally running a Torch model."""
    def __init__(self, seat: int, rules: Dict, model: LSTMActorCritic, device: str = "cpu"):
        self.seat = seat
        self.rules = rules
        self.model = model.to(device)
        self.device = device
        self.hx = (torch.zeros(1,1,model.cfg.lstm, device=device),
                   torch.zeros(1,1,model.cfg.lstm, device=device))
        self.buffer: List[StepRecord] = []
        self.training_mode = True
        self.oracle_on = True

    # ---- utils ----
    def _step_net(self, obs: np.ndarray) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(obs[None,None,:]).float().to(self.device)  # [1,1,D]
        y, self.hx = self.model(x, self.hx)  # [1,1,H]
        out = self.model.step_logits_value(y[:, -1, :])  # dict of [1, A]
        return out

    # ---- decisions ----
    def pick_discard(self, env) -> str:
        obs = build_observation(env, self.seat)
        logits = self._step_net(obs)
        legal = env.legal_discards(self.seat)
        mask = np.zeros((self.model.cfg.discard_head,), dtype=np.float32)
        idxs = []
        for t in legal:
            if t in T2I:
                mask[T2I[t]] = 1.0
                idxs.append(T2I[t])
        if mask.sum() == 0:  # fallback
            return env.players[self.seat].concealed[0]

        with torch.no_grad():
            act, logp = masked_categorical(logits["discard"], torch.from_numpy(mask)[None,:].to(self.device))
            value = logits["value"].squeeze(-1).cpu().item()
        # pick concrete tile closest to act (it’s already direct index)
        inv = {v:k for k,v in T2I.items()}
        choice_idx = int(act.item())
        tile = inv.get(choice_idx, legal[0])

        # Oracle choice for imitation
        oracle_choice = None
        if self.oracle_on:
            oi = OR.oracle_discard(env, self.seat, legal)
            if oi is not None:
                oracle_choice = T2I.get(legal[oi], None)

        self.buffer.append(StepRecord(obs, "discard", mask, choice_idx, float(logp.cpu().item()), value, oracle_choice))
        return tile

    def decide_ron(self, env, tile, points, loser):
        obs = build_observation(env, self.seat)
        logits = self._step_net(obs)
        mask = np.array([1.0, 1.0], dtype=np.float32)  # yes/no always legal
        with torch.no_grad():
            act, logp = masked_categorical(logits["binary"], torch.from_numpy(mask)[None,:].to(self.device))
            value = logits["value"].squeeze(-1).cpu().item()
        choice = int(act.item())
        self.buffer.append(StepRecord(obs, "ron", mask, choice, float(logp.cpu().item()), value, oracle_choice=1))
        return bool(choice)  # 1=True accept

    def decide_pung(self, env, seat, tile):
        # Only binary: 1 accept / 0 pass, but mask to 0 if illegal.
        p = env.players[self.seat]
        legal_yes = p.concealed.count(tile) >= 2
        obs = build_observation(env, self.seat)
        logits = self._step_net(obs)
        mask = np.array([1.0, 1.0 if legal_yes else 0.0], dtype=np.float32)  # [no, yes]
        with torch.no_grad():
            act, logp = masked_categorical(logits["binary"], torch.from_numpy(mask)[None,:].to(self.device))
            value = logits["value"].squeeze(-1).cpu().item()
        choice = int(act.item())
        # Oracle: lightly biased to yes if legal
        oracle_choice = 1 if (self.oracle_on and legal_yes and OR.oracle_yesno(0.8)==1) else 0
        self.buffer.append(StepRecord(obs, "pung", mask, choice, float(logp.cpu().item()), value, oracle_choice))
        return bool(choice)

    def choose_chow(self, env, seat, tile, chow_sets):
        if not chow_sets: return None
        obs = build_observation(env, self.seat)
        logits = self._step_net(obs)
        # up to 3 options + pass last slot
        k = min(3, len(chow_sets))
        mask = np.zeros((4,), dtype=np.float32)
        for i in range(k): mask[i] = 1.0
        mask[3] = 1.0  # pass
        with torch.no_grad():
            act, logp = masked_categorical(logits["chow"], torch.from_numpy(mask)[None,:].to(self.device))
            value = logits["value"].squeeze(-1).cpu().item()
        choice = int(act.item())
        oracle_choice = None
        if self.oracle_on:
            oi = OR.oracle_chow(chow_sets, env, self.seat)
            oracle_choice = oi if oi is not None else 3  # if oracle passes

        self.buffer.append(StepRecord(obs, "chow", mask, choice, float(logp.cpu().item()), value, oracle_choice))
        return None if choice == 3 else chow_sets[choice]

    def decide_open_kong(self, env, seat, tile):  # called right after pung claim sometimes
        # Treat as binary yes/no on the *specific* tile opportunity
        obs = build_observation(env, self.seat)
        logits = self._step_net(obs)
        mask = np.array([1.0, 1.0], dtype=np.float32)
        with torch.no_grad():
            act, logp = masked_categorical(logits["binary"], torch.from_numpy(mask)[None,:].to(self.device))
            value = logits["value"].squeeze(-1).cpu().item()
        choice = int(act.item())
        self.buffer.append(StepRecord(obs, "kong_open", mask, choice, float(logp.cpu().item()), value, oracle_choice=1))
        return bool(choice)

    def decide_add_kong(self, env, seat, tile):
        obs = build_observation(env, self.seat)
        logits = self._step_net(obs)
        mask = np.array([1.0, 1.0], dtype=np.float32)
        with torch.no_grad():
            act, logp = masked_categorical(logits["binary"], torch.from_numpy(mask)[None,:].to(self.device))
            value = logits["value"].squeeze(-1).cpu().item()
        choice = int(act.item())
        self.buffer.append(StepRecord(obs, "kong_add", mask, choice, float(logp.cpu().item()), value, oracle_choice=1))
        return bool(choice)

    def decide_closed_kong(self, env, seat, candidates: List[str]):
        if not candidates: return None
        obs = build_observation(env, self.seat)
        logits = self._step_net(obs)
        # up to 4 candidates + pass
        k = min(4, len(candidates))
        mask = np.zeros((5,), dtype=np.float32)
        for i in range(k): mask[i] = 1.0
        mask[4] = 1.0  # pass
        with torch.no_grad():
            act, logp = masked_categorical(logits["kong"], torch.from_numpy(mask)[None,:].to(self.device))
            value = logits["value"].squeeze(-1).cpu().item()
        choice = int(act.item())
        oracle_choice = 4  # usually pass unless very strong—keep simple
        self.buffer.append(StepRecord(obs, "kong_closed", mask, choice, float(logp.cpu().item()), value, oracle_choice))
        return None if choice == 4 else candidates[choice]
