from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACConfig:
    def __init__(self, obs_dim: int, hidden: int = 256, lstm: int = 256):
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.lstm = lstm
        # Head sizes (max). We will mask at runtime:
        self.discard_head = 34     # choose a tile kind; mask illegal
        self.binary_head  = 2      # yes/no (ron, pung)
        self.chow_head    = 4      # up to 3 chow options + pass
        self.kong_head    = 5      # up to 4 candidates + pass


class LSTMActorCritic(nn.Module):
    def __init__(self, cfg: ACConfig):
        super().__init__()
        self.cfg = cfg

        # Build encoder dynamically from obs_dim
        self.enc = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(cfg.hidden, cfg.lstm, batch_first=True)

        H = cfg.lstm
        self.head_discard = nn.Linear(H, cfg.discard_head)
        self.head_binary  = nn.Linear(H, cfg.binary_head)
        self.head_chow    = nn.Linear(H, cfg.chow_head)
        self.head_kong    = nn.Linear(H, cfg.kong_head)
        self.v = nn.Linear(H, 1)

    def forward(self, obs_seq: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        obs_seq: [B,T,obs_dim]
        hx: optional (h0,c0) with shape [1,B,H]. If None, zeros are used.
        """
        B, T, D = obs_seq.shape

        # Rebuild encoder dynamically if observation size changes (for safety)
        first_layer = self.enc[0]
        if D != first_layer.in_features:
            print(f"[warn] adapting encoder input: {first_layer.in_features}→{D}")
            new_fc = nn.Linear(D, self.cfg.hidden).to(obs_seq.device)
            self.enc[0] = new_fc

        z = self.enc(obs_seq.view(B * T, D)).view(B, T, -1)

        if hx is None:
            H = self.cfg.lstm
            h0 = torch.zeros(1, B, H, device=obs_seq.device)
            c0 = torch.zeros(1, B, H, device=obs_seq.device)
            hx = (h0, c0)

        y, new_hx = self.lstm(z, hx)
        return y, new_hx

    def step_logits_value(self, y_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return policy logits and value predictions from one timestep."""
        return {
            "discard": self.head_discard(y_t),
            "binary":  self.head_binary(y_t),
            "chow":    self.head_chow(y_t),
            "kong":    self.head_kong(y_t),
            "value":   self.v(y_t)
        }


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample from masked categorical distribution."""
    masked = logits + (mask + 1e-8).log()
    pi = F.softmax(masked, dim=-1)
    act = torch.multinomial(pi, num_samples=1).squeeze(-1)
    logp = torch.log(pi.gather(-1, act.unsqueeze(-1)).squeeze(-1) + 1e-12)
    return act, logp
