# algorithm/mahjongrl/pretrain_bc.py
from __future__ import annotations
import argparse, os, time, json, random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from algorithm.rules_io import load_rules
from algorithm.sim_and_train import Env
from algorithm.mahjongrl.env_wrapper import build_observation
from algorithm.mahjongrl.model import ACConfig, LSTMActorCritic


# Mapping numeric head code -> head name used by the model
# This matches the HEAD_MAP used when you generated the .npz files:
#   HEAD_MAP = {"discard":0, "binary":1, "pung":2, "chow":3, "kong":4}
IDX_TO_HEAD: Dict[int, str] = {
    0: "discard",
    1: "binary",
    2: "pung",
    3: "chow",
    4: "kong",
}


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def iter_npz_chunks(bc_dir: str):
    """
    Yield .npz chunk paths in a stable order.
    """
    bc_path = Path(bc_dir)
    files = sorted(bc_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz chunks found in {bc_dir}")
    for f in files:
        yield f


def load_ckpt(path: str) -> Dict:
    """
    Compatible with train.py's load_ckpt.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_ckpt(path: str, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_bc(args):
    # ----------------- Setup / model init -----------------
    set_global_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[bc] device = {device}")

    rules = load_rules(args.rules)

    # Detect obs_dim in exactly the same way as train.py
    dummy_env = Env(rules, seed=123)
    obs0 = build_observation(dummy_env, seat=0)
    obs_dim = int(obs0.shape[0])
    print(f"[bc] detected obs_dim = {obs_dim}")

    cfg = ACConfig(obs_dim=obs_dim, hidden=args.hidden, lstm=args.lstm)
    model = LSTMActorCritic(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    # Optional resume for BC pretrain itself
    if args.resume:
        print(f"[bc] loading checkpoint from {args.resume}")
        state = load_ckpt(args.resume)

        msd = state.get("model_state") or state.get("model")
        if msd:
            model.load_state_dict(msd)
        osd = state.get("opt_state") or state.get("optimizer")
        if osd:
            opt.load_state_dict(osd)

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
        print(f"[bc] resume from epoch {start_epoch}")

    # For sanity: we’ll validate obs_dim against the first npz chunk
    first_chunk_checked = False

    # ----------------- Training loop -----------------
    for epoch in range(start_epoch, args.epochs):
        epoch_loss_sum = 0.0
        epoch_n_samples = 0

        print(f"[bc] epoch {epoch+1}/{args.epochs}")
        for npz_path in iter_npz_chunks(args.bc_dir):
            with np.load(npz_path) as data:
                obs = data["obs"]      # shape (N, D)
                heads = data["heads"]  # shape (N,)
                acts = data["acts"]    # shape (N,)

            if not first_chunk_checked:
                D = obs.shape[1]
                if D != obs_dim:
                    raise ValueError(
                        f"[bc] obs_dim mismatch: dataset has {D}, env.build_observation gives {obs_dim}. "
                        f"Make sure simulate_dataset() used the SAME build_observation."
                    )
                first_chunk_checked = True

            N = obs.shape[0]
            if N == 0:
                continue

            # Shuffle within this chunk
            idx = np.arange(N)
            np.random.shuffle(idx)
            obs = obs[idx]
            heads = heads[idx]
            acts = acts[idx]

            # Mini-batch loop over this chunk
            for start in range(0, N, args.batch_size):
                end = min(start + args.batch_size, N)
                if start >= end:
                    continue

                batch_obs = torch.from_numpy(obs[start:end]).float().to(device)   # (B, D)
                batch_heads = torch.from_numpy(heads[start:end]).long().to(device)
                batch_acts = torch.from_numpy(acts[start:end]).long().to(device)
                B = batch_obs.size(0)

                # LSTM forward: treat these as sequences of length 1
                hx = (
                    torch.zeros(1, B, cfg.lstm, device=device),
                    torch.zeros(1, B, cfg.lstm, device=device),
                )  # (num_layers, batch, hidden)

                # Input shape for model: (batch, seq_len, obs_dim) = (B,1,D)
                y, _ = model(batch_obs.unsqueeze(1), hx)  # y: (B, 1, hidden)
                features = y.squeeze(1)                   # (B, hidden)

                heads_logits = model.step_logits_value(features)
                # heads_logits is a dict like:
                #  {"discard": (B,34), "binary": (B,2), "pung": (B,2),
                #   "chow": (B,4), "kong": (B,5), "value": (B,1), ...}

                total_loss = 0.0
                total_used = 0

                # Loop over each head type, do CE on its subset
                for head_idx, head_name in IDX_TO_HEAD.items():
                    mask = (batch_heads == head_idx)
                    if not mask.any():
                        continue

                    # Model uses "discard" as name, and the rest match directly
                    logits = heads_logits.get(head_name, None)
                    if logits is None:
                        continue

                    logits_h = logits[mask]        # (n_h, C)
                    targets_h = batch_acts[mask]   # (n_h,)

                    C = logits_h.size(-1)
                    # Some heads (e.g. chow/kong) might have more classes than we used in BC;
                    # drop labels that are out of range just in case.
                    valid_mask = targets_h < C
                    if not valid_mask.any():
                        continue

                    logits_h = logits_h[valid_mask]
                    targets_h = targets_h[valid_mask]

                    loss_h = F.cross_entropy(logits_h, targets_h)
                    n_h = targets_h.size(0)

                    total_loss = total_loss + loss_h * n_h
                    total_used += n_h

                if total_used == 0:
                    continue

                total_loss = total_loss / total_used

                opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                epoch_loss_sum += float(total_loss.item()) * total_used
                epoch_n_samples += total_used

        if epoch_n_samples > 0:
            mean_loss = epoch_loss_sum / epoch_n_samples
        else:
            mean_loss = float("nan")
        print(f"[bc] epoch {epoch+1} mean supervised loss = {mean_loss:.4f}")

    # ----------------- Save final checkpoint (train.py-compatible) -----------------
    payload = {
        "epoch": args.epochs - 1,
        "args": {
            "bc_pretrain": vars(args)
        },
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "py_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "timestamp": time.time(),
    }

    out_path = Path(args.out).resolve()
    save_ckpt(str(out_path), payload)
    print(f"[bc] saved pretraining checkpoint to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="Path to rules JSON (same as train.py)")
    ap.add_argument("--bc-dir", required=True, help="Folder containing .npz chunks (from simulate_dataset)")
    ap.add_argument("--out", default="runs/bc_pretrain.pt",
                    help="Where to save the BC checkpoint (train.py --resume will load this)")
    ap.add_argument("--epochs", type=int, default=3, help="Number of passes over the BC dataset")
    ap.add_argument("--batch-size", type=int, default=2048, help="BC mini-batch size in steps")
    ap.add_argument("--hidden", type=int, default=256,
                    help="Hidden size (must match what you'll use in train.py)")
    ap.add_argument("--lstm", type=int, default=256,
                    help="LSTM size (must match what you'll use in train.py)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=123456)
    ap.add_argument("--resume", default=None,
                    help="Optional: resume BC pretraining from an existing .pt checkpoint")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = ap.parse_args()
    train_bc(args)
