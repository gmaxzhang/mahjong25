# algorithm/mahjongrl/pretrain_value_from_returns.py
from __future__ import annotations
import argparse, os, time, random
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


def iter_npz_chunks(value_dir: str):
    """
    Yield .npz chunk paths in a stable order from the given value_dir.
    Each chunk is expected to contain 'obs' and 'G' arrays.
    """
    value_path = Path(value_dir)
    files = sorted(value_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz chunks found in {value_dir}")
    for f in files:
        yield f


def load_ckpt(path: str) -> Dict:
    """
    Compatible with train.py's load_ckpt (same as in pretrain_bc.py).
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_ckpt(path: str, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_value_from_returns(args):
    # ----------------- Setup / model init -----------------
    set_global_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[value-pretrain] device = {device}")

    rules = load_rules(args.rules)

    # Detect obs_dim exactly as train.py does
    dummy_env = Env(rules, seed=123)
    obs0 = build_observation(dummy_env, seat=0)
    obs_dim = int(obs0.shape[0])
    print(f"[value-pretrain] detected obs_dim = {obs_dim}")

    cfg = ACConfig(obs_dim=obs_dim, hidden=args.hidden, lstm=args.lstm)
    model = LSTMActorCritic(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    # Optional resume
    if args.resume:
        print(f"[value-pretrain] loading checkpoint from {args.resume}")
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
        print(f"[value-pretrain] resume from epoch {start_epoch}")

    first_chunk_checked = False

    # ----------------- Training loop -----------------
    for epoch in range(start_epoch, args.epochs):
        epoch_loss_sum = 0.0
        epoch_n_samples = 0

        print(f"[value-pretrain] epoch {epoch+1}/{args.epochs}")
        for npz_path in iter_npz_chunks(args.value_dir):
            with np.load(npz_path) as data:
                obs = data["obs"]  # (N, D)
                G   = data["G"]    # (N,)

            if not first_chunk_checked:
                D = obs.shape[1]
                if D != obs_dim:
                    raise ValueError(
                        f"[value-pretrain] obs_dim mismatch: dataset has {D}, "
                        f"env.build_observation gives {obs_dim}. "
                        f"Make sure bc_dataset_returns_step0b used the SAME build_observation."
                    )
                first_chunk_checked = True

            N = obs.shape[0]
            if N == 0:
                continue

            # Shuffle within this chunk
            idx = np.arange(N)
            np.random.shuffle(idx)
            obs = obs[idx]
            G   = G[idx]

            # Mini-batches
            for start in range(0, N, args.batch_size):
                end = min(start + args.batch_size, N)
                if start >= end:
                    continue

                batch_obs = torch.from_numpy(obs[start:end]).float().to(device)   # (B, D)
                batch_G   = torch.from_numpy(G[start:end]).float().to(device)     # (B,)

                B = batch_obs.size(0)

                # Treat as sequence length 1 for LSTM
                hx = (
                    torch.zeros(1, B, cfg.lstm, device=device),
                    torch.zeros(1, B, cfg.lstm, device=device),
                )

                # Forward
                y, _ = model(batch_obs.unsqueeze(1), hx)  # (B,1,hidden)
                features = y.squeeze(1)                   # (B, hidden)

                heads_logits = model.step_logits_value(features)
                # Expect a "value" head of shape (B,1)
                v_pred = heads_logits["value"].squeeze(-1)  # (B,)

                loss = F.mse_loss(v_pred, batch_G)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                epoch_loss_sum += float(loss.item()) * B
                epoch_n_samples += B

        mean_loss = epoch_loss_sum / epoch_n_samples if epoch_n_samples > 0 else float("nan")
        print(f"[value-pretrain] epoch {epoch+1} mean MSE = {mean_loss:.6f}")

    # ----------------- Save train.py-compatible checkpoint -----------------
    payload = {
        "epoch": args.epochs - 1,
        "args": {
            "value_pretrain": vars(args)
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
    print(f"[value-pretrain] saved value-pretraining checkpoint to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Supervised value-head pretraining on (obs, return) data."
    )
    ap.add_argument("--rules", required=True,
                    help="Path to rules JSON (same as train.py)")
    ap.add_argument("--value-dir", required=True,
                    help="Folder containing (obs, G) .npz chunks from bc_dataset_returns_step0b.py")
    ap.add_argument("--out", default="runs/value_pretrain.pt",
                    help="Where to save the checkpoint (train.py --resume will load this)")
    ap.add_argument("--epochs", type=int, default=3,
                    help="Number of passes over the value dataset")
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="Mini-batch size")
    ap.add_argument("--hidden", type=int, default=256,
                    help="Hidden size (must match train.py)")
    ap.add_argument("--lstm", type=int, default=256,
                    help="LSTM size (must match train.py)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate for Adam")
    ap.add_argument("--seed", type=int, default=123456,
                    help="Global seed")
    ap.add_argument("--resume", default=None,
                    help="Optional: resume value pretraining from an existing .pt checkpoint")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even if CUDA is available")
    args = ap.parse_args()
    train_value_from_returns(args)
