# mahjong_rl/train.py
from __future__ import annotations
import argparse, random, math
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rules_io import load_rules
from sim_and_train import Env
from mahjongrl.env_wrapper import make_lineup_with_rl, build_observation, compute_rl_reward
from mahjongrl.model import ACConfig, LSTMActorCritic
from mahjongrl.agent import RLPolicy

def a2c_update(steps, gamma: float, lam: float,
               alpha_entropy: float, alpha_oracle: float,
               model: LSTMActorCritic, device: str, opt: optim.Optimizer):
    """
    steps: List[StepRecord], all from one episode (sequence).
    We compute returns/advantages with GAE(λ).
    """
    # Collect arrays
    obs = torch.from_numpy(np.stack([s.obs for s in steps], axis=0)).float().to(device)  # [T,D]
    logp = torch.tensor([s.logp for s in steps], dtype=torch.float32, device=device)     # [T]
    vals = torch.tensor([s.value for s in steps], dtype=torch.float32, device=device)    # [T]

    # Re-run the model to get current logits for entropy & policy loss
    cfg = model.cfg
    model.train()
    hx = (torch.zeros(1,1,cfg.lstm, device=device),
          torch.zeros(1,1,cfg.lstm, device=device))
    y, _ = model(obs[None,:,:], hx)  # [1,T,H]
    logits_all = model.step_logits_value(y.squeeze(0))  # dict of [T, A?]
    entropy_terms = []

    # Build per-step policy loss using stored kind/mask/choice
    pol_losses = []
    imit_losses = []
    for t, s in enumerate(steps):
        if s.kind == "discard":
            logits = logits_all["discard"][t:t+1, :]  # [1,34]
        elif s.kind in ("ron","pung"):
            logits = logits_all["binary"][t:t+1, :]
        elif s.kind == "chow":
            logits = logits_all["chow"][t:t+1, :]
        else:  # kongs
            logits = logits_all["kong"][t:t+1, :]

        mask = torch.from_numpy(s.mask)[None,:].to(device)
        masked = logits + (mask + 1e-8).log()
        logpi = torch.log_softmax(masked, dim=-1)
        act = torch.tensor([[s.choice]], dtype=torch.long, device=device)
        logpi_a = logpi.gather(-1, act).squeeze(-1).squeeze(0)  # scalar
        ent = -(logpi.exp() * logpi).sum(dim=-1).mean()
        entropy_terms.append(ent)

        # Policy gradient part uses advantage; placeholder here (we'll compute advantages next)
        pol_losses.append(-logpi_a)  # multiply by advantage later

        # Oracle imitation (optional)
        if s.oracle_choice is not None:
            oc = int(s.oracle_choice)
            if oc >= 0 and oc < logpi.shape[-1] and s.mask[oc] > 0.5:
                imit_losses.append(nn.functional.cross_entropy(logpi.unsqueeze(0).exp().log(), torch.tensor([oc], device=device)))
    # Compute returns/advantages with GAE
    # Note: one payoff at episode end; distribute via same return at each step (undiscounted), or
    # use gamma^k. Here we do standard episodic return equally backprop’d with GAE.
    # We need final reward; stash in steps[-1].value? We pass separately.
    # For simplicity, moments after we call this function, we replace 'R' using last element of returns.

    # We’ll just compute A = R - V_t using the final return R seen on terminal.
    # To do this properly, the caller will set 'final_return'.
    # Here we expect the caller to pass 'adv' and 'ret' directly. For brevity, we fold it in:
    # (Override pol_losses with advantage.)

    # This function will be called with additional tensors:
    return pol_losses, entropy_terms, imit_losses, logits_all, vals

def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    rules = load_rules(args.rules)

    # Create a dummy env to compute obs_dim
    dummy = Env(rules, seed=123)
    obs0 = build_observation(dummy, seat=0)
    cfg = ACConfig(obs_dim=obs0.shape[0], hidden=args.hidden, lstm=args.lstm)
    model = LSTMActorCritic(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    def run_episode_vs(lineup_tags: List[str], seed: int, selfplay: bool = False):
        env = Env(rules, seed=seed)
        rl = RLPolicy(seat=0, rules=rules, model=model, device=device)
        if selfplay:
            # three copies sharing parameters
            lineup = [rl, RLPolicy(1, rules, model, device), RLPolicy(2, rules, model, device), RLPolicy(3, rules, model, device)]
        else:
            lineup = make_lineup_with_rl(rl, rules, lineup_tags)
        env.turn = 0  # East starts
        draws = 0
        while env.wall and not env.terminal and draws < args.max_draws:
            env.step_turn(lineup)
            draws += 1
        reward = compute_rl_reward(env.terminal or {"source":"drawn_game"}, seat=0, rules=rules)
        return rl.buffer, reward

    # Curriculum: phase 1 vs fixed bots, phase 2 self-play
    total_steps = 0
    alpha_oracle = args.oracle_coef

    for epoch in range(args.epochs):
        buffers = []
        rewards = []
        for ep in range(args.episodes_per_epoch):
            seed = random.randint(1, 10**9)
            if epoch < args.vs_bots_epochs:
                lineup = args.lineup.split(",")
                buf, rew = run_episode_vs(lineup, seed, selfplay=False)
            else:
                buf, rew = run_episode_vs([], seed, selfplay=True)
            buffers.append(buf)
            rewards.append(rew)

        # Flatten
        steps = [s for buf in buffers for s in buf]
        if not steps:
            continue

        # Compute simple returns/advantages: same final reward for all steps in each episode
        # We will build a per-step advantage vector ‘adv’ aligned with ‘steps’.
        adv = []
        ret = []
        idx = 0
        for buf, R in zip(buffers, rewards):
            T = len(buf)
            for _ in range(T):
                adv.append(R)  # advantage baseline ~ 0; simple (R - V) handled inside loss.
                ret.append(R)
                idx += 1
        adv = torch.tensor(adv, dtype=torch.float32, device=device)
        ret = torch.tensor(ret, dtype=torch.float32, device=device)

        pol_losses, ent_terms, imit_losses, logits_all, vals = a2c_update(
            steps, gamma=args.gamma, lam=0.95,
            alpha_entropy=args.entropy_coef, alpha_oracle=alpha_oracle,
            model=model, device=device, opt=opt
        )

        # Now weight policy terms by (R - V)
        V = vals
        A = (adv - V).detach()
        # Align sizes
        pol_loss = torch.stack([pl * A[i] for i,pl in enumerate(pol_losses)]).mean()
        ent_loss = - args.entropy_coef * torch.stack(ent_terms).mean()
        val_loss = 0.5 * (ret - V).pow(2).mean()
        imit_loss = (alpha_oracle * torch.stack(imit_losses).mean()) if imit_losses else torch.tensor(0.0, device=device)

        loss = pol_loss + val_loss + ent_loss + imit_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Decay oracle weight across epochs
        alpha_oracle = max(0.0, alpha_oracle * args.oracle_decay)

        avg_r = float(np.mean(rewards)) if rewards else 0.0
        print(f"[epoch {epoch+1}/{args.epochs}] episodes={len(rewards)} avg_reward={avg_r:.2f} "
              f"loss={loss.item():.3f} pol={pol_loss.item():.3f} val={val_loss.item():.3f} "
              f"ent={(-ent_loss).item():.3f} imit={imit_loss.item():.3f} oracle_coef={alpha_oracle:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--lineup", default="aggro,wp,payout", help="3 opponents for phase 1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--episodes-per-epoch", type=int, default=64)
    ap.add_argument("--vs-bots-epochs", type=int, default=12, help="epochs vs fixed bots before self-play")
    ap.add_argument("--max-draws", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lstm", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--oracle-coef", type=float, default=0.5)
    ap.add_argument("--oracle-decay", type=float, default=0.9)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)
