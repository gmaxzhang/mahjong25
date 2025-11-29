import torch
import numpy as np
from copy import deepcopy
from algorithm.sim_and_train import Env
from algorithm.mahjongrl.agent import RLPolicy
from algorithm.mahjongrl.env_wrapper import build_observation, compute_rl_reward

class VecEnv:
    """
    Vectorized environment wrapper for Mahjong.
    Runs multiple Env instances in parallel and batches policy inference on GPU.
    """
    def __init__(self, rules, model, n_envs=8, device="cuda", lineup_tags=None):
        self.rules = rules
        self.device = device
        self.n_envs = n_envs
        self.envs = [Env(rules, seed=i+1) for i in range(n_envs)]
        self.model = model
        self.lineup_tags = lineup_tags or ["hybrid", "hybrid", "hybrid"]
        self.rl_policies = [RLPolicy(0, rules, model, device) for _ in range(n_envs)]
        self.dones = np.zeros(n_envs, dtype=bool)
        self.obs_dim = model.cfg.obs_dim
        self._init_envs()

    def _init_envs(self):
        for env in self.envs:
            if hasattr(env, "reset"):
                try:
                    env.reset()
                except Exception:
                    pass

    def reset(self):
        """Reset all environments and return stacked initial observations."""
        self.dones[:] = False
        obs = []
        for i, env in enumerate(self.envs):
            env.__init__(self.rules)
            ob = build_observation(env, seat=0)
            obs.append(ob)
        return torch.tensor(np.stack(obs), device=self.device, dtype=torch.float32)

    def step(self, actions):
        """
        Step all environments once with provided actions for seat 0.
        Each action is an integer index corresponding to the discard or relevant head.
        """
        batch_obs, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            if self.dones[i]:
                batch_obs.append(np.zeros(self.obs_dim))
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
                continue

            try:
                env.force_action(0, int(actions[i]))
            except Exception:
                pass
            try:
                env.step_turn(self.rl_policies)
            except Exception:
                self.dones[i] = True

            done = bool(getattr(env, "terminal", None))
            self.dones[i] = done

            reward = 0.0
            if done:
                reward = compute_rl_reward(env.terminal, seat=0, rules=self.rules)
            rewards.append(float(reward))
            dones.append(done)
            infos.append(env.terminal if done else {})

            ob = build_observation(env, seat=0)
            batch_obs.append(ob)

        obs_tensor = torch.tensor(np.stack(batch_obs), device=self.device, dtype=torch.float32)
        rew_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        done_tensor = torch.tensor(dones, device=self.device)
        return obs_tensor, rew_tensor, done_tensor, infos
