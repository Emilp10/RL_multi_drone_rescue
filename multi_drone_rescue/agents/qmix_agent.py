from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class QMIXConfig:
    obs_dim: int
    state_dim: int
    n_actions: int = 5
    n_agents: int = 3
    gamma: float = 0.95
    lr: float = 1e-4
    batch_size: int = 128
    buffer_size: int = 50000
    min_buffer: int = 5000
    target_update_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_steps: int = 40000
    hidden_dim: int = 128
    mixer_hidden_dim: int = 64
    tau: float = 0.01  # soft target update rate
    device: str = "cpu"


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class Mixer(nn.Module):
    """Simple 2-layer mixing network with state-dependent weights.
    This is a simplified QMIX mixer.
    """
    def __init__(self, n_agents: int, state_dim: int, hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed1 = nn.Linear(state_dim, n_agents * hidden)
        self.bias1 = nn.Linear(state_dim, hidden)
        self.embed2 = nn.Linear(state_dim, hidden)
        self.bias2 = nn.Linear(state_dim, 1)

    def forward(self, q_agents: torch.Tensor, state: torch.Tensor):
        # q_agents: [B, n_agents]
        B = q_agents.size(0)
        h = self.embed1(state).view(B, self.n_agents, -1)  # [B, n_agents, H]
        b1 = self.bias1(state).unsqueeze(1)  # [B, 1, H]
        x = torch.relu((h * q_agents.unsqueeze(-1)).sum(dim=1) + b1.squeeze(1))  # [B, H]
        w2 = self.embed2(state)  # [B, H]
        b2 = self.bias2(state)   # [B, 1]
        y = (x * w2).sum(dim=1, keepdim=True) + b2  # [B, 1]
        return y.squeeze(1)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []
        self.idx = 0

    def push(self, item: dict):
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.idx] = item
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[dict]:
        idxs = np.random.choice(len(self.data), size=batch_size, replace=False)
        return [self.data[i] for i in idxs]

    def __len__(self):
        return len(self.data)


class QMIXAgent:
    def __init__(self, cfg: QMIXConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q_nets = nn.ModuleList([QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim) for _ in range(cfg.n_agents)])
        self.target_q_nets = nn.ModuleList([QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim) for _ in range(cfg.n_agents)])
        self.mixer = Mixer(cfg.n_agents, cfg.state_dim, cfg.mixer_hidden_dim)
        self.target_mixer = Mixer(cfg.n_agents, cfg.state_dim, cfg.mixer_hidden_dim)
        self.q_nets.to(self.device)
        self.target_q_nets.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self.optimizer = optim.Adam(list(self.q_nets.parameters()) + list(self.mixer.parameters()), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self._update_targets(hard=True)

        self.epsilon = cfg.epsilon_start
        self.eps_decay = (cfg.epsilon_start - cfg.epsilon_end) / max(1, cfg.epsilon_decay_steps)
        self.train_steps = 0

    def act(self, obs_batch: Dict[str, np.ndarray], explore: bool = True, action_masks: Dict[str, np.ndarray] | None = None) -> Dict[str, int]:
        actions = {}
        for i, (aid, obs) in enumerate(obs_batch.items()):
            if explore and random.random() < self.epsilon:
                a = random.randrange(self.cfg.n_actions)
            else:
                with torch.no_grad():
                    q = self.q_nets[i](torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                    if action_masks is not None and aid in action_masks:
                        mask = torch.tensor(action_masks[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
                        # set invalid actions to very negative
                        q = q + (mask - 1.0) * 1e9
                    a = int(q.argmax(dim=1).item())
            actions[aid] = a
        return actions

    def push_transition(self, transition: dict):
        self.buffer.push(transition)

    def _update_targets(self, hard: bool = False, tau: float = 1.0):
        src_q = list(self.q_nets.parameters()) + list(self.mixer.parameters())
        dst_q = list(self.target_q_nets.parameters()) + list(self.target_mixer.parameters())
        if hard:
            for t, s in zip(dst_q, src_q):
                t.data.copy_(s.data)
        else:
            for t, s in zip(dst_q, src_q):
                t.data.copy_(tau * s.data + (1 - tau) * t.data)

    def train_step(self):
        if len(self.buffer) < self.cfg.min_buffer:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        # Stack batch
        obs = torch.tensor(np.stack([b["obs"] for b in batch]), dtype=torch.float32, device=self.device)  # [B, n_agents, obs_dim]
        actions = torch.tensor(np.stack([b["actions"] for b in batch]), dtype=torch.int64, device=self.device)  # [B, n_agents]
        rewards = torch.tensor(np.stack([b["rewards"] for b in batch]), dtype=torch.float32, device=self.device)  # [B, n_agents]
        next_obs = torch.tensor(np.stack([b["next_obs"] for b in batch]), dtype=torch.float32, device=self.device)  # [B, n_agents, obs_dim]
        dones = torch.tensor(np.stack([b["dones"] for b in batch]), dtype=torch.float32, device=self.device)  # [B, n_agents]
        state = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32, device=self.device)  # [B, state_dim]
        next_state = torch.tensor(np.stack([b["next_state"] for b in batch]), dtype=torch.float32, device=self.device)  # [B, state_dim]
        # optional masks
        next_masks = None
        if "next_masks" in batch[0]:
            next_masks = torch.tensor(np.stack([b["next_masks"] for b in batch]), dtype=torch.float32, device=self.device)  # [B, n_agents, A]

        B = obs.size(0)
        n_agents = self.cfg.n_agents

        # Current Q values for actions
        q_vals = []
        with torch.set_grad_enabled(True):
            for i in range(n_agents):
                qi = self.q_nets[i](obs[:, i, :])  # [B, A]
                ai = actions[:, i]  # [B]
                qi_a = qi.gather(1, ai.unsqueeze(1)).squeeze(1)  # [B]
                q_vals.append(qi_a)
            q_vals = torch.stack(q_vals, dim=1)  # [B, n_agents]
            total_q = self.mixer(q_vals, state)  # [B]

        # Double Q with masked selection: argmax from online nets over valid actions; evaluate with target nets
        with torch.no_grad():
            target_q_vals = []
            for i in range(n_agents):
                q_online = self.q_nets[i](next_obs[:, i, :])  # [B, A]
                if next_masks is not None:
                    q_online = q_online + (next_masks[:, i, :] - 1.0) * 1e9
                a_online = q_online.argmax(dim=1)  # [B]
                q_target_all = self.target_q_nets[i](next_obs[:, i, :])  # [B, A]
                qi = q_target_all.gather(1, a_online.unsqueeze(1)).squeeze(1)  # [B]
                target_q_vals.append(qi)
            target_q_vals = torch.stack(target_q_vals, dim=1)  # [B, n_agents]
            target_total_q = self.target_mixer(target_q_vals, next_state)  # [B]

        # Rewards: sum across agents as team reward
        team_r = rewards.sum(dim=1)  # [B]
        # scale and clamp rewards to stabilize targets
        team_r = torch.clamp(team_r * 0.1, -10.0, 10.0)
        done_any = (dones.max(dim=1).values > 0.5).float()  # episode done if any agent done

        y = team_r + self.cfg.gamma * (1.0 - done_any) * target_total_q
        loss = nn.SmoothL1Loss()(total_q, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.q_nets.parameters()) + list(self.mixer.parameters()), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        # epsilon decay
        if self.epsilon > self.cfg.epsilon_end:
            self.epsilon = max(self.cfg.epsilon_end, self.epsilon - self.eps_decay)

        # soft update every step
        self._update_targets(hard=False, tau=self.cfg.tau)
        # and periodic hard sync for safety
        if self.train_steps % self.cfg.target_update_interval == 0:
            self._update_targets(hard=True)

        return float(loss.item())

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "q": self.q_nets.state_dict(),
            "mixer": self.mixer.state_dict(),
            "eps": self.epsilon,
        }, os.path.join(path, "qmix.pt"))

    def load(self, path: str, map_location: str = "cpu"):
        ckpt = torch.load(os.path.join(path, "qmix.pt"), map_location=map_location)
        self.q_nets.load_state_dict(ckpt["q"])  # type: ignore
        self.mixer.load_state_dict(ckpt["mixer"])  # type: ignore
        self._update_targets(hard=True)
        self.epsilon = float(ckpt.get("eps", 0.05))
