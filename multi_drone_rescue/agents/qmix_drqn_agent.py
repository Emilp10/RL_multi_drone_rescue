from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


@dataclass
class DRQNConfig:
    obs_dim: int
    state_dim: int
    n_actions: int = 5
    n_agents: int = 3
    gamma: float = 0.95
    lr: float = 1e-4
    batch_size: int = 8            # batches of sequences
    buffer_episodes: int = 200     # number of episodes to keep
    min_episodes: int = 20
    target_update_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_steps: int = 60000
    hidden_dim: int = 256
    mixer_hidden_dim: int = 64
    seq_len: int = 48
    burn_in: int = 12
    n_step: int = 3
    tau: float = 0.01
    device: str = "cpu"


class QNetRNN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, obs]
        B, T, D = x.shape
        x = self.relu(self.fc1(x))
        y, hn = self.gru(x, h)
        y = self.ln(y)
        y = self.dropout(y)
        q = self.head(y)  # [B, T, A]
        return q, hn

    def step(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, obs]
        return self.forward(x, h)


class MonotonicMixer(nn.Module):
    """QMIX-style monotonic mixing network using state-conditioned hypernets.
    Ensures dQ_total/dQ_i >= 0 via positive weights (softplus).
    """
    def __init__(self, n_agents: int, state_dim: int, hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden = hidden
        # Hypernets produce state-conditioned weights and biases
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden)
        self.hyper_b1 = nn.Linear(state_dim, hidden)
        self.hyper_w2 = nn.Linear(state_dim, hidden)
        self.hyper_b2 = nn.Linear(state_dim, 1)
        self.activation = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, q_agents: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # q_agents: [B, n_agents], state: [B, state_dim]
        B = q_agents.size(0)
        # First layer: [B, hidden] = sum_i softplus(w1_i) * q_i + b1
        w1 = self.softplus(self.hyper_w1(state)).view(B, self.n_agents, self.hidden)  # positive weights
        b1 = self.hyper_b1(state)  # [B, hidden]
        x = (q_agents.unsqueeze(-1) * w1).sum(dim=1) + b1  # [B, hidden]
        x = self.activation(x)
        # Second layer: scalar output
        w2 = self.softplus(self.hyper_w2(state))  # [B, hidden]
        b2 = self.hyper_b2(state)  # [B, 1]
        y = (x * w2).sum(dim=1, keepdim=True) + b2  # [B, 1]
        return y.squeeze(1)


class EpisodeReplay:
    def __init__(self, capacity_episodes: int):
        self.capacity = capacity_episodes
        self.episodes: List[List[dict]] = []

    def push_episode(self, steps: List[dict]):
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(steps)

    def sample(self, batch_size: int, seq_len: int):
        assert self.episodes, "No episodes in buffer"
        batch = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            if len(ep) < seq_len:
                # pad by repeating last transition
                pad = [ep[-1]] * (seq_len - len(ep))
                subseq = ep + pad
                start = 0
            else:
                start = random.randint(0, len(ep) - seq_len)
                subseq = ep[start:start+seq_len]
            batch.append(subseq)
        # batch: [B][T]{dict}
        # convert to arrays
        B = len(batch)
        T = len(batch[0])
        n_agents = len(batch[0][0]["obs"])
        obs = np.stack([[step["obs"] for step in seq] for seq in batch], axis=0)  # [B, T, n_agents, obs_dim]
        actions = np.stack([[step["actions"] for step in seq] for seq in batch], axis=0)  # [B, T, n_agents]
        rewards = np.stack([[step["rewards"] for step in seq] for seq in batch], axis=0)  # [B, T, n_agents]
        next_obs = np.stack([[step["next_obs"] for step in seq] for seq in batch], axis=0)  # [B, T, n_agents, obs_dim]
        dones = np.stack([[step["dones"] for step in seq] for seq in batch], axis=0)  # [B, T, n_agents]
        state = np.stack([[step["state"] for step in seq] for seq in batch], axis=0)  # [B, T, state_dim]
        next_state = np.stack([[step["next_state"] for step in seq] for seq in batch], axis=0)  # [B, T, state_dim]
        next_masks = np.stack([[step["next_masks"] for step in seq] for seq in batch], axis=0)  # [B, T, n_agents, A]
        return obs, actions, rewards, next_obs, dones, state, next_state, next_masks

    def __len__(self):
        return len(self.episodes)


class DRQNAgent:
    def __init__(self, cfg: DRQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q_nets = nn.ModuleList([QNetRNN(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim) for _ in range(cfg.n_agents)])
        self.target_q_nets = nn.ModuleList([QNetRNN(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim) for _ in range(cfg.n_agents)])
        self.mixer = MonotonicMixer(cfg.n_agents, cfg.state_dim, cfg.mixer_hidden_dim)
        self.target_mixer = MonotonicMixer(cfg.n_agents, cfg.state_dim, cfg.mixer_hidden_dim)
        self.q_nets.to(self.device)
        self.target_q_nets.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self.optimizer = optim.Adam(list(self.q_nets.parameters()) + list(self.mixer.parameters()), lr=cfg.lr)
        # Cosine annealing LR scheduler for stability
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-6)
        self.buffer = EpisodeReplay(cfg.buffer_episodes)
        self._update_targets(hard=True)

        self.epsilon = cfg.epsilon_start
        self.eps_decay = (cfg.epsilon_start - cfg.epsilon_end) / max(1, cfg.epsilon_decay_steps)
        # online hidden states for acting (one per agent)
        self._h: List[Optional[torch.Tensor]] = [None for _ in range(cfg.n_agents)]
        # Running observation normalization (EMA mean/var)
        self.obs_mean = torch.zeros(cfg.obs_dim, device=self.device)
        self.obs_var = torch.ones(cfg.obs_dim, device=self.device)
        self.obs_count = 1e-6

    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., obs_dim]
        eps = 1e-6
        mean = self.obs_mean.view(1, -1)
        std = torch.sqrt(self.obs_var + eps).view(1, -1)
        return (x - mean) / std

    def _update_obs_stats(self, batch_obs: torch.Tensor):
        # batch_obs: [B, T, N, obs_dim]
        with torch.no_grad():
            # Reduce over B, T, N
            dims = (0, 1, 2)
            b_mean = batch_obs.mean(dim=dims)
            b_var = batch_obs.var(dim=dims, unbiased=False)
            alpha = 0.01  # EMA factor
            self.obs_mean = (1 - alpha) * self.obs_mean + alpha * b_mean
            self.obs_var = (1 - alpha) * self.obs_var + alpha * b_var
            self.obs_count += float(batch_obs.shape[0] * batch_obs.shape[1] * batch_obs.shape[2])

    def reset_episode(self):
        self._h = [None for _ in range(self.cfg.n_agents)]

    def act(self, obs_batch: Dict[str, np.ndarray], explore: bool = True, action_masks: Dict[str, np.ndarray] | None = None) -> Dict[str, int]:
        actions = {}
        for i, (aid, obs) in enumerate(obs_batch.items()):
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
                # normalize using running stats
                xm = self._normalize_obs(x.view(1, -1)).view(1, 1, -1)
                q, h = self.q_nets[i].step(xm, self._h[i])
                self._h[i] = h
                if action_masks is not None and aid in action_masks:
                    mask = torch.tensor(action_masks[aid], dtype=torch.float32, device=self.device).view(1, 1, -1)
                    q = q + (mask - 1.0) * 1e9
                if explore and random.random() < self.epsilon:
                    a = random.randrange(self.cfg.n_actions)
                else:
                    a = int(q[0, 0].argmax().item())
            actions[aid] = a
        return actions

    def push_episode(self, steps: List[dict]):
        self.buffer.push_episode(steps)

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
        if len(self.buffer) < self.cfg.min_episodes:
            return None
        obs, actions, rewards, next_obs, dones, state, next_state, next_masks = self.buffer.sample(self.cfg.batch_size, self.cfg.seq_len)
        device = self.device
        # to tensors
        obs = torch.tensor(obs, dtype=torch.float32, device=device)            # [B, T, N, obs]
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)  # [B, T, N, obs]
        actions = torch.tensor(actions, dtype=torch.int64, device=device)      # [B, T, N]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)    # [B, T, N]
        dones = torch.tensor(dones, dtype=torch.float32, device=device)        # [B, T, N]
        state = torch.tensor(state, dtype=torch.float32, device=device)        # [B, T, S]
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)  # [B, T, S]
        next_masks = torch.tensor(next_masks, dtype=torch.float32, device=device)  # [B, T, N, A]

        # Update running obs stats and normalize observations
        self._update_obs_stats(obs)
        B, T, N, D = obs.shape
        obs = self._normalize_obs(obs.view(B*T*N, D)).view(B, T, N, D)
        next_obs = self._normalize_obs(next_obs.view(B*T*N, D)).view(B, T, N, D)

        B, T, N, A = obs.shape[0], obs.shape[1], obs.shape[2], self.cfg.n_actions
        burn = self.cfg.burn_in
        # Current Q values and mixed Q
        q_agents_list = []  # list of [B, T]
        for i in range(self.cfg.n_agents):
            qi_all, _ = self.q_nets[i](obs[:, :, i, :])  # [B, T, A]
            ai = actions[:, :, i].unsqueeze(-1)          # [B, T, 1]
            qi_a = qi_all.gather(-1, ai).squeeze(-1)     # [B, T]
            q_agents_list.append(qi_a)
        q_agents = torch.stack(q_agents_list, dim=-1)     # [B, T, N]
        # mix per time step by flattening time into batch
        total_q = self.mixer(q_agents.reshape(B*T, N), state.reshape(B*T, -1)).view(B, T)  # [B, T]

        # Target with masked Double Q
        with torch.no_grad():
            target_list = []
            for i in range(self.cfg.n_agents):
                q_online, _ = self.q_nets[i](next_obs[:, :, i, :])  # [B, T, A]
                q_online = q_online + (next_masks[:, :, i, :] - 1.0) * 1e9
                a_star = q_online.argmax(dim=-1)  # [B, T]
                q_tgt_all, _ = self.target_q_nets[i](next_obs[:, :, i, :])
                qi = q_tgt_all.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)  # [B, T]
                target_list.append(qi)
            target_q = torch.stack(target_list, dim=-1)  # [B, T, N]
            target_total_q = self.target_mixer(target_q.reshape(B*T, N), next_state.reshape(B*T, -1)).view(B, T)

        team_r = rewards.sum(dim=2)  # [B, T]
        team_r = torch.clamp(team_r * 0.1, -10.0, 10.0)
        done_any = (dones.max(dim=2).values > 0.5).float()  # [B, T]

        # n-step targets
        n = self.cfg.n_step
        T_eff = max(1, T - n)
        # discounted n-step reward
        r_n = torch.zeros((B, T_eff), dtype=torch.float32, device=device)
        for k in range(n):
            r_n += (self.cfg.gamma ** k) * team_r[:, k:k+T_eff]
        # bootstrap mask: 1 if no done within next n steps
        # rolling OR over window n
        # build windowed max via loop for readability
        term_within = torch.zeros((B, T_eff), dtype=torch.float32, device=device)
        for k in range(n):
            term_within = torch.maximum(term_within, done_any[:, k:k+T_eff])
        m_boot = 1.0 - term_within
        # bootstrap value from t+n-1 index of target_total_q
        v_boot = target_total_q[:, n-1:n-1+T_eff]
        y = r_n + (self.cfg.gamma ** n) * m_boot * v_boot
        # predictions aligned to first T_eff steps
        pred_all = total_q[:, :T_eff]
        # apply burn-in: compute loss only after burn_in
        b = min(burn, pred_all.shape[1]-1)
        pred = pred_all[:, b:]
        targ = y[:, b:]
        loss = nn.SmoothL1Loss()(pred, targ)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.q_nets.parameters()) + list(self.mixer.parameters()), 10.0)
        self.optimizer.step()
        # step LR scheduler
        try:
            self.lr_scheduler.step()
        except Exception:
            pass

        # epsilon decay
        if self.epsilon > self.cfg.epsilon_end:
            self.epsilon = max(self.cfg.epsilon_end, self.epsilon - self.eps_decay)
        # soft + periodic hard update
        self._update_targets(hard=False, tau=self.cfg.tau)
        if random.randint(1, self.cfg.target_update_interval) == 1:
            self._update_targets(hard=True)
        return float(loss.item())

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "q": self.q_nets.state_dict(),
            "mixer": self.mixer.state_dict(),
            "eps": self.epsilon,
            "cfg": self.cfg.__dict__,
            "obs_norm": {
                "mean": self.obs_mean.detach().cpu(),
                "var": self.obs_var.detach().cpu(),
                "count": self.obs_count,
            },
        }, os.path.join(path, "qmix_drqn.pt"))

    def load(self, path: str, map_location: str = "cpu"):
        ckpt = torch.load(os.path.join(path, "qmix_drqn.pt"), map_location=map_location)
        self.q_nets.load_state_dict(ckpt["q"])  # type: ignore
        # Mixer architecture may change between versions; load non-strictly and continue on mismatch
        try:
            self.mixer.load_state_dict(ckpt.get("mixer", {}), strict=False)  # type: ignore
        except Exception:
            # Fallback: keep randomly initialized mixer
            pass
        self._update_targets(hard=True)
        self.epsilon = float(ckpt.get("eps", 0.05))
        # Load obs normalization if available
        on = ckpt.get("obs_norm")
        if isinstance(on, dict):
            try:
                self.obs_mean = on.get("mean", self.obs_mean).to(self.device)
                self.obs_var = on.get("var", self.obs_var).to(self.device)
                self.obs_count = float(on.get("count", self.obs_count))
            except Exception:
                pass
