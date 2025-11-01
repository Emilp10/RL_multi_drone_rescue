from __future__ import annotations
import os
import csv
import os.path as osp
from typing import Dict, Any
import argparse
import json
import numpy as np
import torch

from environment.rescue_env import RescueConfig, parallel_env
from agents.qmix_drqn_agent import DRQNAgent, DRQNConfig
from agents.qmix_agent import QMIXAgent, QMIXConfig


def _ckpt_dir(n_agents: int, override: str | None = None) -> str:
    if override is not None:
        return override
    return osp.join("checkpoints", f"drqn_agents_{n_agents}", "final")


def _ckpt_path(n_agents: int, override: str | None = None) -> str:
    return osp.join(_ckpt_dir(n_agents, override), "qmix_drqn.pt")


def evaluate_agents(n_agents: int, episodes: int = 50, seed: int | None = 123, ckpt_dir: str | None = None) -> Dict[str, Any]:
    cfg = RescueConfig(grid_size=10, num_agents=n_agents, num_victims=6, obstacle_density=0.15, fov_radius=2, max_steps=200, seed=seed)
    env = parallel_env(cfg)

    # build policy
    obs, _ = env.reset()
    obs_dim = len(next(iter(obs.values())))
    state_dim = len(env.state())
    ckpt = _ckpt_path(n_agents, ckpt_dir)
    if osp.exists(ckpt):
        # Load checkpoint config to match architecture (e.g., hidden sizes)
        ckpt_data = torch.load(ckpt, map_location="cpu")
        ckpt_cfg = ckpt_data.get("cfg", {}) if isinstance(ckpt_data, dict) else {}
        agent = DRQNAgent(DRQNConfig(
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=env.num_actions,
            n_agents=n_agents,
            device="cpu",
            hidden_dim=int(ckpt_cfg.get("hidden_dim", 128)),
            mixer_hidden_dim=int(ckpt_cfg.get("mixer_hidden_dim", 64)),
        ))
        agent.load(_ckpt_dir(n_agents, ckpt_dir))
        agent.epsilon = 0.0
    else:
        agent = QMIXAgent(QMIXConfig(obs_dim=obs_dim, state_dim=state_dim, n_actions=env.num_actions, n_agents=n_agents, device="cpu"))

    total_returns: list[float] = []
    victims_rescued: list[int] = []
    victims_left: list[int] = []

    for ep in range(episodes):
        obs, infos = env.reset()
        if hasattr(agent, "reset_episode"):
            agent.reset_episode()
        ep_return = 0.0
        victims_left_start = int(infos[env.agents[0]]["victims_left"])  # same for all agents
        for t in range(env.max_steps):
            masks = {aid: infos[aid].get("action_mask") for aid in env.agents}
            actions = agent.act(obs, explore=False, action_masks=masks)
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            ep_return += sum(float(r) for r in rewards.values())
            obs = next_obs
            if any(list(terms.values())) or any(list(truncs.values())):
                break
        victims_left_end = int(infos[env.agents[0]]["victims_left"]) if 'infos' in locals() else 0
        victims_rescued.append(max(0, victims_left_start - victims_left_end))
        victims_left.append(victims_left_end)
        total_returns.append(ep_return)

    # write eval log
    os.makedirs("logs", exist_ok=True)
    log_path = osp.join("logs", f"eval_agents_{n_agents}.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return", "victims_rescued", "victims_left"]) 
        for i, (ret, res, left) in enumerate(zip(total_returns, victims_rescued, victims_left), start=1):
            w.writerow([i, ret, res, left])

    success = sum(1 for v in victims_left if v == 0)
    summary = {
        "episodes": len(total_returns),
        "avg_return": float(np.mean(total_returns)) if total_returns else 0.0,
        "success_rate": success / max(1, len(total_returns)),
        "avg_rescued": float(np.mean(victims_rescued)) if victims_rescued else 0.0,
        "path": log_path,
    }

    return {"ok": True, "summary": summary}

def main():
    parser = argparse.ArgumentParser(description="Evaluate DRQN/QMIX agents")
    parser.add_argument("--agents", type=int, choices=[1, 3], default=3, help="Number of agents to evaluate (1 or 3)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Optional checkpoint directory override (path to 'final' folder)")
    args = parser.parse_args()

    res = evaluate_agents(args.agents, episodes=args.episodes, ckpt_dir=args.ckpt_dir)
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
