import os
import csv
import argparse
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from environment.rescue_env import RescueConfig, parallel_env
from agents.qmix_agent import QMIXAgent, QMIXConfig
from agents.qmix_drqn_agent import DRQNAgent, DRQNConfig


@dataclass
class TrainConfig:
    episodes: int = 200
    episode_max_steps: int = 200
    save_every: int = 25
    save_dir: str = "checkpoints"
    log_csv: str = "training_log.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=3, help="Number of agents (1 or 3)")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes")
    parser.add_argument("--save-every", type=int, default=None, help="Override save interval")
    parser.add_argument("--min-episodes", type=int, default=None, help="Override min episodes before training")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--burn-in", type=int, default=None, help="Override burn-in steps")
    parser.add_argument("--n-step", type=int, default=None, help="Override n-step returns")
    parser.add_argument("--epsilon-decay-steps", type=int, default=None, help="Override epsilon decay steps")
    parser.add_argument("--epsilon-end", type=float, default=None, help="Override final epsilon")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dim for QNetRNN")
    parser.add_argument("--mixer-hidden-dim", type=int, default=None, help="Override mixer hidden dim")
    parser.add_argument("--save-suffix", type=str, default="", help="Suffix for save_dir to isolate runs (e.g., hp1)")
    args = parser.parse_args()
    # Env
    env_cfg = RescueConfig(grid_size=10, num_agents=args.agents, num_victims=6, obstacle_density=0.15, fov_radius=2, max_steps=200, seed=42)
    env = parallel_env(env_cfg)

    # Derive dims
    obs_sample, _ = env.reset()
    n_agents = len(env.agents)
    obs_dim = len(next(iter(obs_sample.values())))
    state_dim = len(env.state())

    # Agent (DRQN-QMIX by default)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Improved hyperparameters for better performance
    agent_cfg = DRQNConfig(
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=env.num_actions,
        n_agents=n_agents,
        device=device,
        # Learning
        lr=3e-4,                      # Increased learning rate for faster initial learning
        gamma=0.97,                    # Slightly higher discount for longer-term planning
        # Buffer & Batching
        batch_size=16,                 # Larger batch size for more stable gradients
        buffer_episodes=400,           # Larger buffer for better sample diversity
        min_episodes=40,               # More episodes before training starts
        # Network architecture
        hidden_dim=256,                # Larger network capacity
        mixer_hidden_dim=128,          # Larger mixer network
        # Recurrent sequence
        seq_len=32,                    # Longer sequences for better temporal learning
        burn_in=8,                     # Proportional burn-in period
        n_step=5,                      # Longer n-step returns
        # Exploration
        epsilon_start=1.0,
        epsilon_end=0.05,              # Lower final epsilon for more exploitation
        epsilon_decay_steps=100000,    # Slower decay for better exploration
        # Target network
        target_update_interval=500,    # Less frequent hard updates
        tau=0.005,                     # Soft update coefficient (if using soft updates)
    )
    # Apply CLI overrides if provided
    if args.lr is not None:
        agent_cfg.lr = args.lr
    if args.seq_len is not None:
        agent_cfg.seq_len = args.seq_len
    if args.burn_in is not None:
        agent_cfg.burn_in = args.burn_in
    if args.n_step is not None:
        agent_cfg.n_step = args.n_step
    if args.epsilon_decay_steps is not None:
        agent_cfg.epsilon_decay_steps = args.epsilon_decay_steps
    if args.epsilon_end is not None:
        agent_cfg.epsilon_end = args.epsilon_end
    if args.hidden_dim is not None:
        agent_cfg.hidden_dim = args.hidden_dim
    if args.mixer_hidden_dim is not None:
        agent_cfg.mixer_hidden_dim = args.mixer_hidden_dim
    if args.min_episodes is not None:
        agent_cfg.min_episodes = args.min_episodes
    agent = DRQNAgent(agent_cfg)

    # Extended training configuration for better convergence
    train_cfg = TrainConfig()
    train_cfg.episodes = 1500 if args.episodes is None else args.episodes
    train_cfg.save_every = 50 if args.save_every is None else args.save_every
    # save directory adjusted by agent count
    suffix = f"_{args.save_suffix}" if args.save_suffix else ""
    train_cfg.save_dir = os.path.join("checkpoints", f"drqn_agents_{n_agents}{suffix}")
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    # Logging
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"training_log_agents_{n_agents}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "loss", "victims_rescued", "victims_left"])

    for ep in range(1, train_cfg.episodes + 1):
        obs, infos = env.reset()
        agent.reset_episode()
        total_reward = 0.0
        losses = []
        victims_left_start = int(infos[env.agents[0]]["victims_left"])  # same for all agents
        for t in range(env_cfg.max_steps):
            # collect current masks for action selection
            masks = {aid: infos[aid].get("action_mask") for aid in env.agents}
            actions = agent.act(obs, explore=True, action_masks=masks)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = any(list(terminations.values())) or any(list(truncations.values()))

            # Pack transition for episode buffer
            obs_arr = np.stack([obs[aid] for aid in env.agents], axis=0)
            next_obs_arr = np.stack([next_obs[aid] for aid in env.agents], axis=0)
            actions_arr = np.array([actions[aid] for aid in env.agents], dtype=np.int64)
            rewards_arr = np.array([rewards[aid] for aid in env.agents], dtype=np.float32)
            dones_arr = np.array([float(terminations[aid] or truncations[aid]) for aid in env.agents], dtype=np.float32)
            state = infos[env.agents[0]]["state"]
            next_state = env.state()
            next_masks = np.stack([infos[aid]["action_mask"] for aid in env.agents], axis=0)
            # store per-step dict in a local list
            if t == 0:
                episode_steps = []
            episode_steps.append({
                "obs": obs_arr,
                "actions": actions_arr,
                "rewards": rewards_arr,
                "next_obs": next_obs_arr,
                "dones": dones_arr,
                "state": state,
                "next_state": next_state,
                "next_masks": next_masks,
            })

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            total_reward += rewards_arr.sum()
            obs = next_obs
            if done:
                break

        # push completed episode to buffer
        agent.push_episode(episode_steps)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        victims_left_end = int(infos[env.agents[0]]["victims_left"]) if 'infos' in locals() else 0
        victims_rescued = max(0, victims_left_start - victims_left_end)
        print(f"Episode {ep} | Return={total_reward:.2f} | Loss={avg_loss:.4f} | Rescued={victims_rescued} | Left={victims_left_end} | Eps={agent.epsilon:.2f}")
        # append to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, total_reward, avg_loss, victims_rescued, victims_left_end])

        if ep % train_cfg.save_every == 0:
            out = os.path.join(train_cfg.save_dir, f"ep_{ep}")
            os.makedirs(out, exist_ok=True)
            agent.save(out)

    # final save
    agent.save(os.path.join(train_cfg.save_dir, "final"))


if __name__ == "__main__":
    main()
