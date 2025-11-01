import os
import argparse
import csv
import numpy as np
from environment.rescue_env import RescueConfig, parallel_env
from agents.qmix_agent import QMIXAgent, QMIXConfig
from agents.qmix_drqn_agent import DRQNAgent, DRQNConfig


def evaluate(episodes: int = 30, max_steps: int | None = None, ckpt_dir: str | None = None, agents: int = 3):
    if ckpt_dir is None:
        ckpt_dir = os.path.join("checkpoints", f"drqn_agents_{agents}", "final")
    env_cfg = RescueConfig(grid_size=10, num_agents=3, num_victims=6, obstacle_density=0.15, fov_radius=2, max_steps=200, seed=123)
    env_cfg.num_agents = agents
    if max_steps is not None:
        env_cfg.max_steps = max_steps
    env = parallel_env(env_cfg)

    # derive dims
    obs, infos = env.reset()
    obs_dim = len(next(iter(obs.values())))
    state_dim = len(env.state())

    # Prefer DRQN checkpoint if available
    agent = None
    if os.path.exists(os.path.join(ckpt_dir, "qmix_drqn.pt")):
        agent = DRQNAgent(DRQNConfig(obs_dim=obs_dim, state_dim=state_dim, n_actions=env.num_actions, n_agents=len(env.agents), device="cpu"))
        agent.load(ckpt_dir)
    elif os.path.exists(os.path.join(ckpt_dir, "qmix.pt")):
        agent = QMIXAgent(QMIXConfig(obs_dim=obs_dim, state_dim=state_dim, n_actions=env.num_actions, n_agents=len(env.agents), device="cpu"))
        agent.load(ckpt_dir)
    else:
        agent = QMIXAgent(QMIXConfig(obs_dim=obs_dim, state_dim=state_dim, n_actions=env.num_actions, n_agents=len(env.agents), device="cpu"))
        print("Warning: checkpoint not found; evaluating with untrained model.")
    agent.epsilon = 0.0

    returns = []
    rescues = []
    success = 0
    steps_to_rescue = []

    for ep in range(1, episodes + 1):
        obs, infos = env.reset()
        victims_left_prev = int(infos[env.agents[0]]["victims_left"])  # same for all agents
        total = 0.0
        ep_rescues = 0
        for t in range(env_cfg.max_steps):
            masks = {aid: infos[aid]["action_mask"] for aid in env.agents}
            actions = agent.act(obs, explore=False, action_masks=masks)
            obs, rewards, terms, trunc, infos = env.step(actions)
            total += float(sum(rewards.values()))
            victims_left = int(infos[env.agents[0]]["victims_left"])  # same for all
            if victims_left < victims_left_prev:
                # rescued one or more this step
                ep_rescues += (victims_left_prev - victims_left)
                steps_to_rescue.append(t + 1)
                victims_left_prev = victims_left
            if any(list(terms.values())) or any(list(trunc.values())):
                break
        returns.append(total)
        rescues.append(ep_rescues)
        if int(infos[env.agents[0]]["victims_left"]) == 0:
            success += 1
        print(f"Eval Ep {ep}: return={total:.2f} rescues={ep_rescues} left={int(infos[env.agents[0]]['victims_left'])}")

    print("\nSUMMARY")
    print(f"Episodes={episodes}")
    print(f"Avg Return={np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Avg Rescues={np.mean(rescues):.2f}")
    print(f"Success Rate={(success/episodes)*100:.1f}% (all victims rescued)")
    if steps_to_rescue:
        print(f"Median steps per rescue={int(np.median(steps_to_rescue))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    evaluate(episodes=args.episodes, agents=args.agents)
