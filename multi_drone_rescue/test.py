import os
import sys
import time
from typing import Dict

import numpy as np

from environment.rescue_env import RescueConfig, parallel_env
from agents.qmix_agent import QMIXAgent, QMIXConfig
from agents.qmix_drqn_agent import DRQNAgent, DRQNConfig
from visualizer.grid_display import GridDisplay


def main():
    cfg = RescueConfig(grid_size=10, num_agents=3, num_victims=6, obstacle_density=0.15, fov_radius=2, max_steps=200, seed=7)
    env = parallel_env(cfg)

    # derive dims
    obs_sample, _ = env.reset()
    n_agents = len(env.agents)
    obs_dim = len(next(iter(obs_sample.values())))
    state_dim = len(env.state())

    # load agent
    def build_env_and_agent(n_agents: int):
        cfg = RescueConfig(grid_size=10, num_agents=n_agents, num_victims=6, obstacle_density=0.15, fov_radius=2, max_steps=200, seed=7)
        env = parallel_env(cfg)
        obs_sample, _ = env.reset()
        obs_dim = len(next(iter(obs_sample.values())))
        state_dim = len(env.state())
        # Prefer DRQN (try long-trained first, then regular)
        ckpt_dirs = [
            os.path.join("checkpoints", f"drqn_agents_{n_agents}_long", "final"),
            os.path.join("checkpoints", f"drqn_agents_{n_agents}", "final")
        ]
        
        agent = DRQNAgent(DRQNConfig(obs_dim=obs_dim, state_dim=state_dim, n_actions=env.num_actions, n_agents=n_agents, device="cpu"))
        
        for ckpt_dir in ckpt_dirs:
            if os.path.exists(os.path.join(ckpt_dir, "qmix_drqn.pt")):
                agent.load(ckpt_dir)
                agent.epsilon = 0.0
                model_type = "long-trained" if "_long" in ckpt_dir else "regular"
                print(f"Loaded {model_type} DRQN model from {ckpt_dir}")
                return env, agent
        
        # Fallback to QMIX
        qm_ckpt = os.path.join("checkpoints", "final")
        qagent = QMIXAgent(QMIXConfig(obs_dim=obs_dim, state_dim=state_dim, n_actions=env.num_actions, n_agents=n_agents, device="cpu"))
        if os.path.exists(os.path.join(qm_ckpt, "qmix.pt")):
            qagent.load(qm_ckpt)
            qagent.epsilon = 0.0
            print("Loaded legacy QMIX model.")
            return env, qagent
        print("No trained model found. Using epsilon policy.")
        return env, agent

    env, agent = build_env_and_agent(3)

    display = GridDisplay(cfg.grid_size, cell_size=40)

    def reset_episode():
        o, info = env.reset()
        return o, info, 0.0, 0

    obs, infos, ep_return, step = reset_episode()
    episode = 1
    running = True
    import pygame
    paused = False
    greedy = True
    victims_left_prev = int(env.victims.sum())
    no_progress_steps = 0

    while running:
        # handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, infos, ep_return, step = reset_episode()
                    episode += 1
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_g:
                    greedy = not greedy
                elif event.key == pygame.K_1:
                    env, agent = build_env_and_agent(1)
                    obs, infos, ep_return, step = reset_episode()
                    episode += 1
                elif event.key == pygame.K_3:
                    env, agent = build_env_and_agent(3)
                    obs, infos, ep_return, step = reset_episode()
                    episode += 1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # panel buttons
                if display.clicked_reset(event.pos):
                    obs, infos, ep_return, step = reset_episode()
                    episode += 1
                elif display.click_in_grid(event.pos):
                    x, y = display.handle_click(event.pos)
                    # toggle victim if empty; else toggle obstacle
                    if env.victims[x, y] == 0 and env.obstacles[x, y] == 0:
                        env.victims[x, y] = 1
                    elif env.victims[x, y] == 1:
                        env.victims[x, y] = 0
                    else:
                        env.obstacles[x, y] = 1 - env.obstacles[x, y]

        # defaults to avoid referencing before assignment when paused
        terminations, truncations = {}, {}
        if not paused:
            masks = {aid: infos[aid].get("action_mask") for aid in env.agents}
            actions = agent.act(obs, explore=not greedy, action_masks=masks)
            obs, rewards, terminations, truncations, infos = env.step(actions)

            step += 1
            ep_return += float(sum(rewards.values()))

        victims_left = int(env.victims.sum())
        display.draw(env.obstacles, env.victims, env.pos, fps=8, stats={
            "episode": episode,
            "step": step,
            "ep_return": ep_return,
            "victims_left": victims_left,
            "mode": "Greedy" if greedy else "Explore",
        })

        # auto-reset if stuck (no change in victims left for too long)
        if not paused:
            if victims_left == victims_left_prev:
                no_progress_steps += 1
            else:
                no_progress_steps = 0
            victims_left_prev = victims_left
            if no_progress_steps >= 120:
                obs, infos, ep_return, step = reset_episode()
                episode += 1
                no_progress_steps = 0

        if any(list(terminations.values())) or any(list(truncations.values())):
            obs, infos, ep_return, step = reset_episode()
            episode += 1

    display.close()


if __name__ == "__main__":
    main()
