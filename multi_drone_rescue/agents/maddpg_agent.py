"""
Minimal scaffold for a MADDPG agent. For this project, we prioritize QMIX for stability.
This file documents the intended API and raises NotImplementedError for unimplemented parts.
"""
from dataclasses import dataclass


@dataclass
class MADDPGConfig:
    obs_dim: int
    act_dim: int
    n_agents: int
    gamma: float = 0.99


class MADDPGAgent:
    def __init__(self, cfg: MADDPGConfig):
        self.cfg = cfg
        raise NotImplementedError(
            "MADDPG is not implemented in this starter. Use QMIX or extend this class."
        )
