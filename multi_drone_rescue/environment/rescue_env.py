"""
Multi-Drone Search & Rescue environment (PettingZoo Parallel API style)

Grid codes:
 0 empty, 1 obstacle, 2 victim, 3 drone (rendering only)

Observations per agent (vector):
 [x, y, rescued_count, local_patch_flat...]
 where local_patch is an integer grid of size determined by fov_radius (Manhattan)
 mapped to values: empty=0, obstacle=1, victim=2, drone=3 (drone appears only for visibility consistency)

Global state: concatenation of all agents' (x,y) plus flattened full victim map (binary) and obstacle map (binary).
This is a compact but informative representation for centralized training.

Rewards:
 +10 for finding a victim (removing it)
 -5 for collision with another drone (same cell after move)
 -2 for trying to move into an obstacle or outside (agent stays)
 -0.1 per move
 +50 when all victims rescued (episode end)

This file does not depend on pettingzoo at import-time to make local development easier.
If PettingZoo is installed, this follows the expected ParallelEnv interface.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from environment.utils import in_bounds, fov_cells


@dataclass
class RescueConfig:
    grid_size: int = 10
    num_agents: int = 3
    num_victims: int = 6
    obstacle_density: float = 0.15  # fraction of cells
    fov_radius: int = 2
    max_steps: int = 200
    seed: Optional[int] = None


ACTION_MEANINGS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
    4: (0, 0),   # stay
}


class RescueParallelEnv:
    metadata = {"render_modes": ["human"], "name": "multi_drone_rescue"}

    def __init__(self, config: RescueConfig = RescueConfig()):
        self.cfg = config
        self._rng = random.Random(config.seed)
        self.np_random = np.random.RandomState(config.seed)

        self.grid_size = config.grid_size
        self.num_agents = config.num_agents
        self.max_steps = config.max_steps
        self.fov_radius = config.fov_radius

        # Runtime state
        self.t = 0
        self.agents: List[str] = [f"agent_{i}" for i in range(self.num_agents)]
        self.pos: Dict[str, Tuple[int, int]] = {}
        self.rescued: Dict[str, int] = {aid: 0 for aid in self.agents}
        self.victims: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.obstacles: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.visited: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.stay_steps: Dict[str, int] = {aid: 0 for aid in self.agents}
        self.trail: Dict[str, List[Tuple[int, int]]] = {aid: [] for aid in self.agents}
        self.terminated: Dict[str, bool] = {aid: False for aid in self.agents}
        self.truncated: Dict[str, bool] = {aid: False for aid in self.agents}

        # Precompute fixed local-patch offsets (Manhattan diamond), ensures fixed obs length
        self._patch_offsets = self._compute_patch_offsets()

        # Spaces (defined lazily as tuples to avoid gym dependency)
        # action space: Discrete(5)
        self.action_spaces = {aid: (5,) for aid in self.agents}
        # observation length calculation
        lp_len = len(self._patch_offsets)
        obs_len = 3 + lp_len + lp_len + self.num_agents  # x,y,rescued + local patch + visited patch + agent id one-hot
        self.observation_spaces = {aid: (obs_len,) for aid in self.agents}

        self.reset(seed=config.seed)

    # ------------- Helpers -------------
    def _compute_patch_offsets(self):
        # Stable order of relative offsets within a Manhattan-radius diamond
        r = self.fov_radius
        offsets = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    offsets.append((dx, dy))
        return offsets

    def _random_empty_cell(self) -> Tuple[int, int]:
        while True:
            x = self._rng.randrange(self.grid_size)
            y = self._rng.randrange(self.grid_size)
            if self.obstacles[x, y] == 0 and self.victims[x, y] == 0 and (x, y) not in self.pos.values():
                return (x, y)

    def _place_random(self):
        # obstacles
        total_cells = self.grid_size * self.grid_size
        num_obstacles = int(self.cfg.obstacle_density * total_cells)
        self.obstacles[:, :] = 0
        for _ in range(num_obstacles):
            x = self._rng.randrange(self.grid_size)
            y = self._rng.randrange(self.grid_size)
            self.obstacles[x, y] = 1

        # victims
        self.victims[:, :] = 0
        v = 0
        while v < self.cfg.num_victims:
            x = self._rng.randrange(self.grid_size)
            y = self._rng.randrange(self.grid_size)
            if self.obstacles[x, y] == 0 and self.victims[x, y] == 0:
                self.victims[x, y] = 1
                v += 1

        # agents
        for aid in self.agents:
            self.pos[aid] = self._random_empty_cell()
            self.rescued[aid] = 0
        # reset visited map
        self.visited[:, :] = 0
        # reset stay counters
        self.stay_steps = {aid: 0 for aid in self.agents}

    def _local_patch(self, center: Tuple[int, int]) -> np.ndarray:
        cx, cy = center
        # Map to 0 empty, 1 obstacle, 2 victim, 3 drone
        vals = []
        for dx, dy in self._patch_offsets:
            x, y = cx + dx, cy + dy
            if not in_bounds((x, y), self.grid_size):
                # pad out-of-bounds as obstacle to discourage moving off-map
                vals.append(1)
            else:
                if self.obstacles[x, y] == 1:
                    vals.append(1)
                elif self.victims[x, y] == 1:
                    vals.append(2)
                elif (x, y) in self.pos.values():
                    vals.append(3)
                else:
                    vals.append(0)
        return np.array(vals, dtype=np.float32)

    def _local_visited_patch(self, center: Tuple[int, int]) -> np.ndarray:
        cx, cy = center
        vals = []
        for dx, dy in self._patch_offsets:
            x, y = cx + dx, cy + dy
            if not in_bounds((x, y), self.grid_size):
                vals.append(1.0)  # treat OOB as visited to avoid bias off-map
            else:
                vals.append(float(self.visited[x, y] > 0))
        return np.array(vals, dtype=np.float32)

    def _obs(self, aid: str) -> np.ndarray:
        x, y = self.pos[aid]
        lp = self._local_patch((x, y))
        vp = self._local_visited_patch((x, y))
        onehot = np.zeros((self.num_agents,), dtype=np.float32)
        onehot[self.agents.index(aid)] = 1.0
        return np.concatenate([
            np.array([x, y, self.rescued[aid]], dtype=np.float32),
            lp,
            vp,
            onehot,
        ])

    def _global_state(self) -> np.ndarray:
        # concat agent positions + flattened victims + flattened obstacles
        pos_vec = []
        for aid in self.agents:
            xy = self.pos[aid]
            pos_vec.extend([xy[0], xy[1]])
        pos_vec = np.array(pos_vec, dtype=np.float32)
        victims_flat = self.victims.astype(np.float32).flatten()
        obstacles_flat = self.obstacles.astype(np.float32).flatten()
        return np.concatenate([pos_vec, victims_flat, obstacles_flat])

    def _action_mask(self, aid: str) -> np.ndarray:
        """Return a binary mask of shape (5,) for valid actions.
        1 = valid, 0 = invalid. Stay is always valid.
        """
        x, y = self.pos[aid]
        mask = np.ones((5,), dtype=np.float32)
        # up
        nx, ny = x - 1, y
        if not in_bounds((nx, ny), self.grid_size) or self.obstacles[nx, ny] == 1:
            mask[0] = 0.0
        # down
        nx, ny = x + 1, y
        if not in_bounds((nx, ny), self.grid_size) or self.obstacles[nx, ny] == 1:
            mask[1] = 0.0
        # left
        nx, ny = x, y - 1
        if not in_bounds((nx, ny), self.grid_size) or self.obstacles[nx, ny] == 1:
            mask[2] = 0.0
        # right
        nx, ny = x, y + 1
        if not in_bounds((nx, ny), self.grid_size) or self.obstacles[nx, ny] == 1:
            mask[3] = 0.0
        # stay always valid
        mask[4] = 1.0
        return mask

    # ------------- API -------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)
            self.np_random.seed(seed)
        self.t = 0
        self.terminated = {aid: False for aid in self.agents}
        self.truncated = {aid: False for aid in self.agents}
        self._place_random()
        observations = {aid: self._obs(aid) for aid in self.agents}
        victims_left = int(self.victims.sum())
        infos = {aid: {"state": self._global_state(), "action_mask": self._action_mask(aid), "victims_left": victims_left} for aid in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        assert set(actions.keys()) == set(self.agents)
        self.t += 1

        # Propose moves
        desired: Dict[str, Tuple[int, int]] = {}
        penalties = {aid: 0.0 for aid in self.agents}
        # for shaping: distance to nearest victim before and after
        def nearest_victim_dist(p: Tuple[int, int]) -> Optional[int]:
            if self.victims.sum() == 0:
                return None
            x0, y0 = p
            # compute min manhattan distance to any victim
            vx, vy = np.where(self.victims == 1)
            if vx.size == 0:
                return None
            d = np.min(np.abs(vx - x0) + np.abs(vy - y0))
            return int(d)
        prev_dists = {aid: nearest_victim_dist(self.pos[aid]) for aid in self.agents}
        for aid, a in actions.items():
            dx, dy = ACTION_MEANINGS.get(int(a), (0, 0))
            x, y = self.pos[aid]
            nx, ny = x + dx, y + dy
            # invalid move -> stay and penalize (softer)
            if not in_bounds((nx, ny), self.grid_size) or self.obstacles[nx, ny] == 1:
                desired[aid] = (x, y)
                if (dx, dy) != (0, 0):
                    penalties[aid] -= 0.5
            else:
                desired[aid] = (nx, ny)

        # Resolve collisions: if multiple choose same cell, all penalized and stay
        cell_to_aids: Dict[Tuple[int, int], List[str]] = {}
        for aid, dpos in desired.items():
            cell_to_aids.setdefault(dpos, []).append(aid)
        new_pos = self.pos.copy()
        for cell, ids in cell_to_aids.items():
            if len(ids) == 1:
                new_pos[ids[0]] = cell
            else:
                # collision (softer)
                for aid in ids:
                    penalties[aid] -= 2.0
                    # stay in place
                    new_pos[aid] = self.pos[aid]

        old_pos = self.pos.copy()
        self.pos = new_pos
        # update trails
        for aid in self.agents:
            self.trail[aid].append(self.pos[aid])
            if len(self.trail[aid]) > 20:
                self.trail[aid].pop(0)

        # Victim collection
        found_counts = {aid: 0 for aid in self.agents}
        for aid in self.agents:
            x, y = self.pos[aid]
            if self.victims[x, y] == 1:
                self.victims[x, y] = 0
                found_counts[aid] += 1
                self.rescued[aid] += 1

        # Rewards with improved shaping
        rewards = {aid: 0.0 for aid in self.agents}
        for aid in self.agents:
            rewards[aid] += penalties[aid]
            # step cost (very small to encourage exploration)
            rewards[aid] -= 0.01
            # victim reward (major positive signal)
            if found_counts[aid] > 0:
                rewards[aid] += 15.0 * found_counts[aid]  # Increased from 10.0
            # idle penalty: stayed intentionally without rescuing
            if int(actions[aid]) == 4 and found_counts[aid] == 0:
                rewards[aid] -= 0.1  # Increased penalty for unnecessary staying

        # Enhanced exploration bonus and proximity shaping
        for aid in self.agents:
            x, y = self.pos[aid]
            # first visit to a cell yields bonus (encourage exploration)
            if self.visited[x, y] == 0:
                rewards[aid] += 0.3  # Increased from 0.2
                self.visited[x, y] = 1
            # proximity shaping: reward getting closer to victims
            if self.victims.sum() > 0:
                before = prev_dists[aid]
                after = nearest_victim_dist((x, y))
                if before is not None and after is not None:
                    if after < before:
                        # Reward moving closer (scaled by distance improvement)
                        delta = min(3, max(1, before - after))
                        rewards[aid] += 0.05 * float(delta)  # Increased from 0.02
                    elif after > before:
                        # Small penalty for moving away from victims
                        rewards[aid] -= 0.01
            # stagnation penalty: too many consecutive non-moves
            if self.pos[aid] == old_pos[aid]:
                self.stay_steps[aid] += 1
            else:
                self.stay_steps[aid] = 0
            if self.stay_steps[aid] >= 10:
                rewards[aid] -= 0.2
                self.stay_steps[aid] = 0
            # light revisit penalty for revisiting recent trail
            if self.trail[aid].count((x, y)) > 1:
                rewards[aid] -= 0.02

        done_all = self.victims.sum() == 0
        if done_all:
            for aid in self.agents:
                rewards[aid] += 50.0
                self.terminated[aid] = True
        if self.t >= self.max_steps:
            for aid in self.agents:
                self.truncated[aid] = True

        terminations = self.terminated.copy()
        truncations = self.truncated.copy()
        observations = {aid: self._obs(aid) for aid in self.agents}
        victims_left = int(self.victims.sum())
        infos = {aid: {"state": self._global_state(), "action_mask": self._action_mask(aid), "victims_left": victims_left} for aid in self.agents}
        return observations, rewards, terminations, truncations, infos

    # Rendering support via pygame is provided in visualizer/grid_display.py
    def render(self):
        pass

    def state(self) -> np.ndarray:
        return self._global_state()

    @property
    def num_actions(self) -> int:
        return 5


def parallel_env(config: RescueConfig = RescueConfig()) -> RescueParallelEnv:
    return RescueParallelEnv(config)
