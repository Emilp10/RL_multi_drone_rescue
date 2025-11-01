from __future__ import annotations
import os
from typing import Dict, Any, List, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import torch
import csv
import os.path as osp

from .environment.rescue_env import RescueConfig, parallel_env, ACTION_MEANINGS
from .agents.qmix_drqn_agent import DRQNAgent, DRQNConfig
from .agents.qmix_agent import QMIXAgent, QMIXConfig
from .evaluate import evaluate_agents

app = FastAPI(title="Multi-Drone Rescue Server")

# CORS for local development (Next.js, Vite)
# Allow all origins for development - restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Global session state (single session for demo)
SESSION: Dict[str, Any] = {
    "env": None,
    "agent": None,
    "greedy": True,
    "planner_assist": False,
    "versus": False,
    "manual_agent": 0,
    "record_demo": False,
    "human_path": [],  # list of (x,y)
}

class ResetReq(BaseModel):
    agents: int = 3
    greedy: bool = True
    seed: int | None = 123

class StepReq(BaseModel):
    greedy: bool = True


class OptionsReq(BaseModel):
    planner_assist: bool | None = None
    versus: bool | None = None
    manual_agent: int | None = None
    record_demo: bool | None = None


class ToggleObstacleReq(BaseModel):
    x: int
    y: int


class ManualStepReq(BaseModel):
    action: int
    greedy: bool = True


class RandomizeReq(BaseModel):
    agents: int | None = None
    greedy: bool | None = None
    seed: int | None = None


def encode_state(env) -> Dict[str, Any]:
    pos = {aid: list(map(int, env.pos[aid])) for aid in env.agents}
    victims = np.argwhere(env.victims == 1).tolist()
    obstacles = np.argwhere(env.obstacles == 1).tolist()
    return {
        "grid_size": env.grid_size,
        "agents": env.agents,
        "pos": pos,
        "victims": victims,
        "obstacles": obstacles,
        "t": env.t,
        "victims_left": int(env.victims.sum()),
    }


def _ckpt_path(n_agents: int) -> str:
    return osp.join("checkpoints", f"drqn_agents_{n_agents}", "final", "qmix_drqn.pt")


def _logs_path(n_agents: int | None = None) -> str | None:
    logs_dir = "logs"
    if n_agents == 1:
        p = osp.join(logs_dir, "training_log_agents_1.csv")
        return p if osp.exists(p) else None
    # prefer explicit 3-agent log, else fallback to default
    p3 = osp.join(logs_dir, "training_log_agents_3.csv")
    if osp.exists(p3):
        return p3
    p = osp.join(logs_dir, "training_log.csv")
    return p if osp.exists(p) else None


def _valid_action(mask: List[float], a: int) -> bool:
    return 0 <= a < len(mask) and bool(mask[a])


def _a_star_next(env, start: Tuple[int, int]) -> Tuple[int, int] | None:
    """Return the next cell toward shortest path to nearest victim using BFS (grid is small)."""
    import collections
    if env.victims.sum() == 0:
        return None
    n = env.grid_size
    victims = set(map(tuple, np.argwhere(env.victims == 1)))
    q = collections.deque()
    q.append(start)
    prev = {start: None}
    while q:
        x, y = q.popleft()
        if (x, y) in victims:
            # reconstruct one step
            cur = (x, y)
            while prev[cur] is not None and prev[cur] != start:
                cur = prev[cur]
            return cur
        for a, (dx, dy) in ACTION_MEANINGS.items():
            if a == 4:
                continue
            nx, ny = x + dx, y + dy
            if (0 <= nx < n and 0 <= ny < n and env.obstacles[nx, ny] == 0 and (nx, ny) not in prev):
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return None

@app.get("/")
def root():
    return {
        "message": "Multi-Drone Rescue API",
        "version": "1.0",
        "endpoints": ["/reset", "/step", "/manual_step", "/toggle_obstacle", "/options", "/randomize", "/info", "/metrics"]
    }

@app.post("/reset")
def reset(req: ResetReq):
    cfg = RescueConfig(grid_size=10, num_agents=req.agents, num_victims=6, obstacle_density=0.15, fov_radius=2, max_steps=200, seed=req.seed)
    env = parallel_env(cfg)
    obs, infos = env.reset()
    obs_dim = len(next(iter(obs.values())))
    state_dim = len(env.state())

    ckpt = os.path.join("checkpoints", f"drqn_agents_{req.agents}", "final", "qmix_drqn.pt")
    if os.path.exists(ckpt):
        # Load checkpoint config to instantiate matching architecture
        try:
            ckpt_data = torch.load(ckpt, map_location="cpu")
            ckpt_cfg = ckpt_data.get("cfg", {}) if isinstance(ckpt_data, dict) else {}
        except Exception:
            ckpt_cfg = {}
        agent = DRQNAgent(DRQNConfig(
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=env.num_actions,
            n_agents=req.agents,
            device="cpu",
            hidden_dim=int(ckpt_cfg.get("hidden_dim", 128)),
            mixer_hidden_dim=int(ckpt_cfg.get("mixer_hidden_dim", 64)),
        ))
        agent.load(os.path.dirname(ckpt))
        agent.epsilon = 0.0 if req.greedy else 0.1
    else:
        agent = QMIXAgent(QMIXConfig(obs_dim=obs_dim, state_dim=state_dim, n_actions=env.num_actions, n_agents=req.agents, device="cpu"))

    SESSION["env"] = env
    SESSION["agent"] = agent
    SESSION["greedy"] = req.greedy
    # reset session extras
    SESSION["planner_assist"] = SESSION.get("planner_assist", False)
    SESSION["versus"] = SESSION.get("versus", False)
    SESSION["manual_agent"] = SESSION.get("manual_agent", 0)
    SESSION["record_demo"] = SESSION.get("record_demo", False)
    SESSION["human_path"] = []
    return {"ok": True, "state": encode_state(env)}

@app.post("/step")
def step(req: StepReq):
    env = SESSION.get("env")
    agent = SESSION.get("agent")
    if env is None or agent is None:
        return {"ok": False, "error": "Not initialized; call /reset first"}

    # act (auto mode)
    obs, infos = {aid: env._obs(aid) for aid in env.agents}, {aid: {"action_mask": env._action_mask(aid)} for aid in env.agents}
    masks = {aid: infos[aid]["action_mask"] for aid in env.agents}
    actions = agent.act(obs, explore=not req.greedy, action_masks=masks)

    # planner assist override
    if SESSION.get("planner_assist", False):
        for aid in env.agents:
            # if agent stayed too long or chose invalid, nudge toward nearest victim
            if env.stay_steps.get(aid, 0) >= 3 or not _valid_action(list(masks[aid]), int(actions[aid])):
                start = env.pos[aid]
                nxt = _a_star_next(env, start)
                if nxt is not None:
                    dx, dy = nxt[0] - start[0], nxt[1] - start[1]
                    for a, (adx, ady) in ACTION_MEANINGS.items():
                        if (adx, ady) == (dx, dy) and _valid_action(list(masks[aid]), a):
                            actions[aid] = a
                            break
    next_obs, rewards, terms, truncs, infos2 = env.step(actions)
    done = any(list(terms.values())) or any(list(truncs.values()))

    return {
        "ok": True,
        "state": encode_state(env),
        "rewards": rewards,
        "done": done,
    }


@app.post("/manual_step")
def manual_step(req: ManualStepReq):
    env = SESSION.get("env")
    agent = SESSION.get("agent")
    if env is None or agent is None:
        return {"ok": False, "error": "Not initialized; call /reset first"}
    idx = int(SESSION.get("manual_agent", 0))
    idx = max(0, min(idx, len(env.agents)-1))
    manual_id = env.agents[idx]

    # build actions: manual for selected, policy for rest
    obs = {aid: env._obs(aid) for aid in env.agents}
    masks = {aid: env._action_mask(aid) for aid in env.agents}
    policy_actions = SESSION["agent"].act(obs, explore=not req.greedy, action_masks=masks)
    actions = policy_actions.copy()
    if _valid_action(list(masks[manual_id]), int(req.action)):
        actions[manual_id] = int(req.action)
    else:
        actions[manual_id] = 4  # stay if invalid

    # record demo if requested
    if SESSION.get("record_demo", False):
        demos_dir = "logs"
        os.makedirs(demos_dir, exist_ok=True)
        path = osp.join(demos_dir, f"demos_agents_{env.num_agents}.csv")
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            # store t, agent_index, x, y, action
            x, y = env.pos[manual_id]
            w.writerow([env.t, idx, x, y, int(req.action)])
        # store path for overlay
        SESSION["human_path"].append(tuple(env.pos[manual_id]))

    _, rewards, terms, truncs, _ = env.step(actions)
    done = any(list(terms.values())) or any(list(truncs.values()))
    return {"ok": True, "state": encode_state(env), "rewards": rewards, "done": done}


@app.post("/toggle_obstacle")
def toggle_obstacle(req: ToggleObstacleReq):
    env = SESSION.get("env")
    if env is None:
        return {"ok": False, "error": "Not initialized"}
    x, y = int(req.x), int(req.y)
    if not (0 <= x < env.grid_size and 0 <= y < env.grid_size):
        return {"ok": False, "error": "Out of bounds"}
    # don't allow toggling where agents stand
    if (x, y) in env.pos.values():
        return {"ok": False, "error": "Cell occupied by agent"}
    # don't allow placing obstacle on a victim
    if env.victims[x, y] == 1 and env.obstacles[x, y] == 0:
        return {"ok": False, "error": "Cell has a victim"}
    env.obstacles[x, y] = 0 if env.obstacles[x, y] == 1 else 1
    return {"ok": True, "state": encode_state(env)}


@app.post("/options")
def update_options(req: OptionsReq):
    for k in ["planner_assist", "versus", "manual_agent", "record_demo"]:
        v = getattr(req, k)
        if v is not None:
            SESSION[k] = v
    return {"ok": True, "options": {k: SESSION[k] for k in ["planner_assist", "versus", "manual_agent", "record_demo"]}}


@app.post("/randomize")
def randomize(req: RandomizeReq):
    env = SESSION.get("env")
    agent = SESSION.get("agent")
    if env is None or agent is None:
        # if not initialized, fallback to reset
        r = ResetReq(agents=req.agents or 3, greedy=SESSION.get("greedy", True), seed=req.seed or np.random.randint(0, 10_000))
        return reset(r)
    seed = req.seed if req.seed is not None else int(np.random.randint(0, 1_000_000))
    obs, _ = env.reset(seed=seed)
    SESSION["greedy"] = bool(req.greedy) if req.greedy is not None else SESSION.get("greedy", True)
    SESSION["human_path"] = []
    return {"ok": True, "state": encode_state(env)}


@app.get("/info")
def info():
    env = SESSION.get("env")
    agent = SESSION.get("agent")
    session = {
        "initialized": env is not None and agent is not None,
        "greedy": SESSION.get("greedy", True),
        "n_agents": getattr(env, "num_agents", None) if env is not None else None,
        "grid_size": getattr(env, "grid_size", None) if env is not None else None,
        "agent_type": agent.__class__.__name__ if agent is not None else None,
    }
    availability = {
        "checkpoints": {
            "drqn_agents_1": osp.exists(_ckpt_path(1)),
            "drqn_agents_3": osp.exists(_ckpt_path(3)),
        },
        "logs": {
            "default": _logs_path(None) is not None,
            "agents_1": _logs_path(1) is not None,
        },
    }
    return {"ok": True, "session": session, "availability": availability}


@app.get("/metrics")
def metrics(agents: int | None = None):
    path = _logs_path(agents)
    if path is None:
        return {"ok": False, "error": "No logs found for requested agents"}

    episodes, returns, losses, rescued, left = [], [], [], [], []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(float(row.get("episode", 0))))
                returns.append(float(row.get("return", 0.0)))
                losses.append(float(row.get("loss", 0.0)))
                rescued.append(int(float(row.get("victims_rescued", 0))))
                left.append(int(float(row.get("victims_left", 0))))
    except Exception as e:
        return {"ok": False, "error": f"Failed to read logs: {e}"}

    n = len(episodes)
    if n == 0:
        return {"ok": False, "error": "Empty log file"}

    success = sum(1 for v in left if v == 0)
    avg_return = sum(returns) / n
    avg_rescued = sum(rescued) / n
    summary = {
        "episodes": n,
        "avg_return": avg_return,
        "success_rate": success / n,
        "avg_rescued": avg_rescued,
        "path": path,
    }

    return {
        "ok": True,
        "series": {
            "episode": episodes,
            "return": returns,
            "loss": losses,
            "victims_rescued": rescued,
            "victims_left": left,
        },
        "summary": summary,
    }


def _ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n <= 1:
        return (0.0, 0.0)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1))
    half = 1.96 * s / (n ** 0.5)
    return (max(0.0, m - half), m + half)


@app.get("/evaluation")
def evaluation(agents: int = 3):
    """Return evaluation summary for a given number of agents.
    If an eval log exists under logs/eval_agents_{agents}.csv, aggregate it.
    Otherwise, run a short evaluation (episodes=20) and return its summary.
    """
    path = osp.join("logs", f"eval_agents_{agents}.csv")
    if osp.exists(path):
        # aggregate existing file
        episodes, returns, rescued, left = [], [], [], []
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    episodes.append(int(float(row.get("episode", 0))))
                    returns.append(float(row.get("return", 0.0)))
                    rescued.append(int(float(row.get("victims_rescued", 0))))
                    left.append(int(float(row.get("victims_left", 0))))
        except Exception as e:
            return {"ok": False, "error": f"Failed to read eval logs: {e}"}

        n = len(episodes)
        if n == 0:
            return {"ok": False, "error": "Empty eval log"}
        success = sum(1 for v in left if v == 0)
        avg_return = sum(returns) / n
        avg_rescued = (sum(rescued) / n) if n else 0.0
        r_lo, r_hi = _ci95(returns)
        z_lo, z_hi = _ci95([float(x) for x in rescued])
        # checkpoint info
        ckpt_dir = osp.join("checkpoints", f"drqn_agents_{agents}", "final")
        ckpt_file = osp.join(ckpt_dir, "qmix_drqn.pt")
        ckpt_mtime = osp.getmtime(ckpt_file) if osp.exists(ckpt_file) else None
        summary = {
            "episodes": n,
            "avg_return": avg_return,
            "avg_return_ci95": [r_lo, r_hi],
            "success_rate": success / n,
            "successes": success,
            "avg_rescued": avg_rescued,
            "avg_rescued_ci95": [z_lo, z_hi],
            "path": path,
            "ckpt_dir": ckpt_dir,
            "ckpt_file": ckpt_file,
            "ckpt_mtime": ckpt_mtime,
        }
        return {"ok": True, "summary": summary}

    # otherwise run a quick evaluation
    result = evaluate_agents(agents, episodes=20)
    # Attach checkpoint info to the summary
    ckpt_dir = osp.join("checkpoints", f"drqn_agents_{agents}", "final")
    ckpt_file = osp.join(ckpt_dir, "qmix_drqn.pt")
    ckpt_mtime = osp.getmtime(ckpt_file) if osp.exists(ckpt_file) else None
    if isinstance(result, dict) and "summary" in result:
        result["summary"].update({
            "ckpt_dir": ckpt_dir,
            "ckpt_file": ckpt_file,
            "ckpt_mtime": ckpt_mtime,
        })
    return result


def _summarize_training_log(path: str) -> dict[str, Any]:
    episodes, returns, losses, rescued, left = [], [], [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(float(row.get("episode", 0))))
            returns.append(float(row.get("return", 0.0)))
            losses.append(float(row.get("loss", 0.0)))
            rescued.append(int(float(row.get("victims_rescued", 0))))
            left.append(int(float(row.get("victims_left", 0))))
    n = len(episodes)
    if n == 0:
        return {"ok": False, "error": "Empty log"}
    # last window
    k = min(20, n)
    win_returns = returns[-k:]
    win_losses = losses[-k:]
    win_rescued = rescued[-k:]
    win_left = left[-k:]
    success_20 = sum(1 for v in win_left if v == 0) / float(k)
    return {
        "ok": True,
        "episodes": n,
        "return_last": returns[-1],
        "loss_last": losses[-1],
        "rescued_last": rescued[-1],
        "left_last": left[-1],
        "avg_return_20": float(np.mean(win_returns)),
        "avg_loss_20": float(np.mean(win_losses)),
        "avg_rescued_20": float(np.mean(win_rescued)),
        "success_20": success_20,
        "path": path,
    }


@app.get("/status")
def status(agents: int = 3):
    """Summarize recent training health for the UI."""
    path = _logs_path(agents)
    if not path:
        return {"ok": False, "error": "No training log available"}
    try:
        summary = _summarize_training_log(path)
    except Exception as e:
        return {"ok": False, "error": f"Failed to read training log: {e}"}
    # attach checkpoint info
    ckpt_dir = osp.join("checkpoints", f"drqn_agents_{agents}", "final")
    ckpt_file = osp.join(ckpt_dir, "qmix_drqn.pt")
    ckpt_mtime = osp.getmtime(ckpt_file) if osp.exists(ckpt_file) else None
    summary.update({
        "ckpt_dir": ckpt_dir,
        "ckpt_file": ckpt_file,
        "ckpt_mtime": ckpt_mtime,
    })
    return summary


@app.post("/run_evaluation")
def run_evaluation(req: ResetReq):
    agents = int(req.agents)
    episodes = 50
    return evaluate_agents(agents, episodes=episodes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
