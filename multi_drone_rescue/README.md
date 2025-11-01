# Multi-Drone Search & Rescue Simulator (MARL)

Train multiple drones to find and rescue victims in a 2D grid using Multi-Agent Reinforcement Learning.

## Features
- PettingZoo-style parallel environment (`environment/rescue_env.py`)
- QMIX implementation (PyTorch) for centralized training with decentralized execution
- Pygame visualizer to watch drones explore; click to add obstacles/victims live
- Training and interactive demo scripts
- Smarter exploration: first-visit bonus, proximity shaping; idle/stagnation penalties to prevent freezing
- Interactive demo controls: Reset button/R, pause with Space, toggle Greedy/Explore with G, auto-reset if stuck

## Install
Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, if PyTorch install via pip fails, follow the instructions from https://pytorch.org/ and then install the rest.

## Train
Runs QMIX training and logs a CSV at `logs/training_log.csv`. Models saved under `checkpoints/`.

```bash
python -m multi_drone_rescue.train
```

Alternatively, from the project root:
```bash
python multi_drone_rescue/train.py
```

## Demo (Interactive)
Loads `checkpoints/final/qmix.pt` if present; otherwise runs with an exploratory policy.
- Left-click in grid:
	- Empty -> add victim
	- Victim -> remove victim
	- Otherwise -> toggle obstacle
- Right panel:
	- Reset button or press R to reset episode
	- Space to pause/resume
	- Press G to toggle Greedy vs Explore mode (shown in panel)
	- Auto-reset triggers if victims left doesn’t change for ~120 steps

```bash
python multi_drone_rescue/test.py
```

## Configurations
- Environment config in `RescueConfig` (grid size, victims, obstacle density, fov radius, max steps)
- Agent config in `QMIXConfig` (learning rate, epsilon schedule, buffer sizes)

## Folder Structure
```
multi_drone_rescue/
├── environment/
│   ├── rescue_env.py          # PettingZoo parallel-style environment
│   └── utils.py               # helpers
├── agents/
│   ├── qmix_agent.py          # Simple QMIX implementation
│   └── maddpg_agent.py        # Stub/Skeleton for MADDPG
├── visualizer/
│   └── grid_display.py        # pygame grid renderer
├── train.py                   # training loop
├── test.py                    # interactive demo
├── requirements.txt
└── README.md
```

## Notes and Future Work
- The QMIX here is simplified but works as a strong baseline. Add recurrent Q-nets (DRQN) for partial observability.
- Add battery constraints, communication costs, or weather slowdowns.
- Integrate Weights & Biases or TensorBoard for richer logging.
- Swap in RLlib's QMIX for production-grade scaling.
