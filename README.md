# RL - Multi-Agent Reinforcement Learning Project

This project implements a multi-agent reinforcement learning system for drone rescue scenarios, featuring both training infrastructure and a web-based frontend for visualization and interaction.

## Project Structure

### Core Components

- **`multi_drone_rescue/`** - Main RL implementation with QMIX algorithm
- **`frontend/`** - React/TypeScript web interface for visualization
- **`logs/`** - Training logs and evaluation data
- **`checkpoints/`** - Saved model checkpoints from training runs

### Key Features

- **Multi-Agent Reinforcement Learning**: QMIX implementation for coordinated drone behavior
- **Interactive Environment**: Pygame-based simulator with real-time controls
- **Web Dashboard**: Modern React frontend for monitoring and analysis
- **Hyperparameter Tuning**: Optuna-based optimization system
- **Comprehensive Logging**: Detailed training and evaluation metrics

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)
- Git

### Installation

1. **Clone and setup Python environment:**
```bash
cd multi_drone_rescue
pip install -r requirements.txt
```

2. **Install frontend dependencies:**
```bash
cd frontend
npm install
```

### Running the System

1. **Train agents:**
```bash
python -m multi_drone_rescue.train
```

2. **Run interactive demo:**
```bash
python multi_drone_rescue/test.py
```

3. **Start web interface:**
```bash
cd frontend
npm run dev
```

4. **Start API server:**
```bash
python multi_drone_rescue/server.py
```

## Features

### Multi-Drone Rescue Environment
- **Cooperative Search & Rescue**: Multiple drones work together to find and rescue victims
- **Dynamic Obstacles**: Interactive environment with moveable obstacles
- **Real-time Visualization**: Watch agents learn and adapt in real-time
- **Smart Exploration**: First-visit bonuses and proximity-based rewards

### Training Infrastructure
- **QMIX Algorithm**: Centralized training with decentralized execution
- **Flexible Configuration**: Easy-to-modify hyperparameters and environment settings
- **Checkpoint System**: Automatic model saving and loading
- **Performance Tracking**: Comprehensive logging and metrics

### Web Interface
- **Modern UI**: Built with React, TypeScript, and Tailwind CSS
- **Real-time Monitoring**: Live training progress and metrics
- **Interactive Controls**: Pause, resume, and modify training parameters
- **Data Visualization**: Charts and graphs for performance analysis

## Configuration

### Environment Settings
Modify `RescueConfig` in the environment files:
- Grid size and layout
- Number of victims and obstacles
- Drone field of view
- Episode length limits

### Agent Parameters
Adjust `QMIXConfig` for training:
- Learning rates and schedules
- Exploration strategies
- Buffer sizes and batch parameters
- Network architectures

## Development

### Adding New Algorithms
The project structure supports easy addition of new RL algorithms:
1. Implement in `multi_drone_rescue/agents/`
2. Follow the existing agent interface
3. Update training scripts as needed

### Extending the Environment
The PettingZoo-compatible environment can be extended:
- Add new observation types
- Implement additional reward signals
- Create new scenario types

## Results and Analysis

Training logs and checkpoints are automatically saved:
- **Training metrics**: `logs/training_log_*.csv`
- **Evaluation results**: `logs/eval_*.csv`
- **Model checkpoints**: `checkpoints/` directory
- **Hyperparameter studies**: `optuna_studies.db`

## Contributing

This project demonstrates modern ML engineering practices:
- Modular, extensible codebase
- Comprehensive logging and monitoring
- Interactive visualization tools
- Automated hyperparameter optimization

## Technologies Used

- **Backend**: Python, PyTorch, PettingZoo, Pygame
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **Optimization**: Optuna for hyperparameter tuning
- **Visualization**: Custom Pygame renderer + web dashboard

## License

This project is for educational and research purposes.
