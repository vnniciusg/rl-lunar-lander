# 🚀 Deep Q-Network (DQN) for Lunar Lander

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2+-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete implementation of Deep Q-Network (DQN) reinforcement learning algorithm for solving the OpenAI Gymnasium LunarLander-v3 environment. This project demonstrates how an AI agent can learn to successfully land a spacecraft using deep reinforcement learning techniques.

![Lunar Lander Demo](https://gymnasium.farama.org/_images/lunar_lander.gif)

## 🎯 Project Overview

This repository contains a from-scratch implementation of the DQN algorithm with the following key features:

- **🧠 Deep Q-Network Architecture**: Neural network that learns Q-value estimates for state-action pairs
- **💾 Experience Replay**: Stores and samples past experiences for stable learning
- **🎲 Epsilon-Greedy Exploration**: Balances exploration vs exploitation with decaying randomness
- **🎯 Target Network**: Separate target network for stable Q-learning updates
- **⚙️ Configurable Hyperparameters**: Pydantic-based configuration management
- **📊 Training Visualization**: Comprehensive plotting and progress tracking
- **🎬 Video Recording**: Optional episode recording for performance analysis

## 🏗️ Architecture

The project follows a clean, modular architecture:

```
rl-lunar-lander/
├── 📄 README.md              # Project documentation
├── 🐍 dqn.py                 # Deep Q-Network neural network implementation
├── 🤖 agent.py               # DQN agent with experience replay and training logic
├── ⚙️ config.py              # Pydantic-based configuration management
├── 🏃 train.py               # Training loop and environment interaction
├── 📋 pyproject.toml         # Project dependencies and metadata
├── 📜 LICENSE                # MIT License
└── 📁 videos/                # Generated training videos (created during training)
```

### Key Components

- **`DeepQNetwork`**: PyTorch neural network with 3 fully connected layers
- **`Agent`**: Main DQN agent handling action selection, experience storage, and learning
- **`DQNConfig`**: Pydantic model for hyperparameter validation and management
- **`train_agent()`**: Complete training pipeline with logging and visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/vnniciusg/rl-lunar-lander.git
   cd rl-lunar-lander
   ```

2. **Install dependencies**:

   Using uv (recommended):

   ```bash
   uv sync
   ```

   Or using pip:

   ```bash
   pip install -e .
   ```

3. **Run training**:
   ```bash
   python train.py
   ```

## 🎮 Usage

### Basic Training

```python
from train import train_agent

# Train for 1000 episodes (default)
scores, best_score = train_agent()

# Train for custom number of episodes
scores, best_score = train_agent(n_episodes=500)
```

### Custom Configuration

```python
from config import DQNConfig
from agent import Agent

# Create custom configuration
config = DQNConfig(
    lr=5e-4,           # Learning rate
    gamma=0.95,        # Discount factor
    epsilon=1.0,       # Initial exploration rate
    eps_min=0.01,      # Minimum exploration rate
    eps_dec=0.995,     # Exploration decay rate
    mem_size=10000,    # Replay buffer size
    batch_size=64,     # Training batch size
    target_update=1000 # Target network update frequency
)

# Use custom config with agent
agent = Agent(input_dims=(8,), n_actions=4, config=config)
```

## 📊 Training Results

The training process generates several outputs:

- **📈 Performance Plots**: `training_results.png` showing learning curves
- **🎬 Episode Videos**: Recorded in `videos/` directory (optional)
- **📝 Training Logs**: Detailed progress information

### Example Training Output

```
INFO - EPISODE 0, Reward: -150.2, Best: -150.2, Epsilon: 1.000
INFO - EPISODE 10, Reward: -89.4, Best: -45.6, Epsilon: 0.950
INFO - EPISODE 100, Reward: 120.5, Best: 145.8, Epsilon: 0.605
...
SUCCESS - Training completed!
INFO - Final performance: Best reward = 248.7
INFO - Graph saved as 'training_results.png'
```

## ⚙️ Configuration

The project uses Pydantic for configuration management with automatic validation:

| Parameter       | Default | Description                        |
| --------------- | ------- | ---------------------------------- |
| `lr`            | 1e-3    | Learning rate for neural network   |
| `gamma`         | 0.99    | Discount factor for future rewards |
| `epsilon`       | 1.0     | Initial exploration rate           |
| `eps_min`       | 0.01    | Minimum exploration rate           |
| `eps_dec`       | 0.995   | Exploration decay factor           |
| `mem_size`      | 1000    | Experience replay buffer size      |
| `batch_size`    | 64      | Training batch size                |
| `target_update` | 1000    | Target network update frequency    |

## 🧪 Environment Details

**LunarLander-v3** specifications:

- **Observation Space**: 8-dimensional continuous vector
  - Position (x, y)
  - Velocity (x, y)
  - Angle and angular velocity
  - Left and right leg contact booleans
- **Action Space**: 4 discrete actions
  - 0: Do nothing
  - 1: Fire left orientation engine
  - 2: Fire main engine
  - 3: Fire right orientation engine
- **Reward**: Landing successfully gives +100-140 points, crashing gives -100

## 📈 Performance Metrics

The agent typically achieves:

- **Convergence**: ~500-800 episodes
- **Target Performance**: 200+ average reward over 100 episodes
- **Success Rate**: 90%+ successful landings after training

## 🛠️ Development

### Project Structure

```python
# Core DQN implementation
dqn.py          # Neural network architecture
agent.py        # Agent logic and training
config.py       # Configuration management

# Training and utilities
train.py        # Main training script
pyproject.toml  # Dependencies and project metadata
```

### Dependencies

- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[Gymnasium](https://gymnasium.farama.org/)**: Reinforcement learning environments
- **[Pydantic](https://pydantic.dev/)**: Configuration validation
- **[Loguru](https://loguru.readthedocs.io/)**: Enhanced logging
- **[Matplotlib](https://matplotlib.org/)**: Plotting and visualization

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN Paper
- [Gymnasium Documentation](https://gymnasium.farama.org/) - Environment Documentation
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Deep Learning Framework

---

**Built with ❤️ and lots of ☕ by [vin](https://github.com/vnniciusg)**
