# Reinforcement Learning Workshop

**Authors:** Konrad Fijałkowski & Fabian Fetter

A collection of reinforcement learning agents using Gymnasium and Stable-Baselines3.

## Project Structure

```
reinforcement_learning/
├── agents/                  # Agent scripts
│   ├── battlezone_agent.py
│   ├── humanoid_agent.py
│   ├── autonomous_car.py
│   └── lunar_lander_agent.py
├── models/                  # Trained model files (.zip)
├── media/                   # Recordings and videos
├── requirements.txt
└── README.md
```

## Agents

### 1. Battlezone Agent

- **Environment:** `ALE/BattleZone-v5` (Atari)
- **Algorithm:** DQN
- **Description:** Trains an agent to play the classic Atari BattleZone game.

```bash
python agents/battlezone_agent.py --mode train --timesteps 100000
python agents/battlezone_agent.py --mode play
```

### 2. Humanoid Standup Agent

- **Environment:** `HumanoidStandup-v5` (Mujoco)
- **Algorithm:** PPO
- **Description:** Trains a humanoid robot to stand up from a prone position.

```bash
python agents/humanoid_agent.py --mode train --timesteps 1000000
python agents/humanoid_agent.py --mode play
```

### 3. Autonomous Car Agent

- **Environment:** `highway-fast-v0` (Highway-Env)
- **Algorithm:** DQN
- **Description:** Trains an autonomous car to drive safely on a highway.

```bash
python agents/autonomous_car.py --mode train --timesteps 20000
python agents/autonomous_car.py --mode play
```

### 4. Our Idea Agent - Lunar Lander

- **Environment:** `LunarLander-v3` (Box2D)
- **Algorithm:** PPO
- **Description:** Trains an agent to safely land a spacecraft on the moon.

```bash
python agents/lunar_lander_agent.py --mode train --timesteps 100000
python agents/lunar_lander_agent.py --mode play
```

## Installation

```bash
pip install -r requirements.txt
pip install "gymnasium[mujoco]" "gymnasium[box2d]" highway-env
```

## Requirements

- Python 3.10+
- Gymnasium
- Stable-Baselines3
- PyTorch
- Mujoco (for Humanoid)
- Box2D (for LunarLander)
- Highway-Env (for Autonomous Car)
