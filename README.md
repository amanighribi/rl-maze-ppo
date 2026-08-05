# RL Maze Navigation with PPO

An agent learns to navigate from a start point to a goal in a grid maze while
avoiding obstacles, trained with PPO (Proximal Policy Optimization) using
Stable-Baselines3 and a custom Gymnasium environment.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

Train the agent:
```bash
python train.py
```

Evaluate the trained agent and generate a GIF of it solving the maze:
```bash
python evaluate.py
```

Monitor training live with TensorBoard:
```bash
tensorboard --logdir logs
```

## Project structure

- `maze_env/maze_env.py` — custom Gymnasium environment (state, action, reward, render)
- `train.py` — PPO training loop
- `evaluate.py` — runs the trained policy and exports a GIF

## Status

- [x] Fixed maze environment matching the grid sketch
- [ ] Confirm maze generalization scope with instructor
- [ ] Tune reward shaping
- [ ] Curriculum / difficulty scaling (if generalization is in scope)
