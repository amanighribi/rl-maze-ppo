# RL Maze Navigation with PPO

# RL Maze Navigation with PPO

An agent learns to navigate from a start point to a goal in a procedurally
generated grid maze while avoiding obstacles, trained with PPO (Proximal
Policy Optimization) using Stable-Baselines3 and a custom Gymnasium
environment. A new random maze is generated every episode (guaranteed
solvable via BFS check), so the agent has to learn general navigation
behavior rather than memorizing a single fixed layout.

**Current result: 99% success rate across 100 freshly generated, unseen mazes.**

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

Evaluate the trained agent on one episode and generate a GIF of it solving the maze:
```bash
python evaluate.py
```

Evaluate success rate across 100 fresh random mazes:
```bash
python evaluate_batch.py
```

Inspect failed episodes in detail (maze layout + path taken):
```bash
python evaluate_failures.py
```

Monitor training live with TensorBoard:
```bash
tensorboard --logdir logs
```

## App (React + FastAPI)

Backend (serves the trained model):
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend (in a separate terminal):
```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (usually `http://localhost:5173`). Make sure the backend is running first — the frontend calls it directly.

## Project structure

- `maze_env/maze_env.py` — custom Gymnasium environment (procedural maze generation, state, action, reward, render)
- `train.py` — PPO training loop
- `evaluate.py` — runs the trained policy on one episode and exports a GIF
- `evaluate_batch.py` — measures success rate across many episodes
- `evaluate_failures.py` — isolates and prints failed episodes for debugging
- `backend/main.py` — FastAPI server exposing `/new-maze` and `/solve`
- `frontend/` — React app: generates mazes and animates the agent solving them

## Environment design

- **State (7-dim):** normalized agent position (row, col), whether each of the 4 neighboring cells is open, and a normalized count of how many times the current cell has already been visited this episode
- **Action:** discrete, up/down/left/right
- **Reward:** +1.0 on reaching the goal, -0.01 per normal step, -0.2 for bumping a wall, escalating penalty for revisiting the same cell (discourages loops)
- **Episode:** ends on reaching the goal or after a 50-step limit

## Debugging notes

Two real bugs were found and fixed during development, both via a custom
batch-evaluation + failure-inspection workflow rather than just staring at
training reward curves:

1. **Weak wall penalty.** Early on, bumping a wall (-0.05) was barely worse
   than a normal step (-0.01), so the deterministic policy would get stuck
   repeating the same wall-bump action indefinitely at certain states.
   Fixed by increasing the penalty to -0.2.
2. **State-aliasing loop trap.** After adding procedural mazes, ~15% of
   episodes failed — all due to the agent oscillating between two adjacent
   cells 20+ times. Root cause: the observation had no memory, so revisiting
   a cell looked identical every time, and a deterministic policy kept
   repeating the same action forever. A binary "already visited" flag
   didn't fix it (it saturates after the 2nd visit). The real fix was an
   escalating, uncapped-feeling visit count in both the observation and the
   reward, so the state keeps changing — and getting worse — the longer the
   agent stays stuck. This took the success rate from 85% to 99%.

## Status

- [x] Fixed-maze baseline (validated the pipeline end-to-end)
- [x] Procedural maze generation with BFS solvability check
- [x] Diagnosed and fixed wall-penalty and loop-trap bugs
- [x] 99% success rate across 100 unseen random mazes
- [ ] Scale to larger grids / higher obstacle density
- [ ] Randomize start/goal positions (harder generalization test)
- [ ] Integrate trained model into the existing app (per internship scope)

