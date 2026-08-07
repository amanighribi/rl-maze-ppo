import sys
from pathlib import Path

# Allow importing maze_env from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO

from maze_env.maze_env import MazeEnv, generate_maze, WALL

app = FastAPI(title="RL Maze API")

# Allow the React dev server to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev server port
    allow_methods=["*"],
    allow_headers=["*"],
)

model = PPO.load(str(Path(__file__).resolve().parent.parent / "ppo_maze.zip"))

HEIGHT, WIDTH = 5, 5
START = (0, 0)
GOAL = (4, 4)


class SolveRequest(BaseModel):
    maze: list[list[int]]
    start: list[int]
    goal: list[int]


@app.get("/new-maze")
def new_maze():
    maze = generate_maze(HEIGHT, WIDTH, START, GOAL, obstacle_density=0.25)
    return {
        "maze": maze.tolist(),
        "start": list(START),
        "goal": list(GOAL),
    }


@app.post("/solve")
def solve(req: SolveRequest):
    import numpy as np

    env = MazeEnv(height=len(req.maze), width=len(req.maze[0]),
                   start=tuple(req.start), goal=tuple(req.goal))
    env.maze = np.array(req.maze)
    env.agent_pos = tuple(req.start)
    env._step_count = 0
    env._visited = {tuple(req.start): 1}
    obs = env._get_obs()

    path = [list(env.agent_pos)]
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        path.append(list(env.agent_pos))

    return {
        "path": path,
        "success": terminated,
        "steps": len(path) - 1,
    }
