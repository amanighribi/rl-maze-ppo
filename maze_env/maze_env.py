import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# Grid legend
EMPTY, WALL, AGENT, GOAL = 0, 1, 2, 3

# Fixed maze for now (matches the grid your instructor sketched).
# 0 = open path, 1 = wall/obstacle.
# Replace/generate this later if you and your instructor agree on random mazes.
DEFAULT_MAZE = np.array([
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

START_POS = (0, 0)
GOAL_POS = (4, 4)


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, maze=DEFAULT_MAZE, start=START_POS, goal=GOAL_POS, render_mode=None):
        super().__init__()
        self.maze = maze
        self.start = start
        self.goal = goal
        self.render_mode = render_mode

        self.height, self.width = self.maze.shape

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # State: agent's (row, col) position, normalized.
        # Start simple; you can expand this later (e.g. local obstacle sensing).
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )

        self.agent_pos = None
        self._fig = None

    def _get_obs(self):
        r, c = self.agent_pos
        return np.array([r / (self.height - 1), c / (self.width - 1)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start
        return self._get_obs(), {}

    def step(self, action):
        r, c = self.agent_pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
        dr, dc = moves[action]
        new_r, new_c = r + dr, c + dc

        terminated = False
        truncated = False

        # Check bounds and walls
        if 0 <= new_r < self.height and 0 <= new_c < self.width and self.maze[new_r, new_c] != WALL:
            self.agent_pos = (new_r, new_c)
            reward = -0.01  # small step penalty, encourages efficiency
        else:
            reward = -0.05  # bumped into wall/edge, slightly worse than a normal step

        if self.agent_pos == self.goal:
            reward = 1.0
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        grid = self.maze.copy().astype(int)
        grid[self.agent_pos] = AGENT
        grid[self.goal] = GOAL

        if self.render_mode == "rgb_array" or self.render_mode == "human":
            colors = {EMPTY: "white", WALL: "black", AGENT: "blue", GOAL: "green"}
            img = np.zeros((*grid.shape, 3))
            color_map = {EMPTY: (1, 1, 1), WALL: (0, 0, 0), AGENT: (0.2, 0.4, 1), GOAL: (0.2, 0.8, 0.2)}
            for val, rgb in color_map.items():
                img[grid == val] = rgb

            if self.render_mode == "human":
                plt.imshow(img)
                plt.axis("off")
                plt.pause(0.1)
                plt.clf()
                return None
            else:
                return (img * 255).astype(np.uint8)

    def close(self):
        plt.close("all")
