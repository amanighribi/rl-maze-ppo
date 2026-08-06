import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt

# Grid legend
EMPTY, WALL, AGENT, GOAL = 0, 1, 2, 3

START_POS = (0, 0)
GOAL_POS = (4, 4)


def is_solvable(maze, start, goal):
    """BFS check: is there a path from start to goal avoiding walls?"""
    h, w = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    queue = deque([start])
    visited[start] = True

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and maze[nr, nc] != WALL:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return False


def generate_maze(height, width, start, goal, obstacle_density=0.25, np_random=None, max_attempts=100):
    """Randomly scatter obstacles, retry until the maze is solvable start->goal."""
    rng = np_random if np_random is not None else np.random.default_rng()

    for _ in range(max_attempts):
        maze = (rng.random((height, width)) < obstacle_density).astype(int)
        maze[start] = EMPTY
        maze[goal] = EMPTY
        if is_solvable(maze, start, goal):
            return maze

    # Fallback: if we somehow failed every attempt, return an empty (fully open) maze
    return np.zeros((height, width), dtype=int)


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, height=5, width=5, start=START_POS, goal=GOAL_POS,
                 obstacle_density=0.25, render_mode=None, max_steps=50):
        super().__init__()
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.obstacle_density = obstacle_density
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._step_count = 0

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # State: agent's (row, col) position, normalized, whether each
        # neighboring cell (up, down, left, right) is open (1) or blocked (0),
        # and how many times the agent has already visited its current cell
        # this episode (capped/normalized). A binary "visited" flag isn't
        # enough — it looks the same on the 2nd and the 25th visit, so a
        # deterministic policy can still get stuck in a fixed loop. A rising
        # count keeps the state changing the longer the agent is stuck,
        # giving the policy something to actually react to.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.float32
        )

        self.agent_pos = None
        self._fig = None

    def _get_obs(self):
        r, c = self.agent_pos
        pos = [r / (self.height - 1), c / (self.width - 1)]

        def is_open(rr, cc):
            return 1.0 if 0 <= rr < self.height and 0 <= cc < self.width and self.maze[rr, cc] != WALL else 0.0

        neighbors = [
            is_open(r - 1, c),  # up
            is_open(r + 1, c),  # down
            is_open(r, c - 1),  # left
            is_open(r, c + 1),  # right
        ]
        # Normalized visit count for the current cell, capped at 5+ visits.
        # Keeps rising instead of flatlining, so a stuck agent's observation
        # keeps changing rather than repeating identically forever.
        visit_count = min(self._visited.get((r, c), 0), 5) / 5.0
        return np.array(pos + neighbors + [visit_count], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.maze = generate_maze(
            self.height, self.width, self.start, self.goal,
            obstacle_density=self.obstacle_density, np_random=self.np_random,
        )
        self.agent_pos = self.start
        self._step_count = 0
        self._visited = {self.start: 1}
        return self._get_obs(), {}

    def step(self, action):
        r, c = self.agent_pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
        dr, dc = moves[action]
        new_r, new_c = r + dr, c + dc

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps

        # Check bounds and walls
        if 0 <= new_r < self.height and 0 <= new_c < self.width and self.maze[new_r, new_c] != WALL:
            self.agent_pos = (new_r, new_c)
            visit_count = self._visited.get(self.agent_pos, 0)
            if visit_count > 0:
                # Escalating penalty: the more times this cell has already
                # been visited, the worse it gets — makes prolonged looping
                # increasingly costly instead of a flat, ignorable penalty.
                reward = -0.1 * (visit_count + 1)
            else:
                reward = -0.01  # small step penalty, encourages efficiency
            self._visited[self.agent_pos] = visit_count + 1
        else:
            reward = -0.2  # bumped into wall/edge — clearly worse than a normal step

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