from collections import Counter
from stable_baselines3 import PPO
from maze_env.maze_env import MazeEnv, WALL

N_EPISODES = 100

env = MazeEnv()
model = PPO.load("ppo_maze")


def print_maze(maze, path, start, goal):
    """ASCII render: # wall, . open, S start, G goal, * visited path (last wins)."""
    height, width = maze.shape
    grid = [["#" if maze[r, c] == WALL else "." for c in range(width)] for r in range(height)]

    for (r, c) in path:
        if (r, c) not in (start, goal):
            grid[r][c] = "*"
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"

    for row in grid:
        print(" ".join(row))


failure_count = 0

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    path = [env.agent_pos]

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        path.append(env.agent_pos)

    if terminated:
        continue  # only inspect failures

    failure_count += 1
    print(f"\n{'=' * 40}")
    print(f"FAILURE #{failure_count} (episode {ep})")
    print(f"{'=' * 40}")
    print_maze(env.maze, path, env.start, env.goal)

    # Detect looping: how many times was each cell revisited?
    visit_counts = Counter(path)
    most_common_pos, most_common_count = visit_counts.most_common(1)[0]
    unique_cells = len(visit_counts)

    print(f"\nPath length: {len(path)} steps, visited {unique_cells} unique cells")
    if most_common_count >= 5:
        print(f"-> Looks like LOOPING: cell {most_common_pos} was visited {most_common_count} times")
    elif unique_cells >= len(path) * 0.7:
        print(f"-> Looks like a genuinely LONG/WANDERING path (mostly new cells, ran out of steps)")
    else:
        print(f"-> Mixed pattern: some revisiting, but not a tight loop")

print(f"\n\nTotal failures inspected: {failure_count}/{N_EPISODES}")
