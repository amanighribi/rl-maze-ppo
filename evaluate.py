import imageio
from stable_baselines3 import PPO

from maze_env.maze_env import MazeEnv

env = MazeEnv(render_mode="rgb_array")
model = PPO.load("ppo_maze")

obs, _ = env.reset()
frames = [env.render()]
positions = [env.agent_pos]

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(int(action))
    frames.append(env.render())
    positions.append(env.agent_pos)
    print(f"step {step}: action={int(action)} -> pos={env.agent_pos} reward={reward:.3f}")
    if terminated or truncated:
        break

imageio.mimsave("maze_solution.gif", frames, fps=2, loop=0)
print(f"Saved maze_solution.gif — agent {'reached the goal' if terminated else 'did not finish'}")
print(f"Full path: {positions}")