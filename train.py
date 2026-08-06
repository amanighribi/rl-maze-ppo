from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from maze_env.maze_env import MazeEnv

env = MazeEnv()

# Sanity check: confirms your env respects the Gymnasium API before training
check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
)

model.learn(total_timesteps=200_000)
model.save("ppo_maze")

print("Training done. Model saved to ppo_maze.zip")