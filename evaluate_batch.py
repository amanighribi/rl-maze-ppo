from stable_baselines3 import PPO
from maze_env.maze_env import MazeEnv

N_EPISODES = 100

env = MazeEnv()
model = PPO.load("ppo_maze")

successes = 0
steps_on_success = []
steps_on_failure = []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        steps += 1

    if terminated:
        successes += 1
        steps_on_success.append(steps)
    else:
        steps_on_failure.append(steps)

success_rate = successes / N_EPISODES * 100
avg_steps_success = sum(steps_on_success) / len(steps_on_success) if steps_on_success else float("nan")

print(f"Ran {N_EPISODES} episodes on freshly generated random mazes")
print(f"Success rate: {success_rate:.1f}% ({successes}/{N_EPISODES})")
if steps_on_success:
    print(f"Average steps to goal (successful episodes): {avg_steps_success:.1f}")
if steps_on_failure:
    print(f"Failed episodes: {len(steps_on_failure)} (hit the {env.max_steps}-step limit without reaching the goal)")
