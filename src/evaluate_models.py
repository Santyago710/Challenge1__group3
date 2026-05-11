import gymnasium as gym
import ale_py
import os
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

env_name = "ALE/PrivateEye-v5"


def make_env():
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env


models_dir = "models"
episodes = 5
max_steps = 5000

results = []

model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".zip")])

print("\nEvaluating models...\n")

for model_file in model_files:

    model_path = os.path.join(models_dir, model_file)

    print("Evaluating:", model_file)

    env = make_env()

    model = DQN.load(model_path)

    episode_rewards = []

    for ep in range(episodes):

        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < max_steps:

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            done = terminated or truncated

            step += 1

        episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)

    print("Rewards:", episode_rewards)
    print("Average reward:", avg_reward)
    print()

    results.append((model_file, avg_reward))

    env.close()


print("\n==============================")
print("FINAL RESULTS")
print("==============================")

results.sort(key=lambda x: x[1], reverse=True)

for model, reward in results:
    print(f"{model} -> Avg Reward: {reward}")


best_model = results[0][0]

print("\nBest model:", best_model)
print("Average reward:", results[0][1])