import gymnasium as gym
import ale_py
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

env_name = "ALE/PrivateEye-v5"


def make_env():
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env


models_dir = "models"

# Challenge 3 protocol: 10 deterministic evaluation episodes
episodes  = 10
max_steps = 5000

results = []

model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".zip")])

if not model_files:
    print(f"[ERROR] No .zip models found in '{models_dir}/'")
    print("        Run experiments.py first to train the models.")
    exit(1)

print("\n" + "="*55)
print("  Challenge 3 – PPO Model Evaluator")
print("  Group 3 | ALE/PrivateEye-v5")
print(f"  Models found : {len(model_files)}")
print(f"  Episodes each: {episodes} (deterministic / greedy)")
print("="*55)

for model_file in model_files:

    model_path = os.path.join(models_dir, model_file)

    print(f"\nEvaluating: {model_file}")
    print("─"*45)

    env = make_env()

    model = PPO.load(model_path)

    episode_rewards = []

    for ep in range(episodes):

        obs, _ = env.reset()
        done         = False
        total_reward = 0
        step         = 0

        while not done and step < max_steps:

            # deterministic=True → greedy policy (no exploration)
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            done          = terminated or truncated
            step         += 1

        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward  = np.std(episode_rewards)

    print(f"  Rewards : {episode_rewards}")
    print(f"  Mean    : {mean_reward:.2f} ± {std_reward:.2f}")

    results.append({
        "model":  model_file,
        "mean":   mean_reward,
        "std":    std_reward,
        "min":    np.min(episode_rewards),
        "max":    np.max(episode_rewards),
        "rewards": episode_rewards,
    })

    env.close()

# ─────────────────────────────────────────────────────────────
# Final ranking — sorted by mean reward (descending)
# ─────────────────────────────────────────────────────────────
results.sort(key=lambda x: x["mean"], reverse=True)

print("\n\n" + "="*55)
print("  FINAL RANKING — ALL PPO MODELS")
print("="*55)
print(f"  {'RANK':<6} {'MODEL':<40} {'MEAN ± STD'}")
print(f"  {'─'*6} {'─'*40} {'─'*15}")

for rank, r in enumerate(results, start=1):
    print(f"  {rank:<6} {r['model']:<40} {r['mean']:.2f} ± {r['std']:.2f}")

best = results[0]
print(f"\n  🏆 Best model : {best['model']}")
print(f"     Mean reward: {best['mean']:.2f} ± {best['std']:.2f}")
print("="*55)