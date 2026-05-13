import os
import csv
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# ─────────────────────────────────────────────────────────────
# Evalúa todos los modelos DQN del Challenge 1
# Guarda resultados en logs/comparison/dqn_results.csv
#
# Uso: python src/evaluate_dqn.py
# ─────────────────────────────────────────────────────────────

gym.register_envs(ale_py)

env_name   = "ALE/PrivateEye-v5"
models_dir = "../../models"
output_csv = "logs/comparison/dqn_results.csv"
episodes   = 10
max_steps  = 5000

os.makedirs("logs/comparison", exist_ok=True)

def make_env():
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env

# Map model filename → experiment key
model_files = {
    "exp1": "dqn_privateeye_exp1.zip",
    "exp2": "dqn_privateeye_exp2.zip",
    "exp3": "dqn_privateeye_exp3.zip",
    "exp4": "dqn_privateeye_exp4.zip",
    "exp5": "dqn_privateeye_exp5.zip",
    "exp6": "dqn_privateeye_exp6.zip",
}

print("\n" + "="*55)
print("  Challenge 1 – DQN Model Evaluator")
print("  Group 3 | ALE/PrivateEye-v5")
print(f"  Episodes per model : {episodes} (deterministic)")
print("="*55)

csv_rows = []

for exp, fname in model_files.items():
    model_path = os.path.join(models_dir, fname)

    if not os.path.exists(model_path):
        print(f"\n  [SKIP] {fname} not found")
        csv_rows.append([exp, fname, "nan", "nan", "nan", "nan", "nan"])
        continue

    print(f"\n── {exp} — {fname}")

    env   = make_env()
    model = DQN.load(model_path)

    episode_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done         = False
        total_reward = 0
        step         = 0

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done          = terminated or truncated
            step         += 1

        episode_rewards.append(total_reward)
        print(f"  Episode {ep+1:2d}: {total_reward:.1f}")

    env.close()

    mean_r = np.mean(episode_rewards)
    std_r  = np.std(episode_rewards)
    min_r  = np.min(episode_rewards)
    max_r  = np.max(episode_rewards)

    print(f"  → Mean: {mean_r:.2f} ± {std_r:.2f}  (min={min_r:.1f}, max={max_r:.1f})")

    csv_rows.append([exp, fname,
                     f"{mean_r:.4f}", f"{std_r:.4f}",
                     f"{min_r:.4f}", f"{max_r:.4f}",
                     ",".join(str(r) for r in episode_rewards)])

# Save CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["exp", "model", "mean", "std", "min", "max", "all_rewards"])
    writer.writerows(csv_rows)

print(f"\n✅ Results saved → {output_csv}")

# Final ranking
print("\n" + "="*55)
print("  FINAL RANKING — DQN MODELS")
print("="*55)
sorted_rows = sorted(csv_rows, key=lambda x: float(x[2]) if x[2] != "nan" else -999, reverse=True)
for r in sorted_rows:
    print(f"  {r[0]:<6} {r[1]:<35} mean={float(r[2]):.2f} ± {float(r[3]):.2f}")
print("="*55)