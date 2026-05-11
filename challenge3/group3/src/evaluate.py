import gymnasium as gym
import ale_py
import sys
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Registrar entornos Atari
gym.register_envs(ale_py)

env_name = "ALE/PrivateEye-v5"


def make_env():
    env = gym.make(env_name, render_mode="human")
    env = AtariWrapper(env)
    return env


# Verificar argumento del modelo
if len(sys.argv) < 2:
    print("Uso: python src/evaluate.py <ruta_del_modelo>")
    print("Ejemplo: python src/evaluate.py models/ppo_privateeye_exp1_seed42")
    sys.exit()

model_path = sys.argv[1]

print("Loading model:", model_path)

env = make_env()

model = PPO.load(model_path)

# Challenge 3 protocol: 10 deterministic evaluation episodes
episodes  = 10
max_steps = 5000

episode_rewards = []

for ep in range(episodes):

    obs, _ = env.reset()

    done         = False
    total_reward = 0
    step         = 0

    print(f"\nStarting episode {ep+1}")

    while not done and step < max_steps:

        # deterministic=True → greedy policy (no exploration)
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        done          = terminated or truncated
        step         += 1

        # Mostrar progreso cada 500 pasos
        if step % 500 == 0:
            print("Step:", step, "Reward:", total_reward)

        # Renderizar solo cada 20 pasos para acelerar
        if step % 20 == 0:
            env.render()

        # pequeño delay para estabilidad visual
        time.sleep(0.003)

    print(f"Episode {ep+1} finished")
    print("Total reward:", total_reward)

    episode_rewards.append(total_reward)

env.close()

# ─────────────────────────────────────────────────────────────
# Results — mean ± std as required by Challenge 3 protocol
# ─────────────────────────────────────────────────────────────
print("\n" + "="*45)
print("  EVALUATION RESULTS (PPO)")
print("="*45)
print(f"  Model    : {model_path}")
print(f"  Episodes : {episodes}")
print(f"  Rewards  : {episode_rewards}")
print(f"  Mean     : {np.mean(episode_rewards):.2f}")
print(f"  Std      : {np.std(episode_rewards):.2f}")
print(f"  Min      : {np.min(episode_rewards):.2f}")
print(f"  Max      : {np.max(episode_rewards):.2f}")
print("="*45)