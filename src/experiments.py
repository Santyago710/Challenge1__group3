import gymnasium as gym
import ale_py
import itertools
import os

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register_envs(ale_py)

env_name = "ALE/PrivateEye-v5"


def make_env():
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env


# Espacio de búsqueda de hiperparámetros
learning_rates = [0.00005, 0.0001, 0.0002]
batch_sizes = [32, 64]
buffer_sizes = [50000, 100000]


experiments = list(itertools.product(learning_rates, batch_sizes, buffer_sizes))

print("Total experiments:", len(experiments))


for i, (lr, batch, buffer) in enumerate(experiments):

    print(f"\nRunning experiment {i+1}")

    env = DummyVecEnv([make_env])

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=lr,
        batch_size=batch,
        buffer_size=buffer,
        learning_starts=20000,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2000,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log=f"./logs/exp_{i+1}/"
    )

    model.learn(total_timesteps=200000)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/dqn_privateeye_exp{i+1}")