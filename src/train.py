import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml
import os

gym.register_envs(ale_py)


# Cargar hiperparámetros
with open("configs/hyperparameters.yaml", "r") as f:
    config = yaml.safe_load(f)

env_name = config["env"]
training = config["training"]
exploration = config["exploration"]


def make_env():
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env


env = DummyVecEnv([make_env])


model = DQN(
    "CnnPolicy",
    env,
    learning_rate=training["learning_rate"],
    buffer_size=training["buffer_size"],
    learning_starts=training["learning_starts"],
    batch_size=training["batch_size"],
    gamma=training["gamma"],
    train_freq=training["train_freq"],
    target_update_interval=training["target_update_interval"],
    exploration_fraction=exploration["exploration_fraction"],
    exploration_final_eps=exploration["exploration_final_eps"],
    verbose=1,
    tensorboard_log="./logs/"
)


model.learn(total_timesteps=training["total_timesteps"])


os.makedirs("models", exist_ok=True)
model.save("models/dqn_privateeye")