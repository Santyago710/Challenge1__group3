import gymnasium as gym
import ale_py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

env_name = "ALE/PrivateEye-v5"

def make_env():
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env


models_folder = "models"
results = []

for model_file in os.listdir(models_folder):

    if model_file.endswith(".zip"):

        model_path = os.path.join(models_folder, model_file)

        print("Evaluating:", model_file)

        env = make_env()

        model = DQN.load(model_path)

        total_reward = 0
        episodes = 10

        for ep in range(episodes):

            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)

                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward

        avg_reward = total_reward / episodes

        results.append((model_file, avg_reward))

        print("Average reward:", avg_reward)


print("\nRESULTS")
for r in results:
    print(r)