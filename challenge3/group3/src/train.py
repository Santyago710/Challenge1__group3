import os
import yaml
import argparse
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register_envs(ale_py)

# ─────────────────────────────────────────────────────────────
# Argument parsing
# Allows selecting which experiment and seed to run, e.g.:
#   python train.py --exp exp1 --seed 42
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train PPO on ALE/PrivateEye-v5")
parser.add_argument("--exp",  type=str, default="exp1",
                    help="Experiment key from hyperparameters.yaml")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Load hyperparameters
# ─────────────────────────────────────────────────────────────
with open("configs/hyperparameters.yaml", "r") as f:
    config = yaml.safe_load(f)

env_name = config["env"]
params   = config["experiments"][args.exp]

print(f"\n{'='*55}")
print(f"  Challenge 3 – PPO | Group 3 | ALE/PrivateEye-v5")
print(f"  Experiment : {args.exp}")
print(f"  Seed       : {args.seed}")
print(f"  Timesteps  : {params['total_timesteps']:,}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────────────────────────
# Environment factory
# Uses SB3's AtariWrapper — identical preprocessing to Ch1:
#   grayscale, resize 84x84, frame-skip 4, frame-stack 4
# ─────────────────────────────────────────────────────────────
def make_env():
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env

env = DummyVecEnv([make_env])

# ─────────────────────────────────────────────────────────────
# PPO model
# CnnPolicy uses the same convolutional architecture as DQN
# in Challenge 1, ensuring a fair algorithmic comparison.
# ─────────────────────────────────────────────────────────────
log_dir = f"./logs/{args.exp}/seed_{args.seed}/"
os.makedirs(log_dir, exist_ok=True)

model = PPO(
    "CnnPolicy",
    env,
    learning_rate=params["learning_rate"],
    n_steps=params["n_steps"],
    n_epochs=params["n_epochs"],
    batch_size=params["batch_size"],
    gamma=params["gamma"],
    gae_lambda=params["gae_lambda"],
    clip_range=params["clip_range"],
    ent_coef=params["ent_coef"],
    vf_coef=params["vf_coef"],
    max_grad_norm=params["max_grad_norm"],
    seed=args.seed,
    verbose=1,
    tensorboard_log=log_dir,
)

# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
model.learn(total_timesteps=params["total_timesteps"])

# ─────────────────────────────────────────────────────────────
# Save model
# Naming convention mirrors Challenge 1:
#   ppo_privateeye_{exp}_{seed}.zip
# ─────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
save_path = f"models/ppo_privateeye_{args.exp}_seed{args.seed}"
model.save(save_path)
print(f"\nModel saved → {save_path}.zip")