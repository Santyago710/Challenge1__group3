from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

print("=== DQN TAGS ===")
dqn_dir = "../../logs/exp_1/DQN_1"
ea = EventAccumulator(dqn_dir)
ea.Reload()
print(f"Path: {dqn_dir}")
print(f"Tags: {ea.Tags()}")

print("\n=== PPO TAGS ===")
ppo_dir = "./logs/exp1/seed_42/PPO_1"
ea2 = EventAccumulator(ppo_dir)
ea2.Reload()
print(f"Path: {ppo_dir}")
print(f"Tags: {ea2.Tags()}")
