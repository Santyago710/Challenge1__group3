# Reinforcement Learning for Atari PrivateEye using DQN

## Overview

This project implements a **Deep Q-Network (DQN)** agent to learn how to play the Atari game **PrivateEye** using the **Gymnasium** and **Stable-Baselines3** frameworks.

The goal is to explore how different hyperparameters affect the performance of the reinforcement learning agent. Multiple experiments were conducted and evaluated to identify the best performing model.

This project includes:

* Training a DQN agent on the Atari **PrivateEye** environment
* Running multiple experiments with different hyperparameters
* Saving trained models
* Evaluating and comparing model performance
* Visualizing learning curves
* Running the trained agent in the environment

---

# Project Structure

```
challenge1__3/
│
├── models/                 # Saved trained models
│
├── logs/                   # TensorBoard training logs
│
├── src/
│   ├── train.py            # Script to train a single model
│   ├── experiments.py      # Runs multiple training experiments
│   ├── evaluate.py         # Runs a trained agent in the environment
│   ├── evaluate_models.py  # Evaluates all trained models
│   
├── hyperparameters.yaml    # Training configuration
│
│
│── Paper/ # Folder of the documnent in IEEE format that contains all the development of the challenge 1 and 3
├
├── requirements.txt
│
└── README.md
```

---

# Environment

The project uses the Atari environment:

```
ALE/PrivateEye-v5
```

This environment is part of the **Arcade Learning Environment (ALE)**.

---

# Installation

## 1. Clone the repository

```bash
git clone <repository_url>
cd challenge1__3
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate the environment:

Linux / Mac:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install gymnasium[atari] stable-baselines3[extra] ale-py torch matplotlib tensorboard
```

---

# Training a Model

To train a single model:

```bash
python src/train.py
```

The training configuration is defined in:

```
hyperparameters.yaml
```

Key parameters include:

* total_timesteps
* learning_rate
* batch_size
* buffer_size
* gamma
* exploration parameters

---

# Running Multiple Experiments

To evaluate different hyperparameter configurations:

```bash
python src/experiments.py
```

This script trains multiple models with different:

* learning rates
* batch sizes
* replay buffer sizes

Each trained model is saved in:

```
models/
```

Example:

```
models/dqn_privateeye_exp1.zip
models/dqn_privateeye_exp2.zip
models/dqn_privateeye_exp3.zip
```

---

# Evaluating All Models

To evaluate the performance of all trained models:

```bash
python src/evaluate_models.py
```

The script:

* loads each trained model
* runs several episodes
* computes the **average reward**
* prints a comparison table

Example output:

```
FINAL RESULTS

dqn_privateeye_exp2.zip -> Avg Reward: 1.0
dqn_privateeye_exp3.zip -> Avg Reward: 1.0
dqn_privateeye_exp4.zip -> Avg Reward: 1.0
dqn_privateeye_exp5.zip -> Avg Reward: 1.0
dqn_privateeye_exp6.zip -> Avg Reward: 0.0
dqn_privateeye_exp1.zip -> Avg Reward: -1.0
```

The best performing model is automatically identified.

---

# Watching the Agent Play

To visualize a trained agent interacting with the environment:

```bash
python src/evaluate.py models/dqn_privateeye_exp2.zip
```

A window will open showing the agent playing **PrivateEye**.

---

# Visualizing Training Performance

Training metrics are saved using **TensorBoard**.

Run:

```bash
tensorboard --logdir logs
```

Then open in a browser:

```
http://localhost:6006
```

You can view:

* episode reward
* training loss
* exploration progress

---

# Hardware Used

Training was performed on:

* **CPU**
* **8GB RAM**
* **NVIDIA GTX 1650 GPU**

Note: Atari DQN training is typically **CPU-bound**, since most computation time is spent simulating the environment.

---

# Results Summary

The rewards obtained during evaluation were relatively small due to the complexity of the **PrivateEye** environment and the limited number of training timesteps.

PrivateEye is known to be one of the most challenging Atari environments due to:

* sparse rewards
* long exploration sequences
* delayed feedback

Despite this, some models achieved positive rewards, indicating the agent started learning meaningful behaviors.

---

# Technologies Used

* Python
* PyTorch
* Stable-Baselines3
* Gymnasium
* Atari Learning Environment (ALE)
* TensorBoard
* Matplotlib

---

# Authors
Group #3
David Buitrago 20221020085 //
Cristian Cruz 202210200125 //
Daniel Cuellar 20221020081


Reinforcement Learning Project
