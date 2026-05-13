# Challenge 3 — PPO vs DQN | Group 3 | ALE/PrivateEye-v5

**Course:** Machine Learning — Universidad Distrital Francisco José de Caldas  
**Professor:** Carlos Andrés Sierra, M.Sc.  
**Algorithm:** Proximal Policy Optimization (PPO) via Stable Baselines 3  
**Environment:** `ALE/PrivateEye-v5`  

---

## Project Structure

```
challenge3/group3/
├── configs/
│   └── hyperparameters.yaml     # 6 PPO experiment configs × 3 seeds
├── logs/
│   ├── exp1/ … exp6/            # TensorBoard logs per experiment/seed
│   └── comparison/              # DQN vs PPO plots and metrics CSV
├── models/
│   └── ppo_privateeye_*.zip     # Trained PPO models
├── src/
│   ├── train.py                 # Train a single PPO run
│   ├── experiments.py           # Run all experiments automatically
│   ├── evaluate.py              # Evaluate one model (with rendering)
│   ├── evaluate_models.py       # Evaluate and rank all PPO models
│   ├── evaluate_dqn.py          # Evaluate all DQN models from Ch1
│   └── compare_results.py       # Generate DQN vs PPO comparison plots
├── CHECKLIST.md
└── README.md
```

---

## Setup

```bash
# From the repo root (challenge1__3/)
source venv/bin/activate
pip install -r challenge3/group3/requirements.txt
```

---

## Reproduce Training

### Run all 18 experiments (6 configs × 3 seeds):
```bash
cd challenge3/group3
python src/experiments.py
```

### Run a single experiment:
```bash
python src/experiments.py --exp exp6 --seed 42
```

### Skip already-completed experiments:
```bash
python src/experiments.py --skip exp1 exp2
```

---

## Evaluate Models

### Watch a single PPO model in action:
```bash
python src/evaluate.py models/ppo_privateeye_exp6_seed42
```

### Rank all trained PPO models:
```bash
python src/evaluate_models.py
```

### Evaluate all DQN models from Challenge 1:
```bash
python src/evaluate_dqn.py --models_dir ../../models
```

---

## Generate DQN vs PPO Comparison

```bash
# 1. Evaluate DQN models first (generates dqn_results.csv)
python src/evaluate_dqn.py --models_dir ../../models

# 2. Generate all comparison plots and metrics table
python src/compare_results.py
```

Plots are saved in `logs/comparison/`.

---

## Preprocessing (identical to Challenge 1)

| Parameter | Value |
|-----------|-------|
| Grayscale | ✅ |
| Resize | 84 × 84 |
| Frame skip | 4 |
| Frame stack | 4 |
| Pixel scaling | [0, 1] |
| Wrapper | `SB3 AtariWrapper + Monitor` |

---

## Hyperparameter Search Space

| Experiment | Key change | Hypothesis |
|------------|-----------|------------|
| exp1 | Baseline (PDF starter values) | Reference point |
| exp2 | `ent_coef` 0.02 → 0.05 | More exploration |
| exp3 | `n_steps` 2048 → 4096 | Longer credit assignment |
| exp4 | Lower `lr` + more `n_epochs` | Conservative updates |
| exp5 | Wider `clip_range` + larger batch | Faster adaptation |
| exp6 | High entropy + long horizon + low lr | Best guess for sparse reward |

**Best PPO config:** exp6 (`ent_coef=0.05`, `n_steps=4096`, `lr=1e-4`)  
**Best DQN config:** exp2 (from Challenge 1)

---

## Key Results

| Metric | DQN (best) | PPO (best) |
|--------|-----------|-----------|
| Final mean reward | 1.00 ± 0.00 | ~0.90 ± 0.70 |
| Learning trend | Unknown (no curve logged) | Clear upward trend from −6 to ~1 |
| Steps to converge | N/A | ~150,000 |
| Target score (500) reached | ❌ | ❌ |

Both algorithms fail to obtain significant reward on PrivateEye-v5 within 200k steps,
consistent with published results. PPO shows a measurable learning signal while DQN
reaches a marginally higher final score with its off-policy replay buffer.

---

## References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
- Schulman et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.*
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.*
- Raffin et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.*
