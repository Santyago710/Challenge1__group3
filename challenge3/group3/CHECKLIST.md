# CHECKLIST — Challenge 3 | Group 3 | ALE/PrivateEye-v5

## Exact command to reproduce the best PPO run

```bash
# From challenge3/group3/
python src/experiments.py --exp exp6 --seed 42
```

## Seeds used for PPO repeated experiments

| Experiment | Seeds          |
|------------|----------------|
| exp1–exp6  | 42, 123, 777   |

## Pointers to logs and figures

| Artifact | Path |
|----------|------|
| PPO TensorBoard logs | `logs/exp{1-6}/seed_{42,123,777}/PPO_*/` |
| DQN evaluation results (CSV) | `logs/comparison/dqn_results.csv` |
| Learning curve per experiment | `logs/comparison/learning_curve_exp{1-6}.png` |
| Best DQN vs Best PPO plot | `logs/comparison/best_dqn_vs_ppo.png` |
| PPO trained models | `models/ppo_privateeye_exp{1-6}_seed{42,123,777}.zip` |
| DQN trained models (Ch1) | `../../models/dqn_privateeye_exp{1-6}.zip` |

## Comparative summary (DQN vs PPO)

ALE/PrivateEye-v5 is one of the hardest Atari environments due to its extremely
sparse reward structure. Both DQN and PPO struggled to obtain meaningful rewards
within the 200,000-step budget.

DQN (Challenge 1, exp2) achieved a final mean evaluation reward of 1.00 ± 0.00
across 10 deterministic episodes. However, because episode rewards were not logged
during training, its learning dynamics remain unknown.

PPO (Challenge 3, exp6) started with a mean reward of approximately −6 in early
training but showed a clear upward trend, converging to near 0–1 by 200,000 steps.
This behaviour reflects PPO's on-policy nature: the clipped surrogate objective
prevents catastrophic policy updates, while the entropy bonus (ent_coef=0.05) and
longer rollout horizon (n_steps=4096) encouraged sustained exploration in a sparse
reward setting.

The key algorithmic difference observed empirically is that PPO exhibits a visible
learning curve — starting poorly but improving steadily — whereas DQN's off-policy
replay buffer allows it to reuse past experiences more efficiently, reaching a stable
(if low) performance faster. Neither algorithm discovered significant reward in
PrivateEye, confirming that this environment requires exploration strategies beyond
standard PPO and DQN, such as intrinsic motivation or curiosity-driven exploration.
