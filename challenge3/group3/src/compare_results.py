import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import integrate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─────────────────────────────────────────────────────────────
# Challenge 3 – DQN vs PPO Comparison Script
# Group 3 | ALE/PrivateEye-v5
#
# Generates all plots and metrics required by the challenge:
#   1. Learning curves (reward vs steps) with shaded std bands
#   2. Final performance table (mean ± std per experiment)
#   3. AUC (area under the learning curve, normalised)
#   4. Sample efficiency (steps to reach target score)
#
# Usage:
#   python src/compare_results.py
#
# Assumes this folder structure (run from challenge3/group3/):
#   Ch1 DQN logs : ../../logs/exp_1/DQN_1/  ... DQN_3/
#   Ch3 PPO logs : ./logs/exp1/seed_42/     ... seed_777/
# ─────────────────────────────────────────────────────────────

# ── Output folder for all figures ────────────────────────────
os.makedirs("logs/comparison", exist_ok=True)

# ── Colours ──────────────────────────────────────────────────
DQN_COLOR = "#2196F3"   # blue
PPO_COLOR = "#F44336"   # red

# ─────────────────────────────────────────────────────────────
# 1. TensorBoard reader
# ─────────────────────────────────────────────────────────────
def read_tb_scalar(log_dir, tag):
    """
    Read a scalar tag from a TensorBoard event file.
    Returns (steps, values) as numpy arrays.
    Returns (None, None) if tag not found.
    """
    ea = EventAccumulator(log_dir)
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        return None, None
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


# ─────────────────────────────────────────────────────────────
# 2. Load DQN logs  (Challenge 1)
#    Path : ../../logs/exp_<N>/DQN_<seed_idx>/
#    Tag  : rollout/ep_rew_mean
# ─────────────────────────────────────────────────────────────
DQN_LOG_BASE = "../../logs"
DQN_TAG      = "rollout/ep_rew_mean"

# Map experiment name → subfolder name used in Ch1
DQN_EXPS = {
    "exp1": "exp_1",
    "exp2": "exp_2",
    "exp3": "exp_3",
    "exp4": "exp_4",
    "exp5": "exp_5",
    "exp6": "exp_6",
}

def load_dqn_experiment(exp_key):
    """Load all seeds for one DQN experiment. Returns list of (steps, values)."""
    folder   = DQN_EXPS[exp_key]
    base_dir = os.path.join(DQN_LOG_BASE, folder)
    runs     = []
    for seed_folder in sorted(os.listdir(base_dir)):          # DQN_1, DQN_2, DQN_3
        seed_dir = os.path.join(base_dir, seed_folder)
        if not os.path.isdir(seed_dir):
            continue
        steps, values = read_tb_scalar(seed_dir, DQN_TAG)
        if steps is not None:
            runs.append((steps, values))
    return runs


# ─────────────────────────────────────────────────────────────
# 3. Load PPO logs  (Challenge 3)
#    Path : ./logs/exp1/seed_42/PPO_1/
#    Tag  : rollout/ep_rew_mean
# ─────────────────────────────────────────────────────────────
PPO_LOG_BASE = "./logs"
PPO_TAG      = "rollout/ep_rew_mean"
PPO_SEEDS    = [42, 123, 777]

def load_ppo_experiment(exp_key):
    """Load all seeds for one PPO experiment. Returns list of (steps, values)."""
    base_dir = os.path.join(PPO_LOG_BASE, exp_key)
    runs     = []
    for seed in PPO_SEEDS:
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            print(f"  [WARN] PPO log not found: {seed_dir}")
            continue
        # SB3 creates a PPO_1 subfolder inside the tensorboard log dir
        for sub in sorted(os.listdir(seed_dir)):
            sub_dir = os.path.join(seed_dir, sub)
            if os.path.isdir(sub_dir):
                steps, values = read_tb_scalar(sub_dir, PPO_TAG)
                if steps is not None:
                    runs.append((steps, values))
                    break
    return runs


# ─────────────────────────────────────────────────────────────
# 4. Interpolation helper
#    Puts all seeds on a common step grid for easy averaging
# ─────────────────────────────────────────────────────────────
def interpolate_runs(runs, n_points=200):
    """
    Interpolate all (steps, values) runs onto a common grid.
    Returns (common_steps, matrix) where matrix is (n_seeds, n_points).
    """
    if not runs:
        return None, None
    max_step   = min(r[0][-1] for r in runs)   # shortest run determines limit
    grid       = np.linspace(0, max_step, n_points)
    matrix     = np.array([np.interp(grid, r[0], r[1]) for r in runs])
    return grid, matrix


# ─────────────────────────────────────────────────────────────
# 5. Metrics helpers
# ─────────────────────────────────────────────────────────────
def compute_auc(steps, mean_values):
    """Normalised area under the learning curve."""
    if steps is None:
        return float("nan")
    auc = integrate.trapezoid(mean_values, steps)
    return auc / steps[-1]   # normalise by total steps


def sample_efficiency(steps, mean_values, target):
    """Steps needed to first reach `target` mean reward. None if never reached."""
    if steps is None:
        return None
    indices = np.where(mean_values >= target)[0]
    return int(steps[indices[0]]) if len(indices) > 0 else None


# ─────────────────────────────────────────────────────────────
# 6. Main comparison
# ─────────────────────────────────────────────────────────────
EXPERIMENTS  = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
TARGET_SCORE = 500    # PrivateEye baseline target (adjust if needed)

print("\n" + "="*60)
print("  Challenge 3 – DQN vs PPO Comparison")
print("  Group 3 | ALE/PrivateEye-v5")
print("="*60)

# Storage for metrics table
metrics = []

for exp in EXPERIMENTS:
    print(f"\n── {exp} ──────────────────────────────────────")

    # Load
    dqn_runs = load_dqn_experiment(exp)
    ppo_runs = load_ppo_experiment(exp)

    print(f"  DQN seeds loaded: {len(dqn_runs)} | PPO seeds loaded: {len(ppo_runs)}")

    # Interpolate
    dqn_steps, dqn_mat = interpolate_runs(dqn_runs)
    ppo_steps, ppo_mat = interpolate_runs(ppo_runs)

    # Means and stds
    dqn_mean = np.mean(dqn_mat, axis=0) if dqn_mat is not None else None
    dqn_std  = np.std(dqn_mat,  axis=0) if dqn_mat is not None else None
    ppo_mean = np.mean(ppo_mat, axis=0) if ppo_mat is not None else None
    ppo_std  = np.std(ppo_mat,  axis=0) if ppo_mat is not None else None

    # ── Plot: learning curve ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    if dqn_steps is not None:
        ax.plot(dqn_steps, dqn_mean, color=DQN_COLOR, lw=2, label="DQN (Ch1)")
        ax.fill_between(dqn_steps,
                        dqn_mean - dqn_std,
                        dqn_mean + dqn_std,
                        alpha=0.2, color=DQN_COLOR)

    if ppo_steps is not None:
        ax.plot(ppo_steps, ppo_mean, color=PPO_COLOR, lw=2, label="PPO (Ch3)")
        ax.fill_between(ppo_steps,
                        ppo_mean - ppo_std,
                        ppo_mean + ppo_std,
                        alpha=0.2, color=PPO_COLOR)

    ax.axhline(TARGET_SCORE, color="gray", lw=1, ls="--",
               label=f"Target score ({TARGET_SCORE})")
    ax.set_title(f"DQN vs PPO — {exp} | ALE/PrivateEye-v5", fontsize=13)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = f"logs/comparison/learning_curve_{exp}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  📊 Saved: {fig_path}")

    # ── Metrics ──────────────────────────────────────────────
    dqn_final_mean = float(np.mean(dqn_mat[:, -1])) if dqn_mat is not None else float("nan")
    dqn_final_std  = float(np.std(dqn_mat[:,  -1])) if dqn_mat is not None else float("nan")
    ppo_final_mean = float(np.mean(ppo_mat[:, -1])) if ppo_mat is not None else float("nan")
    ppo_final_std  = float(np.std(ppo_mat[:,  -1])) if ppo_mat is not None else float("nan")

    dqn_auc  = compute_auc(dqn_steps, dqn_mean)
    ppo_auc  = compute_auc(ppo_steps, ppo_mean)
    dqn_seff = sample_efficiency(dqn_steps, dqn_mean, TARGET_SCORE)
    ppo_seff = sample_efficiency(ppo_steps, ppo_mean, TARGET_SCORE)

    metrics.append({
        "exp":           exp,
        "dqn_final":     f"{dqn_final_mean:.1f} ± {dqn_final_std:.1f}",
        "ppo_final":     f"{ppo_final_mean:.1f} ± {ppo_final_std:.1f}",
        "dqn_auc":       f"{dqn_auc:.2f}",
        "ppo_auc":       f"{ppo_auc:.2f}",
        "dqn_seff":      dqn_seff if dqn_seff else "never",
        "ppo_seff":      ppo_seff if ppo_seff else "never",
    })

# ─────────────────────────────────────────────────────────────
# 7. Overall learning curve (best exp of each algorithm)
#    Uses the experiment with the highest final mean reward
# ─────────────────────────────────────────────────────────────
print("\n── Generating overall best-vs-best plot ─────────────")

best_dqn_exp, best_dqn_val = None, -np.inf
best_ppo_exp, best_ppo_val = None, -np.inf

for m in metrics:
    try:
        dqn_val = float(m["dqn_final"].split(" ")[0])
        ppo_val = float(m["ppo_final"].split(" ")[0])
        if dqn_val > best_dqn_val:
            best_dqn_val = dqn_val
            best_dqn_exp = m["exp"]
        if ppo_val > best_ppo_val:
            best_ppo_val = ppo_val
            best_ppo_exp = m["exp"]
    except Exception:
        continue

print(f"  Best DQN experiment: {best_dqn_exp} ({best_dqn_val:.1f})")
print(f"  Best PPO experiment: {best_ppo_exp} ({best_ppo_val:.1f})")

dqn_best_runs = load_dqn_experiment(best_dqn_exp)
ppo_best_runs = load_ppo_experiment(best_ppo_exp)
dqn_steps, dqn_mat = interpolate_runs(dqn_best_runs)
ppo_steps, ppo_mat = interpolate_runs(ppo_best_runs)

fig, ax = plt.subplots(figsize=(10, 6))

if dqn_steps is not None:
    dqn_mean = np.mean(dqn_mat, axis=0)
    dqn_std  = np.std(dqn_mat,  axis=0)
    ax.plot(dqn_steps, dqn_mean, color=DQN_COLOR, lw=2.5,
            label=f"DQN best ({best_dqn_exp})")
    ax.fill_between(dqn_steps,
                    dqn_mean - dqn_std,
                    dqn_mean + dqn_std,
                    alpha=0.2, color=DQN_COLOR)

if ppo_steps is not None:
    ppo_mean = np.mean(ppo_mat, axis=0)
    ppo_std  = np.std(ppo_mat,  axis=0)
    ax.plot(ppo_steps, ppo_mean, color=PPO_COLOR, lw=2.5,
            label=f"PPO best ({best_ppo_exp})")
    ax.fill_between(ppo_steps,
                    ppo_mean - ppo_std,
                    ppo_mean + ppo_std,
                    alpha=0.2, color=PPO_COLOR)

ax.axhline(TARGET_SCORE, color="gray", lw=1.2, ls="--",
           label=f"Target score ({TARGET_SCORE})")
ax.set_title("Best DQN vs Best PPO — ALE/PrivateEye-v5\n(shaded = ±1 std across 3 seeds)",
             fontsize=13)
ax.set_xlabel("Environment Steps", fontsize=11)
ax.set_ylabel("Mean Episode Reward", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

fig_path = "logs/comparison/best_dqn_vs_ppo.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  📊 Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────
# 8. Print metrics table
# ─────────────────────────────────────────────────────────────
print("\n\n" + "="*75)
print("  METRICS TABLE — DQN vs PPO | ALE/PrivateEye-v5")
print("="*75)
print(f"  {'EXP':<6} {'DQN FINAL':>16} {'PPO FINAL':>16} "
      f"{'DQN AUC':>9} {'PPO AUC':>9} {'DQN SEFF':>12} {'PPO SEFF':>12}")
print(f"  {'─'*6} {'─'*16} {'─'*16} {'─'*9} {'─'*9} {'─'*12} {'─'*12}")
for m in metrics:
    print(f"  {m['exp']:<6} {m['dqn_final']:>16} {m['ppo_final']:>16} "
          f"{m['dqn_auc']:>9} {m['ppo_auc']:>9} "
          f"{str(m['dqn_seff']):>12} {str(m['ppo_seff']):>12}")
print("="*75)
print("\n  Figures saved in: logs/comparison/")
print("  Use these plots directly in your IEEE paper.\n")