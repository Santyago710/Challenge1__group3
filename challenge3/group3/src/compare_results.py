import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─────────────────────────────────────────────────────────────
# Challenge 3 – DQN vs PPO Comparison Script
# Group 3 | ALE/PrivateEye-v5
# ─────────────────────────────────────────────────────────────

os.makedirs("logs/comparison", exist_ok=True)

DQN_COLOR = "#2196F3"   # blue
PPO_COLOR = "#F44336"   # red

# ─────────────────────────────────────────────────────────────
# 1. TensorBoard reader
# ─────────────────────────────────────────────────────────────
def read_tb_scalar(log_dir, tags):
    """
    Try multiple tag names and return first match.
    Returns (steps, values) or (None, None).
    """
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                steps  = np.array([e.step  for e in events])
                values = np.array([e.value for e in events])
                if len(steps) > 0:
                    return steps, values
    except Exception as e:
        print(f"    [WARN] Could not read {log_dir}: {e}")
    return None, None

# Tags SB3 uses for episode reward
REWARD_TAGS = [
    "rollout/ep_rew_mean",
    "train/ep_rew_mean",
    "eval/mean_reward",
]

# ─────────────────────────────────────────────────────────────
# 2. Load DQN logs (Challenge 1)
#    ../../logs/exp_1/DQN_1/events...
# ─────────────────────────────────────────────────────────────
DQN_LOG_BASE = "../../logs"
DQN_EXPS = {
    "exp1": "exp_1", "exp2": "exp_2", "exp3": "exp_3",
    "exp4": "exp_4", "exp5": "exp_5", "exp6": "exp_6",
}

def load_dqn_experiment(exp_key):
    folder   = DQN_EXPS[exp_key]
    base_dir = os.path.join(DQN_LOG_BASE, folder)
    runs     = []
    for seed_folder in sorted(os.listdir(base_dir)):       # DQN_1, DQN_2, DQN_3
        seed_dir = os.path.join(base_dir, seed_folder)
        if not os.path.isdir(seed_dir):
            continue
        # Try the seed_dir directly first, then any subdirs
        steps, values = read_tb_scalar(seed_dir, REWARD_TAGS)
        if steps is not None:
            runs.append((steps, values))
            continue
        for sub in sorted(os.listdir(seed_dir)):
            sub_dir = os.path.join(seed_dir, sub)
            if os.path.isdir(sub_dir):
                steps, values = read_tb_scalar(sub_dir, REWARD_TAGS)
                if steps is not None:
                    runs.append((steps, values))
                    break
    return runs

# ─────────────────────────────────────────────────────────────
# 3. Load PPO logs (Challenge 3)
#    ./logs/exp1/seed_42/PPO_1/events...
#    Multiple PPO_N subfolders — take the one with most events
# ─────────────────────────────────────────────────────────────
PPO_LOG_BASE = "./logs"
PPO_SEEDS    = [42, 123, 777]

def load_ppo_experiment(exp_key):
    base_dir = os.path.join(PPO_LOG_BASE, exp_key)
    runs     = []
    for seed in PPO_SEEDS:
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            print(f"    [WARN] Not found: {seed_dir}")
            continue
        # Pick the PPO subfolder with the most data points
        best_steps, best_values = None, None
        for sub in sorted(os.listdir(seed_dir)):
            sub_dir = os.path.join(seed_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            steps, values = read_tb_scalar(sub_dir, REWARD_TAGS)
            if steps is not None:
                if best_steps is None or len(steps) > len(best_steps):
                    best_steps  = steps
                    best_values = values
        if best_steps is not None:
            runs.append((best_steps, best_values))
    return runs

# ─────────────────────────────────────────────────────────────
# 4. Interpolation helper
# ─────────────────────────────────────────────────────────────
def interpolate_runs(runs, n_points=200):
    if not runs:
        return None, None
    max_step = min(r[0][-1] for r in runs)
    grid     = np.linspace(0, max_step, n_points)
    matrix   = np.array([np.interp(grid, r[0], r[1]) for r in runs])
    return grid, matrix

# ─────────────────────────────────────────────────────────────
# 5. Metrics helpers
# ─────────────────────────────────────────────────────────────
def compute_auc(steps, mean_values):
    if steps is None:
        return float("nan")
    return integrate.trapezoid(mean_values, steps) / steps[-1]

def sample_efficiency(steps, mean_values, target):
    if steps is None:
        return None
    indices = np.where(mean_values >= target)[0]
    return int(steps[indices[0]]) if len(indices) > 0 else None

# ─────────────────────────────────────────────────────────────
# 6. Main comparison loop
# ─────────────────────────────────────────────────────────────
EXPERIMENTS  = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
TARGET_SCORE = 500

print("\n" + "="*60)
print("  Challenge 3 – DQN vs PPO Comparison")
print("  Group 3 | ALE/PrivateEye-v5")
print("="*60)

metrics = []

for exp in EXPERIMENTS:
    print(f"\n── {exp} ──────────────────────────────────────")

    dqn_runs = load_dqn_experiment(exp)
    ppo_runs = load_ppo_experiment(exp)

    print(f"  DQN seeds loaded: {len(dqn_runs)} | PPO seeds loaded: {len(ppo_runs)}")

    dqn_steps, dqn_mat = interpolate_runs(dqn_runs)
    ppo_steps, ppo_mat = interpolate_runs(ppo_runs)

    dqn_mean = np.mean(dqn_mat, axis=0) if dqn_mat is not None else None
    dqn_std  = np.std(dqn_mat,  axis=0) if dqn_mat is not None else None
    ppo_mean = np.mean(ppo_mat, axis=0) if ppo_mat is not None else None
    ppo_std  = np.std(ppo_mat,  axis=0) if ppo_mat is not None else None

    # ── Plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    if dqn_steps is not None:
        ax.plot(dqn_steps, dqn_mean, color=DQN_COLOR, lw=2, label="DQN (Ch1)")
        ax.fill_between(dqn_steps, dqn_mean - dqn_std,
                        dqn_mean + dqn_std, alpha=0.2, color=DQN_COLOR)

    if ppo_steps is not None:
        ax.plot(ppo_steps, ppo_mean, color=PPO_COLOR, lw=2, label="PPO (Ch3)")
        ax.fill_between(ppo_steps, ppo_mean - ppo_std,
                        ppo_mean + ppo_std, alpha=0.2, color=PPO_COLOR)

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

    # ── Metrics ─────────────────────────────────────────────
    dqn_final_mean = float(np.mean(dqn_mat[:, -1])) if dqn_mat is not None else float("nan")
    dqn_final_std  = float(np.std(dqn_mat[:,  -1])) if dqn_mat is not None else float("nan")
    ppo_final_mean = float(np.mean(ppo_mat[:, -1])) if ppo_mat is not None else float("nan")
    ppo_final_std  = float(np.std(ppo_mat[:,  -1])) if ppo_mat is not None else float("nan")

    metrics.append({
        "exp":      exp,
        "dqn_final": f"{dqn_final_mean:.1f} ± {dqn_final_std:.1f}",
        "ppo_final": f"{ppo_final_mean:.1f} ± {ppo_final_std:.1f}",
        "dqn_auc":   f"{compute_auc(dqn_steps, dqn_mean):.2f}" if dqn_mean is not None else "nan",
        "ppo_auc":   f"{compute_auc(ppo_steps, ppo_mean):.2f}" if ppo_mean is not None else "nan",
        "dqn_seff":  sample_efficiency(dqn_steps, dqn_mean, TARGET_SCORE) or "never",
        "ppo_seff":  sample_efficiency(ppo_steps, ppo_mean, TARGET_SCORE) or "never",
    })

# ─────────────────────────────────────────────────────────────
# 7. Best vs Best overall plot
# ─────────────────────────────────────────────────────────────
print("\n── Generating overall best-vs-best plot ─────────────")

best_dqn_exp, best_dqn_val = None, -np.inf
best_ppo_exp, best_ppo_val = None, -np.inf

for m in metrics:
    try:
        dval = float(m["dqn_final"].split(" ")[0])
        pval = float(m["ppo_final"].split(" ")[0])
        if not np.isnan(dval) and dval > best_dqn_val:
            best_dqn_val, best_dqn_exp = dval, m["exp"]
        if not np.isnan(pval) and pval > best_ppo_val:
            best_ppo_val, best_ppo_exp = pval, m["exp"]
    except Exception:
        continue

# Fallback to exp1 if nothing loaded
best_dqn_exp = best_dqn_exp or "exp1"
best_ppo_exp = best_ppo_exp or "exp1"

print(f"  Best DQN: {best_dqn_exp} ({best_dqn_val:.1f})")
print(f"  Best PPO: {best_ppo_exp} ({best_ppo_val:.1f})")

dqn_steps, dqn_mat = interpolate_runs(load_dqn_experiment(best_dqn_exp))
ppo_steps, ppo_mat = interpolate_runs(load_ppo_experiment(best_ppo_exp))

fig, ax = plt.subplots(figsize=(10, 6))

if dqn_steps is not None:
    dqn_mean = np.mean(dqn_mat, axis=0)
    dqn_std  = np.std(dqn_mat,  axis=0)
    ax.plot(dqn_steps, dqn_mean, color=DQN_COLOR, lw=2.5,
            label=f"DQN best ({best_dqn_exp})")
    ax.fill_between(dqn_steps, dqn_mean - dqn_std,
                    dqn_mean + dqn_std, alpha=0.2, color=DQN_COLOR)

if ppo_steps is not None:
    ppo_mean = np.mean(ppo_mat, axis=0)
    ppo_std  = np.std(ppo_mat,  axis=0)
    ax.plot(ppo_steps, ppo_mean, color=PPO_COLOR, lw=2.5,
            label=f"PPO best ({best_ppo_exp})")
    ax.fill_between(ppo_steps, ppo_mean - ppo_std,
                    ppo_mean + ppo_std, alpha=0.2, color=PPO_COLOR)

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
# 8. Metrics table
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