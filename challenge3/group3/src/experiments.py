import os
import sys
import time
import yaml
import subprocess
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Challenge 3 – Experiment Runner
# Group 3 | ALE/PrivateEye-v5
#
# Runs all PPO experiments automatically:
#   6 experiments × 3 seeds = 18 total training runs
#
# Usage:
#   python experiments.py                        # run all experiments
#   python experiments.py --exp exp1             # run only exp1 (3 seeds)
#   python experiments.py --exp exp1 --seed 42   # single run
#   python experiments.py --skip exp1            # skip exp1, run the rest
#   python experiments.py --skip exp1 exp2       # skip multiple
# ─────────────────────────────────────────────────────────────

import argparse
parser = argparse.ArgumentParser(description="Run all PPO experiments for Challenge 3")
parser.add_argument("--exp",  type=str, default=None,
                    help="Run only this experiment (e.g. exp1). Default: all.")
parser.add_argument("--seed", type=int, default=None,
                    help="Run only this seed. Default: all seeds from config.")
parser.add_argument("--skip", type=str, nargs="+", default=[],
                    help="Skip one or more experiments (e.g. --skip exp1 exp2).")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────
with open("configs/hyperparameters.yaml", "r") as f:
    config = yaml.safe_load(f)

all_experiments = list(config["experiments"].keys())
all_seeds       = config["seeds"]

# Apply --exp filter
experiments_to_run = [args.exp] if args.exp else all_experiments
seeds_to_run       = [args.seed] if args.seed else all_seeds

# Apply --skip filter
if args.skip:
    experiments_to_run = [e for e in experiments_to_run if e not in args.skip]
    print(f"\n  ⏭️  Skipping: {args.skip}")

# Validate
for exp in experiments_to_run:
    if exp not in config["experiments"]:
        print(f"[ERROR] Experiment '{exp}' not found in hyperparameters.yaml")
        print(f"        Available: {all_experiments}")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Summary before starting
# ─────────────────────────────────────────────────────────────
total_runs = len(experiments_to_run) * len(seeds_to_run)
print("\n" + "="*60)
print("  Challenge 3 – PPO Experiment Runner")
print("  Group 3 | ALE/PrivateEye-v5")
print("="*60)
print(f"  Experiments  : {experiments_to_run}")
print(f"  Seeds        : {seeds_to_run}")
print(f"  Total runs   : {total_runs}")
print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60 + "\n")

# ─────────────────────────────────────────────────────────────
# Run all experiments
# ─────────────────────────────────────────────────────────────
results = []
run_idx  = 0

for exp in experiments_to_run:
    for seed in seeds_to_run:
        run_idx += 1
        run_label = f"[{run_idx}/{total_runs}] {exp} | seed={seed}"

        print(f"\n{'─'*60}")
        print(f"  Starting : {run_label}")
        print(f"  Time     : {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*60}")

        start_time = time.time()

        # Calls train.py with the right args
        cmd = [sys.executable, "src/train.py", "--exp", exp, "--seed", str(seed)]

        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            status  = "OK"
            print(f"\n  ✅ Finished {run_label} in {elapsed/60:.1f} min")

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            status  = "FAILED"
            print(f"\n  ❌ FAILED  {run_label} after {elapsed/60:.1f} min")
            print(f"     Exit code: {e.returncode}")

        results.append({
            "exp":     exp,
            "seed":    seed,
            "status":  status,
            "minutes": round((time.time() - start_time) / 60, 1),
        })

# ─────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("  EXPERIMENT RUNNER — FINAL SUMMARY")
print("="*60)
print(f"  {'EXP':<8} {'SEED':<8} {'STATUS':<10} {'MINUTES'}")
print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
for r in results:
    icon = "✅" if r["status"] == "OK" else "❌"
    print(f"  {r['exp']:<8} {r['seed']:<8} {icon} {r['status']:<8} {r['minutes']} min")

succeeded = sum(1 for r in results if r["status"] == "OK")
failed    = total_runs - succeeded

print(f"\n  Total : {succeeded}/{total_runs} runs succeeded"
      + (f", {failed} failed ⚠️" if failed else " ✅"))
print(f"  Ended : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

if failed > 0:
    print("\n  ⚠️  Some runs failed. Check the output above for details.")
    print("     You can re-run a single failed run with:")
    for r in results:
        if r["status"] == "FAILED":
            print(f"     python src/experiments.py --exp {r['exp']} --seed {r['seed']}")
    sys.exit(1)
