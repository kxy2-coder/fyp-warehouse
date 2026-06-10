# =============================================================================
# eval_best_model.py — Evaluate the saved best_model.zip
# =============================================================================
# Loads best_model.zip (saved by EvalCallback during training),
# runs the convergence test from the default layout, evaluates the
# RL-recommended layout, and compares it against the default baseline.
#
# Usage:
#   python eval_best_model.py
#   python eval_best_model.py --eval-runs 3   # more simulation runs per layout
# =============================================================================

import argparse
import numpy as np

from rl_env import (
    WarehouseLayoutEnv, run_single_simulation, calibrate_kpi_bounds,
    compute_depot_cols,
    DEFAULT_AISLE_WIDTH, DEFAULT_CENTRE_AISLE, DEFAULT_DEPOT_COL,
    DEFAULT_DEPOT_COUNT, DEFAULT_SHELF_START, DEFAULT_SHELF_END,
    DEFAULT_CROSS_AISLE_ROW, DEFAULT_CROSS_AISLE_ON,
    W_PICKS, W_DISTANCE, W_CONGESTION, EVAL_RUNS
)

parser = argparse.ArgumentParser()
parser.add_argument("--eval-runs", type=int, default=EVAL_RUNS,
                    help="Simulation runs per layout (default: same as rl_env.py)")
parser.add_argument("--model", type=str, default="best_model.zip",
                    help="Path to model file (default: best_model.zip)")
args = parser.parse_args()

from stable_baselines3 import PPO

print("=" * 60)
print("  EVALUATING SAVED MODEL:", args.model)
print("=" * 60)

# Load KPI bounds (uses saved kpi_bounds.json automatically)
kpi_bounds = calibrate_kpi_bounds()

# Create environment and load model
env   = WarehouseLayoutEnv(kpi_bounds=kpi_bounds)
model = PPO.load(args.model)
print(f"  Model loaded from {args.model}\n")

# ── Convergence test ──────────────────────────────────────────────────────────
print("  Convergence test from default layout...")
print("-" * 60)

env.reset()
env.aisle_width        = DEFAULT_AISLE_WIDTH
env.centre_aisle_width = DEFAULT_CENTRE_AISLE
env.depot_count        = DEFAULT_DEPOT_COUNT
env.shelf_start_row    = DEFAULT_SHELF_START
env.shelf_end_row      = DEFAULT_SHELF_END
env.cross_aisle_row    = DEFAULT_CROSS_AISLE_ROW
env.cross_aisle_on     = DEFAULT_CROSS_AISLE_ON
env.steps_taken        = 0
obs = env._get_observation()

print(f"  Start: aisle_w={env.aisle_width}, centre={env.centre_aisle_width}, "
      f"depots={env.depot_count} {compute_depot_cols(env.depot_count)}, "
      f"shelf={env.shelf_start_row}-{env.shelf_end_row}, "
      f"cross={'on@'+str(env.cross_aisle_row) if env.cross_aisle_on else 'off'}")
print()

converged_at = None
prev_params  = {
    "aisle_width":        env.aisle_width,
    "centre_aisle_width": env.centre_aisle_width,
    "depot_count":        env.depot_count,
    "shelf_start_row":    env.shelf_start_row,
    "shelf_end_row":      env.shelf_end_row,
    "cross_aisle_row":    env.cross_aisle_row,
    "cross_aisle_on":     env.cross_aisle_on,
}

for step in range(1, 51):
    action, _ = model.predict(obs, deterministic=True)
    env._apply_action(action)
    obs = env._get_observation()

    curr_params = {
        "aisle_width":        env.aisle_width,
        "centre_aisle_width": env.centre_aisle_width,
        "depot_count":        env.depot_count,
        "shelf_start_row":    env.shelf_start_row,
        "shelf_end_row":      env.shelf_end_row,
        "cross_aisle_row":    env.cross_aisle_row,
        "cross_aisle_on":     env.cross_aisle_on,
    }
    changed = [k for k in curr_params if curr_params[k] != prev_params[k]]
    print(f"  Step {step:2d}: aisle_w={curr_params['aisle_width']}, "
          f"centre={curr_params['centre_aisle_width']}, "
          f"depots={curr_params['depot_count']} {compute_depot_cols(curr_params['depot_count'])}, "
          f"shelf={curr_params['shelf_start_row']}-{curr_params['shelf_end_row']}, "
          f"cross={'on@'+str(curr_params['cross_aisle_row']) if curr_params['cross_aisle_on'] else 'off'}  "
          f"[{', '.join(changed) if changed else 'converged'}]")

    if not changed:
        converged_at = step
        break
    prev_params = curr_params
else:
    print("  Reached 50 steps without converging.")

# ── Norm helpers ──────────────────────────────────────────────────────────────
def norm_higher_better(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.5

def norm_lower_better(v, lo, hi):
    return float(np.clip((hi - v) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.5

picks_lo, picks_hi = kpi_bounds["picks_per_hour"]
dist_lo,  dist_hi  = kpi_bounds["dist_per_agent"]
cong_lo,  cong_hi  = kpi_bounds["congestion_rate"]

# ── Evaluate RL layout ────────────────────────────────────────────────────────
print(f"\n  Evaluating RL layout ({args.eval_runs} runs)...")
rl_depot_cols = compute_depot_cols(env.depot_count)
rl_result = run_single_simulation(
    aisle_width=env.aisle_width,
    centre_aisle_width=env.centre_aisle_width,
    depot_col=rl_depot_cols[0],
    shelf_start_row=env.shelf_start_row,
    shelf_end_row=env.shelf_end_row,
    cross_aisle_row=env.cross_aisle_row,
    eval_runs=args.eval_runs,
    depot_cols=rl_depot_cols,
    depot_count=env.depot_count,
    depot_row=0,
    cross_aisle_enabled=bool(env.cross_aisle_on),
)

print()
rl_reward = None
if rl_result is None:
    print("  ERROR: RL layout is invalid.")
else:
    steps_msg = (f"converged at step {converged_at}"
                 if converged_at else "no convergence after 50 steps")
    P = norm_higher_better(rl_result["picks_per_hour"], picks_lo, picks_hi)
    D = norm_lower_better(rl_result["dist_per_agent"],  dist_lo,  dist_hi)
    C = norm_lower_better(rl_result["congestion_rate"], cong_lo,  cong_hi)
    rl_reward = W_PICKS*P + W_DISTANCE*D + W_CONGESTION*C

    print(f"  RL LAYOUT ({steps_msg})")
    print(f"    aisle_w={env.aisle_width}, centre={env.centre_aisle_width}, "
          f"depots={env.depot_count} {rl_depot_cols}, "
          f"shelf={env.shelf_start_row}-{env.shelf_end_row}, "
          f"cross={'on@'+str(env.cross_aisle_row) if env.cross_aisle_on else 'off'}")
    print(f"    Picks/hr   : {rl_result['picks_per_hour']:.2f}  (norm {P:.2f})")
    print(f"    Dist/agent : {rl_result['dist_per_agent']:.1f} m  (norm {D:.2f})")
    print(f"    Congestion : {rl_result['congestion_rate']*100:.2f}%  (norm {C:.2f})")
    print(f"    Item cells : {rl_result['num_items']} (raw, not in reward)")
    print(f"    Reward     : {rl_reward:.3f}")

# ── Evaluate baseline ─────────────────────────────────────────────────────────
print(f"\n  Evaluating BASELINE layout ({args.eval_runs} runs)...")
bl_result = run_single_simulation(
    aisle_width=DEFAULT_AISLE_WIDTH,
    centre_aisle_width=DEFAULT_CENTRE_AISLE,
    depot_col=DEFAULT_DEPOT_COL,
    shelf_start_row=DEFAULT_SHELF_START,
    shelf_end_row=DEFAULT_SHELF_END,
    cross_aisle_row=DEFAULT_CROSS_AISLE_ROW,
    eval_runs=args.eval_runs,
)

print()
bl_reward = None
if bl_result is None:
    print("  ERROR: Baseline layout failed.")
else:
    P_b = norm_higher_better(bl_result["picks_per_hour"], picks_lo, picks_hi)
    D_b = norm_lower_better(bl_result["dist_per_agent"],  dist_lo,  dist_hi)
    C_b = norm_lower_better(bl_result["congestion_rate"], cong_lo,  cong_hi)
    bl_reward = W_PICKS*P_b + W_DISTANCE*D_b + W_CONGESTION*C_b

    print(f"  BASELINE (default layout)")
    print(f"    aisle_w={DEFAULT_AISLE_WIDTH}, centre={DEFAULT_CENTRE_AISLE}, "
          f"depot_col={DEFAULT_DEPOT_COL}, "
          f"shelf={DEFAULT_SHELF_START}-{DEFAULT_SHELF_END}, "
          f"cross_aisle={DEFAULT_CROSS_AISLE_ROW}")
    print(f"    Picks/hr   : {bl_result['picks_per_hour']:.2f}  (norm {P_b:.2f})")
    print(f"    Dist/agent : {bl_result['dist_per_agent']:.1f} m  (norm {D_b:.2f})")
    print(f"    Congestion : {bl_result['congestion_rate']*100:.2f}%  (norm {C_b:.2f})")
    print(f"    Item cells : {bl_result['num_items']} (raw, not in reward)")
    print(f"    Reward     : {bl_reward:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
if rl_reward is not None and bl_reward is not None:
    diff = rl_reward - bl_reward
    sign = "+" if diff >= 0 else ""
    print(f"  RL layout reward  : {rl_reward:.3f}")
    print(f"  Baseline reward   : {bl_reward:.3f}")
    print(f"  Difference        : {sign}{diff:.3f}  "
          f"({'RL better' if diff > 0 else 'Baseline better' if diff < 0 else 'Equal'})")
    if rl_result and bl_result:
        cong_diff = (bl_result['congestion_rate'] - rl_result['congestion_rate']) * 100
        sign_c = "+" if cong_diff >= 0 else ""
        print(f"  Congestion change : {sign_c}{cong_diff:.2f}% "
              f"({'improvement' if cong_diff > 0 else 'worse'})")
print("=" * 60)
