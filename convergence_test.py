# =============================================================================
# convergence_test.py — Re-run RL convergence test from a saved run folder
# =============================================================================
# Loads best_model.zip + kpi_bounds.json from a saved run folder and runs the
# convergence loop (step 5/5 from rl_env.py main) without retraining.
#
# USAGE:
#   python convergence_test.py --run "Results_final/Agents=4,lambda=0.02" \
#                              --agents 4 --lambda-base 0.02
#
#   python convergence_test.py --run runs/2026-05-23_12-47-59_agents6_lambda0.06 \
#                              --agents 6 --lambda-base 0.06
# =============================================================================

import argparse
import json
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Re-run RL convergence test from a saved run folder")
    parser.add_argument("--run", required=True,
                        help="Path to the saved run folder containing "
                             "best_model.zip and kpi_bounds.json")
    parser.add_argument("--agents", type=int, required=True,
                        help="NUM_AGENTS used during that training run")
    parser.add_argument("--lambda-base", type=float, required=True,
                        help="LAMBDA_BASE used during that training run")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max convergence steps (default 50)")
    args = parser.parse_args()

    # Resolve paths
    run_folder = args.run
    model_path = os.path.join(run_folder, "best_model.zip")
    bounds_path = os.path.join(run_folder, "kpi_bounds.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(bounds_path):
        raise FileNotFoundError(f"KPI bounds not found: {bounds_path}")

    # ── Apply NUM_AGENTS / LAMBDA_BASE overrides BEFORE importing rl_env ──
    # so module-level constants pick up the right values.
    import agent as _agent_module
    _agent_module.LAMBDA_BASE = args.lambda_base

    import rl_env
    rl_env.NUM_AGENTS  = args.agents
    rl_env.LAMBDA_BASE = args.lambda_base

    from rl_env import (
        WarehouseLayoutEnv, run_single_simulation, compute_depot_cols,
        DEFAULT_AISLE_WIDTH, DEFAULT_CENTRE_AISLE, DEFAULT_DEPOT_COUNT,
        DEFAULT_SHELF_START, DEFAULT_SHELF_END,
        DEFAULT_CROSS_AISLE_ROW, DEFAULT_CROSS_AISLE_ON,
        W_PICKS, W_DISTANCE, W_CONGESTION, EVAL_RUNS,
    )
    from stable_baselines3 import PPO

    # ── Load KPI bounds from the saved run ────────────────────────────────
    with open(bounds_path, "r") as f:
        bounds = json.load(f)
    kpi_bounds = {
        "picks_per_hour":  tuple(bounds["picks_per_hour"]),
        "dist_per_agent":  tuple(bounds["dist_per_agent"]),
        "congestion_rate": tuple(bounds["congestion_rate"]),
    }

    print("=" * 60)
    print(f"  CONVERGENCE TEST — {run_folder}")
    print("=" * 60)
    print(f"  N agents     : {args.agents}")
    print(f"  λ_base       : {args.lambda_base}")
    print(f"  KPI bounds   : picks {kpi_bounds['picks_per_hour']}")
    print(f"                 dist  {kpi_bounds['dist_per_agent']}")
    print(f"                 cong  {kpi_bounds['congestion_rate']}")
    print(f"  Max steps    : {args.max_steps}")
    print("=" * 60)

    # ── Build env and load model ──────────────────────────────────────────
    env = WarehouseLayoutEnv(kpi_bounds=kpi_bounds, start_mode="default")
    model = PPO.load(model_path)
    print(f"  Loaded model from {model_path}")

    # ── Reset to default layout ───────────────────────────────────────────
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

    def snapshot():
        return {
            "aisle_width":        env.aisle_width,
            "centre_aisle_width": env.centre_aisle_width,
            "depot_count":        env.depot_count,
            "shelf_start_row":    env.shelf_start_row,
            "shelf_end_row":      env.shelf_end_row,
            "cross_aisle_row":    env.cross_aisle_row,
            "cross_aisle_on":     env.cross_aisle_on,
        }

    def fmt(p):
        return (f"aisle_w={p['aisle_width']}, centre={p['centre_aisle_width']}, "
                f"depots={p['depot_count']} {compute_depot_cols(p['depot_count'])}, "
                f"shelf={p['shelf_start_row']}-{p['shelf_end_row']}, "
                f"cross={'on@'+str(p['cross_aisle_row']) if p['cross_aisle_on'] else 'off'}")

    prev = snapshot()
    print(f"\n  Starting layout : {fmt(prev)}\n")

    converged_at = None
    for step in range(1, args.max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        env._apply_action(action)
        obs = env._get_observation()
        cur = snapshot()
        changed = [k for k in cur if cur[k] != prev[k]]
        print(f"  Step {step:2d}: {fmt(cur)}  [changed: {', '.join(changed) if changed else 'none'}]")
        if not changed:
            converged_at = step
            print(f"  Converged.")
            break
        prev = cur
    else:
        print(f"  Reached {args.max_steps} steps without converging.")

    # ── Evaluate final layout ─────────────────────────────────────────────
    def norm_higher_better(value, lo, hi):
        if hi <= lo: return 0.5
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def norm_lower_better(value, lo, hi):
        if hi <= lo: return 0.5
        return float(np.clip((hi - value) / (hi - lo), 0.0, 1.0))

    picks_lo, picks_hi = kpi_bounds["picks_per_hour"]
    dist_lo,  dist_hi  = kpi_bounds["dist_per_agent"]
    cong_lo,  cong_hi  = kpi_bounds["congestion_rate"]

    print(f"\n  Evaluating final layout ({EVAL_RUNS} runs)...")
    depot_cols = compute_depot_cols(env.depot_count)
    final = run_single_simulation(
        aisle_width=env.aisle_width,
        centre_aisle_width=env.centre_aisle_width,
        depot_col=depot_cols[0],
        shelf_start_row=env.shelf_start_row,
        shelf_end_row=env.shelf_end_row,
        cross_aisle_row=env.cross_aisle_row,
        eval_runs=EVAL_RUNS,
        depot_cols=depot_cols,
        depot_count=env.depot_count,
        depot_row=0,
        cross_aisle_enabled=bool(env.cross_aisle_on),
    )

    print()
    if final is None:
        print("  ERROR: Final layout is invalid.")
        return

    P = norm_higher_better(final['picks_per_hour'], picks_lo, picks_hi)
    D = norm_lower_better(final['dist_per_agent'],  dist_lo,  dist_hi)
    C = norm_lower_better(final['congestion_rate'], cong_lo,  cong_hi)
    reward = W_PICKS*P + W_DISTANCE*D + W_CONGESTION*C

    steps_msg = (f"after {converged_at} step(s)"
                 if converged_at is not None
                 else f"after {args.max_steps} steps (no convergence)")
    print("-" * 60)
    print(f"  RL-RECOMMENDED LAYOUT ({steps_msg}):")
    print(f"    {fmt(snapshot())}")
    print(f"    Picks/hr    : {final['picks_per_hour']:.2f}   (norm: {P:.2f})")
    print(f"    Dist/agent  : {final['dist_per_agent']:.1f}  (norm: {D:.2f})")
    print(f"    Congestion  : {final['congestion_rate']:.4f}  (norm: {C:.2f})")
    print(f"    Item cells  : {final['num_items']}")
    print(f"    Reward      : {reward:.3f}")

    # ── Baseline for comparison ───────────────────────────────────────────
    print()
    print("-" * 60)
    print("  BASELINE (default layout):")
    baseline = run_single_simulation(
        aisle_width=DEFAULT_AISLE_WIDTH,
        centre_aisle_width=DEFAULT_CENTRE_AISLE,
        depot_col=compute_depot_cols(DEFAULT_DEPOT_COUNT)[0],
        shelf_start_row=DEFAULT_SHELF_START,
        shelf_end_row=DEFAULT_SHELF_END,
        cross_aisle_row=DEFAULT_CROSS_AISLE_ROW,
        eval_runs=EVAL_RUNS,
        depot_cols=compute_depot_cols(DEFAULT_DEPOT_COUNT),
        depot_count=DEFAULT_DEPOT_COUNT,
        depot_row=0,
        cross_aisle_enabled=bool(DEFAULT_CROSS_AISLE_ON),
    )
    if baseline is not None:
        Pb = norm_higher_better(baseline['picks_per_hour'], picks_lo, picks_hi)
        Db = norm_lower_better(baseline['dist_per_agent'],  dist_lo,  dist_hi)
        Cb = norm_lower_better(baseline['congestion_rate'], cong_lo,  cong_hi)
        baseline_reward = W_PICKS*Pb + W_DISTANCE*Db + W_CONGESTION*Cb
        print(f"    Picks/hr    : {baseline['picks_per_hour']:.2f}   (norm: {Pb:.2f})")
        print(f"    Dist/agent  : {baseline['dist_per_agent']:.1f}  (norm: {Db:.2f})")
        print(f"    Congestion  : {baseline['congestion_rate']:.4f}  (norm: {Cb:.2f})")
        print(f"    Reward      : {baseline_reward:.3f}")
        print()
        print(f"  RL reward     : {reward:.3f}")
        print(f"  Baseline      : {baseline_reward:.3f}")
        print(f"  Δreward       : {reward - baseline_reward:+.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
