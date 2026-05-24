# =============================================================================
# compress.py — Validate compressed simulation against full 8-hour run
# =============================================================================
# Runs the default layout under the compressed (3600-tick) simulation and
# optionally the full (28800-tick) simulation for direct comparison.
# Also supports calibrating kpi_bounds.json from 100 random layouts.
#
# Usage:
#   python compress.py              # compressed only (fast ~10s)
#   python compress.py --full       # compressed + full side by side (~90s)
#   python compress.py --full --runs 5
#   python compress.py --calibrate            # 100 random layouts, ~17 min
#   python compress.py --calibrate --cal-samples 50   # fewer samples, faster
# =============================================================================

import argparse
import random
import rl_env
from rl_env import (
    run_single_simulation,
    DEFAULT_AISLE_WIDTH, DEFAULT_CENTRE_AISLE, DEFAULT_DEPOT_COL,
    DEFAULT_SHELF_START, DEFAULT_SHELF_END, DEFAULT_CROSS_AISLE_ROW,
    TRAINING_TICKS, SHIFT_TICKS, TIME_SCALE,
    GRID_ROWS, GRID_COLS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true",
                    help="Also run full 28800-tick simulation for direct comparison")
parser.add_argument("--runs", type=int, default=3,
                    help="Simulation runs per mode (default 3)")
parser.add_argument("--calibrate", action="store_true",
                    help="Run random layouts to calibrate kpi_bounds.json dist/agent bound")
parser.add_argument("--cal-samples", type=int, default=100,
                    help="Number of random layouts for calibration (default 100)")
parser.add_argument("--agents", type=int, default=rl_env.NUM_AGENTS,
                    help=f"Number of pickers (default {rl_env.NUM_AGENTS}) — "
                         f"overrides rl_env.NUM_AGENTS for this run")
parser.add_argument("--lambda-base", type=float, default=None,
                    help="NHPP base arrival rate per tick (overrides agent.py default)")
args = parser.parse_args()

# Apply --agents / --lambda-base overrides before any simulation runs.
# run_single_simulation reads rl_env.NUM_AGENTS at call-time, and
# nhpp_arrival() reads agent.LAMBDA_BASE — so monkey-patching both
# propagates through identically to how rl_env.py does it.
rl_env.NUM_AGENTS = args.agents
if args.lambda_base is not None:
    import agent as _agent_module
    _agent_module.LAMBDA_BASE = args.lambda_base
    rl_env.LAMBDA_BASE        = args.lambda_base    # used in calibration ceiling

NUM_AGENTS  = rl_env.NUM_AGENTS
LAMBDA_BASE = rl_env.LAMBDA_BASE

W = 60

print("=" * W)
print("  COMPRESSED SIMULATION VALIDATION  —  Default Layout")
print(f"  Compressed : {TRAINING_TICKS} ticks  (TIME_SCALE = x{TIME_SCALE:.0f})")
if args.full:
    print(f"  Full       : {SHIFT_TICKS} ticks  (8hr shift)")
print(f"  Runs       : {args.runs}   Agents: {NUM_AGENTS}   λ_base: {LAMBDA_BASE}")
print("=" * W)

# ── Compressed run ────────────────────────────────────────────────────────────
print(f"\n  Running compressed ({TRAINING_TICKS} ticks, {args.runs} runs)...", end=" ", flush=True)
c = run_single_simulation(
    DEFAULT_AISLE_WIDTH, DEFAULT_CENTRE_AISLE, DEFAULT_DEPOT_COL,
    DEFAULT_SHELF_START, DEFAULT_SHELF_END, DEFAULT_CROSS_AISLE_ROW,
    eval_runs=args.runs,
)
print("done")

print(f"\n  COMPRESSED RESULTS")
print(f"  {'Picks/hr':<30} {c['picks_per_hour']:>8.2f}")
print(f"  {'Dist/agent (m)':<30} {c['dist_per_agent']:>8.1f}")
print(f"  {'Congestion':<30} {c['congestion_rate']*100:>7.2f}%")
print(f"  {'Jobs completed':<30} {c['jobs_completed']:>8.1f}")
print(f"  {'Jobs arrived':<30} {c['jobs_arrived']:>8.1f}")

print(f"\n  x{TIME_SCALE:.0f} EXTRAPOLATION  (multiply counts to estimate 8hr equivalents)")
print(f"  {'Completed x8':<30} {c['jobs_completed']*TIME_SCALE:>8.1f}   (8hr expect ~2400)")
print(f"  {'Dist/agent x8 (m)':<30} {c['dist_per_agent']*TIME_SCALE:>8.1f}   (8hr expect ~13000-15000)")
print(f"  {'Congestion (no x8)':<30} {c['congestion_rate']*100:>7.2f}%  (8hr expect ~30-50%)")

# ── Full run ──────────────────────────────────────────────────────────────────
if args.full:
    import rl_env
    print(f"\n  Running full ({SHIFT_TICKS} ticks, {args.runs} runs)...", end=" ", flush=True)
    rl_env.TRAINING_TICKS = SHIFT_TICKS
    rl_env.TIME_SCALE     = 1
    f = run_single_simulation(
        DEFAULT_AISLE_WIDTH, DEFAULT_CENTRE_AISLE, DEFAULT_DEPOT_COL,
        DEFAULT_SHELF_START, DEFAULT_SHELF_END, DEFAULT_CROSS_AISLE_ROW,
        eval_runs=args.runs,
    )
    rl_env.TRAINING_TICKS = TRAINING_TICKS
    rl_env.TIME_SCALE     = TIME_SCALE
    print("done")

    print(f"\n  SIDE-BY-SIDE COMPARISON")
    print(f"  {'':<28} {'Compressed':>12} {'Full 8hr':>10} {'Error':>8}")
    print("  " + "-" * (W - 2))

    def err(a, b):
        return abs(a - b) / max(abs(b), 1) * 100

    print(f"  {'Picks/hr':<28} {c['picks_per_hour']:>12.2f} {f['picks_per_hour']:>10.2f} {err(c['picks_per_hour'], f['picks_per_hour']):>7.1f}%")
    print(f"  {'Dist/agent (m)':<28} {c['dist_per_agent']:>12.1f} {f['dist_per_agent']:>10.1f} {err(c['dist_per_agent'], f['dist_per_agent']):>7.1f}%")
    print(f"  {'Congestion':<28} {c['congestion_rate']*100:>11.2f}% {f['congestion_rate']*100:>9.2f}% {err(c['congestion_rate'], f['congestion_rate']):>7.1f}%")
    print(f"  {'Jobs completed':<28} {c['jobs_completed']:>12.1f} {f['jobs_completed']:>10.1f}")
    print()
    print(f"  {'Completed x8':<28} {c['jobs_completed']*TIME_SCALE:>12.1f} {f['jobs_completed']:>10.1f} {err(c['jobs_completed']*TIME_SCALE, f['jobs_completed']):>7.1f}%")
    print(f"  {'Dist/agent x8 (m)':<28} {c['dist_per_agent']*TIME_SCALE:>12.1f} {f['dist_per_agent']:>10.1f} {err(c['dist_per_agent']*TIME_SCALE, f['dist_per_agent']):>7.1f}%")

print("\n" + "=" * W)

# ── Calibration run ───────────────────────────────────────────────────────────
if args.calibrate:
    import json, os

    PHYSICS_PICKS_MAX = LAMBDA_BASE * SHIFT_TICKS / 8.0  # = 300.0

    print(f"\n  CALIBRATING KPI BOUNDS FROM VALID LAYOUTS")
    print(f"  Sampling {args.cal_samples} random layouts (1 run each)...")
    print(f"  Parameter ranges (matches RL training space):")
    print(f"    aisle_width   : 2–3")
    print(f"    centre_aisle  : 1–5")
    print(f"    depot_count   : 1–4")
    print(f"    depot_col     : 2–{GRID_COLS - 3}  (per depot, independently)")
    print(f"    depot_row     : 0 (top, fixed — warehouse is vertically symmetric)")
    print(f"    shelf_start   : 2–8")
    print(f"    shelf_end     : {GRID_ROWS - 7}–{GRID_ROWS - 2}")
    print(f"    cross_aisle   : on or off")
    print(f"    cross_row     : shelf_start+1 to shelf_end-1  (only used if on)")
    print("  " + "-" * (W - 2))

    picks_values    = []
    dist_values     = []
    cong_values     = []
    n_valid         = 0
    n_invalid       = 0

    for i in range(args.cal_samples):
        aw          = random.randint(2, 3)
        caw         = random.randint(1, 5)
        ssr         = random.randint(2, 8)
        ser         = random.randint(GRID_ROWS - 7, GRID_ROWS - 2)
        if ssr >= ser:
            ser = ssr + 1
        depot_count = random.randint(1, 4)
        depot_cols  = [random.randint(2, GRID_COLS - 3) for _ in range(depot_count)]
        cross_on    = random.randint(0, 1)
        # Cross-aisle row must sit strictly between shelf_start and shelf_end.
        # Only used when cross_on=1, but always passed so the value is logged.
        cross_row   = random.randint(ssr + 1, ser - 1)

        result = run_single_simulation(
            aw, caw, depot_cols[0], ssr, ser,
            cross_aisle_row=cross_row,
            eval_runs=1,
            depot_cols=depot_cols,
            depot_count=depot_count,
            depot_row=0,
            cross_aisle_enabled=bool(cross_on),
        )

        if result is not None:
            picks_values.append(result["picks_per_hour"])
            dist_values.append(result["dist_per_agent"])
            cong_values.append(result["congestion_rate"])
            n_valid += 1
        else:
            n_invalid += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1:>4}/{args.cal_samples}  valid={n_valid}  invalid={n_invalid}")

    print("  " + "-" * (W - 2))

    if len(picks_values) < 5:
        print("  WARNING: Too few valid layouts (<5). Cannot calibrate reliably.")
    else:
        # ── Apply ±5% buffer, then clip to physical limits ────────────────────
        # picks/hr  : higher is better — lower bound from data, upper capped at
        #             physics ceiling (can't complete more jobs than arrived)
        picks_lo = max(0.0,              min(picks_values) * 0.95)
        picks_hi = min(PHYSICS_PICKS_MAX, max(picks_values) * 1.05)

        # dist/agent: lower is better — both bounds from data; floor clipped to 0
        dist_lo  = max(0.0, min(dist_values) * 0.95)
        dist_hi  =          max(dist_values) * 1.05

        # congestion: lower is better — both bounds from data; clipped to [0, 1]
        cong_lo  = max(0.0, min(cong_values) * 0.95)
        cong_hi  = min(1.0, max(cong_values) * 1.05)

        print(f"\n  CALIBRATION RESULTS  ({n_valid}/{args.cal_samples} valid layouts)")
        print(f"  {'':28} {'min obs':>10} {'max obs':>10}  {'lo bound':>10} {'hi bound':>10}")
        print(f"  {'-'*68}")
        print(f"  {'picks/hr':<28} {min(picks_values):>10.2f} {max(picks_values):>10.2f}  {picks_lo:>10.2f} {picks_hi:>10.2f}")
        print(f"  {'dist/agent (m)':<28} {min(dist_values):>10.1f} {max(dist_values):>10.1f}  {dist_lo:>10.1f} {dist_hi:>10.1f}")
        print(f"  {'congestion':<28} {min(cong_values):>10.3f} {max(cong_values):>10.3f}  {cong_lo:>10.3f} {cong_hi:>10.3f}")

        print()
        print(f"  SUGGESTED kpi_bounds.json  (copy these values manually):")
        print(f"  " + "-" * (W - 2))

        suggested = {
            "picks_per_hour":  [round(picks_lo, 2), round(picks_hi, 2)],
            "dist_per_agent":  [round(dist_lo,  1), round(dist_hi,  1)],
            "congestion_rate": [round(cong_lo,  4), round(cong_hi,  4)],
            "max_storage":     304,
        }
        print("  " + json.dumps(suggested, indent=4).replace("\n", "\n  "))
        print(f"\n  Current kpi_bounds.json path: {os.path.abspath('kpi_bounds.json')}")

    print("\n" + "=" * W)
