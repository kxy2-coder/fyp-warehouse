# =============================================================================
# rl_env.py — Reinforcement Learning Environment for Warehouse Layout (Sim_V3)
# =============================================================================
# Wraps the Sim_V3 warehouse simulation into a Gymnasium environment for
# training a PPO agent (Stable Baselines 3) to optimise warehouse layout.
#
# SIMULATION BACKEND (Sim_V3):
#   - JuPedSim Collision-Free Speed Model (CFSM) for continuous agent movement
#   - Physics-based collision avoidance — no discrete cell-blocking
#   - Congestion measured as fraction of walking ticks where agent speed
#     dropped below 20% of free speed (1.2 m/s) due to crowd pressure
#   - Human factors (fatigue, experience) removed — JuPedSim physics already
#     provides empirically validated speed variation through crowd dynamics
#   - Demand modelled via NHPP with morning/end-of-shift peaks (β=0.3)
#     calibrated for NUM_AGENTS=6 at λ_base=0.0833 (sweet-spot from
#     sensitivity analysis: 94% completion rate, 38% congestion)
#
# STATE (observation) — 7 normalised floats in [0, 1]:
#   [aisle_width, centre_aisle_width, depot_count, shelf_start_row,
#    shelf_end_row, cross_aisle_row, cross_aisle_on]
#
# ACTIONS — MultiDiscrete([3,3,3,3,3,3,2]), each ±-dim maps 0→-1, 1→0, 2→+1:
#   dim 0: aisle_width       ± 1
#   dim 1: centre_aisle      ± 2
#   dim 2: depot_count       ± 1  (1–4 depots; positions auto-computed equally spaced)
#   dim 3: shelf_start_row   ± 1
#   dim 4: shelf_end_row     ± 1
#   dim 5: cross_aisle_row   ± 1
#   dim 6: cross_aisle_on    0 or 1 (direct binary)
#
# REWARD — weighted sum of 3 normalised KPIs (all in [0, 1]):
#   reward = W_PICKS * P + W_DISTANCE * D + W_CONGESTION * C
#   where:
#     P = picks/hr          (higher is better)
#     D = dist/agent        (lower  is better)
#     C = congestion rate   (lower  is better)
#   Returns -1.0 if layout is physically invalid or below minimum storage
#   (the 240-item floor in run_single_simulation already prevents
#    degenerate tiny-warehouse layouts, so storage is no longer a soft term).
#   Weights: picks=0.30, distance=0.30, congestion=0.40
#   Congestion weighted highest — aligns with professor feedback to
#   prioritise reducing depot/aisle congestion.
#
# DEPOT PLACEMENT — equal-spacing rule (justified by Non-Traditional Layout
#   Design for RMFS with Multiple Workstations, Appendix A):
#   spacing = GRID_COLS / (n_depots + 1)
#   col_i   = int(spacing x i)  for i = 1 ... n_depots
#   This rule achieves within 1.38% of optimal travel distance, making it
#   a near-optimal and computationally tractable default configuration.
#
# USAGE:
#   python rl_env.py                            # default training (10k steps)
#   python rl_env.py --timesteps 50000          # longer training
#   python rl_env.py --recalibrate              # fresh KPI bounds calibration
#   python rl_env.py --start-mode random        # random episode start layouts
#   python rl_env.py --save-results             # save RL layout to dashboard
# =============================================================================

import math
import random
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from grid           import Grid
from agent          import (nhpp_arrival, get_multiplier, SHIFT_TICKS,
                            DEMAND_BETA)
from jupedsim_agent import JupedSimAgent


# =============================================================================
# CONFIGURATION
# =============================================================================

GRID_ROWS  = 25
GRID_COLS  = 35
NUM_AGENTS = 6
LAMBDA_BASE = 0.0833  # Base arrival rate for the non-homogeneous Poisson process (NHPP).
DEMAND_BETA = 0.3     # Amplitude parameter for the non-homogeneous Poisson process (NHPP).

EVAL_RUNS = 5
MAX_STEPS_PER_EPISODE = 10
REPLENISH_DELAY = 100

# ── Training simulation length ────────────────────────────────────────────────
# RL training uses a compressed 1-hour simulation instead of the full 8-hour
# shift. The NHPP time axis is scaled so the full demand shape (morning peak,
# steady state, end-of-shift peak) plays out within TRAINING_TICKS, while λ
# stays the same so agents are never overwhelmed.
#
# TIME_SCALE maps each training tick to its equivalent position in the full
# shift: nhpp_arrival(tick * TIME_SCALE) mirrors the 8-hour demand curve.
#
# KPI notes:
#   picks/hr   — ceiling is λ × TRAINING_TICKS / training_hours = 300, same
#                as the full run. No change to bounds.
#   dist/agent — ~8× lower than full run. kpi_bounds.json must reflect this.
#   congestion — ratio (0–1), scale-invariant. Absolute value will be lower
#                than the full run (fewer simultaneous active agents), but
#                relative ranking between layouts is preserved — which is all
#                the RL needs.
#
# Validation: multiply completed × TIME_SCALE and distance × TIME_SCALE to
# get full-run equivalents and compare against actual 8-hour results.
TRAINING_TICKS = 3600                          # 1 compressed hour
TIME_SCALE     = SHIFT_TICKS / TRAINING_TICKS  # = 8

W_PICKS      = 0.30
W_DISTANCE   = 0.30
W_CONGESTION = 0.40

# ── Default layout parameters ───────────────────────
# All resets, baseline comparisons, and calibration validation use these.
# Change here and it propagates everywhere automatically.
DEFAULT_AISLE_WIDTH       = 2
DEFAULT_CENTRE_AISLE      = 3
DEFAULT_DEPOT_COL         = GRID_COLS // 2
DEFAULT_DEPOT_COUNT       = 1
DEFAULT_SHELF_START       = 2
DEFAULT_SHELF_END         = GRID_ROWS - 2
DEFAULT_CROSS_AISLE_ROW   = (DEFAULT_SHELF_START + DEFAULT_SHELF_END) // 2
DEFAULT_CROSS_AISLE_ON    = 1          # 1 = enabled
DEFAULT_CROSS_AISLE_WIDTH = 2   # rows — 2m wide, enough for two agents to pass

# Sensitivity analysis ranges (not used in training loop — for analysis scripts)
LAMBDA_BASE_RANGE = [0.030, 0.055, 0.078, 0.100, 0.130]
BETA_RANGE        = [0.0, 0.2, 0.3, 0.4]

DEBUG = False   # set True to print per-run job arrival diagnostics


# =============================================================================
# DEPOT PLACEMENT HELPER
# =============================================================================

def compute_depot_cols(n_depots, grid_cols=GRID_COLS):
    """
    Compute equally-spaced depot column positions for a given depot count.

    Formula (equal-interval rule from Non-Traditional Layout Design for RMFS
    with Multiple Workstations, Appendix A):
        spacing = grid_cols / (n_depots + 1)
        col_i   = int(spacing * i)  for i = 1 ... n_depots

    Examples for grid_cols=35:
        n=1 → [17]
        n=2 → [11, 23]
        n=3 → [8, 17, 26]
        n=4 → [7, 14, 21, 28]
    """
    spacing = grid_cols / (n_depots + 1)
    return [int(spacing * i) for i in range(1, n_depots + 1)]


# =============================================================================
# CALIBRATION — discover KPI min/max bounds from random layouts
# =============================================================================

def run_single_simulation(aisle_width, centre_aisle_width, depot_col,
                          shelf_start_row, shelf_end_row,
                          cross_aisle_row=None,
                          cross_aisle_width=DEFAULT_CROSS_AISLE_WIDTH,
                          eval_runs=EVAL_RUNS,
                          replenish_delay=REPLENISH_DELAY,
                          depot_cols=None,
                          depot_count=1,
                          depot_row=0,
                          cross_aisle_enabled=True):
    """
    Build a warehouse layout with the given parameters and run the simulation
    eval_runs times. Returns averaged KPIs, or None if the layout is invalid.

    depot_cols     — list of active depot column positions (overrides depot_col
                     when provided). Length should equal depot_count.
    depot_count    — number of active depots (used when depot_cols is None).
    depot_row      — 0 = top row, 1 = bottom row.
    cross_aisle_enabled — False disables the cross-aisle entirely (None to grid).
    """
    # Determine active depot columns
    if depot_cols is not None:
        active_depot_cols = list(depot_cols)
    else:
        active_depot_cols = [depot_col]

    try:
        grid = Grid(
            rows=GRID_ROWS, cols=GRID_COLS,
            aisle_width=aisle_width,
            centre_aisle_width=centre_aisle_width,
            depot_row=depot_row,
            depot_cols=active_depot_cols,
            shelf_start_row=shelf_start_row,
            shelf_end_row=shelf_end_row,
            cross_aisle_row=cross_aisle_row,
            cross_aisle_width=cross_aisle_width,
            cross_aisle_enabled=cross_aisle_enabled,
            replenish_delay=replenish_delay,
        )
    except ValueError:
        return None

    num_items   = len(grid.get_all_item_positions())
    min_storage = 240   # minimum item cells (legacy: JOBS_MEAN * NUM_AGENTS)
    if num_items < min_storage:
        return None

    total_distance  = 0.0
    total_completed = 0.0
    total_arrived   = 0.0
    total_slow_ticks = 0
    total_walk_ticks = 0

    for _ in range(eval_runs):
        grid = Grid(
            rows=GRID_ROWS, cols=GRID_COLS,
            aisle_width=aisle_width,
            centre_aisle_width=centre_aisle_width,
            depot_row=depot_row,
            depot_cols=active_depot_cols,
            shelf_start_row=shelf_start_row,
            shelf_end_row=shelf_end_row,
            cross_aisle_row=cross_aisle_row,
            cross_aisle_width=cross_aisle_width,
            cross_aisle_enabled=cross_aisle_enabled,
            replenish_delay=replenish_delay,
        )

        # JupedSimAgent replaces Agent + pathfinder + resolve_conflicts.
        # JuPedSim handles physical movement and collision avoidance internally.
        try:
            agent = JupedSimAgent(NUM_AGENTS, grid)
        except RuntimeError:
            # JuPedSim geometry constraint violation (e.g. agents spawned too
            # close to a wall for the given depot position). Treat as invalid.
            return None

        for tick in range(TRAINING_TICKS):
            # Scale tick onto the full 8-hour demand curve so the compressed
            # simulation mirrors the complete NHPP shape (morning peak,
            # steady state, end-of-shift peak) within TRAINING_TICKS.
            if nhpp_arrival(int(tick * TIME_SCALE)):
                agent.add_job()
            agent.step()          # JuPedSim physics + state machine
            grid.tick_replenishment()

        if DEBUG:
            print(f"Jobs arrived: {agent.total_orders}, "
                  f"Completed: {sum(agent.orders_completed)}, "
                  f"Queue remaining: {agent.job_queue}")

        raw = agent.collect_raw()
        total_distance   += raw["total_distance"]
        total_completed  += raw["jobs_completed"]
        total_arrived    += raw["jobs_arrived"]
        total_slow_ticks += sum(agent._slow_ticks)
        total_walk_ticks += sum(agent._walk_ticks)

    avg_distance  = total_distance  / eval_runs
    avg_completed = total_completed / eval_runs

    if avg_completed <= 0:
        return None

    training_hours  = TRAINING_TICKS / 3600.0        # = 1.0 for 3600 ticks
    picks_per_hour  = avg_completed / training_hours  # rate — comparable across runs
    dist_per_agent  = avg_distance / NUM_AGENTS       # lower than 8hr run (~÷8)
    # Congestion: fraction of walking ticks where agent speed dropped below
    # 50% free speed AND another agent was within 0.9m (CONGESTION_RADIUS).
    # Both conditions required — excludes natural waypoint deceleration.
    congestion_rate = (total_slow_ticks / total_walk_ticks
                       if total_walk_ticks > 0 else 0.0)

    avg_arrived = total_arrived / eval_runs

    return {
        "picks_per_hour":  picks_per_hour,
        "dist_per_agent":  dist_per_agent,
        "congestion_rate": congestion_rate,
        "avg_distance":    avg_distance,
        "avg_orders":      avg_completed,
        "jobs_arrived":    avg_arrived,
        "jobs_completed":  avg_completed,
        "avg_conflicts":   0,           # no longer applicable (JuPedSim-based)
        "num_items":       num_items,
    }


BOUNDS_FILE = "kpi_bounds.json"


def calibrate_kpi_bounds(num_samples=200, eval_runs=1, force=False):
    """
    Run random layout configurations to discover the realistic min/max
    range for each KPI. These bounds are used to normalise KPIs to 0-1.

    Results are saved to kpi_bounds.json and reloaded on future runs so
    all training runs use identical normalisation bounds.

    Pass force=True (or --recalibrate flag) to ignore the saved file and
    run fresh calibration.
    """
    import json
    import os

    # Load saved bounds if they exist and force=False
    if not force and os.path.exists(BOUNDS_FILE):
        with open(BOUNDS_FILE, "r") as f:
            bounds = json.load(f)
        # Convert lists back to tuples for consistency
        bounds["picks_per_hour"]  = tuple(bounds["picks_per_hour"])
        bounds["dist_per_agent"]  = tuple(bounds["dist_per_agent"])
        bounds["congestion_rate"] = tuple(bounds["congestion_rate"])
        print(f"\n  Loaded saved KPI bounds from {BOUNDS_FILE}")
        print(f"    Picks/hr      : {bounds['picks_per_hour'][0]:.1f} — "
              f"{bounds['picks_per_hour'][1]:.1f}")
        print(f"    Dist/agent    : {bounds['dist_per_agent'][0]:.1f} — "
              f"{bounds['dist_per_agent'][1]:.1f}")
        print(f"    Congestion    : {bounds['congestion_rate'][0]:.4f} — "
              f"{bounds['congestion_rate'][1]:.4f}")
        print(f"    Max storage   : {bounds['max_storage']} items")
        print(f"  (Use --recalibrate to regenerate)")
        return bounds

    print(f"\n  Calibrating KPI bounds ({num_samples} random layouts)...")
    print(f"  Sampling space (matches compress.py --calibrate):")
    print(f"    aisle_width  : 2-3      centre_aisle : 1-5")
    print(f"    shelf_start  : 2-8      shelf_end    : {GRID_ROWS - 7}-{GRID_ROWS - 2}")
    print(f"    depot_count  : 1-4      depot_col    : 2-{GRID_COLS - 3} (per depot)")
    print(f"    cross_aisle  : on/off   cross_row    : shelf_start+1 to shelf_end-1 (if on)")

    picks_values      = []
    distance_values   = []
    congestion_values = []
    storage_values    = []

    for i in range(num_samples):
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
            eval_runs=eval_runs,
            depot_cols=depot_cols,
            depot_count=depot_count,
            depot_row=0,
            cross_aisle_enabled=bool(cross_on),
        )

        if result is not None:
            picks_values.append(result["picks_per_hour"])
            distance_values.append(result["dist_per_agent"])
            congestion_values.append(result["congestion_rate"])
            storage_values.append(result["num_items"])

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{num_samples} layouts sampled "
                  f"({len(picks_values)} valid)")

    if len(picks_values) < 5:
        print("  WARNING: Too few valid layouts found. Using fallback bounds.")
        return {
            "picks_per_hour":  (0.0, LAMBDA_BASE * SHIFT_TICKS / 8.0),
            "dist_per_agent":  (0.0, 25000.0),
            "congestion_rate": (0.0, 1.0),
            "max_storage":     440,
        }

    # ── KPI bounds: same formulas as compress.py --calibrate ─────────────────
    # picks/hr   : ±5% buffer on observed range, clipped above by physics
    #              ceiling λ × TRAINING_TICKS (= total demand per hour, can't
    #              complete more than arrived); floor clipped to 0.
    # dist/agent : ±5% buffer on observed range; floor clipped to 0.
    # congestion : ±5% buffer on observed range, clipped to [0, 1].
    PHYSICS_PICKS_MAX = LAMBDA_BASE * SHIFT_TICKS / 8.0   # = λ × 3600 (jobs/hr)
    picks_lo = max(0.0,               min(picks_values)      * 0.95)
    picks_hi = min(PHYSICS_PICKS_MAX, max(picks_values)      * 1.05)
    dist_lo  = max(0.0,               min(distance_values)   * 0.95)
    dist_hi  =                        max(distance_values)   * 1.05
    cong_lo  = max(0.0,               min(congestion_values) * 0.95)
    cong_hi  = min(1.0,               max(congestion_values) * 1.05)
    bounds = {
        "picks_per_hour":  (picks_lo, picks_hi),
        "dist_per_agent":  (dist_lo,  dist_hi),
        "congestion_rate": (cong_lo,  cong_hi),
        "max_storage":     int(max(storage_values)),
    }

    print(f"\n  Calibration complete ({len(picks_values)} valid layouts):")
    print(f"    Picks/hr      : {bounds['picks_per_hour'][0]:.1f} — "
          f"{bounds['picks_per_hour'][1]:.1f}")
    print(f"    Dist/agent    : {bounds['dist_per_agent'][0]:.1f} — "
          f"{bounds['dist_per_agent'][1]:.1f}")
    print(f"    Congestion    : {bounds['congestion_rate'][0]:.4f} — "
          f"{bounds['congestion_rate'][1]:.4f}")
    print(f"    Max storage   : {bounds['max_storage']} items "
          f"(highest seen across {len(storage_values)} valid layouts)")

    # Validate bounds against default layout (Sim_V3 defaults)
    default_result = run_single_simulation(
        aisle_width=DEFAULT_AISLE_WIDTH,
        centre_aisle_width=DEFAULT_CENTRE_AISLE,
        depot_col=DEFAULT_DEPOT_COL,
        shelf_start_row=DEFAULT_SHELF_START,
        shelf_end_row=DEFAULT_SHELF_END,
        cross_aisle_row=DEFAULT_CROSS_AISLE_ROW,
        eval_runs=3,
    )
    if default_result:
        p_lo, p_hi = bounds["picks_per_hour"]
        d_lo, d_hi = bounds["dist_per_agent"]
        c_lo, c_hi = bounds["congestion_rate"]
        norm_p = np.clip((default_result["picks_per_hour"]  - p_lo) / (p_hi - p_lo), 0, 1)
        norm_d = np.clip((d_hi - default_result["dist_per_agent"])  / (d_hi - d_lo), 0, 1)
        norm_c = np.clip((c_hi - default_result["congestion_rate"]) / (c_hi - c_lo), 0, 1)
        print(f"\n  Default layout scores under these bounds:")
        print(f"    Norm picks    : {norm_p:.2f}")
        print(f"    Norm distance : {norm_d:.2f}")
        print(f"    Norm cong     : {norm_c:.2f}")
        print(f"    Item cells    : {default_result['num_items']} "
              f"(raw, not used in reward)")
        print(f"  (Scores near 0 or 1 may indicate bounds are too tight/loose)")

    # Save bounds to file for reuse
    save_data = {
        "picks_per_hour":  list(bounds["picks_per_hour"]),
        "dist_per_agent":  list(bounds["dist_per_agent"]),
        "congestion_rate": list(bounds["congestion_rate"]),
        "max_storage":     bounds["max_storage"],
    }
    with open(BOUNDS_FILE, "w") as f:
        import json
        json.dump(save_data, f, indent=2)
    print(f"\n  Bounds saved to {BOUNDS_FILE} — will be reused in future runs.")
    print(f"  Delete {BOUNDS_FILE} or use --recalibrate to regenerate.\n")

    return bounds


# =============================================================================
# THE GYMNASIUM ENVIRONMENT
# =============================================================================

class WarehouseLayoutEnv(gym.Env):
    """
    RL environment where the agent optimises warehouse layout parameters.
    (Unchanged from original — only run_single_simulation was updated above.)
    """

    metadata = {"render_modes": []}

    def __init__(self, kpi_bounds=None, start_mode="default"):
        super().__init__()

        self.start_mode = start_mode  # "default" or "random"

        # Action dimensions: [aisle_w, centre, depot_count,
        #                     shelf_start, shelf_end, cross_aisle_row, cross_aisle_on]
        # Dims 0-5: 3 choices (0→-1, 1→stay, 2→+1)
        # Dim 6: binary (0=off, 1=on, direct assignment)
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 2])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.aisle_width_range        = (2, 3)
        self.centre_aisle_width_range = (1, 5)
        self.shelf_start_row_range    = (2, 8)
        self.shelf_end_row_range      = (GRID_ROWS - 7, GRID_ROWS - 2)
        self.cross_aisle_row_range    = (1, GRID_ROWS - 2)
        self.depot_count_range        = (1, 4)

        if kpi_bounds is not None:
            self.kpi_bounds = kpi_bounds
        else:
            self.kpi_bounds = {
                "picks_per_hour":  (0.0, LAMBDA_BASE * SHIFT_TICKS / 8.0),
                "dist_per_agent":  (0.0, 25000.0),
                "congestion_rate": (0.0, 1.0),
            }

        self.steps_taken    = 0
        self.episode_count  = 0
        self.episode_log    = []

        self.aisle_width        = DEFAULT_AISLE_WIDTH
        self.centre_aisle_width = DEFAULT_CENTRE_AISLE
        self.depot_count        = DEFAULT_DEPOT_COUNT
        self.shelf_start_row    = DEFAULT_SHELF_START
        self.shelf_end_row      = DEFAULT_SHELF_END
        self.cross_aisle_row    = DEFAULT_CROSS_AISLE_ROW
        self.cross_aisle_on     = DEFAULT_CROSS_AISLE_ON

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.start_mode == "random":
            self.aisle_width        = random.randint(*self.aisle_width_range)
            self.centre_aisle_width = random.randint(*self.centre_aisle_width_range)
            self.depot_count        = random.randint(*self.depot_count_range)
            self.shelf_start_row    = random.randint(*self.shelf_start_row_range)
            self.shelf_end_row      = random.randint(*self.shelf_end_row_range)
            self.cross_aisle_on     = random.randint(0, 1)
            self.cross_aisle_row    = random.randint(self.shelf_start_row,
                                                     self.shelf_end_row)
            if self.shelf_start_row >= self.shelf_end_row:
                self.shelf_end_row = self.shelf_start_row + 1
            self.cross_aisle_row = max(self.shelf_start_row,
                                       min(self.shelf_end_row, self.cross_aisle_row))
        else:  # "default"
            self.aisle_width        = DEFAULT_AISLE_WIDTH
            self.centre_aisle_width = DEFAULT_CENTRE_AISLE
            self.depot_count        = DEFAULT_DEPOT_COUNT
            self.shelf_start_row    = DEFAULT_SHELF_START
            self.shelf_end_row      = DEFAULT_SHELF_END
            self.cross_aisle_row    = DEFAULT_CROSS_AISLE_ROW
            self.cross_aisle_on     = DEFAULT_CROSS_AISLE_ON

        self.steps_taken = 0

        return self._get_observation(), {}

    def step(self, action):
        self._apply_action(action)
        self.steps_taken += 1

        terminated = (self.steps_taken >= MAX_STEPS_PER_EPISODE)

        if terminated:
            reward, info = self._evaluate_layout()
        else:
            reward = 0.0
            info   = {}

        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        def norm(value, lo, hi):
            if hi == lo:
                return 0.0
            return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

        return np.array([
            norm(self.aisle_width,        *self.aisle_width_range),
            norm(self.centre_aisle_width, *self.centre_aisle_width_range),
            norm(self.depot_count,        *self.depot_count_range),
            norm(self.shelf_start_row,    *self.shelf_start_row_range),
            norm(self.shelf_end_row,      *self.shelf_end_row_range),
            norm(self.cross_aisle_row,    *self.cross_aisle_row_range),
            float(self.cross_aisle_on),
        ], dtype=np.float32)

    def _apply_action(self, action):
        # Dims 0-5: 3-choice (0→-1, 1→stay, 2→+1)
        d = [int(a) - 1 for a in action[:6]]

        self.aisle_width = int(np.clip(
            self.aisle_width + d[0],
            *self.aisle_width_range))

        self.centre_aisle_width = int(np.clip(
            self.centre_aisle_width + d[1] * 2,
            *self.centre_aisle_width_range))

        # Dim 2: depot_count ±1 — positions derived automatically via compute_depot_cols()
        self.depot_count = int(np.clip(
            self.depot_count + d[2],
            *self.depot_count_range))

        self.shelf_start_row = int(np.clip(
            self.shelf_start_row + d[3],
            *self.shelf_start_row_range))

        self.shelf_end_row = int(np.clip(
            self.shelf_end_row + d[4],
            *self.shelf_end_row_range))

        self.cross_aisle_row = self.cross_aisle_row + d[5]

        # Dim 6: binary cross_aisle_on (0 or 1 direct assignment)
        self.cross_aisle_on = int(action[6])

        # Constraints
        if self.shelf_start_row >= self.shelf_end_row:
            self.shelf_end_row = self.shelf_start_row + 1

        self.cross_aisle_row = max(self.shelf_start_row,
                                   min(self.shelf_end_row, self.cross_aisle_row))

    def _evaluate_layout(self):
        depot_cols = compute_depot_cols(self.depot_count)
        result = run_single_simulation(
            aisle_width=self.aisle_width,
            centre_aisle_width=self.centre_aisle_width,
            depot_col=depot_cols[0],
            shelf_start_row=self.shelf_start_row,
            shelf_end_row=self.shelf_end_row,
            cross_aisle_row=self.cross_aisle_row,
            eval_runs=EVAL_RUNS,
            depot_cols=depot_cols,
            depot_count=self.depot_count,
            depot_row=0,
            cross_aisle_enabled=bool(self.cross_aisle_on),
        )

        if result is None:
            return -1.0, {"error": "invalid_layout"}

        def norm_higher_better(value, lo, hi):
            if hi <= lo:
                return 0.5
            return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

        def norm_lower_better(value, lo, hi):
            if hi <= lo:
                return 0.5
            return float(np.clip((hi - value) / (hi - lo), 0.0, 1.0))

        picks_lo, picks_hi = self.kpi_bounds["picks_per_hour"]
        dist_lo,  dist_hi  = self.kpi_bounds["dist_per_agent"]
        cong_lo,  cong_hi  = self.kpi_bounds["congestion_rate"]

        P = norm_higher_better(result["picks_per_hour"],  picks_lo, picks_hi)
        D = norm_lower_better(result["dist_per_agent"],   dist_lo,  dist_hi)
        C = norm_lower_better(result["congestion_rate"],  cong_lo,  cong_hi)

        reward = W_PICKS * P + W_DISTANCE * D + W_CONGESTION * C

        info = {
            "reward":              reward,
            "picks_per_hour":      result["picks_per_hour"],
            "dist_per_agent":      result["dist_per_agent"],
            "congestion_rate":     result["congestion_rate"],
            "num_items":           result["num_items"],
            "norm_picks":          P,
            "norm_distance":       D,
            "norm_congestion":     C,
            "aisle_width":         self.aisle_width,
            "centre_aisle_width":  self.centre_aisle_width,
            "depot_count":         self.depot_count,
            "depot_cols":          str(depot_cols),
            "shelf_start_row":     self.shelf_start_row,
            "shelf_end_row":       self.shelf_end_row,
            "cross_aisle_row":     self.cross_aisle_row,
            "cross_aisle_on":      self.cross_aisle_on,
        }

        self.episode_count += 1
        self.episode_log.append({
            "episode":             self.episode_count,
            "reward":              reward,
            "picks_per_hour":      result["picks_per_hour"],
            "dist_per_agent":      result["dist_per_agent"],
            "congestion_rate":     result["congestion_rate"],
            "num_items":           result["num_items"],
            "norm_picks":          P,
            "norm_distance":       D,
            "norm_congestion":     C,
            "lambda_base":         LAMBDA_BASE,
            "beta":                DEMAND_BETA,
            "jobs_arrived":        result["jobs_arrived"],
            "jobs_completed":      result["jobs_completed"],
            "aisle_width":         self.aisle_width,
            "centre_aisle_width":  self.centre_aisle_width,
            "depot_count":         self.depot_count,
            "depot_cols":          str(depot_cols),
            "shelf_start_row":     self.shelf_start_row,
            "shelf_end_row":       self.shelf_end_row,
            "cross_aisle_row":     self.cross_aisle_row,
            "cross_aisle_on":      self.cross_aisle_on,
        })

        return reward, info


# =============================================================================
# EVAL ENVIRONMENT — always resets to default layout (used by EvalCallback)
# =============================================================================

class EvalWarehouseEnv(WarehouseLayoutEnv):
    """
    Identical to WarehouseLayoutEnv except reset() always returns the
    default layout instead of a random one.

    Used by EvalCallback to produce a clean, consistent evaluation curve
    during training — every evaluation starts from the same default layout
    so results are directly comparable across episodes.
    """

    def reset(self, seed=None, options=None):
        super(WarehouseLayoutEnv, self).reset(seed=seed)

        self.aisle_width        = DEFAULT_AISLE_WIDTH
        self.centre_aisle_width = DEFAULT_CENTRE_AISLE
        self.depot_count        = DEFAULT_DEPOT_COUNT
        self.shelf_start_row    = DEFAULT_SHELF_START
        self.shelf_end_row      = DEFAULT_SHELF_END
        self.cross_aisle_row    = DEFAULT_CROSS_AISLE_ROW
        self.cross_aisle_on     = DEFAULT_CROSS_AISLE_ON
        self.steps_taken        = 0

        return self._get_observation(), {}


# =============================================================================
# TRAINING SCRIPT
# =============================================================================

def main():
    # Globals overridden by --agents / --lambda-base below. Must be declared
    # at the top of main() before any read, otherwise Python raises
    # "name used prior to global declaration".
    global NUM_AGENTS, LAMBDA_BASE, EVAL_RUNS

    parser = argparse.ArgumentParser(
        description="Train RL agent for warehouse layout optimisation")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="Total training timesteps (default 10000)")
    parser.add_argument("--eval-runs", type=int, default=5,
                        help="Simulation runs per layout evaluation (default 5)")
    parser.add_argument("--calibration-runs", type=int, default=100,
                        help="Random layouts for KPI calibration (default 100)")
    parser.add_argument("--start-mode", type=str, default="default",
                        choices=["default", "random"],
                        help="Episode start layout: 'default' (fixed) or 'random' (default: default)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Ignore saved kpi_bounds.json and run fresh calibration")
    parser.add_argument("--save-results", action="store_true",
                        help="After training, run a full experiment with the RL-recommended "
                             "layout and save it to results/ for the dashboard")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Name for the saved dashboard run (default: RL_optimised_<timestamp>)")
    parser.add_argument("--save-runs", type=int, default=10,
                        help="Number of simulation runs for the saved RL result (default 10)")
    parser.add_argument("--agents", type=int, default=NUM_AGENTS,
                        help=f"Number of pickers in the warehouse "
                             f"(default {NUM_AGENTS}) — affects both calibration and training")
    parser.add_argument("--lambda-base", type=float, default=None,
                        help="NHPP base arrival rate per tick (overrides agent.py default) "
                             "— affects both calibration and training")
    parser.add_argument("--save-run", action="store_true",
                        help="At end of training, copy training_log.csv, best_model.zip, "
                             "warehouse_layout_ppo_v3.zip, kpi_bounds.json, evaluations.npz "
                             "and a config.txt (with CLI args) into runs/<timestamp>_<name>/ "
                             "so the run can be revisited later")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional suffix for the run folder name "
                             "(e.g. --run-name agents4_lambda06_10k)")
    args = parser.parse_args()

    # Apply --agents / --lambda-base overrides BEFORE calibration so both
    # calibration and training use the chosen values — once kpi_bounds.json
    # is written, those bounds are locked to whatever NUM_AGENTS/LAMBDA_BASE
    # produced them.
    NUM_AGENTS = args.agents
    EVAL_RUNS  = args.eval_runs
    if args.lambda_base is not None:
        import agent as _agent_module
        _agent_module.LAMBDA_BASE = args.lambda_base   # drives nhpp_arrival()
        LAMBDA_BASE = args.lambda_base                  # picks/hr ceiling here

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    print("=" * 60)
    print("  WAREHOUSE LAYOUT RL")
    print("=" * 60)
    import agent as _agent_module
    print(f"  Grid size         : {GRID_ROWS} x {GRID_COLS}")
    print(f"  Agents            : {NUM_AGENTS}   (--agents)")
    print(f"  Eval runs         : {EVAL_RUNS} (per layout)")
    print(f"  Actions/episode   : {MAX_STEPS_PER_EPISODE}")
    print(f"  Timesteps         : {args.timesteps}")
    print(f"  Calibration runs  : {args.calibration_runs}")
    print(f"  NHPP demand       : λ_base={_agent_module.LAMBDA_BASE}  (--lambda-base), "
          f"β={DEMAND_BETA}, shift={SHIFT_TICKS} ticks (8 hr)")
    print(f"  Storage minimum   : 280 item cells")
    print(f"  Reward weights    : picks={W_PICKS:.2f}, dist={W_DISTANCE:.2f}, "
          f"cong={W_CONGESTION:.2f}")
    print(f"  Start mode        : {args.start_mode}")
    print("=" * 60)

    print("\n[1/5] Calibrating KPI normalisation bounds...")
    kpi_bounds = calibrate_kpi_bounds(
        num_samples=args.calibration_runs, eval_runs=1,
        force=args.recalibrate)

    print("\n[2/5] Creating environment...")
    env      = WarehouseLayoutEnv(kpi_bounds=kpi_bounds, start_mode=args.start_mode)
    eval_env = EvalWarehouseEnv(kpi_bounds=kpi_bounds)

    print("[3/5] Validating environment...")
    check_env(env, warn=True)
    print("  Passed.\n")

    print("[4/5] Training PPO agent...")
    print(f"  (Each episode runs {TRAINING_TICKS}-tick compressed simulation, x{TIME_SCALE:.0f} time scale)\n")

    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=".",
        log_path=".",
        eval_freq=MAX_STEPS_PER_EPISODE * 10,   # evaluate every 10 episodes
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=100,
        batch_size=50,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.001,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    model.save("warehouse_layout_ppo_v3")
    print("\n  Model saved to warehouse_layout_ppo_v3.zip")
    print("  Best model saved to best_model.zip (via EvalCallback)")

    # Load the best model found during training for the convergence test
    import os
    best_model_path = "best_model.zip"
    if os.path.exists(best_model_path):
        model = PPO.load(best_model_path)
        print("  Loaded best_model.zip for convergence test.")
    else:
        print("  best_model.zip not found — using final model.")

    import csv
    log_file = "training_log.csv"
    if env.episode_log:
        keys = env.episode_log[0].keys()
        with open(log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(env.episode_log)
        print(f"  Training log saved to {log_file} "
              f"({len(env.episode_log)} episodes)")
    print("  Run 'python plot_training.py' to visualise the results.\n")

    print("\n[5/5] Convergence test from default layout...")
    print("=" * 60)

    # Reset then immediately override with the default layout
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

    converged_at = None
    prev_params = {
        "aisle_width":        env.aisle_width,
        "centre_aisle_width": env.centre_aisle_width,
        "depot_count":        env.depot_count,
        "shelf_start_row":    env.shelf_start_row,
        "shelf_end_row":      env.shelf_end_row,
        "cross_aisle_row":    env.cross_aisle_row,
        "cross_aisle_on":     env.cross_aisle_on,
    }

    print(f"  Starting layout : aisle_w={env.aisle_width}, "
          f"centre={env.centre_aisle_width}, "
          f"depots={env.depot_count} {compute_depot_cols(env.depot_count)}, "
          f"shelf={env.shelf_start_row}-{env.shelf_end_row}, "
          f"cross_aisle={'on@'+str(env.cross_aisle_row) if env.cross_aisle_on else 'off'}")
    print()

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
              f"[changed: {', '.join(changed) if changed else 'none'}]")

        if not changed:
            print(f"  Converged.")
            converged_at = step
            break

        prev_params = curr_params
    else:
        print("  Reached max 50 steps without converging.")

    # Local norm helpers for reward display
    def norm_higher_better(value, lo, hi):
        if hi <= lo:
            return 0.5
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def norm_lower_better(value, lo, hi):
        if hi <= lo:
            return 0.5
        return float(np.clip((hi - value) / (hi - lo), 0.0, 1.0))

    picks_lo, picks_hi = env.kpi_bounds["picks_per_hour"]
    dist_lo,  dist_hi  = env.kpi_bounds["dist_per_agent"]
    cong_lo,  cong_hi  = env.kpi_bounds["congestion_rate"]

    # Evaluate the final layout the agent settled on
    print()
    print("  Evaluating final layout...")
    _final_depot_cols = compute_depot_cols(env.depot_count)
    final_result = run_single_simulation(
        aisle_width=env.aisle_width,
        centre_aisle_width=env.centre_aisle_width,
        depot_col=_final_depot_cols[0],
        shelf_start_row=env.shelf_start_row,
        shelf_end_row=env.shelf_end_row,
        cross_aisle_row=env.cross_aisle_row,
        eval_runs=EVAL_RUNS,
        depot_cols=_final_depot_cols,
        depot_count=env.depot_count,
        depot_row=0,
        cross_aisle_enabled=bool(env.cross_aisle_on),
    )

    print()
    rl_reward = None
    if final_result is None:
        print("  ERROR: Final layout is invalid.")
    else:
        steps_msg = (f"after {converged_at} step(s)"
                     if converged_at is not None
                     else "after 50 steps (no convergence)")
        P = norm_higher_better(final_result['picks_per_hour'], picks_lo, picks_hi)
        D = norm_lower_better(final_result['dist_per_agent'],  dist_lo,  dist_hi)
        C = norm_lower_better(final_result['congestion_rate'], cong_lo,  cong_hi)
        rl_reward = W_PICKS*P + W_DISTANCE*D + W_CONGESTION*C
        print(f"  RL-RECOMMENDED LAYOUT ({steps_msg}):")
        print(f"    aisle_w={env.aisle_width}, centre={env.centre_aisle_width}, "
              f"depots={env.depot_count} {_final_depot_cols}, "
              f"shelf={env.shelf_start_row}-{env.shelf_end_row}, "
              f"cross={'on@'+str(env.cross_aisle_row) if env.cross_aisle_on else 'off'}")
        print(f"    Picks/hr    : {final_result['picks_per_hour']:.2f}  (norm: {P:.2f})")
        print(f"    Dist/agent  : {final_result['dist_per_agent']:.1f} (norm: {D:.2f})")
        print(f"    Congestion  : {final_result['congestion_rate']:.4f}  (norm: {C:.2f})")
        print(f"    Item cells  : {final_result['num_items']} (raw, not in reward)")
        print(f"    Reward      : {rl_reward:.3f}")

    print("\n" + "-" * 60)
    print("  BASELINE (default layout):")

    baseline = run_single_simulation(
        aisle_width=DEFAULT_AISLE_WIDTH,
        centre_aisle_width=DEFAULT_CENTRE_AISLE,
        depot_col=DEFAULT_DEPOT_COL,
        shelf_start_row=DEFAULT_SHELF_START,
        shelf_end_row=DEFAULT_SHELF_END,
        cross_aisle_row=DEFAULT_CROSS_AISLE_ROW,
        eval_runs=EVAL_RUNS,
    )

    baseline_reward = None
    if baseline is None:
        print("    ERROR: Default layout failed!")
    else:
        P_b = norm_higher_better(baseline['picks_per_hour'], picks_lo, picks_hi)
        D_b = norm_lower_better(baseline['dist_per_agent'],  dist_lo,  dist_hi)
        C_b = norm_lower_better(baseline['congestion_rate'], cong_lo,  cong_hi)
        baseline_reward = W_PICKS*P_b + W_DISTANCE*D_b + W_CONGESTION*C_b
        print(f"    Layout      : aisle_w={DEFAULT_AISLE_WIDTH}, centre={DEFAULT_CENTRE_AISLE}, "
              f"depot_col={DEFAULT_DEPOT_COL}, shelf={DEFAULT_SHELF_START}-{DEFAULT_SHELF_END}, "
              f"cross_aisle_row={DEFAULT_CROSS_AISLE_ROW}")
        print(f"    Picks/hr    : {baseline['picks_per_hour']:.2f}  (norm: {P_b:.2f})")
        print(f"    Dist/agent  : {baseline['dist_per_agent']:.1f} (norm: {D_b:.2f})")
        print(f"    Congestion  : {baseline['congestion_rate']:.4f}  (norm: {C_b:.2f})")
        print(f"    Item cells  : {baseline['num_items']} (raw, not in reward)")
        print(f"    Reward      : {baseline_reward:.3f}")

    print()
    if rl_reward is not None and baseline_reward is not None:
        print(f"  RL layout  reward: {rl_reward:.3f}")
        print(f"  Baseline   reward: {baseline_reward:.3f}")
    print("=" * 60)

    # --save-run: bundle all artifacts (training log, models, bounds, eval
    # log, config) into runs/<timestamp>_<name>/ so the training can be
    # revisited later. Independent of --save-results (which exports the
    # layout for the dashboard).
    if args.save_run:
        import os
        import shutil
        from datetime import datetime as _dt

        ts        = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix    = f"_{args.run_name}" if args.run_name else ""
        folder    = os.path.join("runs", f"{ts}{suffix}")
        os.makedirs(folder, exist_ok=True)

        artifacts = [
            "training_log.csv",
            "best_model.zip",
            "warehouse_layout_ppo_v3.zip",
            "kpi_bounds.json",
            "evaluations.npz",
        ]
        copied  = []
        missing = []
        for f in artifacts:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(folder, f))
                copied.append(f)
            else:
                missing.append(f)

        # Dump the full CLI configuration so the run is self-documenting.
        import agent as _agent_module
        config_path = os.path.join(folder, "config.txt")
        with open(config_path, "w") as cf:
            cf.write(f"# Training run config — {ts}\n")
            cf.write(f"# CLI command used:\n")
            cf.write(f"python rl_env.py")
            for k, v in vars(args).items():
                if v is None or v is False:
                    continue
                cf.write(f" --{k.replace('_', '-')} {v}" if v is not True else f" --{k.replace('_', '-')}")
            cf.write("\n\n")
            cf.write(f"NUM_AGENTS         = {NUM_AGENTS}\n")
            cf.write(f"LAMBDA_BASE        = {_agent_module.LAMBDA_BASE}\n")
            cf.write(f"DEMAND_BETA        = {DEMAND_BETA}\n")
            cf.write(f"GRID_ROWS x COLS   = {GRID_ROWS} x {GRID_COLS}\n")
            cf.write(f"TRAINING_TICKS     = {TRAINING_TICKS}  (TIME_SCALE x{TIME_SCALE:.0f})\n")
            cf.write(f"EVAL_RUNS          = {EVAL_RUNS}\n")
            cf.write(f"MAX_STEPS_PER_EP   = {MAX_STEPS_PER_EPISODE}\n")
            cf.write(f"Reward weights     = picks {W_PICKS}, dist {W_DISTANCE}, "
                     f"cong {W_CONGESTION}\n")
            cf.write(f"Start mode         = {args.start_mode}\n")
            cf.write(f"Timesteps          = {args.timesteps}\n")
            cf.write(f"Calibration runs   = {args.calibration_runs}\n")

        print(f"\n  Run bundle saved → {folder}/")
        print(f"    Copied : {', '.join(copied) if copied else '(none)'}")
        if missing:
            print(f"    Missing: {', '.join(missing)}  (these files weren't produced this run)")
        print(f"    Config : {config_path}")
        print(f"  To revisit: cp {folder}/* . && python plot_training.py")
        print("=" * 60)

    # --save-results: run a full experiment with the RL-recommended layout
    # and save it to results/ so it appears in the dashboard dropdown.
    if args.save_results and final_result is not None:
        import subprocess, sys, os
        from datetime import datetime as _dt

        run_name = args.save_name or (
            "RL_optimised_" + _dt.now().strftime("%Y-%m-%d_%H-%M"))

        print(f"\n  Saving RL-optimised run to results/{run_name} ...")
        print(f"  (Running {args.save_runs} full simulation runs — this may take a moment)")

        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "main.py"),
            "--experiment",
            "--save",
            "--name",            run_name,
            "--runs",            str(args.save_runs),
            "--agents",          str(NUM_AGENTS),
            "--rows",            str(GRID_ROWS),
            "--cols",            str(GRID_COLS),
            "--aisle-width",     str(env.aisle_width),
            "--centre-aisle",    str(env.centre_aisle_width),
            "--depot-col",       str(_final_depot_cols[0]),
            "--shelf-start",     str(env.shelf_start_row),
            "--shelf-end",       str(env.shelf_end_row),
            "--cross-aisle-row", str(env.cross_aisle_row),
        ]
        # NOTE: main.py currently only supports a single --depot-col.
        # Multi-depot runs from --save-results will use the primary depot only.
        # Add --depot-cols support to main.py if full multi-depot saving is needed.

        result_proc = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        if result_proc.returncode == 0:
            print(f"\n  Done. Open the dashboard and select '{run_name}'.")
        else:
            print(f"\n  WARNING: main.py exited with code {result_proc.returncode}.")
        print("=" * 60)


if __name__ == "__main__":
    main()
