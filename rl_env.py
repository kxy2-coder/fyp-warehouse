# =============================================================================
# rl_env.py — Reinforcement Learning Environment for Warehouse Layout (v3)
# =============================================================================
# Wraps the warehouse simulation into a Gymnasium environment for SB3.
#
# CHANGES FROM v2:
#   - Reward: normalised 3-KPI weighted sum (picks/hr, distance, congestion)
#     instead of raw -avg_distance.
#   - Replaced depot_row with depot_col (lateral position matters more).
#   - Centre aisle width: odd constraint removed, step size now +/-1.
#   - Added calibration phase: runs random layouts before training to
#     discover KPI min/max bounds for normalisation.
#   - All 3 KPIs calculated: picks_per_hour, dist_per_agent, congestion_rate.
#
# STATE (observation) — 6 normalised values:
#   [aisle_width, centre_aisle_width, depot_col, shelf_start_row,
#    shelf_end_row, cross_aisle_row]
#
# ACTIONS — 12 discrete actions (+/-1 nudges, except depot_col which is +/-2):
#   0: aisle_width + 1          1: aisle_width - 1
#   2: centre_aisle_width + 1   3: centre_aisle_width - 1
#   4: depot_col + 2            5: depot_col - 2
#   6: shelf_start_row + 1      7: shelf_start_row - 1
#   8: shelf_end_row + 1        9: shelf_end_row - 1
#  10: cross_aisle_row + 1     11: cross_aisle_row - 1
#
# REWARD:
#   reward = (1/3) * P + (1/3) * D + (1/3) * C
#   where P, D, C are normalised picks/hr, distance, congestion (all 0-1).
#   -1.0 if layout is invalid or has insufficient storage.
#
# USAGE:
#   python rl_env.py                          # default training
#   python rl_env.py --timesteps 20000        # longer training
#   python rl_env.py --eval-runs 5            # more accurate evaluation
#   python rl_env.py --calibration-runs 50    # more calibration samples
# =============================================================================

import math
import random
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from grid    import Grid
from agent   import (Agent, nhpp_arrival, get_multiplier, SHIFT_TICKS,
                     LAMBDA_BASE, DEMAND_BETA, BLOCKED_WAIT_TICKS,
                     STATE_TO_DEPOT)
from metrics import MetricsTracker


def resolve_conflicts(agent):
    """
    Reactive conflict resolution — mirrors main.py.
    Called AFTER agent.step(): detects agents sharing the same cell and
    blocks the lower-priority one for BLOCKED_WAIT_TICKS ticks.
    """
    active = [i for i in range(agent.n) if not agent.is_done(i)]

    def priority(i):
        state_score = 1 if agent.state[i] == STATE_TO_DEPOT else 0
        return (state_score, random.random())

    pos_map = {}
    for i in active:
        pos = (int(agent.pos_row[i]), int(agent.pos_col[i]))
        pos_map.setdefault(pos, []).append(i)

    for pos, occupants in pos_map.items():
        if len(occupants) < 2:
            continue
        if pos == tuple(agent.grid.depot):
            continue

        winner = max(occupants, key=priority)
        for loser in occupants:
            if loser == winner:
                continue
            agent.blocked_ticks[loser]  = BLOCKED_WAIT_TICKS
            agent.blocked_events[loser] += 1


# =============================================================================
# CONFIGURATION
# =============================================================================

GRID_ROWS  = 25
GRID_COLS  = 35
NUM_AGENTS = 4

EVAL_RUNS = 5
MAX_STEPS_PER_EPISODE = 5
REPLENISH_DELAY = 100

W_PICKS      = 1.0 / 3.0
W_DISTANCE   = 1.0 / 3.0
W_CONGESTION = 1.0 / 3.0

# Sensitivity analysis ranges (not used in training loop — for analysis scripts)
LAMBDA_BASE_RANGE = [0.030, 0.055, 0.078, 0.100, 0.130]
BETA_RANGE        = [0.0, 0.2, 0.3, 0.4]

DEBUG = False   # set True to print per-run job arrival diagnostics


# =============================================================================
# CALIBRATION — discover KPI min/max bounds from random layouts
# =============================================================================

def run_single_simulation(aisle_width, centre_aisle_width, depot_col,
                          shelf_start_row, shelf_end_row,
                          cross_aisle_row=None, eval_runs=EVAL_RUNS,
                          replenish_delay=REPLENISH_DELAY):
    """
    Build a warehouse layout with the given parameters and run the simulation
    eval_runs times. Returns averaged KPIs, or None if the layout is invalid.
    """
    try:
        grid = Grid(
            rows=GRID_ROWS, cols=GRID_COLS,
            aisle_width=aisle_width,
            centre_aisle_width=centre_aisle_width,
            depot_col=depot_col,
            shelf_start_row=shelf_start_row,
            shelf_end_row=shelf_end_row,
            cross_aisle_row=cross_aisle_row,
            replenish_delay=replenish_delay,
        )
    except ValueError:
        return None

    num_items   = len(grid.get_all_item_positions())
    min_storage = 280   # minimum item cells (legacy: JOBS_MEAN * NUM_AGENTS)
    if num_items < min_storage:
        return None

    total_distance  = 0.0
    total_completed = 0.0
    total_arrived   = 0.0
    total_worktime  = 0.0
    total_conflicts = 0.0

    for _ in range(eval_runs):
        grid = Grid(
            rows=GRID_ROWS, cols=GRID_COLS,
            aisle_width=aisle_width,
            centre_aisle_width=centre_aisle_width,
            depot_col=depot_col,
            shelf_start_row=shelf_start_row,
            shelf_end_row=shelf_end_row,
            cross_aisle_row=cross_aisle_row,
            replenish_delay=replenish_delay,
        )

        # ONE Agent object manages all NUM_AGENTS workers simultaneously
        agent   = Agent(NUM_AGENTS, grid, quota=0)
        metrics = MetricsTracker()

        for tick in range(SHIFT_TICKS):
            if nhpp_arrival(tick):
                agent.add_job()
            agent.step()
            resolve_conflicts(agent)
            grid.tick_replenishment()
            metrics.update(agent, grid.depot)

        if DEBUG:
            print(f"Jobs arrived: {agent.total_orders}, "
                  f"Completed: {agent.orders_completed.sum()}, "
                  f"Queue remaining: {agent.job_queue}")

        raw = metrics.collect_raw(agent)
        total_distance  += raw["total_distance"]
        total_completed += raw["jobs_completed"]
        total_arrived   += raw["jobs_arrived"]
        total_worktime  += raw["total_work_time"]
        total_conflicts += raw["cell_conflicts"]

    avg_distance  = total_distance  / eval_runs
    avg_completed = total_completed / eval_runs
    avg_arrived   = total_arrived   / eval_runs
    avg_worktime  = total_worktime  / eval_runs
    avg_conflicts = total_conflicts / eval_runs

    if avg_completed <= 0:
        return None

    picks_per_hour  = avg_completed / 8.0          # fixed 8-hour denominator
    dist_per_agent  = avg_distance / NUM_AGENTS
    congestion_rate = (avg_conflicts * BLOCKED_WAIT_TICKS) / SHIFT_TICKS

    return {
        "picks_per_hour":  picks_per_hour,
        "dist_per_agent":  dist_per_agent,
        "congestion_rate": congestion_rate,
        "avg_distance":    avg_distance,
        "avg_orders":      avg_completed,
        "jobs_arrived":    avg_arrived,
        "jobs_completed":  avg_completed,
        "avg_conflicts":   avg_conflicts,
        "num_items":       num_items,
    }


def calibrate_kpi_bounds(num_samples=50, eval_runs=1):
    """
    Run random layout configurations to discover the realistic min/max
    range for each KPI. These bounds are used to normalise KPIs to 0-1.
    """
    print(f"\n  Calibrating KPI bounds ({num_samples} random layouts)...")

    picks_values      = []
    distance_values   = []
    congestion_values = []

    for i in range(num_samples):
        aw  = random.randint(1, 3)
        caw = random.randint(1, 3)
        dc  = random.randint(2, GRID_COLS - 3)
        ssr = random.randint(1, 6)
        ser = random.randint(GRID_ROWS - 7, GRID_ROWS - 2)

        if ssr >= ser:
            ser = ssr + 1

        result = run_single_simulation(aw, caw, dc, ssr, ser, eval_runs=eval_runs)

        if result is not None:
            picks_values.append(result["picks_per_hour"])
            distance_values.append(result["dist_per_agent"])
            congestion_values.append(result["congestion_rate"])

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{num_samples} layouts sampled "
                  f"({len(picks_values)} valid)")

    if len(picks_values) < 5:
        print("  WARNING: Too few valid layouts found. Using fallback bounds.")
        return {
            "picks_per_hour":  (30.0, 150.0),
            "dist_per_agent":  (100.0, 2000.0),
            "congestion_rate": (0.0, 1.0),
        }

    def bounds_with_buffer(values):
        lo, hi = min(values), max(values)
        margin = (hi - lo) * 0.05
        if margin < 0.001:
            margin = 0.001
        return (lo - margin, hi + margin)

    bounds = {
        "picks_per_hour":  bounds_with_buffer(picks_values),
        "dist_per_agent":  bounds_with_buffer(distance_values),
        "congestion_rate": bounds_with_buffer(congestion_values),
    }

    print(f"\n  Calibration complete ({len(picks_values)} valid layouts):")
    print(f"    Picks/hr      : {bounds['picks_per_hour'][0]:.1f} — "
          f"{bounds['picks_per_hour'][1]:.1f}")
    print(f"    Dist/agent    : {bounds['dist_per_agent'][0]:.1f} — "
          f"{bounds['dist_per_agent'][1]:.1f}")
    print(f"    Congestion    : {bounds['congestion_rate'][0]:.4f} — "
          f"{bounds['congestion_rate'][1]:.4f}")

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

    def __init__(self, kpi_bounds=None):
        super().__init__()

        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.aisle_width_range        = (1, 3)
        self.centre_aisle_width_range = (1, 3)
        self.depot_col_range          = (2, GRID_COLS - 3)
        self.shelf_start_row_range    = (1, 6)
        self.shelf_end_row_range      = (GRID_ROWS - 7, GRID_ROWS - 2)
        self.cross_aisle_row_range    = (1, GRID_ROWS - 2)

        if kpi_bounds is not None:
            self.kpi_bounds = kpi_bounds
        else:
            self.kpi_bounds = {
                "picks_per_hour":  (30.0, 150.0),
                "dist_per_agent":  (100.0, 2000.0),
                "congestion_rate": (0.0, 1.0),
            }

        self.steps_taken    = 0
        self.episode_count  = 0
        self.episode_log    = []

        self.aisle_width        = 2
        self.centre_aisle_width = 3
        self.depot_col          = GRID_COLS // 2
        self.shelf_start_row    = 1
        self.shelf_end_row      = GRID_ROWS - 2
        self.cross_aisle_row    = (1 + GRID_ROWS - 2) // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.aisle_width        = 2
        self.centre_aisle_width = 3
        self.depot_col          = GRID_COLS // 2
        self.shelf_start_row    = 1
        self.shelf_end_row      = GRID_ROWS - 2
        self.cross_aisle_row    = (1 + GRID_ROWS - 2) // 2
        self.steps_taken        = 0

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
            norm(self.depot_col,          *self.depot_col_range),
            norm(self.shelf_start_row,    *self.shelf_start_row_range),
            norm(self.shelf_end_row,      *self.shelf_end_row_range),
            norm(self.cross_aisle_row,    *self.cross_aisle_row_range),
        ], dtype=np.float32)

    def _apply_action(self, action):
        if action == 0:
            self.aisle_width = min(self.aisle_width + 1,
                                   self.aisle_width_range[1])
        elif action == 1:
            self.aisle_width = max(self.aisle_width - 1,
                                   self.aisle_width_range[0])

        elif action == 2:
            self.centre_aisle_width = min(self.centre_aisle_width + 1,
                                          self.centre_aisle_width_range[1])
        elif action == 3:
            self.centre_aisle_width = max(self.centre_aisle_width - 1,
                                          self.centre_aisle_width_range[0])

        elif action == 4:
            self.depot_col = min(self.depot_col + 2,
                                 self.depot_col_range[1])
        elif action == 5:
            self.depot_col = max(self.depot_col - 2,
                                 self.depot_col_range[0])

        elif action == 6:
            self.shelf_start_row = min(self.shelf_start_row + 1,
                                       self.shelf_start_row_range[1])
        elif action == 7:
            self.shelf_start_row = max(self.shelf_start_row - 1,
                                       self.shelf_start_row_range[0])

        elif action == 8:
            self.shelf_end_row = min(self.shelf_end_row + 1,
                                     self.shelf_end_row_range[1])
        elif action == 9:
            self.shelf_end_row = max(self.shelf_end_row - 1,
                                     self.shelf_end_row_range[0])

        elif action == 10:
            self.cross_aisle_row = self.cross_aisle_row + 1
        elif action == 11:
            self.cross_aisle_row = self.cross_aisle_row - 1

        if self.shelf_start_row >= self.shelf_end_row:
            self.shelf_end_row = self.shelf_start_row + 1

        self.cross_aisle_row = max(self.shelf_start_row,
                                   min(self.shelf_end_row, self.cross_aisle_row))

    def _evaluate_layout(self):
        result = run_single_simulation(
            aisle_width=self.aisle_width,
            centre_aisle_width=self.centre_aisle_width,
            depot_col=self.depot_col,
            shelf_start_row=self.shelf_start_row,
            shelf_end_row=self.shelf_end_row,
            cross_aisle_row=self.cross_aisle_row,
            eval_runs=EVAL_RUNS,
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
            "norm_picks":          P,
            "norm_distance":       D,
            "norm_congestion":     C,
            "num_items":           result["num_items"],
            "aisle_width":         self.aisle_width,
            "centre_aisle_width":  self.centre_aisle_width,
            "depot_col":           self.depot_col,
            "shelf_start_row":     self.shelf_start_row,
            "shelf_end_row":       self.shelf_end_row,
            "cross_aisle_row":     self.cross_aisle_row,
        }

        self.episode_count += 1
        self.episode_log.append({
            "episode":             self.episode_count,
            "reward":              reward,
            "picks_per_hour":      result["picks_per_hour"],
            "dist_per_agent":      result["dist_per_agent"],
            "congestion_rate":     result["congestion_rate"],
            "norm_picks":          P,
            "norm_distance":       D,
            "norm_congestion":     C,
            "lambda_base":         LAMBDA_BASE,
            "beta":                DEMAND_BETA,
            "jobs_arrived":        result["jobs_arrived"],
            "jobs_completed":      result["jobs_completed"],
            "aisle_width":         self.aisle_width,
            "centre_aisle_width":  self.centre_aisle_width,
            "depot_col":           self.depot_col,
            "shelf_start_row":     self.shelf_start_row,
            "shelf_end_row":       self.shelf_end_row,
            "cross_aisle_row":     self.cross_aisle_row,
        })

        return reward, info


# =============================================================================
# TRAINING SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for warehouse layout optimisation")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="Total training timesteps (default 10000)")
    parser.add_argument("--eval-runs", type=int, default=5,
                        help="Simulation runs per layout evaluation (default 5)")
    parser.add_argument("--calibration-runs", type=int, default=200,
                        help="Random layouts for KPI calibration (default 200)")
    args = parser.parse_args()

    global EVAL_RUNS
    EVAL_RUNS = args.eval_runs

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    print("=" * 60)
    print("  WAREHOUSE LAYOUT RL TRAINER (v3)")
    print("=" * 60)
    print(f"  Grid size         : {GRID_ROWS} x {GRID_COLS}")
    print(f"  Agents            : {NUM_AGENTS}")
    print(f"  Eval runs         : {EVAL_RUNS} (per layout)")
    print(f"  Actions/episode   : {MAX_STEPS_PER_EPISODE}")
    print(f"  Timesteps         : {args.timesteps}")
    print(f"  Calibration runs  : {args.calibration_runs}")
    print(f"  NHPP demand       : λ_base={LAMBDA_BASE}, β={DEMAND_BETA}, "
          f"shift={SHIFT_TICKS} ticks (8 hr)")
    print(f"  Storage minimum   : 280 item cells")
    print(f"  Reward weights    : picks={W_PICKS:.2f}, "
          f"dist={W_DISTANCE:.2f}, cong={W_CONGESTION:.2f}")
    print("=" * 60)

    print("\n[1/5] Calibrating KPI normalisation bounds...")
    kpi_bounds = calibrate_kpi_bounds(
        num_samples=args.calibration_runs, eval_runs=1)

    print("\n[2/5] Creating environment...")
    env = WarehouseLayoutEnv(kpi_bounds=kpi_bounds)

    print("[3/5] Validating environment...")
    check_env(env, warn=True)
    print("  Passed.\n")

    print("[4/5] Training PPO agent...")
    print("  (Each episode runs your full warehouse simulation)\n")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=64,
        batch_size=32,
        n_epochs=10,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=args.timesteps)

    model.save("warehouse_layout_ppo_v3")
    print("\n  Model saved to warehouse_layout_ppo_v3.zip")

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

    print("\n[5/5] Testing trained agent...")
    print("=" * 60)

    for episode in range(5):
        obs, _ = env.reset()
        done   = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        if "error" in info:
            print(f"\n  Episode {episode + 1}: REJECTED — {info['error']}")
        else:
            print(f"\n  Episode {episode + 1}:")
            print(f"    Layout      : aisle_w={info['aisle_width']}, "
                  f"centre={info['centre_aisle_width']}, "
                  f"depot_col={info['depot_col']}, "
                  f"shelf={info['shelf_start_row']}-{info['shelf_end_row']}, "
                  f"cross_aisle_row={info['cross_aisle_row']}")
            print(f"    Storage     : {info['num_items']} item cells")
            print(f"    Picks/hr    : {info['picks_per_hour']:.2f}  "
                  f"(norm: {info['norm_picks']:.3f})")
            print(f"    Dist/agent  : {info['dist_per_agent']:.1f}  "
                  f"(norm: {info['norm_distance']:.3f})")
            print(f"    Congestion  : {info['congestion_rate']:.4f}  "
                  f"(norm: {info['norm_congestion']:.3f})")
            print(f"    Reward      : {total_reward:.4f}")

    print("\n" + "-" * 60)
    print("  BASELINE (default layout):")

    default_cross = (1 + GRID_ROWS - 2) // 2
    baseline = run_single_simulation(
        aisle_width=2, centre_aisle_width=3,
        depot_col=GRID_COLS // 2,
        shelf_start_row=1, shelf_end_row=GRID_ROWS - 2,
        cross_aisle_row=default_cross,
        eval_runs=EVAL_RUNS,
    )

    if baseline is None:
        print("    ERROR: Default layout failed!")
    else:
        print(f"    Layout      : aisle_w=2, centre=3, "
              f"depot_col={GRID_COLS // 2}, shelf=1-{GRID_ROWS - 2}, "
              f"cross_aisle_row={default_cross}")
        print(f"    Storage     : {baseline['num_items']} item cells")
        print(f"    Picks/hr    : {baseline['picks_per_hour']:.2f}")
        print(f"    Dist/agent  : {baseline['dist_per_agent']:.1f}")
        print(f"    Congestion  : {baseline['congestion_rate']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
