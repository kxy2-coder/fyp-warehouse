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
import itertools

# ── Your existing simulation modules ─────────────────────────────────────────
from grid    import Grid
from agent   import Agent, draw_job_quota, JOBS_MEAN, JOBS_STD, BLOCKED_WAIT_TICKS
from metrics import MetricsTracker


def resolve_right_of_way(agents):
    """
    Resolve movement conflicts between agents.
    Copied from main.py so this file runs without pygame.
    """
    active = [a for a in agents if not a.is_done()]

    def priority(agent):
        state_score = 1 if agent.state == "to_depot" else 0
        id_score    = -agent.agent_id
        return (state_score, id_score)

    for a1, a2 in itertools.combinations(active, 2):
        next1 = a1.peek_next_pos()
        next2 = a2.peek_next_pos()
        if next1 is None or next2 is None:
            continue

        same_dest = (next1 == next2)
        head_on   = (next1 == a2.pos and next2 == a1.pos)
        if not (same_dest or head_on):
            continue

        if priority(a1) >= priority(a2):
            loser = a2
        else:
            loser = a1

        if loser.blocked_ticks_remaining == 0:
            loser.blocked_count += 1
        loser.blocked_ticks_remaining = BLOCKED_WAIT_TICKS


# =============================================================================
# CONFIGURATION
# =============================================================================

GRID_ROWS  = 25
GRID_COLS  = 35
NUM_AGENTS = 4

# How many simulation runs to average per layout evaluation.
EVAL_RUNS = 5

# How many layout tweaks the RL agent gets per episode.
MAX_STEPS_PER_EPISODE = 5

# Safety limit for simulation ticks.
MAX_SIM_TICKS = 50_000

# Reward weights (equal weighting).
W_PICKS      = 1.0 / 3.0
W_DISTANCE   = 1.0 / 3.0
W_CONGESTION = 1.0 / 3.0


# =============================================================================
# CALIBRATION — discover KPI min/max bounds from random layouts
# =============================================================================

def run_single_simulation(aisle_width, centre_aisle_width, depot_col,
                          shelf_start_row, shelf_end_row,
                          cross_aisle_row=None, eval_runs=EVAL_RUNS):
    """
    Build a warehouse layout with the given parameters and run the simulation
    eval_runs times. Returns averaged KPIs, or None if the layout is invalid.

    cross_aisle_row — horizontal corridor row within the shelf zone.
                      None = use Grid default (middle of shelf zone).
    """
    # ── Build and validate ────────────────────────────────────────────────
    try:
        grid = Grid(
            rows=GRID_ROWS, cols=GRID_COLS,
            aisle_width=aisle_width,
            centre_aisle_width=centre_aisle_width,
            depot_col=depot_col,
            shelf_start_row=shelf_start_row,
            shelf_end_row=shelf_end_row,
            cross_aisle_row=cross_aisle_row,
        )
    except ValueError:
        return None

    # Storage constraint
    num_items = len(grid.get_all_item_positions())
    min_storage = JOBS_MEAN * NUM_AGENTS
    if num_items < min_storage:
        return None

    # ── Run simulation ────────────────────────────────────────────────────
    total_distance  = 0.0
    total_orders    = 0.0
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
        )

        quota  = draw_job_quota()
        agents = [Agent(grid, agent_id=i+1, total_orders=quota)
                  for i in range(NUM_AGENTS)]
        metrics = MetricsTracker()

        ticks = 0
        while not all(a.is_done() for a in agents):
            resolve_right_of_way(agents)
            for agent in agents:
                agent.step()
            metrics.update(agents, grid.depot)
            ticks += 1
            if ticks > MAX_SIM_TICKS:
                break

        raw = metrics.collect_raw(agents)
        total_distance  += raw["total_distance"]
        total_orders    += raw["total_orders"]
        total_worktime  += raw["total_work_time"]
        total_conflicts += raw["cell_conflicts"]

    # ── Average and compute KPIs ──────────────────────────────────────────
    avg_distance  = total_distance  / eval_runs
    avg_orders    = total_orders    / eval_runs
    avg_worktime  = total_worktime  / eval_runs
    avg_conflicts = total_conflicts / eval_runs

    if avg_worktime <= 0 or avg_orders <= 0:
        return None

    picks_per_hour  = avg_orders / avg_worktime
    dist_per_agent  = avg_distance / NUM_AGENTS
    congestion_rate = avg_conflicts / avg_orders

    return {
        "picks_per_hour":  picks_per_hour,
        "dist_per_agent":  dist_per_agent,
        "congestion_rate": congestion_rate,
        "avg_distance":    avg_distance,
        "avg_orders":      avg_orders,
        "avg_conflicts":   avg_conflicts,
        "num_items":       num_items,
    }


def calibrate_kpi_bounds(num_samples=50, eval_runs=1):
    """
    Run random layout configurations to discover the realistic min/max
    range for each KPI. These bounds are used to normalise KPIs to 0-1.

    Uses eval_runs=1 per layout during calibration for speed (the bounds
    only need to be approximate).
    """
    print(f"\n  Calibrating KPI bounds ({num_samples} random layouts)...")

    picks_values      = []
    distance_values   = []
    congestion_values = []

    for i in range(num_samples):
        # Random layout parameters within valid ranges
        aw  = random.randint(1, 3)
        caw = random.randint(1, 3)
        dc  = random.randint(2, GRID_COLS - 3)
        ssr = random.randint(1, 6)
        ser = random.randint(GRID_ROWS - 7, GRID_ROWS - 2)

        # Ensure shelf_start < shelf_end
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

    # Use min/max with a small buffer (5%) to avoid edge normalisation issues
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

    STATE (observation) — 6 normalised values:
        [aisle_width, centre_aisle_width, depot_col, shelf_start_row,
         shelf_end_row, cross_aisle_row]

    ACTIONS — 12 discrete actions:
        0: aisle_width + 1          1: aisle_width - 1
        2: centre_aisle_width + 1   3: centre_aisle_width - 1
        4: depot_col + 2            5: depot_col - 2
        6: shelf_start_row + 1      7: shelf_start_row - 1
        8: shelf_end_row + 1        9: shelf_end_row - 1
       10: cross_aisle_row + 1     11: cross_aisle_row - 1

    REWARD:
        Normalised weighted sum of 3 KPIs (equal weights, 1/3 each).
        -1.0 if layout is invalid or has insufficient storage.
    """

    metadata = {"render_modes": []}

    def __init__(self, kpi_bounds=None):
        super().__init__()

        # ── 12 discrete actions ───────────────────────────────────────────
        self.action_space = spaces.Discrete(12)

        # ── 6 normalised observation values ───────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # ── Parameter bounds ──────────────────────────────────────────────
        self.aisle_width_range        = (1, 3)
        self.centre_aisle_width_range = (1, 3)
        self.depot_col_range          = (2, GRID_COLS - 3)   # 2 to 32
        self.shelf_start_row_range    = (1, 6)
        self.shelf_end_row_range      = (GRID_ROWS - 7, GRID_ROWS - 2)  # 18..23
        # Cross-aisle obs range spans the full possible shelf zone for normalisation.
        # Actual clamping is done dynamically in _apply_action().
        self.cross_aisle_row_range    = (1, GRID_ROWS - 2)   # 1..23

        # ── KPI normalisation bounds (from calibration) ───────────────────
        if kpi_bounds is not None:
            self.kpi_bounds = kpi_bounds
        else:
            self.kpi_bounds = {
                "picks_per_hour":  (30.0, 150.0),
                "dist_per_agent":  (100.0, 2000.0),
                "congestion_rate": (0.0, 1.0),
            }

        # ── Episode tracking ──────────────────────────────────────────────
        self.steps_taken    = 0
        self.episode_count  = 0
        self.episode_log    = []   # stores KPIs from every completed episode

        # ── Current layout parameters (defaults) ─────────────────────────
        self.aisle_width        = 2
        self.centre_aisle_width = 2
        self.depot_col          = GRID_COLS // 2   # 17 (centre)
        self.shelf_start_row    = 1
        self.shelf_end_row      = GRID_ROWS - 2    # 23
        self.cross_aisle_row    = (1 + GRID_ROWS - 2) // 2   # middle of shelf zone

    # =====================================================================
    # reset()
    # =====================================================================

    def reset(self, seed=None, options=None):
        """Start a fresh episode with default layout parameters."""
        super().reset(seed=seed)

        self.aisle_width        = 2
        self.centre_aisle_width = 2
        self.depot_col          = GRID_COLS // 2
        self.shelf_start_row    = 1
        self.shelf_end_row      = GRID_ROWS - 2
        self.cross_aisle_row    = (1 + GRID_ROWS - 2) // 2
        self.steps_taken        = 0

        return self._get_observation(), {}

    # =====================================================================
    # step()
    # =====================================================================

    def step(self, action):
        """
        Apply one action (nudge a parameter).
        After MAX_STEPS_PER_EPISODE, run simulation and return reward.
        Steps 1 to MAX-1 return reward = 0 (no simulation yet).
        """
        self._apply_action(action)
        self.steps_taken += 1

        terminated = (self.steps_taken >= MAX_STEPS_PER_EPISODE)

        if terminated:
            reward, info = self._evaluate_layout()
        else:
            reward = 0.0
            info   = {}

        return self._get_observation(), reward, terminated, False, info

    # =====================================================================
    # Private helpers
    # =====================================================================

    def _get_observation(self):
        """Return normalised layout parameters as a numpy array."""

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
        """
        Modify one layout parameter based on the chosen action.
        Values are clamped to their valid ranges.
        """
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

        # ── Safety: ensure shelf_start < shelf_end ────────────────────────
        if self.shelf_start_row >= self.shelf_end_row:
            self.shelf_end_row = self.shelf_start_row + 1

        # ── Safety: keep cross_aisle_row inside current shelf zone ────────
        self.cross_aisle_row = max(self.shelf_start_row,
                                   min(self.shelf_end_row, self.cross_aisle_row))

    def _evaluate_layout(self):
        """
        Build warehouse, check constraints, run simulation,
        compute normalised 3-KPI reward.
        """
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

        # ── Normalise each KPI to 0-1 ────────────────────────────────────
        def norm_higher_better(value, lo, hi):
            """Normalise where higher raw value = better (picks/hr)."""
            if hi <= lo:
                return 0.5
            return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

        def norm_lower_better(value, lo, hi):
            """Normalise where lower raw value = better (distance, cong)."""
            if hi <= lo:
                return 0.5
            return float(np.clip((hi - value) / (hi - lo), 0.0, 1.0))

        picks_lo, picks_hi = self.kpi_bounds["picks_per_hour"]
        dist_lo,  dist_hi  = self.kpi_bounds["dist_per_agent"]
        cong_lo,  cong_hi  = self.kpi_bounds["congestion_rate"]

        P = norm_higher_better(result["picks_per_hour"],  picks_lo, picks_hi)
        D = norm_lower_better(result["dist_per_agent"],   dist_lo,  dist_hi)
        C = norm_lower_better(result["congestion_rate"],  cong_lo,  cong_hi)

        # ── Weighted sum ──────────────────────────────────────────────────
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

        # ── Log this episode ──────────────────────────────────────────────
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
    min_storage = JOBS_MEAN * NUM_AGENTS
    print(f"  Storage minimum   : {min_storage} item cells "
          f"({JOBS_MEAN} jobs x {NUM_AGENTS} agents)")
    print(f"  Reward weights    : picks={W_PICKS:.2f}, "
          f"dist={W_DISTANCE:.2f}, cong={W_CONGESTION:.2f}")
    print("=" * 60)

    # ── Step 1: Calibrate KPI bounds ──────────────────────────────────────
    print("\n[1/5] Calibrating KPI normalisation bounds...")
    kpi_bounds = calibrate_kpi_bounds(
        num_samples=args.calibration_runs, eval_runs=1)

    # ── Step 2: Create and validate environment ───────────────────────────
    print("\n[2/5] Creating environment...")
    env = WarehouseLayoutEnv(kpi_bounds=kpi_bounds)

    print("[3/5] Validating environment...")
    check_env(env, warn=True)
    print("  Passed.\n")

    # ── Step 3: Train ─────────────────────────────────────────────────────
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

    # ── Save training log to CSV ──────────────────────────────────────────
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

    # ── Step 4: Test the trained agent ────────────────────────────────────
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

    # ── Baseline comparison ───────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  BASELINE (default layout):")

    default_cross = (1 + GRID_ROWS - 2) // 2
    baseline = run_single_simulation(
        aisle_width=2, centre_aisle_width=2,
        depot_col=GRID_COLS // 2,
        shelf_start_row=1, shelf_end_row=GRID_ROWS - 2,
        cross_aisle_row=default_cross,
        eval_runs=EVAL_RUNS,
    )

    if baseline is None:
        print("    ERROR: Default layout failed!")
    else:
        print(f"    Layout      : aisle_w=2, centre=2, "
              f"depot_col={GRID_COLS // 2}, shelf=1-{GRID_ROWS - 2}, "
              f"cross_aisle_row={default_cross}")
        print(f"    Storage     : {baseline['num_items']} item cells")
        print(f"    Picks/hr    : {baseline['picks_per_hour']:.2f}")
        print(f"    Dist/agent  : {baseline['dist_per_agent']:.1f}")
        print(f"    Congestion  : {baseline['congestion_rate']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    