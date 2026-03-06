# =============================================================================
# agent.py 
# =============================================================================
# This file defines the agent — a simulated warehouse worker.
#
# Human factors are modeled based on:
#   Malpas & Relvas (2025) — "Designing a virtual warehouse operator
#   integrating fatigue, recovery, and learning using agent-based modeling"
#
# HUMAN FACTORS IMPLEMENTED:
#
#   1. FATIGUE (exponential buildup during work)
#      I(w) = 1 - e^(-d * w)     d = 0.20
#
#   2. RECOVERY (exponential decay during rest at depot)
#      R(x) = e^(-r * x) - 1     r = 0.25
#
#   3. FATIGUE AFFECTS WALKING SPEED
#      Pause probability = alpha * F / 2
#
#   4. FATIGUE AFFECTS PICKUP DURATION
#      Da = (1 + alpha * F) * E(w,B) * Do
#
#   5. EXPERIENCE / LEARNING CURVE
#      E(w, B) = M + (1 - M) * (w + B)^(-b)
#      Agent 1 = experienced (B=1000h), all others = novice (B=20h)
#
# JOB QUOTA — DISTRIBUTION-BASED:
#   Rather than a fixed order quota, the number of jobs per simulation run
#   is drawn from a normal distribution N(JOBS_MEAN, JOBS_STD).
#   This means each run has a slightly different workload, reflecting
#   realistic day-to-day demand variability in a warehouse.
#
#   Running the simulation many times (NUM_RUNS) and averaging the results
#   gives a robust KPI that holds across varying demand levels — this is
#   a sensitivity analysis approach, meaning layout conclusions are not
#   dependent on one specific demand assumption.
#
#   JOBS_MEAN and JOBS_STD are placeholder values for now.
#   They will be calibrated against literature once KPIs are finalised.
#   NUM_RUNS controls how many silent runs are averaged in --experiment mode.
#
# TIME COSTS PER ACTION:
#   Walking one cell (~1 m/s)  = 1 second  = 1/3600 hours
#   Picking one item (~10 s)   = 10/3600 hours
#   Resting at depot (~30 s)   = 30/3600 hours
#
# STATES:
#   "waiting"    — at depot, shift not started
#   "to_item"    — walking toward target item
#   "picking_up" — dwelling at shelf, physically picking up item
#   "to_depot"   — walking back to depot
#   "resting"    — at depot, recovering before next order
#   "all_done"   — all jobs for this run complete
# =============================================================================

import math
import random
from pathfinder import find_path_to_neighbour, find_path

# ── Job quota distribution parameters ────────────────────────────────────────
# Each simulation run draws the number of jobs from N(JOBS_MEAN, JOBS_STD).
# These are placeholder values — to be calibrated once KPIs are finalised.
# Sensitivity analysis will test low / medium / high demand scenarios.
JOBS_MEAN = 70     # placeholder mean jobs per agent per run
JOBS_STD  = 14     # placeholder std (20% of mean — conventional assumption)

# ── Number of runs for the experiment mode (--experiment flag) ────────────────
# Each layout is evaluated over NUM_RUNS independent runs and results averaged.
NUM_RUNS = 30

# ── Real-time costs per action (in hours) ────────────────────────────────────
WALK_TIME         = 1 / 3600       # 1 second per cell walked
PICKUP_BASE_TIME  = 10 / 3600      # 10 seconds base pickup time
DROPOFF_TIME      = 30 / 3600      # 30 seconds resting/unloading at depot

# ── Base pickup duration in ticks (before fatigue/experience scaling) ─────────
PICKUP_BASE_TICKS  = 5
BLOCKED_WAIT_TICKS = 3             # ticks a blocked agent waits (1 tick = 1 second)

# ── Fatigue model parameters (Malpas & Relvas, 2025) ─────────────────────────
FATIGUE_d     = 0.20
FATIGUE_r     = 0.25
FATIGUE_alpha = 0.40

# ── Experience / learning curve parameters ───────────────────────────────────
LEARNING_RATE = 0.90
AUTOMATION_M  = 0.0

# Experience levels for the two worker types (hours of prior experience).
EXPERIENCE_NOVICE     = 20.0    # novice worker
EXPERIENCE_EXPERT     = 1000.0  # experienced worker
EXPERIENCE_PROB_EXPERT = 0.35   # 35% chance any agent is experienced (realistic mix)

# Legacy dict kept for any direct references — not used for assignment any more.
EXPERIENCE_B = {}


def draw_experience():
    """
    Randomly assign prior experience for one agent.
    Each agent independently has a 35% chance of being experienced.
    This produces a realistic mixed workforce across runs.
    """
    import random
    return EXPERIENCE_EXPERT if random.random() < EXPERIENCE_PROB_EXPERT else EXPERIENCE_NOVICE


def draw_job_quota():
    """
    Draw the number of jobs for one simulation run from N(JOBS_MEAN, JOBS_STD).
    Clamped to a minimum of 1 to avoid zero or negative quotas.
    """
    n = random.gauss(JOBS_MEAN, JOBS_STD)
    return max(1, round(n))


class Agent:
    """
    A warehouse worker that completes a randomly drawn number of pickup trips.
    The job quota is set at creation time by drawing from the distribution.
    Behaviour is governed by fatigue and experience (Malpas & Relvas, 2025).
    """

    def __init__(self, grid, agent_id=1, color=(60, 200, 80), total_orders=None):
        """
        Set up the agent at the depot, ready to start.

        grid         — the Grid object (pathfinding + item removal)
        agent_id     — integer id, used for display and logging
        color        — RGB tuple for drawing this agent on screen
        total_orders — job quota for this run. If None, draws from distribution.
                       Pass an explicit value to give all agents the same quota
                       within a single run (recommended).
        """
        self.grid     = grid
        self.agent_id = agent_id
        self.color    = color

        # Job quota for this run — same value passed to all agents in one run
        self.total_orders = total_orders if total_orders is not None else draw_job_quota()

        self.pos   = grid.depot
        self.state = "waiting"

        self.target     = None
        self.path       = []
        self.path_index = 0

        self.pickup_ticks_remaining = 0

        # ── Real-time tracking ────────────────────────────────────────────────
        self.work_time = 0.0
        self.rest_time = 0.0

        # ── Fatigue ───────────────────────────────────────────────────────────
        self.fatigue = 0.0

        # ── Experience ───────────────────────────────────────────────────────
        self.experience_B = draw_experience()

        # ── Metrics ───────────────────────────────────────────────────────────
        self.distance         = 0
        self.orders_completed = 0

        # ── Blocking ──────────────────────────────────────────────────────────
        self.blocked_ticks_remaining = 0
        self.blocked_count           = 0

    # =========================================================================
    # Human factors
    # =========================================================================

    def _fatigue_buildup(self):
        return 1.0 - math.exp(-FATIGUE_d * self.work_time)

    def _fatigue_recovery(self):
        return math.exp(-FATIGUE_r * self.rest_time) - 1.0

    def _update_fatigue(self):
        raw = self._fatigue_buildup() + self._fatigue_recovery()
        self.fatigue = max(0.0, min(1.0, raw))

    def _experience_factor(self):
        b = -math.log(LEARNING_RATE) / math.log(2)
        w_total = self.work_time + self.experience_B
        if w_total <= 0:
            w_total = 0.001
        return AUTOMATION_M + (1.0 - AUTOMATION_M) * (w_total ** -b)

    def _adjusted_pickup_ticks(self):
        E  = self._experience_factor()
        Da = (1.0 + FATIGUE_alpha * self.fatigue) * E * PICKUP_BASE_TICKS
        return max(1, round(Da))

    def _should_pause_walk(self):
        pause_probability = FATIGUE_alpha * self.fatigue / 2.0
        return random.random() < pause_probability

    # =========================================================================
    # Private navigation helpers
    # =========================================================================

    def _pick_next_item(self):
        available = self.grid.get_all_item_positions()
        if not available:
            return None
        return random.choice(available)

    def _plan_path_to_item(self):
        self.path       = find_path_to_neighbour(self.grid, self.pos, self.target)
        self.path_index = 0
        self.state      = "to_item"

    def _plan_path_to_depot(self):
        self.path       = find_path(self.grid, self.pos, self.grid.depot)
        self.path_index = 0
        self.state      = "to_depot"

    # =========================================================================
    # Public interface
    # =========================================================================

    def peek_next_pos(self):
        if self.blocked_ticks_remaining > 0:
            return None
        if self.state in ("to_item", "to_depot") and self.path_index < len(self.path):
            return self.path[self.path_index]
        return None

    def step(self):
        """Advance the agent by one simulation tick."""

        # ── Waiting: start first order ────────────────────────────────────────
        if self.state == "waiting":
            self.target = self._pick_next_item()
            if self.target:
                self._plan_path_to_item()
            else:
                self.state = "all_done"
            return

        # ── Walking toward item ───────────────────────────────────────────────
        if self.state == "to_item":
            if self.blocked_ticks_remaining > 0:
                self.blocked_ticks_remaining -= 1
                self.work_time += WALK_TIME
                self._update_fatigue()
                return

            if self._should_pause_walk():
                self.work_time += WALK_TIME
                self._update_fatigue()
                return

            if self.path_index < len(self.path):
                self.pos         = self.path[self.path_index]
                self.path_index += 1
                self.distance   += 1
                self.work_time  += WALK_TIME
                self._update_fatigue()
            else:
                self.pickup_ticks_remaining = self._adjusted_pickup_ticks()
                self.state = "picking_up"

        # ── Picking up item ───────────────────────────────────────────────────
        elif self.state == "picking_up":
            self.pickup_ticks_remaining -= 1
            self.work_time += PICKUP_BASE_TIME
            self._update_fatigue()

            if self.pickup_ticks_remaining <= 0:
                self.grid.remove_item(*self.target)
                self.orders_completed += 1
                self._plan_path_to_depot()

        # ── Walking back to depot ─────────────────────────────────────────────
        elif self.state == "to_depot":
            if self.blocked_ticks_remaining > 0:
                self.blocked_ticks_remaining -= 1
                self.work_time += WALK_TIME
                self._update_fatigue()
                return

            if self._should_pause_walk():
                self.work_time += WALK_TIME
                self._update_fatigue()
                return

            if self.path_index < len(self.path):
                self.pos         = self.path[self.path_index]
                self.path_index += 1
                self.distance   += 1
                self.work_time  += WALK_TIME
                self._update_fatigue()
            else:
                self.state = "resting"

        # ── Resting at depot ──────────────────────────────────────────────────
        elif self.state == "resting":
            self.rest_time += DROPOFF_TIME
            self._update_fatigue()

            if self.orders_completed >= self.total_orders:
                self.state = "all_done"
            else:
                self.target = self._pick_next_item()
                if self.target:
                    self._plan_path_to_item()
                else:
                    self.state = "all_done"

    def is_done(self):
        """Returns True when the agent has completed all jobs for this run."""
        return self.state == "all_done"