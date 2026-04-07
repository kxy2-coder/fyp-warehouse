# =============================================================================
# agent.py — Vectorised Warehouse Workers
# =============================================================================
# This file defines the Agent class — but now ONE Agent object manages ALL
# warehouse workers simultaneously using numpy arrays instead of a for-loop
# over individual Agent objects.
#
# WHY VECTORISE?
#   Instead of looping "for each agent, do maths", we store every agent's
#   data in numpy arrays and do ALL agents' maths in one numpy call.
#   Example: fatigue for 4 agents used to be 4 separate calls to math.exp.
#   Now it is ONE call: np.exp(-FATIGUE_d * self.the ) which processes
#   all 4 agents' work_time values in a single C-level operation.
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
#      Each agent independently has a 35% chance of being experienced (B=1000h).
#
# STATE ENCODING (integers instead of strings for numpy compatibility):
#   0 = waiting     — at depot, shift not started
#   1 = to_item     — walking toward target item
#   2 = picking_up  — dwelling at shelf, physically picking up item
#   3 = to_depot    — walking back to depot
#   4 = resting     — at depot, recovering before next order
#   5 = all_done    — all jobs for this run complete
#
# DATA LAYOUT:
#   Every per-agent value is a 1-D numpy array of length n_agents.
#   Index i always refers to agent i (0-indexed internally; displayed as i+1).
#   Paths are kept as a Python list-of-lists (variable length, not numpy-able).
# =============================================================================

import math
import random
import numpy as np
from pathfinder import find_path_to_neighbour, find_path

# ── State integer constants (exported so main.py/rl_env.py can use them) ─────
STATE_WAITING  = 0
STATE_TO_ITEM  = 1
STATE_PICKING  = 2
STATE_TO_DEPOT = 3
STATE_RESTING  = 4
STATE_ALL_DONE = 5

# ── NHPP demand model parameters ─────────────────────────────────────────────
SHIFT_TICKS = 28800    # 8 hours at 1 tick/second
LAMBDA_BASE = 0.0778   # base arrival rate per tick (≈ 70 picks/hr × 4 agents / 3600)
DEMAND_BETA = 0.3      # amplitude parameter (moderate time-varying shape)
# JOBS_MEAN = 70  (deprecated — kept for reference)
# JOBS_STD  = 14  (deprecated — kept for reference)

# ── Number of runs for the experiment mode (--experiment flag) ────────────────
NUM_RUNS = 50

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

EXPERIENCE_NOVICE      = 20.0
EXPERIENCE_EXPERT      = 1000.0
EXPERIENCE_PROB_EXPERT = 0.35

# Precompute the learning-rate exponent once (avoids repeating log/log every tick).
# b = -log(LEARNING_RATE) / log(2)
_LEARNING_EXPONENT = -math.log(LEARNING_RATE) / math.log(2)

# Legacy dict kept for any direct references — not used for assignment.
EXPERIENCE_B = {}


def draw_experience():
    """Randomly assign prior experience hours for one agent."""
    return EXPERIENCE_EXPERT if random.random() < EXPERIENCE_PROB_EXPERT else EXPERIENCE_NOVICE


# draw_job_quota() — deprecated; replaced by NHPP arrival model below.


def get_multiplier(tick):
    """Return NHPP demand multiplier m(t) for the given tick."""
    beta     = DEMAND_BETA
    m_peak   = 1.0 + beta * 0.4
    m_trough = 1.0 - beta * 0.3
    if tick < 7200:
        return m_peak
    elif tick < 21600:
        return m_trough
    else:
        return m_peak


def nhpp_arrival(tick):
    """Return True if a job arrives at this tick (Bernoulli draw)."""
    rate = LAMBDA_BASE * get_multiplier(tick)
    return random.random() < rate


# =============================================================================
# Agent — manages ALL N workers simultaneously via numpy arrays
# =============================================================================

class Agent:
    """
    Manages ALL warehouse workers at once using numpy arrays.

    Instead of creating one Agent per worker, create ONE Agent that holds
    every worker's state in 1-D arrays of length n_agents.

    Parameters
    ----------
    n_agents : int
        Number of workers to simulate.
    grid : Grid
        The warehouse grid (used for pathfinding and item removal).
    quota : int
        Job quota shared by all workers in this run.
    """

    def __init__(self, n_agents, grid, quota):
        self.n    = n_agents
        self.grid = grid
        self.job_queue    = 0   # jobs currently available to pick
        self.total_orders = 0   # tracks total jobs that have arrived

        depot = grid.depot   # (row, col) tuple

        # ── Position arrays ───────────────────────────────────────────────
        # All agents start at the depot.
        self.pos_row = np.full(n_agents, depot[0], dtype=np.int32)
        self.pos_col = np.full(n_agents, depot[1], dtype=np.int32)

        # ── State array (integers 0-5) ─────────────────────────────────────
        # All agents start in STATE_WAITING.
        self.state = np.zeros(n_agents, dtype=np.int32)

        # ── Time accumulators ─────────────────────────────────────────────
        self.work_time = np.zeros(n_agents, dtype=np.float64)
        self.rest_time = np.zeros(n_agents, dtype=np.float64)

        # ── Fatigue ───────────────────────────────────────────────────────
        self.fatigue = np.zeros(n_agents, dtype=np.float64)

        # ── Experience (hours of prior experience) ────────────────────────
        self.experience_B = np.array(
            [draw_experience() for _ in range(n_agents)], dtype=np.float64
        )

        # ── Metrics ───────────────────────────────────────────────────────
        self.distance         = np.zeros(n_agents, dtype=np.int32)
        self.orders_completed = np.zeros(n_agents, dtype=np.int32)
        self.blocked_events   = np.zeros(n_agents, dtype=np.int32)
        self.idle_ticks       = np.zeros(n_agents, dtype=np.int32)

        # ── Blocking ──────────────────────────────────────────────────────
        self.blocked_ticks = np.zeros(n_agents, dtype=np.int32)

        # ── Pickup countdown ──────────────────────────────────────────────
        self.pickup_ticks = np.zeros(n_agents, dtype=np.int32)

        # ── Paths (can't be numpy — each agent has a different-length path) ─
        # paths[i]        : list of (row, col) tuples for agent i
        # path_indices[i] : next step index in paths[i]
        # targets[i]      : (row, col) of current target shelf cell, or None
        self.paths        = [[] for _ in range(n_agents)]
        self.path_indices = np.zeros(n_agents, dtype=np.int32)
        self.targets      = [None] * n_agents

    # =========================================================================
    # Human-factors maths — all vectorised over n_agents
    # =========================================================================

    def _update_fatigue_all(self):
        """
        Recompute fatigue for every agent in one numpy call.

        Fatigue = clip(buildup + recovery, 0, 1)
          buildup  = 1 - exp(-FATIGUE_d * work_time)   [increases with work]
          recovery = exp(-FATIGUE_r * rest_time) - 1   [negative; reduces fatigue]
        """
        buildup  = 1.0 - np.exp(-FATIGUE_d * self.work_time)
        recovery = np.exp(-FATIGUE_r * self.rest_time) - 1.0
        self.fatigue = np.clip(buildup + recovery, 0.0, 1.0)

    def _experience_factor(self):
        """
        Compute E(w, B) = M + (1-M) * (w + B)^(-b) for all agents at once.
        Returns a 1-D array of length n_agents.
        """
        w_total = self.work_time + self.experience_B
        w_total = np.maximum(w_total, 0.001)   # guard against zero
        return AUTOMATION_M + (1.0 - AUTOMATION_M) * np.power(w_total, -_LEARNING_EXPONENT)

    def _calc_pickup_ticks_all(self):
        """
        Da = (1 + alpha * F) * E(w,B) * base_ticks  for all agents.
        Returns an int32 array (minimum 1 tick per agent).
        """
        E  = self._experience_factor()
        Da = (1.0 + FATIGUE_alpha * self.fatigue) * E * PICKUP_BASE_TICKS
        return np.maximum(1, np.round(Da).astype(np.int32))

    # =========================================================================
    # Navigation helpers — still per-agent (paths differ)
    # =========================================================================

    def _pick_next_item(self):
        available = self.grid.get_all_item_positions()
        if not available:
            return None
        return random.choice(available)

    def _plan_path_to_item(self, i):
        pos = (int(self.pos_row[i]), int(self.pos_col[i]))
        self.paths[i]        = find_path_to_neighbour(self.grid, pos, self.targets[i])
        self.path_indices[i] = 0
        self.state[i]        = STATE_TO_ITEM

    def _plan_path_to_depot(self, i):
        pos = (int(self.pos_row[i]), int(self.pos_col[i]))
        self.paths[i]        = find_path(self.grid, pos, self.grid.depot)
        self.path_indices[i] = 0
        self.state[i]        = STATE_TO_DEPOT

    # =========================================================================
    # Public interface
    # =========================================================================

    def peek_next_pos(self, i):
        """Return the next cell agent i intends to move to, or None."""
        if self.blocked_ticks[i] > 0:
            return None
        if self.state[i] in (STATE_TO_ITEM, STATE_TO_DEPOT):
            if self.path_indices[i] < len(self.paths[i]):
                return self.paths[i][self.path_indices[i]]
        return None

    def is_done(self, i):
        """Return True if agent i has finished all its jobs."""
        return bool(self.state[i] == STATE_ALL_DONE)

    def all_done(self):
        """Return True when every agent has finished (fallback safety check)."""
        return bool(np.all(self.state == STATE_ALL_DONE))

    def add_job(self):
        """Add one job to the queue (called by simulation loop each tick)."""
        self.job_queue    += 1
        self.total_orders += 1

    # =========================================================================
    # step() — advance ALL agents by one simulation tick
    # =========================================================================

    def step(self):
        """
        Advance every agent by one tick using vectorised numpy operations.

        IMPORTANT — state masks are captured at the START of the tick.
        This means a transition that happens mid-tick (e.g. to_item → picking_up)
        does not cause double-processing within the same tick.  This exactly
        matches the original single-agent step() behaviour.

        Rough order of operations:
          1. Waiting agents pick a target and start walking (small loop).
          2. Walking agents (blocked / pausing / moving) — vectorised + small loop.
          3. Picking agents count down; deplete item when done (small loop).
          4. Resting agents accumulate rest time; pick next job (small loop).
          5. Fatigue updated for ALL agents in one numpy call.
        """

        # ── Snapshot state masks at tick start ────────────────────────────────
        # Using pre-computed masks ensures that state transitions made below
        # don't accidentally trigger a second state's logic in the same tick.
        waiting   = self.state == STATE_WAITING
        to_item   = self.state == STATE_TO_ITEM
        picking   = self.state == STATE_PICKING
        to_depot  = self.state == STATE_TO_DEPOT
        resting   = self.state == STATE_RESTING

        # ── 1. Waiting → start first order once a job is available ──────────
        self.idle_ticks[waiting] += 1
        waiting_indices = sorted(np.where(waiting)[0],
                                 key=lambda i: self.idle_ticks[i], reverse=True)
        for i in waiting_indices:
            if self.job_queue > 0:
                target = self._pick_next_item()
                if target:
                    self.job_queue -= 1
                    self.targets[i] = target
                    self._plan_path_to_item(i)
                # else: shelves depleted — stay waiting until replenished
            # else: no jobs dispatched yet — stay waiting

        # ── 2. Walking (to_item or to_depot) ─────────────────────────────────
        walking = to_item | to_depot

        # Identify blocked agents BEFORE decrementing (so not_blocked is correct).
        blocked     = walking & (self.blocked_ticks > 0)
        not_blocked = walking & ~blocked

        # Blocked: serve out wait, add work time.
        self.blocked_ticks[blocked] -= 1
        self.work_time[blocked]     += WALK_TIME

        # Not blocked: roll for fatigue-induced pause.
        # pause_probability = alpha * F / 2
        # We use Python's random.random() (same as original) so that the RNG
        # stream is identical to the non-vectorised version for the same seed.
        # Only non-blocked walking agents consume a random number — matching the
        # original call count exactly.
        pausing = np.zeros(self.n, dtype=bool)
        for i in np.where(not_blocked)[0]:
            if random.random() < FATIGUE_alpha * self.fatigue[i] / 2.0:
                pausing[i] = True
        self.work_time[pausing] += WALK_TIME

        # Moving: not blocked AND not pausing.
        moving = not_blocked & ~pausing

        # Pre-compute pickup ticks once for the whole tick.
        # (Only used for agents that reach their item shelf this tick.)
        pickup_ticks_arr = self._calc_pickup_ticks_all()

        for i in np.where(moving)[0]:
            if self.path_indices[i] < len(self.paths[i]):
                # Take next step along path.
                pos = self.paths[i][self.path_indices[i]]
                self.pos_row[i]      = pos[0]
                self.pos_col[i]      = pos[1]
                self.path_indices[i] += 1
                self.distance[i]    += 1
                self.work_time[i]   += WALK_TIME
            else:
                # Reached end of path — transition state.
                if to_item[i]:   # was STATE_TO_ITEM at tick start
                    self.pickup_ticks[i] = pickup_ticks_arr[i]
                    self.state[i]        = STATE_PICKING
                else:            # was STATE_TO_DEPOT
                    self.state[i] = STATE_RESTING

        # ── 3. Picking up ─────────────────────────────────────────────────────
        # Decrement counters and add time for all currently-picking agents.
        self.pickup_ticks[picking] -= 1
        self.work_time[picking]    += PICKUP_BASE_TIME

        # Agents that finished picking this tick.
        finished_picking = picking & (self.pickup_ticks <= 0)
        for i in np.where(finished_picking)[0]:
            self.grid.deplete_item(*self.targets[i])
            self.orders_completed[i] += 1
            self._plan_path_to_depot(i)

        # ── 4. Resting at depot ────────────────────────────────────────────
        self.rest_time[resting]  += DROPOFF_TIME
        self.idle_ticks[resting] += 1

        # Agents needing next job only if queue has jobs available.
        # Do NOT transition to STATE_ALL_DONE here — shift end is tick-based.
        queue_available = resting & (self.job_queue > 0)

        resting_indices = sorted(np.where(queue_available)[0],
                                 key=lambda i: self.idle_ticks[i], reverse=True)
        for i in resting_indices:
            if self.job_queue <= 0:
                break   # queue exhausted by earlier agents this tick
            target = self._pick_next_item()
            if target:
                self.job_queue -= 1
                self.targets[i] = target
                self._plan_path_to_item(i)
            else:
                pass    # shelf depleted — stay resting until replenished

        # ── 5. Update fatigue for every agent in one numpy call ───────────────
        self._update_fatigue_all()