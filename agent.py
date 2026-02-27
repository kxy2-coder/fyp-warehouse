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
#      Fatigue builds up as the agent works, modeled as:
#        I(w) = 1 - e^(-d * w)
#      where w = cumulative work time (hours), d = task intensity coefficient.
#      Calibrated value from paper: d = 0.20
#
#   2. RECOVERY (exponential decay during rest at depot)
#      Fatigue partially recovers while the agent rests at depot:
#        R(x) = e^(-r * x) - 1
#      where x = rest time (hours), r = rest quality coefficient.
#      Calibrated value from paper: r = 0.25
#      Current fatigue: F(w, x) = I(w) + R(x), clamped to [0, 1]
#
#   3. FATIGUE AFFECTS WALKING SPEED
#      A fatigued agent walks slower. Each walking step may be skipped
#      (agent pauses for one tick) based on fatigue level:
#        Sf = S1 * (1 - alpha * F(w,x) / 2)
#      where S1 = base speed, alpha = 0.4 (fatigue binding factor).
#      Implemented as a probability of pausing each step.
#
#   4. FATIGUE AFFECTS PICKUP DURATION
#      Picking up an item takes longer when fatigued:
#        Da = (1 + alpha * F(w,x)) * E(w,B) * Do
#      where Do = base pickup duration, E(w,B) = experience factor.
#
#   5. EXPERIENCE / LEARNING CURVE (S-curve model)
#      Agents improve with accumulated work hours:
#        E(w, B) = M + (1 - M) * (w + B)^(-b)
#      where B = prior experience (hours), M = 0 (fully manual),
#      b = -log(LR)/log(2), LR = learning rate.
#      A lower E means faster task execution.
#      Agent 1 starts as experienced (B=1000h), Agent 2 as novice (B=20h).
#
# TIME COSTS PER ACTION (grounding ticks in real time):
#   Each action adds a real-time cost in hours to work_time or rest_time.
#   Walking one cell (~1 metre at ~1 m/s) = 1 second = 0.000278 hours
#   Picking up one item (~10 seconds)     = 0.00278 hours
#   Resting at depot   (~30 seconds)      = 0.00833 hours
#
# STATES:
#   "waiting"    — at depot, shift not started
#   "to_item"    — walking toward target item
#   "picking_up" — dwelling at shelf, physically picking up item
#   "to_depot"   — walking back to depot
#   "resting"    — at depot, recovering before next order
#   "all_done"   — all TOTAL_ORDERS complete
# =============================================================================

import math
import random
from pathfinder import find_path_to_neighbour, find_path

# ── Order quota ───────────────────────────────────────────────────────────────
TOTAL_ORDERS = 20

# ── Real-time costs per action (in hours) ────────────────────────────────────
WALK_TIME         = 1 / 3600       # 1 second per cell walked
PICKUP_BASE_TIME  = 10 / 3600      # 10 seconds base pickup time
DROPOFF_TIME      = 30 / 3600      # 30 seconds resting/unloading at depot

# ── Base pickup duration in ticks (before fatigue/experience scaling) ─────────
PICKUP_BASE_TICKS  = 5              # visible simulation ticks for pickup
BLOCKED_WAIT_TICKS = 3             # ticks a blocked agent waits (1 tick = 1 second)

# ── Fatigue model parameters (Malpas & Relvas, 2025) ─────────────────────────
FATIGUE_d     = 0.20   # task intensity coefficient  — controls fatigue buildup rate
FATIGUE_r     = 0.25   # rest quality coefficient    — controls recovery rate
FATIGUE_alpha = 0.40   # fatigue binding factor      — scales fatigue effect on tasks

# ── Experience / learning curve parameters ───────────────────────────────────
LEARNING_RATE = 0.90   # LR: fraction of time saved each time experience doubles
AUTOMATION_M  = 0.0    # M: 0 = fully manual (no automation floor)

# Prior experience in hours for each agent
# Agent 1 = experienced worker (proficient ~1 year ≈ 1000h from paper)
# Agent 2 = novice worker      (new starter  ~1 week ≈ 20h from paper)
EXPERIENCE_B = {1: 1000.0, 2: 20.0}


class Agent:
    """
    A warehouse worker that completes TOTAL_ORDERS pickup trips.
    Behavior is governed by fatigue and experience following
    Malpas & Relvas (2025).
    """

    def __init__(self, grid, agent_id=1, color=(60, 200, 80)):
        """
        Set up the agent at the depot, ready to start.

        grid     — the Grid object (pathfinding + item removal)
        agent_id — 1 or 2, used for display and logging
        color    — RGB tuple for drawing this agent on screen
        """
        self.grid     = grid
        self.agent_id = agent_id
        self.color    = color

        self.pos   = grid.depot
        self.state = "waiting"

        # Current target item — (row, col) of shelf being collected
        self.target = None

        # Path being followed
        self.path       = []
        self.path_index = 0

        # ── Pickup dwell ──────────────────────────────────────────────────────
        # Counts down during "picking_up" state
        self.pickup_ticks_remaining = 0

        # ── Real-time tracking ────────────────────────────────────────────────
        self.work_time = 0.0    # cumulative hours worked (drives fatigue buildup)
        self.rest_time = 0.0    # cumulative hours rested (drives fatigue recovery)

        # ── Fatigue ───────────────────────────────────────────────────────────
        self.fatigue = 0.0      # current fatigue level F(w,x) in [0.0, 1.0]

        # ── Experience ───────────────────────────────────────────────────────
        # Prior experience in hours — differs between agents
        self.experience_B = EXPERIENCE_B.get(agent_id, 20.0)

        # ── Metrics ───────────────────────────────────────────────────────────
        self.distance         = 0    # total cells walked
        self.orders_completed = 0    # total items collected

        # ── Blocking / right-of-way ───────────────────────────────────────────
        self.blocked_ticks_remaining = 0   # counts down while agent is waiting to pass
        self.blocked_count           = 0   # number of blocking events (not ticks)

    # =========================================================================
    # Human factors calculations
    # =========================================================================

    def _fatigue_buildup(self):
        """
        I(w) = 1 - e^(-d * w)
        Fatigue accumulated from work_time hours of work.
        Returns a value in [0, 1].
        """
        return 1.0 - math.exp(-FATIGUE_d * self.work_time)

    def _fatigue_recovery(self):
        """
        R(x) = e^(-r * x) - 1
        Recovery from rest_time hours of rest.
        Returns a negative value (reduces fatigue), clamped so
        total fatigue never goes below 0.
        """
        return math.exp(-FATIGUE_r * self.rest_time) - 1.0

    def _update_fatigue(self):
        """
        F(w, x) = I(w) + R(x), clamped to [0.0, 1.0].
        Called after work_time or rest_time changes.
        """
        raw = self._fatigue_buildup() + self._fatigue_recovery()
        self.fatigue = max(0.0, min(1.0, raw))

    def _experience_factor(self):
        """
        E(w, B) = M + (1 - M) * (w + B)^(-b)
        Learning curve from Malpas & Relvas (2025).
        Returns a multiplier: < 1 means faster than baseline,
        approaches M as experience grows.
        w = current work_time, B = prior experience hours.
        """
        b = -math.log(LEARNING_RATE) / math.log(2)
        w_total = self.work_time + self.experience_B
        # Avoid division by zero
        if w_total <= 0:
            w_total = 0.001
        return AUTOMATION_M + (1.0 - AUTOMATION_M) * (w_total ** -b)

    def _adjusted_pickup_ticks(self):
        """
        Da = (1 + alpha * F(w,x)) * E(w,B) * Do
        Returns the number of pickup ticks after fatigue and experience
        scaling. Always at least 1 tick.
        """
        E  = self._experience_factor()
        Da = (1.0 + FATIGUE_alpha * self.fatigue) * E * PICKUP_BASE_TICKS
        return max(1, round(Da))

    def _should_pause_walk(self):
        """
        Sf = S1 * (1 - alpha * F(w,x) / 2)
        Returns True if the agent should pause this tick (fatigue slows them).
        Implemented as a probability: higher fatigue = more frequent pauses.
        Pause probability = alpha * F / 2, so at max fatigue (F=1):
        pause chance = 0.4 * 1.0 / 2 = 0.20 (20% chance of pausing per step).
        """
        pause_probability = FATIGUE_alpha * self.fatigue / 2.0
        return random.random() < pause_probability

    # =========================================================================
    # Private navigation helpers
    # =========================================================================

    def _pick_next_item(self):
        """Randomly choose one available item from the grid."""
        available = self.grid.get_all_item_positions()
        if not available:
            return None
        return random.choice(available)

    def _plan_path_to_item(self):
        """Path to the cell adjacent to the target shelf."""
        self.path       = find_path_to_neighbour(self.grid, self.pos, self.target)
        self.path_index = 0
        self.state      = "to_item"

    def _plan_path_to_depot(self):
        """Path from current position back to the depot."""
        self.path       = find_path(self.grid, self.pos, self.grid.depot)
        self.path_index = 0
        self.state      = "to_depot"

    # =========================================================================
    # Public interface
    # =========================================================================

    def peek_next_pos(self):
        """Return the next cell this agent intends to walk to, or None."""
        if self.blocked_ticks_remaining > 0:
            return None   # already waiting — don't re-trigger a block
        if self.state in ("to_item", "to_depot") and self.path_index < len(self.path):
            return self.path[self.path_index]
        return None

    def step(self):
        """
        Advance the agent by one simulation tick.
        Called once per tick by main.py.
        """

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
            # Blocked by higher-priority agent (countdown set by resolve_right_of_way)
            if self.blocked_ticks_remaining > 0:
                self.blocked_ticks_remaining -= 1
                self.work_time += WALK_TIME
                self._update_fatigue()
                return

            # Check fatigue-based walking pause
            if self._should_pause_walk():
                # Agent pauses this tick — still counts as work time
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
                # Arrived next to shelf — begin pickup dwell
                self.pickup_ticks_remaining = self._adjusted_pickup_ticks()
                self.state = "picking_up"
                print(
                    f"  [Agent {self.agent_id}] Reached shelf {self.target}"
                    f" | fatigue={self.fatigue:.2f}"
                    f" | pickup will take {self.pickup_ticks_remaining} ticks"
                )

        # ── Picking up item (dwell) ───────────────────────────────────────────
        elif self.state == "picking_up":
            self.pickup_ticks_remaining -= 1
            self.work_time += PICKUP_BASE_TIME
            self._update_fatigue()

            if self.pickup_ticks_remaining <= 0:
                self.grid.remove_item(*self.target)
                self.orders_completed += 1
                print(
                    f"  [Agent {self.agent_id}] Picked up item"
                    f" | orders={self.orders_completed}/{TOTAL_ORDERS}"
                    f" | fatigue={self.fatigue:.2f}"
                    f" | exp_factor={self._experience_factor():.3f}"
                )
                self._plan_path_to_depot()

        # ── Walking back to depot ─────────────────────────────────────────────
        elif self.state == "to_depot":
            # Blocked by higher-priority agent (countdown set by resolve_right_of_way)
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
                # Arrived at depot — rest briefly before next order
                self.state = "resting"

        # ── Resting at depot ──────────────────────────────────────────────────
        elif self.state == "resting":
            self.rest_time += DROPOFF_TIME
            self._update_fatigue()

            print(
                f"  [Agent {self.agent_id}] Resting at depot"
                f" | fatigue after rest={self.fatigue:.2f}"
            )

            if self.orders_completed >= TOTAL_ORDERS:
                self.state = "all_done"
                print(f"  [Agent {self.agent_id}] All orders complete!")
            else:
                self.target = self._pick_next_item()
                if self.target:
                    self._plan_path_to_item()
                else:
                    self.state = "all_done"

    def is_done(self):
        """Returns True when the agent has completed all TOTAL_ORDERS trips."""
        return self.state == "all_done"
