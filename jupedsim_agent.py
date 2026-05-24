# =============================================================================
# jupedsim_agent.py — JuPedSim-powered Agent Movement for Sim_V3
# =============================================================================
# Replaces agent.py + pathfinder.py from Sim_V2.
#
# WHAT CHANGES vs Sim_V2:
#   agent.py    — used A* to find a grid path, moved one cell per tick,
#                 conflicts detected when two agents shared a cell.
#   THIS FILE   — JuPedSim moves agents continuously through 2D space,
#                 collision avoidance is handled physically (agents slow
#                 down and steer around each other).
#
# WHAT STAYS THE SAME:
#   - Job assignment (NHPP model, job queue)
#   - State machine (waiting → to_item → picking → to_depot → resting)
#   - Item depletion / replenishment on the grid
#   - KPI format returned to rl_env.py
#
# HUMAN FACTORS (fatigue, experience, learning curve):
#   Not modelled in Sim_V3. JuPedSim's Collision-Free Speed Model already
#   provides empirically validated crowd dynamics (natural speed variation,
#   physical collision avoidance). Layering an uncalibrated fatigue model
#   on top would conflate physiological and physical effects. Acknowledged
#   as future work requiring empirical calibration data.
#
# COORDINATE SYSTEM:
#   grid cell (row, col) maps to continuous position (x, y) where:
#       x = col * CELL_SIZE + CELL_SIZE/2   (horizontal, left→right)
#       y = row * CELL_SIZE + CELL_SIZE/2   (vertical,  top→bottom)
#   CELL_SIZE = 1.0 means 1 grid cell = 1 metre.
# =============================================================================

import math
import random
import numpy as np
import jupedsim as jps
from shapely.geometry import Polygon
from shapely.ops import unary_union

from grid import Grid
from agent import nhpp_arrival, SHIFT_TICKS, LAMBDA_BASE, DEMAND_BETA

# ── State constants (same integers as agent.py for compatibility) ─────────────
STATE_WAITING  = 0
STATE_TO_ITEM  = 1
STATE_PICKING  = 2
STATE_TO_DEPOT = 3
STATE_RESTING  = 4
STATE_ALL_DONE = 5

# ── Physical constants ────────────────────────────────────────────────────────
CELL_SIZE         = 1.0   # 1 grid cell = 1 metre
FREE_SPEED        = 1.2   # mean free walking speed in m/s
SPEED_STD         = 0.12  # std deviation of desired walking speed (10% of FREE_SPEED).
                          # CRITICAL: the Collision-Free Speed Model (Tordeux 2016)
                          # produces a perfectly symmetric stalemate when two agents
                          # with identical desired speeds approach head-on — both
                          # decelerate to ~0 m/s and freeze for minutes at a time.
                          # Real workers walk at varying paces (the Tordeux paper
                          # itself recommends heterogeneous speeds for this reason).
                          # Per-agent Gaussian variation breaks the symmetry: the
                          # faster agent claims the path, the slower one gives way,
                          # and the head-on encounter resolves in 1–2 seconds
                          # instead of stalling indefinitely.
AGENT_RADIUS      = 0.3   # body radius in metres (≈ adult shoulder width, 0.6m diameter)
                          # NOTE: shelf_start_row >= 2 recommended — gives a 2m staging
                          # area (rows 0-1) before shelves start. shelf_start_row=1
                          # leaves only a 1m corridor at row 0 which causes deadlocks
                          # with 6 agents. The RL will discover the optimal value.
ARRIVAL_THRESHOLD  = 0.5              # metres — how close counts as "arrived at destination"
                                      # 0.5m ≈ agent stops at the shelf face (realistic)
                                      # still >> FREE_SPEED * dt (0.12m) so no overshoot misses
CONGESTION_RADIUS  = 3 * AGENT_RADIUS # 0.9m centre-to-centre — minimum meaningful proximity.
                                      # At this distance gap between bodies = 0.3m (nearly
                                      # touching). JuPedSim CFSM has already reduced speed to
                                      # ~50% free speed. Below 2×AGENT_RADIUS (0.6m) is
                                      # physically impossible — the model prevents overlap.
CONGESTION_SPEED   = FREE_SPEED * 0.5 # 0.6 m/s — speed below which an agent is considered
                                      # significantly slowed. At CONGESTION_RADIUS separation
                                      # JuPedSim physics naturally produces speeds near this
                                      # value, so the two thresholds are mutually consistent.
PICKUP_TICKS       = 10               # ticks spent picking up an item (same as Sim_V2)
REST_TICKS         = 10               # ticks resting at depot between jobs

# ── Anti-stuck "step aside" thresholds ────────────────────────────────────────
# CFSM produces persistent slow-downs when multiple agents converge in a
# narrow aisle.  Real workers step aside within a few seconds; we simulate
# this by redirecting frozen agents to a nearby waypoint.
STUCK_SPEED          = 0.10   # m/s — below this we consider the agent "not moving"
STUCK_TICKS          = 20     # consecutive frozen ticks before we step aside
                              # (30 s — generous tolerance, lets CFSM physics
                              # resolve most encounters naturally before we
                              # intervene; only fires for true deadlocks)
SIDESTEP_MIN_DIST    = 0.8    # minimum lateral move (don't sidestep in place)
SIDESTEP_MAX_DIST    = 2.5    # maximum lateral move (keep it a small detour)
SIDESTEP_REACHED_TOL = 0.7    # how close counts as "arrived at the side spot"

# How many JuPedSim iterations to run per simulation tick.
# JuPedSim uses small internal timesteps for accurate physics.
# JPS_STEPS_PER_TICK × JPS_DT = 1.0 second per simulation tick.
JPS_DT            = 0.1   # JuPedSim internal timestep in seconds
JPS_STEPS_PER_TICK = 10   # 10 × 0.1s = 1 second per tick


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def cell_to_xy(row, col):
    """
    Convert a grid cell (row, col) to its continuous-space centre position.
    Returns (x, y) in metres.
    """
    return (
        col * CELL_SIZE + CELL_SIZE / 2.0,
        row * CELL_SIZE + CELL_SIZE / 2.0,
    )


def build_jupedsim_geometry(grid):
    """
    Convert the warehouse grid into a JuPedSim walkable geometry.

    In JuPedSim 1.x the geometry IS a shapely polygon — no GeometryBuilder
    needed. We pass the polygon directly to jps.Simulation(geometry=...).

    Strategy:
      1. Build one 1×1 square for every walkable cell (floor + depot).
      2. Merge all adjacent squares into one polygon using shapely.
         Adjacent cells in the same column merge into tall rectangles,
         making the geometry efficient.
      3. Apply a morphological closing (expand then shrink by CORNER_BUFFER)
         to round off sharp concave corners at every shelf end.

         WHY: unary_union produces 90-degree inward notches at shelf corners.
         JuPedSim's CFSM boundary avoidance zeroes the velocity component
         pointing into each wall face. At a sharp concave corner two wall
         faces meet at a point — the CFSM zeroes BOTH velocity components,
         freezing any agent that reaches the corner. When two agents approach
         the same corner from opposite directions they deadlock: each is
         frozen at the corner and neither backs away because CFSM has no
         give-way rule.

         buffer(+ε).buffer(-ε)  fills in concave notches smaller than ε,
         replacing the zero-width corner point with a tiny arc. The wall
         repulsion now has a single consistent outward direction so agents
         navigate around corners without stalling. Aisle widths shrink by
         at most 2ε (0.1 m total) — negligible for 2 m aisles.

    Returns a shapely Polygon (or MultiPolygon) representing the walkable
    floor area. Shelf and item cells are NOT included — JuPedSim treats
    anything outside this polygon as a wall.
    """
    # Corner-rounding buffer radius (metres).
    # 0.10 m creates a gentle arc at every shelf-end corner without
    # meaningfully changing aisle widths (narrows entrances by 0.20 m max).
    # This addresses true CFSM zero-width corner freezes.
    # It does NOT fix top-corridor throughput congestion — that is a genuine
    # physical constraint (6 agents sharing a 2 m staging zone) which the RL
    # should observe and learn to reduce by adjusting shelf_start_row / depots.
    CORNER_BUFFER = 0.10

    walkable_squares = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.is_walkable(r, c):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                walkable_squares.append(Polygon([
                    (x0,             y0),
                    (x0 + CELL_SIZE, y0),
                    (x0 + CELL_SIZE, y0 + CELL_SIZE),
                    (x0,             y0 + CELL_SIZE),
                ]))

    # Merge adjacent walkable squares into one polygon
    merged = unary_union(walkable_squares)

    # Morphological closing: expand then shrink by CORNER_BUFFER.
    # This fills in sharp concave notches (shelf-end corners) while leaving
    # the bulk of the walkable area unchanged.
    # resolution=16 gives a smooth 16-segment arc per quarter-circle.
    rounded = merged.buffer(+CORNER_BUFFER, resolution=16) \
                    .buffer(-CORNER_BUFFER, resolution=16)

    return rounded


def find_pickup_xy(grid, shelf_row, shelf_col):
    """
    Return the (x, y) position an agent should walk to when picking
    an item from (shelf_row, shelf_col).

    Only the LEFT and RIGHT neighbours are considered (aisle-facing sides).
    Agents approach shelves from the aisle beside them — exactly as in a
    real warehouse where workers walk along the aisle and pick from the
    shelf face to their side.

    Top/bottom neighbours are intentionally excluded because:
      - In the middle of the shelf zone top/bottom cells are other shelf
        cells anyway (unreachable), so this changes nothing for them.
      - For the FIRST shelf row (shelf_start_row) the top neighbour is the
        top corridor — right at the narrow aisle-mouth corner where agents
        transitioning between corridor and aisle already compete. Routing
        agents there concentrates traffic at the worst chokepoint.
      - For the LAST shelf row (shelf_end_row) the bottom neighbour is the
        bottom corridor (often only 1 m wide) — sending agents there causes
        congestion in the narrowest part of the warehouse.
      - Picking from the side (aisle) is also physically realistic: shelf
        racks are accessed from the aisle face, not from above or below.

    Every shelf column in the default layout has exactly one walkable side
    neighbour (the aisle column beside it), so this function always returns
    a valid position and never returns None for a correctly built grid.
    """
    for dc in (-1, +1):                        # left first, then right
        c = shelf_col + dc
        if grid.is_walkable(shelf_row, c):
            return cell_to_xy(shelf_row, c)
    return None                                # should not happen in a valid layout


# =============================================================================
# JupedSimAgent — all workers managed in one object
# =============================================================================

class JupedSimAgent:
    """
    Manages all warehouse workers using JuPedSim for physical movement.

    JuPedSim handles: continuous position updates, collision avoidance,
                      speed adjustment in crowded areas.
    Python handles:   job queue, state machine, item picking, KPIs.

    Parameters
    ----------
    n_agents : int   — number of workers
    grid     : Grid  — the warehouse grid (shelves, depot, walkability)
    """

    def __init__(self, n_agents, grid):
        self.n    = n_agents
        self.grid = grid

        # ── Job tracking ──────────────────────────────────────────────────
        self.job_queue    = 0   # jobs currently in queue
        self.total_orders = 0   # total jobs that have ever arrived

        # ── Build JuPedSim simulation ─────────────────────────────────────
        # build_jupedsim_geometry() returns a shapely polygon.
        # In JuPedSim 1.x, pass it directly as the geometry argument.
        # dt=JPS_DT: small timestep for accurate physics.
        # We call sim.iterate() JPS_STEPS_PER_TICK times per simulation tick
        # so that each tick represents exactly 1 second.
        #
        # MODEL CHOICE — CollisionFreeSpeedModel (Tordeux 2016).
        # Stable, well-validated, mathematically guaranteed collision-free.
        # AnticipationVelocityModel (Xu 2021) was trialled but over-reacted
        # in our scenario: with all 6 agents starting on a single 1×1 depot
        # cell, AVM's 1-second look-ahead made every agent see the others
        # blocking every possible exit and refuse to move. The combination
        # of CFSM + heterogeneous desired_speeds (see SPEED_STD) gives
        # realistic flow without that pathology.
        walkable_polygon = build_jupedsim_geometry(grid)
        self.sim = jps.Simulation(
            model=jps.CollisionFreeSpeedModel(),
            geometry=walkable_polygon,
            dt=JPS_DT,
            trajectory_writer=None,
        )

        # ── Per-agent state ───────────────────────────────────────────────
        # All depot positions (one or more)
        self.depots_xy = [cell_to_xy(r, c) for r, c in grid.depots]
        self.depot_xy  = self.depots_xy[0]   # backward compat: first depot

        self.state        = [STATE_WAITING] * n_agents
        self.targets      = [None]          * n_agents  # (row, col) of target shelf
        # Assign each agent to a depot (round-robin across depots)
        self.dest_xy      = [self.depots_xy[i % len(self.depots_xy)]
                             for i in range(n_agents)]
        self.pickup_ticks = [0]             * n_agents
        self.rest_ticks   = [0]             * n_agents
        self.idle_ticks   = [0]             * n_agents

        # ── Metrics ───────────────────────────────────────────────────────
        self.distance         = [0.0] * n_agents   # total metres walked
        self.orders_completed = np.zeros(n_agents, dtype=np.int32)  # numpy for .sum() compat

        # Congestion tracking:
        #   _walk_ticks[i] = ticks agent i spent walking (TO_ITEM or TO_DEPOT)
        #   _slow_ticks[i] = of those, ticks where agent moved < 50% free speed
        #                    AND another agent was within CONGESTION_RADIUS (0.9m)
        #   _curr_speed[i] = speed (m/tick) computed in the last step() call
        #                    used by main.py for per-cell congestion heatmap
        self._walk_ticks  = [0]   * n_agents
        self._slow_ticks  = [0]   * n_agents
        self._curr_speed  = [0.0] * n_agents

        # Anti-stuck tracking (see step() for the "step aside" reflex):
        #   _stuck_ticks[i]  = consecutive ticks moving below STUCK_SPEED
        #   _sidestep_xy[i]  = current side-step waypoint, or None if not active
        self._stuck_ticks = [0]    * n_agents
        self._sidestep_xy = [None] * n_agents

        # ── Pre-build journey cache ───────────────────────────────────────
        # PERFORMANCE: instead of creating a new JuPedSim journey every trip,
        # pre-build one journey for every possible destination at init time.
        # _set_destination() then just calls switch_agent_journey() (1 C++ call)
        # instead of add_waypoint_stage() + add_journey() + switch (3 C++ calls)
        # that would accumulate ~1,200 dead journey objects over a full shift.
        self._journey_cache = {}  # (x, y) → (journey_id, stage_id)

        def _cache_destination(xy):
            if xy not in self._journey_cache:
                sid = self.sim.add_waypoint_stage(xy, ARRIVAL_THRESHOLD)
                jid = self.sim.add_journey(jps.JourneyDescription([sid]))
                self._journey_cache[xy] = (jid, sid)

        # Cache all depot positions as destinations
        for _dxy in self.depots_xy:
            _cache_destination(_dxy)

        # Pre-cache pickup positions for every shelf/item cell
        for r in range(grid.rows):
            for c in range(grid.cols):
                if not grid.is_walkable(r, c):          # shelf or item cell
                    pickup = find_pickup_xy(grid, r, c)
                    if pickup is not None:
                        _cache_destination(pickup)

        # ── Pre-cache EVERY walkable cell as a possible side-step target ──
        # Used by the anti-stuck mechanism in step(): when an agent has been
        # frozen for too many ticks, we redirect them to a nearby cached
        # waypoint to break the deadlock — mimicking a real worker stepping
        # aside. Pre-caching all walkable cells gives the mechanism a dense
        # network of fall-back positions throughout the warehouse.
        for r in range(grid.rows):
            for c in range(grid.cols):
                if grid.is_walkable(r, c):
                    _cache_destination(cell_to_xy(r, c))

        # ── Per-agent desired speeds (heterogeneous to prevent CFSM freezes) ─
        # See SPEED_STD docstring above for the full rationale.
        # Drawn once at construction time and held constant for the shift, so
        # each "worker" has a stable individual pace (a fast worker stays fast).
        # Clamp to [0.6, 1.8] m/s to keep all agents within human-walking range.
        self._desired_speeds = [
            max(0.6, min(1.8, random.gauss(FREE_SPEED, SPEED_STD)))
            for _ in range(n_agents)
        ]

        # ── Add all agents to JuPedSim, distributed across depots ────────
        self.jps_ids = []
        _margin = AGENT_RADIUS + 0.05
        _min_x  = _margin
        _max_x  = grid.cols * CELL_SIZE - _margin

        num_depots = len(self.depots_xy)
        # Count how many agents are assigned to each depot (for spread calc)
        agents_per_depot = [(n_agents + num_depots - 1 - d) // num_depots
                            for d in range(num_depots)]

        depot_slot_counters = [0] * num_depots  # tracks spread slot per depot

        for i in range(n_agents):
            depot_idx = i % num_depots
            slot      = depot_slot_counters[depot_idx]
            depot_slot_counters[depot_idx] += 1

            dxy = self.depots_xy[depot_idx]
            depot_x, depot_y = dxy
            djid, dsid = self._journey_cache[dxy]

            n_at_depot = agents_per_depot[depot_idx]
            # Spread agents horizontally around their assigned depot
            start_x = depot_x + (slot - n_at_depot / 2.0) * AGENT_RADIUS * 2.5
            start_x = max(_min_x, min(_max_x, start_x))

            jps_id = self.sim.add_agent(
                jps.CollisionFreeSpeedModelAgentParameters(
                    journey_id=djid,
                    stage_id=dsid,
                    position=(start_x, depot_y),
                    desired_speed=self._desired_speeds[i],   # heterogeneous!
                    radius=AGENT_RADIUS,
                )
            )
            self.jps_ids.append(jps_id)

        # Current positions cache — refreshed once per step(), reused everywhere
        # to avoid repeated Python↔C++ boundary crossings for the same tick.
        self._curr_xy = [self.sim.agent(jid).position for jid in self.jps_ids]
        self._prev_xy = list(self._curr_xy)

    # =========================================================================
    # Public interface (mirrors agent.py)
    # =========================================================================

    def add_job(self):
        """Add one job to the queue (called by simulation loop each tick)."""
        self.job_queue    += 1
        self.total_orders += 1

    def is_done(self, i):
        """Return True if agent i has finished all its jobs."""
        return self.state[i] == STATE_ALL_DONE

    def all_done(self):
        """Return True when every agent has finished."""
        return all(s == STATE_ALL_DONE for s in self.state)

    # =========================================================================
    # Compatibility properties — allow main.py to use the same attribute
    # access patterns as Sim_V2's Agent class.
    # =========================================================================

    @property
    def pos_row(self):
        """Row index for each agent — for pygame drawing."""
        return [self.get_row_col(i)[0] for i in range(self.n)]

    @property
    def pos_col(self):
        """Column index for each agent — for pygame drawing."""
        return [self.get_row_col(i)[1] for i in range(self.n)]

    @property
    def slow_ticks(self):
        """
        Per-agent congestion ticks — replaces blocked_events from Sim_V2.
        Counts ticks each agent spent walking at < 20% of free speed,
        indicating physical slowing due to nearby agents.
        Returns a numpy array for consistency with other per-agent metrics.
        """
        return np.array(self._slow_ticks, dtype=np.int32)

    @property
    def walk_ticks(self):
        """Per-agent total walking ticks (TO_ITEM or TO_DEPOT state)."""
        return np.array(self._walk_ticks, dtype=np.int32)

    def get_xy(self, i):
        """
        Return continuous (x, y) position of agent i in metres.
        x = horizontal (col direction), y = vertical (row direction).
        """
        return self.sim.agent(self.jps_ids[i]).position

    def get_row_col(self, i):
        """
        Return the grid cell (row, col) the agent is currently in.
        Converts continuous (x, y) back to integer cell indices.
        Used by visualisation code that draws agents on the grid.
        """
        x, y = self.get_xy(i)
        col = int(x / CELL_SIZE)
        row = int(y / CELL_SIZE)
        # Clamp to valid grid bounds
        row = max(0, min(self.grid.rows - 1, row))
        col = max(0, min(self.grid.cols - 1, col))
        return row, col

    def get_speed(self, i):
        """
        Return the speed of agent i (m/s) computed during the last step().
        Uses the cached _curr_speed value to avoid extra C++ boundary crossings.
        Returns 0.0 before the first step() call.
        """
        return self._curr_speed[i]

    # =========================================================================
    # step() — advance all agents by one simulation tick (1 second)
    # =========================================================================

    def step(self):
        """
        Advance the simulation by one tick.

        1. JuPedSim physics  — moves all agents, handles collision avoidance
        2. Metrics update    — distance walked, congestion detection
        3. State transitions — arrived at shelf/depot, picking countdown
        4. Job assignment    — give waiting agents their next job
        5. Anti-stuck reflex — step aside if frozen for >STUCK_TICKS
        """

        # ── 1. Advance JuPedSim (moves all agents, handles avoidance) ─────
        # JPS_STEPS_PER_TICK small steps (each JPS_DT seconds) = 1 second total.
        self.sim.iterate(JPS_STEPS_PER_TICK)

        # ── Refresh position cache (one C++ call per agent, reused below) ─
        for i, jid in enumerate(self.jps_ids):
            self._curr_xy[i] = self.sim.agent(jid).position

        # ── 2. Update distance and congestion metrics ─────────────────────
        for i in range(self.n):
            cx, cy = self._curr_xy[i]
            px, py = self._prev_xy[i]

            # Distance walked this tick (Euclidean, in metres)
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            self.distance[i]   += d
            self._prev_xy[i]    = (cx, cy)   # roll forward for next tick
            self._curr_speed[i] = d          # speed in m/tick (= m/s at 1 tick/s)

            # Congestion: agent is walking but slowed by another nearby agent.
            # Two conditions must both be true:
            #   1. Speed < CONGESTION_SPEED (50% free speed = 0.6 m/s)
            #      — agent is moving significantly below its unimpeded pace
            #   2. Another agent is within CONGESTION_RADIUS (0.9m centre-to-centre)
            #      — a nearby agent is the physical cause of the slowdown
            # Together these exclude natural waypoint deceleration (agent alone
            # near shelf/depot) and only count genuine agent-agent conflicts.
            if self.state[i] in (STATE_TO_ITEM, STATE_TO_DEPOT):
                self._walk_ticks[i] += 1
                if d < CONGESTION_SPEED:
                    for j in range(self.n):
                        if j != i:
                            cx_j, cy_j = self._curr_xy[j]
                            if math.sqrt((cx - cx_j) ** 2 + (cy - cy_j) ** 2) < CONGESTION_RADIUS:
                                self._slow_ticks[i] += 1
                                break

        # ── 3. State machine transitions ──────────────────────────────────
        for i in range(self.n):
            s = self.state[i]

            # --- Walking to shelf ---
            if s == STATE_TO_ITEM:
                cx, cy = self._curr_xy[i]           # reuse cached position
                dx, dy = self.dest_xy[i]
                dist_to_dest = math.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)
                if dist_to_dest < ARRIVAL_THRESHOLD:
                    # Arrived next to shelf — start picking
                    self.pickup_ticks[i] = PICKUP_TICKS
                    self.state[i]        = STATE_PICKING

            # --- Picking up item ---
            elif s == STATE_PICKING:
                self.pickup_ticks[i] -= 1
                if self.pickup_ticks[i] <= 0:
                    # Item collected — deplete from grid and head back
                    r, c = self.targets[i]
                    self.grid.deplete_item(r, c)
                    self.orders_completed[i] += 1   # numpy int32 supports +=
                    self._go_to_depot(i)

            # --- Walking back to depot ---
            elif s == STATE_TO_DEPOT:
                cx, cy = self._curr_xy[i]           # reuse cached position
                dx, dy = self.dest_xy[i]            # dest already set to nearest depot
                dist_to_depot = math.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)
                if dist_to_depot < ARRIVAL_THRESHOLD:
                    # Back at depot — short rest before next job
                    self.rest_ticks[i] = REST_TICKS
                    self.state[i]      = STATE_RESTING

            # --- Resting at depot ---
            elif s == STATE_RESTING:
                self.rest_ticks[i] -= 1
                if self.rest_ticks[i] <= 0:
                    self.state[i] = STATE_WAITING

            # --- Waiting for a job ---
            elif s == STATE_WAITING:
                self.idle_ticks[i] += 1

        # ── 4. Assign queued jobs to waiting agents ────────────────────────
        for i in range(self.n):
            if self.state[i] == STATE_WAITING and self.job_queue > 0:
                target = self._pick_next_item()
                if target is not None:
                    self.job_queue  -= 1
                    self.targets[i]  = target
                    self._go_to_item(i)

        # ── 5. Anti-stuck "step aside" reflex ─────────────────────────────
        # CFSM can produce persistent slow-downs when multiple agents converge
        # in a narrow aisle (e.g. 3 agents approaching the same aisle entrance
        # from different directions).  In reality a worker would step aside
        # within a few seconds rather than freeze for minutes.  We simulate
        # that: any agent below STUCK_SPEED for STUCK_TICKS in a row gets
        # redirected to a nearby cached waypoint as a "side spot", then
        # restored to its real destination once it gets there.
        for i in range(self.n):
            # Only walking agents can be "stuck" — picking / resting / waiting
            # agents are deliberately stationary, not frozen.
            if self.state[i] not in (STATE_TO_ITEM, STATE_TO_DEPOT):
                self._stuck_ticks[i] = 0
                self._sidestep_xy[i] = None
                continue

            # Already doing a side-step? Check if reached, then restore real dest.
            if self._sidestep_xy[i] is not None:
                cx, cy   = self._curr_xy[i]
                sx, sy   = self._sidestep_xy[i]
                if math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2) < SIDESTEP_REACHED_TOL:
                    # Arrived at the side spot — head back to the real destination
                    self._set_destination(i, self.dest_xy[i])
                    self._sidestep_xy[i] = None
                self._stuck_ticks[i] = 0   # don't accumulate during a sidestep
                continue

            # Normal walking — check if frozen
            if self._curr_speed[i] < STUCK_SPEED:
                self._stuck_ticks[i] += 1
                if self._stuck_ticks[i] >= STUCK_TICKS:
                    self._try_sidestep(i)
            else:
                self._stuck_ticks[i] = 0

    # =========================================================================
    # Navigation
    # =========================================================================

    def _try_sidestep(self, i):
        """
        Redirect a frozen agent to a nearby cached waypoint so it can break
        out of a CFSM deadlock.  Picks a random walkable cell between
        SIDESTEP_MIN_DIST and SIDESTEP_MAX_DIST away — the randomness is
        deliberate: if two agents both go stuck against each other, they pick
        different side spots and the symmetric stalemate is broken.

        The original destination in self.dest_xy[i] is NOT touched, so once
        the side spot is reached the agent resumes its real journey.
        """
        cx, cy = self._curr_xy[i]
        candidates = []
        for wxy in self._journey_cache:
            wx, wy = wxy
            d = math.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
            if SIDESTEP_MIN_DIST < d < SIDESTEP_MAX_DIST:
                candidates.append(wxy)

        if not candidates:
            return

        sidestep_xy = random.choice(candidates)
        self._set_destination(i, sidestep_xy)
        self._sidestep_xy[i] = sidestep_xy
        self._stuck_ticks[i] = 0

    def _pick_next_item(self):
        """Randomly select an available item from the grid."""
        available = self.grid.get_all_item_positions()
        return random.choice(available) if available else None

    def _go_to_item(self, i):
        """Send agent i to the pickup position next to its target shelf."""
        r, c = self.targets[i]
        dest = find_pickup_xy(self.grid, r, c)
        if dest is None:
            return   # no accessible neighbour — rare, skip
        self.dest_xy[i] = dest
        self.state[i]   = STATE_TO_ITEM
        self._set_destination(i, dest)

    def _nearest_depot_xy(self, i):
        """Return the (x, y) of the depot closest to agent i's current position."""
        cx, cy = self._curr_xy[i]
        return min(self.depots_xy,
                   key=lambda d: (cx - d[0]) ** 2 + (cy - d[1]) ** 2)

    def _go_to_depot(self, i):
        """Send agent i back to the nearest depot."""
        nearest = self._nearest_depot_xy(i)
        self.dest_xy[i] = nearest
        self.state[i]   = STATE_TO_DEPOT
        self._set_destination(i, nearest)

    def _set_destination(self, i, dest_xy):
        """
        Redirect agent i to dest_xy using the pre-built journey cache.

        PERFORMANCE: all possible destinations were pre-registered at init time
        in self._journey_cache.  No new JuPedSim objects are created here —
        we just call switch_agent_journey() (one C++ call) to redirect the agent.
        """
        journey_id, stage_id = self._journey_cache[dest_xy]
        self.sim.switch_agent_journey(self.jps_ids[i], journey_id, stage_id)

    # =========================================================================
    # KPI collection — compatible with rl_env.py's expected format
    # =========================================================================

    def congestion_rate(self):
        """
        Agent-conflict congestion rate: fraction of walking ticks where an agent
        was both (a) moving below 50% of free speed (< 0.6 m/s) AND (b) had
        another agent within 0.9m centre-to-centre (CONGESTION_RADIUS).

        Condition (a) alone would falsely count natural JuPedSim deceleration as
        agents approach their destination. Condition (b) alone would fire even
        when two agents pass each other freely at full speed. Both together isolate
        genuine agent-agent conflicts: one agent is physically forcing another to slow.

        Range 0.0 (no conflicts) to 1.0 (always conflicted while walking).
        """
        total_walk = sum(self._walk_ticks)
        if total_walk == 0:
            return 0.0
        return sum(self._slow_ticks) / total_walk

    def collect_raw(self):
        """
        Return raw metrics in the format expected by rl_env.py.
        Matches the dict returned by MetricsTracker.collect_raw() in Sim_V2.
        """
        return {
            "total_distance":  sum(self.distance),
            "jobs_completed":  int(self.orders_completed.sum()),
            "jobs_arrived":    self.total_orders,
            "cell_conflicts":  sum(self._slow_ticks),   # repurposed: slow-ticks
        }


# =============================================================================
# QUICK SMOKE TEST — run directly to verify JuPedSim works with the grid
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("=" * 55)
    print("  jupedsim_agent.py — smoke test")
    print("=" * 55)

    # Build warehouse with shelf_start_row=3 to give agents a 3m staging area
    # near the depot. Without this, row 0 is only 1m wide — 4 agents with 0.6m
    # body width deadlock trying to pass each other in the top corridor.
    # shelf_start_row is a key layout parameter the RL will optimise.
    from grid import Grid
    grid = Grid(rows=25, cols=35, shelf_start_row=3)
    print(f"  Grid built: {grid.rows}×{grid.cols}, "
          f"depot at {grid.depot}, "
          f"{len(grid.get_all_item_positions())} items  "
          f"(shelf_start_row=3)")

    # Build geometry (smoke-test it independently before creating agents)
    print("  Building JuPedSim geometry...")
    geom = build_jupedsim_geometry(grid)
    print(f"  Geometry built OK  (shapely type: {geom.geom_type})")

    # Create agents (this also builds the JuPedSim Simulation internally)
    print("  Creating 4 JuPedSim agents...")
    agent = JupedSimAgent(n_agents=4, grid=grid)
    print("  Agents created OK")

    # Show starting positions
    print(f"  Agent starting positions:")
    for i, jid in enumerate(agent.jps_ids):
        print(f"    Agent {i+1}: {agent.sim.agent(jid).position}")

    # Run 500 ticks with periodic position snapshots
    print("\n  Running 500 ticks (snapshots every 100 ticks)...")
    for tick in range(500):
        if nhpp_arrival(tick):
            agent.add_job()
        agent.step()
        grid.tick_replenishment()

        # Print snapshot every 100 ticks
        if (tick + 1) % 100 == 0:
            positions = [agent.sim.agent(jid).position for jid in agent.jps_ids]
            states    = agent.state
            state_names = {0:"wait",1:"→item",2:"pick",3:"→depot",4:"rest",5:"done"}
            print(f"\n  Tick {tick+1:4d}:")
            for i in range(agent.n):
                x, y = positions[i]
                print(f"    Agent {i+1}: ({x:.1f},{y:.1f})  "
                      f"state={state_names[states[i]]:7s}  "
                      f"jobs={agent.orders_completed[i]}  "
                      f"dist={agent.distance[i]:.1f}m")

    raw = agent.collect_raw()
    print(f"\n  After 500 ticks:")
    print(f"    Jobs arrived   : {raw['jobs_arrived']}")
    print(f"    Jobs completed : {raw['jobs_completed']}")
    print(f"    Total distance : {raw['total_distance']:.1f} m")
    print(f"    Congestion rate: {agent.congestion_rate():.4f}")
    print("\n  Smoke test PASSED")
    print("=" * 55)
