# =============================================================================
# metrics.py — Simulation Metrics
# =============================================================================
# Owns ALL metric state and logic for the warehouse simulation.
# To add a new metric in the future:
#   1. Add its counter/state variables in __init__
#   2. Update it inside update() each simulation tick
#   3. Expose it as a property or attribute for draw_panel / print_summary
#
# Current metrics:
#   - cell_conflicts : steps where both agents share the same non-depot cell
# =============================================================================

from agent import TOTAL_ORDERS

FLASH_FRAMES = 12   # how many display frames the red conflict highlight lasts


class MetricsTracker:
    """
    Tracks all simulation metrics and provides helper methods for
    logging and displaying them.
    """

    def __init__(self):
        # ── Cell conflict metric ──────────────────────────────────────────────
        # Incremented each simulation step where both agents occupy the same
        # cell (depot excluded — agents always start/end there together).
        self.cell_conflicts = 0

        # Visual flash state — used by draw_grid to highlight conflict cells
        self.conflict_cell  = None   # (row, col) of the most recent conflict
        self.flash_timer    = 0      # frames remaining for the red highlight

        # ── Add future metrics below ──────────────────────────────────────────
        # Example:
        #   self.total_wait_steps = 0   # steps an agent spent waiting

    # -------------------------------------------------------------------------
    # Called once per simulation tick (after both agents have stepped)
    # -------------------------------------------------------------------------

    def update(self, agent1, agent2, depot):
        """
        Check all metric conditions and update counters.
        Call this every simulation tick, after agent1.step() and agent2.step().

        agent1, agent2 — Agent objects
        depot          — (row, col) of the depot cell (excluded from conflicts)
        """
        self._check_cell_conflict(agent1, agent2, depot)

        # ── Add future metric updates here ────────────────────────────────────

    def _check_cell_conflict(self, agent1, agent2, depot):
        """Detect and record cell-sharing conflicts."""
        if agent1.is_done() or agent2.is_done():
            return
        if agent1.pos == agent2.pos and agent1.pos != depot:
            self.cell_conflicts += 1
            self.conflict_cell   = agent1.pos
            self.flash_timer     = FLASH_FRAMES

    # -------------------------------------------------------------------------
    # Called once per display frame (to animate the conflict flash)
    # -------------------------------------------------------------------------

    def tick_flash(self):
        """Decrement the conflict highlight timer. Call once per frame."""
        if self.flash_timer > 0:
            self.flash_timer -= 1

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def print_step(self, agent1, agent2):
        """Print a one-line summary of the current simulation step."""
        print(
            f"  A1[{agent1.orders_completed:2d}/{TOTAL_ORDERS}] "
            f"{agent1.state:15s} {str(agent1.pos):12s} | "
            f"A2[{agent2.orders_completed:2d}/{TOTAL_ORDERS}] "
            f"{agent2.state:15s} {str(agent2.pos):12s} | "
            f"Conflicts: {self.cell_conflicts}"
        )

    def print_summary(self, agent1, agent2):
        """Print the final results when the simulation finishes."""
        print("=" * 50)
        print("  SIMULATION COMPLETE")
        print(f"  Agent 1 distance : {agent1.distance} cells")
        print(f"  Agent 2 distance : {agent2.distance} cells")
        print(f"  Cell conflicts   : {self.cell_conflicts}")
        # ── Print future metric summaries here ────────────────────────────────
        print("=" * 50)
