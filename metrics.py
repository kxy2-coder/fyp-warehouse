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

from agent import TOTAL_ORDERS, BLOCKED_WAIT_TICKS

FLASH_FRAMES = 12   # how many display frames the red conflict highlight lasts


class MetricsTracker:
    """
    Tracks all simulation metrics and provides helper methods for
    logging and displaying them.
    """

    def __init__(self):
        # ── Cell conflict metric ──────────────────────────────────────────────
        self.cell_conflicts = 0
        self.conflict_cell  = None
        self.flash_timer    = 0

    # -------------------------------------------------------------------------

    def update(self, agent1, agent2, depot):
        """
        Check all metric conditions and update counters.
        Call after agent1.step() and agent2.step() each tick.
        """
        self._check_cell_conflict(agent1, agent2, depot)

    def _check_cell_conflict(self, agent1, agent2, depot):
        """Detect and record cell-sharing conflicts (depot excluded)."""
        if agent1.is_done() or agent2.is_done():
            return
        if agent1.pos == agent2.pos and agent1.pos != depot:
            self.cell_conflicts += 1
            self.conflict_cell   = agent1.pos
            self.flash_timer     = FLASH_FRAMES

    # -------------------------------------------------------------------------

    def tick_flash(self):
        """Decrement conflict highlight timer. Call once per display frame."""
        if self.flash_timer > 0:
            self.flash_timer -= 1

    # -------------------------------------------------------------------------

    def print_step(self, agent1, agent2):
        """Print a one-line terminal summary per simulation tick."""
        print(
            f"  A1[{agent1.orders_completed:2d}/{TOTAL_ORDERS}]"
            f" {agent1.state:12s} fat={agent1.fatigue:.2f}"
            f" | A2[{agent2.orders_completed:2d}/{TOTAL_ORDERS}]"
            f" {agent2.state:12s} fat={agent2.fatigue:.2f}"
            f" | conflicts={self.cell_conflicts}"
        )

    def print_summary(self, agent1, agent2):
        """Print final results when both agents finish."""
        print("=" * 55)
        print("  SIMULATION COMPLETE")
        print(f"  Agent 1 (experienced) :")
        print(f"    Distance   : {agent1.distance} cells")
        print(f"    Work time  : {agent1.work_time:.4f} hrs")
        print(f"    Final fatigue: {agent1.fatigue:.3f}")
        print(f"  Agent 2 (novice) :")
        print(f"    Distance   : {agent2.distance} cells")
        print(f"    Work time  : {agent2.work_time:.4f} hrs")
        print(f"    Final fatigue: {agent2.fatigue:.3f}")
        print(f"  Cell conflicts : {self.cell_conflicts}")
        print(f"  Agent 1 blocked: {agent1.blocked_count}x ({agent1.blocked_count * BLOCKED_WAIT_TICKS}s waited)")
        print(f"  Agent 2 blocked: {agent2.blocked_count}x ({agent2.blocked_count * BLOCKED_WAIT_TICKS}s waited)")
        print("=" * 55)
