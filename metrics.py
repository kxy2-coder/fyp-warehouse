# =============================================================================
# metrics.py — Simulation Metrics
# =============================================================================
# Owns ALL metric state and logic for the warehouse simulation.
# Works with any number of agents (N agents).
#
# Current metrics:
#   - cell_conflicts : ticks where two or more agents share the same non-depot cell
# =============================================================================

from agent import BLOCKED_WAIT_TICKS

FLASH_FRAMES = 12   # how many display frames the red conflict highlight lasts


class MetricsTracker:
    """
    Tracks all simulation metrics and provides helper methods for
    logging and displaying them.
    """

    def __init__(self):
        self.cell_conflicts = 0
        self.conflict_cell  = None
        self.flash_timer    = 0

    # -------------------------------------------------------------------------

    def update(self, agents, depot):
        """
        Check all metric conditions and update counters.
        Call after all agents have stepped each tick.

        agents — list of Agent objects
        depot  — (row, col) of the depot cell
        """
        self._check_cell_conflict(agents, depot)

    def _check_cell_conflict(self, agents, depot):
        """Detect and record cell-sharing conflicts (depot excluded)."""
        active = [a for a in agents if not a.is_done()]
        # Build a dict: position -> list of agents there
        pos_map = {}
        for agent in active:
            pos_map.setdefault(agent.pos, []).append(agent)

        for pos, occupants in pos_map.items():
            if len(occupants) >= 2 and pos != depot:
                self.cell_conflicts += 1
                self.conflict_cell   = pos
                self.flash_timer     = FLASH_FRAMES
                break   # count once per tick even if multiple conflicts

    # -------------------------------------------------------------------------

    def tick_flash(self):
        """Decrement conflict highlight timer. Call once per display frame."""
        if self.flash_timer > 0:
            self.flash_timer -= 1

    # -------------------------------------------------------------------------

    def print_step(self, agents):
        """Print a one-line terminal summary per simulation tick."""
        parts = []
        for agent in agents:
            parts.append(
                f"A{agent.agent_id}[{agent.orders_completed:2d}/{agent.total_orders}]"
                f" {agent.state:12s} fat={agent.fatigue:.2f}"
            )
        print("  " + " | ".join(parts) + f" | conflicts={self.cell_conflicts}")

    def print_summary(self, agents):
        """Print final results when all agents finish."""
        print("=" * 55)
        print("  SIMULATION COMPLETE")
        for agent in agents:
            exp_label = "Experienced" if agent.experience_B >= 100 else "Novice"
            print(f"  Agent {agent.agent_id} ({exp_label}) :")
            print(f"    Distance     : {agent.distance} cells")
            print(f"    Work time    : {agent.work_time:.4f} hrs")
            print(f"    Final fatigue: {agent.fatigue:.3f}")
            print(f"    Blocked      : {agent.blocked_count}x ({agent.blocked_count * BLOCKED_WAIT_TICKS}s waited)")
        print(f"  Cell conflicts : {self.cell_conflicts}")
        print("=" * 55)
    def collect_raw(self, agents):
        return {
            "cell_conflicts":    self.cell_conflicts,
            "total_distance":    sum(a.distance for a in agents),
            "total_work_time":   sum(a.work_time for a in agents),
            "total_orders":      sum(a.orders_completed for a in agents),
            "total_blocked":     sum(a.blocked_count for a in agents),
            "avg_final_fatigue": sum(a.fatigue for a in agents) / len(agents),
            "job_quota":         agents[0].total_orders,
        }