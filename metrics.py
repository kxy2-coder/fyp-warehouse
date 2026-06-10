# =============================================================================
# metrics.py — Metric tracking and per-agent trace logging
# =============================================================================
# Provides two utilities used by main.py and the analysis scripts:
#
#   TraceLogger     — records per-agent KPI snapshots (picks, distance,
#                     idle ticks, walk ticks, slow/congestion ticks) at a
#                     fixed tick interval. Writes the bin-level history to
#                     agent_traces.csv for plot_agent_traces.py to read.
#
#   MetricsTracker  — accumulates totals during a single run (picks,
#                     distance, congestion ticks, replenishment events)
#                     and exposes them to main.py / plot_workload.py.
#
# Both work with the JupedSimAgent (Sim_V3) instance shape; the legacy
# Sim_V2 Agent class is no longer the primary client of this module.
# =============================================================================

import csv
import numpy as np
from agent import BLOCKED_WAIT_TICKS

FLASH_FRAMES = 12   # how many display frames the red conflict highlight lasts


class TraceLogger:
    """
    Records per-agent performance snapshots at fixed tick intervals.
    Call snapshot() every bin_size ticks; call save_csv() at run end.
    """

    def __init__(self, n_agents, bin_size=600):
        self.n_agents = n_agents
        self.bin_size = bin_size
        self.records  = []

    def snapshot(self, tick, agent):
        """Record one dict per agent at this tick."""
        for i in range(self.n_agents):
            self.records.append({
                "tick":       tick,
                "agent_id":   i + 1,
                "picks":      int(agent.orders_completed[i]),
                "distance":   round(float(agent.distance[i]), 1),
                "idle_ticks": int(agent.idle_ticks[i]),
                # slow_ticks / walk_ticks: JuPedSim physics-based congestion.
                # slow_ticks = ticks agent walked at < 20% free speed.
                # walk_ticks = total ticks agent was walking.
                # Congestion rate % = slow_ticks / walk_ticks × 100.
                "slow_ticks": int(agent.slow_ticks[i]),
                "walk_ticks": int(agent.walk_ticks[i]),
            })

    def save_csv(self, filename):
        """Write all recorded snapshots to a CSV file."""
        if not self.records:
            return
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)


class MetricsTracker:
    """
    Tracks all simulation metrics and provides helper methods for
    logging and displaying them.
    """

    def __init__(self):
        self.cell_conflicts = 0
        self.conflict_cell  = None
        self.flash_timer    = 0
        self.spatial_log    = []   # list of (tick, row, col) per blocking event

    # -------------------------------------------------------------------------

    def update(self, agent, depot, tick=None):
        """
        Check all metric conditions and update counters.
        Call after all agents have stepped each tick.

        agent — vectorised Agent object (manages all workers)
        depot — (row, col) of the depot cell
        tick  — current simulation tick (used for spatial logging)
        """
        self._check_cell_conflict(agent, depot, tick)

    def _check_cell_conflict(self, agent, depot, tick=None):
        """Detect and record cell-sharing conflicts (depot excluded)."""
        pos_map = {}
        for i in range(agent.n):
            if not agent.is_done(i):
                pos = (int(agent.pos_row[i]), int(agent.pos_col[i]))
                pos_map.setdefault(pos, []).append(i)

        for pos, occupants in pos_map.items():
            if len(occupants) >= 2 and pos != depot:
                self.cell_conflicts += 1
                self.conflict_cell   = pos
                self.flash_timer     = FLASH_FRAMES
                if tick is not None:
                    self.spatial_log.append((tick, pos[0], pos[1]))

    # -------------------------------------------------------------------------

    def tick_flash(self):
        """Decrement conflict highlight timer. Call once per display frame."""
        if self.flash_timer > 0:
            self.flash_timer -= 1

    # -------------------------------------------------------------------------

    def print_step(self, agent):
        """Print a one-line terminal summary per simulation tick."""
        state_names = {
            0: "waiting",
            1: "to_item",
            2: "picking_up",
            3: "to_depot",
            4: "resting",
            5: "all_done",
        }
        parts = []
        for i in range(agent.n):
            parts.append(
                f"A{i+1}[{agent.orders_completed[i]:2d}/{agent.total_orders}]"
                f" {state_names[int(agent.state[i])]:12s} fat={agent.fatigue[i]:.2f}"
            )
        print("  " + " | ".join(parts) + f" | conflicts={self.cell_conflicts}")

    def print_summary(self, agent):
        """Print final results when all agents finish."""
        print("=" * 55)
        print("  SIMULATION COMPLETE")
        print(f"  Jobs arrived: {agent.total_orders}  |  "
              f"Jobs completed: {int(agent.orders_completed.sum())}")
        for i in range(agent.n):
            exp_label = "Experienced" if agent.experience_B[i] >= 100 else "Novice"
            print(f"  Agent {i+1} ({exp_label}) :")
            print(f"    Distance     : {agent.distance[i]} cells")
            print(f"    Work time    : {agent.work_time[i]:.4f} hrs")
            print(f"    Final fatigue: {agent.fatigue[i]:.3f}")
        print(f"  Total collisions : {self.cell_conflicts} "
              f"(each loser waited {BLOCKED_WAIT_TICKS}s)")
        print("=" * 55)

    def get_heatmap(self, grid_rows, grid_cols):
        """Return a 2D numpy array (grid_rows × grid_cols) with conflict counts."""
        heatmap = np.zeros((grid_rows, grid_cols), dtype=np.int32)
        for _, row, col in self.spatial_log:
            if 0 <= row < grid_rows and 0 <= col < grid_cols:
                heatmap[row, col] += 1
        return heatmap

    def save_spatial_csv(self, filename, grid_rows, grid_cols):
        """
        Write aggregated (row, col, count) to CSV.
        Only cells with at least one conflict are written.
        """
        heatmap = self.get_heatmap(grid_rows, grid_cols)
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row", "col", "count"])
            for r in range(grid_rows):
                for c in range(grid_cols):
                    if heatmap[r, c] > 0:
                        writer.writerow([r, c, int(heatmap[r, c])])

    def collect_raw(self, agent):
        return {
            "cell_conflicts":  self.cell_conflicts,
            "total_distance":  int(agent.distance.sum()),
            "jobs_arrived":    int(agent.total_orders),
            "jobs_completed":  int(agent.orders_completed.sum()),
        }
