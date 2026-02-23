# =============================================================================
# agent.py — The Warehouse Worker
# =============================================================================
# This file defines the agent — a simulated warehouse worker.
#
# Each agent completes TOTAL_ORDERS (10) pickup trips in sequence:
#   1. Stand at the depot (top middle)
#   2. Walk to a randomly chosen item (A* shortest path to adjacent cell)
#   3. Pick up the item (removes it from the shelf)
#   4. Walk back to the depot
#   5. Repeat from step 2 until all orders are done
#
# Two agents can run simultaneously; they ghost through each other
# (no collision avoidance — conflicts are counted externally).
#
# STATES:
#   "waiting"    — at depot, about to start first order
#   "to_item"    — walking toward the current target item
#   "picking_up" — arrived next to item, about to grab it
#   "to_depot"   — walking back to depot
#   "all_done"   — all TOTAL_ORDERS complete, agent stops
# =============================================================================

import random
from pathfinder import find_path_to_neighbour, find_path

TOTAL_ORDERS = 10


class Agent:
    """
    A warehouse worker that completes TOTAL_ORDERS pickup trips one by one.
    """

    def __init__(self, grid, agent_id=1, color=(60, 200, 80)):
        """
        Set up the agent at the depot, ready to start.

        grid     — the Grid object (needed for pathfinding and item removal)
        agent_id — 1 or 2, used for display / terminal logging
        color    — RGB tuple for drawing this agent on screen
        """
        self.grid     = grid
        self.agent_id = agent_id
        self.color    = color

        self.pos   = grid.depot   # start at the depot
        self.state = "waiting"    # hasn't moved yet

        # Current target item — (row, col) of the shelf being collected
        self.target = None

        # Path the agent is following: list of (row, col) steps
        self.path       = []
        self.path_index = 0

        # Metrics
        self.distance        = 0   # total cells walked across all orders
        self.orders_completed = 0  # how many full trips are done

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _pick_next_item(self):
        """
        Randomly choose one available item from the grid right now.
        Because items are removed as they are collected, this always
        returns a still-stocked shelf (or None if the grid is empty).
        """
        available = self.grid.get_all_item_positions()
        if not available:
            return None
        return random.choice(available)

    def _plan_path_to_item(self):
        """
        Calculate the shortest path from current position to the cell
        adjacent to the target (because the agent cannot stand on a shelf).
        """
        self.path       = find_path_to_neighbour(self.grid, self.pos, self.target)
        self.path_index = 0
        self.state      = "to_item"

    def _plan_path_to_depot(self):
        """
        Calculate the shortest path from current position back to the depot.
        """
        self.path       = find_path(self.grid, self.pos, self.grid.depot)
        self.path_index = 0
        self.state      = "to_depot"

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def step(self):
        """
        Advance the agent by ONE cell (or one action).
        Called once per simulation tick by main.py.
        """

        # --- First step: leave the waiting state and start order 1 ---
        if self.state == "waiting":
            self.target = self._pick_next_item()
            if self.target:
                self._plan_path_to_item()
            else:
                self.state = "all_done"   # no items at all (edge case)
            return

        # --- Walking toward the item ---
        if self.state == "to_item":
            if self.path_index < len(self.path):
                self.pos         = self.path[self.path_index]
                self.path_index += 1
                self.distance   += 1
            else:
                # Reached the cell beside the item — ready to pick up
                self.state = "picking_up"

        # --- Picking up the item ---
        elif self.state == "picking_up":
            self.grid.remove_item(*self.target)   # shelf becomes empty rack
            self.orders_completed += 1
            self._plan_path_to_depot()            # head back home

        # --- Walking back to depot ---
        elif self.state == "to_depot":
            if self.path_index < len(self.path):
                self.pos         = self.path[self.path_index]
                self.path_index += 1
                self.distance   += 1
            else:
                # Arrived at depot
                if self.orders_completed < TOTAL_ORDERS:
                    # More orders to do — pick the next item immediately
                    self.target = self._pick_next_item()
                    if self.target:
                        self._plan_path_to_item()
                    else:
                        self.state = "all_done"  # grid exhausted early
                else:
                    self.state = "all_done"      # all 10 orders complete

    def is_done(self):
        """Returns True when the agent has completed all TOTAL_ORDERS trips."""
        return self.state == "all_done"
