# =============================================================================
# grid.py — The Warehouse Layout
# =============================================================================
# This file builds the warehouse as a grid (like a spreadsheet).
# Each cell in the grid is one of these types:
#
#   EMPTY (0) — a floor tile the agent can walk on
#   SHELF (1) — a shelf block the agent CANNOT walk through
#   ITEM  (2) — a shelf that has a item/good on it (all shelves start full)
#   DEPOT (3) — the home base where agents start and return to
#
# The grid is stored as a 2D list: grid[row][col]
# Row 0 is the TOP of the warehouse.
#
# LAYOUT PARAMETERS (all adjustable — no magic numbers):
#
#   rows, cols           — overall warehouse dimensions
#   aisle_width          — walking aisle width between shelf blocks
#   centre_aisle_width   — main vertical centre aisle width (must be odd)
#   depot_row            — which row the depot sits on (default 0 = top)
#   depot_col            — which column the depot sits on (default = centre)
#   shelf_start_row      — first row where shelves appear (default 1)
#   shelf_end_row        — last row where shelves appear (default rows-2)
#   cross_aisle_row      — row of the horizontal cross-aisle cutting through
#                          the shelf zone (default = middle of shelf zone).
#                          Set to None to disable the cross-aisle entirely.
#
# Adjusting shelf_start_row / shelf_end_row create clear staging
# areas near the depot, or compress shelves into a smaller zone — both of
# which directly affect agent travel distance and are key layout variables
# for optimization.
#
# The cross_aisle_row creates a horizontal corridor across the full width
# of the warehouse, allowing agents to cut across without travelling to the
# centre or border aisles. Its position within the shelf zone is a key
# layout variable — moving it closer to the depot shortens return trips
# for nearby shelves but splits the shelf zone unevenly.
# =============================================================================

import random
import numpy as np

# Cell type numbers
EMPTY = 0
SHELF = 1
ITEM  = 2
DEPOT = 3


class Grid:
    """
    Builds and stores the warehouse layout.
    """

    def __init__(self, rows=25, cols=35, aisle_width=2,
                 centre_aisle_width=3, depot_row=None, depot_col=None,
                 depot_cols=None,
                 shelf_start_row=None, shelf_end_row=None,
                 cross_aisle_row=None, cross_aisle_width=2,
                 cross_aisle_enabled=True,
                 replenish_delay=100):
        """
        Create the warehouse grid.

        rows               — how many rows tall (default 25)
        cols               — how many columns wide (default 35, odd recommended)
        aisle_width        — aisle width between shelf blocks in columns (default 2)
        centre_aisle_width — main centre aisle width, must be odd (default 3)
        depot_row          — row the depot sits on (default 0 = top row)
        depot_col          — column the depot sits on (default = centre column)
        shelf_start_row    — first row where shelves appear (default 1)
                             Increase this to create a clear staging area near
                             the depot — agents have open floor to manoeuvre before
                             entering the shelf zone. Useful when depot_row = 0.
        shelf_end_row      — last row where shelves appear (default rows-2)
                             Decrease this to leave clear floor at the bottom.
                             Together with shelf_start_row, controls the vertical
                             extent of the shelf zone — a key layout variable for
                             optimizing average travel distance.
        cross_aisle_row    — row of the horizontal cross-aisle within the shelf
                             zone (default = middle row of the shelf zone).
                             Clears the full row to walkable floor, cutting the
                             shelf zone into two halves. Pass None to disable.
        cross_aisle_width  — how many rows the cross-aisle spans (default 2).
                             A width of 1 (1m) is too narrow for two agents to
                             pass — agents cannot physically pass in a 1m gap.
                             Width of 2 gives a 2m corridor where agents can
                             pass side-by-side without slowing each other.
        """
        # centre_aisle_width can be any positive integer (odd constraint removed)

        centre  = cols // 2
        half_ca = centre_aisle_width // 2
        if centre - half_ca - 2 < 1:
            raise ValueError("Grid too narrow for the given aisle parameters")

        self.rows = rows
        self.cols = cols

        # Layout parameters — shelf width is fixed at 2
        self.shelf_width        = 2
        self.aisle_width        = aisle_width
        self.centre_aisle_width = centre_aisle_width

        # Depot position(s) — defaults to single depot at top-centre
        self.depot_row = depot_row if depot_row is not None else 0

        if depot_cols is not None:
            # Multiple depot columns provided explicitly (from RL multi-depot)
            _active_cols = list(depot_cols)
        else:
            # Single depot column (backward-compatible path)
            _active_cols = [depot_col if depot_col is not None else cols // 2]

        self.depots    = [(self.depot_row, c) for c in _active_cols]
        self.depot     = self.depots[0]          # backward compat: first depot
        self.depot_col = self.depot[1]           # backward compat: first col

        # Shelf zone rows
        # shelf_start_row=3  → 3m staging area at top  (rows 0-2 clear)
        # shelf_end_row=rows-3 → 2m corridor at bottom (last 2 rows clear)
        # This ensures agents can pass each other at both ends of the shelf zone.
        self.shelf_start_row = shelf_start_row if shelf_start_row is not None else 3
        self.shelf_end_row   = shelf_end_row   if shelf_end_row   is not None else rows - 3

        # Cross-aisle width — number of rows cleared for the horizontal corridor
        self.cross_aisle_width = max(1, cross_aisle_width)

        # Clamp to valid range
        self.shelf_start_row = max(1, self.shelf_start_row)
        self.shelf_end_row   = min(rows - 2, self.shelf_end_row)

        if self.shelf_start_row > self.shelf_end_row:
            raise ValueError(
                f"shelf_start_row ({self.shelf_start_row}) must be <= "
                f"shelf_end_row ({self.shelf_end_row})"
            )

        # Cross-aisle row — horizontal corridor cutting through the shelf zone.
        # cross_aisle_enabled=False disables it entirely (self.cross_aisle_row=None).
        # When enabled and cross_aisle_row=None, defaults to middle of shelf zone.
        if not cross_aisle_enabled:
            self.cross_aisle_row = None
        elif cross_aisle_row is None:
            self.cross_aisle_row = (self.shelf_start_row + self.shelf_end_row) // 2
        else:
            # Clamp to shelf zone bounds so it always falls inside the shelves
            self.cross_aisle_row = max(self.shelf_start_row,
                                       min(self.shelf_end_row, cross_aisle_row))

        # Replenishment: delay (in ticks) before a picked cell restocks.
        # Randomised per cell between 80%–150% of this value so cells don't
        # all restock simultaneously.
        self.replenish_delay = replenish_delay

        # Tracks depleted cells: {(row, col): ticks_remaining}
        self._depleted = {}

        # 2D numpy array of cell types — starts all empty
        # dtype int8: each cell stores a small integer (0-3), int8 uses 1 byte
        # per cell vs ~28 bytes for a Python int, so much more memory efficient.
        self.cells = np.full((rows, cols), EMPTY, dtype=np.int8)

        # Shelf reference labels — populated by _label_shelves()
        self.shelf_labels = {}

        # Build layout then assign labels
        self._build()
        self._label_shelves()

    def _build(self):
        """
        Draw the warehouse onto self.cells using all layout parameters.

        Structure:
          depot_row                       — walkway row containing the depot
          shelf_start_row..shelf_end_row  — shelf zone (shelves + aisles)
          all other rows                  — open walkable floor

        Shelves are shelf_width (2) columns wide.
        Aisles between shelf blocks are aisle_width columns wide.
        The centre aisle (centre_aisle_width wide) is always kept clear.
        Border columns (0 and cols-1) are always kept as walkways.
        """
        centre  = self.cols // 2
        half_ca = self.centre_aisle_width // 2
        step    = self.shelf_width + self.aisle_width

        # --- Determine which columns are shelf columns ---
        shelf_cols = set()

        # Left side — expand outward from centre
        pos = centre - half_ca - self.shelf_width
        while pos >= 1:
            shelf_cols.add(pos)
            shelf_cols.add(pos + 1)
            pos -= step

        # Right side — expand outward from centre
        pos = centre + half_ca + 1
        while pos + 1 <= self.cols - 2:
            shelf_cols.add(pos)
            shelf_cols.add(pos + 1)
            pos += step

        self._shelf_cols = shelf_cols

        # --- Fill every cell using numpy operations (no nested loop needed) ---

        # 1. Everything starts as walkable floor (already set in __init__,
        #    but reset here so _build() is self-contained if called again).
        self.cells[:] = EMPTY

        # 2. Place ITEM in every shelf column within the shelf zone.
        #    np.ix_ creates the right index structure for a rectangular region.
        shelf_rows = np.arange(self.shelf_start_row, self.shelf_end_row + 1)
        shelf_cols_arr = np.array(sorted(shelf_cols))
        self.cells[np.ix_(shelf_rows, shelf_cols_arr)] = ITEM

        # 3. Clear the cross-aisle rows (only when enabled).
        if self.cross_aisle_row is not None:
            for _r in range(self.cross_aisle_row,
                            min(self.rows, self.cross_aisle_row + self.cross_aisle_width)):
                self.cells[_r, :] = EMPTY

        # 4. Place all depots last so they are never overwritten.
        for _dr, _dc in self.depots:
            self.cells[_dr, _dc] = DEPOT

        # 5. Precompute a plain Python bool list-of-lists for fast walkability
        #    checks in the A* pathfinder.  Single-element numpy array access has
        #    per-call overhead; Python list indexing is ~5-10x faster and
        #    is_walkable() is called hundreds of thousands of times per run.
        #    self.cells stays as a numpy array for batch operations elsewhere.
        #    This table never needs updating: ITEM→SHELF (remove_item) keeps the
        #    cell non-walkable, and the depot never changes.
        self._walkable = ((self.cells == EMPTY) | (self.cells == DEPOT)).tolist()

    def _label_shelves(self):
        """
        Assign alphanumeric reference labels to every shelf cell.

        Each shelf column gets its own letter (A, B, C … left-to-right).
        Rows within the shelf zone are numbered 1, 2, 3 … top-to-bottom.
        Labels beyond Z roll over to AA, AB, etc.

        Only cells within shelf_start_row..shelf_end_row are labelled,
        matching exactly what _build() places as ITEM cells.
        """
        cols_sorted = sorted(self._shelf_cols)

        def col_letter(n):
            """Convert 0-based index to spreadsheet-style letter: 0→A, 25→Z, 26→AA…"""
            s = ""
            n += 1
            while n:
                n, r = divmod(n - 1, 26)
                s = chr(65 + r) + s
            return s

        # All rows cleared by the cross-aisle (empty set when disabled)
        if self.cross_aisle_row is not None:
            cross_aisle_rows = set(range(self.cross_aisle_row,
                                         min(self.rows,
                                             self.cross_aisle_row + self.cross_aisle_width)))
        else:
            cross_aisle_rows = set()

        for idx, col in enumerate(cols_sorted):
            letter = col_letter(idx)
            row_number = 1
            for row in range(self.shelf_start_row, self.shelf_end_row + 1):
                if row in cross_aisle_rows:
                    # Cross-aisle rows are walkable floor — no shelf label here
                    continue
                self.shelf_labels[(row, col)] = f"{letter}{row_number}"
                row_number += 1

    def is_walkable(self, row, col):
        """
        Returns True if the agent can step on this cell.
        The agent can walk on EMPTY and DEPOT cells only.
        ITEM and SHELF cells are physical objects — cannot walk through them.
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False  # outside the warehouse
        return self._walkable[row][col]

    def get_all_item_positions(self):
        """
        Returns a list of (row, col) for every cell that has an item.
        Used when randomly picking which item to collect.
        np.argwhere returns all positions where the condition is True,
        much faster than a nested loop on large grids.
        """
        return [tuple(p) for p in np.argwhere(self.cells == ITEM)]

    def deplete_item(self, row, col):
        """
        Called when an agent picks an item. Marks the cell as temporarily
        depleted (SHELF) and schedules it for restocking after a randomised
        delay in the range [80%, 150%] of replenish_delay.
        """
        if self.cells[row, col] == ITEM:
            self.cells[row, col] = SHELF
            lo = max(1, int(self.replenish_delay * 0.8))
            hi = int(self.replenish_delay * 1.5)
            self._depleted[(row, col)] = random.randint(lo, hi)

    def tick_replenishment(self):
        """
        Advance all replenishment timers by one tick.
        Any cell whose timer reaches zero is restocked (SHELF → ITEM).
        Call once per simulation tick.
        """
        expired = [pos for pos, t in self._depleted.items() if t <= 1]
        for pos in expired:
            self.cells[pos[0], pos[1]] = ITEM
            del self._depleted[pos]
        for pos in self._depleted:
            self._depleted[pos] -= 1

    def remove_item(self, row, col):
        """
        Permanently remove an item (ITEM → SHELF, no restock timer).
        Kept for backward compatibility; prefer deplete_item() in normal use.
        """
        if self.cells[row, col] == ITEM:
            self.cells[row, col] = SHELF

    def storage_map(self):
        """
        Returns a 2D numpy array (same shape as self.cells) where:
            1 = storage cell (SHELF or ITEM — a rack exists here)
            0 = everything else (floor, depot, aisle)

        Useful for reporting layout utilisation and visualising the storage
        footprint independently of item state (full vs picked).
        """
        return ((self.cells == ITEM) | (self.cells == SHELF)).astype(np.int8)