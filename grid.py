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
#
# Adjusting shelf_start_row / shelf_end_row lets you create clear staging
# areas near the depot, or compress shelves into a smaller zone — both of
# which directly affect agent travel distance and are key layout variables
# for optimization.
# =============================================================================

# Cell type numbers
EMPTY = 0
SHELF = 1
ITEM  = 2
DEPOT = 3


class Grid:
    """
    Builds and stores the warehouse layout.
    """

    def __init__(self, rows=15, cols=17, aisle_width=2,
                 centre_aisle_width=3, depot_row=None, depot_col=None,
                 shelf_start_row=None, shelf_end_row=None):
        """
        Create the warehouse grid.

        rows               — how many rows tall (default 15)
        cols               — how many columns wide (default 17, odd recommended)
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
        """
        if centre_aisle_width % 2 == 0:
            raise ValueError("centre_aisle_width must be odd")

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

        # Depot position — defaults to top-centre
        self.depot_row = depot_row if depot_row is not None else 0
        self.depot_col = depot_col if depot_col is not None else cols // 2
        self.depot     = (self.depot_row, self.depot_col)

        # Shelf zone rows — defaults to full interior height
        # shelf_start_row must be at least 1 (row 0 is always a walkway)
        # shelf_end_row must be at most rows-2 (last row is always a walkway)
        self.shelf_start_row = shelf_start_row if shelf_start_row is not None else 1
        self.shelf_end_row   = shelf_end_row   if shelf_end_row   is not None else rows - 2

        # Clamp to valid range
        self.shelf_start_row = max(1, self.shelf_start_row)
        self.shelf_end_row   = min(rows - 2, self.shelf_end_row)

        if self.shelf_start_row > self.shelf_end_row:
            raise ValueError(
                f"shelf_start_row ({self.shelf_start_row}) must be <= "
                f"shelf_end_row ({self.shelf_end_row})"
            )

        # 2D grid of cell types — starts all empty
        self.cells = [[EMPTY for _ in range(cols)] for _ in range(rows)]

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

        # --- Fill every cell ---
        dr, dc = self.depot_row, self.depot_col

        for r in range(self.rows):
            for c in range(self.cols):

                if r == dr and c == dc:
                    # Depot cell
                    self.cells[r][c] = DEPOT

                elif self.shelf_start_row <= r <= self.shelf_end_row and c in shelf_cols:
                    # Inside shelf zone and a shelf column — place an item
                    self.cells[r][c] = ITEM

                else:
                    # Everything else: walkable floor
                    self.cells[r][c] = EMPTY

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

        for idx, col in enumerate(cols_sorted):
            letter = col_letter(idx)
            row_number = 1
            for row in range(self.shelf_start_row, self.shelf_end_row + 1):
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
        return self.cells[row][col] in (EMPTY, DEPOT)

    def get_all_item_positions(self):
        """
        Returns a list of (row, col) for every cell that has an item.
        Used when randomly picking which item to collect.
        """
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.cells[r][c] == ITEM
        ]

    def remove_item(self, row, col):
        """
        Called when the agent picks up an item.
        The shelf becomes empty (SHELF=1) — the rack is still there, just no item.
        """
        if self.cells[row][col] == ITEM:
            self.cells[row][col] = SHELF