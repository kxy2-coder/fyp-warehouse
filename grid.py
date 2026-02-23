# =============================================================================
# grid.py — The Warehouse Layout
# =============================================================================
# This file builds the warehouse as a grid (like a spreadsheet).
# Each cell in the grid is one of these types:
#
#   EMPTY (0) — a floor tile the agent can walk on
#   SHELF (1) — a shelf block the agent CANNOT walk through
#   ITEM  (2) — a shelf that has a item/good on it (all shelves start full)
#   DEPOT (3) — the top-middle cell where the agent starts and returns to
#
# The grid is stored as a 2D list: grid[row][col]
# Row 0 is the TOP of the warehouse.
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

    def __init__(self, rows=15, cols=17):
        """
        Create the warehouse grid.
        rows = how many rows tall  (default 15)
        cols = how many columns wide (default 17, must be odd so there's a centre)
        """
        self.rows = rows
        self.cols = cols

        # Create a 2D list filled with EMPTY
        # Think of it as: a list of rows, each row is a list of cells
        self.cells = [[EMPTY for _ in range(cols)] for _ in range(rows)]

        # Depot is always top-middle
        self.depot = (0, cols // 2)

        # Build the layout
        self._build()

    def _build(self):
        """
        Draw the warehouse onto self.cells.

        Layout:
          Row 0         — top walkway, depot in the centre
          Rows 1 to N-2 — shelf columns with aisles between them
          Last row      — bottom walkway

        Shelf columns are 2 cells wide, separated by 2-cell aisles.
        The centre column is always kept clear (main vertical aisle).
        """
        centre = self.cols // 2

        # Work out which column indices should be shelves.
        # Build each side independently, expanding OUTWARD from the centre.
        # This keeps the layout symmetric for any grid width.
        #
        # Left side:  pairs at (centre-3, centre-2), (centre-7, centre-6), ...
        # Right side: pairs at (centre+2, centre+3), (centre+6, centre+7), ...
        # Centre column and its immediate neighbours (centre±1) stay as aisle.
        # Columns 0 and cols-1 stay as border walkways.
        shelf_cols = set()

        # Left side — work outward from centre
        pos = centre - 3
        while pos >= 1:
            shelf_cols.add(pos)
            shelf_cols.add(pos + 1)
            pos -= 4

        # Right side — work outward from centre
        pos = centre + 2
        while pos + 1 <= self.cols - 2:
            shelf_cols.add(pos)
            shelf_cols.add(pos + 1)
            pos += 4

        # Fill every cell
        for r in range(self.rows):
            for c in range(self.cols):

                if r == 0:
                    # Top row: depot in middle, otherwise empty
                    self.cells[r][c] = DEPOT if c == centre else EMPTY

                elif r == self.rows - 1:
                    # Bottom row: all walkable
                    self.cells[r][c] = EMPTY

                else:
                    # Middle rows: shelf or aisle
                    if c in shelf_cols:
                        self.cells[r][c] = ITEM  # ALL shelf cells start with an item
                    else:
                        self.cells[r][c] = EMPTY

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
