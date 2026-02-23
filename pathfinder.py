# =============================================================================
# pathfinder.py — Shortest Path (A* Algorithm)
# =============================================================================
# This file answers one question:
#   "What is the shortest route between two cells?"
#
# We use A* (A-star) — a well known pathfinding algorithm used in games
# and robotics. It's efficient because it makes smart guesses about which
# direction to search first, rather than trying every possible path.
#
# HOW IT WORKS (simple version):
#   - Keep a list of cells to check, sorted by "most promising first"
#   - "Most promising" = steps taken so far + estimated steps remaining
#   - Always check the most promising cell next
#   - When we reach the goal, trace back the route we took
#
# The agent moves UP, DOWN, LEFT, RIGHT only (no diagonal).
# =============================================================================

import heapq  # built-in Python tool — always gives us the smallest item first


def heuristic(a, b):
    """
    Estimate the distance between cell a and cell b.
    We use Manhattan distance: move in straight lines only, no diagonals.
    Example: from (0,0) to (3,4) = |3| + |4| = 7 steps minimum.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_path(grid, start, goal):
    """
    Find the shortest walkable path from start to goal.

    grid  — the Grid object (used to check if cells are walkable)
    start — (row, col) where the agent is now
    goal  — (row, col) where the agent wants to go

    Returns a list of (row, col) steps from start to goal (inclusive).
    Returns [] if no path exists.
    """
    if start == goal:
        return [start]

    # Priority queue: entries are (f_score, steps_taken, position)
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))

    # Track which cell we came from (to reconstruct the path at the end)
    came_from = {}

    # Track the actual number of steps to reach each cell
    steps_to = {start: 0}

    while open_set:
        _, current_steps, current = heapq.heappop(open_set)

        # Reached the goal!
        if current == goal:
            return _build_path(came_from, current)

        # Check all 4 neighbours: up, down, left, right
        row, col = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour = (row + dr, col + dc)

            # Skip if not walkable — UNLESS it's the goal itself
            # (agent walks NEXT TO items, so goal may be a non-walkable shelf)
            if neighbour != goal and not grid.is_walkable(*neighbour):
                continue

            new_steps = current_steps + 1

            if neighbour not in steps_to or new_steps < steps_to[neighbour]:
                steps_to[neighbour] = new_steps
                came_from[neighbour] = current
                f = new_steps + heuristic(neighbour, goal)
                heapq.heappush(open_set, (f, new_steps, neighbour))

    return []  # no path found


def find_path_to_neighbour(grid, start, goal):
    """
    Find a path to any walkable cell that is directly next to the goal.
    Used when the agent cannot step ON the goal (e.g. a shelf/item cell).

    Tries all 4 cells adjacent to goal, returns the shortest path found.
    """
    row, col = goal
    neighbours = [
        (row + dr, col + dc)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if grid.is_walkable(row + dr, col + dc)
    ]

    best = []
    for n in neighbours:
        path = find_path(grid, start, n)
        if path and (not best or len(path) < len(best)):
            best = path
    return best


def _build_path(came_from, current):
    """Trace backwards through came_from to get the full path."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
