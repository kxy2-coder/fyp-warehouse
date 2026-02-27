# =============================================================================
# main.py — Run the Simulation
# =============================================================================
# Runs a two-agent warehouse simulation.
# Each agent completes 10 pickup orders independently.
# Human factors (fatigue, experience) are modeled per Malpas & Relvas (2025).
# Congestion is measured as "cell conflicts": simulation steps where both
# agents occupy the same cell simultaneously.
#
# Usage:
#   python main.py                              # default 20x21 grid
#   python main.py --rows 20 --cols 25          # larger grid
#   python main.py --speed 0.15                 # faster animation
#   python main.py --aisle-width 3              # wider aisles
#   python main.py --centre-aisle 5             # wider centre aisle (must be odd)
#   python main.py --depot-col 5                # shift depot left
# =============================================================================

import argparse
import pygame
import sys
import time

from grid    import Grid, EMPTY, SHELF, ITEM, DEPOT
from agent   import Agent, TOTAL_ORDERS, EXPERIENCE_B, BLOCKED_WAIT_TICKS
from metrics import MetricsTracker


# =============================================================================
# CLI ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Two-agent warehouse simulator")
    parser.add_argument("--rows",          type=int,   default=25,   help="Grid rows (default 25)")
    parser.add_argument("--cols",          type=int,   default=35,   help="Grid cols (default 35, odd recommended)")
    parser.add_argument("--speed",         type=float, default=0.15, help="Seconds between steps (default 0.15)")
    parser.add_argument("--aisle-width",   type=int,   default=2,    help="Aisle width between shelf blocks (default 2)")
    parser.add_argument("--centre-aisle",  type=int,   default=3,    help="Centre aisle width, must be odd (default 3)")
    parser.add_argument("--depot-row",     type=int,   default=None, help="Depot row (default 0 = top row)")
    parser.add_argument("--depot-col",     type=int,   default=None, help="Depot column (default = centre)")
    parser.add_argument("--shelf-start",   type=int,   default=None, help="First row shelves appear (default 1)")
    parser.add_argument("--shelf-end",     type=int,   default=None, help="Last row shelves appear (default rows-2)")
    parser.add_argument("--cell-size",     type=int,   default=35,   help="Pixel size per cell (default 35)")
    return parser.parse_args()


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

CELL_SIZE   = 40
MARGIN      = 2
PANEL_WIDTH = 260

# Colours
COL_BG           = (15,  20,  30)
COL_EMPTY        = (30,  38,  50)
COL_SHELF        = (50,  58,  72)
COL_ITEM_DOT     = (220, 160,  40)
COL_DEPOT        = (30,  70, 130)
COL_DEPOT_LABEL  = (100, 160, 255)
COL_AGENT1       = ( 60, 200,  80)
COL_AGENT2       = ( 60, 180, 220)
COL_AGENT1_DOT   = (200, 255, 210)
COL_AGENT2_DOT   = (200, 240, 255)
COL_TARGET1      = (220, 100,  50)
COL_TARGET2      = (180,  60, 200)
COL_PATH1        = ( 40,  80, 120)
COL_PATH2        = ( 40, 110,  90)
COL_CONFLICT     = (220,  50,  50)
COL_PANEL        = (20,  26,  36)
COL_WHITE        = (230, 235, 245)
COL_MUTED        = (100, 115, 135)
COL_DONE         = ( 60, 200,  80)
COL_FATIGUE_LOW  = ( 60, 200,  80)   # green  — low fatigue
COL_FATIGUE_MED  = (220, 180,  40)   # yellow — medium fatigue
COL_FATIGUE_HIGH = (220,  60,  50)   # red    — high fatigue
COL_RESTING      = ( 80, 140, 220)   # blue ring when agent is resting


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def cell_rect(row, col):
    x = col * (CELL_SIZE + MARGIN) + MARGIN
    y = row * (CELL_SIZE + MARGIN) + MARGIN
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def fatigue_colour(fatigue):
    """Interpolate colour from green -> yellow -> red based on fatigue 0-1."""
    if fatigue < 0.5:
        t = fatigue / 0.5
        r = int(COL_FATIGUE_LOW[0] + t * (COL_FATIGUE_MED[0] - COL_FATIGUE_LOW[0]))
        g = int(COL_FATIGUE_LOW[1] + t * (COL_FATIGUE_MED[1] - COL_FATIGUE_LOW[1]))
        b = int(COL_FATIGUE_LOW[2] + t * (COL_FATIGUE_MED[2] - COL_FATIGUE_LOW[2]))
    else:
        t = (fatigue - 0.5) / 0.5
        r = int(COL_FATIGUE_MED[0] + t * (COL_FATIGUE_HIGH[0] - COL_FATIGUE_MED[0]))
        g = int(COL_FATIGUE_MED[1] + t * (COL_FATIGUE_HIGH[1] - COL_FATIGUE_MED[1]))
        b = int(COL_FATIGUE_MED[2] + t * (COL_FATIGUE_HIGH[2] - COL_FATIGUE_MED[2]))
    return (r, g, b)


def draw_fatigue_bar(surface, x, y, width, fatigue, font_small):
    """Draw a labelled fatigue progress bar."""
    bar_h    = 10
    filled_w = int(width * fatigue)
    col      = fatigue_colour(fatigue)

    # Background
    pygame.draw.rect(surface, (40, 48, 60), pygame.Rect(x, y, width, bar_h), border_radius=3)
    # Fill
    if filled_w > 0:
        pygame.draw.rect(surface, col, pygame.Rect(x, y, filled_w, bar_h), border_radius=3)
    # Border
    pygame.draw.rect(surface, (70, 82, 100), pygame.Rect(x, y, width, bar_h), width=1, border_radius=3)
    # Label
    label = font_small.render(f"Fatigue: {fatigue:.0%}", True, col)
    surface.blit(label, (x, y + bar_h + 2))
    return y + bar_h + 16


def draw_grid(surface, grid, agent1, agent2, metrics, font_small):
    """Draw the warehouse grid, agents, paths, targets, conflict flash."""
    upcoming1 = set(agent1.path[agent1.path_index:])
    upcoming2 = set(agent2.path[agent2.path_index:])

    for r in range(grid.rows):
        for c in range(grid.cols):
            rect = cell_rect(r, c)
            cell = grid.cells[r][c]
            pos  = (r, c)

            # Base colour
            if cell == DEPOT:
                colour = COL_DEPOT
            elif cell in (SHELF, ITEM):
                colour = COL_SHELF
            else:
                colour = COL_EMPTY

            pygame.draw.rect(surface, colour, rect, border_radius=3)

            # Item dot
            if cell == ITEM:
                dot_size = CELL_SIZE // 5
                dot_rect = pygame.Rect(
                    rect.centerx - dot_size // 2,
                    rect.centery - dot_size // 2,
                    dot_size, dot_size
                )
                pygame.draw.rect(surface, COL_ITEM_DOT, dot_rect, border_radius=2)

            # Shelf reference label (e.g. "A1", "B3") — hidden below 18px cells
            if CELL_SIZE >= 18 and cell in (SHELF, ITEM):
                label = grid.shelf_labels.get((r, c), "")
                if label:
                    lbl_surf = font_small.render(label, True, COL_MUTED)
                    lbl_rect = lbl_surf.get_rect(center=rect.center)
                    if cell == ITEM:
                        lbl_rect.y = rect.y + 3   # nudge up to avoid the dot
                    surface.blit(lbl_surf, lbl_rect)

            # Target rings
            if not agent1.is_done() and pos == agent1.target:
                pygame.draw.rect(surface, COL_TARGET1, rect, width=2, border_radius=3)
            if not agent2.is_done() and pos == agent2.target:
                pygame.draw.rect(surface, COL_TARGET2, rect, width=2, border_radius=3)

            # Path dots
            dot_r = CELL_SIZE // 8
            cx, cy = rect.center
            if pos in upcoming1 and pos != agent1.pos and not agent1.is_done():
                pygame.draw.circle(surface, COL_PATH1, (cx, cy), dot_r)
            if pos in upcoming2 and pos != agent2.pos and not agent2.is_done():
                pygame.draw.circle(surface, COL_PATH2, (cx, cy), dot_r)

            # Conflict flash
            if metrics.flash_timer > 0 and pos == metrics.conflict_cell:
                pygame.draw.rect(surface, COL_CONFLICT, rect, width=3, border_radius=3)

    # Draw agents on top
    for agent, col_fill, col_dot in [
        (agent1, COL_AGENT1, COL_AGENT1_DOT),
        (agent2, COL_AGENT2, COL_AGENT2_DOT),
    ]:
        if not agent.is_done():
            r = cell_rect(*agent.pos)
            # Blue ring when resting, fatigue-coloured ring when picking up
            if agent.state == "resting":
                pygame.draw.circle(surface, COL_RESTING, r.center,
                                   CELL_SIZE // 2 - 2, width=3)
            elif agent.state == "picking_up":
                pygame.draw.circle(surface, fatigue_colour(agent.fatigue),
                                   r.center, CELL_SIZE // 2 - 2, width=3)
            pygame.draw.circle(surface, col_fill, r.center, CELL_SIZE // 2 - 4)
            pygame.draw.circle(surface, col_dot,  r.center, CELL_SIZE // 8)


def draw_panel(surface, grid, agent1, agent2, metrics,
               font_big, font_med, font_small):
    """Right-side info panel with agent status, fatigue bars, experience."""
    panel_x = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    pygame.draw.rect(surface, COL_PANEL,
                     pygame.Rect(panel_x, 0, PANEL_WIDTH, surface.get_height()))
    pygame.draw.line(surface, (40, 50, 65),
                     (panel_x, 0), (panel_x, surface.get_height()), 1)

    x          = panel_x + 16
    y          = 20
    bar_width  = PANEL_WIDTH - 36
    divider_end = panel_x + PANEL_WIDTH - 16

    surface.blit(font_big.render("WAREHOUSE SIM", True, COL_DEPOT_LABEL), (x, y))
    y += 36

    state_label = {
        "waiting":    "Starting...",
        "to_item":    "Heading to item",
        "picking_up": "Picking up item",
        "to_depot":   "Returning to depot",
        "resting":    "Resting at depot",
        "all_done":   "Shift complete!",
    }

    def divider():
        nonlocal y
        pygame.draw.line(surface, (40, 50, 65), (x, y), (divider_end, y))
        y += 10

    def agent_panel(agent, col_agent, col_target, label):
        nonlocal y
        divider()

        # Header: agent label + experience level
        exp_label = "Experienced" if agent.experience_B >= 100 else "Novice"
        surface.blit(font_small.render(
            f"{label}  [{exp_label}, B={agent.experience_B:.0f}h]",
            True, col_agent), (x, y));  y += 18

        # State
        status = state_label.get(agent.state, agent.state)
        col    = COL_DONE if agent.is_done() else COL_WHITE
        surface.blit(font_med.render(status, True, col), (x, y));  y += 22

        # Orders / distance / yields
        surface.blit(font_small.render(
            f"Orders: {agent.orders_completed}/{TOTAL_ORDERS}"
            f"   Dist: {agent.distance}",
            True, COL_MUTED), (x, y));  y += 18
        surface.blit(font_small.render(
            f"Blocked: {agent.blocked_count}x  ({agent.blocked_count * BLOCKED_WAIT_TICKS}s waited)",
            True, COL_MUTED), (x, y));  y += 18

        # Experience factor
        exp_factor = agent._experience_factor()
        surface.blit(font_small.render(
            f"Exp factor: {exp_factor:.3f}  "
            f"(work: {agent.work_time*60:.1f} min)",
            True, COL_MUTED), (x, y));  y += 18

        # Pickup ticks remaining (when picking up)
        if agent.state == "picking_up":
            surface.blit(font_small.render(
                f"Pickup ticks left: {agent.pickup_ticks_remaining}",
                True, COL_ITEM_DOT), (x, y));  y += 16

        # Fatigue bar
        y = draw_fatigue_bar(surface, x, y, bar_width, agent.fatigue, font_small)
        y += 4

    agent_panel(agent1, COL_AGENT1, COL_TARGET1, "AGENT 1")
    agent_panel(agent2, COL_AGENT2, COL_TARGET2, "AGENT 2")

    # Congestion
    divider()
    surface.blit(font_small.render("CONGESTION", True, COL_MUTED), (x, y));       y += 18
    surface.blit(font_small.render("CELL CONFLICTS", True, COL_MUTED), (x, y));   y += 16
    conflict_col = COL_CONFLICT if metrics.cell_conflicts > 0 else COL_WHITE
    surface.blit(font_big.render(str(metrics.cell_conflicts), True, conflict_col), (x, y)); y += 36
    surface.blit(font_small.render("steps sharing a cell", True, COL_MUTED), (x, y)); y += 24

    # Simulation complete summary
    if agent1.is_done() and agent2.is_done():
        divider()
        surface.blit(font_med.render("Simulation complete!", True, COL_DONE), (x, y)); y += 22
        surface.blit(font_small.render(
            f"A1: {agent1.distance} cells | fatigue={agent1.fatigue:.2f}",
            True, COL_AGENT1), (x, y)); y += 16
        surface.blit(font_small.render(
            f"A2: {agent2.distance} cells | fatigue={agent2.fatigue:.2f}",
            True, COL_AGENT2), (x, y)); y += 16
        surface.blit(font_small.render(
            f"Conflicts: {metrics.cell_conflicts}",
            True, COL_CONFLICT), (x, y)); y += 16
        surface.blit(font_small.render("(close window to exit)", True, COL_MUTED), (x, y))

    # Legend
    legend_y = surface.get_height() - 170
    pygame.draw.line(surface, (40, 50, 65), (x, legend_y), (divider_end, legend_y))
    legend_y += 10
    surface.blit(font_small.render("LEGEND", True, COL_MUTED), (x, legend_y)); legend_y += 18

    legend_items = [
        (COL_EMPTY,        "Floor"),
        (COL_SHELF,        "Shelf / empty rack"),
        (COL_ITEM_DOT,     "Item on shelf"),
        (COL_DEPOT,        "Depot"),
        (COL_AGENT1,       "Agent 1 (experienced)"),
        (COL_AGENT2,       "Agent 2 (novice)"),
        (COL_RESTING,      "Resting at depot"),
        (COL_CONFLICT,     "Conflict cell"),
    ]
    for colour, text in legend_items:
        sq = pygame.Rect(x, legend_y + 2, 10, 10)
        pygame.draw.rect(surface, colour, sq, border_radius=2)
        surface.blit(font_small.render(text, True, COL_MUTED), (x + 16, legend_y))
        legend_y += 16


# =============================================================================
# RIGHT-OF-WAY RESOLUTION
# =============================================================================

def resolve_right_of_way(agent1, agent2):
    """
    Called before each tick. If both agents are heading to the same cell,
    or are about to swap positions head-on, mark the lower-priority one to yield.

    Priority (highest first):
      1. State: to_depot (loaded, returning) beats to_item (going to pick up)
      2. Tiebreaker: Agent 1 (experienced) beats Agent 2 (novice)
    """
    if agent1.is_done() or agent2.is_done():
        return

    next1 = agent1.peek_next_pos()
    next2 = agent2.peek_next_pos()

    if next1 is None or next2 is None:
        return

    same_dest = (next1 == next2)
    head_on   = (next1 == agent2.pos and next2 == agent1.pos)
    if not (same_dest or head_on):
        return

    def priority(agent):
        state_score = 1 if agent.state == "to_depot" else 0
        id_score    = 1 if agent.agent_id == 1 else 0
        return (state_score, id_score)

    if priority(agent1) >= priority(agent2):
        if agent2.blocked_ticks_remaining == 0:   # new event — log it once
            agent2.blocked_count += 1
            print(
                f"  [Agent 2] Blocked by Agent 1"
                f" | event #{agent2.blocked_count}"
                f" | waiting {BLOCKED_WAIT_TICKS}s"
            )
        agent2.blocked_ticks_remaining = BLOCKED_WAIT_TICKS
    else:
        if agent1.blocked_ticks_remaining == 0:
            agent1.blocked_count += 1
            print(
                f"  [Agent 1] Blocked by Agent 2"
                f" | event #{agent1.blocked_count}"
                f" | waiting {BLOCKED_WAIT_TICKS}s"
            )
        agent1.blocked_ticks_remaining = BLOCKED_WAIT_TICKS


# =============================================================================
# HEADLESS RUNNER — for optimization and batch experiments
# =============================================================================

def run_headless(
    rows=25, cols=35,
    aisle_width=2, centre_aisle_width=3,
    depot_row=None, depot_col=None,
    shelf_start_row=None, shelf_end_row=None,
    max_ticks=50000,
    seed=None,
):
    """
    Run the full simulation without any pygame graphics.
    Returns a dictionary of metrics when both agents finish.

    This is the function your optimization / RL loop will call.
    Pass in layout parameters, get back performance metrics.

    Parameters
    ----------
    rows, cols              — warehouse dimensions
    aisle_width             — aisle width between shelf blocks
    centre_aisle_width      — main centre aisle width (must be odd)
    depot_row               — depot row (default 0 = top)
    depot_col               — depot column (default = centre)
    shelf_start_row         — first row shelves appear (default 1)
    shelf_end_row           — last row shelves appear (default rows-2)
    max_ticks               — safety cap to prevent infinite loops
    seed                    — random seed for reproducibility (None = random)

    Returns
    -------
    dict with keys:
        total_distance      — combined cells walked by both agents
        agent1_distance     — cells walked by agent 1
        agent2_distance     — cells walked by agent 2
        cell_conflicts      — ticks both agents shared the same cell
        agent1_work_time    — agent 1 total work time in hours
        agent2_work_time    — agent 2 total work time in hours
        agent1_fatigue      — agent 1 final fatigue level [0, 1]
        agent2_fatigue      — agent 2 final fatigue level [0, 1]
        agent1_blocked      — number of times agent 1 was blocked
        agent2_blocked      — number of times agent 2 was blocked
        ticks               — total simulation ticks to completion
        throughput          — orders completed per hour of combined work time
                              (primary metric for optimization)
        completed           — True if both agents finished, False if max_ticks hit
    """
    import random as _random
    if seed is not None:
        _random.seed(seed)

    grid   = Grid(
        rows=rows, cols=cols,
        aisle_width=aisle_width,
        centre_aisle_width=centre_aisle_width,
        depot_row=depot_row,
        depot_col=depot_col,
        shelf_start_row=shelf_start_row,
        shelf_end_row=shelf_end_row,
    )
    agent1  = Agent(grid, agent_id=1, color=(0, 0, 0))
    agent2  = Agent(grid, agent_id=2, color=(0, 0, 0))
    metrics = MetricsTracker()

    ticks = 0
    while not (agent1.is_done() and agent2.is_done()):
        if ticks >= max_ticks:
            break
        resolve_right_of_way(agent1, agent2)
        agent1.step()
        agent2.step()
        metrics.update(agent1, agent2, grid.depot)
        ticks += 1

    total_work = agent1.work_time + agent2.work_time
    total_orders = agent1.orders_completed + agent2.orders_completed
    throughput = total_orders / total_work if total_work > 0 else 0.0

    return {
        "total_distance":   agent1.distance + agent2.distance,
        "agent1_distance":  agent1.distance,
        "agent2_distance":  agent2.distance,
        "cell_conflicts":   metrics.cell_conflicts,
        "agent1_work_time": agent1.work_time,
        "agent2_work_time": agent2.work_time,
        "agent1_fatigue":   agent1.fatigue,
        "agent2_fatigue":   agent2.fatigue,
        "agent1_blocked":   agent1.blocked_count,
        "agent2_blocked":   agent2.blocked_count,
        "ticks":            ticks,
        "throughput":       throughput,
        "completed":        agent1.is_done() and agent2.is_done(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    global CELL_SIZE
    CELL_SIZE = args.cell_size

    pygame.init()

    grid   = Grid(
        rows=args.rows, cols=args.cols,
        aisle_width=args.aisle_width,
        centre_aisle_width=args.centre_aisle,
        depot_row=args.depot_row,
        depot_col=args.depot_col,
        shelf_start_row=args.shelf_start,
        shelf_end_row=args.shelf_end,
    )
    agent1 = Agent(grid, agent_id=1, color=COL_AGENT1)
    agent2 = Agent(grid, agent_id=2, color=COL_AGENT2)

    print("=" * 55)
    print("  Warehouse Simulator")
    print("=" * 55)
    print(f"  Grid         : {grid.rows} rows x {grid.cols} cols")
    print(f"  Aisle width  : {grid.aisle_width}  |  Centre aisle: {grid.centre_aisle_width}")
    print(f"  Depot        : row {grid.depot[0]}, col {grid.depot[1]}")
    print(f"  Shelf zone   : rows {grid.shelf_start_row} to {grid.shelf_end_row}")
    print(f"  Orders       : {TOTAL_ORDERS} per agent")
    print(f"  Agent 1      : Experienced (B={EXPERIENCE_B[1]:.0f}h)")
    print(f"  Agent 2      : Novice      (B={EXPERIENCE_B[2]:.0f}h)")
    print(f"  Step delay   : {args.speed}s")
    print("=" * 55)

    grid_px_w = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    grid_px_h = grid.rows * (CELL_SIZE + MARGIN) + MARGIN
    win_w     = grid_px_w + PANEL_WIDTH
    win_h     = max(grid_px_h, 600)

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        f"Warehouse Sim — {grid.rows}x{grid.cols}"
    )

    font_big   = pygame.font.SysFont("monospace", 18, bold=True)
    font_med   = pygame.font.SysFont("monospace", 14)
    font_small = pygame.font.SysFont("monospace", 11)
    font_grid  = pygame.font.SysFont("monospace", max(7, CELL_SIZE // 4))

    clock          = pygame.time.Clock()
    metrics        = MetricsTracker()
    last_step_time = time.time()

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now       = time.time()
        both_done = agent1.is_done() and agent2.is_done()

        if not both_done and (now - last_step_time) >= args.speed:
            resolve_right_of_way(agent1, agent2)
            agent1.step()
            agent2.step()
            last_step_time = now

            metrics.update(agent1, agent2, grid.depot)
            metrics.print_step(agent1, agent2)

            if agent1.is_done() and agent2.is_done():
                metrics.print_summary(agent1, agent2)

        metrics.tick_flash()

        screen.fill(COL_BG)
        draw_grid(screen, grid, agent1, agent2, metrics, font_grid)
        draw_panel(screen, grid, agent1, agent2, metrics,
                   font_big, font_med, font_small)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

