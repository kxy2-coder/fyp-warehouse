# =============================================================================
# main.py — Run the Simulation
# =============================================================================
# Runs a two-agent warehouse simulation.
# Each agent completes 10 pickup orders independently.
# Congestion is measured as "cell conflicts": the number of simulation
# steps where both agents occupy the same cell simultaneously.
#
# Usage:
#   python main.py                        # default 15x17 grid
#   python main.py --rows 20 --cols 25    # larger grid
#   python main.py --speed 0.15           # faster animation
# =============================================================================

import argparse
import pygame
import sys
import time

from grid    import Grid, EMPTY, SHELF, ITEM, DEPOT
from agent   import Agent, TOTAL_ORDERS
from metrics import MetricsTracker


# =============================================================================
# CLI ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Two-agent warehouse simulator")
    parser.add_argument("--rows",  type=int,   default=20,  help="Grid rows (default 15)")
    parser.add_argument("--cols",  type=int,   default=21,  help="Grid cols (default 17, odd recommended)")
    parser.add_argument("--speed", type=float, default=0.2, help="Seconds between steps (default 0.3)")
    return parser.parse_args()


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

CELL_SIZE   = 40    # pixels per cell
MARGIN      = 2     # gap between cells
PANEL_WIDTH = 240   # info panel width

# Colours (R, G, B)
COL_BG           = (15,  20,  30)
COL_EMPTY        = (30,  38,  50)
COL_SHELF        = (50,  58,  72)
COL_ITEM_DOT     = (220, 160,  40)
COL_DEPOT        = (30,  70, 130)
COL_DEPOT_LABEL  = (100, 160, 255)
COL_AGENT1       = ( 60, 200,  80)   # green
COL_AGENT2       = ( 60, 180, 220)   # cyan
COL_AGENT1_DOT   = (200, 255, 210)
COL_AGENT2_DOT   = (200, 240, 255)
COL_TARGET1      = (220, 100,  50)   # orange ring — agent 1 target
COL_TARGET2      = (180,  60, 200)   # purple ring — agent 2 target
COL_PATH1        = ( 40,  80, 120)
COL_PATH2        = ( 40, 110,  90)
COL_CONFLICT     = (220,  50,  50)   # red flash on conflict cell
COL_PANEL        = (20,  26,  36)
COL_WHITE        = (230, 235, 245)
COL_MUTED        = (100, 115, 135)
COL_DONE         = ( 60, 200,  80)


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def cell_rect(row, col):
    x = col * (CELL_SIZE + MARGIN) + MARGIN
    y = row * (CELL_SIZE + MARGIN) + MARGIN
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def draw_grid(surface, grid, agent1, agent2, metrics):
    """
    Draw the full warehouse grid including both agents, their paths,
    their targets, and the conflict flash.
    """
    upcoming1 = set(agent1.path[agent1.path_index:])
    upcoming2 = set(agent2.path[agent2.path_index:])

    for r in range(grid.rows):
        for c in range(grid.cols):
            rect = cell_rect(r, c)
            cell = grid.cells[r][c]
            pos  = (r, c)

            # --- Base cell colour ---
            if cell == DEPOT:
                colour = COL_DEPOT
            elif cell in (SHELF, ITEM):
                colour = COL_SHELF
            else:
                colour = COL_EMPTY

            pygame.draw.rect(surface, colour, rect, border_radius=3)

            # --- Item dot ---
            if cell == ITEM:
                dot_size = CELL_SIZE // 5
                dot_rect = pygame.Rect(
                    rect.centerx - dot_size // 2,
                    rect.centery - dot_size // 2,
                    dot_size, dot_size
                )
                pygame.draw.rect(surface, COL_ITEM_DOT, dot_rect, border_radius=2)

            # --- Target rings ---
            if not agent1.is_done() and pos == agent1.target:
                pygame.draw.rect(surface, COL_TARGET1, rect, width=2, border_radius=3)
            if not agent2.is_done() and pos == agent2.target:
                pygame.draw.rect(surface, COL_TARGET2, rect, width=2, border_radius=3)

            # --- Path dots ---
            dot_r = CELL_SIZE // 8
            cx, cy = rect.center
            if pos in upcoming1 and pos != agent1.pos and not agent1.is_done():
                pygame.draw.circle(surface, COL_PATH1, (cx, cy), dot_r)
            if pos in upcoming2 and pos != agent2.pos and not agent2.is_done():
                pygame.draw.circle(surface, COL_PATH2, (cx, cy), dot_r)

            # --- Conflict flash ---
            if metrics.flash_timer > 0 and pos == metrics.conflict_cell:
                pygame.draw.rect(surface, COL_CONFLICT, rect, width=3, border_radius=3)

        # (end column loop)
    # (end row loop)

    # --- Draw agents (on top of everything else) ---
    for agent, col_fill, col_dot in [
        (agent1, COL_AGENT1, COL_AGENT1_DOT),
        (agent2, COL_AGENT2, COL_AGENT2_DOT),
    ]:
        if not agent.is_done():
            r = cell_rect(*agent.pos)
            pygame.draw.circle(surface, col_fill, r.center, CELL_SIZE // 2 - 4)
            pygame.draw.circle(surface, col_dot,  r.center, CELL_SIZE // 8)


def _panel_text(surface, font, text, colour, x, y):
    surface.blit(font.render(text, True, colour), (x, y))
    return y + font.get_linesize() + 2


def draw_panel(surface, grid, agent1, agent2, metrics,
               font_big, font_med, font_small):
    """
    Right-side info panel showing both agents' status and congestion metric.
    """
    panel_x = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    pygame.draw.rect(surface, COL_PANEL,
                     pygame.Rect(panel_x, 0, PANEL_WIDTH, surface.get_height()))
    pygame.draw.line(surface, (40, 50, 65),
                     (panel_x, 0), (panel_x, surface.get_height()), 1)

    x = panel_x + 16
    y = 20

    # Title
    surface.blit(font_big.render("WAREHOUSE SIM v3", True, COL_DEPOT_LABEL), (x, y))
    y += 36

    state_label = {
        "waiting":    "Starting...",
        "to_item":    "Heading to item",
        "picking_up": "Picking up item",
        "to_depot":   "Returning to depot",
        "all_done":   "All done!",
    }

    divider_end = panel_x + PANEL_WIDTH - 16

    def divider():
        nonlocal y
        pygame.draw.line(surface, (40, 50, 65), (x, y), (divider_end, y))
        y += 10

    # --- Agent 1 ---
    divider()
    surface.blit(font_small.render("AGENT 1", True, COL_AGENT1), (x, y));  y += 18
    status1 = state_label.get(agent1.state, agent1.state)
    col1 = COL_DONE if agent1.is_done() else COL_WHITE
    surface.blit(font_med.render(status1, True, col1), (x, y));  y += 22
    surface.blit(font_small.render(
        f"Orders: {agent1.orders_completed}/{TOTAL_ORDERS}   Dist: {agent1.distance}",
        True, COL_MUTED), (x, y));  y += 18
    if not agent1.is_done() and agent1.target:
        surface.blit(font_small.render(
            f"Target: row {agent1.target[0]}, col {agent1.target[1]}",
            True, COL_TARGET1), (x, y))
    y += 22

    # --- Agent 2 ---
    divider()
    surface.blit(font_small.render("AGENT 2", True, COL_AGENT2), (x, y));  y += 18
    status2 = state_label.get(agent2.state, agent2.state)
    col2 = COL_DONE if agent2.is_done() else COL_WHITE
    surface.blit(font_med.render(status2, True, col2), (x, y));  y += 22
    surface.blit(font_small.render(
        f"Orders: {agent2.orders_completed}/{TOTAL_ORDERS}   Dist: {agent2.distance}",
        True, COL_MUTED), (x, y));  y += 18
    if not agent2.is_done() and agent2.target:
        surface.blit(font_small.render(
            f"Target: row {agent2.target[0]}, col {agent2.target[1]}",
            True, COL_TARGET2), (x, y))
    y += 22

    # --- Congestion metric ---
    divider()
    surface.blit(font_small.render("CONGESTION", True, COL_MUTED), (x, y));  y += 18
    surface.blit(font_small.render("CELL CONFLICTS", True, COL_MUTED), (x, y));  y += 16
    conflict_col = COL_CONFLICT if metrics.cell_conflicts > 0 else COL_WHITE
    surface.blit(font_big.render(str(metrics.cell_conflicts), True, conflict_col), (x, y));  y += 36
    surface.blit(font_small.render("steps sharing a cell", True, COL_MUTED), (x, y));  y += 24

    # --- Both done summary ---
    if agent1.is_done() and agent2.is_done():
        divider()
        surface.blit(font_med.render("Simulation complete!", True, COL_DONE), (x, y));  y += 24
        surface.blit(font_small.render(f"Agent 1: {agent1.distance} cells", True, COL_AGENT1), (x, y));  y += 18
        surface.blit(font_small.render(f"Agent 2: {agent2.distance} cells", True, COL_AGENT2), (x, y));  y += 18
        surface.blit(font_small.render(f"Conflicts: {metrics.cell_conflicts}", True, COL_CONFLICT), (x, y));  y += 18
        surface.blit(font_small.render("(close window to exit)", True, COL_MUTED), (x, y));  y += 18

    # --- Legend at bottom ---
    legend_y = surface.get_height() - 160
    pygame.draw.line(surface, (40, 50, 65), (x, legend_y), (divider_end, legend_y))
    legend_y += 10
    surface.blit(font_small.render("LEGEND", True, COL_MUTED), (x, legend_y));  legend_y += 18

    legend_items = [
        (COL_EMPTY,    "Floor"),
        (COL_SHELF,    "Shelf / rack"),
        (COL_ITEM_DOT, "Item on shelf"),
        (COL_DEPOT,    "Depot"),
        (COL_AGENT1,   "Agent 1 (green)"),
        (COL_AGENT2,   "Agent 2 (cyan)"),
        (COL_CONFLICT, "Conflict cell"),
    ]
    for colour, text in legend_items:
        sq = pygame.Rect(x, legend_y + 2, 10, 10)
        pygame.draw.rect(surface, colour, sq, border_radius=2)
        surface.blit(font_small.render(text, True, COL_MUTED), (x + 16, legend_y))
        legend_y += 18


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    pygame.init()

    grid   = Grid(rows=args.rows, cols=args.cols)
    agent1 = Agent(grid, agent_id=1, color=COL_AGENT1)
    agent2 = Agent(grid, agent_id=2, color=COL_AGENT2)

    # Print run config to terminal
    print("=" * 50)
    print("  Warehouse Simulator v3  —  Two Agents")
    print("=" * 50)
    print(f"  Grid      : {grid.rows} rows x {grid.cols} cols")
    print(f"  Depot     : row {grid.depot[0]}, col {grid.depot[1]}")
    print(f"  Orders    : {TOTAL_ORDERS} per agent  ({TOTAL_ORDERS * 2} total)")
    print(f"  Step delay: {args.speed}s")
    print("=" * 50)

    # Window dimensions
    grid_px_w = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    grid_px_h = grid.rows * (CELL_SIZE + MARGIN) + MARGIN
    win_w     = grid_px_w + PANEL_WIDTH
    win_h     = max(grid_px_h, 560)

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        f"Warehouse Sim — {grid.rows}x{grid.cols} grid — 2 agents × {TOTAL_ORDERS} orders"
    )

    font_big   = pygame.font.SysFont("monospace", 18, bold=True)
    font_med   = pygame.font.SysFont("monospace", 14)
    font_small = pygame.font.SysFont("monospace", 11)

    clock = pygame.time.Clock()

    metrics = MetricsTracker()

    last_step_time = time.time()

    running = True
    while running:

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Simulation step ---
        now = time.time()
        both_done = agent1.is_done() and agent2.is_done()

        if not both_done and (now - last_step_time) >= args.speed:
            agent1.step()
            agent2.step()
            last_step_time = now

            metrics.update(agent1, agent2, grid.depot)
            metrics.print_step(agent1, agent2)

            if both_done:
                metrics.print_summary(agent1, agent2)

        metrics.tick_flash()

        # --- Draw ---
        screen.fill(COL_BG)
        draw_grid(screen, grid, agent1, agent2, metrics)
        draw_panel(screen, grid, agent1, agent2, metrics,
                   font_big, font_med, font_small)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
