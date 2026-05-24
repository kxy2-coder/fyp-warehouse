# =============================================================================
# plot_heatmap.py — Spatial Conflict Heatmap
# =============================================================================
# Reads conflict_heatmap.csv (row, col, avg_count) and overlays a heatmap on
# the warehouse grid layout.
#
# Usage:
#   python plot_heatmap.py                                  # default CSV + grid
#   python plot_heatmap.py --file conflict_heatmap.csv
#   python plot_heatmap.py --rows 25 --cols 35
#   python plot_heatmap.py --compare layout_a.csv layout_b.csv --labels "A" "B"
# =============================================================================

import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from grid import Grid, SHELF, ITEM


def _load_csv(filename):
    """Read (row, col, avg_count) CSV; return dict {(row, col): avg_count}."""
    data = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[(int(row["row"]), int(row["col"]))] = float(row["avg_count"])
    return data


def _build_arrays(data, grid_rows, grid_cols):
    """Convert dict to 2D numpy array."""
    arr = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    for (r, c), v in data.items():
        if 0 <= r < grid_rows and 0 <= c < grid_cols:
            arr[r, c] = v
    return arr


def _make_grid_mask(grid_rows, grid_cols, **grid_kwargs):
    """Return (shelf_mask, depot_positions) from a Grid instance."""
    g = Grid(rows=grid_rows, cols=grid_cols, **grid_kwargs)
    # Shelves start fully stocked (ITEM=2); depleted ones become SHELF=1.
    # Both values mark a physical shelf cell that agents cannot walk through.
    shelf_mask = (g.cells == SHELF) | (g.cells == ITEM)
    return shelf_mask, [tuple(d) for d in g.depots]


def _draw_panel(ax, heatmap, shelf_mask, depot_positions, title, norm=None):
    """
    Draw a single heatmap panel:
      - light steel-blue floor for zero-conflict walkable cells
      - dark blue-grey for shelves with outlined borders
      - log-scaled YlOrRd heatmap on conflict cells
      - blue depot marker
    Returns (im, norm) so the caller can share the colour scale.
    """
    grid_rows, grid_cols = heatmap.shape

    # ── 1. Background: floor colour vs shelf colour ──────────────────────────
    bg = np.ones((grid_rows, grid_cols, 3))
    bg[:, :] = [0.88, 0.91, 0.96]          # light steel-blue floor
    bg[shelf_mask] = [0.30, 0.32, 0.38]     # dark blue-grey shelves
    ax.imshow(bg, aspect="auto", origin="upper",
              extent=[-0.5, grid_cols - 0.5, grid_rows - 0.5, -0.5], zorder=1)

    # ── 3. Heatmap with sqrt power-norm ──────────────────────────────────────
    # Square-root scale compresses the range just enough to reveal variation
    # across the warehouse without making every 1-count cell glow like log does.
    vals = heatmap[heatmap > 0]
    if vals.size > 0:
        if norm is None:
            vmax_val = float(np.percentile(vals, 98))
            vmax_val = max(vmax_val, 2.0)
            norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_val)

        # Mask zeros AND shelf cells — shelves are never walkable so
        # they should never show heatmap colour, only the grey background.
        display = np.ma.masked_where((heatmap == 0) | shelf_mask, heatmap.astype(float))
        im = ax.imshow(display, cmap="YlOrRd", norm=norm, alpha=0.88,
                       aspect="auto", origin="upper",
                       extent=[-0.5, grid_cols - 0.5, grid_rows - 0.5, -0.5],
                       zorder=2)
    else:
        dummy = np.ma.masked_all((grid_rows, grid_cols))
        norm  = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=2)
        im    = ax.imshow(dummy, cmap="YlOrRd", norm=norm, alpha=0,
                          aspect="auto", origin="upper",
                          extent=[-0.5, grid_cols - 0.5, grid_rows - 0.5, -0.5],
                          zorder=2)

    # ── 4. Depot markers (one per depot) ─────────────────────────────────────
    for dr, dc in depot_positions:
        ax.plot(dc, dr, marker="s", markersize=9, color="#3B82F6",
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        ax.text(dc, dr, "D", ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold", zorder=6)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Column", fontsize=8)
    ax.set_ylabel("Row", fontsize=8)
    ax.tick_params(labelsize=7)

    return im, norm


def plot_from_csv(filename="traffic_heatmap.csv",
                  grid_rows=25, grid_cols=35,
                  out_file="traffic_heatmap.png",
                  **grid_kwargs):
    """
    Single-layout heatmap with its own independent colour scale.
    Called from main.py when --plot-heatmap is set — once for traffic
    density and once for congestion hotspots, each with full colour range.
    """
    data = _load_csv(filename)
    heatmap = _build_arrays(data, grid_rows, grid_cols)
    shelf_mask, depot_pos = _make_grid_mask(grid_rows, grid_cols, **grid_kwargs)

    # Auto-detect which heatmap this is from the filename for the title
    is_congestion = "congestion" in filename.lower()
    if is_congestion:
        suptitle  = "Congestion Hotspot Heatmap\n(avg ticks per cell — speed < 50% free speed AND another agent within 0.9 m)"
        cbar_label = "Avg slow-walk ticks per cell"
    else:
        suptitle  = "Traffic Density Heatmap\n(avg ticks agents spent walking through each cell)"
        cbar_label = "Avg agent-ticks per cell"

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    im, _ = _draw_panel(ax, heatmap, shelf_mask, depot_pos,
                        title=f"{grid_rows}×{grid_cols} warehouse layout")

    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"  Heatmap saved → {out_file}")
    plt.close(fig)


def plot_comparison(file_a, file_b,
                    label_a="Layout A", label_b="Layout B",
                    grid_rows=25, grid_cols=35,
                    out_file="conflict_heatmap_comparison.png",
                    grid_kwargs_a=None, grid_kwargs_b=None):
    """
    Side-by-side heatmap comparing two layout CSVs.
    Colour scale is shared so intensities are directly comparable.
    """
    grid_kwargs_a = grid_kwargs_a or {}
    grid_kwargs_b = grid_kwargs_b or {}

    data_a = _load_csv(file_a)
    data_b = _load_csv(file_b)
    hm_a = _build_arrays(data_a, grid_rows, grid_cols)
    hm_b = _build_arrays(data_b, grid_rows, grid_cols)

    sm_a, dp_a = _make_grid_mask(grid_rows, grid_cols, **grid_kwargs_a)
    sm_b, dp_b = _make_grid_mask(grid_rows, grid_cols, **grid_kwargs_b)

    # Shared sqrt scale using the true raw max across both layouts so absolute
    # magnitudes are directly comparable — layout A at 82 looks darker than B at 40
    vmax_shared = max(float(hm_a.max()), float(hm_b.max()), 2.0)
    shared_norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_shared)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Traffic Density vs Congestion Hotspots\n"
                 "Left: all walking ticks per cell  |  Right: congestion ticks only (speed < 50% free speed AND agent within 0.9 m)",
                 fontsize=11, fontweight="bold")

    im1, _ = _draw_panel(ax1, hm_a, sm_a, dp_a, title=label_a, norm=shared_norm)
    im2, _ = _draw_panel(ax2, hm_b, sm_b, dp_b, title=label_b, norm=shared_norm)

    fig.colorbar(im2, ax=[ax1, ax2], label="Avg agent-ticks per cell", shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"  Comparison heatmap saved → {out_file}")
    plt.close(fig)


# =============================================================================
# Clean layout figure (paper-style black/white)
# =============================================================================

def plot_layout(grid_rows=25, grid_cols=35,
                out_file="layout.png",
                **grid_kwargs):
    """
    Produce a clean black/white warehouse layout figure suitable for papers.

    Shelves = black  |  Aisles = white  |  Depot = dark square with 'D' label

    Cells are always equal-sized squares.
    """
    from grid import Grid, SHELF, ITEM, DEPOT, EMPTY

    g = Grid(rows=grid_rows, cols=grid_cols, **grid_kwargs)

    # Build binary image: 1 = shelf (black), 0 = walkable (white)
    img = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if g.cells[r, c] in (SHELF, ITEM):
                img[r, c] = 1.0

    # Figure size: each cell = 0.22 inches so cells are square.
    cell_in   = 0.22
    margin_t  = 0.15
    margin_r  = 0.15
    margin_bl = 0.15
    fig_w = grid_cols * cell_in + margin_r  + margin_bl
    fig_h = grid_rows * cell_in + margin_t  + margin_bl

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Position axes so grid cells are exactly square
    ax = fig.add_axes([
        margin_bl / fig_w,          # left
        margin_bl / fig_h,          # bottom
        grid_cols * cell_in / fig_w, # width
        grid_rows * cell_in / fig_h, # height
    ])

    ax.imshow(img, cmap="binary", aspect="equal", origin="upper",
              extent=[-0.5, grid_cols - 0.5, grid_rows - 0.5, -0.5],
              vmin=0, vmax=1, interpolation="nearest")

    # Thin cell grid lines
    for r in range(grid_rows + 1):
        ax.axhline(r - 0.5, color="lightgrey", linewidth=0.25, zorder=2)
    for c in range(grid_cols + 1):
        ax.axvline(c - 0.5, color="lightgrey", linewidth=0.25, zorder=2)

    # Depot markers
    for dr, dc in g.depots:
        ax.add_patch(plt.Rectangle((dc - 0.5, dr - 0.5), 1, 1,
                                   facecolor="#444444", zorder=3))
        ax.text(dc, dr, "D", ha="center", va="center",
                fontsize=max(4, cell_in * 18), color="white",
                fontweight="bold", zorder=4)

    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"  Layout figure saved → {out_file}")
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Conflict heatmap visualisation")
    parser.add_argument("--file",    default="conflict_heatmap.csv",
                        help="CSV file to read (default: conflict_heatmap.csv)")
    parser.add_argument("--out",     default="conflict_heatmap.png",
                        help="Output PNG filename (default: conflict_heatmap.png)")
    parser.add_argument("--rows",    type=int, default=25, help="Grid rows (default 25)")
    parser.add_argument("--cols",    type=int, default=35, help="Grid cols (default 35)")
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
                        help="Compare two heatmap CSVs side-by-side")
    parser.add_argument("--labels",  nargs=2, metavar=("LABEL_A", "LABEL_B"),
                        default=["Layout A", "Layout B"],
                        help="Labels for comparison panels")
    args = parser.parse_args()

    if args.compare:
        plot_comparison(args.compare[0], args.compare[1],
                        label_a=args.labels[0], label_b=args.labels[1],
                        grid_rows=args.rows, grid_cols=args.cols,
                        out_file=args.out if args.out != "conflict_heatmap.png"
                                 else "conflict_heatmap_comparison.png")
    else:
        plot_from_csv(filename=args.file, grid_rows=args.rows, grid_cols=args.cols,
                      out_file=args.out)


if __name__ == "__main__":
    main()
