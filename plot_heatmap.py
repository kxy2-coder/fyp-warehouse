# =============================================================================
# plot_heatmap.py — Spatial Traffic / Congestion Heatmap
# =============================================================================
# Reads a per-cell heatmap CSV (row, col, avg_count) and overlays it on the
# warehouse grid layout. Auto-styles the colour bar based on filename:
#
#   traffic_heatmap.csv     → "Avg walking time (s)"
#   congestion_heatmap.csv  → "Avg congestion time (s)"
#
# Supports a side-by-side comparison mode for two layouts (e.g. baseline
# vs PPO-optimised) using the --compare flag.
#
# Usage:
#   python plot_heatmap.py                                  # default: traffic_heatmap.csv
#   python plot_heatmap.py --file congestion_heatmap.csv
#   python plot_heatmap.py --rows 25 --cols 35
#   python plot_heatmap.py --shelf-start 2 --shelf-end 23
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
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelbottom=False, labelleft=False, length=0)

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
        suptitle  = "Congestion Hotspot Heatmap"
        cbar_label = "Avg congestion time (s)"
    else:
        suptitle  = "Traffic Density Heatmap\n(Average time spent walking through each cell)"
        cbar_label = "Avg walking time (s)"

    fig, ax = plt.subplots(figsize=(10, 7))

    im, _ = _draw_panel(ax, heatmap, shelf_mask, depot_pos, title=suptitle)

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
                annotate=True,
                **grid_kwargs):
    """
    Produce a clean black/white warehouse layout figure suitable for papers.

    Shelves = black  |  Aisles = white  |  Depot = dark square with 'D' label

    Cells are always equal-sized squares.

    annotate=True overlays parameter labels (w_a, w_c, n_d, r_s, r_e, r_ca, b_ca)
    to match Table 3.1 in the methodology section.
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
    # Larger margins reserve space for annotation labels around the grid edges.
    cell_in   = 0.30 if annotate else 0.22
    margin_t  = 0.60 if annotate else 0.15
    margin_r  = 1.50 if annotate else 0.15
    margin_l  = 1.00 if annotate else 0.15
    margin_b  = 0.40 if annotate else 0.15
    fig_w = grid_cols * cell_in + margin_r + margin_l
    fig_h = grid_rows * cell_in + margin_t + margin_b

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Position axes so grid cells are exactly square
    ax = fig.add_axes([
        margin_l / fig_w,            # left
        margin_b / fig_h,            # bottom
        grid_cols * cell_in / fig_w, # width
        grid_rows * cell_in / fig_h, # height
    ])

    ax.imshow(img, cmap="binary", aspect="equal", origin="upper",
              extent=[-0.5, grid_cols - 0.5, grid_rows - 0.5, -0.5],
              vmin=0, vmax=1, interpolation="nearest")

    # Cell grid lines (more visible for clarity)
    for r in range(grid_rows + 1):
        ax.axhline(r - 0.5, color="#888888", linewidth=0.5, zorder=2)
    for c in range(grid_cols + 1):
        ax.axvline(c - 0.5, color="#888888", linewidth=0.5, zorder=2)

    # Depot markers
    for dr, dc in g.depots:
        ax.add_patch(plt.Rectangle((dc - 0.5, dr - 0.5), 1, 1,
                                   facecolor="#444444", zorder=3))
        ax.text(dc, dr, "D", ha="center", va="center",
                fontsize=max(4, cell_in * 18), color="white",
                fontweight="bold", zorder=4)

    # ── Parameter annotations ──────────────────────────────────────────────
    if annotate:
        LBL_FS    = 14         # font size for variable labels
        ARR_COL   = "black"    # bold black for arrows/labels
        ARR_KW    = dict(arrowstyle="<->", color=ARR_COL, linewidth=1.6)
        TXT_KW    = dict(fontsize=LBL_FS, color=ARR_COL,
                         fontweight="bold", clip_on=False)

        # ---- Find one pick aisle and the centre aisle ----
        # Scan a row that DOES contain shelves so aisle gaps can be detected.
        # Avoid the cross-aisle band (rows cleared horizontally) and shelf-zone
        # boundary rows by picking a row inside the upper shelf block.
        ca_top  = g.cross_aisle_row if g.cross_aisle_row is not None else g.shelf_end_row
        ca_bot  = (g.cross_aisle_row + g.cross_aisle_width - 1
                   if g.cross_aisle_row is not None else g.shelf_end_row)
        shelf_scan_row = (g.shelf_start_row + ca_top - 1) // 2
        # Guard: if scan row landed in the cross-aisle band, nudge it up
        if g.shelf_start_row <= shelf_scan_row <= ca_top - 1:
            pass  # already valid
        else:
            shelf_scan_row = g.shelf_start_row + 1
        cells_row = g.cells[shelf_scan_row, :]
        in_aisle = (cells_row != SHELF) & (cells_row != ITEM)

        # Find contiguous aisle runs (col ranges where in_aisle is True)
        aisle_runs = []
        start = None
        for c in range(grid_cols):
            if in_aisle[c] and start is None:
                start = c
            elif (not in_aisle[c]) and start is not None:
                aisle_runs.append((start, c - 1))
                start = None
        if start is not None:
            aisle_runs.append((start, grid_cols - 1))

        # Centre aisle = widest aisle run; pick aisle = a narrow one (not boundary)
        centre_run = max(aisle_runs, key=lambda r: r[1] - r[0])
        narrow_runs = [r for r in aisle_runs
                       if r != centre_run and r[0] > 0 and r[1] < grid_cols - 1]
        pick_run = narrow_runs[0] if narrow_runs else aisle_runs[0]

        # ---- w_a: pick aisle width (arrow INSIDE the aisle, near shelf-start) ----
        y_w = g.shelf_start_row + 0.5     # one row inside the shelf zone
        ax.annotate("", xy=(pick_run[0] - 0.5, y_w),
                    xytext=(pick_run[1] + 0.5, y_w),
                    arrowprops=ARR_KW, annotation_clip=False)
        ax.text((pick_run[0] + pick_run[1]) / 2, y_w + 0.9,
                r"$w_a$", ha="center", va="top", **TXT_KW)

        # ---- w_c: centre aisle width (arrow INSIDE the centre aisle) ----
        ax.annotate("", xy=(centre_run[0] - 0.5, y_w),
                    xytext=(centre_run[1] + 0.5, y_w),
                    arrowprops=ARR_KW, annotation_clip=False)
        ax.text((centre_run[0] + centre_run[1]) / 2, y_w + 0.9,
                r"$w_c$", ha="center", va="top", **TXT_KW)

        # ---- n_d: depot count (label near the first depot) ----
        n_d = len(g.depots)
        first_dr, first_dc = g.depots[0]
        ax.text(first_dc, first_dr - 1.2,
                rf"$n_d = {n_d}$", ha="center", va="bottom", **TXT_KW)

        # ---- r_s and r_e: horizontal arrows aligned with the GRIDLINE at the
        # top of each row, so the arrow visually rests on a grid line. The tip
        # stops exactly at the grid boundary (x = -0.5).
        ROW_ARR = dict(arrowstyle="->", color=ARR_COL, linewidth=1.6,
                       mutation_scale=18)

        # r_s: top gridline of the first shelf row (y = shelf_start_row - 0.5)
        y_rs = g.shelf_start_row - 0.5
        ax.annotate("", xy=(-0.5, y_rs),
                    xytext=(-2.5, y_rs),
                    arrowprops=ROW_ARR, annotation_clip=False)
        ax.text(-2.7, y_rs,
                rf"$r_s = {g.shelf_start_row}$", ha="right", va="center",
                **TXT_KW)

        # r_e: bottom gridline of the last shelf row (y = shelf_end_row + 0.5)
        y_re = g.shelf_end_row + 0.5
        ax.annotate("", xy=(-0.5, y_re),
                    xytext=(-2.5, y_re),
                    arrowprops=ROW_ARR, annotation_clip=False)
        ax.text(-2.7, y_re,
                rf"$r_e = {g.shelf_end_row}$", ha="right", va="center",
                **TXT_KW)

        # ---- r_ca and b_ca: horizontal arrow on the right pointing at the
        # gridline at the top of the cross-aisle row.
        if g.cross_aisle_row is not None:
            y_rca = g.cross_aisle_row - 0.5
            ax.annotate("", xy=(grid_cols - 0.5, y_rca),
                        xytext=(grid_cols + 1.5, y_rca),
                        arrowprops=ROW_ARR, annotation_clip=False)
            ax.text(grid_cols + 1.7, y_rca,
                    rf"$r_{{ca}} = {g.cross_aisle_row}$  $(b_{{ca}} = 1)$",
                    ha="left", va="center", **TXT_KW)
        else:
            ax.text(grid_cols + 1.7, (g.shelf_start_row + g.shelf_end_row) / 2,
                    r"$b_{ca} = 0$ (off)",
                    ha="left", va="center", **TXT_KW)

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
    parser.add_argument("--file",    default="traffic_heatmap.csv",
                        help="CSV file to read (default: traffic_heatmap.csv)")
    parser.add_argument("--out",     default=None,
                        help="Output PNG filename (default: derived from input filename)")
    parser.add_argument("--rows",        type=int, default=25, help="Grid rows (default 25)")
    parser.add_argument("--cols",        type=int, default=35, help="Grid cols (default 35)")
    parser.add_argument("--shelf-start", type=int, default=2,  help="First shelf row (default 2)")
    parser.add_argument("--shelf-end",   type=int, default=23, help="Last shelf row (default 23)")
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
        out = args.out or args.file.replace(".csv", ".png")
        plot_from_csv(filename=args.file, grid_rows=args.rows, grid_cols=args.cols,
                      out_file=out,
                      shelf_start_row=args.shelf_start,
                      shelf_end_row=args.shelf_end)


if __name__ == "__main__":
    main()
