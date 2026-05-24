# =============================================================================
# dashboard.py  Warehouse Simulation Dashboard  Imperial College London
# Run:   streamlit run dashboard.py
# Save:  python main.py --experiment --save --name "my_run"
# =============================================================================
import os, sys, json
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Warehouse Simulation - Imperial College London",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Imperial College London colours
NAVY      = "#002147"
BLUE      = "#003E74"
LBLUE     = "#D4EFFC"
LGREY     = "#EBEEEE"
CGREY     = "#9D9D9D"
DGREY     = "#373A36"
WHITE     = "#FFFFFF"
GOOD      = "#1B5E20"
AMBER_COL = "#E65100"
BAD       = "#B71C1C"
SERIES    = ["#003E74","#0277BD","#00838F","#558B2F",
             "#6A1B9A","#F57F17","#1565C0","#2E7D32"]

# =============================================================================
# CSS
# =============================================================================
st.markdown(f"""
<style>
.stApp {{background-color:#F7F9FB;}}
[data-testid="stSidebar"] {{background-color:{NAVY};}}
[data-testid="stSidebar"] * {{color:white !important;}}
[data-testid="stSidebar"] [data-baseweb="select"] {{background-color:rgba(255,255,255,0.12) !important;border-color:rgba(255,255,255,0.25) !important;border-radius:4px;}}
[data-testid="stSidebar"] [data-baseweb="select"] * {{color:white !important;}}
[data-testid="stSidebar"] [data-baseweb="popover"] * {{color:{NAVY} !important;background-color:white;}}
[data-testid="stSidebar"] .stButton>button {{
    background-color:{BLUE};border:1px solid {LBLUE};
    border-radius:4px;width:100%;color:white !important;}}
[data-testid="stSidebar"] .stButton>button:hover {{background-color:#004A9A;}}
.page-header {{
    background:linear-gradient(135deg,{NAVY} 0%,{BLUE} 100%);
    color:white;padding:22px 28px;border-radius:8px;margin-bottom:20px;}}
.page-header h1 {{margin:0 0 5px 0;font-size:24px;font-weight:700;color:white;}}
.page-header p  {{margin:0;font-size:12px;color:{LBLUE};opacity:0.9;}}
.sec {{font-size:13px;font-weight:700;color:{NAVY};text-transform:uppercase;
    letter-spacing:0.7px;border-bottom:2px solid {LBLUE};
    padding-bottom:5px;margin-bottom:14px;}}
.kpi-card {{background:white;border-radius:8px;padding:20px 18px 16px 18px;
    border-left:5px solid {BLUE};box-shadow:0 1px 4px rgba(0,33,71,0.09);}}
.kpi-card.good    {{border-left-color:{GOOD};}}
.kpi-card.amber   {{border-left-color:{AMBER_COL};}}
.kpi-card.bad     {{border-left-color:{BAD};}}
.kpi-card.neutral {{border-left-color:{BLUE};}}
.kpi-label {{font-size:11px;font-weight:700;color:{CGREY};
    text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;}}
.kpi-value {{font-size:36px;font-weight:800;color:{NAVY};
    line-height:1;margin-bottom:4px;}}
.kpi-unit  {{font-size:14px;font-weight:400;color:{CGREY};margin-left:3px;}}
.kpi-badge {{display:inline-block;padding:3px 10px;border-radius:20px;
    font-size:11px;font-weight:700;margin:7px 0;}}
.badge-good    {{background:#E8F5E9;color:{GOOD};}}
.badge-amber   {{background:#FFF3E0;color:{AMBER_COL};}}
.badge-bad     {{background:#FFEBEE;color:{BAD};}}
.badge-neutral {{background:{LBLUE};color:{BLUE};}}
.kpi-desc {{font-size:12px;color:{CGREY};line-height:1.5;margin-top:5px;}}
.kpi-dir  {{font-size:11px;color:{CGREY};margin-top:3px;font-style:italic;}}
.cfg-card {{background:white;border-radius:8px;padding:18px;
    box-shadow:0 1px 4px rgba(0,33,71,0.09);}}
.cfg-grp  {{font-size:11px;font-weight:700;color:{BLUE};text-transform:uppercase;
    letter-spacing:0.6px;margin:12px 0 5px 0;
    border-bottom:1px solid {LGREY};padding-bottom:3px;}}
.cfg-grp:first-child {{margin-top:0;}}
.cfg-row  {{display:flex;justify-content:space-between;
    align-items:center;padding:3px 0;font-size:13px;}}
.cfg-k    {{color:{DGREY};}}
.cfg-v    {{color:{NAVY};font-weight:600;}}
.exp-card {{background:white;border-radius:8px;padding:18px;
    box-shadow:0 1px 4px rgba(0,33,71,0.09);}}
.exp-stat {{background:{LGREY};border-radius:6px;padding:10px 14px;margin-bottom:6px;}}
.exp-sv   {{font-size:22px;font-weight:700;color:{NAVY};}}
.exp-sl   {{font-size:11px;color:{CGREY};text-transform:uppercase;letter-spacing:0.5px;}}
.exp-cap  {{font-size:11px;color:{CGREY};margin-top:2px;}}
.exp-body {{font-size:12px;color:{DGREY};line-height:1.6;}}
.exp-h    {{font-size:13px;font-weight:700;color:{NAVY};margin:14px 0 8px 0;}}
.exp-h:first-child {{margin-top:0;}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PLOTLY HELPERS
# =============================================================================
DEF_LEGEND = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11))

def plotly_layout(title="", height=370):
    return dict(
        title=dict(text=title, font=dict(color=NAVY, size=12), x=0),
        height=height, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(family="Arial, sans-serif", size=11, color=DGREY),
        legend=DEF_LEGEND,
        margin=dict(l=52, r=20, t=46, b=42),
        xaxis=dict(gridcolor=LGREY, linecolor=LGREY, zerolinecolor=LGREY,
                   tickfont=dict(size=10)),
        yaxis=dict(gridcolor=LGREY, linecolor=LGREY, zerolinecolor=LGREY,
                   tickfont=dict(size=10)),
    )

def shade(fig):
    for x0, x1, col in [(0,2,"rgba(180,30,30,0.05)"),
                         (2,6,"rgba(30,80,180,0.05)"),
                         (6,8,"rgba(180,30,30,0.05)")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=col, layer="below", line_width=0)
    for xv in (2, 6):
        fig.add_vline(x=xv, line_dash="dot", line_color=CGREY, line_width=1)

# =============================================================================
# DATA HELPERS
# =============================================================================
@st.cache_data
def get_saved_runs():
    if not os.path.exists(RESULTS_DIR):
        return []
    return [n for n in sorted(os.listdir(RESULTS_DIR), reverse=True)
            if os.path.exists(os.path.join(RESULTS_DIR, n, "run_summary.json"))]

@st.cache_data
def load_summary(rd):
    with open(os.path.join(rd, "run_summary.json")) as f:
        return json.load(f)

@st.cache_data
def load_all_kpis():
    if not os.path.exists(RESULTS_DIR):
        return {}
    out = {}
    for name in os.listdir(RESULTS_DIR):
        p = os.path.join(RESULTS_DIR, name, "run_summary.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    out[name] = json.load(f)["kpis"]
            except Exception:
                pass
    return out

@st.cache_data
def load_traces_cached(csv_path):
    from plot_agent_traces import load_traces
    return dict(load_traces(csv_path))

@st.cache_data
def load_heatmap_cached(csv_path):
    from plot_heatmap import _load_csv
    return _load_csv(csv_path)

@st.cache_data
def load_workload_cached(npz_path):
    npz = np.load(npz_path)
    return npz["arrivals"], npz["completions"]

def per_bin(records, field):
    vals = [r[field] for r in records]
    return [vals[i] - vals[i-1] if i > 0 else vals[0] for i in range(len(vals))]

# =============================================================================
# KPI RATING  (compare against all saved runs)
# =============================================================================
def kpi_rating(current, all_kpis, key, higher_is_better=True, current_name=None):
    entries = {n: v[key] for n, v in all_kpis.items() if key in v}
    if len(entries) < 2:
        return "neutral", "Only run saved"

    # With exactly 2 runs: compare directly against the other run
    if len(entries) == 2:
        other_name, other_val = next(
            (n, v) for n, v in entries.items() if n != current_name)
        if other_val == 0:
            return "neutral", "No comparison"
        pct = (current - other_val) / other_val * 100
        short = other_name[:18] + "…" if len(other_name) > 18 else other_name
        if higher_is_better:
            if pct >  1: return "good",  f"+{pct:.1f}% vs {short}"
            if pct < -1: return "bad",   f"{pct:.1f}% vs {short}"
            return "amber", f"Tied ({pct:+.1f}%) vs {short}"
        else:
            if pct < -1: return "good",  f"{abs(pct):.1f}% better vs {short}"
            if pct >  1: return "bad",   f"+{pct:.1f}% worse vs {short}"
            return "amber", f"Tied ({pct:+.1f}%) vs {short}"

    # 3+ runs: compare against the mean of all saved runs
    mean = float(np.mean(list(entries.values())))
    if mean == 0:
        return "neutral", "No comparison"
    pct = (current - mean) / mean * 100
    if higher_is_better:
        if pct >  5: return "good",  f"+{pct:.1f}% above avg"
        if pct < -5: return "bad",   f"{pct:.1f}% below avg"
        return "amber", f"Near average ({pct:+.1f}%)"
    else:
        if pct < -5: return "good",  f"{abs(pct):.1f}% below avg"
        if pct >  5: return "bad",   f"+{pct:.1f}% above avg"
        return "amber", f"Near average ({pct:+.1f}%)"

def kpi_card(label, val, unit, status, badge, desc, direction):
    return f"""<div class="kpi-card {status}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}<span class="kpi-unit">{unit}</span></div>
      <span class="kpi-badge badge-{status}">{badge}</span>
      <div class="kpi-desc">{desc}</div>
      <div class="kpi-dir">{direction}</div>
    </div>"""

def cfg_row(k, v):
    return (f'<div class="cfg-row">'
            f'<span class="cfg-k">{k}</span>'
            f'<span class="cfg-v">{v}</span></div>')

def cfg_grp(t):
    return f'<div class="cfg-grp">{t}</div>'

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 18px 0;">
      <div style="font-size:17px;font-weight:800;letter-spacing:0.5px;">WAREHOUSE SIM</div>
      <div style="font-size:10px;opacity:0.65;margin-top:3px;">Imperial College London</div>
    </div>""", unsafe_allow_html=True)

    runs = get_saved_runs()
    if not runs:
        st.warning("No saved runs found.")
        st.markdown("Save a run first:\n```\npython main.py --experiment "
                    "--save --name \"my_run\"\n```")
        st.stop()

    selected = st.selectbox("Select run", runs, label_visibility="collapsed")
    run_dir  = os.path.join(RESULTS_DIR, selected)
    summary  = load_summary(run_dir)
    cfg      = summary["config"]
    kpis     = summary["kpis"]
    raw      = summary["raw_averages"]

    st.markdown(f"""
    <div style="font-size:11px;opacity:0.6;margin-top:6px;line-height:1.7;">
      {summary['timestamp'][:16].replace('T',' ')}<br>
      Grid: {cfg['rows']}&times;{cfg['cols']}
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Refresh run list"):
        get_saved_runs.clear()
        load_all_kpis.clear()
        st.rerun()

all_kpis = load_all_kpis()

# =============================================================================
# TABS
# =============================================================================
tab_ov, tab_hm, tab_wl, tab_tr, tab_rl = st.tabs([
    "Overview", "Conflict Heatmap", "Workload Analysis",
    "Agent Traces", "RL Training"
])

# =============================================================================
# TAB 1 - OVERVIEW
# =============================================================================
with tab_ov:
    st.markdown(f"""<div class="page-header">
      <h1>Warehouse Simulation Results</h1>
      <p>Run: <strong>{summary['name']}</strong> &nbsp;&middot;&nbsp;
         {summary['timestamp'][:16].replace('T',' ')} &nbsp;&middot;&nbsp;
         Averaged over {cfg['num_runs']} independent simulation runs</p>
    </div>""", unsafe_allow_html=True)

    # Row 1: Layout + Config
    col_lay, col_cfg = st.columns([3, 2], gap="medium")

    with col_lay:
        st.markdown('<div class="sec">Warehouse Layout</div>',
                    unsafe_allow_html=True)
        try:
            from grid import Grid, EMPTY, SHELF, ITEM, DEPOT
            g = Grid(
                rows=cfg["rows"], cols=cfg["cols"],
                aisle_width=cfg["aisle_width"],
                centre_aisle_width=cfg["centre_aisle"],
                depot_row=cfg.get("depot_row", 0),
                depot_col=cfg.get("depot_col", cfg["cols"] // 2),
                shelf_start_row=cfg.get("shelf_start"),
                shelf_end_row=cfg.get("shelf_end"),
                cross_aisle_row=cfg.get("cross_aisle_row"),
            )
            img = np.ones((g.rows, g.cols, 3))
            for r in range(g.rows):
                for c in range(g.cols):
                    cell = g.cells[r, c]
                    if cell == DEPOT:
                        img[r, c] = [0.00, 0.13, 0.28]
                    elif cell in (SHELF, ITEM):
                        img[r, c] = [0.22, 0.23, 0.27]
                    else:
                        img[r, c] = [0.91, 0.95, 0.98]

            fig_l, ax = plt.subplots(figsize=(9, 6))
            fig_l.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.imshow(img, aspect="auto", origin="upper",
                      extent=[-0.5, g.cols-0.5, g.rows-0.5, -0.5])
            dr, dc = g.depot
            ax.plot(dc, dr, marker="s", markersize=10, color=LBLUE,
                    markeredgecolor=NAVY, markeredgewidth=1.5, zorder=5)
            ax.text(dc, dr, "D", ha="center", va="center",
                    fontsize=6, color=NAVY, fontweight="bold", zorder=6)
            ax.set_title(
                f"{g.rows} rows \u00d7 {g.cols} cols",
                fontsize=10, color=DGREY, pad=8)
            ax.set_xlabel("Column", fontsize=9, color=CGREY)
            ax.set_ylabel("Row",    fontsize=9, color=CGREY)
            ax.tick_params(colors=CGREY, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(LGREY)
            ax.legend(handles=[
                Patch(facecolor=[0.91,0.95,0.98], edgecolor="lightgray",
                      label="Floor / Aisle"),
                Patch(facecolor=[0.22,0.23,0.27], label="Shelf / Item"),
                Patch(facecolor=[0.00,0.13,0.28], label="Depot"),
            ], loc="upper right", fontsize=8, framealpha=0.9, edgecolor=LGREY)
            plt.tight_layout()
            st.pyplot(fig_l)
            plt.close(fig_l)
        except Exception as e:
            st.error(f"Could not render layout: {e}")

    with col_cfg:
        st.markdown('<div class="sec">Configuration</div>',
                    unsafe_allow_html=True)
        depot_r = cfg.get("depot_row", 0)
        depot_c = cfg.get("depot_col", cfg["cols"] // 2)
        shelf_s = cfg.get("shelf_start", 1)
        shelf_e = cfg.get("shelf_end", cfg["rows"] - 2)
        h  = '<div class="cfg-card">'
        h += cfg_grp("Grid Dimensions")
        h += cfg_row("Rows",         cfg["rows"])
        h += cfg_row("Columns",      cfg["cols"])
        h += cfg_row("Total cells",  cfg["rows"] * cfg["cols"])
        h += cfg_grp("Agents & Simulation")
        h += cfg_row("Number of agents", cfg["agents"])
        h += cfg_row("Simulation runs",  cfg["num_runs"])
        h += cfg_row("Shift duration",   "8 hours")
        h += cfg_row("Time resolution",  "1 tick = 1 second")
        h += cfg_grp("Layout Parameters")
        h += cfg_row("Aisle width",        f"{cfg['aisle_width']} cols")
        h += cfg_row("Centre aisle width", f"{cfg['centre_aisle']} cols")
        h += cfg_row("Depot position",
                     f"row {depot_r}, col {depot_c}")
        h += cfg_row("Shelf zone",         f"rows {shelf_s} to {shelf_e}")
        h += cfg_row("Cross-aisle row",
                     str(cfg.get("cross_aisle_row", "N/A")))
        h += '</div>'
        st.markdown(h, unsafe_allow_html=True)

    # Row 2: KPI Cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Key Performance Indicators</div>',
                unsafe_allow_html=True)

    ph_st, ph_bd = kpi_rating(kpis["picks_per_hour"],  all_kpis,
                               "picks_per_hour",  True,  selected)
    d_st,  d_bd  = kpi_rating(kpis["dist_per_agent"],  all_kpis,
                               "dist_per_agent",  False, selected)
    cg_st, cg_bd = kpi_rating(kpis["congestion_rate"], all_kpis,
                               "congestion_rate", False, selected)

    kc1, kc2, kc3 = st.columns(3, gap="medium")
    with kc1:
        st.markdown(kpi_card(
            "Picks per Hour", f"{kpis['picks_per_hour']:.2f}", "",
            ph_st, ph_bd,
            "Total completed picks divided by the 8-hour shift. "
            "Reflects overall throughput across all agents.",
            "Higher is better"), unsafe_allow_html=True)
    with kc2:
        st.markdown(kpi_card(
            "Avg Travel Distance / Agent",
            f"{kpis['dist_per_agent']:.1f}", "cells",
            d_st, d_bd,
            "Average grid cells walked per agent over the full shift. "
            "Shorter distances indicate a more efficient layout.",
            "Lower is better"), unsafe_allow_html=True)
    with kc3:
        st.markdown(kpi_card(
            "Congestion Rate",
            f"{kpis['congestion_rate']:.2f}", "%",
            cg_st, cg_bd,
            "Percentage of shift ticks where agents shared a cell, "
            "causing one to wait. Reflects aisle conflict frequency.",
            "Lower is better"), unsafe_allow_html=True)

    # Row 3: Raw averages
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Simulation Averages</div>',
                unsafe_allow_html=True)
    comp_pct = raw["jobs_completed"] / max(raw["jobs_arrived"], 1) * 100
    ra1, ra2, ra3, ra4 = st.columns(4, gap="small")
    ra1.metric("Jobs Arrived",   f"{int(raw['jobs_arrived']):,}",
               help="Total NHPP demand over the 8-hour shift")
    ra2.metric("Jobs Completed", f"{int(raw['jobs_completed']):,}",
               help=f"Completion rate: {comp_pct:.1f}%")
    ra3.metric("Total Distance", f"{int(raw['total_distance']):,} m",
               help="Total metres walked across all agents combined")
    ra4.metric("Slow-walk Ticks", f"{int(raw['cell_conflicts']):,}",
               help="Ticks agents spent below 20% free speed — congestion proxy")

# =============================================================================
# TAB 2 - CONFLICT HEATMAP
# =============================================================================
with tab_hm:
    st.markdown(f"""<div class="page-header">
      <h1>Spatial Conflict Heatmap</h1>
      <p>Average agent conflicts per grid cell overlaid on the warehouse layout.
         Brighter cells are congestion hot-spots.</p>
    </div>""", unsafe_allow_html=True)

    hm_csv = os.path.join(run_dir, "conflict_heatmap.csv")
    if not os.path.exists(hm_csv):
        st.info("No heatmap data for this run.")
    else:
        try:
            from plot_heatmap import _build_arrays, _make_grid_mask, _draw_panel
            data      = load_heatmap_cached(hm_csv)
            heatmap   = _build_arrays(data, cfg["rows"], cfg["cols"])
            gkw = dict(
                aisle_width=cfg["aisle_width"],
                centre_aisle_width=cfg["centre_aisle"],
                depot_row=cfg.get("depot_row", 0),
                depot_col=cfg.get("depot_col", cfg["cols"] // 2),
                shelf_start_row=cfg.get("shelf_start"),
                shelf_end_row=cfg.get("shelf_end"),
                cross_aisle_row=cfg.get("cross_aisle_row"),
            )
            shelf_mask, depot_pos = _make_grid_mask(
                cfg["rows"], cfg["cols"], **gkw)
            walkable = int((~shelf_mask).sum())
            vals = heatmap[heatmap > 0]

            col_hm, col_ex = st.columns([3, 1], gap="medium")
            with col_hm:
                fig_hm, ax_hm = plt.subplots(figsize=(11, 7))
                fig_hm.patch.set_facecolor("white")
                im, _ = _draw_panel(
                    ax_hm, heatmap, shelf_mask, depot_pos,
                    title=(f"Conflict Heatmap  {cfg['rows']}x{cfg['cols']} "
                           f"warehouse  ({cfg['num_runs']} runs averaged)"))
                plt.colorbar(im, ax=ax_hm,
                             label="Avg conflicts per cell", shrink=0.85)
                ax_hm.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_hm)
                plt.close(fig_hm)

            with col_ex:
                mean_str = f"{vals.mean():.2f}" if vals.size else "none"
                stats_html = ""
                for stat_name, stat_val, stat_cap in [
                    ("Peak conflict cell",
                     f"{heatmap.max():.1f}",
                     "Highest avg conflict count on any single cell"),
                    ("Mean (hot cells only)",
                     mean_str,
                     "Average among cells with at least one conflict"),
                    ("Cells with conflicts",
                     f"{int((heatmap > 0).sum())}",
                     "Walkable cells that recorded any conflict"),
                    ("Total walkable cells",
                     f"{walkable}",
                     "Floor and aisle cells agents can move through"),
                ]:
                    stats_html += (
                        f'<div class="exp-stat">'
                        f'<div class="exp-sv">{stat_val}</div>'
                        f'<div class="exp-sl">{stat_name}</div>'
                        f'<div class="exp-cap">{stat_cap}</div>'
                        f'</div>'
                    )
                st.markdown(f"""
<div class="exp-card">
  <div class="exp-h" style="margin-top:0">How to read this map</div>
  <div class="exp-body">Each cell shows the <em>average number of times</em>
  two or more agents occupied it simultaneously across all {cfg["num_runs"]} runs.
  <br><br>
  <strong>Yellow &rarr; Orange &rarr; Red</strong> = increasing conflict intensity.
  Dark grey = shelves (impassable). Light blue = conflict-free floor.</div>
  <div class="exp-h">Statistics</div>
  {stats_html}
  <div class="exp-h">What causes conflicts?</div>
  <div class="exp-body">Conflicts occur when agents converge on narrow aisles
  or bottlenecks near the depot. Hot-spots suggest layout changes — wider aisles,
  a repositioned depot, or an adjusted cross-aisle — could reduce congestion.</div>
</div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error rendering heatmap: {e}")

# =============================================================================
# TAB 3 - WORKLOAD ANALYSIS
# =============================================================================
with tab_wl:
    st.markdown(f"""<div class="page-header">
      <h1>NHPP Workload Analysis</h1>
      <p>Non-homogeneous Poisson Process demand model - arrival rate vs
         completion rate across the 8-hour shift.
         Averaged over {cfg['num_runs']} runs.</p>
    </div>""", unsafe_allow_html=True)

    wl_npz = os.path.join(run_dir, "workload_data.npz")
    if not os.path.exists(wl_npz):
        st.info("Workload data not available. "
                "Future runs saved with --save include it automatically.")
    else:
        try:
            from plot_workload import bin_data, theoretical_arrivals_per_bin
            from agent import SHIFT_TICKS, LAMBDA_BASE, DEMAND_BETA

            all_arr, all_comp = load_workload_cached(wl_npz)
            bin_size = 600
            bin_hrs  = bin_size / 3600
            avg_arr  = np.mean(all_arr,  axis=0)
            avg_comp = np.mean(all_comp, axis=0)
            arr_bin  = bin_data(avg_arr,  bin_size)
            comp_bin = bin_data(avg_comp, bin_size)
            theory   = theoretical_arrivals_per_bin(bin_size)
            n_bins   = len(arr_bin)
            x_left   = np.arange(n_bins) * bin_hrs
            x_mid    = x_left + bin_hrs / 2
            tick_hrs = np.arange(SHIFT_TICKS) / 3600
            cum_arr  = np.cumsum(avg_arr)
            cum_comp = np.cumsum(avg_comp)
            backlog  = cum_arr - cum_comp
            total_arr  = int(round(avg_arr.sum()))
            total_comp = int(round(avg_comp.sum()))
            comp_pct   = 100.0 * total_comp / max(total_arr, 1)

            sm1,sm2,sm3,sm4 = st.columns(4, gap="small")
            sm1.metric("Jobs Arrived (avg)",   f"{total_arr}")
            sm2.metric("Jobs Completed (avg)", f"{total_comp}")
            sm3.metric("Completion Rate",      f"{comp_pct:.1f}%")
            sm4.metric("Final Queue Backlog",  f"{int(round(backlog[-1]))}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="sec">Arrival Rate vs Completion Rate - '
                '10-min Bins</div>', unsafe_allow_html=True)

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=x_left, y=arr_bin, name="Actual arrivals",
                marker_color=BLUE, opacity=0.82,
                width=bin_hrs*0.42, offset=0))
            fig1.add_trace(go.Bar(
                x=x_left+bin_hrs*0.42, y=comp_bin, name="Completions",
                marker_color=GOOD, opacity=0.82,
                width=bin_hrs*0.42, offset=0))
            fig1.add_trace(go.Scatter(
                x=x_mid, y=theory,
                name="Theoretical lambda(t)*dt",
                mode="lines+markers",
                line=dict(color=AMBER_COL, width=2, dash="dash"),
                marker=dict(size=4)))
            shade(fig1)
            for xc, lbl in [(1,"Morning peak"),
                            (4,"Mid-shift lull"),
                            (7,"End-of-shift rush")]:
                fig1.add_annotation(
                    x=xc, y=1, yref="paper", text=lbl,
                    showarrow=False,
                    font=dict(size=9, color=CGREY), yanchor="top")
            fig1.update_layout(**plotly_layout(height=340), barmode="overlay")
            fig1.update_legends(orientation="h", y=1.12, x=0)
            fig1.update_xaxes(
                title="Shift time (hours)",
                tickvals=list(range(9)), range=[0,8])
            fig1.update_yaxes(
                title=f"Jobs per {bin_size//60}-min bin")
            st.plotly_chart(fig1, use_container_width=True)

            col_c, col_b = st.columns(2, gap="medium")
            with col_c:
                st.markdown(
                    '<div class="sec">Cumulative Arrivals vs Completions</div>',
                    unsafe_allow_html=True)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=tick_hrs, y=cum_arr, name="Cumulative arrived",
                    line=dict(color=BLUE, width=2)))
                fig2.add_trace(go.Scatter(
                    x=tick_hrs, y=cum_comp, name="Cumulative completed",
                    line=dict(color=GOOD, width=2),
                    fill="tonexty",
                    fillcolor="rgba(230,160,0,0.13)"))
                shade(fig2)
                fig2.update_layout(**plotly_layout(height=310))
                fig2.update_xaxes(
                    title="Shift time (hours)",
                    tickvals=list(range(9)), range=[0,8])
                fig2.update_yaxes(title="Cumulative jobs")
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                st.markdown(
                    '<div class="sec">Queue Backlog (Arrived minus '
                    'Completed)</div>', unsafe_allow_html=True)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=tick_hrs, y=backlog, name="Queue backlog",
                    line=dict(color=BAD, width=2),
                    fill="tozeroy",
                    fillcolor="rgba(183,28,28,0.11)"))
                shade(fig3)
                fig3.update_layout(**plotly_layout(height=310))
                fig3.update_xaxes(
                    title="Shift time (hours)",
                    tickvals=list(range(9)), range=[0,8])
                fig3.update_yaxes(
                    title="Jobs waiting in queue",
                    rangemode="tozero")
                st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering workload: {e}")

# =============================================================================
# TAB 4 - AGENT TRACES
# =============================================================================
with tab_tr:
    st.markdown(f"""<div class="page-header">
      <h1>Per-Agent Performance Traces</h1>
      <p>Individual agent behaviour averaged across {cfg['num_runs']} runs,
         broken into 10-minute bins over the 8-hour shift.</p>
    </div>""", unsafe_allow_html=True)

    traces_csv = os.path.join(run_dir, "agent_traces.csv")
    if not os.path.exists(traces_csv):
        st.info("No agent trace data found for this run.")
    else:
        try:
            agents_data = load_traces_cached(traces_csv)
            agent_ids   = sorted(agents_data.keys())
            ticks_hrs   = [r["tick"]/3600 for r in agents_data[agent_ids[0]]]

            panels = [
                ("picks",
                 "Picks per 10-min Bin",
                 "Orders completed per 10-min window per agent",
                 "Picks", False),
                ("distance",
                 "Distance per 10-min Bin",
                 "Cells walked per 10-min window per agent",
                 "Cells", False),
                ("idle_ticks",
                 "Idle Ticks per 10-min Bin",
                 "Ticks spent waiting for a job assignment",
                 "Ticks idle", True),
                ("blocked_events",
                 "Blocked Events per 10-min Bin",
                 "Conflict-induced wait events per window",
                 "Collisions", True),
            ]

            for i in range(0, len(panels), 2):
                cols = st.columns(2, gap="medium")
                for j, col in enumerate(cols):
                    if i+j >= len(panels):
                        break
                    field, title, subtitle, ylabel, zero_based = panels[i+j]
                    with col:
                        st.markdown(
                            f'<div class="sec">{title}</div>',
                            unsafe_allow_html=True)
                        st.caption(subtitle)
                        fig = go.Figure()
                        for idx, aid in enumerate(agent_ids):
                            fig.add_trace(go.Scatter(
                                x=ticks_hrs,
                                y=per_bin(agents_data[aid], field),
                                name=f"Agent {aid}",
                                mode="lines",
                                line=dict(
                                    color=SERIES[idx % len(SERIES)],
                                    width=1.8),
                                opacity=0.85))
                        layout = plotly_layout(height=300)
                        if zero_based:
                            layout["yaxis"]["rangemode"] = "tozero"
                        fig.update_layout(**layout)
                        fig.update_xaxes(
                            title="Shift time (hours)",
                            tickvals=list(range(9)),
                            range=[0, max(ticks_hrs)+0.1])
                        fig.update_yaxes(title=ylabel)
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering agent traces: {e}")

# =============================================================================
# TAB 5 - RL TRAINING
# =============================================================================
with tab_rl:
    st.markdown("""
    <div class="page-header"
         style="background:linear-gradient(135deg,#1B3A5C 0%,#2E5986 100%);">
      <h1>RL Layout Optimisation - Training Results</h1>
      <p>The PPO agent learns to adjust warehouse layout parameters to
         maximise picks per hour and minimise travel distance and congestion.</p>
    </div>""", unsafe_allow_html=True)

    st.warning("**Work in Progress** - RL model training is ongoing. "
               "Results reflect the current state of training.")

    training_csv = os.path.join(os.path.dirname(__file__), "training_log.csv")
    if not os.path.exists(training_csv):
        st.info("No training log found. Run python rl_env.py to begin RL training.")
    else:
        try:
            df = pd.read_csv(training_csv)
            i1, i2, i3 = st.columns(3)
            i1.metric("Episodes completed", f"{len(df)}")
            i2.metric("Best reward",
                      f"{df['reward'].max():.4f}"
                      if "reward" in df.columns else "n/a")
            i3.metric("Final avg (last 20)",
                      f"{df['reward'].tail(20).mean():.4f}"
                      if "reward" in df.columns else "n/a")

            window = st.slider(
                "Rolling average window (episodes)",
                min_value=5, max_value=50, value=20, step=5)

            st.markdown("<br>", unsafe_allow_html=True)

            rl_panels = [
                ("reward",          "Overall Reward (0 to 1)",         "#534AB7", True),
                ("picks_per_hour",  "Picks per Hour",                  "#1D9E75", True),
                ("dist_per_agent",  "Avg Distance / Agent (cells)",    "#D85A30", False),
                ("congestion_rate", "Congestion Rate",                 "#D4537E", False),
            ]
            for i in range(0, len(rl_panels), 2):
                cols = st.columns(2, gap="medium")
                for j, col in enumerate(cols):
                    if i+j >= len(rl_panels):
                        break
                    col_name, ylabel, color, hib = rl_panels[i+j]
                    if col_name not in df.columns:
                        continue
                    with col:
                        st.markdown(
                            f'<div class="sec">{ylabel}</div>',
                            unsafe_allow_html=True)
                        rolling  = df[col_name].rolling(
                            window=window, min_periods=1).mean()
                        start_v  = rolling.iloc[min(window, len(rolling)-1)]
                        end_v    = rolling.iloc[-1]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df["episode"], y=df[col_name],
                            mode="markers",
                            marker=dict(color=color, size=3, opacity=0.2),
                            showlegend=False))
                        fig.add_trace(go.Scatter(
                            x=df["episode"], y=rolling,
                            mode="lines",
                            line=dict(color=color, width=2),
                            name=f"Rolling avg (w={window})"))
                        direction = ("Higher is better"
                                     if hib else "Lower is better")
                        fig.update_layout(
                            **plotly_layout(height=280),
                            title=dict(
                                text=(f"{direction}  -  "
                                      f"Start {start_v:.3f} to End {end_v:.3f}"),
                                font=dict(size=11, color=CGREY), x=0))
                        fig.update_xaxes(title="Episode")
                        fig.update_yaxes(title=ylabel)
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_p, col_n = st.columns(2, gap="medium")

            with col_p:
                st.markdown(
                    '<div class="sec">Layout Parameters Explored by Agent</div>',
                    unsafe_allow_html=True)
                params = [
                    ("aisle_width",        "Aisle width",      "#534AB7"),
                    ("centre_aisle_width", "Centre aisle",     "#1D9E75"),
                    ("depot_col",          "Depot col",        "#D85A30"),
                    ("shelf_start_row",    "Shelf start row",  "#378ADD"),
                    ("shelf_end_row",      "Shelf end row",    "#D4537E"),
                    ("cross_aisle_row",    "Cross-aisle row",  "#E8A020"),
                ]
                fig_p = go.Figure()
                for cn, label, color in params:
                    if cn in df.columns:
                        rolling = df[cn].rolling(
                            window=window, min_periods=1).mean()
                        fig_p.add_trace(go.Scatter(
                            x=df["episode"], y=rolling,
                            mode="lines", name=label,
                            line=dict(color=color, width=1.8)))
                fig_p.update_layout(**plotly_layout(height=300))
                fig_p.update_xaxes(title="Episode")
                fig_p.update_yaxes(title="Parameter value")
                st.plotly_chart(fig_p, use_container_width=True)

            with col_n:
                st.markdown(
                    '<div class="sec">Normalised KPI Contributions to Reward</div>',
                    unsafe_allow_html=True)
                fig_n = go.Figure()
                for cn, label, color in [
                    ("norm_picks",      "Norm picks/hr",   "#1D9E75"),
                    ("norm_distance",   "Norm distance",   "#D85A30"),
                    ("norm_congestion", "Norm congestion", "#D4537E"),
                ]:
                    if cn in df.columns:
                        rolling = df[cn].rolling(
                            window=window, min_periods=1).mean()
                        fig_n.add_trace(go.Scatter(
                            x=df["episode"], y=rolling,
                            mode="lines", name=label,
                            line=dict(color=color, width=1.8)))
                fig_n.update_layout(**plotly_layout(height=300))
                fig_n.update_xaxes(title="Episode")
                fig_n.update_yaxes(
                    title="Normalised value (0 to 1)",
                    range=[-0.05, 1.05])
                st.plotly_chart(fig_n, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering RL training results: {e}")
