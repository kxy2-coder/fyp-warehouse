# Warehouse Layout Optimisation Using Reinforcement Learning

This repository contains the simulation, reinforcement learning, and analysis code for a final-year project on automated warehouse layout design. 

---

## Quickstart

```bash
# Create environment (Python 3.12, conda recommended)
conda create -n sim_v3 python=3.12
conda activate sim_v3
pip install -r requirements.txt

# 1. Calibrate KPI normalisation bounds
python compress.py --calibrate --cal-samples 100

# 2. Train the PPO agent
python rl_env.py --timesteps 10000

# 3. Visualise training progress
python plot_training.py

# 4. Run a single deterministic simulation for inspection
python main.py --agents 6
```

---

## File-by-file overview

### Core simulation backbone

| File | Role |
|---|---|
| `grid.py` | Builds the warehouse grid (shelves, aisles, depots, cross-aisle). All layout parameters are constructor arguments — no magic numbers. |
| `jupedsim_agent.py` | Picker movement using the JuPedSim Collision-Free Speed Model (Tordeux et al., 2016). Handles continuous-space navigation, physical collision avoidance, congestion detection, and stuck-detection sidestep. |
| `agent.py` | Demand model (`nhpp_arrival`, `get_multiplier`) + shared constants (`SHIFT_TICKS`, `LAMBDA_BASE`, state-machine integers). Also contains a legacy Sim_V2 `Agent` class kept solely for `plot_workload.py`. |
| `pathfinder.py` | A\* shortest-path used by the legacy `Agent` class. Not used by the RL pipeline. |
| `metrics.py` | `TraceLogger` (per-bin KPI snapshots → `agent_traces.csv`) and `MetricsTracker` (run-level totals). |

### Entry points

| File | Role |
|---|---|
| `main.py` | Pygame-based interactive visualiser. Runs a single simulation with on-screen rendering of the grid, agents, paths, and KPIs. Use this for live demos and debugging. |
| `rl_env.py` | Gymnasium environment wrapping the simulator + PPO training loop (Stable Baselines 3). Saves `best_model.zip`, `training_log.csv`, and runs a convergence test from the default layout at the end of training. |
| `compress.py` | (a) Validates that the 3600-tick compressed simulation used during training agrees with the full 8-hour shift, and (b) calibrates `kpi_bounds.json` from random layouts (`--calibrate`). |
| `record_replay.py` | Headless simulation runner that exports every agent position to a JSON replay file for the Godot 3D visualiser. |

### Analysis & plotting

| File | Role |
|---|---|
| `sensitivity_analysis.py` | One-at-a-time sensitivity run over all seven layout decision variables. Produces a 4×2 KPI response figure and a ranked sensitivity bar chart. Re-plot from CSV with `--plot-only`. |
| `plot_training.py` | Reward, KPI, and layout-parameter curves over training episodes. |
| `plot_training_compare.py` | Overlays reward curves across multiple `(N agents, λ)` runs stored under `Results_final/`. |
| `plot_agent_traces.py` | 2×2 per-agent figure (picks, distance, idle time, congestion time) from `agent_traces.csv`. Also plots per-tick walking speed from `speed_trace.csv`. |
| `plot_heatmap.py` | Spatial overlay of `traffic_heatmap.csv` or `congestion_heatmap.csv` on the warehouse grid. Supports side-by-side comparison via `--compare`. |
| `plot_workload.py` | NHPP demand visualisation — arrivals vs completions over a shift. |

---

## Decision variables and ranges (used by the RL agent)

| Symbol | Variable | Range | Step |
|---|---|---|---|
| `w_a` | Aisle width | [2, 3] | ±1 |
| `w_c` | Centre aisle width | [1, 5] | ±2 |
| `n_d` | Depot count | [1, 4] | ±1 |
| `r_s` | Shelf start row | [2, 8] | ±1 |
| `r_e` | Shelf end row | [18, 23] | ±1 |
| `r_ca` | Cross-aisle row | shelf-zone interior | ±1 |
| `b_ca` | Cross-aisle enabled | on / off | binary |

Depot column positions are auto-derived from `n_d` using the equal-spacing rule.

---

## KPI definitions

| KPI | Definition | Direction |
|---|---|---|
| Picks per hour | Jobs completed / total shift hour | higher better |
| Distance per agent |  Travel distance per agent | lower better |
| Congestion rate | Fraction of walking ticks where agent speed < 0.6 m/s **and** another agent within 0.9 m | lower better |

The reward used by PPO is a weighted sum after each KPI is normalised to `[0, 1]` against bounds in `kpi_bounds.json`:

```
reward = 0.30 · P + 0.30 · D + 0.40 · C
```

---

## Reproducing the main results

```bash
# Calibrate KPI bounds at the target operating condition
python compress.py --calibrate --lambda-base 0.0833 --cal-samples 100

# Train PPO for 10k steps
python rl_env.py --timesteps 10000

# Plot training curves
python plot_training.py

# Run sensitivity analysis on the default layout
python sensitivity_analysis.py --eval-runs 10 --lambda-base 0.1

# Export Godot replay of the converged layout
python record_replay.py --agents 6 --ticks 3600 --out replay.json
```

---

## Generated files (excluded from git via `.gitignore`)

These are produced by the scripts and are not committed:

- `best_model.zip`, `warehouse_layout_ppo_v3.zip` — trained PPO weights
- `training_log.csv`, `evaluations.npz` — training history
- `kpi_bounds.json`, `sensitivity_baseline.json` — cached calibration
- `agent_traces.csv`, `speed_trace.csv`, `traffic_heatmap.csv`, `congestion_heatmap.csv` — per-run outputs
- `*.png`, `*.pdf` — figures (regenerate from the plot scripts)
- `runs/`, `Results_final/` — saved experiment bundles

---

## Key dependencies

- Python 3.12
- `jupedsim` (Collision-Free Speed Model)
- `stable-baselines3`, `gymnasium`
- `numpy`, `pandas`, `matplotlib`
- `pygame` (for the interactive `main.py` visualiser)

Full list in `requirements.txt`.

---
