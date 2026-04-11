# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Post-processing and figure generation for the cylinder print validation.

Parses step_metrics.csv from the most recent run and generates comparison
plots against the experimental results.

Run:
    python -m validation.cylinder_print.figure
    python -m validation.cylinder_print.figure path/to/step_metrics.csv
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False

COLOR_SIM = '#4A6274'
COLOR_SIM_COLLAPSE = '#003366'
COLOR_EXP = '#a52a2a'

# Wolfs, Bos & Salet (2018) Cement Concr. Res. 106, 103-116
# Individual specimen failures: 30, 25, 31, 27, 31
EXPERIMENTAL_SPECIMENS = np.array([30, 25, 31, 27, 31])
EXPERIMENTAL_COLLAPSE_MEAN = float(np.mean(EXPERIMENTAL_SPECIMENS))   # 28.8
EXPERIMENTAL_COLLAPSE_STD = float(np.std(EXPERIMENTAL_SPECIMENS))     # 2.7
EXPERIMENTAL_COLLAPSE_MIN = int(np.min(EXPERIMENTAL_SPECIMENS))       # 25
EXPERIMENTAL_COLLAPSE_MAX = int(np.max(EXPERIMENTAL_SPECIMENS))       # 31
COLOR_EXP_BAND = '#a52a2a'
T_INTERVAL_S = 2.0 * math.pi * 250.0 / (5000.0 / 60.0)  # ≈ 18.85 s


def load_metrics(csv_path):
    """Load step_metrics.csv into a list of dicts with numeric conversion."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        parsed[k] = v
            rows.append(parsed)
    return rows


def compute_active_layers(time_s):
    """Compute how many layers are active at a given simulation time."""
    return int(time_s / T_INTERVAL_S) + 1


def find_failure_step(rows):
    """Find the first step where Newton solver did not converge."""
    for i, row in enumerate(rows):
        converged = row.get("converged", 1)
        if isinstance(converged, str):
            converged = converged.strip().lower() not in ("0", "false", "")
        if not converged:
            return i
    return None


def find_collapse_layer(rows):
    """Detect collapse layer from displacement acceleration or divergence.

    Returns the layer number at which collapse is predicted, or None.
    Collapse is detected as Newton divergence, or a sudden displacement
    jump exceeding 3x the running median increment.
    """
    fail_idx = find_failure_step(rows)
    if fail_idx is not None:
        return compute_active_layers(rows[fail_idx]["time_s"])

    times = np.array([r["time_s"] for r in rows])
    max_disp = np.array([r.get("max_disp_mm", 0.0) for r in rows])
    layers = np.array([compute_active_layers(t) for t in times])

    unique_layers = np.unique(layers)
    disp_per_layer = np.array([
        np.mean(max_disp[layers == l]) for l in unique_layers
    ])

    if len(disp_per_layer) < 5:
        return None

    increments = np.diff(disp_per_layer)
    for i in range(3, len(increments)):
        median_prev = np.median(increments[max(0, i - 3):i])
        if median_prev > 0 and increments[i] > 3.0 * median_prev:
            return int(unique_layers[i + 1])
    return None


def find_latest_run(base_dir="validation/cylinder_print/output"):
    """Find the most recent run directory."""
    base = Path(base_dir)
    if not base.exists():
        return None
    run_dirs = sorted(base.glob("run_*"), key=lambda p: p.name)
    if not run_dirs:
        return None
    csv_path = run_dirs[-1] / "step_metrics.csv"
    return csv_path if csv_path.exists() else None


def main():
    # Determine CSV path
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = find_latest_run()
        if csv_path is None:
            print("No step_metrics.csv found. Run the simulation first.")
            sys.exit(1)

    print(f"Loading: {csv_path}")
    rows = load_metrics(csv_path)
    if not rows:
        print("Empty metrics file.")
        sys.exit(1)

    # Extract arrays
    times = np.array([r["time_s"] for r in rows])
    active_layers = np.array([compute_active_layers(t) for t in times])
    max_disp = np.array([r.get("max_disp_mm", 0.0) for r in rows])
    yielding_cells = np.array([r.get("yielding_cells", 0) for r in rows])
    active_cells = np.array([r.get("active_cells", 1) for r in rows])
    max_plastic_strain = np.array([r.get("max_plastic_strain", 0.0) for r in rows])

    yielding_fraction = np.where(
        active_cells > 0,
        yielding_cells / active_cells * 100.0,
        0.0,
    )

    # Detect collapse
    collapse_layer = find_collapse_layer(rows)
    if collapse_layer is not None:
        print(f"\nPredicted collapse: layer {collapse_layer}")
    else:
        print("\nNo collapse detected.")
    print(f"  Max displacement at end: {max_disp[-1]:.3f} mm")
    print(f"\nWolfs et al. (2018) experiment: mean={EXPERIMENTAL_COLLAPSE_MEAN:.1f}, "
          f"std={EXPERIMENTAL_COLLAPSE_STD:.1f}, range={EXPERIMENTAL_COLLAPSE_MIN}-{EXPERIMENTAL_COLLAPSE_MAX} layers")

    if not _HAS_PLOT:
        print("\nMatplotlib/seaborn not available; skipping figure generation.")
        return

    # Smooth the data by averaging per-layer (multiple steps per layer)
    unique_layers = np.unique(active_layers)
    disp_per_layer = np.array([
        np.mean(max_disp[active_layers == l]) for l in unique_layers
    ])
    yield_per_layer = np.array([
        np.mean(yielding_fraction[active_layers == l]) for l in unique_layers
    ])
    plastic_per_layer = np.array([
        np.max(max_plastic_strain[active_layers == l]) for l in unique_layers
    ])

    # Common legend entries for all panels
    exp_label = f"Experiment mean ({EXPERIMENTAL_COLLAPSE_MEAN:.0f} layers)"
    exp_band_label = f"Experiment range ({EXPERIMENTAL_COLLAPSE_MIN}\u2013{EXPERIMENTAL_COLLAPSE_MAX})"
    sim_collapse_label = (
        f"Predicted collapse ({collapse_layer} layers)"
        if collapse_layer is not None else None
    )

    def _add_reference_lines(ax):
        ax.axvspan(EXPERIMENTAL_COLLAPSE_MIN, EXPERIMENTAL_COLLAPSE_MAX,
                   color=COLOR_EXP_BAND, alpha=0.12, label=exp_band_label, zorder=1)
        ax.axvline(EXPERIMENTAL_COLLAPSE_MEAN, color=COLOR_EXP, linestyle="--",
                   lw=1.8, label=exp_label, zorder=3)
        if collapse_layer is not None:
            ax.axvline(collapse_layer, color=COLOR_SIM_COLLAPSE, linestyle="--",
                       lw=1.8, label=sim_collapse_label, zorder=3)

    # ── Figure: side-by-side ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # ── a  Max displacement vs active layers ──────────────────────────
    ax = axes[0]
    ax.plot(unique_layers, disp_per_layer, '-', color=COLOR_SIM, lw=2.5,
            ms=0, alpha=0.88, label="Present model", zorder=4)
    _add_reference_lines(ax)
    ax.set_xlabel("Active layers", fontsize=13)
    ax.set_ylabel("Max displacement [mm]", fontsize=13)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_xlim(0, unique_layers[-1])
    ax.set_title("a", fontweight="bold", fontsize=13, loc="left", pad=8)
    ax.grid(True, which="major", alpha=0.3)

    # ── b  Yielding fraction ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(unique_layers, yield_per_layer, '-', color=COLOR_SIM, lw=2.5,
            ms=0, alpha=0.88, label=r"Yielded ($f > 0$)", zorder=4)
    _add_reference_lines(ax)
    ax.set_xlabel("Active layers", fontsize=13)
    ax.set_ylabel("Yielding cells [%]", fontsize=13)
    ax.set_xlim(0, unique_layers[-1])
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_title("b", fontweight="bold", fontsize=13, loc="left", pad=8)
    ax.grid(True, which="major", alpha=0.3)

    # ── c  Max plastic strain ─────────────────────────────────────────
    ax = axes[2]
    ax.plot(unique_layers, plastic_per_layer * 100.0, '-', color=COLOR_SIM,
            lw=2.5, ms=0, alpha=0.88, label="Max plastic strain", zorder=4)
    _add_reference_lines(ax)
    ax.set_xlabel("Active layers", fontsize=13)
    ax.set_ylabel("Max plastic strain [%]", fontsize=13)
    ax.set_xlim(0, unique_layers[-1])
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_title("c", fontweight="bold", fontsize=13, loc="left", pad=8)
    ax.grid(True, which="major", alpha=0.3)

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.14, top=0.90,
                        wspace=0.30)

    fig_path = csv_path.parent / "cylinder_print_validation.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"\nFigures saved: {fig_path} and {fig_path.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
