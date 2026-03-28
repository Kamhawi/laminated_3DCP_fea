"""Run all Phase 4 cohesive law convexity and stability diagnostics.

Usage::

    python -m diagnostics.run_phase4 [--small] [--config PATH] [--skip-irreversibility] [--g-factor G]

The ``--small`` flag uses a 4x4x2 mesh (32 cells, 768 DOFs) for fast
iteration.  ``--skip-irreversibility`` skips the time-stepping damage
tracking tests (Tests 4.2, 4.3 — most expensive).  ``--g-factor`` sets
the gravity amplification for the damage tracking tests (default 50).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI

from diagnostics.setup import build_diagnostic_state
from diagnostics.stiffness_hierarchy import compute_cell_diameters
from diagnostics.cohesive_analysis import (
    run_tangent_stiffness_analysis,
    run_mixed_mode_convexity_check,
    _run_time_stepping_with_damage_tracking,
    run_damage_irreversibility_check,
    run_contact_penalty_check,
)
from diagnostics.run_phase1 import _TeeWriter, _results_to_serializable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Cohesive law convexity and stability diagnostics",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use small mesh (4x4x2 = 32 cells) for fast testing",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--skip-irreversibility", action="store_true",
        help="Skip damage tracking tests (Tests 4.2, 4.3 — most expensive)",
    )
    parser.add_argument(
        "--g-factor", type=float, default=50.0,
        help="Gravity amplification factor for damage tracking (default: 50)",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    mesh_overrides = None
    if args.small:
        mesh_overrides = {"n_length": 4, "n_span": 4, "n_thickness": 2}

    tee = None
    if comm.rank == 0:
        tee = _TeeWriter(sys.stdout)
        sys.stdout = tee

    t0 = time.time()

    # -------------------------------------------------------------------
    # Build diagnostic state
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("=" * 72)
        print("  PHASE 4: COHESIVE LAW CONVEXITY AND STABILITY")
        print("=" * 72)
        if args.small:
            print("  Mode: SMALL mesh (4x4x2)")
        if args.skip_irreversibility:
            print("  Skipping: Tests 4.2+4.3 (damage tracking)")
        else:
            print(f"  Gravity amplification: {args.g_factor}x")
        print(flush=True)

    state = build_diagnostic_state(
        config_path=args.config, mesh_overrides=mesh_overrides, comm=comm,
    )

    cfg = state.cfg
    results = {}

    # Representative ages and element sizes for analytical tests.
    bt = state.birth_times_dolfinx
    t_min = float(bt.min())
    t_max = float(bt.max())
    ages = [5, 30, 60, 120, 300]  # seconds since deposition

    h_arr = compute_cell_diameters(state.msh)
    h_mean = float(np.mean(h_arr)) if h_arr.size > 0 else 100.0
    h_values = [h_mean]

    # -------------------------------------------------------------------
    # Test 4.1: Tangent stiffness (analytical, fast)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 4.1: Mode-I Tangent Stiffness and Snap-Back Ratio")
        print("-" * 72, flush=True)

    results["tangent_stiffness"] = run_tangent_stiffness_analysis(
        cfg, ages, h_values, t_open=0.0, verbose=(comm.rank == 0),
    )

    # -------------------------------------------------------------------
    # Test 4.4: Mixed-mode convexity (analytical, fast)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 4.4: Mixed-Mode Hessian Convexity")
        print("-" * 72, flush=True)

    results["mixed_mode_convexity"] = run_mixed_mode_convexity_check(
        cfg, ages, h_values, t_open=0.0, verbose=(comm.rank == 0),
    )

    # -------------------------------------------------------------------
    # Tests 4.2 + 4.3: Damage tracking (Newton solves, expensive)
    # -------------------------------------------------------------------
    if not args.skip_irreversibility:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Tests 4.2+4.3: Damage Tracking Under Amplified Gravity")
            print("-" * 72)
            print(f"    g_factor = {args.g_factor}")
            print(f"    Running time-stepping loop...", flush=True)

        tracking = _run_time_stepping_with_damage_tracking(
            state, n_steps=20, g_factor=args.g_factor, max_iter=50,
            verbose=(comm.rank == 0),
        )

        # Test 4.2: Damage irreversibility.
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 4.2: Damage Irreversibility Analysis")
            print("-" * 72, flush=True)

        results["damage_irreversibility"] = run_damage_irreversibility_check(
            tracking, verbose=(comm.rank == 0),
        )

        # Test 4.3: Contact penalty.
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 4.3: Contact Penalty Adequacy")
            print("-" * 72, flush=True)

        results["contact_penalty"] = run_contact_penalty_check(
            tracking, verbose=(comm.rank == 0),
        )

        # Store tracking summary (without large arrays) for JSON.
        tracking_summary = {
            "n_steps": tracking["n_steps"],
            "g_factor": tracking["g_factor"],
            "h_elem": tracking["h_elem"],
            "steps": [
                {k: v for k, v in s.items()
                 if k not in ("damage", "jump_n", "jump_t_mag", "delta_n_open")}
                for s in tracking.get("steps", [])
            ],
        }
        results["tracking_summary"] = tracking_summary

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - t0

    if comm.rank == 0:
        print("\n" + "=" * 72)
        print("  SUMMARY")
        print("=" * 72)

        # 4.1: Tangent stiffness.
        ts = results.get("tangent_stiffness", {})
        sb = ts.get("max_snap_back_ratio", 0)
        snap = ts.get("snap_back_detected", False)
        status = "WARN (snap-back)" if snap else "OK (no snap-back)"
        print(f"  Test 4.1  Tangent stiffness:      {status}  "
              f"(max ratio={sb:.4f})")

        # 4.4: Mixed-mode convexity.
        mc = results.get("mixed_mode_convexity", {})
        mc_rows = mc.get("rows", [])
        if mc_rows:
            max_nc = max(r["non_convex_fraction"] for r in mc_rows)
            min_eig = min(r["min_eigenvalue_global"] for r in mc_rows)
            print(f"  Test 4.4  Mixed-mode convexity:   "
                  f"non-convex fraction up to {max_nc:.1%}  "
                  f"(min eig={min_eig:.4e})")

        # 4.2: Damage irreversibility.
        if not args.skip_irreversibility:
            di = results.get("damage_irreversibility", {})
            healing = di.get("any_healing", False)
            status = "HEALING DETECTED" if healing else "NO HEALING"
            worst = di.get("worst_decrease", 0)
            n_events = di.get("n_healing_events", 0)
            print(f"  Test 4.2  Damage irreversibility: {status}  "
                  f"(events={n_events}, worst={worst:.6f})")

            # 4.3: Contact penalty.
            cp = results.get("contact_penalty", {})
            adequate = cp.get("all_adequate", True)
            worst_r = cp.get("worst_ratio", 0)
            status = "ADEQUATE" if adequate else "INADEQUATE"
            print(f"  Test 4.3  Contact penalty:        {status}  "
                  f"(worst ratio={worst_r:.4e})")

        print(f"\n  Total time: {elapsed:.1f} s")
        print("=" * 72)

    # -------------------------------------------------------------------
    # Save outputs (rank 0 only)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        if tee is not None:
            sys.stdout = tee._original

        output_dir = Path("diagnostics") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mesh_tag = "small" if args.small else "production"

        log_path = output_dir / f"phase4_{mesh_tag}_{timestamp}.log"
        log_path.write_text(tee.get_log() if tee else "")

        json_path = output_dir / f"phase4_{mesh_tag}_{timestamp}.json"
        json_data = {
            "timestamp": timestamp,
            "mesh_tag": mesh_tag,
            "elapsed_s": elapsed,
            "g_factor": args.g_factor,
            "results": _results_to_serializable(results),
        }
        json_path.write_text(json.dumps(json_data, indent=2))

        print(f"\n  Output saved to:")
        print(f"    Log:  {log_path}")
        print(f"    JSON: {json_path}")


if __name__ == "__main__":
    main()
