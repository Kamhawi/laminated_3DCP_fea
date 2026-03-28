"""Run all Phase 3 activation mechanics diagnostics.

Usage::

    python -m diagnostics.run_phase3 [--small] [--config PATH] [--skip-sweeps] [--skip-stepping]

The ``--small`` flag uses a 4x4x2 mesh (32 cells, 768 DOFs) for fast
iteration.  ``--skip-sweeps`` skips sharpness and alpha_min Newton-solve
sweeps (Tests 3.1b, 3.1c).  ``--skip-stepping`` skips the step-by-step
activation load test (Test 3.4, most expensive).
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
from diagnostics.activation_analysis import (
    run_alpha_contrast_analysis,
    run_sharpness_sweep,
    run_alpha_min_sweep,
    run_newly_activated_residual_check,
    run_pinning_completeness_check,
    run_step_by_step_activation_load,
)
from diagnostics.run_phase1 import _TeeWriter, _results_to_serializable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Activation mechanics diagnostics",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use small mesh (4x4x2 = 32 cells) for fast testing",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--skip-sweeps", action="store_true",
        help="Skip sharpness and alpha_min sweeps (Tests 3.1b, 3.1c)",
    )
    parser.add_argument(
        "--skip-stepping", action="store_true",
        help="Skip step-by-step activation load (Test 3.4, most expensive)",
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
        print("  PHASE 3: ACTIVATION MECHANICS DIAGNOSTICS")
        print("=" * 72)
        if args.small:
            print("  Mode: SMALL mesh (4x4x2)")
        print(flush=True)

    state = build_diagnostic_state(
        config_path=args.config, mesh_overrides=mesh_overrides, comm=comm,
    )

    results = {}

    # -------------------------------------------------------------------
    # Test 3.1a: Alpha contrast ratio (pure NumPy, fast)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 3.1a: Alpha Contrast Ratio")
        print("-" * 72, flush=True)

    results["alpha_contrast"] = run_alpha_contrast_analysis(state)

    # -------------------------------------------------------------------
    # Test 3.3: Pinning completeness (pure NumPy, fast)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 3.3: Inactive DOF Pinning Completeness")
        print("-" * 72, flush=True)

    results["pinning_completeness"] = run_pinning_completeness_check(state)

    # -------------------------------------------------------------------
    # Test 3.1b: Sharpness sweep (Newton solves, form rebuilds)
    # -------------------------------------------------------------------
    if not args.skip_sweeps:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 3.1b: Sharpness Sweep")
            print("-" * 72, flush=True)

        results["sharpness_sweep"] = run_sharpness_sweep(state)

    # -------------------------------------------------------------------
    # Test 3.1c: Alpha_min sweep (Newton solves, form rebuilds)
    # -------------------------------------------------------------------
    if not args.skip_sweeps:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 3.1c: Alpha_min Sweep")
            print("-" * 72, flush=True)

        results["alpha_min_sweep"] = run_alpha_min_sweep(state)

    # -------------------------------------------------------------------
    # Test 3.2: Newly-activated residual check (assembly only)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 3.2: Newly-Activated Cell Residual Contribution")
        print("-" * 72, flush=True)

    results["newly_activated_residual"] = run_newly_activated_residual_check(state)

    # -------------------------------------------------------------------
    # Test 3.4: Step-by-step activation load (full Newton, expensive)
    # -------------------------------------------------------------------
    if not args.skip_stepping:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 3.4: Step-by-Step Activation Load")
            print("-" * 72, flush=True)

        results["activation_load"] = run_step_by_step_activation_load(state)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - t0

    if comm.rank == 0:
        print("\n" + "=" * 72)
        print("  SUMMARY")
        print("=" * 72)

        # 3.1a: contrast ratio.
        ac = results.get("alpha_contrast", {})
        ac_records = ac.get("records", [])
        if ac_records:
            max_ratio = max(r["contrast_ratio"] for r in ac_records)
            n_warn = sum(1 for r in ac_records if r["exceeds_1e6"])
            status = "WARN" if n_warn > 0 else "OK"
            print(f"  Test 3.1a  Alpha contrast:       {status}  "
                  f"(max ratio={max_ratio:.2e}, "
                  f"{n_warn}/{len(ac_records)} time points > 1e6)")

        # 3.3: pinning completeness.
        pc = results.get("pinning_completeness", {})
        pc_records = pc.get("records", [])
        if pc_records:
            max_unpinned = max(r["n_unpinned_soft"] for r in pc_records)
            max_frac = max(r["fraction"] for r in pc_records)
            status = "WARN" if max_unpinned > 0 else "OK"
            print(f"  Test 3.3   Pinning completeness:  {status}  "
                  f"(max unpinned soft={max_unpinned}, "
                  f"max fraction={max_frac:.4f})")

        # 3.1b: sharpness sweep.
        if not args.skip_sweeps:
            ss = results.get("sharpness_sweep", {})
            sweep = ss.get("sweep", [])
            if sweep:
                n_conv = sum(1 for e in sweep if e["converged"])
                default = ss.get("default_value")
                default_entry = next(
                    (e for e in sweep if e["value"] == default), None,
                )
                default_iters = default_entry["n_iter"] if default_entry else "?"
                iter_range = (f"{min(e['n_iter'] for e in sweep)}-"
                              f"{max(e['n_iter'] for e in sweep)}")
                status = "PASS" if n_conv == len(sweep) else f"{n_conv}/{len(sweep)} conv"
                print(f"  Test 3.1b  Sharpness sweep:      {status}  "
                      f"(default={default} -> {default_iters} iters, "
                      f"range={iter_range})")

        # 3.1c: alpha_min sweep.
        if not args.skip_sweeps:
            ams = results.get("alpha_min_sweep", {})
            sweep = ams.get("sweep", [])
            if sweep:
                n_conv = sum(1 for e in sweep if e["converged"])
                default = ams.get("default_value")
                default_entry = next(
                    (e for e in sweep if e["value"] == default), None,
                )
                default_iters = default_entry["n_iter"] if default_entry else "?"
                iter_range = (f"{min(e['n_iter'] for e in sweep)}-"
                              f"{max(e['n_iter'] for e in sweep)}")
                status = "PASS" if n_conv == len(sweep) else f"{n_conv}/{len(sweep)} conv"
                print(f"  Test 3.1c  Alpha_min sweep:      {status}  "
                      f"(default={default} -> {default_iters} iters, "
                      f"range={iter_range})")

        # 3.2: newly-activated residual.
        nar = results.get("newly_activated_residual", {})
        nar_steps = nar.get("steps", [])
        if nar_steps:
            max_frac = max(r["fraction"] for r in nar_steps)
            mean_frac = float(np.mean([r["fraction"] for r in nar_steps]))
            print(f"  Test 3.2   New-cell residual:     "
                  f"max fraction={max_frac:.4f}, mean={mean_frac:.4f}")

        # 3.4: activation load.
        if not args.skip_stepping:
            al = results.get("activation_load", {})
            al_steps = al.get("steps", [])
            corr = al.get("correlations", {})
            if al_steps:
                n_conv = sum(1 for s in al_steps if s["converged"])
                all_conv = n_conv == len(al_steps)
                status = "PASS" if all_conv else f"{n_conv}/{len(al_steps)} conv"
                iter_range = (f"{min(s['n_iter'] for s in al_steps)}-"
                              f"{max(s['n_iter'] for s in al_steps)}")
                print(f"  Test 3.4   Activation load:       {status}  "
                      f"(iters={iter_range})")
                if corr:
                    c1 = corr.get("n_new_vs_n_iter")
                    c2 = corr.get("n_new_vs_initial_residual")
                    c1_str = f"{c1:+.3f}" if c1 is not None else "N/A"
                    c2_str = f"{c2:+.3f}" if c2 is not None else "N/A"
                    print(f"             corr(n_new, n_iter)={c1_str}  "
                          f"corr(n_new, ||F_init||)={c2_str}")

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

        log_path = output_dir / f"phase3_{mesh_tag}_{timestamp}.log"
        log_path.write_text(tee.get_log() if tee else "")

        json_path = output_dir / f"phase3_{mesh_tag}_{timestamp}.json"
        json_data = {
            "timestamp": timestamp,
            "mesh_tag": mesh_tag,
            "elapsed_s": elapsed,
            "results": _results_to_serializable(results),
        }
        json_path.write_text(json.dumps(json_data, indent=2))

        print(f"\n  Output saved to:")
        print(f"    Log:  {log_path}")
        print(f"    JSON: {json_path}")


if __name__ == "__main__":
    main()
