"""Run all Phase 6 residual decomposition diagnostics.

Usage::

    python -m diagnostics.run_phase6 [--small] [--config PATH] [--n-steps N]
           [--g-factor G]

The ``--small`` flag uses a 4x4x2 mesh (32 cells, 768 DOFs) for fast
iteration.  Test 6.2 requires no Newton solves.  Test 6.1 solves Newton
at each time step with amplified gravity to get non-trivial displacement.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from mpi4py import MPI

from diagnostics.setup import build_diagnostic_state
from diagnostics.residual_decomposition import (
    verify_gravity_loading,
    decompose_residual_over_time,
)
from diagnostics.run_phase1 import _TeeWriter, _results_to_serializable


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Residual decomposition diagnostics",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use small mesh (4x4x2 = 32 cells) for fast testing",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--n-steps", type=int, default=10,
        help="Number of time sample points (default: 10)",
    )
    parser.add_argument(
        "--g-factor", type=int, default=50,
        help="Gravity amplification for Test 6.1 (default: 50)",
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
    # Banner
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("=" * 72)
        print("  PHASE 6: RESIDUAL DECOMPOSITION")
        print("=" * 72)
        if args.small:
            print("  Mode: SMALL mesh (4x4x2)")
        print(f"  Time sample points: {args.n_steps}")
        print(f"  Gravity amplification (6.1): {args.g_factor}x")
        print(flush=True)

    # -------------------------------------------------------------------
    # Build diagnostic state
    # -------------------------------------------------------------------
    state = build_diagnostic_state(
        config_path=args.config, mesh_overrides=mesh_overrides, comm=comm,
    )

    results = {}

    # -------------------------------------------------------------------
    # Test 6.2: Gravity loading consistency (quick, no Newton)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 6.2: Gravity Loading Consistency")
        print("-" * 72, flush=True)

    results["gravity_verification"] = verify_gravity_loading(
        state, verbose=True,
    )

    # -------------------------------------------------------------------
    # Test 6.1: Residual decomposition over time
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 6.1: Residual Decomposition Over Time")
        print("-" * 72, flush=True)

    results["residual_decomposition"] = decompose_residual_over_time(
        state, n_steps=args.n_steps, g_factor=args.g_factor, verbose=True,
    )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - t0

    if comm.rank == 0:
        print("\n" + "=" * 72)
        print("  SUMMARY")
        print("=" * 72)

        # 6.2: Gravity.
        gv = results["gravity_verification"]
        gv_status = "PASS" if gv["passed"] else "FAIL"
        print(f"  Test 6.2  Gravity consistency:    {gv_status}  "
              f"(unit_err={gv['unit_conversion_relerr']:.1e}, "
              f"norm/force={gv['ratio_grav_norm_to_force']:.3f})")

        # 6.1: Decomposition.
        rd = results["residual_decomposition"]
        dom = rd["dominant_term_counts"]
        if dom:
            top_term = max(dom.items(), key=lambda x: x[1])
            print(f"  Test 6.1  Residual decomposition: "
                  f"dominant={top_term[0]} ({top_term[1]}/{len(rd['records'])} steps)")
        else:
            print(f"  Test 6.1  Residual decomposition: no steps")

        # Actionable findings.
        concerns = []
        if not gv["passed"]:
            concerns.append("Gravity loading verification FAILED — check unit conversion")
        if dom:
            top = max(dom.items(), key=lambda x: x[1])
            if top[0] == "gravity" and top[1] == len(rd["records"]):
                concerns.append(
                    "Gravity dominates at every step — consider load ramping for first step"
                )
            if top[0] == "bulk_EVP" and top[1] > len(rd["records"]) // 2:
                concerns.append(
                    "Bulk EVP (including eps_vp stagger) dominates — "
                    "stagger error is the main residual source"
                )

        if concerns:
            print(f"\n  Findings:")
            for c in concerns:
                print(f"    - {c}")

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

        log_path = output_dir / f"phase6_{mesh_tag}_{timestamp}.log"
        log_path.write_text(tee.get_log() if tee else "")

        json_path = output_dir / f"phase6_{mesh_tag}_{timestamp}.json"
        json_data = {
            "timestamp": timestamp,
            "mesh_tag": mesh_tag,
            "elapsed_s": elapsed,
            "n_steps": args.n_steps,
            "g_factor": args.g_factor,
            "results": _results_to_serializable(results),
        }
        json_path.write_text(json.dumps(json_data, indent=2))

        print(f"\n  Output saved to:")
        print(f"    Log:  {log_path}")
        print(f"    JSON: {json_path}")


if __name__ == "__main__":
    main()
