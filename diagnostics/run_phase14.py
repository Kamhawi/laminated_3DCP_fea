"""Run all Phase 14 code-level check diagnostics.

Usage::

    python -m diagnostics.run_phase14 [--small] [--config PATH] [--n-steps N]
           [--g-factor G] [--skip-alpha-facet] [--skip-epsvp-ratio]
           [--skip-quadrature]

The ``--small`` flag uses a 4x4x2 mesh (32 cells, 768 DOFs) for fast
iteration.  Test 14.5 requires no Newton solves.  Tests 14.4 and 14.1
solve Newton at each time step with amplified gravity.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from mpi4py import MPI

from diagnostics.setup import build_diagnostic_state
from diagnostics.code_level_checks import (
    check_quadrature_degrees,
    check_epsvp_strain_ratio,
    compare_alpha_facet_modes,
)
from diagnostics.run_phase1 import _TeeWriter, _results_to_serializable


def main():
    parser = argparse.ArgumentParser(
        description="Phase 14: Selected code-level check diagnostics",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use small mesh (4x4x2 = 32 cells) for fast testing",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--n-steps", type=int, default=5,
        help="Number of time sample points (default: 5)",
    )
    parser.add_argument(
        "--g-factor", type=int, default=50,
        help="Gravity amplification for Tests 14.1/14.4 (default: 50)",
    )
    parser.add_argument(
        "--skip-alpha-facet", action="store_true",
        help="Skip Test 14.1 (alpha_facet mode comparison)",
    )
    parser.add_argument(
        "--skip-epsvp-ratio", action="store_true",
        help="Skip Test 14.4 (eps_vp/strain ratio)",
    )
    parser.add_argument(
        "--skip-quadrature", action="store_true",
        help="Skip Test 14.5 (quadrature degree inspection)",
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
        print("  PHASE 14: SELECTED CODE-LEVEL CHECKS")
        print("=" * 72)
        if args.small:
            print("  Mode: SMALL mesh (4x4x2)")
        print(f"  Time sample points: {args.n_steps}")
        skipped = []
        if args.skip_quadrature:
            skipped.append("14.5")
        if args.skip_epsvp_ratio:
            skipped.append("14.4")
        if args.skip_alpha_facet:
            skipped.append("14.1")
        if skipped:
            print(f"  Skipping: Tests {', '.join(skipped)}")
        else:
            print(f"  Gravity amplification (14.1/14.4): {args.g_factor}x")
        print(flush=True)

    # -------------------------------------------------------------------
    # Build diagnostic state
    # -------------------------------------------------------------------
    state = build_diagnostic_state(
        config_path=args.config, mesh_overrides=mesh_overrides, comm=comm,
    )

    results = {}

    # -------------------------------------------------------------------
    # Test 14.5: Quadrature degree inspection (quick, no solves)
    # -------------------------------------------------------------------
    if not args.skip_quadrature:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 14.5: Quadrature Degree Inspection")
            print("-" * 72, flush=True)

        results["quadrature_degrees"] = check_quadrature_degrees(
            state, verbose=True,
        )

    # -------------------------------------------------------------------
    # Test 14.4: eps_vp / strain ratio (time-stepping)
    # -------------------------------------------------------------------
    if not args.skip_epsvp_ratio:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 14.4: eps_vp / epsilon(u) Freezing Ratio")
            print("-" * 72, flush=True)

        results["epsvp_ratio"] = check_epsvp_strain_ratio(
            state, n_steps=args.n_steps, g_factor=args.g_factor, verbose=True,
        )

    # -------------------------------------------------------------------
    # Test 14.1: Alpha_facet mode comparison (3x form rebuilds)
    # -------------------------------------------------------------------
    if not args.skip_alpha_facet:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 14.1: Alpha_facet Mode Comparison")
            print("-" * 72, flush=True)

        results["alpha_facet"] = compare_alpha_facet_modes(
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

        # 14.5: Quadrature.
        if "quadrature_degrees" in results:
            qd = results["quadrature_degrees"]
            n_terms = len(qd["per_term"])
            n_with_info = sum(
                1 for t in qd["per_term"]
                if not t["jit_failed"] and t["degrees"]
            )
            print(f"  Test 14.5  Quadrature degrees:    "
                  f"{n_with_info}/{n_terms} terms have metadata")
        else:
            print(f"  Test 14.5  Quadrature degrees:    SKIPPED")

        # 14.4: eps_vp ratio.
        if "epsvp_ratio" in results:
            er = results["epsvp_ratio"]
            if er["any_ratio_flag"]:
                print(f"  Test 14.4  eps_vp/strain ratio:   WARN  "
                      f"(max={er['worst_max_ratio']:.4e})")
            else:
                print(f"  Test 14.4  eps_vp/strain ratio:   OK  "
                      f"(max={er['worst_max_ratio']:.4e})")
        else:
            print(f"  Test 14.4  eps_vp/strain ratio:   SKIPPED")

        # 14.1: Alpha_facet.
        if "alpha_facet" in results:
            af = results["alpha_facet"]
            summary = af["summary"]
            parts = []
            for mode in af["modes"]:
                s = summary[mode]
                parts.append(f"{mode}={s['total_iters']}iters")
            print(f"  Test 14.1  Alpha_facet modes:     {', '.join(parts)}")
            if any(s["n_failed"] > 0 for s in summary.values()):
                failed_modes = [
                    m for m in af["modes"] if summary[m]["n_failed"] > 0
                ]
                print(f"             Convergence failures:  {', '.join(failed_modes)}")
        else:
            print(f"  Test 14.1  Alpha_facet modes:     SKIPPED")

        # Findings.
        concerns = []
        if "epsvp_ratio" in results and results["epsvp_ratio"]["any_ratio_flag"]:
            concerns.append(
                f"eps_vp/strain ratio exceeds {0.5} — "
                f"VP freezing error may be significant"
            )
        if "alpha_facet" in results:
            af = results["alpha_facet"]
            iters = {m: af["summary"][m]["total_iters"] for m in af["modes"]}
            best = min(iters, key=iters.get)
            worst = max(iters, key=iters.get)
            if iters[worst] > 1.5 * iters[best]:
                concerns.append(
                    f"alpha_facet mode '{best}' uses {iters[best]} total iters "
                    f"vs '{worst}' at {iters[worst]} — "
                    f"consider switching from current 'min' if '{best}' is better"
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

        log_path = output_dir / f"phase14_{mesh_tag}_{timestamp}.log"
        log_path.write_text(tee.get_log() if tee else "")

        json_path = output_dir / f"phase14_{mesh_tag}_{timestamp}.json"
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
