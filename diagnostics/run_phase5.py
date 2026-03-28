"""Run all Phase 5 material property conditioning diagnostics.

Usage::

    python -m diagnostics.run_phase5 [--small] [--config PATH] [--n-steps N]
           [--skip-vp-accuracy] [--skip-stagger] [--g-factor G]

The ``--small`` flag uses a 4x4x2 mesh (32 cells, 768 DOFs) for fast
iteration.  Tests 5.0–5.3 require no Newton solves.  Tests 5.4–5.5
solve Newton at each time point with amplified gravity.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from mpi4py import MPI

from diagnostics.setup import build_diagnostic_state
from diagnostics.property_conditioning import (
    compute_analytical_property_evolution,
    check_property_contrasts,
    check_poisson_locking,
    check_viscosity_stiffness_ratio,
    check_vp_substep_accuracy,
    check_stagger_error,
)
from diagnostics.run_phase1 import _TeeWriter, _results_to_serializable


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Material property conditioning diagnostics",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use small mesh (4x4x2 = 32 cells) for fast testing",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--n-steps", type=int, default=20,
        help="Number of time sample points (default: 20)",
    )
    parser.add_argument(
        "--skip-vp-accuracy", action="store_true",
        help="Skip Test 5.4 (VP sub-step accuracy)",
    )
    parser.add_argument(
        "--skip-stagger", action="store_true",
        help="Skip Test 5.5 (stagger error)",
    )
    parser.add_argument(
        "--g-factor", type=int, default=50,
        help="Gravity amplification for Tests 5.4/5.5 (default: 50)",
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
        print("  PHASE 5: MATERIAL PROPERTY CONDITIONING")
        print("=" * 72)
        if args.small:
            print("  Mode: SMALL mesh (4x4x2)")
        print(f"  Time sample points: {args.n_steps}")
        skipped = []
        if args.skip_vp_accuracy:
            skipped.append("5.4")
        if args.skip_stagger:
            skipped.append("5.5")
        if skipped:
            print(f"  Skipping: Tests {', '.join(skipped)}")
        else:
            print(f"  Gravity amplification (5.4/5.5): {args.g_factor}x")
        print(flush=True)

    # -------------------------------------------------------------------
    # Build diagnostic state
    # -------------------------------------------------------------------
    state = build_diagnostic_state(
        config_path=args.config, mesh_overrides=mesh_overrides, comm=comm,
    )

    results = {}

    # -------------------------------------------------------------------
    # Test 5.0: Analytical property evolution (no mesh)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 5.0: Analytical Property Evolution")
        print("-" * 72, flush=True)

    results["analytical_evolution"] = compute_analytical_property_evolution(
        state.cfg, verbose=(comm.rank == 0),
    )

    # -------------------------------------------------------------------
    # Test 5.1: Property contrast monitoring
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 5.1: Property Contrast Monitoring")
        print("-" * 72, flush=True)

    results["property_contrasts"] = check_property_contrasts(
        state, n_steps=args.n_steps, verbose=True,
    )

    # -------------------------------------------------------------------
    # Test 5.2: Poisson ratio locking
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 5.2: Poisson Ratio Locking")
        print("-" * 72, flush=True)

    results["poisson_locking"] = check_poisson_locking(
        state, n_steps=args.n_steps, verbose=True,
    )

    # -------------------------------------------------------------------
    # Test 5.3: Viscosity-stiffness ratio
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 5.3: Viscosity-Stiffness Ratio")
        print("-" * 72, flush=True)

    results["viscosity_stiffness"] = check_viscosity_stiffness_ratio(
        state, n_steps=args.n_steps, verbose=True,
    )

    # -------------------------------------------------------------------
    # Test 5.4: VP sub-step accuracy
    # -------------------------------------------------------------------
    if not args.skip_vp_accuracy:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 5.4: VP Sub-Step Accuracy")
            print("-" * 72, flush=True)

        results["vp_accuracy"] = check_vp_substep_accuracy(
            state, n_steps=args.n_steps, g_factor=args.g_factor, verbose=True,
        )

    # -------------------------------------------------------------------
    # Test 5.5: Stagger error
    # -------------------------------------------------------------------
    if not args.skip_stagger:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 5.5: Stagger Error")
            print("-" * 72, flush=True)

        results["stagger_error"] = check_stagger_error(
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

        # 5.0: Analytical.
        print("  Test 5.0  Analytical evolution:   (see table above)")

        # 5.1: Property contrasts.
        pc = results["property_contrasts"]
        flags_5_1 = []
        if pc["any_E_ratio_flag"]:
            flags_5_1.append(f"E_ratio={pc['worst_E_ratio']:.1e}")
        if pc["any_eta_ratio_flag"]:
            flags_5_1.append(f"eta_ratio={pc['worst_eta_ratio']:.1e}")
        if pc["any_nu_range_flag"]:
            flags_5_1.append(f"nu_range={pc['worst_nu_range']:.2f}")
        if pc["any_nu_max_flag"]:
            flags_5_1.append(f"nu_max={pc['worst_nu_max']:.4f}")
        if pc["any_E_min_flag"]:
            flags_5_1.append(f"E_min={pc['worst_E_min']:.1e}")
        if flags_5_1:
            print(f"  Test 5.1  Property contrasts:     WARN  ({', '.join(flags_5_1)})")
        else:
            print(f"  Test 5.1  Property contrasts:     OK")

        # 5.2: Poisson locking.
        pl = results["poisson_locking"]
        if pl["any_locking_flag"]:
            max_frac = max(r["frac_above"] for r in pl["records"])
            print(f"  Test 5.2  Poisson locking:        WARN  "
                  f"(max ratio={pl['worst_locking_ratio']:.1f}, "
                  f"{max_frac:.0%} cells above threshold)")
        else:
            print(f"  Test 5.2  Poisson locking:        OK  "
                  f"(max ratio={pl['worst_locking_ratio']:.1f})")

        # 5.3: VP predictor.
        vs = results["viscosity_stiffness"]
        if vs["any_vp_ratio_flag"]:
            print(f"  Test 5.3  VP predictor quality:   WARN  "
                  f"(max dt/min(eta/E)={vs['worst_vp_ratio']:.1f})")
        else:
            print(f"  Test 5.3  VP predictor quality:   OK  "
                  f"(max dt/min(eta/E)={vs['worst_vp_ratio']:.1f})")

        # 5.4: VP accuracy.
        if "vp_accuracy" in results:
            va = results["vp_accuracy"]
            if va["any_accuracy_flag"]:
                print(f"  Test 5.4  VP sub-step accuracy:  WARN  "
                      f"(max evp_err={va['worst_max_evp_err']:.2e}, "
                      f"sig_err={va['worst_max_sig_err']:.2e})")
            else:
                print(f"  Test 5.4  VP sub-step accuracy:  OK  "
                      f"(max evp_err={va['worst_max_evp_err']:.2e})")
        else:
            print(f"  Test 5.4  VP sub-step accuracy:  SKIPPED")

        # 5.5: Stagger error.
        if "stagger_error" in results:
            se = results["stagger_error"]
            if se["any_stagger_flag"]:
                print(f"  Test 5.5  Stagger error:         WARN  "
                      f"(max ratio={se['worst_stagger_ratio']:.2f})")
            else:
                print(f"  Test 5.5  Stagger error:         OK  "
                      f"(max ratio={se['worst_stagger_ratio']:.2f})")
        else:
            print(f"  Test 5.5  Stagger error:         SKIPPED")

        # Conditioning concerns.
        from diagnostics.property_conditioning import E_RATIO_THRESHOLD

        concerns = []
        if pc["any_E_ratio_flag"]:
            first_flag = next(
                r["t_val"] for r in pc["records"] if r["E_ratio_flag"]
            )
            concerns.append(
                f"E_max/E_min crosses {E_RATIO_THRESHOLD:.0e} at t={first_flag:.1f} s"
            )
        if pc["any_nu_max_flag"]:
            concerns.append(
                f"nu_max = {pc['worst_nu_max']:.4f} (fresh material always nearly incompressible)"
            )
        if vs["any_vp_ratio_flag"]:
            concerns.append(
                f"Elastic predictor ratio >> 1: operator split may need smaller dt"
            )
        if pl["any_locking_flag"]:
            concerns.append(
                f"Locking ratio = {pl['worst_locking_ratio']:.1f} for fresh cells: "
                f"consider B-bar or selective reduced integration"
            )
        if "vp_accuracy" in results and results["vp_accuracy"]["any_accuracy_flag"]:
            concerns.append(
                f"VP sub-stepping inaccurate: consider smaller dt or higher-order scheme"
            )
        if "stagger_error" in results and results["stagger_error"]["any_stagger_flag"]:
            concerns.append(
                f"Stagger error dominates residual (ratio={results['stagger_error']['worst_stagger_ratio']:.1f}): "
                f"Newton corrects operator-split error, not load increment"
            )

        if concerns:
            print(f"\n  Conditioning concerns:")
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

        log_path = output_dir / f"phase5_{mesh_tag}_{timestamp}.log"
        log_path.write_text(tee.get_log() if tee else "")

        json_path = output_dir / f"phase5_{mesh_tag}_{timestamp}.json"
        json_data = {
            "timestamp": timestamp,
            "mesh_tag": mesh_tag,
            "elapsed_s": elapsed,
            "n_steps": args.n_steps,
            "results": _results_to_serializable(results),
        }
        json_path.write_text(json.dumps(json_data, indent=2))

        print(f"\n  Output saved to:")
        print(f"    Log:  {log_path}")
        print(f"    JSON: {json_path}")


if __name__ == "__main__":
    main()
