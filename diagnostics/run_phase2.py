"""Run all Phase 2 penalty parameter adequacy diagnostics.

Usage::

    python -m diagnostics.run_phase2 [--small] [--config PATH] [--skip-sweep]

The ``--small`` flag uses a 4x4x2 mesh (32 cells, 768 DOFs) for fast
iteration.  ``--skip-sweep`` skips the Newton-solve sweeps (sub-test 2.1),
which is the most expensive part.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import json

import numpy as np
from mpi4py import MPI

from diagnostics.setup import build_diagnostic_state, prepare_state_at_time
from diagnostics.penalty_sweep import run_penalty_sweep
from diagnostics.stiffness_hierarchy import (
    extract_interlayer_facet_data,
    extract_intralayer_facet_data,
    check_stiffness_hierarchy,
    check_dirichlet_penalty,
    compute_kn_kmin_scaling,
)
from diagnostics.tsl_analysis import (
    compute_tsl_comparison,
    compute_fix2_recommendations,
    _print_fix2_recommendations,
)
from diagnostics.run_phase1 import _TeeWriter, _to_json_safe, _results_to_serializable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Penalty parameter adequacy diagnostics",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use small mesh (4x4x2 = 32 cells) for fast testing",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip Newton-solve sweeps (sub-test 2.1, most expensive)",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    mesh_overrides = None
    if args.small:
        mesh_overrides = {"n_length": 4, "n_span": 4, "n_thickness": 2}

    # Output capture (rank 0 only).
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
        print("  PHASE 2: PENALTY PARAMETER ADEQUACY")
        print("=" * 72)
        if args.small:
            print("  Mode: SMALL mesh (4x4x2)")
        print(flush=True)

    state = build_diagnostic_state(
        config_path=args.config, mesh_overrides=mesh_overrides, comm=comm,
    )

    bt = state.birth_times_dolfinx
    t_max = float(bt.max())
    t_min = float(bt.min())
    t_midpoint = 0.5 * (t_min + t_max)
    t_all_active = t_max * 1.1

    results = {}

    # -------------------------------------------------------------------
    # Sub-test 2.1: Penalty parameter sweeps
    # -------------------------------------------------------------------
    if not args.skip_sweep:
        # 2.1a: gamma_bonded_mult
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 2.1a: gamma_bonded_mult sweep")
            print(f"  t_val = {t_midpoint:.2f} s (~50% active)")
            print("-" * 72, flush=True)

        results["sweep_gamma_bonded"] = run_penalty_sweep(
            state, "gamma_bonded_mult",
            values=[5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
            t_val=t_midpoint,
        )

        # 2.1b: gamma_boundary_mult
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 2.1b: gamma_boundary_mult sweep")
            print(f"  t_val = {t_midpoint:.2f} s (~50% active)")
            print("-" * 72, flush=True)

        results["sweep_gamma_boundary"] = run_penalty_sweep(
            state, "gamma_boundary_mult",
            values=[10.0, 50.0, 100.0, 300.0, 500.0, 1000.0, 2000.0],
            t_val=t_midpoint,
        )

        # 2.1c: K_min_mult
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 2.1c: K_min_mult sweep")
            print(f"  t_val = {t_midpoint:.2f} s (~50% active)")
            print("-" * 72, flush=True)

        results["sweep_K_min"] = run_penalty_sweep(
            state, "K_min_mult",
            values=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            t_val=t_midpoint,
        )

    # -------------------------------------------------------------------
    # Sub-test 2.2: Stiffness hierarchy
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 2.2: Stiffness Hierarchy Check")
        print(f"  t_val = {t_midpoint:.2f} s (~50% active)")
        print("-" * 72, flush=True)

    prepare_state_at_time(state, t_midpoint, u_mode="zero")
    interlayer_data = extract_interlayer_facet_data(state, t_midpoint)
    intralayer_data = extract_intralayer_facet_data(state, t_midpoint)
    results["stiffness_hierarchy"] = check_stiffness_hierarchy(
        interlayer_data, intralayer_data,
    )

    # -------------------------------------------------------------------
    # Sub-test 2.3: Dirichlet penalty check
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 2.3: Dirichlet Penalty Check")
        print("-" * 72, flush=True)

    t_points = [
        t_min + 0.1 * (t_max - t_min),
        t_midpoint,
        t_max * 0.9,
        t_all_active,
    ]
    results["dirichlet_penalty"] = check_dirichlet_penalty(
        state, t_points, interlayer_data=interlayer_data,
    )

    # -------------------------------------------------------------------
    # Sub-test 2.4: K_n / K_min scaling analysis (analytical)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 2.4: K_n/K_min Scaling Analysis (analytical)")
        print("-" * 72, flush=True)

    # Use ORIGINAL config values (not overridden --small values) for
    # production mesh estimate. Re-read the config file if a config path
    # was specified, otherwise use default values from config.yaml.
    import yaml
    orig_cfg_path = args.config or str(Path("config") / "config.yaml")
    with open(orig_cfg_path, "r") as f:
        orig_cfg = yaml.safe_load(f)
    orig_geom = orig_cfg["geometry"]
    orig_mesh = orig_cfg["mesh"]
    dx = orig_geom["length"] / orig_mesh["n_length"]
    dy = orig_geom["span"] / orig_mesh["n_span"]
    dz = orig_geom["thickness"] / orig_mesh["n_thickness"]
    h_prod_est = np.sqrt(dx**2 + dy**2 + dz**2)

    # Estimate t_open from layer deposition time on the production mesh.
    tcp_speed = orig_cfg["activation"]["tcp_speed"]
    n_passes_per_layer = orig_mesh["n_span"]
    layer_print_path = n_passes_per_layer * orig_geom["length"]
    t_open_est = layer_print_path / tcp_speed

    h_values = [5.0, 10.0, 15.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    ages = [0.0, 30.0, 100.0, 150.0, 300.0]

    results["kn_kmin_scaling"] = compute_kn_kmin_scaling(
        state.cfg, h_values=h_values, ages=ages, t_open=t_open_est,
        verbose=(comm.rank == 0),
    )

    if comm.rank == 0:
        print(f"\n    Production mesh estimate: h ≈ {h_prod_est:.1f} mm, "
              f"t_open ≈ {t_open_est:.1f} s")

    # -------------------------------------------------------------------
    # Sub-test 2.5: TSL floor distortion analysis (analytical)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 2.5: TSL Floor Distortion Analysis (analytical)")
        print("-" * 72, flush=True)

    results["tsl_comparison"] = compute_tsl_comparison(
        orig_cfg,
        ages=ages,
        h_values=[h_prod_est],
        t_open=t_open_est,
        verbose=(comm.rank == 0),
    )

    # Cross-reference with K_min_mult sweep for Fix 2 recommendation.
    sweep_k_min = results.get("sweep_K_min", None)
    results["fix2_recommendation"] = compute_fix2_recommendations(
        results["tsl_comparison"], sweep_result=sweep_k_min,
    )
    if comm.rank == 0:
        _print_fix2_recommendations(results["fix2_recommendation"])

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - t0

    if comm.rank == 0:
        print("\n" + "=" * 72)
        print("  SUMMARY")
        print("=" * 72)

        # Sweep summaries.
        if not args.skip_sweep:
            for key, label in [
                ("sweep_gamma_bonded", "gamma_bonded_mult"),
                ("sweep_gamma_boundary", "gamma_boundary_mult"),
                ("sweep_K_min", "K_min_mult"),
            ]:
                sw = results.get(key, {})
                sweep = sw.get("sweep", [])
                if sweep:
                    default = sw.get("default_value")
                    # Find default result.
                    default_entry = next(
                        (e for e in sweep if e["value"] == default), None,
                    )
                    all_conv = all(e["converged"] for e in sweep)
                    n_conv = sum(1 for e in sweep if e["converged"])
                    iter_range = f"{min(e['n_iter'] for e in sweep)}-{max(e['n_iter'] for e in sweep)}"
                    status = "PASS" if all_conv else f"{n_conv}/{len(sweep)} conv"
                    default_iters = default_entry["n_iter"] if default_entry else "?"
                    print(f"  Test 2.1  {label:<25s}: {status}  "
                          f"(default={default} -> {default_iters} iters, "
                          f"range={iter_range})")

        # Hierarchy summary.
        sh = results.get("stiffness_hierarchy", {})
        ds1 = sh.get("dS1", {})
        if ds1.get("n_facets", 0) > 0:
            checks = [ds1.get("Kn_gb_ok", False), ds1.get("Kt_gb_ok", False)]
            n_ok = sum(checks)
            status = "PASS" if all(checks) else f"WARN ({n_ok}/2)"
            max_Kn_gb = ds1.get("K_n_over_gamma_bonded", {}).get("max", 0)
            print(f"  Test 2.2  Stiffness hierarchy:    {status}  "
                  f"(max K_n/gamma_bonded = {max_Kn_gb:.4f})")
        else:
            print(f"  Test 2.2  Stiffness hierarchy:    N/A (no inter-layer facets)")

        # Dirichlet summary.
        dp = results.get("dirichlet_penalty", {})
        mult = dp.get("gamma_boundary_mult", 0)
        in_range = dp.get("in_range_10_1000", False)
        # Find worst conditioning ratio across time points.
        worst_cr = None
        for entry in dp.get("per_time", []):
            cr = entry.get("conditioning_ratio")
            if cr is not None:
                if worst_cr is None or cr > worst_cr:
                    worst_cr = cr
        cr_str = f"max(gamma_bdy)/min(K_n)={worst_cr:.2e}" if worst_cr else ""
        status = "PASS" if in_range else "WARN"
        print(f"  Test 2.3  Dirichlet penalty:      {status}  "
              f"(mult={mult}  {cr_str})")

        # K_n/K_min scaling summary.
        scaling = results.get("kn_kmin_scaling", {})
        scaling_rows = scaling.get("rows", [])
        if scaling_rows:
            # Find the ratio at h ≈ 15mm (production mesh) for the oldest age.
            h_ref = 15.0
            oldest = scaling_rows[-1]
            ratio_at_href = oldest.get("ratio_n_per_h", {}).get(h_ref, None)
            h_star = oldest.get("h_star_n", 0)
            if ratio_at_href is not None:
                floor_status = "FLOOR" if ratio_at_href < 1.0 else "PHYS"
                print(f"  Test 2.4  K_n/K_min scaling:      {floor_status}  "
                      f"(ratio={ratio_at_href:.3f} at h=15mm/age={oldest['age']:.0f}s, "
                      f"h*={h_star:.0f}mm)")
            else:
                print(f"  Test 2.4  K_n/K_min scaling:      (see detailed output)")

        # TSL distortion summary.
        tsl = results.get("tsl_comparison", {})
        tsl_rows = tsl.get("rows", [])
        if tsl_rows:
            worst_peak = max(r["peak_ratio"] for r in tsl_rows)
            worst_delta = min(r["delta_c_ratio"] for r in tsl_rows)
            print(f"  Test 2.5  TSL floor distortion:   "
                  f"peak_ratio={worst_peak:.1f}x  "
                  f"delta_c_ratio={worst_delta:.3f}x")

        # Fix 2 direction.
        fix2 = results.get("fix2_recommendation", {})
        if fix2:
            direction = fix2.get("direction", "?")
            print(f"  Fix 2     Direction:              {direction}")

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

        log_path = output_dir / f"phase2_{mesh_tag}_{timestamp}.log"
        log_path.write_text(tee.get_log() if tee else "")

        json_path = output_dir / f"phase2_{mesh_tag}_{timestamp}.json"
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
