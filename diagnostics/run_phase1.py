"""Run all Phase 1 Jacobian consistency diagnostics.

Usage::

    python -m diagnostics.run_phase1 [--small] [--config PATH] [--skip-componentwise]

The ``--small`` flag uses a 4x4x2 mesh (32 cells, 768 DOFs) for fast
iteration (~seconds).  Without it, the production mesh from config.yaml
is used.
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dolfinx.fem.petsc import assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc

from diagnostics.setup import build_diagnostic_state, prepare_state_at_time
from diagnostics.taylor_test import (
    run_taylor_test,
    run_componentwise_taylor_test,
    _assemble_residual_into,
)
from diagnostics.symmetry_check import check_jacobian_symmetry
from diagnostics.fd_column_check import run_fd_column_check
from diagnostics.error_localization import run_error_localization
from solver.newton import get_inactive_dofs


# ---------------------------------------------------------------------------
# Output capture: tee stdout to both terminal and log file
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Write to both the original stdout and a StringIO buffer."""

    def __init__(self, original):
        self._original = original
        self._buffer = io.StringIO()

    def write(self, s):
        self._original.write(s)
        self._buffer.write(s)

    def flush(self):
        self._original.flush()

    def get_log(self):
        return self._buffer.getvalue()


def _to_json_safe(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _results_to_serializable(results):
    """Convert results dict to JSON-serializable form (numpy → lists)."""
    return _to_json_safe(results)


# ---------------------------------------------------------------------------
# Phase 0: Residual assembly sanity check
# ---------------------------------------------------------------------------

def run_residual_sanity_check(state, verbose=True):
    """Verify that F(0) at full activation matches the gravity load.

    At u=0 with all cells fully active, the only residual contribution
    should come from the body force (gravity) term.  The bulk, cohesive,
    SIP, and Nitsche terms all vanish at u=0.

    Args:
        state: DiagnosticState (will be modified: u set to 0, t_val
            set so all cells are active).
        verbose: Print results on rank 0.

    Returns:
        dict: ``F0_norm``, ``gravity_estimate``, ``ratio``, ``passed``.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx
    t_all_active = float(bt.max()) * 1.1

    prepare_state_at_time(state, t_all_active, u_mode="zero")

    # Assemble F(0).
    b = create_vector(state.F_form)
    _assemble_residual_into(b, state.F_form)
    F0_norm = b.norm()
    b.destroy()

    # Rough analytical estimate of gravity load norm.
    # At u=0 the gravity term is: -alpha * inner(g_vec, v) * dx
    # For DG1 with full activation (alpha≈1), the assembled vector
    # magnitude scales as rho * g * V_mesh.
    # We use the material constants directly.
    rho = state.materials.rho_ns2_mm4  # N*s^2/mm^4
    g = state.materials.g_val  # mm/s^2
    geom = state.cfg["geometry"]
    # Approximate mesh volume from geometry.
    span = geom["span"]
    length = geom["length"]
    thickness = geom["thickness"]
    # Barrel vault volume ~ span * length * thickness (rough, ignores curvature)
    V_approx = span * length * thickness  # mm^3
    gravity_estimate = rho * g * V_approx

    ratio = F0_norm / max(gravity_estimate, 1e-30)
    passed = 0.01 < ratio < 100.0  # Very broad sanity check

    if verbose and comm.rank == 0:
        print("\n  Phase 0: Residual assembly sanity check")
        print(f"    ||F(u=0)||           = {F0_norm:.4e}")
        print(f"    rho*g*V (estimate)   = {gravity_estimate:.4e}")
        print(f"    Ratio                = {ratio:.4f}")
        status = "PASS" if passed else "FAIL"
        print(f"    Status: {status}")
        if not passed:
            print("    WARNING: F(0) does not match gravity estimate — "
                  "possible assembly bug")

    return {
        "F0_norm": F0_norm,
        "gravity_estimate": gravity_estimate,
        "ratio": ratio,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Jacobian consistency diagnostics for 3DCP FEA",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use small mesh (4x4x2 = 32 cells) for fast testing",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--skip-componentwise", action="store_true",
        help="Skip component-wise Taylor test (slowest)",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    mesh_overrides = None
    if args.small:
        mesh_overrides = {"n_length": 4, "n_span": 4, "n_thickness": 2}

    # Set up output capture (rank 0 only).
    tee = None
    if comm.rank == 0:
        tee = _TeeWriter(sys.stdout)
        sys.stdout = tee

    t0 = time.time()

    # -----------------------------------------------------------------------
    # Build diagnostic state
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("=" * 72)
        print("  PHASE 1: JACOBIAN CONSISTENCY VERIFICATION")
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
    t_all_active = t_max * 1.1
    t_midpoint = 0.5 * (t_min + t_max)
    # Just past last birth: youngest interface is fresh.
    t_fresh_interface = t_max + 0.01 * (t_max - t_min)

    results = {}

    # -----------------------------------------------------------------------
    # Phase 0: Residual sanity check
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Phase 0: Residual Assembly Sanity Check")
        print("-" * 72, flush=True)

    results["phase0"] = run_residual_sanity_check(state)

    # -----------------------------------------------------------------------
    # Test 1.1 (baseline): Global Taylor test — all cells active
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 1.1 (baseline): Global Taylor — all cells active")
        print(f"  t_val = {t_all_active:.2f} s")
        print("-" * 72, flush=True)

    prepare_state_at_time(state, t_all_active, u_mode="random_small")
    results["taylor_baseline"] = run_taylor_test(
        state, state.F_form, state.J_form,
        label="global (all active)",
    )

    # -----------------------------------------------------------------------
    # Test 1.1 (partial): Global Taylor test — ~50% cells active
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 1.1 (partial): Global Taylor — ~50% active")
        print(f"  t_val = {t_midpoint:.2f} s")
        print("-" * 72, flush=True)

    prepare_state_at_time(state, t_midpoint, u_mode="random_small")
    results["taylor_partial"] = run_taylor_test(
        state, state.F_form, state.J_form,
        label="global (~50% active)",
    )

    # -----------------------------------------------------------------------
    # Test 1.2: Component-wise Taylor test
    # -----------------------------------------------------------------------
    if not args.skip_componentwise:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 1.2: Component-wise Taylor — all active (K_min branch)")
            print(f"  t_val = {t_all_active:.2f} s")
            print("-" * 72, flush=True)

        prepare_state_at_time(state, t_all_active, u_mode="random_small")
        results["componentwise_allactive"] = run_componentwise_taylor_test(state)

        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 1.2: Component-wise Taylor — fresh interface")
            print(f"  t_val = {t_fresh_interface:.2f} s")
            print("-" * 72, flush=True)

        prepare_state_at_time(state, t_fresh_interface, u_mode="random_small")
        results["componentwise_fresh"] = run_componentwise_taylor_test(state)

    # -----------------------------------------------------------------------
    # Test 1.3: Jacobian symmetry check
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 1.3: Jacobian Symmetry Check")
        print(f"  t_val = {t_all_active:.2f} s")
        print("-" * 72, flush=True)

    prepare_state_at_time(state, t_all_active, u_mode="random_small")
    results["symmetry"] = check_jacobian_symmetry(state)

    # -----------------------------------------------------------------------
    # Test 1.4: FD column spot-check
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 1.4: FD Jacobian Column Spot-Check")
        print(f"  t_val = {t_all_active:.2f} s")
        print("-" * 72, flush=True)

    prepare_state_at_time(state, t_all_active, u_mode="random_small")
    results["fd_columns"] = run_fd_column_check(state)

    # -----------------------------------------------------------------------
    # Test 1.5: Error localization at partial activation
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 1.5: Error Localization (~50% active)")
        print(f"  t_val = {t_midpoint:.2f} s")
        print("-" * 72, flush=True)

    prepare_state_at_time(state, t_midpoint, u_mode="random_small")
    results["error_localization"] = run_error_localization(state)

    # -----------------------------------------------------------------------
    # Test 1.6: Inactive DOF consistency check
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 1.6: Inactive DOF Consistency Check")
        print("-" * 72, flush=True)

    # Re-compute inactive DOFs at the same t_val and verify they match.
    verify_inactive = get_inactive_dofs(
        state.t_val, state.birth_times_dolfinx, state.cell_to_dofs,
    )
    num_owned = len(state.u.x.array)
    verify_inactive = verify_inactive[verify_inactive < num_owned].astype(
        np.int32, copy=False,
    )
    dof_match = np.array_equal(state.inactive_dofs, verify_inactive)
    results["inactive_dof_check"] = {
        "n_setup": len(state.inactive_dofs),
        "n_verify": len(verify_inactive),
        "match": bool(dof_match),
    }
    if comm.rank == 0:
        print(f"    Inactive DOFs at setup:  {len(state.inactive_dofs)}")
        print(f"    Inactive DOFs at verify: {len(verify_inactive)}")
        print(f"    Status: {'PASS (match)' if dof_match else 'FAIL (mismatch)'}")

    # -----------------------------------------------------------------------
    # Test 1.7: Birth time boundary check
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 1.7: Birth Time Boundary Check")
        print("-" * 72, flush=True)

    n_exact = int(np.sum(state.birth_times_dolfinx == state.t_val))
    n_exact_global = comm.allreduce(n_exact, op=MPI.SUM)
    results["birth_boundary"] = {
        "t_val": state.t_val,
        "n_cells_exact_match": n_exact_global,
    }
    if comm.rank == 0:
        print(f"    t_val = {state.t_val:.6f} s")
        print(f"    Cells with birth_time == t_val: {n_exact_global}")
        if n_exact_global == 0:
            print("    Status: OK (no boundary ambiguity)")
        else:
            print("    Status: NOTE — cells sit exactly on the activation "
                  "boundary (strict > vs >= matters)")

    # -----------------------------------------------------------------------
    # Test 1.1c: Component-wise Taylor at partial activation
    # -----------------------------------------------------------------------
    if not args.skip_componentwise:
        if comm.rank == 0:
            print("\n" + "-" * 72)
            print("  Test 1.1c: Component-wise Taylor — ~50% active")
            print(f"  t_val = {t_midpoint:.2f} s")
            print("-" * 72, flush=True)

        prepare_state_at_time(state, t_midpoint, u_mode="random_small")
        results["componentwise_partial"] = run_componentwise_taylor_test(state)

    # -----------------------------------------------------------------------
    # Test 1.4b: FD column check at partial activation
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "-" * 72)
        print("  Test 1.4b: FD Column Spot-Check (~50% active)")
        print(f"  t_val = {t_midpoint:.2f} s")
        print("-" * 72, flush=True)

    prepare_state_at_time(state, t_midpoint, u_mode="random_small")
    results["fd_columns_partial"] = run_fd_column_check(state)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0

    if comm.rank == 0:
        print("\n" + "=" * 72)
        print("  SUMMARY")
        print("=" * 72)

        # Phase 0
        p0 = results["phase0"]
        print(f"  Phase 0  Residual sanity:     "
              f"{'PASS' if p0['passed'] else 'FAIL'}  "
              f"(ratio={p0['ratio']:.3f})")

        # Taylor baseline
        tb = results["taylor_baseline"]
        _print_taylor_summary("Test 1.1a", "Taylor (all active)", tb)

        # Taylor partial
        tp = results["taylor_partial"]
        _print_taylor_summary("Test 1.1b", "Taylor (~50% active)", tp)

        # Component-wise
        if not args.skip_componentwise:
            cw = results.get("componentwise_fresh", [])
            if cw:
                # Find the cohesive term (index 1).
                cohesive_result = cw[1] if len(cw) > 1 else None
                if cohesive_result:
                    _print_taylor_summary(
                        "Test 1.2",
                        "Cohesive term (fresh iface)",
                        cohesive_result,
                    )

        # Symmetry
        sym = results["symmetry"]
        sym_status = "PASS" if sym["ratio"] < 1e-6 else "WARN"
        print(f"  Test 1.3  Sym (full Jacobian):  "
              f"{sym_status}  (defect={sym['ratio']:.2e})")

        # FD columns
        fd = results["fd_columns"]
        max_err = 0.0
        for errs in fd["errors"]:
            max_err = max(max_err, min(errs))  # best error across h values
        fd_status = "PASS" if max_err < 1e-4 else "FAIL"
        print(f"  Test 1.4  FD columns:          "
              f"{fd_status}  (max_err={max_err:.2e})")

        # Error localization
        el = results.get("error_localization", {})
        if el and "fixed" in el:
            el_norm = el["fixed"].get("total", 0.0)
            el_status = "PASS" if el_norm < 1e-6 else "WARN"
            print(f"  Test 1.5  Error localization:   "
                  f"{el_status}  (||c||_fix={el_norm:.2e})")
        elif el:
            print(f"  Test 1.5  Error localization:   N/A (all active)")

        # Inactive DOF consistency
        idc = results.get("inactive_dof_check", {})
        if idc:
            idc_status = "PASS" if idc["match"] else "FAIL"
            print(f"  Test 1.6  Inactive DOF check:   "
                  f"{idc_status}  ({idc['n_setup']} DOFs)")

        # Birth time boundary
        bb = results.get("birth_boundary", {})
        if bb:
            n_ex = bb["n_cells_exact_match"]
            bb_status = "OK" if n_ex == 0 else "NOTE"
            print(f"  Test 1.7  Birth time boundary:  "
                  f"{bb_status}  ({n_ex} cells on boundary)")

        # Component-wise partial
        if not args.skip_componentwise:
            cp = results.get("componentwise_partial", [])
            if cp:
                cohesive_p = cp[1] if len(cp) > 1 else None
                if cohesive_p:
                    _print_taylor_summary(
                        "Test 1.1c",
                        "Cohesive (~50% active)",
                        cohesive_p,
                    )

        # FD columns partial
        fdp = results.get("fd_columns_partial", {})
        if fdp and "errors" in fdp:
            max_err_p = 0.0
            for errs in fdp["errors"]:
                max_err_p = max(max_err_p, min(errs))
            fdp_status = "PASS" if max_err_p < 1e-4 else "FAIL"
            print(f"  Test 1.4b FD columns (partial): "
                  f"{fdp_status}  (max_err={max_err_p:.2e})")

        print(f"\n  Total time: {elapsed:.1f} s")
        print("=" * 72)

    # -------------------------------------------------------------------
    # Save outputs (rank 0 only)
    # -------------------------------------------------------------------
    if comm.rank == 0:
        # Restore stdout before saving.
        if tee is not None:
            sys.stdout = tee._original

        output_dir = Path("diagnostics") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mesh_tag = "small" if args.small else "production"

        # Save log file.
        log_path = output_dir / f"phase1_{mesh_tag}_{timestamp}.log"
        log_path.write_text(tee.get_log() if tee else "")

        # Save structured results as JSON.
        json_path = output_dir / f"phase1_{mesh_tag}_{timestamp}.json"
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


def _print_taylor_summary(test_id, label, result):
    """Print a one-line summary for a Taylor test result."""
    # Handle JIT-failed or empty results.
    if result.get("jit_failed", False) or len(result.get("remainder", [])) == 0:
        print(f"  {test_id:<10s} {label:<25s}: "
              f"SKIP  (JIT compilation failed)")
        return

    rates = result["rates"]
    remainder = result["remainder"]
    eps = result["eps"]

    # Detect noise floor: if max remainder is near machine precision
    # relative to the function/Jacobian norms, it's an exact (bilinear) form.
    noise_floor = max(result.get("F0_norm", 0), result.get("Jdu_norm", 0)) * 1e-14
    if np.max(remainder) < max(noise_floor, 1e-12):
        print(f"  {test_id:<10s} {label:<25s}: "
              f"PASS  (r≈0, bilinear)")
        return

    mask = np.isfinite(rates)
    mid = mask.copy()
    for i in range(len(rates)):
        ep_lo = min(eps[i], eps[i + 1])
        ep_hi = max(eps[i], eps[i + 1])
        if ep_lo < 1e-7 or ep_hi > 1e-2:
            mid[i] = False
        # Also exclude intervals where remainder is in noise floor.
        if remainder[i] < noise_floor or remainder[i + 1] < noise_floor:
            mid[i] = False
    reliable = rates[mid]
    if len(reliable) > 0:
        mean_rate = np.mean(reliable)
        status = "PASS" if mean_rate > 1.8 else "FAIL"
        print(f"  {test_id:<10s} {label:<25s}: "
              f"{status}  (rate={mean_rate:.2f})")
    else:
        print(f"  {test_id:<10s} {label:<25s}: "
              f"N/A   (no reliable data)")


if __name__ == "__main__":
    main()
