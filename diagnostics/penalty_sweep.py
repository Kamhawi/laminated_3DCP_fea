"""Penalty parameter sweep diagnostics (Test 2.1).

Rebuilds the weak form with varied penalty multipliers and runs Newton
solves to measure convergence sensitivity.  Optionally estimates the
Jacobian condition number at each sweep point.
"""

import copy
import gc

import numpy as np
from dolfinx.fem.petsc import assemble_matrix, create_vector
from mpi4py import MPI
from petsc4py import PETSc

from diagnostics.setup import prepare_state_at_time
from diagnostics.taylor_test import _assemble_residual_into
from physics.weak_form import build_evp_cohesive_weak_form
from solver.newton import solve_newton


# ---------------------------------------------------------------------------
# Form rebuild helper
# ---------------------------------------------------------------------------

def _rebuild_forms_with_override(state, param_name, value):
    """Deep-copy config, override one penalty parameter, rebuild forms.

    Returns ``(F_form, J_form)``.  Side effect: replaces
    ``state.materials.t_current`` with a new ``fem.Constant`` — the caller
    must call ``prepare_state_at_time`` afterwards to set its value.
    """
    cfg_copy = copy.deepcopy(state.cfg)
    cfg_copy["interface"][param_name] = value
    F_new, J_new = build_evp_cohesive_weak_form(
        state.msh, state.V, state.materials,
        state.birth_time_func,
        state.interior_facet_tags, state.dirichlet_tags,
        cfg_copy,
    )
    return F_new, J_new


# ---------------------------------------------------------------------------
# Condition number estimation (optional)
# ---------------------------------------------------------------------------

def _estimate_condition_number(J_form, state):
    """Rough condition number estimate via PETSc matrix norms.

    Assembles J, pins inactive DOFs, then computes:
        kappa_lower_bound = ||J||_F * ||J^{-1}e||_2 / ||e||_2

    For a more rigorous estimate, SLEPc eigenvalue solves would be needed.
    This function returns a norm-based estimate or ``None`` on failure.

    Args:
        J_form: Compiled Jacobian form.
        state: DiagnosticState with inactive_dofs set.

    Returns:
        dict with ``J_norm``, ``J_inv_norm_est``, ``kappa_est`` or ``None``.
    """
    try:
        A = assemble_matrix(J_form)
        A.assemble()
        if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
            A.zeroRowsLocal(state.inactive_dofs, diag=1.0)

        J_norm = A.norm(PETSc.NormType.FROBENIUS)

        # Estimate ||J^{-1}|| via one linear solve with a random RHS.
        rng = np.random.RandomState(42)
        b_vec = A.createVecLeft()
        x_vec = A.createVecLeft()
        b_arr = rng.randn(len(b_vec.array))
        b_arr /= np.linalg.norm(b_arr)
        b_vec.array[:] = b_arr
        b_vec.assemble()

        ksp = PETSc.KSP().create(state.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setUp()
        ksp.solve(b_vec, x_vec)

        x_norm = x_vec.norm()
        kappa_est = J_norm * x_norm  # ||J|| * ||J^{-1} e|| / ||e|| (||e||=1)

        ksp.destroy()
        x_vec.destroy()
        b_vec.destroy()
        A.destroy()

        return {
            "J_norm": float(J_norm),
            "J_inv_norm_est": float(x_norm),
            "kappa_est": float(kappa_est),
        }
    except Exception:
        return None
    finally:
        try:
            PETSc.garbage_cleanup(state.comm)
        except Exception:
            pass
        gc.collect()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_penalty_sweep(state, param_name, values, t_val, max_iter=50,
                      estimate_cond=False, verbose=True):
    """Sweep one penalty parameter and record Newton convergence.

    For each value in ``values``, rebuilds the weak form with the modified
    parameter, runs a Newton solve from zero displacement, and records the
    convergence behaviour.

    Args:
        state: DiagnosticState.
        param_name: Config key under ``interface``, e.g.
            ``"gamma_bonded_mult"``.
        values: List of multiplier values to sweep.
        t_val: Simulation time at which to run the solves.
        max_iter: Maximum Newton iterations per sweep point.
        estimate_cond: If True, estimate kappa(J) at each sweep point.
        verbose: Print results on rank 0.

    Returns:
        dict with sweep results.
    """
    comm = state.comm
    default_value = state.cfg["interface"].get(param_name, None)

    # Save original forms and t_current so we can restore afterwards.
    orig_F = state.F_form
    orig_J = state.J_form
    orig_t_current = state.materials.t_current

    sweep_results = []

    for val in values:
        if verbose and comm.rank == 0:
            print(f"    {param_name} = {val:.4g} ... ", end="", flush=True)

        # Rebuild forms with overridden parameter.
        F_new, J_new = _rebuild_forms_with_override(state, param_name, val)

        # Prepare state: sets t_current on the new Constant, updates materials,
        # resets u to zero, computes inactive DOFs.
        prepare_state_at_time(state, t_val, u_mode="zero")

        # Run Newton solve (creates and destroys its own workspace).
        n_iter, converged, msg = solve_newton(
            state.u, state.V, state.msh, F_new, J_new,
            t_val=t_val,
            birth_times=state.birth_times_dolfinx,
            cell_to_dofs=state.cell_to_dofs,
            max_iter=max_iter,
            workspace=None,
            debug=False,
        )

        # Record max displacement.
        max_disp_local = float(np.max(np.abs(state.u.x.array)))
        max_disp = comm.allreduce(max_disp_local, op=MPI.MAX)

        # Assemble residual at final state for the residual norm.
        b = create_vector(F_new)
        _assemble_residual_into(b, F_new)
        if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
            b.array[state.inactive_dofs] = 0.0
        final_res = b.norm()
        b.destroy()

        # Optional condition number estimation.
        cond = None
        if estimate_cond:
            # Reset to zero for a clean J assembly.
            state.u.x.array[:] = 0.0
            if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
                state.u.x.array[state.inactive_dofs] = 0.0
            state.u.x.scatter_forward()
            cond = _estimate_condition_number(J_new, state)

        entry = {
            "value": float(val),
            "n_iter": int(n_iter),
            "converged": bool(converged),
            "final_residual": float(final_res),
            "max_displacement": float(max_disp),
            "message": msg,
            "condition": cond,
        }
        sweep_results.append(entry)

        if verbose and comm.rank == 0:
            status = "CONV" if converged else "FAIL"
            cond_str = ""
            if cond is not None:
                cond_str = f"  kappa~{cond['kappa_est']:.2e}"
            print(f"{status} in {n_iter} iters  "
                  f"||R||={final_res:.2e}  |u|_max={max_disp:.2e}{cond_str}")

    # Restore original forms and t_current.
    state.F_form = orig_F
    state.J_form = orig_J
    state.materials.t_current = orig_t_current

    result = {
        "param_name": param_name,
        "default_value": default_value,
        "t_val": float(t_val),
        "sweep": sweep_results,
    }

    if verbose and comm.rank == 0:
        _print_sweep_summary(result)

    return result


def _print_sweep_summary(result):
    """Print a compact summary table for the sweep."""
    param = result["param_name"]
    default = result["default_value"]
    print(f"\n  Sweep summary: {param}  (default={default})")
    print(f"    {'Value':>10s}  {'Iters':>5s}  {'Status':>6s}  "
          f"{'||R||':>10s}  {'|u|_max':>10s}")
    print(f"    {'-'*10}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}")
    for entry in result["sweep"]:
        val = entry["value"]
        n = entry["n_iter"]
        status = "CONV" if entry["converged"] else "FAIL"
        marker = " *" if default is not None and val == default else "  "
        print(f"    {val:10.4g}  {n:5d}  {status:>6s}  "
              f"{entry['final_residual']:10.2e}  "
              f"{entry['max_displacement']:10.2e}{marker}")
