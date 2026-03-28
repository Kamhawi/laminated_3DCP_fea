"""Activation mechanics analysis (Phase 3 diagnostics).

Provides tests for:
  3.1a  Alpha contrast ratio across time
  3.1b  Sharpness sweep (Newton convergence)
  3.1c  Alpha_min sweep (Newton convergence)
  3.2   Newly-activated cell residual contribution
  3.3   Inactive DOF pinning completeness
  3.4   Step-by-step activation load correlation

All tests operate on a DiagnosticState from ``diagnostics.setup``.
"""

import copy

import numpy as np
from dolfinx.fem.petsc import create_vector
from mpi4py import MPI

from diagnostics.setup import prepare_state_at_time
from diagnostics.taylor_test import _assemble_residual_into
from physics.weak_form import build_evp_cohesive_weak_form
from solver.kinematics import (
    init_newly_activated_displacement,
    zero_inactive_cells,
)
from solver.newton import solve_newton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_alpha_values(t_val, birth_times, sharpness, alpha_min):
    """Compute per-cell activation alpha via the tanh ramp formula.

    Matches ``physics/weak_form.py`` lines 91-92 exactly.

    Args:
        t_val: Current simulation time.
        birth_times: Array of cell birth times (DOLFINx ordering).
        sharpness: Tanh sharpness parameter k.
        alpha_min: Minimum activation floor.

    Returns:
        np.ndarray of per-cell alpha values.
    """
    alpha_raw = 0.5 * (1.0 + np.tanh(sharpness * (t_val - birth_times)))
    return alpha_min + (1.0 - alpha_min) * alpha_raw


def _rebuild_forms_activation(state, sharpness=None, alpha_min=None):
    """Rebuild weak forms with overridden activation parameters.

    Deep-copies config, overrides activation keys, calls
    ``build_evp_cohesive_weak_form``.

    Side effect: replaces ``state.materials.t_current`` with a new
    ``fem.Constant``.  Caller must call ``prepare_state_at_time``
    afterwards.

    Returns:
        (F_form, J_form)
    """
    cfg_copy = copy.deepcopy(state.cfg)
    if sharpness is not None:
        cfg_copy["activation"]["sharpness"] = sharpness
    if alpha_min is not None:
        cfg_copy["activation"]["alpha_min"] = alpha_min
    F_new, J_new = build_evp_cohesive_weak_form(
        state.msh, state.V, state.materials,
        state.birth_time_func,
        state.interior_facet_tags, state.dirichlet_tags,
        cfg_copy,
    )
    return F_new, J_new


def _generate_time_points(state, n_steps=20):
    """Replicate the simulation time grid for the first *n_steps* steps.

    Matches ``solver/time_stepper.py`` lines 489-493.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx
    time_cfg = state.cfg["time_stepping"]

    local_t_min = float(bt.min()) if bt.size > 0 else np.inf
    local_t_max = float(bt.max()) if bt.size > 0 else -np.inf
    global_t_min = comm.allreduce(local_t_min, op=MPI.MIN)
    global_t_max = comm.allreduce(local_t_max, op=MPI.MAX)

    full_times = np.linspace(
        global_t_min + time_cfg["start_offset"],
        global_t_max * time_cfg["end_multiplier"],
        time_cfg["n_steps"],
    )
    return full_times[:n_steps]


# ---------------------------------------------------------------------------
# Test 3.1a: Alpha contrast ratio
# ---------------------------------------------------------------------------

def run_alpha_contrast_analysis(state, time_points=None, verbose=True):
    """Compute alpha contrast ratio at multiple time points.

    For each time, computes ``max_alpha / min_alpha`` over cells with
    ``alpha >= 1e-6``.  Flags if the ratio exceeds 1e6 (near-singular
    Jacobian risk).

    Returns:
        dict with per-time-point records.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx
    sharpness = state.cfg["activation"]["sharpness"]
    alpha_min_val = state.cfg["activation"]["alpha_min"]

    if time_points is None:
        local_t_min = float(bt.min()) if bt.size > 0 else np.inf
        local_t_max = float(bt.max()) if bt.size > 0 else -np.inf
        global_t_min = comm.allreduce(local_t_min, op=MPI.MIN)
        global_t_max = comm.allreduce(local_t_max, op=MPI.MAX)
        time_points = np.linspace(global_t_min, global_t_max * 1.05, 20)

    threshold = 1e-6
    records = []

    for t_val in time_points:
        alpha = compute_alpha_values(float(t_val), bt, sharpness, alpha_min_val)
        mask = alpha >= threshold
        n_above = int(np.sum(mask))

        # MPI-consistent: all ranks have same birth_times in this setup.
        if n_above > 0:
            max_a = float(np.max(alpha[mask]))
            min_a = float(np.min(alpha[mask]))
            ratio = max_a / max(min_a, 1e-30)
        else:
            max_a = 0.0
            min_a = 0.0
            ratio = 1.0

        records.append({
            "t_val": float(t_val),
            "n_cells_above_threshold": n_above,
            "n_total_cells": len(bt),
            "max_alpha": max_a,
            "min_alpha_filtered": min_a,
            "contrast_ratio": ratio,
            "exceeds_1e6": ratio > 1e6,
        })

    if verbose and comm.rank == 0:
        print(f"\n  Alpha Contrast Analysis (threshold={threshold})")
        print(f"    sharpness = {sharpness},  alpha_min = {alpha_min_val}")
        print(f"\n    {'t_val':>10s}  {'n_above':>7s}  {'max_alpha':>10s}  "
              f"{'min_alpha':>10s}  {'ratio':>12s}  {'flag':>5s}")
        print(f"    {'-'*10}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*5}")
        for r in records:
            flag = "WARN" if r["exceeds_1e6"] else "ok"
            print(f"    {r['t_val']:10.2f}  {r['n_cells_above_threshold']:7d}  "
                  f"{r['max_alpha']:10.6f}  {r['min_alpha_filtered']:10.2e}  "
                  f"{r['contrast_ratio']:12.2e}  {flag:>5s}")

    return {
        "sharpness": sharpness,
        "alpha_min": alpha_min_val,
        "threshold": threshold,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Test 3.1b: Sharpness sweep
# ---------------------------------------------------------------------------

def run_sharpness_sweep(state, sharpness_values=None, t_val=None,
                        max_iter=50, verbose=True):
    """Sweep activation sharpness and record Newton convergence.

    For each value, rebuilds forms, runs Newton from zero displacement.

    Returns:
        dict with sweep results.
    """
    if sharpness_values is None:
        sharpness_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    comm = state.comm
    bt = state.birth_times_dolfinx

    if t_val is None:
        local_t_min = float(bt.min()) if bt.size > 0 else np.inf
        local_t_max = float(bt.max()) if bt.size > 0 else -np.inf
        global_t_min = comm.allreduce(local_t_min, op=MPI.MIN)
        global_t_max = comm.allreduce(local_t_max, op=MPI.MAX)
        t_val = 0.5 * (global_t_min + global_t_max)

    default_value = state.cfg["activation"]["sharpness"]

    # Save originals for restore.
    orig_F = state.F_form
    orig_J = state.J_form
    orig_t_current = state.materials.t_current

    sweep_results = []

    for val in sharpness_values:
        if verbose and comm.rank == 0:
            print(f"    sharpness = {val:.4g} ... ", end="", flush=True)

        F_new, J_new = _rebuild_forms_activation(state, sharpness=val)
        prepare_state_at_time(state, t_val, u_mode="zero")

        n_iter, converged, msg = solve_newton(
            state.u, state.V, state.msh, F_new, J_new,
            t_val=t_val,
            birth_times=state.birth_times_dolfinx,
            cell_to_dofs=state.cell_to_dofs,
            max_iter=max_iter,
            workspace=None,
            debug=False,
        )

        max_disp_local = float(np.max(np.abs(state.u.x.array)))
        max_disp = comm.allreduce(max_disp_local, op=MPI.MAX)

        b = create_vector(F_new)
        _assemble_residual_into(b, F_new)
        if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
            b.array[state.inactive_dofs] = 0.0
        final_res = float(b.norm())
        b.destroy()

        entry = {
            "value": float(val),
            "n_iter": int(n_iter),
            "converged": bool(converged),
            "final_residual": final_res,
            "max_displacement": max_disp,
            "message": msg,
        }
        sweep_results.append(entry)

        if verbose and comm.rank == 0:
            status = "CONV" if converged else "FAIL"
            marker = " *" if val == default_value else ""
            print(f"{status} in {n_iter} iters  "
                  f"||R||={final_res:.2e}  |u|_max={max_disp:.2e}{marker}")

    # Restore originals.
    state.F_form = orig_F
    state.J_form = orig_J
    state.materials.t_current = orig_t_current

    result = {
        "param_name": "sharpness",
        "default_value": default_value,
        "t_val": float(t_val),
        "sweep": sweep_results,
    }

    if verbose and comm.rank == 0:
        _print_sweep_summary(result)

    return result


# ---------------------------------------------------------------------------
# Test 3.1c: Alpha_min sweep
# ---------------------------------------------------------------------------

def run_alpha_min_sweep(state, alpha_min_values=None, t_val=None,
                        max_iter=50, verbose=True):
    """Sweep activation alpha_min and record Newton convergence.

    Returns:
        dict with sweep results.
    """
    if alpha_min_values is None:
        alpha_min_values = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

    comm = state.comm
    bt = state.birth_times_dolfinx

    if t_val is None:
        local_t_min = float(bt.min()) if bt.size > 0 else np.inf
        local_t_max = float(bt.max()) if bt.size > 0 else -np.inf
        global_t_min = comm.allreduce(local_t_min, op=MPI.MIN)
        global_t_max = comm.allreduce(local_t_max, op=MPI.MAX)
        t_val = 0.5 * (global_t_min + global_t_max)

    default_value = state.cfg["activation"]["alpha_min"]

    orig_F = state.F_form
    orig_J = state.J_form
    orig_t_current = state.materials.t_current

    sweep_results = []

    for val in alpha_min_values:
        if verbose and comm.rank == 0:
            print(f"    alpha_min = {val:.4g} ... ", end="", flush=True)

        F_new, J_new = _rebuild_forms_activation(state, alpha_min=val)
        prepare_state_at_time(state, t_val, u_mode="zero")

        n_iter, converged, msg = solve_newton(
            state.u, state.V, state.msh, F_new, J_new,
            t_val=t_val,
            birth_times=state.birth_times_dolfinx,
            cell_to_dofs=state.cell_to_dofs,
            max_iter=max_iter,
            workspace=None,
            debug=False,
        )

        max_disp_local = float(np.max(np.abs(state.u.x.array)))
        max_disp = comm.allreduce(max_disp_local, op=MPI.MAX)

        b = create_vector(F_new)
        _assemble_residual_into(b, F_new)
        if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
            b.array[state.inactive_dofs] = 0.0
        final_res = float(b.norm())
        b.destroy()

        entry = {
            "value": float(val),
            "n_iter": int(n_iter),
            "converged": bool(converged),
            "final_residual": final_res,
            "max_displacement": max_disp,
            "message": msg,
        }
        sweep_results.append(entry)

        if verbose and comm.rank == 0:
            status = "CONV" if converged else "FAIL"
            marker = " *" if val == default_value else ""
            print(f"{status} in {n_iter} iters  "
                  f"||R||={final_res:.2e}  |u|_max={max_disp:.2e}{marker}")

    state.F_form = orig_F
    state.J_form = orig_J
    state.materials.t_current = orig_t_current

    result = {
        "param_name": "alpha_min",
        "default_value": default_value,
        "t_val": float(t_val),
        "sweep": sweep_results,
    }

    if verbose and comm.rank == 0:
        _print_sweep_summary(result)

    return result


def _print_sweep_summary(result):
    """Print compact sweep summary table."""
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


# ---------------------------------------------------------------------------
# Test 3.2: Newly-activated cell residual contribution
# ---------------------------------------------------------------------------

def run_newly_activated_residual_check(state, n_steps=20, verbose=True):
    """Measure residual contribution from newly-activated cells.

    Simulates a mini time-stepping loop (no Newton solves).  At each
    step, identifies newly-activated cells, initialises their
    displacement, assembles F, and reports the residual fraction from
    newly-active DOFs.

    Note: The residual at newly-active DOFs includes both the cell's
    bulk stress AND facet integral contributions (SIP, cohesive) coupling
    it to already-equilibrated neighbors.  A high fraction reflects the
    total residual burden from activation, not purely initialization
    quality.

    Returns:
        dict with per-step records.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx
    times = _generate_time_points(state, n_steps)

    if len(times) < 2:
        return {"steps": [], "n_steps": 0}

    dt = times[1] - times[0]
    t_prev = times[0] - dt

    # Reset displacement.
    state.u.x.array[:] = 0.0
    state.u.x.scatter_forward()

    b = create_vector(state.F_form)
    num_owned = len(state.u.x.array)

    records = []

    for i, t_val in enumerate(times):
        t_val_f = float(t_val)

        # Find newly-active cells.
        newly_active = np.where((bt <= t_val_f) & (bt > t_prev))[0]
        n_total_active = int(np.sum(bt <= t_val_f))

        # Update materials.
        state.materials.t_current.value = t_val_f
        state.materials.update_properties(t_val_f)

        # Initialize newly-active displacement.
        init_newly_activated_displacement(
            state.u, newly_active, state.support, state.cell_to_dofs,
        )

        # Zero inactive cells.
        zero_inactive_cells(state.u, t_val_f, bt, state.cell_to_dofs)

        # Assemble residual.
        _assemble_residual_into(b, state.F_form)

        # Zero inactive DOFs in residual for fair comparison.
        inactive_cells = np.where(bt > t_val_f)[0]
        if inactive_cells.size > 0:
            inactive_dofs = np.unique(state.cell_to_dofs[inactive_cells].ravel())
            inactive_dofs_owned = inactive_dofs[inactive_dofs < num_owned]
            b.array[inactive_dofs_owned] = 0.0

        r_total = float(b.norm())

        # Residual from newly-active DOFs.
        r_newly_active = 0.0
        n_newly_active = len(newly_active)
        if n_newly_active > 0:
            newly_active_dofs = np.unique(
                state.cell_to_dofs[newly_active].ravel()
            )
            owned_mask = newly_active_dofs < num_owned
            owned_dofs = newly_active_dofs[owned_mask]
            local_sq = float(np.sum(b.array[owned_dofs] ** 2))
            global_sq = comm.allreduce(local_sq, op=MPI.SUM)
            r_newly_active = np.sqrt(global_sq)

        fraction = r_newly_active / max(r_total, 1e-30) if r_total > 0 else 0.0

        records.append({
            "step": i,
            "t_val": t_val_f,
            "n_newly_active": n_newly_active,
            "n_total_active": n_total_active,
            "r_total": r_total,
            "r_newly_active": r_newly_active,
            "fraction": fraction,
        })

        t_prev = t_val_f

    b.destroy()

    if verbose and comm.rank == 0:
        print(f"\n  Newly-Activated Residual Check ({len(times)} steps)")
        print(f"    Note: fraction includes facet coupling (SIP/cohesive),")
        print(f"    not purely initialization quality.")
        print(f"\n    {'step':>4s}  {'t_val':>10s}  {'n_new':>5s}  {'n_act':>5s}  "
              f"{'r_total':>10s}  {'r_new':>10s}  {'frac':>8s}")
        print(f"    {'-'*4}  {'-'*10}  {'-'*5}  {'-'*5}  "
              f"{'-'*10}  {'-'*10}  {'-'*8}")
        for r in records:
            print(f"    {r['step']:4d}  {r['t_val']:10.2f}  "
                  f"{r['n_newly_active']:5d}  {r['n_total_active']:5d}  "
                  f"{r['r_total']:10.2e}  {r['r_newly_active']:10.2e}  "
                  f"{r['fraction']:8.4f}")

    return {"steps": records, "n_steps": len(times)}


# ---------------------------------------------------------------------------
# Test 3.3: Inactive DOF pinning completeness
# ---------------------------------------------------------------------------

def run_pinning_completeness_check(state, time_points=None,
                                   alpha_threshold=1e-4, verbose=True):
    """Check for cells that are born but effectively zero-stiffness.

    Counts cells where ``birth_time <= t_val`` (not pinned by the binary
    threshold in ``get_inactive_dofs``) but ``alpha(t) < alpha_threshold``
    (near-zero stiffness).  These contribute near-zero diagonal entries
    to J without being identity-pinned.

    Returns:
        dict with per-time-point records.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx
    sharpness = state.cfg["activation"]["sharpness"]
    alpha_min_val = state.cfg["activation"]["alpha_min"]

    if time_points is None:
        local_t_min = float(bt.min()) if bt.size > 0 else np.inf
        local_t_max = float(bt.max()) if bt.size > 0 else -np.inf
        global_t_min = comm.allreduce(local_t_min, op=MPI.MIN)
        global_t_max = comm.allreduce(local_t_max, op=MPI.MAX)
        time_points = np.linspace(global_t_min, global_t_max * 1.05, 20)

    records = []

    for t_val in time_points:
        t_val_f = float(t_val)
        alpha = compute_alpha_values(t_val_f, bt, sharpness, alpha_min_val)

        # Born (not pinned) but effectively zero-stiffness.
        born_mask = bt <= t_val_f
        soft_mask = alpha < alpha_threshold
        unpinned_soft = np.where(born_mask & soft_mask)[0]

        n_total_active = int(np.sum(born_mask))
        n_unpinned_soft = len(unpinned_soft)

        bt_range = None
        if n_unpinned_soft > 0:
            bt_range = [float(bt[unpinned_soft].min()),
                        float(bt[unpinned_soft].max())]

        records.append({
            "t_val": t_val_f,
            "n_unpinned_soft": n_unpinned_soft,
            "n_total_active": n_total_active,
            "fraction": n_unpinned_soft / max(n_total_active, 1),
            "birth_time_range": bt_range,
        })

    if verbose and comm.rank == 0:
        print(f"\n  Pinning Completeness Check (alpha_threshold={alpha_threshold})")
        print(f"    Cells with birth_time <= t but alpha < threshold are")
        print(f"    NOT pinned by get_inactive_dofs but have near-zero stiffness.")
        print(f"\n    {'t_val':>10s}  {'n_unpinned':>10s}  {'n_active':>8s}  "
              f"{'fraction':>8s}  {'bt_range':>20s}")
        print(f"    {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*20}")
        for r in records:
            bt_str = ""
            if r["birth_time_range"] is not None:
                bt_str = f"[{r['birth_time_range'][0]:.1f}, {r['birth_time_range'][1]:.1f}]"
            print(f"    {r['t_val']:10.2f}  {r['n_unpinned_soft']:10d}  "
                  f"{r['n_total_active']:8d}  {r['fraction']:8.4f}  "
                  f"{bt_str:>20s}")

    return {
        "alpha_threshold": alpha_threshold,
        "sharpness": sharpness,
        "alpha_min": alpha_min_val,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Test 3.4: Step-by-step activation load
# ---------------------------------------------------------------------------

def run_step_by_step_activation_load(state, n_steps=20, max_iter=50,
                                     verbose=True):
    """Record activation statistics and Newton convergence per step.

    Runs a mini time-stepping loop with full Newton solves, recording
    newly-active cell counts, initial residual, and iteration counts.

    Computes two correlation metrics:
    - corr(n_newly_active, n_iter): includes path-dependent effects
    - corr(n_newly_active, initial_residual): cleaner activation burden

    Returns:
        dict with per-step records and correlation analysis.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx
    times = _generate_time_points(state, n_steps)

    if len(times) < 2:
        return {"steps": [], "n_steps": 0, "correlations": {}}

    dt = times[1] - times[0]
    t_prev = times[0] - dt

    # Reset displacement.
    state.u.x.array[:] = 0.0
    state.u.x.scatter_forward()

    b = create_vector(state.F_form)
    num_owned = len(state.u.x.array)

    records = []

    for i, t_val in enumerate(times):
        t_val_f = float(t_val)

        # Find newly-active cells.
        newly_active = np.where((bt <= t_val_f) & (bt > t_prev))[0]
        n_newly_active = len(newly_active)
        n_total_active = int(np.sum(bt <= t_val_f))

        bt_range = None
        if n_newly_active > 0:
            bt_range = [float(bt[newly_active].min()),
                        float(bt[newly_active].max())]

        # Initialize newly-active displacement.
        init_newly_activated_displacement(
            state.u, newly_active, state.support, state.cell_to_dofs,
        )

        # Update materials.
        state.materials.t_current.value = t_val_f
        state.materials.update_properties(t_val_f)

        # Assemble residual for initial norm.
        _assemble_residual_into(b, state.F_form)
        inactive_cells = np.where(bt > t_val_f)[0]
        if inactive_cells.size > 0:
            inactive_dofs = np.unique(state.cell_to_dofs[inactive_cells].ravel())
            inactive_dofs_owned = inactive_dofs[inactive_dofs < num_owned]
            b.array[inactive_dofs_owned] = 0.0
        initial_residual = float(b.norm())

        # Newton solve.
        n_iter, converged, msg = solve_newton(
            state.u, state.V, state.msh, state.F_form, state.J_form,
            t_val=t_val_f,
            birth_times=bt,
            cell_to_dofs=state.cell_to_dofs,
            max_iter=max_iter,
            workspace=None,
            debug=False,
        )

        # Post-solve cleanup.
        zero_inactive_cells(state.u, t_val_f, bt, state.cell_to_dofs)

        max_disp_local = float(np.max(np.abs(state.u.x.array)))
        max_disp = comm.allreduce(max_disp_local, op=MPI.MAX)

        records.append({
            "step": i,
            "t_val": t_val_f,
            "n_newly_active": n_newly_active,
            "n_total_active": n_total_active,
            "bt_range_newly_active": bt_range,
            "n_iter": int(n_iter),
            "converged": bool(converged),
            "initial_residual": initial_residual,
            "max_displacement": max_disp,
            "message": msg,
        })

        t_prev = t_val_f

    b.destroy()

    # Correlation analysis.
    correlations = {}
    if len(records) > 2:
        n_new_arr = np.array([r["n_newly_active"] for r in records], dtype=float)
        n_iter_arr = np.array([r["n_iter"] for r in records], dtype=float)
        init_res_arr = np.array([r["initial_residual"] for r in records], dtype=float)

        # Only compute if there's variance.
        if np.std(n_new_arr) > 0 and np.std(n_iter_arr) > 0:
            correlations["n_new_vs_n_iter"] = float(
                np.corrcoef(n_new_arr, n_iter_arr)[0, 1]
            )
        if np.std(n_new_arr) > 0 and np.std(init_res_arr) > 0:
            correlations["n_new_vs_initial_residual"] = float(
                np.corrcoef(n_new_arr, init_res_arr)[0, 1]
            )

    if verbose and comm.rank == 0:
        print(f"\n  Step-by-Step Activation Load ({len(times)} steps)")
        print(f"\n    {'step':>4s}  {'t_val':>10s}  {'n_new':>5s}  {'n_act':>5s}  "
              f"{'||F_init||':>10s}  {'iters':>5s}  {'status':>6s}  {'|u|_max':>10s}")
        print(f"    {'-'*4}  {'-'*10}  {'-'*5}  {'-'*5}  "
              f"{'-'*10}  {'-'*5}  {'-'*6}  {'-'*10}")
        for r in records:
            status = "CONV" if r["converged"] else "FAIL"
            print(f"    {r['step']:4d}  {r['t_val']:10.2f}  "
                  f"{r['n_newly_active']:5d}  {r['n_total_active']:5d}  "
                  f"{r['initial_residual']:10.2e}  {r['n_iter']:5d}  "
                  f"{status:>6s}  {r['max_displacement']:10.2e}")

        if correlations:
            print(f"\n    Correlations:")
            for key, val in correlations.items():
                strength = "STRONG" if abs(val) > 0.5 else "weak"
                print(f"      {key}: {val:+.3f}  ({strength})")
            print(f"    Note: n_new_vs_n_iter includes path-dependent effects.")
            print(f"    n_new_vs_initial_residual is the cleaner diagnostic.")

    return {
        "steps": records,
        "n_steps": len(times),
        "correlations": correlations,
    }
