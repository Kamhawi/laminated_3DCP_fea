"""Phase 14: Selected code-level checks.

Tests:
    14.1  Alpha_facet mode comparison (min vs max vs avg).
    14.4  eps_vp / epsilon(u) freezing ratio.
    14.5  Quadrature degree inspection.
"""

import numpy as np
from mpi4py import MPI

from diagnostics.setup import prepare_state_at_time
from diagnostics.property_conditioning import (
    _build_dg0_projection_tools,
    _generate_full_time_grid,
    _run_substepped_perzyna,
    _n_local,
)
from diagnostics.taylor_test import build_individual_form_terms
from physics.weak_form import build_evp_cohesive_weak_form
from solver.newton import get_inactive_dofs, solve_newton


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
EPSVP_RATIO_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Test 14.5: Quadrature degree inspection
# ---------------------------------------------------------------------------

def check_quadrature_degrees(state, verbose=True):
    """Extract and report quadrature degrees from compiled forms.

    Inspects both the combined residual form and per-term forms to
    identify potentially elevated quadrature degrees from nonlinear
    UFL operators (exp, max_value, etc.).

    Args:
        state: DiagnosticState.
        verbose: Print results on rank 0.

    Returns:
        dict with per-integral quadrature degree info.
    """
    comm = state.comm

    result = {
        "combined": _extract_form_degrees(state.F_form),
        "per_term": [],
    }

    if comm.rank == 0 and verbose:
        print("    Building individual form terms for quadrature inspection...",
              flush=True)

    terms = build_individual_form_terms(state)

    for t in terms:
        entry = {"label": t["label"], "jit_failed": t.get("jit_failed", False)}
        if not t.get("jit_failed"):
            entry["degrees"] = _extract_form_degrees(t["F_form"])
        else:
            entry["degrees"] = {}
        result["per_term"].append(entry)

    if verbose and comm.rank == 0:
        _print_quadrature_degrees(result)

    return result


def _extract_form_degrees(compiled_form):
    """Extract quadrature degrees from a compiled DOLFINx form.

    Uses the ufcx_form interface to read integral metadata.
    Wraps in try/except for API compatibility across DOLFINx versions.
    """
    degrees = {}
    try:
        ufcx = compiled_form.ufcx_form
        integral_names = {0: "cell", 1: "exterior_facet", 2: "interior_facet"}

        for int_type, int_name in integral_names.items():
            ids = ufcx.integral_ids(int_type)
            if len(ids) > 0:
                # Try to get quadrature degree from form integral metadata.
                integrals = compiled_form.integrals(int_type)
                for i, subdomain_id in enumerate(ids):
                    key = f"{int_name}"
                    if len(ids) > 1:
                        key = f"{int_name}({subdomain_id})"
                    # Access metadata if available.
                    try:
                        if i < len(integrals):
                            md = integrals[i].metadata
                            if md and "quadrature_degree" in md:
                                degrees[key] = md["quadrature_degree"]
                                continue
                    except (AttributeError, TypeError):
                        pass
                    degrees[key] = "present"
    except AttributeError:
        # Fallback: try form_integral_types.
        try:
            for integral in compiled_form.integrals:
                itype = integral.integral_type
                degrees[str(itype)] = "present"
        except Exception:
            degrees["error"] = "Could not extract quadrature info"

    # Also try to get estimated degree from UFL form data if available.
    try:
        form_data = compiled_form.form_data
        if form_data is not None:
            for integral_data in form_data.integral_data:
                key = f"ufl_{integral_data.integral_type}({integral_data.subdomain_id})"
                if hasattr(integral_data, "metadata"):
                    md = integral_data.metadata
                    if "quadrature_degree" in md:
                        degrees[key] = md["quadrature_degree"]
    except (AttributeError, TypeError):
        pass

    return degrees


def _print_quadrature_degrees(result):
    """Print quadrature degree inspection results."""
    print("\n  Quadrature Degree Inspection")

    print("\n    Combined residual form:")
    combined = result["combined"]
    if combined:
        for key, val in sorted(combined.items()):
            print(f"      {key:<30s}  degree={val}")
    else:
        print("      (no quadrature metadata extracted)")

    print("\n    Per-term forms:")
    for entry in result["per_term"]:
        label = entry["label"]
        if entry["jit_failed"]:
            print(f"      {label:<24s}  JIT FAILED")
            continue
        degs = entry["degrees"]
        if degs:
            parts = [f"{k}={v}" for k, v in sorted(degs.items())]
            print(f"      {label:<24s}  {', '.join(parts)}")
        else:
            print(f"      {label:<24s}  (no metadata)")


# ---------------------------------------------------------------------------
# Test 14.4: eps_vp / epsilon(u) freezing ratio
# ---------------------------------------------------------------------------

def check_epsvp_strain_ratio(
    state, times=None, n_steps=10, g_factor=50, verbose=True,
):
    """Monitor ||eps_vp|| / ||epsilon(u)|| ratio at each time step.

    Steps through time with amplified gravity. At each converged step,
    projects strain to DG0, runs VP update, then computes per-cell
    Frobenius norm ratio.

    Args:
        state: DiagnosticState.
        times: Optional explicit time array.
        n_steps: Number of time sample points.
        g_factor: Gravity amplification.
        verbose: Print table on rank 0.

    Returns:
        dict with per-step ratio records.
    """
    comm = state.comm
    if times is None:
        times = _generate_full_time_grid(state, n_steps)

    msh = state.msh
    n_loc = _n_local(msh)
    bt = state.birth_times_dolfinx

    project_strain = _build_dg0_projection_tools(state)

    # Amplify gravity.
    g_base = state.materials.g_val
    state.materials.g_const.value = g_base * g_factor

    # Initialize at first time point.
    prepare_state_at_time(state, float(times[0]), u_mode="zero")

    records = []

    for i in range(len(times) - 1):
        t_val = float(times[i + 1])
        dt = float(times[i + 1] - times[i])

        # Update materials to new time.
        state.materials.t_current.value = t_val
        state.materials.update_properties(t_val)

        active = bt[:n_loc] <= t_val
        n_active_local = int(np.sum(active))
        n_active = comm.allreduce(n_active_local, op=MPI.SUM)

        inactive_dofs = get_inactive_dofs(t_val, bt, state.cell_to_dofs)
        num_owned = len(state.u.x.array)
        inactive_owned = inactive_dofs[inactive_dofs < num_owned].astype(
            np.int32, copy=False,
        )

        # Solve Newton.
        n_iter, converged, msg = solve_newton(
            state.u, state.V, state.msh, state.F_form, state.J_form,
            t_val=t_val,
            birth_times=bt,
            cell_to_dofs=state.cell_to_dofs,
            max_iter=50,
            debug=False,
        )

        if not converged:
            if comm.rank == 0 and verbose:
                print(f"    t={t_val:.2f}: Newton did not converge ({msg}), skipping")
            records.append({
                "step": i, "t_val": t_val, "dt": dt,
                "converged": False, "n_iter": int(n_iter), "n_active": n_active,
                "max_ratio": 0.0, "mean_ratio": 0.0, "ratio_flag": False,
            })
            continue

        # Zero inactive DOFs after Newton.
        if inactive_owned.size > 0:
            state.u.x.array[inactive_owned] = 0.0
        state.u.x.scatter_forward()

        # Project strain to DG0.
        project_strain()
        strain_arr = state.materials.strain.x.array.reshape((-1, 3, 3))
        eps_vp_prev = state.materials.eps_vp.x.array.reshape((-1, 3, 3)).copy()

        active_full = np.zeros(strain_arr.shape[0], dtype=bool)
        active_full[:n_loc] = active

        # Compute n_sub and run VP update.
        E_active = state.materials.e_arr[active_full]
        eta_active = state.materials.eta_arr[active_full]
        eta_over_E = eta_active / np.maximum(E_active, 1e-12)
        min_eoe_local = (
            float(np.min(eta_over_E)) if len(eta_over_E) > 0 else np.inf
        )
        min_eoe = comm.allreduce(min_eoe_local, op=MPI.MIN)
        dt_limit = max(min_eoe, 1e-12)
        n_sub = max(1, int(np.ceil(dt / dt_limit)))

        eps_vp_new, _, _, _ = _run_substepped_perzyna(
            strain_arr, eps_vp_prev, state.materials, active_full, dt, n_sub,
        )

        # Compute per-cell ratio: ||eps_vp|| / ||epsilon(u)||.
        evp_active = eps_vp_new[:n_loc][active]
        strain_active = strain_arr[:n_loc][active]

        evp_norms = np.sqrt(np.sum(evp_active ** 2, axis=(1, 2)))
        strain_norms = np.sqrt(np.sum(strain_active ** 2, axis=(1, 2)))
        ratios = evp_norms / np.maximum(strain_norms, 1e-12)

        max_ratio_l = float(np.max(ratios)) if len(ratios) > 0 else 0.0
        mean_ratio_l = float(np.mean(ratios)) if len(ratios) > 0 else 0.0

        max_ratio = comm.allreduce(max_ratio_l, op=MPI.MAX)
        mean_ratio = (
            comm.allreduce(mean_ratio_l * n_active_local, op=MPI.SUM)
            / max(n_active, 1)
        )

        flag = max_ratio > EPSVP_RATIO_THRESHOLD

        records.append({
            "step": i,
            "t_val": t_val,
            "dt": dt,
            "converged": True,
            "n_iter": int(n_iter),
            "n_active": n_active,
            "n_sub": n_sub,
            "max_ratio": max_ratio,
            "mean_ratio": mean_ratio,
            "ratio_flag": flag,
        })

        # Write eps_vp_new back for next step.
        state.materials.eps_vp.x.array[:] = eps_vp_new.ravel()
        state.materials.eps_vp.x.scatter_forward()

        if comm.rank == 0 and verbose:
            fs = "WARN" if flag else "ok"
            print(f"    Step {i}: t={t_val:.2f}  n_act={n_active}  "
                  f"iters={n_iter}  n_sub={n_sub}  "
                  f"max_ratio={max_ratio:.4e}  mean_ratio={mean_ratio:.4e}  {fs}")

    # Restore gravity.
    state.materials.g_const.value = g_base

    worst_max = max((r["max_ratio"] for r in records), default=0.0)
    result = {
        "records": records,
        "g_factor": g_factor,
        "worst_max_ratio": worst_max,
        "any_ratio_flag": any(r["ratio_flag"] for r in records),
    }

    if verbose and comm.rank == 0:
        _print_epsvp_ratio(result)

    return result


def _print_epsvp_ratio(result):
    """Print eps_vp/strain ratio summary."""
    records = result["records"]
    print(f"\n  eps_vp / epsilon(u) Ratio Summary (g_factor={result['g_factor']})")
    print(f"    Threshold: max_ratio > {EPSVP_RATIO_THRESHOLD}")

    print(f"\n    {'step':>4s}  {'t_val':>8s}  {'n_act':>6s}  {'iters':>5s}  "
          f"{'n_sub':>6s}  {'max_ratio':>10s}  {'mean_ratio':>10s}  {'flag':>5s}")
    print(f"    {'-'*4}  {'-'*8}  {'-'*6}  {'-'*5}  "
          f"{'-'*6}  {'-'*10}  {'-'*10}  {'-'*5}")

    for r in records:
        if not r["converged"]:
            print(f"    {r['step']:4d}  {r['t_val']:8.2f}  {r['n_active']:6d}  "
                  f"{'—':>5s}  {'—':>6s}  {'—':>10s}  {'—':>10s}  {'SKIP':>5s}")
            continue
        flag = "WARN" if r["ratio_flag"] else "ok"
        print(f"    {r['step']:4d}  {r['t_val']:8.2f}  {r['n_active']:6d}  "
              f"{r['n_iter']:5d}  {r.get('n_sub', 0):6d}  "
              f"{r['max_ratio']:10.4e}  {r['mean_ratio']:10.4e}  {flag:>5s}")


# ---------------------------------------------------------------------------
# Test 14.1: Alpha_facet mode comparison
# ---------------------------------------------------------------------------

def compare_alpha_facet_modes(
    state, times=None, n_steps=5, g_factor=50, verbose=True,
):
    """Compare Newton convergence for min/max/avg alpha_facet modes.

    Rebuilds forms with each mode up front (all sharing the same
    materials object, including damage_max). Steps through time with
    amplified gravity. At each step, tests all three modes from the
    same u_prev, then advances using the "min" result.

    Args:
        state: DiagnosticState.
        times: Optional explicit time array.
        n_steps: Number of time sample points.
        g_factor: Gravity amplification.
        verbose: Print table on rank 0.

    Returns:
        dict with per-mode, per-step convergence records.
    """
    comm = state.comm
    modes = ["min", "max", "avg"]

    if times is None:
        times = _generate_full_time_grid(state, n_steps)

    msh = state.msh
    n_loc = _n_local(msh)
    bt = state.birth_times_dolfinx

    # --- Rebuild forms for each mode ---
    if comm.rank == 0 and verbose:
        print("    Building forms for each alpha_facet mode...", flush=True)

    mode_forms = {}
    for mode in modes:
        F_form, J_form = build_evp_cohesive_weak_form(
            msh, state.V, state.materials, state.birth_time_func,
            state.interior_facet_tags, state.dirichlet_tags, state.cfg,
            alpha_facet_mode=mode,
        )
        mode_forms[mode] = (F_form, J_form)
        if comm.rank == 0 and verbose:
            print(f"      {mode}: forms built", flush=True)

    project_strain = _build_dg0_projection_tools(state)

    # Amplify gravity.
    g_base = state.materials.g_val
    state.materials.g_const.value = g_base * g_factor

    # Initialize at first time point.
    prepare_state_at_time(state, float(times[0]), u_mode="zero")

    records = {mode: [] for mode in modes}

    for i in range(len(times) - 1):
        t_val = float(times[i + 1])
        dt = float(times[i + 1] - times[i])

        # Update materials to new time.
        state.materials.t_current.value = t_val
        state.materials.update_properties(t_val)

        active = bt[:n_loc] <= t_val
        n_active_local = int(np.sum(active))
        n_active = comm.allreduce(n_active_local, op=MPI.SUM)

        inactive_dofs = get_inactive_dofs(t_val, bt, state.cell_to_dofs)
        num_owned = len(state.u.x.array)
        inactive_owned = inactive_dofs[inactive_dofs < num_owned].astype(
            np.int32, copy=False,
        )

        # Save u_prev for resetting between modes.
        u_prev = state.u.x.array.copy()

        for mode in modes:
            # Reset to u_prev.
            state.u.x.array[:] = u_prev
            state.u.x.scatter_forward()

            F_form, J_form = mode_forms[mode]

            # Assemble ||F(u_prev)|| using this mode's form.
            from dolfinx.fem.petsc import create_vector
            from diagnostics.taylor_test import _assemble_residual_into
            b = create_vector(F_form)
            _assemble_residual_into(b, F_form)
            if len(inactive_owned) > 0:
                b.array[inactive_owned] = 0.0
            F_before = b.norm()
            b.destroy()

            # Solve Newton with this mode's forms.
            n_iter, converged, msg = solve_newton(
                state.u, state.V, state.msh, F_form, J_form,
                t_val=t_val,
                birth_times=bt,
                cell_to_dofs=state.cell_to_dofs,
                max_iter=50,
                debug=False,
            )

            # Zero inactive DOFs.
            if inactive_owned.size > 0:
                state.u.x.array[inactive_owned] = 0.0
            state.u.x.scatter_forward()

            # Measure converged residual.
            b2 = create_vector(F_form)
            _assemble_residual_into(b2, F_form)
            if len(inactive_owned) > 0:
                b2.array[inactive_owned] = 0.0
            F_after = b2.norm()
            b2.destroy()

            records[mode].append({
                "step": i,
                "t_val": t_val,
                "dt": dt,
                "n_active": n_active,
                "n_iter": int(n_iter),
                "converged": bool(converged),
                "F_before": F_before,
                "F_after": F_after,
            })

            if comm.rank == 0 and verbose:
                status = "CONV" if converged else "FAIL"
                print(f"    Step {i}: t={t_val:.2f}  n_act={n_active}  "
                      f"mode={mode:<3s}  iters={n_iter:2d}  {status}  "
                      f"||F_before||={F_before:.4e}  ||F_after||={F_after:.4e}")

        # Advance state using "min" result.
        # Restore "min" converged displacement.
        state.u.x.array[:] = u_prev
        state.u.x.scatter_forward()

        # Re-solve with min forms to get the actual converged state.
        F_min, J_min = mode_forms["min"]
        solve_newton(
            state.u, state.V, state.msh, F_min, J_min,
            t_val=t_val,
            birth_times=bt,
            cell_to_dofs=state.cell_to_dofs,
            max_iter=50,
            debug=False,
        )
        if inactive_owned.size > 0:
            state.u.x.array[inactive_owned] = 0.0
        state.u.x.scatter_forward()

        # VP update for next step.
        project_strain()
        strain_arr = state.materials.strain.x.array.reshape((-1, 3, 3))
        eps_vp_prev = state.materials.eps_vp.x.array.reshape((-1, 3, 3)).copy()

        active_full = np.zeros(strain_arr.shape[0], dtype=bool)
        active_full[:n_loc] = active

        E_active = state.materials.e_arr[active_full]
        eta_active = state.materials.eta_arr[active_full]
        eta_over_E = eta_active / np.maximum(E_active, 1e-12)
        min_eoe_local = (
            float(np.min(eta_over_E)) if len(eta_over_E) > 0 else np.inf
        )
        min_eoe = comm.allreduce(min_eoe_local, op=MPI.MIN)
        dt_limit = max(min_eoe, 1e-12)
        n_sub = max(1, int(np.ceil(dt / dt_limit)))

        eps_vp_new, _, _, _ = _run_substepped_perzyna(
            strain_arr, eps_vp_prev, state.materials, active_full, dt, n_sub,
        )
        state.materials.eps_vp.x.array[:] = eps_vp_new.ravel()
        state.materials.eps_vp.x.scatter_forward()

    # Restore gravity.
    state.materials.g_const.value = g_base

    # Summary: total iterations per mode.
    summary = {}
    for mode in modes:
        total_iters = sum(r["n_iter"] for r in records[mode])
        n_converged = sum(1 for r in records[mode] if r["converged"])
        n_failed = sum(1 for r in records[mode] if not r["converged"])
        summary[mode] = {
            "total_iters": total_iters,
            "n_converged": n_converged,
            "n_failed": n_failed,
        }

    result = {
        "records": records,
        "summary": summary,
        "g_factor": g_factor,
        "modes": modes,
    }

    if verbose and comm.rank == 0:
        _print_alpha_facet_comparison(result)

    return result


def _print_alpha_facet_comparison(result):
    """Print alpha_facet mode comparison summary."""
    print(f"\n  Alpha_facet Mode Comparison Summary (g_factor={result['g_factor']})")
    modes = result["modes"]
    summary = result["summary"]

    print(f"\n    {'mode':<6s}  {'total_iters':>11s}  {'converged':>9s}  {'failed':>6s}")
    print(f"    {'-'*6}  {'-'*11}  {'-'*9}  {'-'*6}")
    for mode in modes:
        s = summary[mode]
        print(f"    {mode:<6s}  {s['total_iters']:11d}  {s['n_converged']:9d}  "
              f"{s['n_failed']:6d}")

    # Per-step comparison.
    records = result["records"]
    n_steps = len(records[modes[0]])
    print(f"\n    {'step':>4s}  {'t_val':>8s}  {'n_act':>6s}  ", end="")
    for mode in modes:
        print(f"{'iters_'+mode:>10s}  ", end="")
    print()
    print(f"    {'-'*4}  {'-'*8}  {'-'*6}  ", end="")
    for _ in modes:
        print(f"{'-'*10}  ", end="")
    print()

    for j in range(n_steps):
        r0 = records[modes[0]][j]
        print(f"    {r0['step']:4d}  {r0['t_val']:8.2f}  {r0['n_active']:6d}  ", end="")
        for mode in modes:
            r = records[mode][j]
            if r["converged"]:
                print(f"{r['n_iter']:10d}  ", end="")
            else:
                print(f"{'FAIL':>10s}  ", end="")
        print()
