"""Phase 6: Residual decomposition diagnostics.

Decomposes the residual F(u) into its 10 constituent weak-form terms
to identify which terms dominate the initial imbalance at each time step.

Tests:
    6.1  Term-by-term residual decomposition over time.
    6.2  Gravity loading consistency verification.
"""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import create_vector
from mpi4py import MPI

from diagnostics.setup import prepare_state_at_time
from diagnostics.taylor_test import _assemble_residual_into, build_individual_form_terms
from diagnostics.property_conditioning import (
    _build_dg0_projection_tools,
    _generate_full_time_grid,
    _run_substepped_perzyna,
)
from solver.newton import get_inactive_dofs, solve_newton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_local(msh):
    """Number of locally-owned cells (excludes ghosts)."""
    return msh.topology.index_map(msh.topology.dim).size_local


def _assemble_term_norm(term_F_form, inactive_dofs):
    """Assemble a single form term's residual and return its L2 norm."""
    b = create_vector(term_F_form)
    _assemble_residual_into(b, term_F_form)
    if len(inactive_dofs) > 0:
        b.array[inactive_dofs] = 0.0
    norm = b.norm()
    b.destroy()
    return norm


def _decompose_residual(terms, inactive_dofs):
    """Assemble all term norms.  Returns list of (label, norm) tuples."""
    norms = []
    for t in terms:
        if t.get("jit_failed"):
            norms.append((t["label"], 0.0))
        else:
            norms.append((t["label"], _assemble_term_norm(t["F_form"], inactive_dofs)))
    return norms


def _assemble_total_norm(state, inactive_dofs):
    """Assemble total ||F(u)|| from the combined form."""
    b = create_vector(state.F_form)
    _assemble_residual_into(b, state.F_form)
    if len(inactive_dofs) > 0:
        b.array[inactive_dofs] = 0.0
    norm = b.norm()
    b.destroy()
    return norm


def _print_decomposition(norms, total_norm, label=""):
    """Print a single decomposition table."""
    print(f"    {'Term':<24s}  {'||F_i||':>11s}  {'%total':>7s}")
    print(f"    {'-'*24}  {'-'*11}  {'-'*7}")
    sum_norms = 0.0
    for name, norm in sorted(norms, key=lambda x: -x[1]):
        pct = 100.0 * norm / max(total_norm, 1e-30)
        print(f"    {name:<24s}  {norm:11.4e}  {pct:6.1f}%")
        sum_norms += norm
    print(f"    {'Total (assembled)':<24s}  {total_norm:11.4e}")
    print(f"    {'Sum of term norms':<24s}  {sum_norms:11.4e}")


# ---------------------------------------------------------------------------
# Test 6.1: Residual decomposition over time
# ---------------------------------------------------------------------------

def decompose_residual_over_time(
    state, times=None, n_steps=10, g_factor=50, verbose=True,
):
    """Decompose the initial residual into 10 terms at each time step.

    Steps through time with amplified gravity and VP updates (same
    approach as Test 5.5).  At each step, decomposes F(u_prev) into
    10 term norms BEFORE Newton solves.

    Args:
        state: DiagnosticState.
        times: Optional explicit time array.
        n_steps: Number of time points if ``times`` is None.
        g_factor: Gravity amplification factor.
        verbose: Print decomposition tables on rank 0.

    Returns:
        dict with per-step decomposition records.
    """
    comm = state.comm
    if times is None:
        times = _generate_full_time_grid(state, n_steps)

    msh = state.msh
    n_loc = _n_local(msh)
    bt = state.birth_times_dolfinx

    # Build individual form terms.
    if comm.rank == 0 and verbose:
        print("    Building individual form terms...", flush=True)
    terms = build_individual_form_terms(state)

    # Build DG0 projection tools for VP update.
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

        # --- BEFORE Newton: decompose F(u_prev) ---
        before_norms = _decompose_residual(terms, inactive_owned)
        total_before = _assemble_total_norm(state, inactive_owned)

        if comm.rank == 0 and verbose:
            print(f"\n  Step {i}: t={t_val:.2f}  dt={dt:.4f}  "
                  f"n_active={n_active}")
            print(f"    BEFORE Newton:")
            _print_decomposition(before_norms, total_before)

        # --- Solve Newton ---
        n_iter, converged, msg = solve_newton(
            state.u, state.V, state.msh, state.F_form, state.J_form,
            t_val=t_val,
            birth_times=bt,
            cell_to_dofs=state.cell_to_dofs,
            max_iter=50,
            debug=False,
        )

        if comm.rank == 0 and verbose:
            status = "CONV" if converged else "FAIL"
            print(f"    Newton: {status} in {n_iter} iters ({msg})")

        # Zero inactive DOFs after Newton.
        if inactive_owned.size > 0:
            state.u.x.array[inactive_owned] = 0.0
        state.u.x.scatter_forward()

        # --- AFTER Newton: total converged residual ---
        total_after = _assemble_total_norm(state, inactive_owned)
        if comm.rank == 0 and verbose:
            print(f"    AFTER Newton: ||F(u_conv)|| = {total_after:.4e}")

        # --- VP update for next step ---
        n_sub = 0
        if converged:
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

        # Store record.
        record = {
            "step": i,
            "t_val": t_val,
            "dt": dt,
            "n_active": n_active,
            "converged": bool(converged),
            "n_iter": int(n_iter),
            "n_sub": n_sub,
            "total_before": total_before,
            "total_after": total_after,
            "before_norms": {name: norm for name, norm in before_norms},
        }
        records.append(record)

    # Restore gravity.
    state.materials.g_const.value = g_base

    # Summary: dominant term at each step.
    dominant_terms = {}
    for r in records:
        if r["total_before"] < 1e-12:
            continue
        dom = max(r["before_norms"].items(), key=lambda x: x[1])
        dominant_terms[dom[0]] = dominant_terms.get(dom[0], 0) + 1

    result = {
        "records": records,
        "g_factor": g_factor,
        "dominant_term_counts": dominant_terms,
    }

    if verbose and comm.rank == 0:
        _print_decomposition_summary(result)

    return result


def _print_decomposition_summary(result):
    """Print summary of residual decomposition."""
    records = result["records"]
    print(f"\n  Residual Decomposition Summary (g_factor={result['g_factor']})")
    print(f"    Dominant term at each step:")
    for name, count in sorted(
        result["dominant_term_counts"].items(), key=lambda x: -x[1],
    ):
        print(f"      {name:<24s}  {count} steps")

    # Compact table.
    print(f"\n    {'step':>4s}  {'t_val':>8s}  {'n_act':>6s}  {'iters':>5s}  "
          f"{'||F_before||':>12s}  {'||F_after||':>12s}  "
          f"{'dominant_term':<24s}  {'dom_%':>6s}")
    print(f"    {'-'*4}  {'-'*8}  {'-'*6}  {'-'*5}  "
          f"{'-'*12}  {'-'*12}  {'-'*24}  {'-'*6}")
    for r in records:
        dom_name, dom_norm = max(r["before_norms"].items(), key=lambda x: x[1])
        dom_pct = 100.0 * dom_norm / max(r["total_before"], 1e-30)
        print(f"    {r['step']:4d}  {r['t_val']:8.2f}  {r['n_active']:6d}  "
              f"{r['n_iter']:5d}  {r['total_before']:12.4e}  "
              f"{r['total_after']:12.4e}  {dom_name:<24s}  {dom_pct:5.1f}%")


# ---------------------------------------------------------------------------
# Test 6.2: Gravity loading consistency
# ---------------------------------------------------------------------------

def verify_gravity_loading(state, verbose=True):
    """Verify gravity body force assembly and unit conversion.

    Two-level verification:

    1. Compute exact mesh volume and analytical gravity force.
    2. Compare with assembled gravity term norm.
    3. Independent unit conversion check.

    Args:
        state: DiagnosticState.
        verbose: Print results on rank 0.

    Returns:
        dict with volume, forces, ratios.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx
    t_all_active = float(bt.max()) * 1.1

    # Set state to all active, u=0.
    prepare_state_at_time(state, t_all_active, u_mode="zero")

    # --- Exact mesh volume ---
    dx = ufl.Measure("dx", domain=state.msh)
    one = fem.Constant(state.msh, 1.0)
    vol_form = fem.form(one * dx)
    V_local = fem.assemble_scalar(vol_form)
    V_exact = comm.allreduce(float(V_local), op=MPI.SUM)

    # --- Material constants ---
    rho_ns2_mm4 = state.materials.rho_ns2_mm4
    g_val = state.materials.g_val
    rho_kg_m3 = state.materials.rho_kg_m3

    # --- Independent unit conversion check ---
    rho_check = rho_kg_m3 * 1e-12  # kg/m³ → N·s²/mm⁴
    body_force_density = rho_check * g_val  # N/mm³
    total_gravity_force = body_force_density * V_exact  # N

    # --- Assembled gravity term ---
    if comm.rank == 0 and verbose:
        print("    Building individual form terms for gravity extraction...",
              flush=True)
    terms = build_individual_form_terms(state)

    # Find the gravity term (index 9).
    gravity_term = None
    for t in terms:
        if t["label"] == "gravity":
            gravity_term = t
            break

    inactive_owned = state.inactive_dofs
    if inactive_owned is None:
        inactive_owned = np.array([], dtype=np.int32)

    gravity_norm = 0.0
    if gravity_term is not None and not gravity_term.get("jit_failed"):
        gravity_norm = _assemble_term_norm(gravity_term["F_form"], inactive_owned)

    # Also assemble full F(0).
    total_F0 = _assemble_total_norm(state, inactive_owned)

    # --- Ratios ---
    # Ratio of assembled gravity norm to analytical force.
    ratio_grav = gravity_norm / max(total_gravity_force, 1e-30)
    # Ratio of total F(0) to gravity-only term (should be ≈1 at u=0).
    ratio_total_grav = total_F0 / max(gravity_norm, 1e-30)
    # Unit conversion match.
    unit_match = abs(rho_ns2_mm4 - rho_check) / max(abs(rho_check), 1e-30)

    result = {
        "V_exact_mm3": V_exact,
        "rho_kg_m3": rho_kg_m3,
        "rho_ns2_mm4": rho_ns2_mm4,
        "rho_check_ns2_mm4": rho_check,
        "unit_conversion_relerr": unit_match,
        "g_val_mm_s2": g_val,
        "body_force_density_N_mm3": body_force_density,
        "total_gravity_force_N": total_gravity_force,
        "gravity_term_norm": gravity_norm,
        "total_F0_norm": total_F0,
        "ratio_grav_norm_to_force": ratio_grav,
        "ratio_F0_to_gravity_term": ratio_total_grav,
        "passed": unit_match < 1e-10 and 0.001 < ratio_grav < 1000.0,
    }

    if verbose and comm.rank == 0:
        _print_gravity_verification(result)

    return result


def _print_gravity_verification(result):
    """Print gravity loading verification."""
    print(f"\n  Gravity Loading Verification")
    print(f"    Mesh volume (exact):       {result['V_exact_mm3']:.2f} mm³")
    print(f"    Density:                   {result['rho_kg_m3']:.0f} kg/m³")
    print(f"      → rho (config):          {result['rho_ns2_mm4']:.6e} N·s²/mm⁴")
    print(f"      → rho (independent):     {result['rho_check_ns2_mm4']:.6e} N·s²/mm⁴")
    print(f"      → unit conversion err:   {result['unit_conversion_relerr']:.2e}")
    print(f"    Gravity:                   {result['g_val_mm_s2']:.0f} mm/s²")
    print(f"    Body force density:        {result['body_force_density_N_mm3']:.6e} N/mm³")
    print(f"    Total gravity force:       {result['total_gravity_force_N']:.6e} N")
    print(f"    ||F_gravity|| (assembled): {result['gravity_term_norm']:.6e}")
    print(f"    ||F(u=0)|| (total):        {result['total_F0_norm']:.6e}")
    print(f"    ||F_grav|| / F_analytical: {result['ratio_grav_norm_to_force']:.4f}")
    print(f"    ||F(0)|| / ||F_grav||:     {result['ratio_F0_to_gravity_term']:.4f}")
    status = "PASS" if result["passed"] else "FAIL"
    print(f"    Status: {status}")
