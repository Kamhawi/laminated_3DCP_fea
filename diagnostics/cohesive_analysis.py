"""Cohesive law convexity and stability analysis (Phase 4 diagnostics).

Provides tests for:
  4.1  Traction-separation tangent stiffness and snap-back ratio
  4.2  Damage irreversibility check (per-facet tracking across steps)
  4.3  Contact penalty adequacy (interpenetration check)
  4.4  Mixed-mode Hessian convexity check

Tests 4.1 and 4.4 are pure analytical (NumPy).  Tests 4.2 and 4.3 share
a mini time-stepping loop with amplified gravity.
"""

import numpy as np
from mpi4py import MPI

from diagnostics.stiffness_hierarchy import compute_cell_diameters
from diagnostics.tsl_analysis import compute_tsl_curve
from diagnostics.validate_interface_opening import _extract_interface_jumps
from materials.time_models import (
    compute_bond_quality,
    compute_poisson_ratio,
    compute_shear_modulus_mpa,
    compute_young_modulus_mpa,
    compute_yield_stress_mpa,
    pa_to_mpa,
)
from solver.kinematics import (
    init_newly_activated_displacement,
    zero_inactive_cells,
)
from solver.newton import solve_newton


# ---------------------------------------------------------------------------
# Test 4.1: Tangent stiffness analysis
# ---------------------------------------------------------------------------

def _compute_analytical_tangent(K_n, G_Ic_eff, r_s, n_points=500):
    """Compute the analytical tangent dT_n/d(delta_n) for mode-I TSL.

    Returns arrays and scalar metrics including snap-back ratio.
    """
    K_n = float(K_n)
    G_Ic_eff = max(float(G_Ic_eff), 1e-30)
    r_s = float(r_s)

    alpha = K_n / (2.0 * G_Ic_eff)
    delta_peak_approx = np.sqrt(G_Ic_eff / max(K_n, 1e-30))
    delta_max = 5.0 * delta_peak_approx
    delta = np.linspace(0, delta_max, n_points)

    xi = alpha * delta ** 2
    exp_neg_xi = np.exp(-xi)

    # Traction.
    traction = K_n * (r_s + (1.0 - r_s) * exp_neg_xi) * delta

    # Analytical tangent: dT/d(delta).
    tangent = K_n * (r_s + (1.0 - r_s) * exp_neg_xi * (1.0 - 2.0 * alpha * delta ** 2))

    i_peak = np.argmax(traction)
    peak_traction = float(traction[i_peak])
    delta_at_peak = float(delta[i_peak])

    # Softening branch: everything past the peak.
    min_tangent = float(np.min(tangent[i_peak:])) if i_peak < len(tangent) - 1 else 0.0
    tangent_at_peak = float(tangent[i_peak])

    # Snap-back ratio: |min_tangent| / K_n.
    # If > 1.0, local tangent loses positive-definiteness.
    initial_stiffness = K_n * (r_s + (1.0 - r_s))  # = K_n at delta=0
    snap_back_ratio = abs(min_tangent) / max(initial_stiffness, 1e-30)

    # Damage at delta_c_99.
    damage = 1.0 - exp_neg_xi
    mask_99 = damage >= 0.99
    delta_c_99 = float(delta[mask_99][0]) if np.any(mask_99) else float(delta[-1])

    return {
        "delta": delta,
        "traction": traction,
        "tangent": tangent,
        "K_n": K_n,
        "G_Ic_eff": G_Ic_eff,
        "r_s": r_s,
        "peak_traction": peak_traction,
        "delta_at_peak": delta_at_peak,
        "min_tangent": min_tangent,
        "tangent_at_peak": tangent_at_peak,
        "snap_back_ratio": snap_back_ratio,
        "delta_c_99": delta_c_99,
    }


def run_tangent_stiffness_analysis(cfg, ages, h_values, t_open=0.0,
                                   verbose=True):
    """Analyse mode-I TSL tangent stiffness and snap-back ratio.

    Pure analytical (NumPy).  For each (age, h) combination, computes
    the analytical tangent dT_n/d(delta_n), finds the minimum (most
    negative) slope on the softening branch, and reports the snap-back
    ratio |min_tangent| / K_n.

    Returns:
        dict with per-(age, h) records and global summary.
    """
    material_cfg = cfg["material"]
    hardening_cfg = cfg["hardening"]
    interface_cfg = cfg["interface"]

    tau_0_pa = float(material_cfg["tau_0"])
    a_thix_pa_s = float(material_cfg["A_thix"])
    tau_0_mpa = float(pa_to_mpa(tau_0_pa))
    a_thix_mpa_s = float(pa_to_mpa(a_thix_pa_s))
    gamma_c = float(material_cfg["gamma_c"])

    t_set = float(hardening_cfg["t_set"])
    e_inf = float(hardening_cfg["E_inf"])
    nu_fresh = float(hardening_cfg["nu_fresh"])
    nu_hard = float(hardening_cfg["nu_hard"])
    n_h = float(hardening_cfg.get("n_h", 0.7))

    g_i_c = float(interface_cfg["G_Ic"])
    nozzle_pressure_pa = float(interface_cfg["nozzle_pressure"])
    k_min_mult = float(interface_cfg.get("K_min_mult", 0.1))
    r_s = float(interface_cfg.get("residual_stiffness_ratio", 0.05))

    ages = np.asarray(ages, dtype=float)
    h_values = np.asarray(h_values, dtype=float)

    beta = float(compute_bond_quality(
        t_open, tau_0_pa, a_thix_pa_s, nozzle_pressure_pa,
    ))
    beta = np.clip(beta, 1e-8, 1.0)
    G_Ic_eff = beta * g_i_c

    rows = []
    for age in ages:
        age_arr = np.array([age])
        tau_y = float(compute_yield_stress_mpa(
            age_arr, tau_0_mpa, a_thix_mpa_s, t_set, n_h,
        )[0])
        sigma_y = np.sqrt(3.0) * tau_y
        nu = float(compute_poisson_ratio(age_arr, nu_fresh, nu_hard, t_set)[0])
        G = float(compute_shear_modulus_mpa(
            np.array([tau_y]), gamma_c, e_inf, nu_hard,
        )[0])
        E = float(compute_young_modulus_mpa(np.array([G]), np.array([nu]))[0])

        K_n_phys = beta * sigma_y ** 2 / (2.0 * g_i_c)

        for h in h_values:
            K_min = k_min_mult * E / h
            K_n_actual = max(K_n_phys, K_min)

            result = _compute_analytical_tangent(K_n_actual, G_Ic_eff, r_s)

            rows.append({
                "age": float(age),
                "h": float(h),
                "E": E,
                "sigma_y": sigma_y,
                "K_n_phys": K_n_phys,
                "K_min": K_min,
                "K_n_actual": K_n_actual,
                "peak_traction": result["peak_traction"],
                "delta_at_peak": result["delta_at_peak"],
                "min_tangent": result["min_tangent"],
                "tangent_at_peak": result["tangent_at_peak"],
                "snap_back_ratio": result["snap_back_ratio"],
                "delta_c_99": result["delta_c_99"],
            })

    max_snap_back = max(r["snap_back_ratio"] for r in rows) if rows else 0.0
    any_snap_back = max_snap_back > 1.0

    out = {
        "beta_bond": beta,
        "t_open": float(t_open),
        "G_Ic_eff": G_Ic_eff,
        "r_s": r_s,
        "rows": rows,
        "max_snap_back_ratio": max_snap_back,
        "snap_back_detected": any_snap_back,
    }

    if verbose:
        _print_tangent_analysis(out)

    return out


def _print_tangent_analysis(result):
    """Print tangent stiffness analysis table."""
    print(f"\n  Tangent Stiffness Analysis (mode-I)")
    print(f"    beta = {result['beta_bond']:.4f},  "
          f"G_Ic_eff = {result['G_Ic_eff']:.4e} N/mm,  "
          f"r_s = {result['r_s']}")

    rows = result["rows"]
    if not rows:
        return

    print(f"\n    {'age':>6s}  {'h':>6s}  {'K_n':>10s}  "
          f"{'T_peak':>10s}  {'d_peak':>10s}  "
          f"{'min_tang':>10s}  {'snap_back':>10s}  {'flag':>5s}")
    print(f"    {'-'*6}  {'-'*6}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*5}")

    for r in rows:
        flag = "WARN" if r["snap_back_ratio"] > 1.0 else "ok"
        print(f"    {r['age']:6.0f}  {r['h']:6.1f}  "
              f"{r['K_n_actual']:10.4e}  {r['peak_traction']:10.4e}  "
              f"{r['delta_at_peak']:10.4e}  {r['min_tangent']:10.4e}  "
              f"{r['snap_back_ratio']:10.4f}  {flag:>5s}")

    sb = result["max_snap_back_ratio"]
    status = "NO SNAP-BACK" if sb < 1.0 else "SNAP-BACK DETECTED"
    print(f"\n    Max snap-back ratio: {sb:.4f}  -> {status}")
    if sb < 1.0:
        print(f"    Newton can handle softening without arc-length control.")


# ---------------------------------------------------------------------------
# Test 4.4: Mixed-mode convexity check
# ---------------------------------------------------------------------------

def run_mixed_mode_convexity_check(cfg, ages, h_values, t_open=0.0,
                                   n_grid=100, verbose=True):
    """Check convexity of the mixed-mode cohesive traction potential.

    Computes the 2x2 Hessian of T w.r.t. (delta_n, delta_t) and checks
    eigenvalues on a 2D grid.  Pure analytical (NumPy).

    Returns:
        dict with per-(age, h) convexity metrics.
    """
    material_cfg = cfg["material"]
    hardening_cfg = cfg["hardening"]
    interface_cfg = cfg["interface"]

    tau_0_pa = float(material_cfg["tau_0"])
    a_thix_pa_s = float(material_cfg["A_thix"])
    tau_0_mpa = float(pa_to_mpa(tau_0_pa))
    a_thix_mpa_s = float(pa_to_mpa(a_thix_pa_s))
    gamma_c = float(material_cfg["gamma_c"])

    t_set = float(hardening_cfg["t_set"])
    e_inf = float(hardening_cfg["E_inf"])
    nu_fresh = float(hardening_cfg["nu_fresh"])
    nu_hard = float(hardening_cfg["nu_hard"])
    n_h = float(hardening_cfg.get("n_h", 0.7))

    g_i_c = float(interface_cfg["G_Ic"])
    g_ii_c = float(interface_cfg["G_IIc"])
    nozzle_pressure_pa = float(interface_cfg["nozzle_pressure"])
    k_min_mult = float(interface_cfg.get("K_min_mult", 0.1))
    r_s = float(interface_cfg.get("residual_stiffness_ratio", 0.05))

    ages = np.asarray(ages, dtype=float)
    h_values = np.asarray(h_values, dtype=float)

    beta = float(compute_bond_quality(
        t_open, tau_0_pa, a_thix_pa_s, nozzle_pressure_pa,
    ))
    beta = np.clip(beta, 1e-8, 1.0)
    G_Ic_eff = beta * g_i_c
    G_IIc_eff = beta * g_ii_c

    rows = []
    for age in ages:
        age_arr = np.array([age])
        tau_y = float(compute_yield_stress_mpa(
            age_arr, tau_0_mpa, a_thix_mpa_s, t_set, n_h,
        )[0])
        sigma_y = np.sqrt(3.0) * tau_y
        nu = float(compute_poisson_ratio(age_arr, nu_fresh, nu_hard, t_set)[0])
        G_mod = float(compute_shear_modulus_mpa(
            np.array([tau_y]), gamma_c, e_inf, nu_hard,
        )[0])
        E = float(compute_young_modulus_mpa(
            np.array([G_mod]), np.array([nu]),
        )[0])

        K_n_phys = beta * sigma_y ** 2 / (2.0 * g_i_c)
        K_t_phys = beta * tau_y ** 2 / (2.0 * g_ii_c)

        for h in h_values:
            K_min = k_min_mult * E / h
            K_n = max(K_n_phys, K_min)
            K_t = max(K_t_phys, K_min)

            # Grid of (delta_n, delta_t).
            delta_n_peak = np.sqrt(max(G_Ic_eff / max(K_n, 1e-30), 0))
            delta_t_peak = np.sqrt(max(G_IIc_eff / max(K_t, 1e-30), 0))
            dn_max = 5.0 * max(delta_n_peak, 1e-12)
            dt_max = 5.0 * max(delta_t_peak, 1e-12)

            dn = np.linspace(1e-12, dn_max, n_grid)
            dt = np.linspace(1e-12, dt_max, n_grid)
            DN, DT = np.meshgrid(dn, dt, indexing="ij")

            alpha_n = K_n / (2.0 * max(G_Ic_eff, 1e-30))
            alpha_t = K_t / (2.0 * max(G_IIc_eff, 1e-30))
            XI = alpha_n * DN ** 2 + alpha_t * DT ** 2
            EXP = np.exp(-XI)

            # Hessian entries (∂T_i/∂δ_j for opening regime).
            H_nn = K_n * (r_s + (1.0 - r_s) * EXP * (1.0 - 2.0 * alpha_n * DN ** 2))
            H_tt = K_t * (r_s + (1.0 - r_s) * EXP * (1.0 - 2.0 * alpha_t * DT ** 2))
            # Cross term from shared damage variable.
            H_nt = -(1.0 - r_s) * EXP * 2.0 * alpha_n * DN * K_t * DT * alpha_t / max(alpha_n, 1e-30)
            # Simplify: H_nt = -(1-r_s) * exp(-xi) * K_n * K_t * dn * dt / (G_Ic_eff * ...)
            # More precisely: ∂²(T_n δ_n + T_t δ_t)/∂δ_n∂δ_t
            # T_n = K_n [r_s + (1-r_s)exp(-xi)] dn  => ∂T_n/∂dt = K_n(1-r_s)(-exp(-xi)) * 2*alpha_t*dt * dn
            H_nt = -(1.0 - r_s) * EXP * K_n * DN * 2.0 * alpha_t * DT

            # Eigenvalues of 2x2 symmetric matrix.
            trace = H_nn + H_tt
            det = H_nn * H_tt - H_nt ** 2
            discriminant = np.maximum(trace ** 2 - 4.0 * det, 0.0)
            lam_min = 0.5 * (trace - np.sqrt(discriminant))

            non_convex_mask = lam_min < 0
            non_convex_fraction = float(np.mean(non_convex_mask))

            # Min eigenvalue at the mode-I peak (delta_n = delta_n_peak, delta_t ≈ 0).
            i_peak_n = min(np.searchsorted(dn, delta_n_peak), n_grid - 1)
            min_eig_at_peak = float(lam_min[i_peak_n, 0])

            rows.append({
                "age": float(age),
                "h": float(h),
                "K_n": K_n,
                "K_t": K_t,
                "non_convex_fraction": non_convex_fraction,
                "min_eigenvalue_at_peak": min_eig_at_peak,
                "min_eigenvalue_global": float(np.min(lam_min)),
            })

    out = {
        "beta_bond": beta,
        "G_Ic_eff": G_Ic_eff,
        "G_IIc_eff": G_IIc_eff,
        "r_s": r_s,
        "n_grid": n_grid,
        "rows": rows,
    }

    if verbose:
        _print_convexity_check(out)

    return out


def _print_convexity_check(result):
    """Print mixed-mode convexity table."""
    print(f"\n  Mixed-Mode Convexity Check")
    print(f"    beta = {result['beta_bond']:.4f},  "
          f"G_Ic_eff = {result['G_Ic_eff']:.4e},  "
          f"G_IIc_eff = {result['G_IIc_eff']:.4e},  "
          f"r_s = {result['r_s']}")

    rows = result["rows"]
    if not rows:
        return

    print(f"\n    {'age':>6s}  {'h':>6s}  {'K_n':>10s}  {'K_t':>10s}  "
          f"{'non_convex':>10s}  {'min_eig_pk':>10s}  {'min_eig_gl':>10s}")
    print(f"    {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}")

    for r in rows:
        print(f"    {r['age']:6.0f}  {r['h']:6.1f}  "
              f"{r['K_n']:10.4e}  {r['K_t']:10.4e}  "
              f"{r['non_convex_fraction']:9.1%}  "
              f"{r['min_eigenvalue_at_peak']:10.4e}  "
              f"{r['min_eigenvalue_global']:10.4e}")

    print(f"\n    Note: Non-convex regions beyond the peak are expected for")
    print(f"    any softening law. Concern arises only if the solver path")
    print(f"    enters the non-convex region without damping.")


# ---------------------------------------------------------------------------
# Tests 4.2 + 4.3: Shared time-stepping with damage tracking
# ---------------------------------------------------------------------------

def _run_time_stepping_with_damage_tracking(state, n_steps=20, g_factor=50,
                                            max_iter=50, verbose=True):
    """Run a mini time-stepping loop with amplified gravity, tracking damage.

    At each converged step, extracts per-facet damage, displacement jumps,
    and tractions.  Returns the full history for analysis by Tests 4.2
    and 4.3.

    Args:
        state: DiagnosticState.
        n_steps: Number of time steps.
        g_factor: Gravity amplification factor.
        max_iter: Max Newton iterations per step.
        verbose: Print progress on rank 0.

    Returns:
        dict with per-step records including per-facet damage arrays.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx

    # Build time grid spanning the full activation range (not just first N steps).
    local_t_min = float(bt.min()) if bt.size > 0 else np.inf
    local_t_max = float(bt.max()) if bt.size > 0 else -np.inf
    global_t_min = comm.allreduce(local_t_min, op=MPI.MIN)
    global_t_max = comm.allreduce(local_t_max, op=MPI.MAX)
    time_cfg = state.cfg["time_stepping"]
    t_start = global_t_min + time_cfg["start_offset"]
    t_end = global_t_max * time_cfg["end_multiplier"]
    times = np.linspace(t_start, t_end, n_steps)

    if len(times) < 2:
        return {"steps": [], "n_steps": 0, "g_factor": g_factor}

    dt = times[1] - times[0]
    t_prev = times[0] - dt

    # Amplify gravity.
    g_base = state.materials.g_const.value
    state.materials.g_const.value = g_base * g_factor

    # Reset displacement and damage_max history.
    state.u.x.array[:] = 0.0
    state.u.x.scatter_forward()
    state.materials.damage_max.x.array[:] = 0.0
    state.materials.damage_max.x.scatter_forward()

    # Get element diameter for contact check.
    h_arr = compute_cell_diameters(state.msh)
    h_elem = float(np.mean(h_arr)) if h_arr.size > 0 else 1.0

    # Precompute inter-layer facet connectivity and normals for damage_max.
    msh = state.msh
    tags = state.interior_facet_tags
    _il_mask = tags.values == 1
    _il_facet_indices = tags.indices[_il_mask]
    n_il_facets = len(_il_facet_indices)
    il_cell_plus = np.empty(n_il_facets, dtype=np.int32)
    il_cell_minus = np.empty(n_il_facets, dtype=np.int32)
    il_facet_normals = np.empty((n_il_facets, 3), dtype=np.float64)
    if n_il_facets > 0:
        tdim = msh.topology.dim
        fdim = tdim - 1
        msh.topology.create_connectivity(fdim, tdim)
        msh.topology.create_connectivity(fdim, 0)
        _f_to_c = msh.topology.connectivity(fdim, tdim)
        _f_to_v = msh.topology.connectivity(fdim, 0)
        for _i, _f in enumerate(_il_facet_indices):
            _cells = _f_to_c.links(_f)
            il_cell_plus[_i] = _cells[0]
            il_cell_minus[_i] = _cells[1] if len(_cells) > 1 else _cells[0]
        from dolfinx.mesh import entities_to_geometry as _e2g
        _geom_x = msh.geometry.x
        for _i, _f in enumerate(_il_facet_indices):
            _verts = _f_to_v.links(_f)
            _geom_dofs = _e2g(
                msh, 0, _verts.astype(np.int32), False
            ).reshape(-1)
            _coords = _geom_x[_geom_dofs]
            _normal = np.cross(_coords[1] - _coords[0], _coords[3] - _coords[0])
            _norm = np.linalg.norm(_normal)
            il_facet_normals[_i] = _normal / _norm if _norm > 1e-12 else _normal

        # Precompute cell diameters.
        from dolfinx.cpp.mesh import h as _dolfinx_h
        _map_c = msh.topology.index_map(tdim)
        _all_cells = np.arange(
            _map_c.size_local + _map_c.num_ghosts, dtype=np.int32
        )
        il_h_cells = np.asarray(_dolfinx_h(msh._cpp_object, tdim, _all_cells))

    records = []

    for i, t_val in enumerate(times):
        t_val_f = float(t_val)

        # Find newly-active cells.
        newly_active = np.where((bt <= t_val_f) & (bt > t_prev))[0]

        # Initialize newly-active displacement.
        init_newly_activated_displacement(
            state.u, newly_active, state.support, state.cell_to_dofs,
        )

        # Update materials.
        state.materials.t_current.value = t_val_f
        state.materials.update_properties(t_val_f)

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

        # Update damage_max history variable.
        if n_il_facets > 0:
            active_mask = bt <= t_val_f
            from materials.damage_update import update_damage_max_numpy
            update_damage_max_numpy(
                state.u, state.materials, state.cell_to_dofs, bt,
                state.cfg, il_cell_plus, il_cell_minus,
                il_facet_normals, il_h_cells, active_mask,
            )

        # Extract per-facet interface data.
        state.t_val = t_val_f
        iface = _extract_interface_jumps(state)

        max_disp_local = float(np.max(np.abs(state.u.x.array)))
        max_disp = comm.allreduce(max_disp_local, op=MPI.MAX)

        entry = {
            "step": i,
            "t_val": t_val_f,
            "n_iter": int(n_iter),
            "converged": bool(converged),
            "max_displacement": max_disp,
            "n_facets": iface.get("n_facets", 0),
        }

        if iface["n_facets"] > 0:
            entry["damage"] = iface["damage"].copy()
            entry["jump_n"] = iface["jump_n"].copy()
            entry["jump_t_mag"] = iface["jump_t_mag"].copy()
            entry["delta_n_open"] = iface["delta_n_open"].copy()
            entry["max_damage"] = float(np.max(iface["damage"]))
            entry["n_damaged"] = int(np.sum(iface["damage"] > 0.01))
            entry["min_jump_n"] = float(np.min(iface["jump_n"]))
            entry["max_opening"] = float(np.max(iface["delta_n_open"]))
        else:
            entry["damage"] = np.array([])
            entry["jump_n"] = np.array([])
            entry["max_damage"] = 0.0
            entry["n_damaged"] = 0
            entry["min_jump_n"] = 0.0
            entry["max_opening"] = 0.0

        records.append(entry)
        t_prev = t_val_f

        if verbose and comm.rank == 0:
            status = "CONV" if converged else "FAIL"
            print(f"    step {i:3d}  t={t_val_f:8.2f}  {status} in {n_iter:2d}  "
                  f"|u|={max_disp:.2e}  "
                  f"d_max={entry['max_damage']:.4f}  "
                  f"n_dmg={entry['n_damaged']}")

    # Restore gravity.
    state.materials.g_const.value = g_base

    return {
        "steps": records,
        "n_steps": len(times),
        "g_factor": g_factor,
        "h_elem": h_elem,
    }


# ---------------------------------------------------------------------------
# Test 4.2: Damage irreversibility
# ---------------------------------------------------------------------------

def run_damage_irreversibility_check(tracking_result, verbose=True):
    """Analyse per-facet damage history for irreversibility violations.

    Args:
        tracking_result: dict from ``_run_time_stepping_with_damage_tracking``.
        verbose: Print results.

    Returns:
        dict with healing detection results.
    """
    steps = tracking_result.get("steps", [])
    if len(steps) < 2:
        return {"any_healing": False, "n_steps_checked": 0}

    healing_events = []
    prev_damage = None

    for entry in steps:
        damage = entry.get("damage", np.array([]))
        if damage.size == 0:
            prev_damage = None
            continue

        if prev_damage is not None and prev_damage.size == damage.size:
            decrease = prev_damage - damage
            healing_mask = decrease > 1e-6
            if np.any(healing_mask):
                healing_events.append({
                    "step": entry["step"],
                    "t_val": entry["t_val"],
                    "n_healing_facets": int(np.sum(healing_mask)),
                    "max_decrease": float(np.max(decrease[healing_mask])),
                    "mean_decrease": float(np.mean(decrease[healing_mask])),
                    "facet_indices": np.where(healing_mask)[0].tolist(),
                })

        prev_damage = damage.copy()

    any_healing = len(healing_events) > 0
    worst_decrease = 0.0
    if healing_events:
        worst_decrease = max(e["max_decrease"] for e in healing_events)

    result = {
        "any_healing": any_healing,
        "n_healing_events": len(healing_events),
        "worst_decrease": worst_decrease,
        "healing_events": healing_events,
        "n_steps_checked": len(steps) - 1,
    }

    if verbose:
        _print_irreversibility_check(result)

    return result


def _print_irreversibility_check(result):
    """Print damage irreversibility results."""
    status = "HEALING DETECTED" if result["any_healing"] else "NO HEALING"
    print(f"\n  Damage Irreversibility Check: {status}")
    print(f"    Steps checked: {result['n_steps_checked']}")

    if result["any_healing"]:
        print(f"    Healing events: {result['n_healing_events']}")
        print(f"    Worst decrease: {result['worst_decrease']:.6f}")
        print(f"\n    {'step':>4s}  {'t_val':>8s}  {'n_facets':>8s}  "
              f"{'max_decr':>10s}  {'mean_decr':>10s}")
        print(f"    {'-'*4}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
        for e in result["healing_events"]:
            print(f"    {e['step']:4d}  {e['t_val']:8.2f}  "
                  f"{e['n_healing_facets']:8d}  "
                  f"{e['max_decrease']:10.6f}  {e['mean_decrease']:10.6f}")

        print(f"\n    Recommendation: Implement damage history variable.")
        print(f"    damage_max = max(damage_current, damage_prev) per facet.")
    else:
        print(f"    No facet damage decreased between consecutive converged steps.")
        print(f"    Note: This does not guarantee irreversibility under all loading")
        print(f"    paths — only under this specific gravity amplification test.")


# ---------------------------------------------------------------------------
# Test 4.3: Contact penalty adequacy
# ---------------------------------------------------------------------------

def run_contact_penalty_check(tracking_result, verbose=True):
    """Check interpenetration magnitudes from the tracked time-stepping.

    Args:
        tracking_result: dict from ``_run_time_stepping_with_damage_tracking``.
        verbose: Print results.

    Returns:
        dict with interpenetration metrics.
    """
    steps = tracking_result.get("steps", [])
    h_elem = tracking_result.get("h_elem", 1.0)

    records = []
    for entry in steps:
        jump_n = entry.get("jump_n", np.array([]))
        if jump_n.size == 0:
            continue

        min_jn = float(np.min(jump_n))
        interpenetration = abs(min(min_jn, 0.0))
        ratio = interpenetration / max(h_elem, 1e-12)

        records.append({
            "step": entry["step"],
            "t_val": entry["t_val"],
            "min_jump_n": min_jn,
            "interpenetration": interpenetration,
            "ratio_h_elem": ratio,
            "adequate": ratio < 0.01,
        })

    worst_ratio = max(r["ratio_h_elem"] for r in records) if records else 0.0
    all_adequate = all(r["adequate"] for r in records) if records else True

    result = {
        "h_elem": h_elem,
        "records": records,
        "worst_ratio": worst_ratio,
        "all_adequate": all_adequate,
    }

    if verbose:
        _print_contact_penalty_check(result)

    return result


def _print_contact_penalty_check(result):
    """Print contact penalty check results."""
    status = "ADEQUATE" if result["all_adequate"] else "INADEQUATE"
    print(f"\n  Contact Penalty Check: {status}")
    print(f"    h_elem = {result['h_elem']:.1f} mm")
    print(f"    Threshold: |interpenetration| / h_elem < 0.01")

    records = result["records"]
    if not records:
        return

    print(f"\n    {'step':>4s}  {'t_val':>8s}  {'min_jn':>10s}  "
          f"{'interpen':>10s}  {'ratio':>10s}  {'ok':>4s}")
    print(f"    {'-'*4}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*4}")
    for r in records:
        ok = "ok" if r["adequate"] else "WARN"
        print(f"    {r['step']:4d}  {r['t_val']:8.2f}  "
              f"{r['min_jump_n']:10.2e}  {r['interpenetration']:10.2e}  "
              f"{r['ratio_h_elem']:10.2e}  {ok:>4s}")

    if not result["all_adequate"]:
        print(f"\n    Worst ratio: {result['worst_ratio']:.4e}")
        print(f"    Recommendation: Use a separate, stiffer contact penalty")
        print(f"    parameter not tied to K_n (which is now physical and soft).")
