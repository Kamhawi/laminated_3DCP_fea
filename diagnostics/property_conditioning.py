"""Phase 5: Material property conditioning diagnostics.

Evaluates age-dependent material property contrasts (E, nu, eta) across
active cells at representative time points.

Tests:
    5.0  Analytical property evolution (no mesh).
    5.1  Property contrast monitoring (E, eta, nu ratios).
    5.2  Poisson ratio locking indicator (lambda/(2*mu)).
    5.3  Viscosity-stiffness ratio (elastic predictor quality).
    5.4  VP sub-step accuracy (baseline vs 10x reference).
    5.5  Stagger error (operator-split residual measurement).
"""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc

from diagnostics.setup import prepare_state_at_time
from diagnostics.taylor_test import _assemble_residual_into
from materials.material_state import update_perzyna_state_cellwise
from materials.time_models import (
    compute_poisson_ratio,
    compute_shear_modulus_mpa,
    compute_viscosity_mpa_s,
    compute_yield_stress_mpa,
    compute_young_modulus_mpa,
    pa_to_mpa,
)
from solver.kinematics import epsilon
from solver.newton import get_inactive_dofs, solve_newton

# ---------------------------------------------------------------------------
# Concern thresholds
# ---------------------------------------------------------------------------
E_RATIO_THRESHOLD = 1e4
ETA_RATIO_THRESHOLD = 1e6
NU_RANGE_THRESHOLD = 0.2
NU_MAX_THRESHOLD = 0.49
E_MIN_THRESHOLD = 1e-3  # MPa
LOCKING_RATIO_THRESHOLD = 12.0
VP_RATIO_THRESHOLD = 1.0
VP_ACCURACY_THRESHOLD = 0.01  # 1% relative difference
STAGGER_RATIO_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_full_time_grid(state, n_steps=20):
    """Generate time grid spanning the full birth time range.

    Same pattern as ``cohesive_analysis.py`` time grid: ``np.linspace``
    from ``t_start`` to ``t_end`` covering all activations.
    """
    comm = state.comm
    bt = state.birth_times_dolfinx
    time_cfg = state.cfg["time_stepping"]

    global_t_min = comm.allreduce(float(bt.min()), op=MPI.MIN)
    global_t_max = comm.allreduce(float(bt.max()), op=MPI.MAX)

    t_start = global_t_min + time_cfg["start_offset"]
    t_end = global_t_max * time_cfg["end_multiplier"]
    return np.linspace(t_start, t_end, n_steps)


def _n_local(msh):
    """Number of locally-owned cells (excludes ghosts)."""
    return msh.topology.index_map(msh.topology.dim).size_local


# ---------------------------------------------------------------------------
# Test 5.0: Analytical property evolution
# ---------------------------------------------------------------------------

def compute_analytical_property_evolution(cfg, ages=None, verbose=True):
    """Evaluate material property evolution analytically (no mesh needed).

    Args:
        cfg: Config dict with ``material`` and ``hardening`` sections.
        ages: Array of material ages [s].  Defaults to representative set.
        verbose: Print table on rank 0.

    Returns:
        dict with per-age property values and derived quantities.
    """
    mat = cfg["material"]
    hrd = cfg["hardening"]

    tau_0_mpa = float(pa_to_mpa(mat["tau_0"]))
    a_thix_mpa_s = float(pa_to_mpa(mat["A_thix"]))
    mu_p_mpa_s = float(pa_to_mpa(mat["mu_p"]))
    gamma_c = float(mat["gamma_c"])
    t_set = float(hrd["t_set"])
    e_inf = float(hrd["E_inf"])
    nu_fresh = float(hrd["nu_fresh"])
    nu_hard = float(hrd["nu_hard"])
    n_h = float(hrd.get("n_h", 0.7))

    if ages is None:
        ages = np.array([0, 1, 5, 10, 30, 60, 150, 300, 600, 900], dtype=float)
    else:
        ages = np.asarray(ages, dtype=float)

    tau_y = compute_yield_stress_mpa(ages, tau_0_mpa, a_thix_mpa_s, t_set, n_h)
    sigma_y = np.sqrt(3.0) * tau_y
    nu = compute_poisson_ratio(ages, nu_fresh, nu_hard, t_set)
    G = compute_shear_modulus_mpa(tau_y, gamma_c, e_inf, nu_hard)
    E = compute_young_modulus_mpa(G, nu)
    eta = compute_viscosity_mpa_s(ages, mu_p_mpa_s, t_set)

    denom = np.maximum(1.0 - 2.0 * nu, 1e-12)
    locking_ratio = nu / denom
    eta_over_E = eta / np.maximum(E, 1e-30)

    result = {
        "ages": ages.tolist(),
        "E_mpa": E.tolist(),
        "nu": nu.tolist(),
        "eta_mpa_s": eta.tolist(),
        "tau_y_mpa": tau_y.tolist(),
        "sigma_y_mpa": sigma_y.tolist(),
        "locking_ratio": locking_ratio.tolist(),
        "eta_over_E_s": eta_over_E.tolist(),
        "params": {
            "tau_0_pa": float(mat["tau_0"]),
            "A_thix_pa_s": float(mat["A_thix"]),
            "mu_p_pa_s": float(mat["mu_p"]),
            "gamma_c": gamma_c,
            "t_set": t_set,
            "E_inf": e_inf,
            "nu_fresh": nu_fresh,
            "nu_hard": nu_hard,
            "n_h": n_h,
        },
    }

    if verbose:
        _print_analytical_evolution(result)

    return result


def _print_analytical_evolution(result):
    """Pretty-print analytical property evolution table."""
    print("\n  Analytical Property Evolution (no mesh)")
    p = result["params"]
    print(f"    tau_0 = {p['tau_0_pa']:.0f} Pa, A_thix = {p['A_thix_pa_s']:.0f} Pa/s, "
          f"mu_p = {p['mu_p_pa_s']:.0f} Pa*s")
    print(f"    gamma_c = {p['gamma_c']}, t_set = {p['t_set']:.0f} s, "
          f"E_inf = {p['E_inf']:.0f} MPa")
    print(f"    nu_fresh = {p['nu_fresh']}, nu_hard = {p['nu_hard']}, "
          f"n_h = {p['n_h']}")

    ages = result["ages"]
    print(f"\n    {'age [s]':>9s}  {'E [MPa]':>9s}  {'nu':>7s}  {'eta [MPa*s]':>12s}  "
          f"{'sigma_y':>9s}  {'lock_ratio':>10s}  {'eta/E [s]':>10s}")
    print(f"    {'-'*9}  {'-'*9}  {'-'*7}  {'-'*12}  "
          f"{'-'*9}  {'-'*10}  {'-'*10}")

    for i, age in enumerate(ages):
        print(f"    {age:9.1f}  {result['E_mpa'][i]:9.4f}  {result['nu'][i]:7.4f}  "
              f"{result['eta_mpa_s'][i]:12.4e}  {result['sigma_y_mpa'][i]:9.4f}  "
              f"{result['locking_ratio'][i]:10.3f}  {result['eta_over_E_s'][i]:10.4e}")


# ---------------------------------------------------------------------------
# Test 5.1: Property contrast monitoring
# ---------------------------------------------------------------------------

def check_property_contrasts(state, times=None, n_steps=20, verbose=True):
    """Monitor E, eta, nu contrasts across active cells over time.

    Args:
        state: DiagnosticState from ``build_diagnostic_state``.
        times: Optional explicit time array.
        n_steps: Number of time points if ``times`` is None.
        verbose: Print table on rank 0.

    Returns:
        dict with per-step records and worst-case values.
    """
    comm = state.comm
    if times is None:
        times = _generate_full_time_grid(state, n_steps)

    msh = state.msh
    n_loc = _n_local(msh)
    bt = state.birth_times_dolfinx

    records = []
    for t_val in times:
        t_val = float(t_val)
        prepare_state_at_time(state, t_val, u_mode="zero")

        active = bt[:n_loc] <= t_val
        n_active_local = int(np.sum(active))
        n_active = comm.allreduce(n_active_local, op=MPI.SUM)

        if n_active == 0:
            records.append({
                "t_val": t_val, "n_active": 0,
                "E_max": 0.0, "E_min": 0.0, "E_ratio": 1.0,
                "eta_max": 0.0, "eta_min": 0.0, "eta_ratio": 1.0,
                "nu_max": 0.0, "nu_min": 0.0, "nu_range": 0.0,
                "E_ratio_flag": False, "eta_ratio_flag": False,
                "nu_range_flag": False, "nu_max_flag": False, "E_min_flag": False,
            })
            continue

        E_arr = state.materials.E.x.array[:n_loc]
        nu_arr = state.materials.nu.x.array[:n_loc]
        eta_arr = state.materials.eta.x.array[:n_loc]

        E_active = E_arr[active]
        nu_active = nu_arr[active]
        eta_active = eta_arr[active]

        # Local extrema.
        E_max_l = float(np.max(E_active))
        E_min_l = float(np.min(E_active))
        eta_max_l = float(np.max(eta_active))
        eta_min_l = float(np.min(eta_active))
        nu_max_l = float(np.max(nu_active))
        nu_min_l = float(np.min(nu_active))

        # Global via MPI.
        E_max = comm.allreduce(E_max_l, op=MPI.MAX)
        E_min = comm.allreduce(E_min_l, op=MPI.MIN)
        eta_max = comm.allreduce(eta_max_l, op=MPI.MAX)
        eta_min = comm.allreduce(eta_min_l, op=MPI.MIN)
        nu_max = comm.allreduce(nu_max_l, op=MPI.MAX)
        nu_min = comm.allreduce(nu_min_l, op=MPI.MIN)

        E_ratio = E_max / max(E_min, 1e-30)
        eta_ratio = eta_max / max(eta_min, 1e-30)
        nu_range = nu_max - nu_min

        records.append({
            "t_val": t_val,
            "n_active": n_active,
            "E_max": E_max, "E_min": E_min, "E_ratio": E_ratio,
            "eta_max": eta_max, "eta_min": eta_min, "eta_ratio": eta_ratio,
            "nu_max": nu_max, "nu_min": nu_min, "nu_range": nu_range,
            "E_ratio_flag": E_ratio > E_RATIO_THRESHOLD,
            "eta_ratio_flag": eta_ratio > ETA_RATIO_THRESHOLD,
            "nu_range_flag": nu_range > NU_RANGE_THRESHOLD,
            "nu_max_flag": nu_max > NU_MAX_THRESHOLD,
            "E_min_flag": E_min < E_MIN_THRESHOLD,
        })

    # Aggregate.
    worst_E_ratio = max(r["E_ratio"] for r in records)
    worst_eta_ratio = max(r["eta_ratio"] for r in records)
    worst_nu_range = max(r["nu_range"] for r in records)
    worst_nu_max = max(r["nu_max"] for r in records)
    worst_E_min = min(r["E_min"] for r in records) if records else 0.0

    result = {
        "records": records,
        "worst_E_ratio": worst_E_ratio,
        "worst_eta_ratio": worst_eta_ratio,
        "worst_nu_range": worst_nu_range,
        "worst_nu_max": worst_nu_max,
        "worst_E_min": worst_E_min,
        "any_E_ratio_flag": any(r["E_ratio_flag"] for r in records),
        "any_eta_ratio_flag": any(r["eta_ratio_flag"] for r in records),
        "any_nu_range_flag": any(r["nu_range_flag"] for r in records),
        "any_nu_max_flag": any(r["nu_max_flag"] for r in records),
        "any_E_min_flag": any(r["E_min_flag"] for r in records),
    }

    if verbose and comm.rank == 0:
        _print_property_contrasts(result)

    return result


def _print_property_contrasts(result):
    """Pretty-print property contrast table."""
    records = result["records"]
    print(f"\n  Property Contrast Monitoring ({len(records)} steps)")
    print(f"    Thresholds: E_ratio>{E_RATIO_THRESHOLD:.0e}  "
          f"eta_ratio>{ETA_RATIO_THRESHOLD:.0e}  "
          f"nu_range>{NU_RANGE_THRESHOLD}  "
          f"nu_max>{NU_MAX_THRESHOLD}  "
          f"E_min<{E_MIN_THRESHOLD} MPa")

    print(f"\n    {'t_val':>8s}  {'n_act':>6s}  {'E_ratio':>10s}  {'eta_ratio':>10s}  "
          f"{'nu_range':>8s}  {'nu_max':>6s}  {'E_min':>10s}  {'flags':>12s}")
    print(f"    {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  "
          f"{'-'*8}  {'-'*6}  {'-'*10}  {'-'*12}")

    for r in records:
        flags = []
        if r["E_ratio_flag"]:
            flags.append("E_r")
        if r["eta_ratio_flag"]:
            flags.append("eta_r")
        if r["nu_range_flag"]:
            flags.append("nu_r")
        if r["nu_max_flag"]:
            flags.append("nu_m")
        if r["E_min_flag"]:
            flags.append("E_m")
        flag_str = ",".join(flags) if flags else "ok"

        print(f"    {r['t_val']:8.2f}  {r['n_active']:6d}  {r['E_ratio']:10.2e}  "
              f"{r['eta_ratio']:10.2e}  {r['nu_range']:8.4f}  {r['nu_max']:6.4f}  "
              f"{r['E_min']:10.4e}  {flag_str:>12s}")


# ---------------------------------------------------------------------------
# Test 5.2: Poisson ratio locking
# ---------------------------------------------------------------------------

def check_poisson_locking(state, times=None, n_steps=20, verbose=True):
    """Track volumetric locking indicator lambda/(2*mu) = nu/(1-2*nu).

    Args:
        state: DiagnosticState.
        times: Optional explicit time array.
        n_steps: Number of time points if ``times`` is None.
        verbose: Print table on rank 0.

    Returns:
        dict with per-step records and worst-case locking ratio.
    """
    comm = state.comm
    if times is None:
        times = _generate_full_time_grid(state, n_steps)

    msh = state.msh
    n_loc = _n_local(msh)
    bt = state.birth_times_dolfinx

    records = []
    for t_val in times:
        t_val = float(t_val)
        prepare_state_at_time(state, t_val, u_mode="zero")

        active = bt[:n_loc] <= t_val
        n_active_local = int(np.sum(active))
        n_active = comm.allreduce(n_active_local, op=MPI.SUM)

        if n_active == 0:
            records.append({
                "t_val": t_val, "n_active": 0,
                "max_locking_ratio": 0.0, "mean_locking_ratio": 0.0,
                "n_above": 0, "frac_above": 0.0, "max_nu": 0.0,
                "locking_flag": False,
            })
            continue

        nu_active = state.materials.nu.x.array[:n_loc][active]
        denom = np.maximum(1.0 - 2.0 * nu_active, 1e-12)
        lr = nu_active / denom

        max_lr_l = float(np.max(lr))
        sum_lr_l = float(np.sum(lr))
        n_above_l = int(np.sum(lr > LOCKING_RATIO_THRESHOLD))
        max_nu_l = float(np.max(nu_active))

        max_lr = comm.allreduce(max_lr_l, op=MPI.MAX)
        sum_lr = comm.allreduce(sum_lr_l, op=MPI.SUM)
        n_above = comm.allreduce(n_above_l, op=MPI.SUM)
        max_nu = comm.allreduce(max_nu_l, op=MPI.MAX)
        mean_lr = sum_lr / max(n_active, 1)
        frac_above = n_above / max(n_active, 1)

        records.append({
            "t_val": t_val,
            "n_active": n_active,
            "max_locking_ratio": max_lr,
            "mean_locking_ratio": mean_lr,
            "n_above": n_above,
            "frac_above": frac_above,
            "max_nu": max_nu,
            "locking_flag": max_lr > LOCKING_RATIO_THRESHOLD,
        })

    worst = max(r["max_locking_ratio"] for r in records) if records else 0.0
    result = {
        "records": records,
        "worst_locking_ratio": worst,
        "any_locking_flag": any(r["locking_flag"] for r in records),
    }

    if verbose and comm.rank == 0:
        _print_poisson_locking(result)

    return result


def _print_poisson_locking(result):
    """Pretty-print Poisson locking table."""
    records = result["records"]
    print(f"\n  Poisson Ratio Locking (lambda/(2*mu) = nu/(1-2*nu))")
    print(f"    Threshold: locking_ratio > {LOCKING_RATIO_THRESHOLD}")

    print(f"\n    {'t_val':>8s}  {'n_act':>6s}  {'max_ratio':>10s}  {'mean_ratio':>10s}  "
          f"{'n_above':>7s}  {'frac':>6s}  {'max_nu':>6s}  {'flag':>5s}")
    print(f"    {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  "
          f"{'-'*7}  {'-'*6}  {'-'*6}  {'-'*5}")

    for r in records:
        flag = "WARN" if r["locking_flag"] else "ok"
        print(f"    {r['t_val']:8.2f}  {r['n_active']:6d}  "
              f"{r['max_locking_ratio']:10.3f}  {r['mean_locking_ratio']:10.3f}  "
              f"{r['n_above']:7d}  {r['frac_above']:6.4f}  "
              f"{r['max_nu']:6.4f}  {flag:>5s}")


# ---------------------------------------------------------------------------
# Test 5.3: Viscosity-stiffness ratio
# ---------------------------------------------------------------------------

def check_viscosity_stiffness_ratio(state, times=None, n_steps=20, verbose=True):
    """Evaluate elastic predictor quality: dt / min(eta/E).

    When this ratio >> 1, the operator-split frozen ``eps_vp`` deviates
    significantly from the converged viscoplastic state.

    Args:
        state: DiagnosticState.
        times: Optional explicit time array.
        n_steps: Number of time points if ``times`` is None.
        verbose: Print table on rank 0.

    Returns:
        dict with per-interval records and worst-case ratio.
    """
    comm = state.comm
    if times is None:
        times = _generate_full_time_grid(state, n_steps)

    msh = state.msh
    n_loc = _n_local(msh)
    bt = state.birth_times_dolfinx

    records = []
    for i in range(len(times) - 1):
        t_val = float(times[i + 1])
        dt = float(times[i + 1] - times[i])

        prepare_state_at_time(state, t_val, u_mode="zero")

        active = bt[:n_loc] <= t_val
        n_active_local = int(np.sum(active))
        n_active = comm.allreduce(n_active_local, op=MPI.SUM)

        if n_active == 0:
            records.append({
                "t_val": t_val, "dt": dt, "n_active": 0,
                "min_eta_over_E": 0.0, "max_eta_over_E": 0.0,
                "vp_ratio": 0.0, "vp_ratio_flag": False,
            })
            continue

        E_active = state.materials.E.x.array[:n_loc][active]
        eta_active = state.materials.eta.x.array[:n_loc][active]
        eta_over_E = eta_active / np.maximum(E_active, 1e-30)

        min_eoe_l = float(np.min(eta_over_E))
        max_eoe_l = float(np.max(eta_over_E))

        min_eoe = comm.allreduce(min_eoe_l, op=MPI.MIN)
        max_eoe = comm.allreduce(max_eoe_l, op=MPI.MAX)
        vp_ratio = dt / max(min_eoe, 1e-30)

        records.append({
            "t_val": t_val,
            "dt": dt,
            "n_active": n_active,
            "min_eta_over_E": min_eoe,
            "max_eta_over_E": max_eoe,
            "vp_ratio": vp_ratio,
            "vp_ratio_flag": vp_ratio > VP_RATIO_THRESHOLD,
        })

    worst = max(r["vp_ratio"] for r in records) if records else 0.0
    result = {
        "records": records,
        "worst_vp_ratio": worst,
        "any_vp_ratio_flag": any(r["vp_ratio_flag"] for r in records),
    }

    if verbose and comm.rank == 0:
        _print_viscosity_stiffness(result)

    return result


def _print_viscosity_stiffness(result):
    """Pretty-print viscosity-stiffness ratio table."""
    records = result["records"]
    print(f"\n  Viscosity-Stiffness Ratio (dt / min(eta/E))")
    print(f"    When >> 1, the frozen eps_vp predictor deviates significantly.")
    print(f"    Threshold: vp_ratio > {VP_RATIO_THRESHOLD}")

    print(f"\n    {'t_val':>8s}  {'dt':>8s}  {'n_act':>6s}  {'min(eta/E)':>11s}  "
          f"{'max(eta/E)':>11s}  {'vp_ratio':>10s}  {'flag':>5s}")
    print(f"    {'-'*8}  {'-'*8}  {'-'*6}  {'-'*11}  "
          f"{'-'*11}  {'-'*10}  {'-'*5}")

    for r in records:
        flag = "WARN" if r["vp_ratio_flag"] else "ok"
        print(f"    {r['t_val']:8.2f}  {r['dt']:8.4f}  {r['n_active']:6d}  "
              f"{r['min_eta_over_E']:11.4e}  {r['max_eta_over_E']:11.4e}  "
              f"{r['vp_ratio']:10.1f}  {flag:>5s}")


# ---------------------------------------------------------------------------
# Helpers for Tests 5.4 and 5.5
# ---------------------------------------------------------------------------

def _build_dg0_projection_tools(state):
    """Build DG0 tensor projection infrastructure.

    Mirrors ``solver/time_stepper.py`` lines 248–293.  Returns a callable
    ``project_strain()`` that projects ``epsilon(u)`` into
    ``state.materials.strain``.
    """
    msh = state.msh
    materials = state.materials
    u = state.u
    dx = ufl.Measure("dx", domain=msh)

    proj_trial = ufl.TrialFunction(materials.V_DG0_tensor)
    proj_test = ufl.TestFunction(materials.V_DG0_tensor)

    a_proj_form = fem.form(ufl.inner(proj_trial, proj_test) * dx)
    A_proj = assemble_matrix(a_proj_form)
    A_proj.assemble()

    diag_proj = A_proj.getDiagonal()
    inv_diag_proj = diag_proj.copy()
    inv_diag_proj.array[:] = 1.0 / np.maximum(inv_diag_proj.array, 1e-30)

    L_strain_form = fem.form(ufl.inner(epsilon(u), proj_test) * dx)
    b_strain = create_vector(L_strain_form)

    def project_strain():
        """Project epsilon(u) into materials.strain (DG0 tensor)."""
        with b_strain.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b_strain, L_strain_form)
        b_strain.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE,
        )
        materials.strain.x.petsc_vec.pointwiseMult(inv_diag_proj, b_strain)
        materials.strain.x.scatter_forward()

    return project_strain


def _run_substepped_perzyna(
    strain_arr, eps_vp_prev, materials, active_mask, dt, n_sub,
):
    """Run explicit Perzyna VP update with ``n_sub`` sub-steps.

    Replicates the production sub-stepping loop from
    ``solver/time_stepper.py`` lines 1007–1096.

    Returns:
        tuple: ``(eps_vp_new, sigma_new, vm_new, f_trial)``.
    """
    sub_dt = dt / max(n_sub, 1)
    eps_vp_tmp = eps_vp_prev.copy()

    # Sub-step 1: all active cells → identify yielding cells.
    eps_vp_tmp, sigma_new, vm_new, _, f_trial = update_perzyna_state_cellwise(
        strain_total=strain_arr,
        eps_vp_prev=eps_vp_tmp,
        e_arr=materials.e_arr,
        nu_arr=materials.nu_arr,
        sigma_y_arr=materials.sigma_y_arr,
        eta_arr=materials.eta_arr,
        dt=sub_dt,
        active_mask=active_mask,
    )

    yielding_mask = (f_trial > 0.0) & active_mask
    if np.any(yielding_mask) and n_sub > 1:
        strain_y = strain_arr[yielding_mask]
        eps_vp_y = eps_vp_tmp[yielding_mask]
        e_y = materials.e_arr[yielding_mask]
        nu_y = materials.nu_arr[yielding_mask]
        sy_y = materials.sigma_y_arr[yielding_mask]
        eta_y = materials.eta_arr[yielding_mask]

        for _ in range(2, n_sub + 1):
            eps_vp_y, _, _, _, _ = update_perzyna_state_cellwise(
                strain_total=strain_y,
                eps_vp_prev=eps_vp_y,
                e_arr=e_y,
                nu_arr=nu_y,
                sigma_y_arr=sy_y,
                eta_arr=eta_y,
                dt=sub_dt,
                active_mask=None,
            )

        eps_vp_tmp[yielding_mask] = eps_vp_y

        # Final dt=0 pass for consistent stress diagnostics.
        eps_vp_tmp, sigma_new, vm_new, _, f_trial = update_perzyna_state_cellwise(
            strain_total=strain_arr,
            eps_vp_prev=eps_vp_tmp,
            e_arr=materials.e_arr,
            nu_arr=materials.nu_arr,
            sigma_y_arr=materials.sigma_y_arr,
            eta_arr=materials.eta_arr,
            dt=0.0,
            active_mask=active_mask,
        )

    return eps_vp_tmp, sigma_new, vm_new, f_trial


def _assemble_residual_norm(state, inactive_dofs):
    """Assemble ``||F(u)||`` with inactive DOFs zeroed."""
    b = create_vector(state.F_form)
    _assemble_residual_into(b, state.F_form)
    if len(inactive_dofs) > 0:
        b.array[inactive_dofs] = 0.0
    norm = b.norm()
    b.destroy()
    return norm


# ---------------------------------------------------------------------------
# Test 5.4: VP sub-step accuracy
# ---------------------------------------------------------------------------

def check_vp_substep_accuracy(
    state, times=None, n_steps=10, g_factor=50, verbose=True,
):
    """Compare baseline sub-stepped Perzyna update vs 10x reference.

    At each time point, solves Newton with amplified gravity to get
    non-trivial strains, then compares VP updates at baseline ``n_sub``
    vs ``10*n_sub``.

    Args:
        state: DiagnosticState.
        times: Optional explicit time array.
        n_steps: Number of time points if ``times`` is None.
        g_factor: Gravity amplification factor.
        verbose: Print table on rank 0.

    Returns:
        dict with per-step accuracy records.
    """
    comm = state.comm
    if times is None:
        times = _generate_full_time_grid(state, n_steps)

    msh = state.msh
    n_loc = _n_local(msh)
    bt = state.birth_times_dolfinx

    # Build DG0 projection infrastructure.
    project_strain = _build_dg0_projection_tools(state)

    # Amplify gravity.
    g_base = state.materials.g_val
    state.materials.g_const.value = g_base * g_factor

    records = []
    for i in range(len(times) - 1):
        t_val = float(times[i + 1])
        dt = float(times[i + 1] - times[i])

        # Prepare state and solve Newton.
        prepare_state_at_time(state, t_val, u_mode="zero")

        n_iter, converged, msg = solve_newton(
            state.u, state.V, state.msh, state.F_form, state.J_form,
            t_val=t_val,
            birth_times=state.birth_times_dolfinx,
            cell_to_dofs=state.cell_to_dofs,
            max_iter=50,
            debug=False,
        )

        if not converged:
            if comm.rank == 0 and verbose:
                print(f"    t={t_val:.2f}: Newton did not converge ({msg}), skipping")
            records.append({
                "t_val": t_val, "dt": dt, "converged": False,
                "n_iter": int(n_iter), "n_active": 0,
                "n_sub_base": 0, "n_sub_ref": 0,
                "max_evp_err": 0.0, "mean_evp_err": 0.0,
                "max_sig_err": 0.0, "mean_sig_err": 0.0,
                "accuracy_flag": False,
            })
            continue

        # Project strain to DG0.
        project_strain()
        strain_arr = state.materials.strain.x.array.reshape((-1, 3, 3))
        eps_vp_prev = state.materials.eps_vp.x.array.reshape((-1, 3, 3)).copy()

        active = bt[:n_loc] <= t_val
        n_active_local = int(np.sum(active))
        n_active = comm.allreduce(n_active_local, op=MPI.SUM)

        # Pad active mask to full array length (including ghosts).
        active_full = np.zeros(strain_arr.shape[0], dtype=bool)
        active_full[:n_loc] = active

        # Compute baseline n_sub (same formula as time_stepper.py:997).
        E_active = state.materials.e_arr[active_full]
        eta_active = state.materials.eta_arr[active_full]
        eta_over_E = eta_active / np.maximum(E_active, 1e-12)
        min_eoe_local = float(np.min(eta_over_E)) if len(eta_over_E) > 0 else np.inf
        min_eoe = comm.allreduce(min_eoe_local, op=MPI.MIN)
        dt_limit = max(min_eoe, 1e-12)
        n_sub_base = max(1, int(np.ceil(dt / dt_limit)))
        n_sub_ref = 10 * n_sub_base

        # Run baseline and reference.
        evp_base, sig_base, vm_base, _ = _run_substepped_perzyna(
            strain_arr, eps_vp_prev, state.materials, active_full, dt, n_sub_base,
        )
        evp_ref, sig_ref, vm_ref, _ = _run_substepped_perzyna(
            strain_arr, eps_vp_prev, state.materials, active_full, dt, n_sub_ref,
        )

        # Error metrics on active cells.
        evp_diff = evp_base[active_full] - evp_ref[active_full]
        evp_ref_act = evp_ref[active_full]
        evp_norm_per_cell = np.sqrt(np.sum(evp_diff ** 2, axis=(1, 2)))
        evp_ref_norm = np.sqrt(np.sum(evp_ref_act ** 2, axis=(1, 2)))
        evp_rel = evp_norm_per_cell / np.maximum(evp_ref_norm, 1e-12)

        sig_diff = sig_base[active_full] - sig_ref[active_full]
        sig_ref_act = sig_ref[active_full]
        sig_norm_per_cell = np.sqrt(np.sum(sig_diff ** 2, axis=(1, 2)))
        sig_ref_norm = np.sqrt(np.sum(sig_ref_act ** 2, axis=(1, 2)))
        sig_rel = sig_norm_per_cell / np.maximum(sig_ref_norm, 1e-12)

        max_evp_l = float(np.max(evp_rel)) if len(evp_rel) > 0 else 0.0
        mean_evp_l = float(np.mean(evp_rel)) if len(evp_rel) > 0 else 0.0
        max_sig_l = float(np.max(sig_rel)) if len(sig_rel) > 0 else 0.0
        mean_sig_l = float(np.mean(sig_rel)) if len(sig_rel) > 0 else 0.0

        max_evp = comm.allreduce(max_evp_l, op=MPI.MAX)
        max_sig = comm.allreduce(max_sig_l, op=MPI.MAX)
        # Weighted mean would be more correct, but max is the diagnostic here.
        mean_evp = comm.allreduce(mean_evp_l * n_active_local, op=MPI.SUM) / max(n_active, 1)
        mean_sig = comm.allreduce(mean_sig_l * n_active_local, op=MPI.SUM) / max(n_active, 1)

        flag = max_evp > VP_ACCURACY_THRESHOLD or max_sig > VP_ACCURACY_THRESHOLD

        records.append({
            "t_val": t_val,
            "dt": dt,
            "converged": True,
            "n_iter": int(n_iter),
            "n_active": n_active,
            "n_sub_base": n_sub_base,
            "n_sub_ref": n_sub_ref,
            "max_evp_err": max_evp,
            "mean_evp_err": mean_evp,
            "max_sig_err": max_sig,
            "mean_sig_err": mean_sig,
            "accuracy_flag": flag,
        })

        if comm.rank == 0 and verbose:
            fs = "WARN" if flag else "ok"
            print(f"    t={t_val:.2f}  n_sub={n_sub_base}/{n_sub_ref}  "
                  f"evp_err={max_evp:.2e}/{mean_evp:.2e}  "
                  f"sig_err={max_sig:.2e}/{mean_sig:.2e}  {fs}")

    # Restore gravity.
    state.materials.g_const.value = g_base

    worst_evp = max((r["max_evp_err"] for r in records), default=0.0)
    worst_sig = max((r["max_sig_err"] for r in records), default=0.0)
    result = {
        "records": records,
        "g_factor": g_factor,
        "worst_max_evp_err": worst_evp,
        "worst_max_sig_err": worst_sig,
        "any_accuracy_flag": any(r["accuracy_flag"] for r in records),
    }

    if verbose and comm.rank == 0:
        _print_vp_accuracy(result)

    return result


def _print_vp_accuracy(result):
    """Pretty-print VP sub-step accuracy table."""
    records = result["records"]
    print(f"\n  VP Sub-Step Accuracy (baseline n_sub vs 10x reference, "
          f"g_factor={result['g_factor']})")
    print(f"    Threshold: max relative error > {VP_ACCURACY_THRESHOLD}")

    print(f"\n    {'t_val':>8s}  {'n_act':>6s}  {'n_sub':>7s}  {'n_ref':>7s}  "
          f"{'max_evp':>10s}  {'mean_evp':>10s}  {'max_sig':>10s}  {'mean_sig':>10s}  "
          f"{'flag':>5s}")
    print(f"    {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*5}")

    for r in records:
        if not r["converged"]:
            print(f"    {r['t_val']:8.2f}  {'—':>6s}  {'—':>7s}  {'—':>7s}  "
                  f"{'—':>10s}  {'—':>10s}  {'—':>10s}  {'—':>10s}  {'SKIP':>5s}")
            continue
        flag = "WARN" if r["accuracy_flag"] else "ok"
        print(f"    {r['t_val']:8.2f}  {r['n_active']:6d}  {r['n_sub_base']:7d}  "
              f"{r['n_sub_ref']:7d}  {r['max_evp_err']:10.2e}  "
              f"{r['mean_evp_err']:10.2e}  {r['max_sig_err']:10.2e}  "
              f"{r['mean_sig_err']:10.2e}  {flag:>5s}")


# ---------------------------------------------------------------------------
# Test 5.5: Stagger error
# ---------------------------------------------------------------------------

def check_stagger_error(
    state, times=None, n_steps=10, g_factor=50, verbose=True,
):
    """Measure operator-split stagger error across time steps.

    Simulates a sequence of time steps.  At each step, measures:

    - ``F_initial``: ``||F(u_prev)||`` after material update (the "natural"
      load change from activation/aging).
    - ``F_stagger``: ``||F(u_converged, eps_vp_new)||`` after VP update
      (the stagger error introduced by the operator split).

    The ``F_stagger`` at step *i* should approximately equal ``F_initial``
    at step *i+1*, with the difference being only the material property
    update.  Both are reported side-by-side for consistency checking.

    Args:
        state: DiagnosticState.
        times: Optional explicit time array.
        n_steps: Number of time points if ``times`` is None.
        g_factor: Gravity amplification factor.
        verbose: Print table on rank 0.

    Returns:
        dict with per-step stagger error records.
    """
    comm = state.comm
    if times is None:
        times = _generate_full_time_grid(state, n_steps)

    msh = state.msh
    n_loc = _n_local(msh)
    bt = state.birth_times_dolfinx

    # Build DG0 projection infrastructure.
    project_strain = _build_dg0_projection_tools(state)

    # Amplify gravity.
    g_base = state.materials.g_val
    state.materials.g_const.value = g_base * g_factor

    # Initialize at first time point.
    prepare_state_at_time(state, float(times[0]), u_mode="zero")

    records = []
    prev_F_stagger = None  # For consistency check

    for i in range(len(times) - 1):
        t_val = float(times[i + 1])
        dt = float(times[i + 1] - times[i])

        # --- Update materials to new time ---
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

        # --- Measure F_initial: ||F(u_prev)|| at new time ---
        F_initial = _assemble_residual_norm(state, inactive_owned)

        # --- Solve Newton ---
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
                "t_val": t_val, "dt": dt, "converged": False,
                "n_iter": int(n_iter), "n_active": n_active,
                "F_initial": F_initial, "F_stagger": 0.0,
                "stagger_ratio": 0.0,
                "prev_F_stagger": prev_F_stagger,
                "consistency_ratio": 0.0,
                "stagger_flag": False,
            })
            prev_F_stagger = None
            continue

        # Zero inactive DOFs after Newton.
        if inactive_owned.size > 0:
            state.u.x.array[inactive_owned] = 0.0
        state.u.x.scatter_forward()

        # --- Project strain and run VP update ---
        project_strain()
        strain_arr = state.materials.strain.x.array.reshape((-1, 3, 3))
        eps_vp_prev = state.materials.eps_vp.x.array.reshape((-1, 3, 3)).copy()

        active_full = np.zeros(strain_arr.shape[0], dtype=bool)
        active_full[:n_loc] = active

        # Compute n_sub.
        E_active = state.materials.e_arr[active_full]
        eta_active = state.materials.eta_arr[active_full]
        eta_over_E = eta_active / np.maximum(E_active, 1e-12)
        min_eoe_local = float(np.min(eta_over_E)) if len(eta_over_E) > 0 else np.inf
        min_eoe = comm.allreduce(min_eoe_local, op=MPI.MIN)
        dt_limit = max(min_eoe, 1e-12)
        n_sub = max(1, int(np.ceil(dt / dt_limit)))

        eps_vp_new, _, _, _ = _run_substepped_perzyna(
            strain_arr, eps_vp_prev, state.materials, active_full, dt, n_sub,
        )

        # Write eps_vp_new back so the form picks it up.
        state.materials.eps_vp.x.array[:] = eps_vp_new.ravel()
        state.materials.eps_vp.x.scatter_forward()

        # --- Measure F_stagger: ||F(u_converged, eps_vp_new)|| ---
        F_stagger = _assemble_residual_norm(state, inactive_owned)

        stagger_ratio = F_stagger / max(F_initial, 1e-12)

        # Consistency check: F_stagger[i-1] ≈ F_initial[i].
        consistency_ratio = 0.0
        if prev_F_stagger is not None and prev_F_stagger > 1e-12:
            consistency_ratio = F_initial / prev_F_stagger

        records.append({
            "t_val": t_val,
            "dt": dt,
            "converged": True,
            "n_iter": int(n_iter),
            "n_active": n_active,
            "n_sub": n_sub,
            "F_initial": F_initial,
            "F_stagger": F_stagger,
            "stagger_ratio": stagger_ratio,
            "prev_F_stagger": prev_F_stagger,
            "consistency_ratio": consistency_ratio,
            "stagger_flag": stagger_ratio > STAGGER_RATIO_THRESHOLD,
        })

        if comm.rank == 0 and verbose:
            fs = "WARN" if stagger_ratio > STAGGER_RATIO_THRESHOLD else "ok"
            cons = f"  cons={consistency_ratio:.3f}" if prev_F_stagger is not None else ""
            print(f"    t={t_val:.2f}  iters={n_iter}  "
                  f"F_init={F_initial:.2e}  F_stag={F_stagger:.2e}  "
                  f"ratio={stagger_ratio:.2f}  {fs}{cons}")

        prev_F_stagger = F_stagger

    # Restore gravity.
    state.materials.g_const.value = g_base

    worst_ratio = max((r["stagger_ratio"] for r in records), default=0.0)
    result = {
        "records": records,
        "g_factor": g_factor,
        "worst_stagger_ratio": worst_ratio,
        "any_stagger_flag": any(r["stagger_flag"] for r in records),
    }

    if verbose and comm.rank == 0:
        _print_stagger_error(result)

    return result


def _print_stagger_error(result):
    """Pretty-print stagger error table."""
    records = result["records"]
    print(f"\n  Stagger Error (g_factor={result['g_factor']})")
    print(f"    F_initial = ||F(u_prev)|| at new time (natural load change)")
    print(f"    F_stagger = ||F(u_conv, eps_vp_new)|| (operator-split error)")
    print(f"    Consistency: F_stagger[i] should ≈ F_initial[i+1]")
    print(f"    Threshold: stagger_ratio > {STAGGER_RATIO_THRESHOLD}")

    print(f"\n    {'t_val':>8s}  {'dt':>8s}  {'n_act':>6s}  {'iters':>5s}  "
          f"{'F_initial':>11s}  {'F_stagger':>11s}  {'ratio':>7s}  "
          f"{'prev_stag':>11s}  {'cons':>7s}  {'flag':>5s}")
    print(f"    {'-'*8}  {'-'*8}  {'-'*6}  {'-'*5}  "
          f"{'-'*11}  {'-'*11}  {'-'*7}  "
          f"{'-'*11}  {'-'*7}  {'-'*5}")

    for r in records:
        if not r["converged"]:
            prev_s = f"{r['prev_F_stagger']:11.4e}" if r["prev_F_stagger"] is not None else f"{'—':>11s}"
            print(f"    {r['t_val']:8.2f}  {r['dt']:8.4f}  {r['n_active']:6d}  "
                  f"{'—':>5s}  {r['F_initial']:11.4e}  {'—':>11s}  {'—':>7s}  "
                  f"{prev_s}  {'—':>7s}  {'SKIP':>5s}")
            continue
        flag = "WARN" if r["stagger_flag"] else "ok"
        prev_s = f"{r['prev_F_stagger']:11.4e}" if r["prev_F_stagger"] is not None else f"{'—':>11s}"
        cons_s = f"{r['consistency_ratio']:7.3f}" if r["prev_F_stagger"] is not None else f"{'—':>7s}"
        print(f"    {r['t_val']:8.2f}  {r['dt']:8.4f}  {r['n_active']:6d}  "
              f"{r['n_iter']:5d}  {r['F_initial']:11.4e}  {r['F_stagger']:11.4e}  "
              f"{r['stagger_ratio']:7.2f}  {prev_s}  {cons_s}  {flag:>5s}")
