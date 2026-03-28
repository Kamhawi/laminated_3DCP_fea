"""Taylor remainder tests for Jacobian consistency verification.

Implements:
  1.1  Global Taylor test on the full residual/Jacobian pair.
  1.2  Component-wise Taylor test on each of the 10 residual terms.

The Taylor remainder is  r(eps) = ||F(u + eps*du) - F(u) - eps*J(u)*du||.
For a correct Jacobian, r(eps) = O(eps^2) so the log-log slope is ~2.
"""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc

from solver.kinematics import epsilon, sigma, sigma_from_strain
from diagnostics.setup import prepare_state_at_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assemble_residual_into(b, F_form):
    """Zero, assemble, and ghost-update a residual vector in-place."""
    with b.localForm() as b_local:
        b_local.set(0.0)
    assemble_vector(b, F_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


def _assemble_jacobian(J_form):
    """Assemble and return a new Jacobian matrix."""
    A = assemble_matrix(J_form)
    A.assemble()
    return A


def make_perturbation(state, seed=42):
    """Create a normalized random perturbation, zeroed on inactive DOFs.

    Uses deterministic per-global-DOF seeding for MPI consistency.

    Returns:
        np.ndarray: Perturbation (same shape as ``state.u.x.array``),
            globally unit-normalized.
    """
    from diagnostics.setup import _make_mpi_consistent_perturbation

    du = _make_mpi_consistent_perturbation(state, scale=1.0, seed=seed)

    # Zero inactive DOFs.
    if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
        du[state.inactive_dofs] = 0.0

    # Global L2 normalization.
    local_sq = np.dot(du[: len(state.u.x.array)], du[: len(state.u.x.array)])
    global_sq = state.comm.allreduce(float(local_sq), op=MPI.SUM)
    norm = np.sqrt(global_sq)
    if norm > 0:
        du /= norm
    return du


# ---------------------------------------------------------------------------
# 1.1  Global Taylor remainder test
# ---------------------------------------------------------------------------

def run_taylor_test(state, F_form, J_form, label="global",
                    eps_values=None, seed=42, verbose=True):
    """Run the Taylor remainder test on a (residual, Jacobian) pair.

    Args:
        state: DiagnosticState with prepared u and inactive_dofs.
        F_form: Compiled residual form.
        J_form: Compiled Jacobian form.
        label: Human-readable label for output.
        eps_values: Perturbation magnitudes.  Default: 13 points in
            [1e-2, 1e-8].
        seed: Random seed for perturbation.
        verbose: Print results table on rank 0.

    Returns:
        dict with keys ``eps``, ``remainder``, ``rates``, ``F0_norm``,
        ``Jdu_norm``.
    """
    if eps_values is None:
        eps_values = np.logspace(-2, -8, 13)

    comm = state.comm
    u = state.u
    u0 = u.x.array.copy()

    # Build perturbation vector.
    du_arr = make_perturbation(state, seed=seed)

    # Assemble F(u0).
    b = create_vector(F_form)
    _assemble_residual_into(b, F_form)
    if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
        b.array[state.inactive_dofs] = 0.0
    F0 = b.array.copy()
    F0_norm = b.norm()

    # Assemble J(u0) and compute J*du.
    A = _assemble_jacobian(J_form)

    # Store du in a PETSc Vec for MatVec.
    du_func = fem.Function(state.V)
    du_func.x.array[:] = du_arr
    du_func.x.scatter_forward()

    Jdu_vec = A.createVecLeft()
    A.mult(du_func.x.petsc_vec, Jdu_vec)

    # Zero Jdu at inactive DOFs to match Newton solver's pin_inactive_dofs.
    # DG cross-facet coupling causes Jdu[inactive] != 0 even though
    # du[inactive] == 0, producing a spurious O(eps) remainder.
    if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
        Jdu_vec.array[state.inactive_dofs] = 0.0

    Jdu = Jdu_vec.array.copy()
    Jdu_norm = Jdu_vec.norm()

    # Sweep epsilon values.
    remainders = []
    for eps in eps_values:
        u.x.array[:] = u0 + eps * du_arr
        u.x.scatter_forward()

        _assemble_residual_into(b, F_form)
        if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
            b.array[state.inactive_dofs] = 0.0

        # r = F(u0+eps*du) - F(u0) - eps*J*du
        r_local = b.array - F0 - eps * Jdu
        local_sq = float(np.dot(r_local, r_local))
        global_sq = comm.allreduce(local_sq, op=MPI.SUM)
        remainders.append(np.sqrt(global_sq))

    # Restore u.
    u.x.array[:] = u0
    u.x.scatter_forward()

    # Cleanup PETSc objects.
    Jdu_vec.destroy()
    A.destroy()
    b.destroy()
    du_func.x.petsc_vec.destroy()

    remainders = np.array(remainders)

    # Compute convergence rates between consecutive eps values.
    rates = np.full(len(eps_values) - 1, np.nan)
    for i in range(len(rates)):
        if remainders[i] > 0 and remainders[i + 1] > 0:
            rates[i] = (np.log(remainders[i + 1]) - np.log(remainders[i])) / (
                np.log(eps_values[i + 1]) - np.log(eps_values[i])
            )

    # Detect if remainder has hit machine precision (noise floor).
    # If the remainder is below a relative threshold of ||F0|| * eps_mach,
    # subsequent rates are meaningless.
    noise_floor = max(F0_norm, Jdu_norm) * 1e-14

    if verbose and comm.rank == 0:
        print(f"\n  Taylor test: {label}")
        print(f"  ||F(u0)|| = {F0_norm:.4e},  ||J*du|| = {Jdu_norm:.4e}")
        print(f"  {'eps':>12s}  {'||r(eps)||':>12s}  {'rate':>8s}  {'note':>12s}")
        print(f"  {'---':>12s}  {'---':>12s}  {'---':>8s}  {'---':>12s}")
        for i, eps in enumerate(eps_values):
            rate_str = f"{rates[i - 1]:.2f}" if i > 0 and np.isfinite(rates[i - 1]) else "  —"
            note = "noise floor" if remainders[i] < noise_floor else ""
            print(f"  {eps:12.2e}  {remainders[i]:12.4e}  {rate_str:>8s}  {note:>12s}")

        # Mean rate in the reliable range: eps in [1e-2, 1e-7] AND
        # remainder above the noise floor.
        mask = np.isfinite(rates)
        mid = mask.copy()
        for i in range(len(rates)):
            ep_lo = min(eps_values[i], eps_values[i + 1])
            ep_hi = max(eps_values[i], eps_values[i + 1])
            if ep_lo < 1e-7 or ep_hi > 1e-2:
                mid[i] = False
            # Exclude intervals where either endpoint is in the noise floor.
            if remainders[i] < noise_floor or remainders[i + 1] < noise_floor:
                mid[i] = False
        reliable = rates[mid]
        if len(reliable) > 0:
            mean_rate = np.mean(reliable)
            print(f"  Mean rate (reliable range): {mean_rate:.3f}")
        elif np.max(remainders) < noise_floor:
            mean_rate = np.inf
            print("  Remainder ≈ 0 everywhere → Jacobian exact (bilinear form)")
        else:
            mean_rate = np.nan
            print("  Mean rate: N/A (no reliable data)")

    result = {
        "eps": eps_values,
        "remainder": remainders,
        "rates": rates,
        "F0_norm": F0_norm,
        "Jdu_norm": Jdu_norm,
        "label": label,
    }
    return result


# ---------------------------------------------------------------------------
# 1.2  Component-wise form builder
# ---------------------------------------------------------------------------

# Labels and nonlinearity flags for the 10 terms.
TERM_INFO = [
    ("bulk_EVP",              False),
    ("cohesive_traction",     True),
    ("SIP_consistency",       False),
    ("SIP_symmetry",          False),
    ("SIP_penalty",           False),
    ("global_stabilization",  False),
    ("Nitsche_consistency",   False),
    ("Nitsche_symmetry",      False),
    ("Nitsche_penalty",       False),
    ("gravity",               False),
]


def build_individual_form_terms(state):
    """Build each of the 10 residual terms as separate compiled forms.

    Mirrors the logic of ``build_evp_cohesive_weak_form`` in
    ``physics/weak_form.py`` but returns individual terms with their
    individual Jacobians.

    Args:
        state: DiagnosticState with msh, V, materials, birth_time_func,
            interior_facet_tags, dirichlet_tags, cfg.

    Returns:
        list[dict]: Each dict has keys ``label``, ``F_form``, ``J_form``,
            ``is_nonlinear``.
    """
    msh = state.msh
    V = state.V
    materials = state.materials
    birth_time_func = state.birth_time_func
    interior_facet_tags = state.interior_facet_tags
    dirichlet_tags = state.dirichlet_tags
    cfg = state.cfg

    u = materials.u
    v = ufl.TestFunction(V)
    du_trial = ufl.TrialFunction(V)

    activation_cfg = cfg["activation"]
    material_cfg = cfg["material"]
    interface_cfg = cfg["interface"]

    dx = ufl.Measure("dx", domain=msh)
    dS = ufl.Measure("dS", domain=msh, subdomain_data=interior_facet_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=dirichlet_tags)

    mu = materials.E / (2 * (1 + materials.nu))
    lam = materials.E * materials.nu / ((1 + materials.nu) * (1 - 2 * materials.nu))

    n = ufl.FacetNormal(msh)
    h = ufl.avg(ufl.CellDiameter(msh))
    h_boundary = ufl.CellDiameter(msh)

    gamma_bonded = interface_cfg["gamma_bonded_mult"] * ufl.avg(materials.E) / h
    gamma_boundary = interface_cfg["gamma_boundary_mult"] * materials.E / h_boundary

    # Reuse the *existing* t_current Constant (already created during
    # build_evp_cohesive_weak_form and stored on materials).
    sharpness = activation_cfg["sharpness"]
    alpha_min = activation_cfg["alpha_min"]
    alpha_raw = 0.5 * (1.0 + ufl.tanh(
        sharpness * (materials.t_current - birth_time_func)
    ))
    alpha = alpha_min + (1.0 - alpha_min) * alpha_raw
    alpha_facet = ufl.min_value(alpha("+"), alpha("-"))

    # Cohesive quantities.
    jump_u = ufl.jump(u)
    jump_n_scalar = ufl.dot(jump_u, n("+"))
    delta_n_open = ufl.max_value(jump_n_scalar, 0.0)
    delta_t = jump_u - jump_n_scalar * n("+")
    delta_t_sq = ufl.inner(delta_t, delta_t)

    tau_0_pa = material_cfg["tau_0"]
    a_thix_pa_s = material_cfg["A_thix"]
    g_i_c = interface_cfg["G_Ic"]
    g_ii_c = interface_cfg["G_IIc"]
    nozzle_pressure_pa = interface_cfg["nozzle_pressure"]
    residual_stiffness_ratio = interface_cfg.get("residual_stiffness_ratio", 0.01)
    k_min_mult = interface_cfg.get("K_min_mult", 0.1)

    t_open_facet = abs(birth_time_func("+") - birth_time_func("-"))
    tau_sub_pa_facet = tau_0_pa + a_thix_pa_s * t_open_facet
    p_deposit_pa = nozzle_pressure_pa
    phi_facet = p_deposit_pa / ufl.max_value(tau_sub_pa_facet, 1.0e-12)
    beta_bond_facet = 1.0 - ufl.exp(-phi_facet)
    beta_bond_facet = ufl.min_value(1.0, ufl.max_value(beta_bond_facet, 1.0e-8))

    g_i_c_interface = ufl.max_value(beta_bond_facet * g_i_c, 1.0e-12)
    g_ii_c_interface = ufl.max_value(beta_bond_facet * g_ii_c, 1.0e-12)

    sigma_bond = beta_bond_facet * ufl.avg(materials.sigma_y)
    tau_bond = beta_bond_facet * ufl.avg(materials.tau_y)

    K_n = sigma_bond * sigma_bond / (2.0 * g_i_c_interface)
    K_t = tau_bond * tau_bond / (2.0 * g_ii_c_interface)
    K_min = k_min_mult * ufl.avg(materials.E) / h
    K_n = ufl.max_value(K_n, K_min)
    K_t = ufl.max_value(K_t, K_min)

    delta_n_sq = delta_n_open * delta_n_open
    damage_arg_n = delta_n_sq / ufl.max_value(
        2.0 * g_i_c_interface / ufl.max_value(K_n, 1.0e-12), 1.0e-12,
    )
    damage_arg_t = delta_t_sq / ufl.max_value(
        2.0 * g_ii_c_interface / ufl.max_value(K_t, 1.0e-12), 1.0e-12,
    )
    damage = 1.0 - ufl.exp(-(damage_arg_n + damage_arg_t))
    damage = ufl.min_value(1.0, ufl.max_value(damage, 0.0))

    K_n_res = residual_stiffness_ratio * K_n
    K_t_res = residual_stiffness_ratio * K_t

    T_n = (
        ((1.0 - damage) * K_n + damage * K_n_res) * delta_n_open
        + K_n * ufl.min_value(jump_n_scalar, 0.0)
    )
    T_t = ((1.0 - damage) * K_t + damage * K_t_res) * delta_t
    T_cohesive = T_n * n("+") + T_t

    sigma_u = sigma_from_strain(epsilon(u) - materials.eps_vp, lam, mu)

    gamma_safe = 0.05 * ufl.avg(materials.E) / h
    g_vec = ufl.as_vector([0.0, 0.0, -materials.rho_const * materials.g_const])

    # Build each term as an individual UFL form.
    terms_ufl = [
        alpha * ufl.inner(sigma_u, epsilon(v)) * dx,
        ufl.inner(alpha_facet * T_cohesive, ufl.jump(v)) * dS(1),
        -ufl.inner(alpha_facet * ufl.avg(sigma_u) * n("+"), ufl.jump(v)) * dS(2),
        -ufl.inner(
            alpha_facet * ufl.avg(sigma(v, lam, mu)) * n("+"), ufl.jump(u),
        ) * dS(2),
        alpha_facet * gamma_bonded * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS(2),
        gamma_safe * ufl.inner(ufl.jump(u), ufl.jump(v)) * (dS(0) + dS(2)),
        -alpha * ufl.inner(sigma_u * n, v) * ds(1),
        -alpha * ufl.inner(sigma(v, lam, mu) * n, u) * ds(1),
        alpha * gamma_boundary * ufl.inner(u, v) * ds(1),
        -alpha * ufl.inner(g_vec, v) * dx,
    ]

    # Work around conda clang LTO linker issue on macOS by temporarily
    # using the system clang for JIT compilation if needed.
    import os
    orig_cc = os.environ.get("CC")
    jit_fix_applied = False
    if os.path.exists("/usr/bin/clang"):
        os.environ["CC"] = "/usr/bin/clang"
        jit_fix_applied = True

    results = []
    for i, F_i_ufl in enumerate(terms_ufl):
        label, is_nonlinear = TERM_INFO[i]
        J_i_ufl = ufl.derivative(F_i_ufl, u, du_trial)
        try:
            F_compiled = fem.form(F_i_ufl)
            J_compiled = fem.form(J_i_ufl)
        except Exception as e:
            if state.comm.rank == 0:
                print(f"    WARNING: JIT compilation failed for term '{label}': "
                      f"{type(e).__name__}: {e}")
                print(f"    Skipping this term.")
            results.append({
                "label": label,
                "F_form": None,
                "J_form": None,
                "is_nonlinear": is_nonlinear,
                "jit_failed": True,
            })
            continue
        results.append({
            "label": label,
            "F_form": F_compiled,
            "J_form": J_compiled,
            "is_nonlinear": is_nonlinear,
            "jit_failed": False,
        })

    # Restore original CC.
    if jit_fix_applied:
        if orig_cc is None:
            os.environ.pop("CC", None)
        else:
            os.environ["CC"] = orig_cc

    return results


# ---------------------------------------------------------------------------
# 1.2  Component-wise Taylor test runner
# ---------------------------------------------------------------------------

def run_componentwise_taylor_test(state, eps_values=None, seed=42, verbose=True):
    """Run Taylor tests on each of the 10 residual terms independently.

    Args:
        state: DiagnosticState with prepared u and inactive_dofs.
        eps_values: Perturbation magnitudes.
        seed: Random seed.
        verbose: Print per-term results.

    Returns:
        list[dict]: Per-term results.  Each has keys from ``run_taylor_test``
            plus ``label`` and ``is_nonlinear``.
    """
    comm = state.comm
    terms = build_individual_form_terms(state)

    if verbose and comm.rank == 0:
        print("\n  Component-wise Taylor test")
        print(f"  {'#':>3s}  {'Term':<24s}  {'Nonlinear':>9s}  "
              f"{'max ||r||':>12s}  {'Mean rate':>9s}")
        print(f"  {'---':>3s}  {'---':<24s}  {'---':>9s}  "
              f"{'---':>12s}  {'---':>9s}")

    all_results = []
    for i, term in enumerate(terms):
        label = term["label"]
        is_nl = term["is_nonlinear"]

        if term.get("jit_failed", False):
            if verbose and comm.rank == 0:
                nl_str = "YES" if is_nl else "no"
                print(f"  {i + 1:3d}  {label:<24s}  {nl_str:>9s}  "
                      f"{'SKIP (JIT)':>12s}  {'—':>9s}")
            all_results.append({
                "label": label, "is_nonlinear": is_nl, "jit_failed": True,
                "eps": np.array([]), "remainder": np.array([]),
                "rates": np.array([]), "F0_norm": 0.0, "Jdu_norm": 0.0,
            })
            continue

        result = run_taylor_test(
            state, term["F_form"], term["J_form"],
            label=label, eps_values=eps_values, seed=seed,
            verbose=False,
        )
        result["is_nonlinear"] = is_nl
        result["jit_failed"] = False

        max_r = np.max(result["remainder"])
        remainders = result["remainder"]
        rates = result["rates"]
        mask = np.isfinite(rates)
        # Restrict to reliable range and exclude noise floor.
        if eps_values is None:
            eps_used = np.logspace(-2, -8, 13)
        else:
            eps_used = eps_values
        noise_floor = max(result["F0_norm"], result["Jdu_norm"]) * 1e-14
        mid = mask.copy()
        for j in range(len(rates)):
            ep_lo = min(eps_used[j], eps_used[j + 1])
            ep_hi = max(eps_used[j], eps_used[j + 1])
            if ep_lo < 1e-7 or ep_hi > 1e-2:
                mid[j] = False
            if remainders[j] < noise_floor or remainders[j + 1] < noise_floor:
                mid[j] = False
        reliable = rates[mid]
        mean_rate = np.mean(reliable) if len(reliable) > 0 else np.nan

        if verbose and comm.rank == 0:
            nl_str = "YES" if is_nl else "no"
            if max_r < max(noise_floor, 1e-12):
                rate_str = "r≈0 (ok)"
            elif np.isfinite(mean_rate):
                rate_str = f"{mean_rate:.2f}"
            else:
                rate_str = "N/A"
            print(f"  {i + 1:3d}  {label:<24s}  {nl_str:>9s}  "
                  f"{max_r:12.4e}  {rate_str:>9s}")

        all_results.append(result)

    return all_results
