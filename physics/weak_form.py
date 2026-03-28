# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Weak-form construction for bulk EVP + cohesive interfaces.

This module builds the nonlinear residual and Jacobian forms used by the
monolithic solve. The algebra mirrors the original implementation.

Physics:
    The formulation combines:
    - small-strain bulk elastoviscoplasticity (with explicit ``eps_vp`` update
      outside this module),
    - activation weighting for not-yet-born material,
    - mixed-mode cohesive tractions on inter-layer interfaces,
    - SIP-like stabilization/consistency terms on selected interior facets.
"""

import ufl
from dolfinx import fem

from solver.kinematics import epsilon, sigma, sigma_from_strain


def build_evp_cohesive_weak_form(
    msh,
    V,
    materials,
    birth_time_func,
    interior_facet_tags,
    dirichlet_tags,
    cfg,
    return_ufl=False,
    alpha_facet_mode="min",
):
    """Build and compile nonlinear residual/Jacobian for the simulation.

    Args:
        msh: DOLFINx mesh.
        V: Displacement function space (DG1 vector).
        materials: MaterialStateManager containing DG0 fields and `u`.
        birth_time_func: DG0 cell birth-time field.
        interior_facet_tags: Meshtags for interior interfaces.
        dirichlet_tags: Meshtags for weak Dirichlet boundaries.
        cfg: Parsed configuration dictionary.

    Returns:
        tuple: ``(F_form, J_form)`` compiled DOLFINx forms.  When
            ``return_ufl=True``, returns ``(F_form, J_form, F_ufl, J_ufl)``
            with the pre-compilation UFL expressions for diagnostic use.

    Raises:
        ValueError: If ``materials.u`` has not been assigned before form build.

    Math overview:
        - Activation: alpha = alpha_min + (1-alpha_min)*0.5*(1+tanh(k(t-t_birth)))
        - Bulk stress: sigma = lambda tr(eps-eps_vp) I + 2 mu (eps-eps_vp)
        - Cohesive traction: mixed normal/tangential penalty with exponential
          softening damage and residual stiffness.
    """
    if not hasattr(materials, "u"):
        raise ValueError(
            "materials.u is required. Assign the displacement fem.Function before building forms."
        )

    u = materials.u
    v = ufl.TestFunction(V)

    activation_cfg = cfg["activation"]
    material_cfg = cfg["material"]
    interface_cfg = cfg["interface"]

    dx = ufl.Measure("dx", domain=msh)
    # Facet tags are interpreted by convention:
    # dS(1) -> inter-layer cohesive, dS(2) -> intra-layer bonded terms.
    dS = ufl.Measure("dS", domain=msh, subdomain_data=interior_facet_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=dirichlet_tags)

    mu = materials.E / (2 * (1 + materials.nu))
    lam = materials.E * materials.nu / ((1 + materials.nu) * (1 - 2 * materials.nu))

    n = ufl.FacetNormal(msh)
    h = ufl.avg(ufl.CellDiameter(msh))
    h_boundary = ufl.CellDiameter(msh)

    gamma_bonded = interface_cfg["gamma_bonded_mult"] * ufl.avg(materials.E) / h
    gamma_boundary = interface_cfg["gamma_boundary_mult"] * materials.E / h_boundary

    # Smooth birth activation via tanh ramp.
    materials.t_current = fem.Constant(msh, 0.0)
    sharpness = activation_cfg["sharpness"]
    alpha_min = activation_cfg["alpha_min"]
    alpha_raw = 0.5 * (1.0 + ufl.tanh(sharpness * (materials.t_current - birth_time_func)))
    alpha = alpha_min + (1.0 - alpha_min) * alpha_raw
    if alpha_facet_mode == "min":
        alpha_facet = ufl.min_value(alpha("+"), alpha("-"))
    elif alpha_facet_mode == "max":
        alpha_facet = ufl.max_value(alpha("+"), alpha("-"))
    elif alpha_facet_mode == "avg":
        alpha_facet = 0.5 * (alpha("+") + alpha("-"))
    else:
        raise ValueError(f"Unknown alpha_facet_mode: {alpha_facet_mode!r}")
    # Cohesive interfaces use max: couple at the mature side's strength
    alpha_facet_cohesive = ufl.max_value(alpha("+"), alpha("-"))

    # ``ufl.jump(u)`` is the displacement discontinuity across an interior
    # facet. It drives cohesive opening/sliding laws on dS(1).
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
    # Bond quality from nozzle-pressure penetration index:
    #   beta = 1 - exp(-phi)
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

    # Exponential damage driven by mode-I and mode-II displacement energies.
    delta_n_sq = delta_n_open * delta_n_open
    damage_arg_n = delta_n_sq / ufl.max_value(
        2.0 * g_i_c_interface / ufl.max_value(K_n, 1.0e-12), 1.0e-12
    )
    damage_arg_t = delta_t_sq / ufl.max_value(
        2.0 * g_ii_c_interface / ufl.max_value(K_t, 1.0e-12), 1.0e-12
    )
    damage = 1.0 - ufl.exp(-(damage_arg_n + damage_arg_t))
    damage = ufl.min_value(1.0, ufl.max_value(damage, 0.0))
    # Enforce damage irreversibility via per-cell history variable.
    damage = ufl.max_value(damage, ufl.max_value(materials.damage_max("+"), materials.damage_max("-")))

    K_n_res = residual_stiffness_ratio * K_n
    K_t_res = residual_stiffness_ratio * K_t

    # Cohesive traction law with:
    # - opening response in the normal direction,
    # - compressive contact-like penalty for negative normal jump,
    # - tangential traction with damage-softened stiffness.
    T_n = ((1.0 - damage) * K_n + damage * K_n_res) * delta_n_open + K_n * ufl.min_value(
        jump_n_scalar, 0.0
    )
    T_t = ((1.0 - damage) * K_t + damage * K_t_res) * delta_t
    T_cohesive = T_n * n("+") + T_t

    sigma_u = sigma_from_strain(epsilon(u) - materials.eps_vp, lam, mu)

    # Residual form:
    # - bulk internal virtual work,
    # - cohesive traction work on inter-layer facets,
    # - SIP terms on bonded interfaces and weak Dirichlet boundaries,
    # - body force loading.
    F = alpha * ufl.inner(sigma_u, epsilon(v)) * dx
    # Cohesive traction virtual work on inter-layer facets.
    F += ufl.inner(alpha_facet_cohesive * T_cohesive, ufl.jump(v)) * dS(1)
    # Intra-layer bonded interfaces: consistent + symmetric SIP couplings.
    F += -ufl.inner(alpha_facet * ufl.avg(sigma_u) * n("+"), ufl.jump(v)) * dS(2)
    F += -ufl.inner(alpha_facet * ufl.avg(sigma(v, lam, mu)) * n("+"), ufl.jump(u)) * dS(2)
    F += alpha_facet * gamma_bonded * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS(2)
    # Jump stabilization on non-cohesive interior facets only.
    # dS(1) excluded: cohesive law (K_n, K_t) provides interface coupling there.
    gamma_safe = 0.05 * ufl.avg(materials.E) / h
    F += alpha_facet * gamma_safe * ufl.inner(ufl.jump(u), ufl.jump(v)) * (dS(0) + dS(2))
    # Weak Dirichlet (Nitsche-like) boundary terms on ds(1).
    F += -alpha * ufl.inner(sigma_u * n, v) * ds(1)
    F += -alpha * ufl.inner(sigma(v, lam, mu) * n, u) * ds(1)
    F += alpha * gamma_boundary * ufl.inner(u, v) * ds(1)
    g_vec = ufl.as_vector([0.0, 0.0, -materials.rho_const * materials.g_const])
    F += -alpha * ufl.inner(g_vec, v) * dx

    J = ufl.derivative(F, u, ufl.TrialFunction(V))
    if return_ufl:
        return fem.form(F), fem.form(J), F, J
    return fem.form(F), fem.form(J)
