"""NumPy-based damage_max history update for inter-layer facets.

Shared by the production time stepper (``solver/time_stepper.py``) and
the diagnostic time stepper (``diagnostics/cohesive_analysis.py``).
"""

import numpy as np


def update_damage_max_numpy(
    u,
    materials,
    cell_to_dofs,
    birth_times,
    cfg,
    il_cell_plus,
    il_cell_minus,
    il_facet_normals,
    il_h_cells,
    active_mask=None,
):
    """Update ``materials.damage_max`` from the current displacement state.

    Computes per-facet damage on inter-layer facets using the same
    exponential damage law as ``physics/weak_form.py`` (lines 132-141),
    then takes the per-cell maximum over all adjacent inter-layer facets.

    Args:
        u: Displacement fem.Function (DG1 vector).
        materials: MaterialStateManager with ``damage_max``, ``sigma_y``,
            ``tau_y``, ``E`` DG0 fields.
        cell_to_dofs: (n_cells, dofs_per_cell) int array mapping cells
            to interleaved vector DOF indices.
        birth_times: Per-cell birth time array (DOLFINx ordering).
        cfg: Full simulation config dict.
        il_cell_plus: (n_il,) int32 array — "+" cell for each inter-layer facet.
        il_cell_minus: (n_il,) int32 array — "-" cell for each inter-layer facet.
        il_facet_normals: (n_il, 3) float64 — unit normals per facet.
        il_h_cells: (n_cells,) float64 — cell diameters.
        active_mask: Optional bool array (n_cells,).  If given, inactive
            cells have their ``damage_max`` zeroed after the update.
    """
    n_il = len(il_cell_plus)
    if n_il == 0:
        return

    u_arr = u.x.array
    n_comps = 3

    # Vectorized per-cell average displacement for both sides.
    dofs_plus = cell_to_dofs[il_cell_plus]
    dofs_minus = cell_to_dofs[il_cell_minus]
    u_plus = u_arr[dofs_plus].reshape(n_il, -1, n_comps).mean(axis=1)
    u_minus = u_arr[dofs_minus].reshape(n_il, -1, n_comps).mean(axis=1)
    jump_vec = u_plus - u_minus

    # Normal and tangential jumps using actual facet normals.
    jump_n = np.sum(jump_vec * il_facet_normals, axis=1)
    delta_n_open = np.maximum(jump_n, 0.0)
    jump_t_vec = jump_vec - jump_n[:, None] * il_facet_normals
    delta_t_sq = np.sum(jump_t_vec ** 2, axis=1)

    # Per-facet averaged material properties.
    sigma_y_arr = materials.sigma_y.x.array
    tau_y_arr = materials.tau_y.x.array
    e_arr = materials.E.x.array

    sigma_y_avg = 0.5 * (sigma_y_arr[il_cell_plus] + sigma_y_arr[il_cell_minus])
    tau_y_avg = 0.5 * (tau_y_arr[il_cell_plus] + tau_y_arr[il_cell_minus])
    e_avg = 0.5 * (e_arr[il_cell_plus] + e_arr[il_cell_minus])

    # Bond quality (same formula as weak_form.py lines 111-118).
    tau_0_pa = float(cfg["material"]["tau_0"])
    a_thix_pa_s = float(cfg["material"]["A_thix"])
    nozzle_pressure_pa = float(cfg["interface"]["nozzle_pressure"])
    g_i_c = float(cfg["interface"]["G_Ic"])
    g_ii_c = float(cfg["interface"]["G_IIc"])
    k_min_mult = float(cfg["interface"].get("K_min_mult", 0.1))

    bt_plus = birth_times[il_cell_plus]
    bt_minus = birth_times[il_cell_minus]
    t_open = np.abs(bt_plus - bt_minus)
    tau_sub = tau_0_pa + a_thix_pa_s * t_open
    phi = nozzle_pressure_pa / np.maximum(tau_sub, 1e-12)
    beta = np.clip(1.0 - np.exp(-phi), 1e-8, 1.0)

    G_Ic_eff = np.maximum(beta * g_i_c, 1e-12)
    G_IIc_eff = np.maximum(beta * g_ii_c, 1e-12)

    # Stiffness (same as weak_form.py lines 123-130).
    K_n = beta * sigma_y_avg ** 2 / (2.0 * G_Ic_eff)
    K_t = beta * tau_y_avg ** 2 / (2.0 * G_IIc_eff)
    h_avg = 0.5 * (il_h_cells[il_cell_plus] + il_h_cells[il_cell_minus])
    K_min = k_min_mult * e_avg / h_avg
    K_n = np.maximum(K_n, K_min)
    K_t = np.maximum(K_t, K_min)

    # Damage (same as weak_form.py lines 132-141).
    damage_arg_n = delta_n_open ** 2 / np.maximum(
        2.0 * G_Ic_eff / np.maximum(K_n, 1e-12), 1e-12
    )
    damage_arg_t = delta_t_sq / np.maximum(
        2.0 * G_IIc_eff / np.maximum(K_t, 1e-12), 1e-12
    )
    facet_damage = np.clip(1.0 - np.exp(-(damage_arg_n + damage_arg_t)), 0.0, 1.0)

    # Per-cell max: accumulate over inter-layer facets.
    dmg_arr = materials.damage_max.x.array
    np.maximum.at(dmg_arr, il_cell_plus, facet_damage)
    np.maximum.at(dmg_arr, il_cell_minus, facet_damage)

    if active_mask is not None:
        dmg_arr[~active_mask] = 0.0

    materials.damage_max.x.scatter_forward()
