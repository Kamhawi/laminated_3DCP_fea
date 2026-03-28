"""Penalty/stiffness hierarchy extraction and diagnostics (Tests 2.2–2.4).

Extracts numerical values of cohesive stiffnesses (K_n, K_t), SIP penalties
(gamma_bonded, gamma_safe), and Nitsche penalty (gamma_boundary) on a
per-facet basis and checks the expected stiffness hierarchy.

Also provides an analytical K_n/K_min scaling analysis (Test 2.4) to
determine whether the K_min floor is a coarse-mesh artifact or structural.

All computations use raw DG0 arrays and mesh geometry — no UFL evaluation.
"""

import numpy as np
from dolfinx.cpp.mesh import h as _dolfinx_h
from mpi4py import MPI

from diagnostics.setup import prepare_state_at_time
from materials.time_models import (
    compute_bond_quality,
    compute_yield_stress_mpa,
    compute_poisson_ratio,
    compute_shear_modulus_mpa,
    compute_young_modulus_mpa,
    pa_to_mpa,
)


# ---------------------------------------------------------------------------
# Cell diameter computation
# ---------------------------------------------------------------------------

def compute_cell_diameters(msh):
    """Compute per-cell diameters matching ``ufl.CellDiameter``.

    Returns:
        np.ndarray of shape ``(n_local + n_ghost,)`` with cell diameters.
    """
    tdim = msh.topology.dim
    map_c = msh.topology.index_map(tdim)
    ncells = map_c.size_local + map_c.num_ghosts
    all_cells = np.arange(ncells, dtype=np.int32)
    return np.asarray(_dolfinx_h(msh._cpp_object, tdim, all_cells))


# ---------------------------------------------------------------------------
# Facet data extraction
# ---------------------------------------------------------------------------

def _get_facet_cell_pairs(msh, facet_indices):
    """Return (cell_plus, cell_minus) arrays for given interior facet indices.

    Each interior facet has exactly two adjacent cells.

    Returns:
        tuple of two int32 arrays, each of length ``len(facet_indices)``.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    n = len(facet_indices)
    cell_plus = np.empty(n, dtype=np.int32)
    cell_minus = np.empty(n, dtype=np.int32)
    for i, f in enumerate(facet_indices):
        cells = f_to_c.links(f)
        cell_plus[i] = cells[0]
        cell_minus[i] = cells[1] if len(cells) > 1 else cells[0]
    return cell_plus, cell_minus


def extract_interlayer_facet_data(state, t_val):
    """Extract per-facet penalty/stiffness data on inter-layer (dS(1)) facets.

    Reproduces the cohesive parameter formulas from ``weak_form.py`` using
    pure NumPy on the DG0 material arrays.

    Args:
        state: DiagnosticState with materials updated at ``t_val``.
        t_val: Simulation time (for reference only; materials must already
            be updated via ``prepare_state_at_time``).

    Returns:
        dict with per-facet arrays and config values.
    """
    msh = state.msh
    cfg = state.cfg
    interface_cfg = cfg["interface"]
    material_cfg = cfg["material"]

    # Config scalars.
    gamma_bonded_mult = interface_cfg["gamma_bonded_mult"]
    k_min_mult = interface_cfg.get("K_min_mult", 0.1)
    g_i_c = interface_cfg["G_Ic"]
    g_ii_c = interface_cfg["G_IIc"]
    nozzle_pressure = interface_cfg["nozzle_pressure"]
    residual_ratio = interface_cfg.get("residual_stiffness_ratio", 0.01)
    tau_0 = material_cfg["tau_0"]
    a_thix = material_cfg["A_thix"]

    # Filter for tag==1 facets.
    tags = state.interior_facet_tags
    mask = tags.values == 1
    facet_indices = tags.indices[mask]

    if len(facet_indices) == 0:
        return {"n_facets": 0}

    cell_plus, cell_minus = _get_facet_cell_pairs(msh, facet_indices)

    # Per-cell DG0 arrays (owned + ghost, already scattered).
    E_arr = np.asarray(state.materials.E.x.array)
    sigma_y_arr = np.asarray(state.materials.sigma_y.x.array)
    tau_y_arr = np.asarray(state.materials.tau_y.x.array)
    birth_arr = state.birth_times_dolfinx
    diameters = compute_cell_diameters(msh)

    # Per-facet cell values.
    E_plus = E_arr[cell_plus]
    E_minus = E_arr[cell_minus]
    sigma_y_plus = sigma_y_arr[cell_plus]
    sigma_y_minus = sigma_y_arr[cell_minus]
    tau_y_plus = tau_y_arr[cell_plus]
    tau_y_minus = tau_y_arr[cell_minus]
    h_plus = diameters[cell_plus]
    h_minus = diameters[cell_minus]
    birth_plus = birth_arr[cell_plus]
    birth_minus = birth_arr[cell_minus]

    # Facet-averaged quantities (matching ufl.avg).
    avg_E = 0.5 * (E_plus + E_minus)
    avg_sigma_y = 0.5 * (sigma_y_plus + sigma_y_minus)
    avg_tau_y = 0.5 * (tau_y_plus + tau_y_minus)
    h_avg = 0.5 * (h_plus + h_minus)

    # Bond quality (reproducing weak_form.py lines 111-118).
    t_open = np.abs(birth_plus - birth_minus)
    tau_sub = tau_0 + a_thix * t_open  # Pa
    phi = nozzle_pressure / np.maximum(tau_sub, 1.0e-12)
    beta_bond = 1.0 - np.exp(-phi)
    beta_bond = np.clip(beta_bond, 1.0e-8, 1.0)

    # Cohesive energies.
    g_i_c_eff = np.maximum(beta_bond * g_i_c, 1.0e-12)
    g_ii_c_eff = np.maximum(beta_bond * g_ii_c, 1.0e-12)

    # Cohesive strengths (avg of DG0 yield stresses, scaled by bond quality).
    # Note: sigma_y is in MPa (= N/mm^2), same unit system as weak_form.
    sigma_bond = beta_bond * avg_sigma_y
    tau_bond = beta_bond * avg_tau_y

    # Cohesive stiffnesses.
    K_n_raw = sigma_bond**2 / (2.0 * g_i_c_eff)
    K_t_raw = tau_bond**2 / (2.0 * g_ii_c_eff)
    K_min = k_min_mult * avg_E / h_avg
    K_n = np.maximum(K_n_raw, K_min)
    K_t = np.maximum(K_t_raw, K_min)

    # Penalty parameters.
    gamma_bonded = gamma_bonded_mult * avg_E / h_avg
    gamma_safe = 0.05 * avg_E / h_avg

    return {
        "n_facets": len(facet_indices),
        "facet_indices": facet_indices,
        "cell_plus": cell_plus,
        "cell_minus": cell_minus,
        "avg_E": avg_E,
        "h_avg": h_avg,
        "beta_bond": beta_bond,
        "sigma_bond": sigma_bond,
        "tau_bond": tau_bond,
        "K_n_raw": K_n_raw,
        "K_t_raw": K_t_raw,
        "K_min": K_min,
        "K_n": K_n,
        "K_t": K_t,
        "gamma_bonded": gamma_bonded,
        "gamma_safe": gamma_safe,
        # Config echoed for reference.
        "gamma_bonded_mult": gamma_bonded_mult,
        "k_min_mult": k_min_mult,
        "residual_ratio": residual_ratio,
    }


def extract_intralayer_facet_data(state, t_val):
    """Extract per-facet penalty data on intra-layer bonded (dS(2)) facets.

    Args:
        state: DiagnosticState with materials updated at ``t_val``.
        t_val: Simulation time (reference).

    Returns:
        dict with per-facet arrays.
    """
    msh = state.msh
    gamma_bonded_mult = state.cfg["interface"]["gamma_bonded_mult"]

    tags = state.interior_facet_tags
    mask = tags.values == 2
    facet_indices = tags.indices[mask]

    if len(facet_indices) == 0:
        return {"n_facets": 0}

    cell_plus, cell_minus = _get_facet_cell_pairs(msh, facet_indices)

    E_arr = np.asarray(state.materials.E.x.array)
    diameters = compute_cell_diameters(msh)

    avg_E = 0.5 * (E_arr[cell_plus] + E_arr[cell_minus])
    h_avg = 0.5 * (diameters[cell_plus] + diameters[cell_minus])

    gamma_bonded = gamma_bonded_mult * avg_E / h_avg
    gamma_safe = 0.05 * avg_E / h_avg

    return {
        "n_facets": len(facet_indices),
        "facet_indices": facet_indices,
        "avg_E": avg_E,
        "h_avg": h_avg,
        "gamma_bonded": gamma_bonded,
        "gamma_safe": gamma_safe,
    }


# ---------------------------------------------------------------------------
# Hierarchy / ratio checks
# ---------------------------------------------------------------------------

def _stats(arr):
    """Return min/max/mean dict for a 1-D array."""
    if len(arr) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def check_stiffness_hierarchy(interlayer_data, intralayer_data, verbose=True):
    """Check ratio-based stiffness hierarchy on interior facets.

    On dS(1) inter-layer facets:
        - gamma_safe / K_n, gamma_safe / K_t  should be << 1
        - K_n / gamma_bonded, K_t / gamma_bonded  should be < 1

    On dS(2) intra-layer bonded facets:
        - gamma_safe / gamma_bonded  should be << 1

    Args:
        interlayer_data: From ``extract_interlayer_facet_data``.
        intralayer_data: From ``extract_intralayer_facet_data``.
        verbose: Print results on rank 0.

    Returns:
        dict with ratio statistics and pass/fail per check.
    """
    result = {}

    # --- dS(1) inter-layer checks ---
    # gamma_safe is excluded from dS(1) — cohesive law (K_n, K_t) is the
    # sole interface coupling.  Only check K_n, K_t scale vs gamma_bonded.
    if interlayer_data["n_facets"] > 0:
        K_n = interlayer_data["K_n"]
        K_t = interlayer_data["K_t"]
        gs = interlayer_data["gamma_safe"]
        gb = interlayer_data["gamma_bonded"]

        ratio_Kn_gb = K_n / np.maximum(gb, 1e-30)
        ratio_Kt_gb = K_t / np.maximum(gb, 1e-30)

        result["dS1"] = {
            "n_facets": interlayer_data["n_facets"],
            "K_n_over_gamma_bonded": _stats(ratio_Kn_gb),
            "K_t_over_gamma_bonded": _stats(ratio_Kt_gb),
            "abs_gamma_safe": _stats(gs),
            "abs_K_min": _stats(interlayer_data["K_min"]),
            "abs_K_n": _stats(K_n),
            "abs_K_t": _stats(K_t),
            "abs_gamma_bonded": _stats(gb),
            "beta_bond": _stats(interlayer_data["beta_bond"]),
            # Pass criteria.
            "Kn_gb_ok": bool(np.max(ratio_Kn_gb) < 1.0),
            "Kt_gb_ok": bool(np.max(ratio_Kt_gb) < 1.0),
        }
    else:
        result["dS1"] = {"n_facets": 0}

    # --- dS(2) intra-layer checks ---
    if intralayer_data["n_facets"] > 0:
        gs2 = intralayer_data["gamma_safe"]
        gb2 = intralayer_data["gamma_bonded"]
        ratio_gs_gb = gs2 / np.maximum(gb2, 1e-30)

        result["dS2"] = {
            "n_facets": intralayer_data["n_facets"],
            "gamma_safe_over_gamma_bonded": _stats(ratio_gs_gb),
            "abs_gamma_safe": _stats(gs2),
            "abs_gamma_bonded": _stats(gb2),
            "gs_gb_ok": bool(np.max(ratio_gs_gb) < 0.1),
        }
    else:
        result["dS2"] = {"n_facets": 0}

    if verbose:
        _print_hierarchy(result)

    return result


def _print_hierarchy(result):
    """Pretty-print hierarchy results."""
    print("\n  Stiffness hierarchy (ratio-based checks)")

    ds1 = result.get("dS1", {})
    if ds1.get("n_facets", 0) > 0:
        print(f"\n  dS(1) inter-layer facets: {ds1['n_facets']}")
        print(f"    {'Metric':<30s}  {'min':>10s}  {'max':>10s}  {'mean':>10s}  {'Status':>8s}")
        print(f"    {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

        for key, label, ok_key in [
            ("K_n_over_gamma_bonded", "K_n / gamma_bonded", "Kn_gb_ok"),
            ("K_t_over_gamma_bonded", "K_t / gamma_bonded", "Kt_gb_ok"),
        ]:
            s = ds1[key]
            status = "OK" if ds1[ok_key] else "WARN"
            print(f"    {label:<30s}  {s['min']:10.4f}  {s['max']:10.4f}  "
                  f"{s['mean']:10.4f}  {status:>8s}")

        print(f"\n    Absolute values:")
        for key, label in [
            ("abs_gamma_safe", "gamma_safe"),
            ("abs_K_min", "K_min"),
            ("abs_K_n", "K_n"),
            ("abs_K_t", "K_t"),
            ("abs_gamma_bonded", "gamma_bonded"),
        ]:
            s = ds1[key]
            print(f"      {label:<20s}  min={s['min']:.4e}  max={s['max']:.4e}  "
                  f"mean={s['mean']:.4e}")

        s_beta = ds1["beta_bond"]
        print(f"      {'beta_bond':<20s}  min={s_beta['min']:.4f}  "
              f"max={s_beta['max']:.4f}  mean={s_beta['mean']:.4f}")

    ds2 = result.get("dS2", {})
    if ds2.get("n_facets", 0) > 0:
        print(f"\n  dS(2) intra-layer facets: {ds2['n_facets']}")
        s = ds2["gamma_safe_over_gamma_bonded"]
        status = "OK" if ds2["gs_gb_ok"] else "WARN"
        print(f"    gamma_safe / gamma_bonded:  min={s['min']:.4f}  "
              f"max={s['max']:.4f}  mean={s['mean']:.4f}  {status}")


# ---------------------------------------------------------------------------
# Dirichlet penalty check
# ---------------------------------------------------------------------------

def check_dirichlet_penalty(state, t_values, interlayer_data=None,
                            verbose=True):
    """Evaluate Dirichlet penalty magnitudes and conditioning ratio.

    For each time point, computes:
        gamma_boundary_abs = gamma_boundary_mult * E_cell / h_cell

    The key conditioning metric is:
        max(gamma_boundary_abs) / min(K_n)
    across boundary and inter-layer facets.

    Args:
        state: DiagnosticState.
        t_values: List of simulation times.
        interlayer_data: Optional pre-computed interlayer data (from
            ``extract_interlayer_facet_data`` at a single time). If None,
            the conditioning ratio is skipped.
        verbose: Print results on rank 0.

    Returns:
        dict with per-time results.
    """
    msh = state.msh
    comm = state.comm
    gamma_boundary_mult = state.cfg["interface"]["gamma_boundary_mult"]

    # Identify boundary facets.
    dirichlet_tags = state.dirichlet_tags
    if dirichlet_tags is None or len(dirichlet_tags.indices) == 0:
        if verbose and comm.rank == 0:
            print("\n  Dirichlet penalty check: N/A (no boundary facets)")
        return {"n_boundary_facets": 0}

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    boundary_facets = dirichlet_tags.indices[dirichlet_tags.values == 1]
    # Exterior facets have one adjacent cell.
    boundary_cells = np.array(
        [f_to_c.links(f)[0] for f in boundary_facets], dtype=np.int32,
    )

    diameters = compute_cell_diameters(msh)

    per_time = []
    for t_val in t_values:
        prepare_state_at_time(state, float(t_val), u_mode="zero")

        E_arr = np.asarray(state.materials.E.x.array)
        E_boundary = E_arr[boundary_cells]
        h_boundary = diameters[boundary_cells]

        gamma_abs = gamma_boundary_mult * E_boundary / h_boundary

        # Conditioning ratio: how much larger is the boundary penalty
        # than the weakest cohesive stiffness?
        cond_ratio = None
        if interlayer_data is not None and interlayer_data["n_facets"] > 0:
            min_Kn_local = float(np.min(interlayer_data["K_n"]))
            max_gamma_local = float(np.max(gamma_abs))
            # Global reduction for MPI.
            min_Kn = comm.allreduce(min_Kn_local, op=MPI.MIN)
            max_gamma = comm.allreduce(max_gamma_local, op=MPI.MAX)
            cond_ratio = max_gamma / max(min_Kn, 1e-30)

        entry = {
            "t_val": float(t_val),
            "n_boundary_facets": int(comm.allreduce(len(boundary_facets), op=MPI.SUM)),
            "gamma_abs": _stats(gamma_abs),
            "E_boundary": _stats(E_boundary),
            "h_boundary": _stats(h_boundary),
            "conditioning_ratio": cond_ratio,
        }
        per_time.append(entry)

    result = {
        "gamma_boundary_mult": gamma_boundary_mult,
        "dimensionless_ratio": gamma_boundary_mult,
        "in_range_10_1000": 10.0 <= gamma_boundary_mult <= 1000.0,
        "per_time": per_time,
    }

    if verbose and comm.rank == 0:
        _print_dirichlet(result)

    return result


def _print_dirichlet(result):
    """Pretty-print Dirichlet penalty results."""
    mult = result["gamma_boundary_mult"]
    in_range = result["in_range_10_1000"]
    status = "PASS" if in_range else "WARN"
    print(f"\n  Dirichlet penalty check")
    print(f"    gamma_boundary_mult = {mult:.1f}  (dimensionless)  {status}")

    for entry in result["per_time"]:
        t = entry["t_val"]
        ga = entry["gamma_abs"]
        eb = entry["E_boundary"]
        cr = entry["conditioning_ratio"]
        cr_str = f"{cr:.2e}" if cr is not None else "N/A"
        print(f"\n    t = {t:.2f} s  ({entry['n_boundary_facets']} boundary facets)")
        print(f"      gamma_boundary_abs:  [{ga['min']:.4e}, {ga['max']:.4e}]  "
              f"mean={ga['mean']:.4e}")
        print(f"      E on boundary cells: [{eb['min']:.4e}, {eb['max']:.4e}]  "
              f"mean={eb['mean']:.4e}")
        print(f"      max(gamma_bdy) / min(K_n) = {cr_str}")


# ---------------------------------------------------------------------------
# Analytical K_n / K_min scaling analysis (Test 2.4)
# ---------------------------------------------------------------------------

def compute_kn_kmin_scaling(cfg, h_values, ages, t_open=0.0, verbose=True):
    """Analytical K_n/K_min scaling check across mesh sizes and material ages.

    Computes K_n_physical (mesh-independent) and K_min (mesh-dependent) for
    each (h, age) combination to determine whether the K_min floor dominates
    on the production mesh or only on coarse diagnostic meshes.

    Args:
        cfg: Full config dict (with "material", "hardening", "interface" keys).
        h_values: Array of cell diameters [mm] to evaluate.
        ages: Array of material ages [s] to evaluate.
        t_open: Interface opening time [s] for beta_bond computation.
        verbose: Print results on rank 0.

    Returns:
        dict with per-age results and crossover diameters.
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

    ages = np.asarray(ages, dtype=float)
    h_values = np.asarray(h_values, dtype=float)

    # Bond quality at the given t_open.
    beta = float(compute_bond_quality(
        t_open, tau_0_pa, a_thix_pa_s, nozzle_pressure_pa,
    ))
    beta = np.clip(beta, 1e-8, 1.0)

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

        # Physical cohesive stiffnesses (mesh-independent).
        K_n_phys = beta * sigma_y**2 / (2.0 * g_i_c)
        K_t_phys = beta * tau_y**2 / (2.0 * g_ii_c)

        # Per-h values.
        K_min_arr = k_min_mult * E / h_values
        ratio_n = K_n_phys / np.maximum(K_min_arr, 1e-30)
        ratio_t = K_t_phys / np.maximum(K_min_arr, 1e-30)

        # Crossover cell diameter: h where K_n_phys = K_min.
        h_star_n = k_min_mult * E * 2.0 * g_i_c / max(beta * sigma_y**2, 1e-30)
        h_star_t = k_min_mult * E * 2.0 * g_ii_c / max(beta * tau_y**2, 1e-30)

        rows.append({
            "age": float(age),
            "tau_y": tau_y,
            "sigma_y": sigma_y,
            "E": E,
            "K_n_phys": K_n_phys,
            "K_t_phys": K_t_phys,
            "K_min_per_h": {float(h): float(km) for h, km in zip(h_values, K_min_arr)},
            "ratio_n_per_h": {float(h): float(r) for h, r in zip(h_values, ratio_n)},
            "ratio_t_per_h": {float(h): float(r) for h, r in zip(h_values, ratio_t)},
            "h_star_n": h_star_n,
            "h_star_t": h_star_t,
        })

    result = {
        "beta_bond": beta,
        "t_open": float(t_open),
        "k_min_mult": k_min_mult,
        "h_values": h_values.tolist(),
        "rows": rows,
    }

    if verbose:
        _print_kn_kmin_scaling(result)

    return result


def _print_kn_kmin_scaling(result):
    """Pretty-print K_n/K_min scaling table."""
    beta = result["beta_bond"]
    t_open = result["t_open"]
    h_values = result["h_values"]
    rows = result["rows"]

    print(f"\n  K_n/K_min scaling analysis (analytical)")
    print(f"    beta_bond = {beta:.4f}  (t_open = {t_open:.1f} s)")
    print(f"    K_min_mult = {result['k_min_mult']}")

    # Header row.
    h_cols = "  ".join(f"h={h:g}" for h in h_values)
    print(f"\n    {'Age [s]':>8s}  {'tau_y':>8s}  {'E':>8s}  "
          f"{'K_n_phys':>10s}  {'h*_n [mm]':>9s}  {h_cols}")
    print(f"    {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*9}  "
          f"{'  '.join('-' * max(5, len(f'h={h:g}')) for h in h_values)}")

    for row in rows:
        ratios = "  ".join(
            f"{row['ratio_n_per_h'][h]:>{max(5, len(f'h={h:g}'))}.3f}"
            for h in h_values
        )
        print(f"    {row['age']:8.0f}  {row['tau_y']:8.4f}  {row['E']:8.2f}  "
              f"{row['K_n_phys']:10.4e}  {row['h_star_n']:9.1f}  {ratios}")

    print(f"\n    Values show K_n_physical / K_min ratio at each (age, h).")
    print(f"    h*_n = crossover cell diameter where K_n_phys = K_min.")
    print(f"    Ratio < 1 means K_min floor dominates (artificial stiffness).")
