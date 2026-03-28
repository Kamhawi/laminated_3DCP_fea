"""Analytical traction-separation law (TSL) analysis (Test 2.5).

Computes mode-I TSL curves from the exponential damage formulation in
``physics/weak_form.py`` and quantifies distortion caused by the K_min
numerical stability floor.

All computations are pure NumPy — no FEA or UFL required.
"""

import numpy as np

from materials.time_models import (
    compute_bond_quality,
    compute_poisson_ratio,
    compute_shear_modulus_mpa,
    compute_young_modulus_mpa,
    compute_yield_stress_mpa,
    pa_to_mpa,
)


# ---------------------------------------------------------------------------
# Single TSL curve evaluation
# ---------------------------------------------------------------------------

def compute_tsl_curve(K_n, G_Ic_eff, r_s=0.05, n_points=500):
    """Evaluate the mode-I traction-separation law analytically.

    Reproduces the exponential damage formulation from ``weak_form.py``
    (lines 132–154) for pure mode-I opening (δ_t = 0).

    Math::

        ξ(δ) = δ² K_n / (2 G_Ic_eff)
        d(δ) = 1 - exp(-ξ)
        T_n(δ) = K_n [r_s + (1 - r_s) exp(-ξ)] δ

    Args:
        K_n: Normal cohesive stiffness [MPa/mm = N/mm³].
        G_Ic_eff: Effective mode-I fracture energy (= beta * G_Ic) [N/mm].
        r_s: Residual stiffness ratio (``residual_stiffness_ratio``).
        n_points: Number of evaluation points.

    Returns:
        dict with curve arrays and scalar metrics.
    """
    K_n = float(K_n)
    G_Ic_eff = max(float(G_Ic_eff), 1e-30)
    r_s = float(r_s)

    # Approximate peak location (exact for r_s=0).
    delta_peak_approx = np.sqrt(G_Ic_eff / max(K_n, 1e-30))
    delta_max = 5.0 * delta_peak_approx
    delta = np.linspace(0, delta_max, n_points)

    alpha = K_n / (2.0 * G_Ic_eff)
    xi = alpha * delta ** 2
    exp_neg_xi = np.exp(-xi)

    damage = 1.0 - exp_neg_xi
    traction = K_n * (r_s + (1.0 - r_s) * exp_neg_xi) * delta

    # Peak traction (numerical).
    i_peak = np.argmax(traction)
    peak_traction = float(traction[i_peak])
    delta_at_peak = float(delta[i_peak])

    # Critical displacement: δ where damage first reaches 0.99.
    mask_99 = damage >= 0.99
    delta_c_99 = float(delta[mask_99][0]) if np.any(mask_99) else float(delta[-1])

    # Fracture energy check: area under T_n(δ) curve.
    G_Ic_numerical = float(np.trapz(traction, delta))

    # Tangent stiffness array (numerical central differences).
    tangent = np.gradient(traction, delta)
    min_tangent = float(np.min(tangent[i_peak:]))  # Softening branch only.

    return {
        "delta": delta,
        "traction": traction,
        "damage": damage,
        "K_n": K_n,
        "G_Ic_eff": G_Ic_eff,
        "r_s": r_s,
        "peak_traction": peak_traction,
        "delta_at_peak": delta_at_peak,
        "delta_c_99": delta_c_99,
        "G_Ic_numerical": G_Ic_numerical,
        "min_tangent": min_tangent,
    }


# ---------------------------------------------------------------------------
# TSL comparison: physical vs floor-dominated
# ---------------------------------------------------------------------------

def compute_tsl_comparison(cfg, ages, h_values, t_open=0.0, verbose=True):
    """Compare physical and floor-dominated TSL at representative conditions.

    For each (age, h) combination, computes two TSL curves:
    - Physical: K_n = K_n_phys (ignoring floor)
    - Actual: K_n = max(K_n_phys, K_min) (with floor)

    Args:
        cfg: Full config dict.
        ages: Array of material ages [s].
        h_values: Array of cell diameters [mm].
        t_open: Interface opening time [s] for beta_bond.
        verbose: Print results.

    Returns:
        dict with per-(age, h) comparison metrics.
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

    # Bond quality.
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

        # Physical K_n (mesh-independent).
        K_n_phys = beta * sigma_y ** 2 / (2.0 * g_i_c)

        for h in h_values:
            K_min = k_min_mult * E / h
            K_n_actual = max(K_n_phys, K_min)
            regime = "PHYS" if K_n_phys >= K_min else "FLOOR"

            # TSL curves.
            tsl_phys = compute_tsl_curve(K_n_phys, G_Ic_eff, r_s)
            tsl_actual = compute_tsl_curve(K_n_actual, G_Ic_eff, r_s)

            peak_ratio = (
                tsl_actual["peak_traction"] / max(tsl_phys["peak_traction"], 1e-30)
            )
            delta_c_ratio = (
                tsl_actual["delta_c_99"] / max(tsl_phys["delta_c_99"], 1e-30)
            )
            G_Ic_ratio = (
                tsl_actual["G_Ic_numerical"] / max(tsl_phys["G_Ic_numerical"], 1e-30)
            )

            rows.append({
                "age": float(age),
                "h": float(h),
                "E": E,
                "sigma_y": sigma_y,
                "tau_y": tau_y,
                "K_n_phys": K_n_phys,
                "K_min": K_min,
                "K_n_actual": K_n_actual,
                "regime": regime,
                "peak_traction_phys": tsl_phys["peak_traction"],
                "peak_traction_actual": tsl_actual["peak_traction"],
                "peak_ratio": peak_ratio,
                "delta_c_phys": tsl_phys["delta_c_99"],
                "delta_c_actual": tsl_actual["delta_c_99"],
                "delta_c_ratio": delta_c_ratio,
                "G_Ic_numerical_phys": tsl_phys["G_Ic_numerical"],
                "G_Ic_numerical_actual": tsl_actual["G_Ic_numerical"],
                "G_Ic_ratio": G_Ic_ratio,
                "min_tangent_phys": tsl_phys["min_tangent"],
                "min_tangent_actual": tsl_actual["min_tangent"],
            })

    result = {
        "beta_bond": beta,
        "t_open": float(t_open),
        "G_Ic": g_i_c,
        "G_Ic_eff": G_Ic_eff,
        "k_min_mult": k_min_mult,
        "r_s": r_s,
        "rows": rows,
    }

    if verbose:
        _print_tsl_comparison(result)

    return result


def _print_tsl_comparison(result):
    """Print compact TSL comparison table."""
    print(f"\n  TSL Floor Distortion Analysis")
    print(f"    beta_bond = {result['beta_bond']:.4f}  "
          f"(t_open = {result['t_open']:.1f} s)")
    print(f"    G_Ic = {result['G_Ic']:.4f} N/mm  "
          f"G_Ic_eff = {result['G_Ic_eff']:.4f} N/mm  "
          f"K_min_mult = {result['k_min_mult']}  "
          f"r_s = {result['r_s']}")

    rows = result["rows"]
    if not rows:
        return

    print(f"\n    {'age [s]':>8s}  {'h [mm]':>8s}  {'K_n_phys':>10s}  "
          f"{'K_min':>10s}  {'regime':>6s}  {'peak_ratio':>10s}  "
          f"{'δ_c_ratio':>10s}  {'G_Ic_ratio':>10s}")
    print(f"    {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*6}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}")

    for row in rows:
        print(f"    {row['age']:8.0f}  {row['h']:8.1f}  "
              f"{row['K_n_phys']:10.4e}  {row['K_min']:10.4e}  "
              f"{row['regime']:>6s}  {row['peak_ratio']:9.1f}x  "
              f"{row['delta_c_ratio']:9.3f}x  {row['G_Ic_ratio']:9.3f}x")

    # Physical TSL reference values.
    print(f"\n    Physical TSL reference (K_n = K_n_phys, no floor):")
    for row in rows:
        print(f"      age={row['age']:.0f}s: T_peak={row['peak_traction_phys']:.4e} MPa  "
              f"δ_c={row['delta_c_phys']:.4e} mm  "
              f"G_Ic_num={row['G_Ic_numerical_phys']:.4e} N/mm")


# ---------------------------------------------------------------------------
# Fix 2 recommendation
# ---------------------------------------------------------------------------

def compute_fix2_recommendations(comparison_result, sweep_result=None):
    """Synthesize TSL comparison and optional sweep into a direction.

    Args:
        comparison_result: From ``compute_tsl_comparison``.
        sweep_result: Optional dict from ``run_penalty_sweep`` for K_min_mult
            (Test 2.1c). If None, stability assessment is skipped.

    Returns:
        dict with recommendation and reasoning.
    """
    rows = comparison_result.get("rows", [])
    if not rows:
        return {"direction": "INSUFFICIENT_DATA", "rationale": "No TSL rows."}

    peak_ratios = [r["peak_ratio"] for r in rows]
    floor_count = sum(1 for r in rows if r["regime"] == "FLOOR")
    floor_fraction = floor_count / len(rows)
    max_peak_ratio = max(peak_ratios)
    median_peak_ratio = float(np.median(peak_ratios))

    distortion_significant = median_peak_ratio > 1.5

    # Sweep stability assessment (provisional).
    sweep_info = {}
    if sweep_result is not None:
        sweep_entries = sweep_result.get("sweep", [])
        for entry in sweep_entries:
            val = entry["value"]
            sweep_info[val] = entry["converged"]

    # Determine direction.
    if not distortion_significant:
        direction = "NONE"
        rationale = (
            f"Floor distortion is minor (median peak ratio = {median_peak_ratio:.1f}x). "
            f"No fix needed."
        )
    elif sweep_info:
        # Check if low K_min_mult values converge.
        low_vals_stable = all(
            sweep_info.get(v, False) for v in [0.001, 0.01]
            if v in sweep_info
        )
        if low_vals_stable:
            direction = "A"
            rationale = (
                f"Floor distortion significant (median {median_peak_ratio:.1f}x, "
                f"max {max_peak_ratio:.1f}x). "
                f"K_min_mult reduction converges provisionally — try Option A."
            )
        else:
            direction = "B_or_C"
            rationale = (
                f"Floor distortion significant (median {median_peak_ratio:.1f}x) "
                f"but low K_min_mult values fail to converge. "
                f"Evaluate Options B (parameter adjustment) or C (reformulation)."
            )
    else:
        direction = "A_tentative"
        rationale = (
            f"Floor distortion significant (median {median_peak_ratio:.1f}x, "
            f"max {max_peak_ratio:.1f}x). "
            f"No sweep data available — run full Phase 2 to assess stability."
        )

    result = {
        "direction": direction,
        "rationale": rationale,
        "distortion_significant": distortion_significant,
        "median_peak_ratio": median_peak_ratio,
        "max_peak_ratio": max_peak_ratio,
        "floor_fraction": floor_fraction,
        "sweep_provisional": sweep_info if sweep_info else None,
    }

    return result


def _print_fix2_recommendations(rec):
    """Print Fix 2 recommendation block."""
    print(f"\n  Fix 2 Analysis:")
    sig = "SIGNIFICANT" if rec["distortion_significant"] else "MINOR"
    print(f"    Floor distortion:          {sig} "
          f"(median {rec['median_peak_ratio']:.1f}x, "
          f"max {rec['max_peak_ratio']:.1f}x peak traction)")
    print(f"    Floor-dominated fraction:  "
          f"{rec['floor_fraction']:.0%} of (age, h) points")

    if rec["sweep_provisional"] is not None:
        print(f"    Stability (provisional*):")
        for val in sorted(rec["sweep_provisional"]):
            status = "YES" if rec["sweep_provisional"][val] else "NO"
            print(f"      K_min_mult={val:<8g} conv: {status}")
        print(f"    * Gravity-only on small mesh (trivial load case).")
        print(f"      Real validation requires interface opening.")
    else:
        print(f"    Stability:                 No sweep data (run without --skip-sweep)")

    print(f"    Direction:                 {rec['direction']}")
    print(f"    Rationale:                 {rec['rationale']}")
