# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Material-state storage and constitutive local updates.

This module contains:
1. `MaterialStateManager` for DG0 material/property fields and reusable arrays.
2. `update_perzyna_state_cellwise` for explicit cell-wise J2 Perzyna updates.

The implementation preserves the original constitutive logic and only organizes
it into reusable components.

Physics:
    The constitutive update follows a small-strain J2-Perzyna viscoplastic
    model with an elastic predictor and explicit viscoplastic correction.
"""

import logging

import numpy as np
from dolfinx import fem

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when numba is unavailable.
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False

from .time_models import (
    compute_poisson_ratio,
    compute_shear_modulus_mpa,
    compute_viscosity_mpa_s,
    compute_yield_stress_mpa,
    compute_young_modulus_mpa,
    kg_m3_to_ns2_mm4,
    pa_to_mpa,
)

logger = logging.getLogger(__name__)


class MaterialStateManager:
    """Manage cell-wise material fields and age-dependent updates.

    Args:
        msh: Distributed DOLFINx mesh.
        material_cfg: Material/rheology section from configuration.
        hardening_cfg: Hardening/time-evolution section from configuration.
        birth_times_dolfinx: Cell birth times in DOLFINx local+ghost ordering.

    Raises:
        KeyError: If required material/hardening keys are missing.
        ValueError: If configuration values cannot be cast to floats.

    Notes:
        - DG0 fields are cell-wise constants.
        - NumPy work arrays are allocated once and reused every time step to
          avoid repeated allocations in the time loop.
    """

    def __init__(self, msh, material_cfg, hardening_cfg, birth_times_dolfinx):
        """Initialize function spaces, fields, constants, and initial state.

        Args:
            msh: Distributed DOLFINx mesh.
            material_cfg: Material/rheology configuration mapping.
            hardening_cfg: Hardening/time-evolution configuration mapping.
            birth_times_dolfinx: Cell birth times in DOLFINx ordering.

        Returns:
            None.

        Raises:
            KeyError: If required configuration keys are not present.
            ValueError: If numeric values cannot be converted to floats.
        """
        self.msh = msh
        self.material_cfg = dict(material_cfg)
        self.hardening_cfg = dict(hardening_cfg)
        self.birth_times = np.asarray(birth_times_dolfinx, dtype=float)

        self.V_DG0 = fem.functionspace(msh, ("DG", 0))
        self.V_DG0_tensor = fem.functionspace(msh, ("DG", 0, (3, 3)))

        self.E = fem.Function(self.V_DG0, name="young_modulus")
        self.G = fem.Function(self.V_DG0, name="shear_modulus")
        self.nu = fem.Function(self.V_DG0, name="poisson_ratio")
        self.rho = fem.Function(self.V_DG0, name="density")
        self.eta = fem.Function(self.V_DG0, name="viscosity")
        self.tau_y = fem.Function(self.V_DG0, name="yield_stress_shear")
        self.sigma_y = fem.Function(self.V_DG0, name="yield_stress_vm")
        self.yield_function_trial = fem.Function(self.V_DG0, name="yield_function_trial")

        self.von_mises = fem.Function(self.V_DG0, name="von_mises")
        self.max_principal_stress = fem.Function(
            self.V_DG0, name="max_principal_stress"
        )
        self.damage_max = fem.Function(self.V_DG0, name="damage_max")

        self.strain = fem.Function(self.V_DG0_tensor, name="strain")
        self.stress = fem.Function(self.V_DG0_tensor, name="stress")
        self.eps_vp = fem.Function(self.V_DG0_tensor, name="eps_vp")
        self.eps_vp.x.array[:] = 0.0
        # Keep ghosted tensor state synchronized immediately after initialization.
        self.eps_vp.x.scatter_forward()

        self.tau_0_pa = float(self.material_cfg["tau_0"])
        self.a_thix_pa_s = float(self.material_cfg["A_thix"])
        self.mu_p_pa_s = float(self.material_cfg["mu_p"])
        self.gamma_c = float(self.material_cfg["gamma_c"])
        self.rho_kg_m3 = float(self.material_cfg["rho"])
        self.g_val = float(self.material_cfg["g"])

        self.t_set = float(self.hardening_cfg["t_set"])
        self.e_inf = float(self.hardening_cfg["E_inf"])
        self.nu_fresh = float(self.hardening_cfg["nu_fresh"])
        self.nu_hard = float(self.hardening_cfg["nu_hard"])
        self.n_h = float(self.hardening_cfg.get("n_h", 0.7))

        self.tau_0_mpa = float(pa_to_mpa(self.tau_0_pa))
        self.a_thix_mpa_s = float(pa_to_mpa(self.a_thix_pa_s))
        self.mu_p_mpa_s = float(pa_to_mpa(self.mu_p_pa_s))
        self.rho_ns2_mm4 = float(kg_m3_to_ns2_mm4(self.rho_kg_m3))

        self.rho_const = fem.Constant(msh, self.rho_ns2_mm4)
        self.g_const = fem.Constant(msh, self.g_val)

        # Persistent scalar work arrays (no per-step reallocation).
        # These are the arrays passed as `out=` buffers to time_models.
        self.age = np.empty_like(self.birth_times)
        self.tau_arr = np.empty_like(self.birth_times)
        self.nu_arr = np.empty_like(self.birth_times)
        self.g_arr = np.empty_like(self.birth_times)
        self.e_arr = np.empty_like(self.birth_times)
        self.eta_arr = np.empty_like(self.birth_times)
        self.sigma_y_arr = np.empty_like(self.birth_times)

        # Initial state at age = 0 for all cells.
        age0 = np.zeros_like(self.birth_times)
        compute_yield_stress_mpa(
            age0,
            self.tau_0_mpa,
            self.a_thix_mpa_s,
            self.t_set,
            self.n_h,
            out=self.tau_arr,
        )
        compute_poisson_ratio(
            age0, self.nu_fresh, self.nu_hard, self.t_set, out=self.nu_arr
        )
        compute_shear_modulus_mpa(
            self.tau_arr,
            self.gamma_c,
            self.e_inf,
            self.nu_hard,
            out=self.g_arr,
        )
        compute_young_modulus_mpa(self.g_arr, self.nu_arr, out=self.e_arr)
        compute_viscosity_mpa_s(age0, self.mu_p_mpa_s, self.t_set, out=self.eta_arr)
        np.multiply(np.sqrt(3.0), self.tau_arr, out=self.sigma_y_arr)

        self.tau_y.x.array[:] = self.tau_arr
        self.sigma_y.x.array[:] = self.sigma_y_arr
        self.nu.x.array[:] = self.nu_arr
        self.G.x.array[:] = self.g_arr
        self.E.x.array[:] = self.e_arr
        self.eta.x.array[:] = self.eta_arr
        self.rho.x.array[:] = self.rho_ns2_mm4
        self.yield_function_trial.x.array[:] = 0.0
        self.von_mises.x.array[:] = 0.0
        self.max_principal_stress.x.array[:] = 0.0
        self.damage_max.x.array[:] = 0.0
        self.stress.x.array[:] = 0.0
        self.strain.x.array[:] = 0.0

        # All DG0/DG0-tensor field writes are followed by ghost synchronization
        # so later distributed reads (forms/output/diagnostics) see consistent data.
        self.tau_y.x.scatter_forward()
        self.sigma_y.x.scatter_forward()
        self.nu.x.scatter_forward()
        self.G.x.scatter_forward()
        self.E.x.scatter_forward()
        self.eta.x.scatter_forward()
        self.rho.x.scatter_forward()
        self.yield_function_trial.x.scatter_forward()
        self.von_mises.x.scatter_forward()
        self.max_principal_stress.x.scatter_forward()
        self.damage_max.x.scatter_forward()
        self.stress.x.scatter_forward()
        self.strain.x.scatter_forward()

    def update_properties(self, t_val):
        """Update age-dependent material properties at simulation time `t_val`.

        Args:
            t_val: Current simulation time.

        Returns:
            None.

        Raises:
            None.

        Math:
            age = max(t_val - t_birth, 0)
            sigma_y = sqrt(3) * tau_y

        The evolution laws for tau_y, nu, G, E, and eta are evaluated by
        `materials.time_models` using pre-allocated arrays.
        """
        # Compute non-negative material age in seconds.
        np.subtract(float(t_val), self.birth_times, out=self.age)
        np.maximum(self.age, 0.0, out=self.age)

        compute_yield_stress_mpa(
            self.age,
            self.tau_0_mpa,
            self.a_thix_mpa_s,
            self.t_set,
            self.n_h,
            out=self.tau_arr,
        )
        compute_poisson_ratio(
            self.age, self.nu_fresh, self.nu_hard, self.t_set, out=self.nu_arr
        )
        compute_shear_modulus_mpa(
            self.tau_arr,
            self.gamma_c,
            self.e_inf,
            self.nu_hard,
            out=self.g_arr,
        )
        compute_young_modulus_mpa(self.g_arr, self.nu_arr, out=self.e_arr)
        compute_viscosity_mpa_s(self.age, self.mu_p_mpa_s, self.t_set, out=self.eta_arr)
        np.multiply(np.sqrt(3.0), self.tau_arr, out=self.sigma_y_arr)

        self.tau_y.x.array[:] = self.tau_arr
        self.sigma_y.x.array[:] = self.sigma_y_arr
        self.nu.x.array[:] = self.nu_arr
        self.G.x.array[:] = self.g_arr
        self.E.x.array[:] = self.e_arr
        self.eta.x.array[:] = self.eta_arr
        self.rho.x.array[:] = self.rho_ns2_mm4

        self.tau_y.x.scatter_forward()
        self.sigma_y.x.scatter_forward()
        self.nu.x.scatter_forward()
        self.G.x.scatter_forward()
        self.E.x.scatter_forward()
        self.eta.x.scatter_forward()
        self.rho.x.scatter_forward()


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _cbrt_numba(x):
        """Real cube root for scalar `x` (Numba-safe)."""
        if x >= 0.0:
            return x ** (1.0 / 3.0)
        return -((-x) ** (1.0 / 3.0))


    @njit(cache=True)
    def _max_eigval_sym33_cardano_components(
        m00, m01, m02, m10, m11, m12, m20, m21, m22
    ):
        """Largest eigenvalue of one 3x3 symmetric tensor via Cardano/trig."""
        # Explicit symmetrization to mirror the NumPy path.
        a00 = m00
        a11 = m11
        a22 = m22
        a01 = 0.5 * (m01 + m10)
        a02 = 0.5 * (m02 + m20)
        a12 = 0.5 * (m12 + m21)

        # Characteristic invariants:
        # lambda^3 - I1*lambda^2 + I2*lambda - I3 = 0
        i1 = a00 + a11 + a22
        i2 = a00 * a11 + a11 * a22 + a22 * a00 - (a01 * a01 + a02 * a02 + a12 * a12)
        i3 = (
            a00 * (a11 * a22 - a12 * a12)
            - a01 * (a01 * a22 - a12 * a02)
            + a02 * (a01 * a12 - a11 * a02)
        )

        # Depressed cubic x^3 + p*x + q = 0 with lambda = x + I1/3.
        p = i2 - (i1 * i1) / 3.0
        q = (-2.0 * i1 * i1 * i1) / 27.0 + (i1 * i2) / 3.0 - i3

        # Degenerate / near-hydrostatic branch (prevents NaN).
        if p >= -1.0e-14:
            if p <= 1.0e-14:
                return i1 / 3.0

            # Robust real-root fallback if round-off pushes p > 0.
            half_q = -0.5 * q
            p_over_3 = p / 3.0
            disc = half_q * half_q + p_over_3 * p_over_3 * p_over_3
            if disc < 0.0:
                disc = 0.0
            sqrt_disc = np.sqrt(disc)
            u = _cbrt_numba(half_q + sqrt_disc)
            v = _cbrt_numba(half_q - sqrt_disc)
            return (u + v) + i1 / 3.0

        r = (3.0 * q / (2.0 * p)) * np.sqrt(-3.0 / p)
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0

        phi = np.arccos(r) / 3.0
        amp = 2.0 * np.sqrt(-p / 3.0)
        x0 = amp * np.cos(phi)
        x1 = amp * np.cos(phi + 2.0943951023931953)  # 2*pi/3
        x2 = amp * np.cos(phi + 4.1887902047863905)  # 4*pi/3

        x_max = x0
        if x1 > x_max:
            x_max = x1
        if x2 > x_max:
            x_max = x2
        return x_max + i1 / 3.0


    @njit(cache=True)
    def _max_eigval_sym33_cardano(sig_cell):
        """Largest eigenvalue of one 3x3 symmetric tensor via Cardano/trig."""
        return _max_eigval_sym33_cardano_components(
            sig_cell[0, 0],
            sig_cell[0, 1],
            sig_cell[0, 2],
            sig_cell[1, 0],
            sig_cell[1, 1],
            sig_cell[1, 2],
            sig_cell[2, 0],
            sig_cell[2, 1],
            sig_cell[2, 2],
        )


    @njit(cache=True, inline="always")
    def _update_perzyna_state_single_cell_numba(
        c,
        strain_total,
        eps_vp_prev,
        e_arr,
        nu_arr,
        sigma_y_arr,
        eta_arr,
        dt_eff,
        eps_vp_new,
        sigma_new,
        vm_new,
        max_ps_new,
        f_trial,
    ):
        """Single-cell J2-Perzyna update writing directly into output arrays."""
        e_cell = e_arr[c]
        nu_cell = nu_arr[c]
        sigma_y_cell = sigma_y_arr[c]
        eta_cell = eta_arr[c]

        mu = e_cell / (2.0 * (1.0 + nu_cell))
        lam = e_cell * nu_cell / ((1.0 + nu_cell) * (1.0 - 2.0 * nu_cell))

        e00 = strain_total[c, 0, 0] - eps_vp_prev[c, 0, 0]
        e01 = strain_total[c, 0, 1] - eps_vp_prev[c, 0, 1]
        e02 = strain_total[c, 0, 2] - eps_vp_prev[c, 0, 2]
        e10 = strain_total[c, 1, 0] - eps_vp_prev[c, 1, 0]
        e11 = strain_total[c, 1, 1] - eps_vp_prev[c, 1, 1]
        e12 = strain_total[c, 1, 2] - eps_vp_prev[c, 1, 2]
        e20 = strain_total[c, 2, 0] - eps_vp_prev[c, 2, 0]
        e21 = strain_total[c, 2, 1] - eps_vp_prev[c, 2, 1]
        e22 = strain_total[c, 2, 2] - eps_vp_prev[c, 2, 2]
        tr_eps_trial = e00 + e11 + e22

        s00 = 2.0 * mu * e00 + lam * tr_eps_trial
        s01 = 2.0 * mu * e01
        s02 = 2.0 * mu * e02
        s10 = 2.0 * mu * e10
        s11 = 2.0 * mu * e11 + lam * tr_eps_trial
        s12 = 2.0 * mu * e12
        s20 = 2.0 * mu * e20
        s21 = 2.0 * mu * e21
        s22 = 2.0 * mu * e22 + lam * tr_eps_trial

        p_trial = (s00 + s11 + s22) / 3.0
        s00 -= p_trial
        s11 -= p_trial
        s22 -= p_trial

        s_norm = np.sqrt(
            s00 * s00
            + s01 * s01
            + s02 * s02
            + s10 * s10
            + s11 * s11
            + s12 * s12
            + s20 * s20
            + s21 * s21
            + s22 * s22
        )
        vm_trial = np.sqrt(1.5) * s_norm
        f_trial_cell = vm_trial - sigma_y_cell

        eta_safe = eta_cell
        if eta_safe < 1.0e-12:
            eta_safe = 1.0e-12
        positive_f = f_trial_cell
        if positive_f < 0.0:
            positive_f = 0.0
        s_norm_safe = s_norm
        if s_norm_safe < 1.0e-12:
            s_norm_safe = 1.0e-12

        flow_scale = (dt_eff / (2.0 * eta_safe)) * positive_f / s_norm_safe

        vp00 = eps_vp_prev[c, 0, 0] + flow_scale * s00
        vp01 = eps_vp_prev[c, 0, 1] + flow_scale * s01
        vp02 = eps_vp_prev[c, 0, 2] + flow_scale * s02
        vp10 = eps_vp_prev[c, 1, 0] + flow_scale * s10
        vp11 = eps_vp_prev[c, 1, 1] + flow_scale * s11
        vp12 = eps_vp_prev[c, 1, 2] + flow_scale * s12
        vp20 = eps_vp_prev[c, 2, 0] + flow_scale * s20
        vp21 = eps_vp_prev[c, 2, 1] + flow_scale * s21
        vp22 = eps_vp_prev[c, 2, 2] + flow_scale * s22

        eps_vp_new[c, 0, 0] = vp00
        eps_vp_new[c, 0, 1] = vp01
        eps_vp_new[c, 0, 2] = vp02
        eps_vp_new[c, 1, 0] = vp10
        eps_vp_new[c, 1, 1] = vp11
        eps_vp_new[c, 1, 2] = vp12
        eps_vp_new[c, 2, 0] = vp20
        eps_vp_new[c, 2, 1] = vp21
        eps_vp_new[c, 2, 2] = vp22

        e00n = strain_total[c, 0, 0] - vp00
        e01n = strain_total[c, 0, 1] - vp01
        e02n = strain_total[c, 0, 2] - vp02
        e10n = strain_total[c, 1, 0] - vp10
        e11n = strain_total[c, 1, 1] - vp11
        e12n = strain_total[c, 1, 2] - vp12
        e20n = strain_total[c, 2, 0] - vp20
        e21n = strain_total[c, 2, 1] - vp21
        e22n = strain_total[c, 2, 2] - vp22
        tr_eps_new = e00n + e11n + e22n

        sn00 = 2.0 * mu * e00n + lam * tr_eps_new
        sn01 = 2.0 * mu * e01n
        sn02 = 2.0 * mu * e02n
        sn10 = 2.0 * mu * e10n
        sn11 = 2.0 * mu * e11n + lam * tr_eps_new
        sn12 = 2.0 * mu * e12n
        sn20 = 2.0 * mu * e20n
        sn21 = 2.0 * mu * e21n
        sn22 = 2.0 * mu * e22n + lam * tr_eps_new

        sigma_new[c, 0, 0] = sn00
        sigma_new[c, 0, 1] = sn01
        sigma_new[c, 0, 2] = sn02
        sigma_new[c, 1, 0] = sn10
        sigma_new[c, 1, 1] = sn11
        sigma_new[c, 1, 2] = sn12
        sigma_new[c, 2, 0] = sn20
        sigma_new[c, 2, 1] = sn21
        sigma_new[c, 2, 2] = sn22

        p_new = (sn00 + sn11 + sn22) / 3.0
        d00 = sn00 - p_new
        d11 = sn11 - p_new
        d22 = sn22 - p_new
        j2_term = d00 * d00 + d11 * d11 + d22 * d22 + 2.0 * (
            sn01 * sn01 + sn02 * sn02 + sn12 * sn12
        )

        vm_new[c] = np.sqrt(1.5 * j2_term)
        max_ps_new[c] = _max_eigval_sym33_cardano_components(
            sn00, sn01, sn02, sn10, sn11, sn12, sn20, sn21, sn22
        )
        f_trial[c] = f_trial_cell


    @njit(parallel=True, cache=True)
    def _update_perzyna_state_cellwise_numba(
        strain_total,
        eps_vp_prev,
        e_arr,
        nu_arr,
        sigma_y_arr,
        eta_arr,
        dt,
        active_mask,
    ):
        """Parallel outer loop over cells (Numba, prange)."""
        n_cells = strain_total.shape[0]

        eps_vp_new = np.empty_like(eps_vp_prev)
        sigma_new = np.empty_like(strain_total)
        vm_new = np.empty(n_cells, dtype=strain_total.dtype)
        max_ps_new = np.zeros(n_cells, dtype=strain_total.dtype)
        f_trial = np.empty(n_cells, dtype=strain_total.dtype)

        dt_eff = dt
        if dt_eff < 0.0:
            dt_eff = 0.0

        for c in prange(n_cells):
            if not active_mask[c]:
                eps_vp_new[c, 0, 0] = 0.0
                eps_vp_new[c, 0, 1] = 0.0
                eps_vp_new[c, 0, 2] = 0.0
                eps_vp_new[c, 1, 0] = 0.0
                eps_vp_new[c, 1, 1] = 0.0
                eps_vp_new[c, 1, 2] = 0.0
                eps_vp_new[c, 2, 0] = 0.0
                eps_vp_new[c, 2, 1] = 0.0
                eps_vp_new[c, 2, 2] = 0.0

                sigma_new[c, 0, 0] = 0.0
                sigma_new[c, 0, 1] = 0.0
                sigma_new[c, 0, 2] = 0.0
                sigma_new[c, 1, 0] = 0.0
                sigma_new[c, 1, 1] = 0.0
                sigma_new[c, 1, 2] = 0.0
                sigma_new[c, 2, 0] = 0.0
                sigma_new[c, 2, 1] = 0.0
                sigma_new[c, 2, 2] = 0.0

                vm_new[c] = 0.0
                max_ps_new[c] = 0.0
                f_trial[c] = 0.0
                continue

            _update_perzyna_state_single_cell_numba(
                c,
                strain_total,
                eps_vp_prev,
                e_arr,
                nu_arr,
                sigma_y_arr,
                eta_arr,
                dt_eff,
                eps_vp_new,
                sigma_new,
                vm_new,
                max_ps_new,
                f_trial,
            )

        return eps_vp_new, sigma_new, vm_new, max_ps_new, f_trial

else:

    def _update_perzyna_state_cellwise_numba(*args, **kwargs):
        raise RuntimeError("Numba is not available.")


_NUMBA_WARMED_UP = False
_PERZYNA_BACKEND_LOGGED = False


def _log_perzyna_backend_once(using_numba):
    """Log once per process which constitutive backend is active."""
    global _PERZYNA_BACKEND_LOGGED
    if _PERZYNA_BACKEND_LOGGED:
        return

    if using_numba:
        msg = "[Perzyna] Using Numba-accelerated cellwise constitutive update."
    else:
        msg = "[Perzyna] Numba unavailable; using NumPy constitutive fallback."

    logger.info(msg)
    print(msg, flush=True)
    _PERZYNA_BACKEND_LOGGED = True


def _warmup_update_perzyna_numba():
    """Compile Numba kernels once with tiny dummy arrays."""
    global _NUMBA_WARMED_UP
    if (not _NUMBA_AVAILABLE) or _NUMBA_WARMED_UP:
        return

    strain_dummy = np.zeros((1, 3, 3), dtype=np.float64)
    eps_vp_dummy = np.zeros((1, 3, 3), dtype=np.float64)
    e_dummy = np.ones(1, dtype=np.float64)
    nu_dummy = np.full(1, 0.3, dtype=np.float64)
    sigma_y_dummy = np.ones(1, dtype=np.float64)
    eta_dummy = np.ones(1, dtype=np.float64)
    mask_dummy = np.ones(1, dtype=np.bool_)

    _update_perzyna_state_cellwise_numba(
        strain_dummy,
        eps_vp_dummy,
        e_dummy,
        nu_dummy,
        sigma_y_dummy,
        eta_dummy,
        0.0,
        mask_dummy,
    )
    _NUMBA_WARMED_UP = True


def _update_perzyna_state_cellwise_numpy(
    strain_total,
    eps_vp_prev,
    e_arr,
    nu_arr,
    sigma_y_arr,
    eta_arr,
    dt,
    active_mask=None,
):
    """Pure NumPy cell-wise explicit J2-Perzyna update.

    Args:
        strain_total (np.ndarray): Shape (n_cells, 3, 3).
        eps_vp_prev (np.ndarray): Shape (n_cells, 3, 3).
        e_arr (np.ndarray): Shape (n_cells,).
        nu_arr (np.ndarray): Shape (n_cells,).
        sigma_y_arr (np.ndarray): Shape (n_cells,).
        eta_arr (np.ndarray): Shape (n_cells,).
        dt (float): Time increment.
        active_mask (np.ndarray | None): Shape (n_cells,), optional.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (eps_vp_new, sigma_new, vm_new, max_ps_new, f_trial) with shapes
            matching the original implementation.
    """
    dt_eff = max(float(dt), 0.0)
    n_cells = strain_total.shape[0]

    if active_mask is not None:
        active_mask_arr = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask_arr.shape[0] != n_cells:
            raise ValueError(
                "active_mask length mismatch in update_perzyna_state_cellwise."
            )

        eps_vp_new = np.zeros_like(eps_vp_prev)
        sigma_new = np.zeros_like(strain_total)
        vm_new = np.zeros(n_cells, dtype=strain_total.dtype)
        max_ps_new = np.zeros(n_cells, dtype=strain_total.dtype)
        f_trial = np.zeros(n_cells, dtype=strain_total.dtype)

        if not np.any(active_mask_arr):
            return eps_vp_new, sigma_new, vm_new, max_ps_new, f_trial

        strain_eval = strain_total[active_mask_arr]
        eps_vp_prev_eval = eps_vp_prev[active_mask_arr]
        e_eval = e_arr[active_mask_arr]
        nu_eval = nu_arr[active_mask_arr]
        sigma_y_eval = sigma_y_arr[active_mask_arr]
        eta_eval = eta_arr[active_mask_arr]
    else:
        active_mask_arr = None
        strain_eval = strain_total
        eps_vp_prev_eval = eps_vp_prev
        e_eval = e_arr
        nu_eval = nu_arr
        sigma_y_eval = sigma_y_arr
        eta_eval = eta_arr

    eps_e_trial = strain_eval - eps_vp_prev_eval

    mu = e_eval / (2.0 * (1.0 + nu_eval))
    lam = e_eval * nu_eval / ((1.0 + nu_eval) * (1.0 - 2.0 * nu_eval))

    tr_eps_trial = eps_e_trial[:, 0, 0] + eps_e_trial[:, 1, 1] + eps_e_trial[:, 2, 2]

    sigma_trial = 2.0 * mu[:, None, None] * eps_e_trial
    sigma_trial[:, 0, 0] += lam * tr_eps_trial
    sigma_trial[:, 1, 1] += lam * tr_eps_trial
    sigma_trial[:, 2, 2] += lam * tr_eps_trial

    tr_sig_trial = sigma_trial[:, 0, 0] + sigma_trial[:, 1, 1] + sigma_trial[:, 2, 2]
    p_trial = tr_sig_trial / 3.0
    sigma_trial[:, 0, 0] -= p_trial
    sigma_trial[:, 1, 1] -= p_trial
    sigma_trial[:, 2, 2] -= p_trial

    s_norm = np.sqrt(np.sum(sigma_trial * sigma_trial, axis=(1, 2)))
    vm_trial = np.sqrt(1.5) * s_norm
    f_trial_eval = vm_trial - sigma_y_eval

    eta_safe = np.maximum(np.asarray(eta_eval, dtype=float), 1.0e-12)
    positive_f = np.maximum(f_trial_eval, 0.0)
    flow_scale = (dt_eff / (2.0 * eta_safe)) * positive_f / np.maximum(s_norm, 1.0e-12)
    eps_vp_new_eval = eps_vp_prev_eval + flow_scale[:, None, None] * sigma_trial

    eps_e_new = strain_eval - eps_vp_new_eval
    tr_eps_new = eps_e_new[:, 0, 0] + eps_e_new[:, 1, 1] + eps_e_new[:, 2, 2]

    sigma_new_eval = 2.0 * mu[:, None, None] * eps_e_new
    sigma_new_eval[:, 0, 0] += lam * tr_eps_new
    sigma_new_eval[:, 1, 1] += lam * tr_eps_new
    sigma_new_eval[:, 2, 2] += lam * tr_eps_new

    tr_sig_new = sigma_new_eval[:, 0, 0] + sigma_new_eval[:, 1, 1] + sigma_new_eval[:, 2, 2]
    p_new = tr_sig_new / 3.0
    s00 = sigma_new_eval[:, 0, 0] - p_new
    s11 = sigma_new_eval[:, 1, 1] - p_new
    s22 = sigma_new_eval[:, 2, 2] - p_new
    s01 = sigma_new_eval[:, 0, 1]
    s02 = sigma_new_eval[:, 0, 2]
    s12 = sigma_new_eval[:, 1, 2]
    j2_term = s00 * s00 + s11 * s11 + s22 * s22 + 2.0 * (s01 * s01 + s02 * s02 + s12 * s12)
    vm_new_eval = np.sqrt(1.5 * j2_term)

    sigma_sym = 0.5 * (sigma_new_eval + np.swapaxes(sigma_new_eval, 1, 2))
    max_ps_eval = np.linalg.eigvalsh(sigma_sym)[:, 2]

    if active_mask_arr is None:
        return eps_vp_new_eval, sigma_new_eval, vm_new_eval, max_ps_eval, f_trial_eval

    eps_vp_new[active_mask_arr] = eps_vp_new_eval
    sigma_new[active_mask_arr] = sigma_new_eval
    vm_new[active_mask_arr] = vm_new_eval
    max_ps_new[active_mask_arr] = max_ps_eval
    f_trial[active_mask_arr] = f_trial_eval
    return eps_vp_new, sigma_new, vm_new, max_ps_new, f_trial


def update_perzyna_state_cellwise(
    strain_total,
    eps_vp_prev,
    e_arr,
    nu_arr,
    sigma_y_arr,
    eta_arr,
    dt,
    active_mask=None,
):
    """Cell-wise explicit J2-Perzyna viscoplastic update.

    Args:
        strain_total (np.ndarray): Shape (n_cells, 3, 3).
        eps_vp_prev (np.ndarray): Shape (n_cells, 3, 3).
        e_arr (np.ndarray): Shape (n_cells,).
        nu_arr (np.ndarray): Shape (n_cells,).
        sigma_y_arr (np.ndarray): Shape (n_cells,).
        eta_arr (np.ndarray): Shape (n_cells,).
        dt (float): Time increment.
        active_mask (np.ndarray | None): Shape (n_cells,), optional.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (eps_vp_new, sigma_new, vm_new, max_ps_new, f_trial), with the same
            shapes/order as the legacy implementation.
    """
    n_cells = strain_total.shape[0]
    active_mask_arr = None
    if active_mask is not None:
        active_mask_arr = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask_arr.shape[0] != n_cells:
            raise ValueError("active_mask length mismatch in update_perzyna_state_cellwise.")

    if _NUMBA_AVAILABLE:
        _log_perzyna_backend_once(True)
        _warmup_update_perzyna_numba()
        if active_mask_arr is None:
            active_mask_numba = np.ones(n_cells, dtype=np.bool_)
        else:
            active_mask_numba = active_mask_arr

        return _update_perzyna_state_cellwise_numba(
            strain_total,
            eps_vp_prev,
            e_arr,
            nu_arr,
            sigma_y_arr,
            eta_arr,
            float(dt),
            active_mask_numba,
        )

    _log_perzyna_backend_once(False)
    return _update_perzyna_state_cellwise_numpy(
        strain_total,
        eps_vp_prev,
        e_arr,
        nu_arr,
        sigma_y_arr,
        eta_arr,
        dt,
        active_mask=active_mask_arr,
    )
