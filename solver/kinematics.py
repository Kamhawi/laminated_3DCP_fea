# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Kinematic operators and activation-aware displacement utilities for 3DCP FEA.

This module provides small-strain continuum operators and helper routines used
inside the transient simulation loop to manage activation-dependent displacement
states. In the global pipeline, these utilities are called before/after each
nonlinear solve to enforce physically consistent initialization for newly born
material and to keep not-yet-born material kinematically inactive.

Physics:
    The constitutive kinematics are based on small-strain linearized measures:

    - Strain tensor:
      epsilon(u) = sym(grad(u))
    - Linear isotropic Cauchy stress:
      sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon

Parallel Notes:
    Calls to ``u.x.scatter_forward()`` and ``is_active_func.x.scatter_forward()``
    are intentionally retained at their exact synchronization points to ensure
    owned and ghost values remain consistent across MPI ranks.
"""

import numpy as np
import ufl


def epsilon(u):
    """Compute the small-strain tensor from displacement.

    Args:
        u: UFL-compatible displacement field.

    Returns:
        ufl.core.expr.Expr: Symmetric small-strain tensor ``sym(grad(u))``.

    Raises:
        None.

    Math:
        epsilon(u) = 1/2 * (grad(u) + grad(u)^T)
    """
    return ufl.sym(ufl.grad(u))


def sigma_from_strain(eps_tensor, lam, mu):
    """Compute linear isotropic stress from a given strain tensor.

    Args:
        eps_tensor: Strain tensor expression.
        lam: First Lame parameter ``lambda``.
        mu: Shear modulus ``mu``.

    Returns:
        ufl.core.expr.Expr: Cauchy stress tensor for linear isotropic elasticity.

    Raises:
        None.

    Math:
        sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon
    """
    return lam * ufl.tr(eps_tensor) * ufl.Identity(3) + 2 * mu * eps_tensor


def sigma(u, lam, mu):
    """Compute linear isotropic stress directly from displacement.

    Args:
        u: UFL-compatible displacement field.
        lam: First Lame parameter ``lambda``.
        mu: Shear modulus ``mu``.

    Returns:
        ufl.core.expr.Expr: Cauchy stress tensor.

    Raises:
        None.

    Math:
        sigma(u) = lambda * tr(epsilon(u)) * I + 2 * mu * epsilon(u)
    """
    return sigma_from_strain(epsilon(u), lam, mu)


def zero_inactive_cells(u, t_val, birth_times, cell_to_dofs):
    """Enforce zero displacement on cells that are not yet active.

    Args:
        u: Displacement function updated in place.
        t_val: Current physical time.
        birth_times: Cell birth-time array (DOLFINx cell ordering).
        cell_to_dofs: Cell-to-vector-DOF map with shape
            ``(num_cells, dofs_per_cell)``.

    Returns:
        None.

    Raises:
        None.

    Physics:
        Material deposited in the future (``birth_time > t``) must not carry
        displacement before activation.
    """
    inactive_cells = np.where(birth_times > t_val)[0]
    if inactive_cells.size > 0:
        # Flatten selected cell DOFs, remove duplicates, then zero the unique
        # vector DOF entries owned on this rank.
        inactive_dofs = np.unique(cell_to_dofs[inactive_cells].reshape(-1))
        u.x.array[inactive_dofs] = 0.0
    # MPI ghost synchronization: ensure neighboring ranks see the updated
    # inactive-cell displacement values before subsequent assembly/evaluation.
    u.x.scatter_forward()


def zero_newly_activated_cells(u, t_val, t_prev, birth_times, cell_to_dofs):
    """Reset displacement on cells activated within the current time increment.

    Args:
        u: Displacement function updated in place.
        t_val: Current physical time.
        t_prev: Previous physical time.
        birth_times: Cell birth-time array (DOLFINx cell ordering).
        cell_to_dofs: Cell-to-vector-DOF map with shape
            ``(num_cells, dofs_per_cell)``.

    Returns:
        None.

    Raises:
        None.

    Physics:
        Newly deposited material should start each activation event from a
        controlled initial displacement state to avoid inheriting stale values.
    """
    newly_active = np.where((birth_times <= t_val) & (birth_times > t_prev))[0]
    if newly_active.size > 0:
        # Use unique flattened DOFs so each vector entry is set once even when
        # multiple selected cells reference shared entities.
        newly_active_dofs = np.unique(cell_to_dofs[newly_active].reshape(-1))
        u.x.array[newly_active_dofs] = 0.0
    # Required to propagate writes to ghost entries before any later read.
    u.x.scatter_forward()


def init_newly_activated_displacement(
    u, newly_active_cells, support, cell_to_dofs, birth_times=None, t_prev=None
):
    """Initialize new cells by copying full DOF state from support cell.

    For each newly active cell with a valid support cell that was already
    active at ``t_prev``, the routine copies all displacement DOFs from the
    support cell. Cells whose support cell is not yet active (or has no
    support) are initialized to zero.

    Args:
        u: Displacement function updated in place.
        newly_active_cells: Iterable of newly activated cell indices.
        support: Per-cell support mapping; ``support[c]`` gives supporting cell
            index or ``-1`` when unavailable. May be ``None``.
        cell_to_dofs: Cell-to-vector-DOF map with shape
            ``(num_cells, dofs_per_cell)``.
        birth_times: Cell birth-time array. When provided with ``t_prev``,
            only copies from support cells that were active at ``t_prev``.
        t_prev: Previous step time. Support cells with ``birth_time > t_prev``
            are treated as inactive (u=0) and skipped.

    Returns:
        None.

    Raises:
        None.

    Physics:
        The initialization copies the full deformation state from the underlying
        support cell at activation time, reducing artificial transients.
    """
    cells = np.asarray(newly_active_cells, dtype=np.int64)
    if cells.size == 0:
        # Even with no updates, keep ghost values synchronized for downstream
        # routines that may read this field collectively.
        u.x.scatter_forward()
        return

    if support is None:
        # Fallback: no support relation available, so initialize new material
        # from a stress-free/zero-displacement state.
        u.x.array[cell_to_dofs[cells].reshape(-1)] = 0.0
        u.x.scatter_forward()
        return

    # Split new cells by support availability to avoid branchy per-cell loops.
    has_support = support[cells] >= 0

    # Additionally check that the support cell is already active (was born
    # before the current step). Inactive support cells have u=0, so copying
    # from them is useless.
    if has_support.any() and birth_times is not None and t_prev is not None:
        support_ids = support[cells[has_support]]
        support_active = birth_times[support_ids] <= t_prev
        # Demote cells whose support is not yet active to unsupported.
        active_idx = np.where(has_support)[0]
        has_support[active_idx[~support_active]] = False

    supported_cells = cells[has_support]
    unsupported_cells = cells[~has_support]

    if supported_cells.size > 0:
        support_cells = support[supported_cells].astype(np.int64, copy=False)
        support_dofs = cell_to_dofs[support_cells]
        new_dofs = cell_to_dofs[supported_cells]
        # Full DOF copy: newly activated cells inherit the complete deformed
        # shape from the support cell below, preserving internal strain.
        u.x.array[new_dofs] = u.x.array[support_dofs]

    if unsupported_cells.size > 0:
        # Unsupported new cells remain zero-initialized by design.
        u.x.array[cell_to_dofs[unsupported_cells].reshape(-1)] = 0.0

    # MPI ghost synchronization is required after direct array writes so all
    # ranks observe identical activation-initialized displacement values.
    u.x.scatter_forward()


def update_active_indicator(is_active_func, t_val, birth_times):
    """Update DG0 activation indicator field from birth-time data.

    Args:
        is_active_func: DG0 function storing per-cell active flags.
        t_val: Current physical time.
        birth_times: Cell birth-time array (DOLFINx cell ordering).

    Returns:
        None.

    Raises:
        None.

    Physics:
        Cells satisfy ``is_active = 1`` when ``birth_time <= t`` and
        ``is_active = 0`` otherwise.
    """
    # DG0 layout is one value per cell; vectorized thresholding is faster and
    # avoids Python-side per-cell loops.
    is_active_func.x.array[:] = (birth_times <= t_val).astype(float)
    # Ensure indicator values are synchronized on ghost cells before use in
    # distributed post-processing or weak-form evaluation.
    is_active_func.x.scatter_forward()
