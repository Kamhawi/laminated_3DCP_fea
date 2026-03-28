"""Error vector localization (Test 1.5).

Computes the Taylor error vector c = (F(u+eps*du) - F(u))/eps - J*du
and decomposes its L2 norm by DOF region:
  - Interior active DOFs (no facet shared with inactive cells)
  - Activation-boundary DOFs (active cells sharing a facet with inactive)
  - Inactive DOFs (cells with birth_time > t_val)

This confirms whether the rate=1 Taylor test finding is caused by
inactive DOF cross-facet coupling in the Jacobian matvec.
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, create_vector
from mpi4py import MPI

from diagnostics.taylor_test import make_perturbation, _assemble_residual_into


def classify_dofs(state):
    """Partition owned DOFs into interior-active, boundary-active, inactive.

    Args:
        state: DiagnosticState with prepared t_val and inactive_dofs.

    Returns:
        dict with keys ``interior_active``, ``boundary_active``,
        ``inactive`` — each a sorted np.ndarray of owned DOF indices.
    """
    msh = state.msh
    tdim = msh.topology.dim
    fdim = tdim - 1
    num_owned = len(state.u.x.array)

    active_cells = np.where(state.birth_times_dolfinx <= state.t_val)[0]
    inactive_cells = np.where(state.birth_times_dolfinx > state.t_val)[0]

    if inactive_cells.size == 0:
        # All cells active — everything is interior active.
        all_owned = np.arange(num_owned, dtype=np.int32)
        return {
            "interior_active": all_owned,
            "boundary_active": np.array([], dtype=np.int32),
            "inactive": np.array([], dtype=np.int32),
        }

    active_set = set(active_cells.tolist())
    inactive_set = set(inactive_cells.tolist())

    # Find active cells sharing a facet with an inactive cell.
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    boundary_active_cells = set()
    for cell in active_cells:
        facets = c_to_f.links(cell)
        for facet in facets:
            neighbors = f_to_c.links(facet)
            for nb in neighbors:
                if nb != cell and nb in inactive_set:
                    boundary_active_cells.add(cell)
                    break
            if cell in boundary_active_cells:
                break

    interior_active_cells = active_set - boundary_active_cells

    # Map cells to owned DOFs.
    def _cells_to_owned_dofs(cell_set):
        if not cell_set:
            return np.array([], dtype=np.int32)
        cells = np.array(sorted(cell_set))
        dofs = np.unique(state.cell_to_dofs[cells].reshape(-1))
        return dofs[dofs < num_owned].astype(np.int32)

    inactive_dofs = _cells_to_owned_dofs(inactive_set)
    boundary_active_dofs = _cells_to_owned_dofs(boundary_active_cells)
    interior_active_dofs = _cells_to_owned_dofs(interior_active_cells)

    # Some DOFs may be shared between boundary-active and inactive cells
    # (DG doesn't share, but just in case). Inactive takes priority.
    inactive_set_dofs = set(inactive_dofs.tolist())
    boundary_active_dofs = np.array(
        sorted(set(boundary_active_dofs.tolist()) - inactive_set_dofs),
        dtype=np.int32,
    )
    interior_active_dofs = np.array(
        sorted(set(interior_active_dofs.tolist()) - inactive_set_dofs
               - set(boundary_active_dofs.tolist())),
        dtype=np.int32,
    )

    return {
        "interior_active": interior_active_dofs,
        "boundary_active": boundary_active_dofs,
        "inactive": inactive_dofs,
    }


def compute_error_vector(state, F_form, J_form, eps=1e-4, seed=42):
    """Compute the Taylor error vector and decompose by DOF region.

    Computes c = (F(u+eps*du) - F(u))/eps - J*du WITHOUT zeroing Jdu
    at inactive DOFs (to reveal the test infrastructure bug), then also
    computes the fixed version (with zeroing) for comparison.

    Args:
        state: DiagnosticState with prepared u and inactive_dofs.
        F_form: Compiled residual form.
        J_form: Compiled Jacobian form.
        eps: Perturbation magnitude for the probe.
        seed: Random seed.

    Returns:
        dict with regional norms (unfixed and fixed) and DOF counts.
    """
    comm = state.comm
    u = state.u
    u0 = u.x.array.copy()

    du_arr = make_perturbation(state, seed=seed)

    # Assemble F(u0).
    b = create_vector(F_form)
    _assemble_residual_into(b, F_form)
    if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
        b.array[state.inactive_dofs] = 0.0
    F0 = b.array.copy()

    # Assemble J(u0) and compute J*du (WITHOUT zeroing inactive DOFs).
    A = assemble_matrix(J_form)
    A.assemble()

    du_func = fem.Function(state.V)
    du_func.x.array[:] = du_arr
    du_func.x.scatter_forward()

    Jdu_vec = A.createVecLeft()
    A.mult(du_func.x.petsc_vec, Jdu_vec)
    Jdu_raw = Jdu_vec.array.copy()  # Before zeroing — shows the bug.

    # Fixed version: zero inactive DOFs.
    Jdu_fixed = Jdu_raw.copy()
    if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
        Jdu_fixed[state.inactive_dofs] = 0.0

    # Assemble F(u0 + eps*du).
    u.x.array[:] = u0 + eps * du_arr
    u.x.scatter_forward()
    _assemble_residual_into(b, F_form)
    if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
        b.array[state.inactive_dofs] = 0.0
    F_eps = b.array.copy()

    # Restore u.
    u.x.array[:] = u0
    u.x.scatter_forward()

    # Cleanup PETSc objects.
    Jdu_vec.destroy()
    A.destroy()
    b.destroy()
    du_func.x.petsc_vec.destroy()

    # Error vectors.
    c_raw = (F_eps - F0) / eps - Jdu_raw
    c_fixed = (F_eps - F0) / eps - Jdu_fixed

    # Classify DOFs and compute regional norms.
    regions = classify_dofs(state)

    def _regional_norm(vec, indices):
        if indices.size == 0:
            return 0.0
        local_sq = float(np.dot(vec[indices], vec[indices]))
        return comm.allreduce(local_sq, op=MPI.SUM)

    result = {"dof_counts": {}, "raw": {}, "fixed": {}}
    total_raw_sq = 0.0
    total_fixed_sq = 0.0

    for name, dof_indices in regions.items():
        result["dof_counts"][name] = comm.allreduce(len(dof_indices), op=MPI.SUM)
        raw_sq = _regional_norm(c_raw, dof_indices)
        fixed_sq = _regional_norm(c_fixed, dof_indices)
        result["raw"][name] = np.sqrt(raw_sq)
        result["fixed"][name] = np.sqrt(fixed_sq)
        total_raw_sq += raw_sq
        total_fixed_sq += fixed_sq

    result["raw"]["total"] = np.sqrt(total_raw_sq)
    result["fixed"]["total"] = np.sqrt(total_fixed_sq)
    result["dof_counts"]["total"] = sum(result["dof_counts"].values())
    result["eps"] = eps

    return result


def run_error_localization(state, verbose=True):
    """Run error localization at the current state.

    Args:
        state: DiagnosticState with prepared u and inactive_dofs.
        verbose: Print results on rank 0.

    Returns:
        dict from ``compute_error_vector``.
    """
    comm = state.comm

    if state.inactive_dofs is None or state.inactive_dofs.size == 0:
        total_inactive = comm.allreduce(0, op=MPI.SUM)
        if total_inactive == 0:
            if verbose and comm.rank == 0:
                print("\n  Error localization: N/A (all cells active)")
            return {"raw": {"total": 0.0}, "fixed": {"total": 0.0}}

    result = compute_error_vector(state, state.F_form, state.J_form)

    if verbose and comm.rank == 0:
        eps = result["eps"]
        print(f"\n  Error localization (eps={eps:.1e}, t_val={state.t_val:.2f})")
        print(f"  {'DOF region':<28s}  {'Count':>6s}  "
              f"{'||c|| (raw)':>12s}  {'||c|| (fix)':>12s}  {'Frac (raw)':>10s}")
        print(f"  {'-' * 28}  {'-' * 6}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

        total_raw = max(result["raw"]["total"], 1e-30)
        for name in ["interior_active", "boundary_active", "inactive"]:
            count = result["dof_counts"][name]
            raw = result["raw"][name]
            fixed = result["fixed"][name]
            frac = raw / total_raw
            print(f"  {name:<28s}  {count:6d}  {raw:12.4e}  {fixed:12.4e}  {frac:10.4f}")

        print(f"  {'-' * 28}  {'-' * 6}  {'-' * 12}  {'-' * 12}  {'-' * 10}")
        count = result["dof_counts"]["total"]
        raw = result["raw"]["total"]
        fixed = result["fixed"]["total"]
        print(f"  {'TOTAL':<28s}  {count:6d}  {raw:12.4e}  {fixed:12.4e}")

    return result
