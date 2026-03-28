"""Finite-difference Jacobian column spot-check (Test 1.4).

For selected DOFs j, computes
    J_FD[:, j] = (F(u + h*e_j) - F(u - h*e_j)) / (2h)
and compares to the corresponding column from the assembled analytical
Jacobian.  Uses two step sizes (h=1e-5, h=1e-7) to distinguish
truncation error from floating-point cancellation.
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc

from diagnostics.taylor_test import _assemble_residual_into


def select_representative_dofs(state, n_per_region=1, seed=42):
    """Select DOFs from different mesh regions for spot-checking.

    Picks one DOF each from:
      - An interior active cell (bulk)
      - A cell adjacent to an inter-layer facet (cohesive / dS(1))
      - A cell adjacent to an intra-layer facet (SIP / dS(2))
      - A cell on a Dirichlet boundary (Nitsche / ds(1))

    Cycles through x, y, z components.

    Args:
        state: DiagnosticState.
        n_per_region: How many DOFs per region.
        seed: Random seed for tie-breaking.

    Returns:
        list[dict]: Each has ``dof_index`` (local), ``description``.
    """
    msh = state.msh
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Active cells at current t_val.
    active_cells = np.where(state.birth_times_dolfinx <= state.t_val)[0]
    if active_cells.size == 0:
        return []

    # Build facet-to-cell connectivity.
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    # Classify cells by the facet types they touch.
    interior_tags = state.interior_facet_tags
    tag_values = interior_tags.values
    tag_indices = interior_tags.indices

    cohesive_cells = set()
    bonded_cells = set()
    for i, facet in enumerate(tag_indices):
        cells = f_to_c.links(facet)
        if tag_values[i] == 1:
            cohesive_cells.update(cells)
        elif tag_values[i] == 2:
            bonded_cells.update(cells)

    # Boundary cells.
    dirichlet_tags = state.dirichlet_tags
    boundary_cells = set()
    if dirichlet_tags is not None:
        for facet in dirichlet_tags.indices:
            cells = f_to_c.links(facet)
            boundary_cells.update(cells)

    active_set = set(active_cells.tolist())
    cohesive_active = sorted(active_set & cohesive_cells)
    bonded_active = sorted(active_set & bonded_cells)
    boundary_active = sorted(active_set & boundary_cells)
    # Interior = active but not on any special facet.
    special = cohesive_cells | bonded_cells | boundary_cells
    interior_active = sorted(active_set - special)

    rng = np.random.RandomState(seed)
    results = []
    component = 0  # cycle through x=0, y=1, z=2

    def _pick(cell_list, desc):
        nonlocal component
        if not cell_list:
            return
        cell = cell_list[rng.randint(len(cell_list))]
        cell_dofs = state.cell_to_dofs[cell]
        # Pick a specific vector component DOF.
        dof = int(cell_dofs[component])
        results.append({"dof_index": dof, "description": desc,
                        "cell_index": cell, "component": component})
        component = (component + 1) % 3

    _pick(interior_active, "interior bulk (active)")
    _pick(cohesive_active, "inter-layer facet (cohesive)")
    _pick(bonded_active, "intra-layer facet (SIP)")
    _pick(boundary_active, "Dirichlet boundary (Nitsche)")

    return results


def _fd_jacobian_column(state, dof_index, h):
    """Compute one Jacobian column by central finite differences.

    Args:
        state: DiagnosticState.
        dof_index: Local DOF index to perturb.
        h: FD step size.

    Returns:
        np.ndarray: FD column (owned entries only).
    """
    u = state.u
    u0 = u.x.array.copy()
    F_form = state.F_form

    b = create_vector(F_form)

    # Forward: u + h*e_j
    u.x.array[:] = u0
    u.x.array[dof_index] += h
    u.x.scatter_forward()
    _assemble_residual_into(b, F_form)
    F_plus = b.array.copy()

    # Backward: u - h*e_j
    u.x.array[:] = u0
    u.x.array[dof_index] -= h
    u.x.scatter_forward()
    _assemble_residual_into(b, F_form)
    F_minus = b.array.copy()

    # Restore.
    u.x.array[:] = u0
    u.x.scatter_forward()

    b.destroy()

    return (F_plus - F_minus) / (2.0 * h)


def _analytical_jacobian_column(A, dof_index, state):
    """Extract column j from the assembled Jacobian via MatVec with e_j.

    Args:
        A: Assembled PETSc Jacobian matrix.
        dof_index: Local column DOF index.
        state: DiagnosticState (for vector template).

    Returns:
        np.ndarray: Analytical column (owned entries).
    """
    e_j = A.createVecRight()
    e_j.set(0.0)
    e_j.array[dof_index] = 1.0
    # Ghost synchronization so other ranks see the unit entry if needed.
    e_j.assemblyBegin()
    e_j.assemblyEnd()

    col = A.createVecLeft()
    A.mult(e_j, col)

    result = col.array.copy()
    e_j.destroy()
    col.destroy()
    return result


def run_fd_column_check(state, dof_indices=None, h_values=None, verbose=True):
    """Run FD column check for selected DOFs at multiple step sizes.

    Args:
        state: DiagnosticState with prepared u and inactive_dofs.
        dof_indices: List of dicts from ``select_representative_dofs``.
            If None, auto-selects.
        h_values: FD step sizes.  Default: ``[1e-5, 1e-7]``.
        verbose: Print comparison table on rank 0.

    Returns:
        dict with keys ``dof_info``, ``h_values``, ``errors`` (nested
        list: ``errors[i_dof][i_h]``).
    """
    comm = state.comm

    if dof_indices is None:
        dof_indices = select_representative_dofs(state)
    if h_values is None:
        h_values = [1e-5, 1e-7]

    # Assemble the analytical Jacobian once.
    A = assemble_matrix(state.J_form)
    A.assemble()

    errors = []
    col_norms_analytical = []
    col_norms_fd = []

    for dof_info in dof_indices:
        dof = dof_info["dof_index"]
        desc = dof_info["description"]

        col_a = _analytical_jacobian_column(A, dof, state)

        dof_errors = []
        dof_fd_norms = []
        for h in h_values:
            col_fd = _fd_jacobian_column(state, dof, h)

            diff = col_a - col_fd
            local_diff_sq = float(np.dot(diff, diff))
            local_fd_sq = float(np.dot(col_fd, col_fd))

            global_diff_sq = comm.allreduce(local_diff_sq, op=MPI.SUM)
            global_fd_sq = comm.allreduce(local_fd_sq, op=MPI.SUM)

            diff_norm = np.sqrt(global_diff_sq)
            fd_norm = np.sqrt(global_fd_sq)
            rel_error = diff_norm / max(fd_norm, 1e-30)

            dof_errors.append(rel_error)
            dof_fd_norms.append(fd_norm)

        local_a_sq = float(np.dot(col_a, col_a))
        global_a_sq = comm.allreduce(local_a_sq, op=MPI.SUM)
        a_norm = np.sqrt(global_a_sq)

        errors.append(dof_errors)
        col_norms_analytical.append(a_norm)
        col_norms_fd.append(dof_fd_norms)

    A.destroy()

    if verbose and comm.rank == 0:
        print("\n  FD Jacobian column spot-check")
        h_headers = "  ".join(f"err(h={h:.0e})" for h in h_values)
        print(f"  {'DOF':>6s}  {'Region':<30s}  {'||J[:,j]||':>10s}  {h_headers}  Assessment")
        print(f"  {'---':>6s}  {'---':<30s}  {'---':>10s}  "
              + "  ".join("---" + " " * 8 for _ in h_values) + "  ---")

        for i, dof_info in enumerate(dof_indices):
            dof = dof_info["dof_index"]
            desc = dof_info["description"]
            a_norm = col_norms_analytical[i]

            err_strs = []
            for j, h in enumerate(h_values):
                err_strs.append(f"{errors[i][j]:11.4e}")

            # Assessment: check consistency between h values.
            if len(h_values) >= 2:
                e0, e1 = errors[i][0], errors[i][1]
                if max(e0, e1) < 1e-5:
                    assessment = "PASS"
                elif e1 > 10 * e0:
                    assessment = "cancellation"
                elif e0 > 10 * e1:
                    assessment = "truncation"
                else:
                    assessment = "FAIL"
            else:
                assessment = "PASS" if errors[i][0] < 1e-4 else "FAIL"

            print(f"  {dof:6d}  {desc:<30s}  {a_norm:10.4e}  "
                  + "  ".join(err_strs) + f"  {assessment}")

    result = {
        "dof_info": dof_indices,
        "h_values": h_values,
        "errors": errors,
        "col_norms_analytical": col_norms_analytical,
        "col_norms_fd": col_norms_fd,
    }
    return result
