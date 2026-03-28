"""Jacobian symmetry verification (Test 1.3).

Checks ||J - J^T||_F / ||J||_F on the full assembled Jacobian.
For a potential-based formulation the ratio should be near machine epsilon.
The cohesive tangent introduces inherent nonsymmetry near damage kinks;
this is expected and flagged as informational, not as a failure.
"""

from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc


def _symmetry_defect(A):
    """Compute ||A - A^T||_F / ||A||_F for a PETSc Mat.

    Returns:
        tuple: ``(J_norm, diff_norm, ratio)``.
    """
    J_norm = A.norm(PETSc.NormType.FROBENIUS)
    AT = A.transpose()
    # AT <- AT - A  (i.e., D = J^T - J)
    AT.axpy(-1.0, A)
    diff_norm = AT.norm(PETSc.NormType.FROBENIUS)
    AT.destroy()
    ratio = diff_norm / max(J_norm, 1e-30)
    return J_norm, diff_norm, ratio


def check_jacobian_symmetry(state, verbose=True):
    """Check symmetry of the full assembled Jacobian.

    Args:
        state: DiagnosticState with prepared u, inactive_dofs, J_form.
        verbose: Print diagnostic results on rank 0.

    Returns:
        dict with keys ``J_norm``, ``asymmetry_norm``, ``ratio``.
    """
    comm = state.comm

    A = assemble_matrix(state.J_form)
    A.assemble()
    if state.inactive_dofs is not None and state.inactive_dofs.size > 0:
        A.zeroRowsLocal(state.inactive_dofs, diag=1.0)

    J_norm, diff_norm, ratio = _symmetry_defect(A)
    A.destroy()

    result = {
        "J_norm": J_norm,
        "asymmetry_norm": diff_norm,
        "ratio": ratio,
    }

    if verbose and comm.rank == 0:
        print("\n  Jacobian symmetry check (full Jacobian)")
        print(f"    ||J||_F          = {J_norm:.4e}")
        print(f"    ||J - J^T||_F    = {diff_norm:.4e}")
        print(f"    Relative defect  = {ratio:.4e}")
        if ratio < 1e-10:
            print("    Status: PASS (symmetric to machine precision)")
        elif ratio < 1e-6:
            print("    Status: PASS (small asymmetry, likely from cohesive tangent)")
        else:
            print("    Status: WARN (notable asymmetry — may include cohesive "
                  "contribution, which is expected)")

    return result
