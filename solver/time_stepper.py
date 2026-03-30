# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Time-stepping driver for the modular 3DCP finite element simulation.

This module owns the transient solve loop once the mesh, weak form, and state
fields are already constructed by the orchestrator. It integrates the major
runtime phases of the simulation pipeline:

1. Activation-aware displacement initialization for newly born cells.
2. Nonlinear Newton solve for the global equilibrium step.
3. DG0 tensor projection for strain post-processing.
4. Explicit J2-Perzyna constitutive update (cell-wise local integration).
5. MPI-safe global diagnostics and output I/O.

Physics:
    The loop combines an implicit global solve for displacement with an
    explicit local constitutive update for viscoplastic strain/stress fields.

Parallel Notes:
    MPI collective calls and ghost synchronization are intentionally preserved.
    In particular, all calls to `allreduce`, `scatter_forward`, and
    `ghostUpdate` remain at their original synchronization points.
"""

import csv
import os
import resource
import time
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.io import VTXWriter
from mpi4py import MPI
from petsc4py import PETSc

from config.config_utils import get_checkpoint_dir
from materials.material_state import update_perzyna_state_cellwise
from solver.kinematics import (
    epsilon,
    init_newly_activated_displacement,
    update_active_indicator,
    zero_inactive_cells,
)
from solver.newton import NewtonLinearWorkspace, solve_newton


def build_cell_to_dofs_and_support(msh, V, vault_mesh, cells_lst, perm):
    """Build cell-to-DOF connectivity and support-cell map.

    Args:
        msh: Distributed DOLFINx mesh.
        V: Displacement function space.
        vault_mesh: Original custom barrel-vault mesh object.
        cells_lst: Flat list of original custom cells.
        perm: DOLFINx-to-original cell permutation array.

    Returns:
        tuple[np.ndarray, np.ndarray, int]:
            - `cell_to_dofs`: `(n_cells, dofs_per_cell)` vector-DOF map.
            - `support`: Per-cell support mapping for inter-layer inheritance.
            - `support_missing`: Number of unresolved support links.

    Raises:
        None.
    """
    map_c = msh.topology.index_map(msh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts

    bs = V.dofmap.bs
    cell_dofs_scalar = np.array(
        [V.dofmap.cell_dofs(cell) for cell in range(num_cells)], dtype=np.int32
    )
    # Expand scalar dof ids to interleaved vector dofs:
    # scalar dof d -> [d*bs + 0, d*bs + 1, ..., d*bs + bs-1]
    cell_to_dofs = (
        cell_dofs_scalar[:, :, None] * bs + np.arange(bs, dtype=np.int32)[None, None, :]
    ).reshape(num_cells, -1)

    original_to_dolfinx = np.full(len(cells_lst), -1, dtype=np.int32)
    original_to_dolfinx[perm] = np.arange(num_cells, dtype=np.int32)
    custom_cell_to_original = {id(cell): i for i, cell in enumerate(cells_lst)}
    support = np.full(num_cells, -1, dtype=np.int32)

    support_missing = 0
    for iface in vault_mesh.inter_layer_interfaces:
        lower_cell, upper_cell = iface.get_connected_cells()
        if lower_cell is None or upper_cell is None:
            continue

        lower_original = custom_cell_to_original.get(id(lower_cell), -1)
        upper_original = custom_cell_to_original.get(id(upper_cell), -1)
        if upper_original < 0:
            continue

        upper_dolfinx = int(original_to_dolfinx[upper_original])
        if upper_dolfinx < 0:
            continue

        if lower_original < 0:
            support_missing += 1
            continue

        lower_dolfinx = int(original_to_dolfinx[lower_original])
        if lower_dolfinx < 0:
            support_missing += 1
            continue

        support[upper_dolfinx] = lower_dolfinx

    return cell_to_dofs, support, support_missing


def _print_step_diagnostics(
    comm, num_owned_cells, step, t_val, birth_times, log_message=None
):
    """Print global active-cell diagnostics for the current step.

    Args:
        comm: MPI communicator.
        num_owned_cells: Number of owned cells on this rank.
        step: Current time-step index.
        t_val: Current simulation time.
        birth_times: Birth-time array in DOLFINx ordering.
        log_message: Optional callable used for rank-0 logging.

    Returns:
        None.

    Raises:
        None.
    """
    active_owned = birth_times[:num_owned_cells] <= t_val
    n_active_local = int(np.sum(active_owned))
    n_total_local = int(num_owned_cells)
    n_active = comm.allreduce(n_active_local, op=MPI.SUM)
    n_total = comm.allreduce(n_total_local, op=MPI.SUM)

    if comm.rank == 0:
        if log_message is None:
            print(f"\n{'=' * 72}")
            print(f"  STEP {step}:  t = {t_val:.4f} s")
            print(f"{'=' * 72}")
            print(
                f"  Active cells: {n_active}/{n_total} ({100 * n_active / n_total:.1f}%)",
                flush=True,
            )
        else:
            log_message(f"\n{'=' * 72}")
            log_message(f"  STEP {step}:  t = {t_val:.4f} s")
            log_message(f"{'=' * 72}")
            log_message(
                f"  Active cells: {n_active}/{n_total} ({100 * n_active / n_total:.1f}%)"
            )


_CSV_COLUMNS = [
    # ── Step identification ──
    "step",
    "time_s",
    "dt_s",
    # ── Mesh / parallel ──
    "mpi_ranks_total",
    "mpi_ranks_active",
    "active_cells",
    "total_cells",
    "active_dofs",
    "inactive_dofs",
    # ── Newton solver ──
    "newton_iters",
    "converged",
    "newton_msg",
    "res0",
    "res_final",
    "rel_res_final",
    "total_ksp_its",
    "total_ls_its",
    "final_omega",
    # ── Timings ──
    "time_newton_s",
    "time_perzyna_s",
    "time_proj_s",
    "time_io_s",
    "time_step_total_s",
    # ── Constitutive ──
    "n_sub_steps",
    "yielding_cells",
    "max_yield_f_MPa",
    # ── Field statistics ──
    "max_disp_mm",
    "z_sag_mm",
    "max_von_mises_MPa",
    "max_principal_MPa",
    "max_plastic_strain",
    # ── Stability / resource ──
    "dt_stable_limit_s",
    "rss_GiB",
    # ── Cumulatives ──
    "cumul_newton_iters",
    "cumul_ksp_its",
    "cumul_wall_s",
]


def _current_rss_gib() -> float:
    """Return current process RSS in GiB (platform-aware)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kB.
    import sys
    if sys.platform == "darwin":
        return ru / (1024 ** 3)
    return ru * 1024 / (1024 ** 3)


def run_simulation(
    msh,
    V,
    u,
    F_form,
    J_form,
    materials,
    birth_times_dolfinx,
    birth_time_func,
    is_active_func,
    cells_layers_func,
    mpi_rank_func,
    cell_to_dofs,
    support,
    cfg,
    disp_path,
    cell_path,
    log_path,
    interior_facet_tags=None,
    simulation_start_time=None,
):
    """Execute the full transient simulation loop.

    Args:
        msh: Distributed DOLFINx mesh.
        V: Displacement function space.
        u: Displacement solution function (updated in place).
        F_form: Compiled nonlinear residual form.
        J_form: Compiled nonlinear Jacobian form.
        materials: Material state manager with DG0/DG0-tensor fields.
        birth_times_dolfinx: Cell birth-time array in DOLFINx ordering.
        birth_time_func: DG0 output field of cell birth times.
        is_active_func: DG0 output field for active/inactive status.
        cells_layers_func: DG0 output field for layer id.
        mpi_rank_func: DG0 output field for owning MPI rank.
        cell_to_dofs: Map from cell id to vector DOF ids.
        support: Support-cell mapping for newly activated cells.
        cfg: Full simulation configuration dictionary.
        disp_path: Output path for displacement VTX writer.
        cell_path: Output path for cell-data VTX writer.
        log_path: Rank-0 log file path.
        simulation_start_time: Optional wall-clock start time for the full
            simulation workflow.

    Returns:
        None.

    Raises:
        RuntimeError: If the global birth-time range is invalid.

    Physics:
        - Global displacement is solved implicitly each step (Newton method).
        - Cell-wise viscoplastic state is updated explicitly using the
          J2-Perzyna flow rule.

    Math:
        The constitutive explicit stability monitor uses:
            dt_sub <= eta / E
        evaluated on active cells.

    Parallel Notes:
        All MPI collectives and ghost synchronizations are intentionally kept in
        their original places to preserve rank consistency and avoid deadlocks.
    """
    comm = msh.comm
    map_c = msh.topology.index_map(msh.topology.dim)
    num_owned_cells = map_c.size_local

    def log_message(message=""):
        """Print and append a Rank-0 message to the simulation log.

        Args:
            message: Message line to print and append.

        Returns:
            None.

        Raises:
            OSError: If log file append fails on rank 0.
        """
        if comm.rank != 0:
            return
        tee_active = os.environ.get("SIM_TEE_ACTIVE") == "1"
        print(message, flush=True)
        if tee_active:
            # In tee mode, printing already mirrors output to the log file.
            return
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{message}\n")

    dx = ufl.Measure("dx", domain=msh)

    # Step 12: DG0 tensor projection setup.
    # The DG0 mass matrix is diagonal, so L2 projection reduces to element-wise
    # scaling rather than solving a global linear system.
    proj_trial = ufl.TrialFunction(materials.V_DG0_tensor)
    proj_test = ufl.TestFunction(materials.V_DG0_tensor)

    a_proj_form = fem.form(ufl.inner(proj_trial, proj_test) * dx)
    A_proj = assemble_matrix(a_proj_form)
    A_proj.assemble()

    diag_proj = A_proj.getDiagonal()
    inv_diag_proj = diag_proj.copy()
    # DG0 mass matrix is diagonal (cell-wise constant basis); inversion is exact.
    inv_diag_proj.array[:] = 1.0 / np.maximum(inv_diag_proj.array, 1.0e-30)

    L_strain_form = fem.form(ufl.inner(epsilon(u), proj_test) * dx)
    b_strain = create_vector(L_strain_form)

    def project_tensor_to_dg0(L_form, b_vec, out_func):
        """Project a tensor field to DG0 by diagonal mass scaling.

        Args:
            L_form: Linear form for projection RHS.
            b_vec: PETSc vector used as RHS buffer.
            out_func: DG0 tensor function receiving projected values.

        Returns:
            None.

        Raises:
            None.

        Math:
            Find q in DG0 such that (q, w) = (rhs, w) for all DG0 test w.
            With DG0 basis, the mass matrix M is diagonal, so q = M^{-1} b.
        """
        with b_vec.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b_vec, L_form)
        # Reverse ghost accumulation is required before reading RHS values that
        # include contributions from shared entities across ranks.
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        out_func.x.petsc_vec.pointwiseMult(inv_diag_proj, b_vec)
        # Forward scatter makes projected DG0 values consistent on ghost cells.
        out_func.x.scatter_forward()

    time_cfg = cfg["time_stepping"]
    solver_cfg = cfg["solver"]
    checkpoint_cfg = cfg["checkpoint"]
    debug_cfg = cfg.get("debug", {})
    checkpoint_save_every = max(1, int(checkpoint_cfg["save_every"]))
    checkpoint_resume_enabled = bool(checkpoint_cfg["resume_enabled"])
    checkpoint_dir = get_checkpoint_dir(cfg)
    comm.Barrier()

    latest_checkpoint_filename = "checkpoint_latest.npz"

    def write_checkpoint_metadata(checkpoint_dir, step, t_val, checkpoint_filename):
        """Persist latest checkpoint descriptor for automated restarts."""
        if comm.rank != 0:
            return
        checkpoint_basename = os.path.basename(str(checkpoint_filename))
        metadata = {
            "step": int(step),
            "time": float(t_val),
            "file_name": checkpoint_basename,
            # Keep legacy key for backward compatibility with older readers.
            "file_prefix": os.path.splitext(checkpoint_basename)[0],
        }
        metadata_path = checkpoint_dir / "latest_checkpoint.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")

    def read_checkpoint_metadata(checkpoint_dir):
        """Read latest checkpoint metadata on rank 0 and broadcast."""
        metadata = None
        metadata_path = checkpoint_dir / "latest_checkpoint.json"
        if comm.rank == 0 and metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        return comm.bcast(metadata, root=0)

    def _field_checkpoint_layout(field):
        """Safely map blocked local DOFs to flattened global scalar DOFs."""
        index_map = field.function_space.dofmap.index_map
        bs = field.function_space.dofmap.index_map_bs
        
        # 1. Get the number of locally owned blocks (nodes)
        num_blocks_local = index_map.size_local
        
        # The true flattened sizes
        size_local = num_blocks_local * bs
        size_global = index_map.size_global * bs
        
        # 2. Generate local block indices (0, 1, 2, ..., N_blocks - 1)
        local_block_indices = np.arange(num_blocks_local, dtype=np.int32)
        
        # 3. Map local blocks to global block IDs safely
        global_block_indices = index_map.local_to_global(local_block_indices)
        
        # 4. Expand global block IDs to flattened scalar DOF IDs
        # e.g., block 5 with bs=3 becomes scalar DOFs 15, 16, 17
        global_indices = (
            global_block_indices[:, None] * bs + np.arange(bs, dtype=np.int64)[None, :]
        ).flatten()
        
        return size_local, size_global, global_indices

    def _assemble_global_from_owned_chunks(
        gathered_indices,
        gathered_values,
        expected_size,
        field_label,
    ):
        all_indices = np.concatenate(gathered_indices).astype(np.int64, copy=False)
        all_values = np.concatenate(gathered_values)
        order = np.argsort(all_indices)
        sorted_indices = all_indices[order]
        sorted_values = all_values[order]
        if sorted_indices.size != expected_size:
            raise RuntimeError(
                f"Checkpoint gather size mismatch for {field_label}: "
                f"{sorted_indices.size} vs expected {expected_size}."
            )
        if sorted_indices.size > 0:
            unique_indices = np.unique(sorted_indices)
            if unique_indices.size != expected_size:
                raise RuntimeError(
                    f"Checkpoint gather contains duplicate/missing global indices "
                    f"for {field_label}: unique={unique_indices.size}, "
                    f"expected={expected_size}."
                )
            if unique_indices[0] < 0 or unique_indices[-1] >= expected_size:
                raise RuntimeError(
                    f"Checkpoint gather contains out-of-range global indices for "
                    f"{field_label}."
                )
        global_values = np.empty(expected_size, dtype=sorted_values.dtype)
        global_values[sorted_indices] = sorted_values
        return global_values

    u_size_local, u_size_global, u_global_indices = _field_checkpoint_layout(u)
    eps_vp = materials.eps_vp
    eps_vp_size_local, eps_vp_size_global, eps_vp_global_indices = _field_checkpoint_layout(
        eps_vp
    )
    damage_max_field = materials.damage_max
    dmg_size_local, dmg_size_global, dmg_global_indices = _field_checkpoint_layout(
        damage_max_field
    )

    # ------------------------------------------------------------------
    # Precompute inter-layer facet data for damage_max update.
    # ------------------------------------------------------------------
    n_il = 0
    il_cell_plus = np.empty(0, dtype=np.int32)
    il_cell_minus = np.empty(0, dtype=np.int32)
    il_facet_normals = np.empty((0, 3), dtype=np.float64)
    il_h_cells = np.empty(0, dtype=np.float64)
    if interior_facet_tags is not None:
        _il_mask = interior_facet_tags.values == 1
        _il_facet_indices = interior_facet_tags.indices[_il_mask]
        n_il = len(_il_facet_indices)
        if n_il > 0:
            tdim = msh.topology.dim
            fdim = tdim - 1
            msh.topology.create_connectivity(fdim, tdim)
            msh.topology.create_connectivity(fdim, 0)
            _f_to_c = msh.topology.connectivity(fdim, tdim)
            _f_to_v = msh.topology.connectivity(fdim, 0)
            il_cell_plus = np.empty(n_il, dtype=np.int32)
            il_cell_minus = np.empty(n_il, dtype=np.int32)
            for _i, _f in enumerate(_il_facet_indices):
                _cells = _f_to_c.links(_f)
                il_cell_plus[_i] = _cells[0]
                il_cell_minus[_i] = _cells[1] if len(_cells) > 1 else _cells[0]

            # Compute per-facet unit normals from vertex coordinates.
            from dolfinx.mesh import entities_to_geometry as _e2g
            _geom_x = msh.geometry.x
            il_facet_normals = np.empty((n_il, 3), dtype=np.float64)
            for _i, _f in enumerate(_il_facet_indices):
                _verts = _f_to_v.links(_f)
                _geom_dofs = _e2g(
                    msh, 0, _verts.astype(np.int32), False
                ).reshape(-1)
                _coords = _geom_x[_geom_dofs]
                _normal = np.cross(_coords[1] - _coords[0], _coords[3] - _coords[0])
                _norm = np.linalg.norm(_normal)
                il_facet_normals[_i] = _normal / _norm if _norm > 1e-12 else _normal

            # Precompute cell diameters.
            from dolfinx.cpp.mesh import h as _dolfinx_h
            _map_c = msh.topology.index_map(tdim)
            _all_cells = np.arange(
                _map_c.size_local + _map_c.num_ghosts, dtype=np.int32
            )
            il_h_cells = np.asarray(_dolfinx_h(msh._cpp_object, tdim, _all_cells))

    def save_checkpoint(step_idx, t_val):
        """Save a single rank-agnostic checkpoint file for the current step."""
        u_local_values = u.x.array[:u_size_local].copy()
        eps_vp_local_values = eps_vp.x.array[:eps_vp_size_local].copy()
        dmg_local_values = damage_max_field.x.array[:dmg_size_local].copy()
        gathered_u_indices = comm.gather(u_global_indices, root=0)
        gathered_u_values = comm.gather(u_local_values, root=0)
        gathered_eps_indices = comm.gather(eps_vp_global_indices, root=0)
        gathered_eps_values = comm.gather(eps_vp_local_values, root=0)
        gathered_dmg_indices = comm.gather(dmg_global_indices, root=0)
        gathered_dmg_values = comm.gather(dmg_local_values, root=0)

        if comm.rank == 0:
            u_global = _assemble_global_from_owned_chunks(
                gathered_u_indices,
                gathered_u_values,
                u_size_global,
                "u",
            )
            eps_vp_global = _assemble_global_from_owned_chunks(
                gathered_eps_indices,
                gathered_eps_values,
                eps_vp_size_global,
                "eps_vp",
            )
            dmg_global = _assemble_global_from_owned_chunks(
                gathered_dmg_indices,
                gathered_dmg_values,
                dmg_size_global,
                "damage_max",
            )
            checkpoint_file = checkpoint_dir / latest_checkpoint_filename
            np.savez(
                checkpoint_file,
                u=u_global,
                eps_vp=eps_vp_global,
                damage_max=dmg_global,
            )
            write_checkpoint_metadata(
                checkpoint_dir,
                step_idx,
                t_val,
                checkpoint_file.name,
            )
            # Remove legacy per-step checkpoint files from older runs.
            for old_file in checkpoint_dir.glob("checkpoint_step_*.npz"):
                try:
                    old_file.unlink()
                except OSError:
                    pass
            log_message(
                "  Checkpoint saved: "
                f"step={int(step_idx)}, time={float(t_val):.6f} s, "
                f"file={checkpoint_file}"
            )
        comm.Barrier()

    def restore_checkpoint(checkpoint_path):
        """Restore distributed fields from a rank-agnostic global checkpoint."""
        if comm.rank == 0:
            with np.load(checkpoint_path) as data:
                if "u" not in data or "eps_vp" not in data:
                    raise KeyError(
                        "Checkpoint file must contain 'u' and 'eps_vp' arrays."
                    )
                u_global = np.asarray(data["u"], dtype=u.x.array.dtype)
                eps_vp_global = np.asarray(data["eps_vp"], dtype=eps_vp.x.array.dtype)
                # Backwards compatibility: old checkpoints lack damage_max.
                if "damage_max" in data:
                    dmg_global = np.asarray(
                        data["damage_max"], dtype=damage_max_field.x.array.dtype
                    )
                else:
                    dmg_global = np.zeros(dmg_size_global, dtype=damage_max_field.x.array.dtype)
        else:
            u_global = np.empty(0, dtype=u.x.array.dtype)
            eps_vp_global = np.empty(0, dtype=eps_vp.x.array.dtype)
            dmg_global = np.empty(0, dtype=damage_max_field.x.array.dtype)

        u_count = np.array([u_global.size], dtype=np.int64)
        eps_count = np.array([eps_vp_global.size], dtype=np.int64)
        dmg_count = np.array([dmg_global.size], dtype=np.int64)
        comm.Bcast(u_count, root=0)
        comm.Bcast(eps_count, root=0)
        comm.Bcast(dmg_count, root=0)

        if comm.rank != 0:
            u_global = np.empty(int(u_count[0]), dtype=u.x.array.dtype)
            eps_vp_global = np.empty(int(eps_count[0]), dtype=eps_vp.x.array.dtype)
            dmg_global = np.empty(int(dmg_count[0]), dtype=damage_max_field.x.array.dtype)

        comm.Bcast(u_global, root=0)
        comm.Bcast(eps_vp_global, root=0)
        comm.Bcast(dmg_global, root=0)

        if int(u_count[0]) != u_size_global:
            raise RuntimeError(
                f"Checkpoint u size mismatch: {int(u_count[0])} vs {u_size_global}."
            )
        if int(eps_count[0]) != eps_vp_size_global:
            raise RuntimeError(
                "Checkpoint eps_vp size mismatch: "
                f"{int(eps_count[0])} vs {eps_vp_size_global}."
            )

        u.x.array[:u_size_local] = u_global[u_global_indices]
        eps_vp.x.array[:eps_vp_size_local] = eps_vp_global[eps_vp_global_indices]
        if int(dmg_count[0]) == dmg_size_global:
            damage_max_field.x.array[:dmg_size_local] = dmg_global[dmg_global_indices]
        u.x.scatter_forward()
        eps_vp.x.scatter_forward()
        damage_max_field.x.scatter_forward()

    n_steps = time_cfg["n_steps"]
    local_birth_owned = birth_times_dolfinx[:num_owned_cells]
    local_t_min = float(np.min(local_birth_owned)) if num_owned_cells > 0 else np.inf
    local_t_max = float(np.max(local_birth_owned)) if num_owned_cells > 0 else -np.inf
    global_t_min = comm.allreduce(local_t_min, op=MPI.MIN)
    global_t_max = comm.allreduce(local_t_max, op=MPI.MAX)

    if not (np.isfinite(global_t_min) and np.isfinite(global_t_max)):
        raise RuntimeError("Invalid global birth-time range for time stepping.")

    full_times = np.linspace(
        global_t_min + time_cfg["start_offset"],
        global_t_max * time_cfg["end_multiplier"],
        n_steps,
    )
    max_steps = int(time_cfg.get("max_steps", 0))
    if max_steps > 0:
        full_times = full_times[:max_steps]
    times = full_times.copy()
    start_idx = 0
    resume_step = None
    resume_time = None
    if checkpoint_resume_enabled:
        metadata_path = checkpoint_dir / "latest_checkpoint.json"
        metadata = read_checkpoint_metadata(checkpoint_dir)
        if metadata is None:
            raise FileNotFoundError(
                "Checkpoint resume requested but metadata was not found: "
                f"{metadata_path}"
            )
        try:
            resume_step = int(metadata["step"])
            resume_time = float(metadata["time"])
            checkpoint_name = metadata.get("file_name")
            if checkpoint_name is None:
                file_prefix = str(metadata["file_prefix"])
                checkpoint_name = f"{file_prefix}.npz"
            checkpoint_name = os.path.basename(str(checkpoint_name))
            if not checkpoint_name.endswith(".npz"):
                checkpoint_name = f"{checkpoint_name}.npz"
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Invalid checkpoint metadata. Expected keys: "
                "{'step': int, 'time': float, and one of "
                "'file_name' or 'file_prefix'}."
            ) from exc

        checkpoint_path = checkpoint_dir / checkpoint_name
        checkpoint_exists = (
            checkpoint_path.exists()
            if comm.rank == 0
            else None
        )
        checkpoint_exists = comm.bcast(checkpoint_exists, root=0)
        if not checkpoint_exists:
            raise FileNotFoundError(
                "Checkpoint resume requested but data file was not found: "
                f"{checkpoint_path}"
            )
        if comm.rank == 0:
            print(
                "Resuming from checkpoint metadata: "
                f"step={resume_step}, time={resume_time:.6f} s, file={checkpoint_path}",
                flush=True,
            )
        restore_checkpoint(checkpoint_path)
        times = full_times[full_times > resume_time]
        start_idx = int(np.searchsorted(full_times, resume_time, side="right"))

    dt_default = (full_times[1] - full_times[0]) if n_steps > 1 else 1.0
    collective_debug = bool(debug_cfg.get("collective_debug", False))
    collective_debug_max_iter = max(
        1, int(debug_cfg.get("collective_debug_max_iter", 2))
    )
    collective_debug_barrier = bool(debug_cfg.get("collective_debug_barrier", False))
    debug_output_io = bool(debug_cfg.get("output_io", False))
    step_setup_debug = bool(debug_cfg.get("step_setup", False))
    step_setup_debug_max_steps = max(
        1, int(debug_cfg.get("step_setup_max_steps", 3))
    )
    newton_memory_debug = bool(debug_cfg.get("newton_memory_tracking", True))
    newton_memory_every_iter = max(
        1, int(debug_cfg.get("newton_memory_every_iter", 1))
    )
    newton_memory_collect_garbage = bool(
        debug_cfg.get("newton_memory_collect_garbage", False)
    )
    newton_memory_track_mumps = bool(
        debug_cfg.get("newton_memory_track_mumps", True)
    )

    total_newton_iters = 0
    total_ksp_its_all = 0
    total_perzyna_steps = 0
    total_time_newton = 0.0
    total_time_perzyna = 0.0
    total_time_proj = 0.0
    total_time_io = 0.0

    final_global_max_u = 0.0
    final_global_min_z = 0.0
    final_global_max_vm = 0.0
    final_global_max_ps = 0.0
    final_global_yielding = 0
    newton_workspace = None

    def step_probe(step, label):
        """Optionally print rank-local debug progress during step setup.

        Args:
            step: Time-step index.
            label: Human-readable probe label.

        Returns:
            None.

        Raises:
            None.
        """
        if (not step_setup_debug) or (step >= step_setup_debug_max_steps):
            return
        print(f"[DBG][rank {comm.rank}] step {step}: {label}", flush=True)
        if collective_debug_barrier:
            # Optional synchronization barrier to localize parallel stalls in
            # debug mode. Kept conditional to avoid production overhead.
            comm.Barrier()

    if comm.rank == 0 and os.environ.get("SIM_TEE_ACTIVE") != "1":
        # Start each run with a fresh log file.
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("")

    if times.size == 0:
        if comm.rank == 0:
            log_message("\nTime stepping: 0 steps")
            if checkpoint_resume_enabled:
                log_message(
                    "  Resume source: "
                    f"step={resume_step}, time={resume_time:.6f} s"
                )
                log_message(
                    "  No remaining times satisfy full_times > resume_time."
                )
            else:
                log_message("  No time steps configured.")
        return

    newton_workspace = NewtonLinearWorkspace(V, msh, F_form, solver_cfg=solver_cfg)

    track_mumps_memory_primary = (
        newton_memory_track_mumps
        and (newton_workspace.linear_solver_mode == "direct")
    )

    if comm.rank == 0:
        log_message(f"\nTime stepping: {len(times)} steps")
        log_message(f"  MPI ranks: {comm.size}")
        if checkpoint_resume_enabled:
            log_message(
                "  Resume source: "
                f"step={resume_step}, time={resume_time:.6f} s"
            )
        log_message(f"  Global time range: [{times[0]:.4f}, {times[-1]:.4f}] s")
        log_message(
            f"  Linear solver mode: {newton_workspace.linear_solver_mode}"
        )
        if newton_workspace.linear_solver_mode == "iterative":
            iterative_cfg = solver_cfg.get("iterative", {})
            gmres_restart = iterative_cfg.get("gmres_restart", 30)
            log_message(
                f"  Iterative solver: gmres/bjacobi+lu on full pinned system"
                f" (restart={gmres_restart})"
            )
        if collective_debug:
            log_message(
                "  DEBUG: collective probes enabled "
                f"(max_iter={collective_debug_max_iter}, barrier={collective_debug_barrier})"
            )
        if debug_output_io:
            log_message("  DEBUG: output I/O probes enabled")
        if step_setup_debug:
            log_message(
                f"  DEBUG: step-setup probes enabled (max_steps={step_setup_debug_max_steps})"
            )
        if newton_memory_debug:
            log_message(
                "  DEBUG: Newton memory tracker enabled "
                f"(every_iter={newton_memory_every_iter}, "
                f"collect_gc={newton_memory_collect_garbage})"
            )
            if track_mumps_memory_primary:
                log_message("  DEBUG: Newton memory tracker includes MUMPS INFOG stats")
            elif newton_memory_track_mumps:
                log_message(
                    "  DEBUG: MUMPS INFOG tracking skipped (linear solver is iterative)"
                )

    newton_workspace.ensure_jacobian(J_form)
    if comm.rank == 0 and newton_memory_debug:
        log_message("  DEBUG: Newton Jacobian sparsity preallocated once and reused")

    try:
        with VTXWriter(msh.comm, str(disp_path), [u]) as vtx_disp, VTXWriter(
            msh.comm,
            str(cell_path),
            [
                birth_time_func,
                is_active_func,
                cells_layers_func,
                mpi_rank_func,
                materials.E,
                materials.G,
                materials.nu,
                materials.rho,
                materials.eta,
                materials.tau_y,
                materials.sigma_y,
                materials.yield_function_trial,
                materials.von_mises,
                materials.max_principal_stress,
                materials.strain,
                materials.stress,
                materials.eps_vp,
                materials.damage_max,
            ],
        ) as vtx_cell:
            t_prev = float(resume_time) if checkpoint_resume_enabled else (times[0] - dt_default)
            # Exact wall-clock start of the time-stepping loop (excludes setup).
            loop_start_time = time.time()

            # CSV metrics logger (rank 0 only).
            csv_path = Path(disp_path).parent / "step_metrics.csv"
            csv_file = None
            csv_writer = None
            if comm.rank == 0:
                csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLUMNS)
                csv_writer.writeheader()
                csv_file.flush()

            for local_idx, t_val in enumerate(times):
                step = start_idx + local_idx
                step_probe(step, "loop entry")
                if hasattr(materials, "t_current"):
                    materials.t_current.value = t_val
                dt = dt_default if step == 0 else max(t_val - t_prev, 0.0)

                if step == 0 and start_idx == 0:
                    u.x.array[:] = 0.0
                    step_probe(step, "before u.scatter_forward (initial zero)")
                    # Scatter after direct array edit so ghosts see consistent u.
                    u.x.scatter_forward()
                    step_probe(step, "after u.scatter_forward (initial zero)")
                else:
                    # Newly activated cells inherit rigid translation from the
                    # underlying support layer to avoid spurious jump transients.
                    newly_active_cells = np.where(
                        (birth_times_dolfinx <= t_val) & (birth_times_dolfinx > t_prev)
                    )[0]
                    step_probe(step, f"newly_active_cells={len(newly_active_cells)}")
                    init_newly_activated_displacement(
                        u,
                        newly_active_cells,
                        support,
                        cell_to_dofs,
                        birth_times=birth_times_dolfinx,
                        t_prev=t_prev,
                    )
                    step_probe(step, "after init_newly_activated_displacement")
                    if newly_active_cells.size > 0 and comm.rank == 0:
                        _new_dofs = cell_to_dofs[newly_active_cells].reshape(-1)
                        _u_new = u.x.array[_new_dofs]
                        log_message(
                            f"  Init disp: {len(newly_active_cells)} new cells, "
                            f"max|u|={np.max(np.abs(_u_new)):.4e}, "
                            f"mean|u|={np.mean(np.abs(_u_new)):.4e}"
                        )
                    newly_active_mask = np.zeros_like(birth_times_dolfinx, dtype=bool)
                    newly_active_mask[newly_active_cells] = True
                    if np.any(newly_active_mask):
                        eps_vp_arr = materials.eps_vp.x.array.reshape((-1, 3, 3))
                        eps_vp_arr[newly_active_mask, :, :] = 0.0
                        materials.damage_max.x.array[newly_active_mask] = 0.0
                    # Must be called on all ranks to avoid rank-divergent ghost updates.
                    step_probe(step, "before eps_vp.scatter_forward")
                    materials.eps_vp.x.scatter_forward()
                    materials.damage_max.x.scatter_forward()
                    step_probe(step, "after eps_vp.scatter_forward")

                t0 = time.time()
                # Age-dependent material update uses preallocated arrays and then
                # scatters all updated DG0 properties internally.
                materials.update_properties(t_val)
                _print_step_diagnostics(
                    comm,
                    num_owned_cells,
                    step,
                    t_val,
                    birth_times_dolfinx,
                    log_message=log_message,
                )

                active_mask = birth_times_dolfinx <= t_val
                dt_stable_limit = np.inf
                if np.any(active_mask):
                    # Explicit constitutive stability monitor: eta/E on active cells.
                    dt_stable_limit = np.min(
                        materials.eta_arr[active_mask]
                        / np.maximum(materials.e_arr[active_mask], 1.0e-12)
                    )

                # Rebuild forms with cell + facet restricted assembly.
                # Cell restriction: only integrate dx over active cells.
                # Facet restriction: only integrate dS over facets adjacent
                # to at least one active cell.  Combined, this makes assembly
                # cost O(active) instead of O(total_mesh).
                u_backup = u.x.array.copy()

                t_newton_start = time.time()
                n_iter, converged, msg, newton_stats = solve_newton(
                    u,
                    V,
                    msh,
                    F_form,
                    J_form,
                    t_val=t_val,
                    birth_times=birth_times_dolfinx,
                    cell_to_dofs=cell_to_dofs,
                    max_iter=solver_cfg["max_iterations"],
                    rtol=solver_cfg["rtol"],
                    atol=solver_cfg["atol"],
                    debug=(comm.rank == 0),
                    collective_debug=collective_debug,
                    collective_debug_max_iter=collective_debug_max_iter,
                    collective_debug_barrier=collective_debug_barrier,
                    workspace=newton_workspace,
                    memory_tracking=newton_memory_debug,
                    memory_tracking_every_iter=newton_memory_every_iter,
                    memory_tracking_collect_garbage=newton_memory_collect_garbage,
                    memory_tracking_mumps=track_mumps_memory_primary,
                )
                time_newton = time.time() - t_newton_start

                if not converged:
                    if comm.rank == 0:
                        log_message("  REVERTING: restoring displacement from before failed step")
                    u.x.array[:] = u_backup
                    u.x.scatter_forward()

                zero_inactive_cells(u, t_val, birth_times_dolfinx, cell_to_dofs)
                update_active_indicator(is_active_func, t_val, birth_times_dolfinx)

                t_proj_start = time.time()
                project_tensor_to_dg0(L_strain_form, b_strain, materials.strain)
                time_proj = time.time() - t_proj_start

                strain_arr = materials.strain.x.array.reshape((-1, 3, 3))
                eps_vp_prev_arr = materials.eps_vp.x.array.reshape((-1, 3, 3))
                dt_update = dt if converged else 0.0
                dt_eff = max(float(dt_update), 0.0)
                # Stability-oriented constitutive substepping:
                # choose n_sub so sub_dt <= eta/E for active cells.
                if dt_eff > 0.0 and np.isfinite(dt_stable_limit):
                    dt_limit = max(float(dt_stable_limit), 1.0e-12)
                    n_sub = max(1, int(np.ceil(dt_eff / dt_limit)))
                else:
                    n_sub = 1
                sub_dt = dt_eff / n_sub if n_sub > 0 else 0.0

                if comm.rank == 0:
                    log_message(
                        f"  Constitutive Update: integrating {n_sub} sub-steps (dt_sub = {sub_dt:.2e}s)..."
                    )

                eps_vp_tmp = eps_vp_prev_arr.copy()
                t_perzyna_start = time.time()
                progress_every = max(1, n_sub // 10)

                i_sub = 1
                if comm.rank == 0 and n_sub >= 10:
                    if (i_sub % progress_every == 0) or (i_sub == 1) or (i_sub == n_sub):
                        log_message(f"    -> sub-step {i_sub}/{n_sub}...")
                # Sub-step 1: evaluate all active cells once to identify yielding cells.
                (
                    eps_vp_tmp,
                    sigma_new_arr,
                    vm_new_arr,
                    max_ps_new_arr,
                    f_trial_arr,
                ) = update_perzyna_state_cellwise(
                    strain_total=strain_arr,
                    eps_vp_prev=eps_vp_tmp,
                    e_arr=materials.e_arr,
                    nu_arr=materials.nu_arr,
                    sigma_y_arr=materials.sigma_y_arr,
                    eta_arr=materials.eta_arr,
                    dt=sub_dt,
                    active_mask=active_mask,
                )

                yielding_mask = (f_trial_arr > 0.0) & active_mask
                n_yielding = int(np.sum(yielding_mask))
                n_active = int(np.sum(active_mask))
                n_sub_computed = n_sub if n_yielding > 0 else 1

                if comm.rank == 0:
                    log_message(
                        "  Constitutive Update: integrating "
                        f"{n_sub_computed} sub-steps (dt_sub = {sub_dt:.2e}s), "
                        f"{n_yielding} yielding cells out of {n_active} active"
                    )

                if n_yielding > 0 and n_sub > 1:
                    strain_yielding = strain_arr[yielding_mask]
                    eps_vp_yielding = eps_vp_tmp[yielding_mask]
                    e_yielding = materials.e_arr[yielding_mask]
                    nu_yielding = materials.nu_arr[yielding_mask]
                    sigma_y_yielding = materials.sigma_y_arr[yielding_mask]
                    eta_yielding = materials.eta_arr[yielding_mask]
                    f_trial_yielding = f_trial_arr[yielding_mask].copy()

                    for i_sub in range(2, n_sub + 1):
                        if comm.rank == 0 and n_sub >= 10:
                            if (i_sub % progress_every == 0) or (i_sub == 1) or (i_sub == n_sub):
                                log_message(f"    -> sub-step {i_sub}/{n_sub}...")
                        # Local explicit Perzyna return/update on yielding cells only.
                        (
                            eps_vp_yielding,
                            _sigma_new_yielding,
                            _vm_new_yielding,
                            _max_ps_new_yielding,
                            f_trial_yielding,
                        ) = update_perzyna_state_cellwise(
                            strain_total=strain_yielding,
                            eps_vp_prev=eps_vp_yielding,
                            e_arr=e_yielding,
                            nu_arr=nu_yielding,
                            sigma_y_arr=sigma_y_yielding,
                            eta_arr=eta_yielding,
                            dt=sub_dt,
                            active_mask=None,
                        )

                    eps_vp_tmp[yielding_mask] = eps_vp_yielding
                    # Recompute full-cell stress diagnostics from the converged eps_vp state.
                    (
                        eps_vp_tmp,
                        sigma_new_arr,
                        vm_new_arr,
                        max_ps_new_arr,
                        f_trial_dt0_arr,
                    ) = update_perzyna_state_cellwise(
                        strain_total=strain_arr,
                        eps_vp_prev=eps_vp_tmp,
                        e_arr=materials.e_arr,
                        nu_arr=materials.nu_arr,
                        sigma_y_arr=materials.sigma_y_arr,
                        eta_arr=materials.eta_arr,
                        dt=0.0,
                        active_mask=active_mask,
                    )
                    # Preserve the legacy final f_trial semantics for yielding cells.
                    f_trial_arr = f_trial_dt0_arr
                    f_trial_arr[yielding_mask] = f_trial_yielding

                n_sub = n_sub_computed
                time_perzyna = time.time() - t_perzyna_start

                local_yielding = int(np.sum((f_trial_arr > 0) & active_mask))
                local_max_f = (
                    float(np.max(f_trial_arr[active_mask])) if np.any(active_mask) else 0.0
                )
                # Global yielding diagnostics require MPI reductions so step-level
                # reporting reflects the full distributed mesh.
                global_yielding = comm.allreduce(local_yielding, op=MPI.SUM)
                global_max_f = comm.allreduce(local_max_f, op=MPI.MAX)
                eps_vp_new_arr = eps_vp_tmp

                inactive_mask = ~active_mask
                eps_vp_new_arr[inactive_mask, :, :] = 0.0
                sigma_new_arr[inactive_mask, :, :] = 0.0
                vm_new_arr[inactive_mask] = 0.0
                max_ps_new_arr[inactive_mask] = 0.0
                f_trial_arr[inactive_mask] = 0.0

                materials.eps_vp.x.array[:] = eps_vp_new_arr.reshape(-1)
                materials.stress.x.array[:] = sigma_new_arr.reshape(-1)
                materials.von_mises.x.array[:] = vm_new_arr
                materials.max_principal_stress.x.array[:] = max_ps_new_arr
                materials.yield_function_trial.x.array[:] = f_trial_arr

                # Scatter all updated constitutive fields before any diagnostics or
                # output, so ghost values are synchronized across partitions.
                materials.eps_vp.x.scatter_forward()
                materials.stress.x.scatter_forward()
                materials.von_mises.x.scatter_forward()
                materials.max_principal_stress.x.scatter_forward()
                materials.yield_function_trial.x.scatter_forward()

                # Update damage_max history on inter-layer facets.
                if n_il > 0:
                    from materials.damage_update import update_damage_max_numpy
                    update_damage_max_numpy(
                        u, materials, cell_to_dofs, birth_times_dolfinx,
                        cfg, il_cell_plus, il_cell_minus,
                        il_facet_normals, il_h_cells, active_mask,
                    )

                # u.x.array is interleaved vector storage [ux0, uy0, uz0, ux1, ...].
                # Reshape to (n_nodes_local, 3) to compute vector magnitudes.
                u_vec = u.x.array.reshape((-1, 3))
                local_max_u = (
                    float(np.max(np.linalg.norm(u_vec, axis=1))) if u_vec.shape[0] > 0 else 0.0
                )

                vm_arr = materials.von_mises.x.array
                local_max_vm = float(np.max(vm_arr)) if vm_arr.size > 0 else 0.0

                # eps_vp stores full 3x3 tensor per cell as 9 flattened entries.
                # Reshape to (n_cells_local, 9) for per-cell tensor norm diagnostics.
                epsvp_vec = materials.eps_vp.x.array.reshape((-1, 9))
                local_max_epsvp = (
                    float(np.max(np.linalg.norm(epsvp_vec, axis=1)))
                    if epsvp_vec.shape[0] > 0
                    else 0.0
                )

                global_max_u = comm.allreduce(local_max_u, op=MPI.MAX)
                global_max_vm = comm.allreduce(local_max_vm, op=MPI.MAX)
                global_max_epsvp = comm.allreduce(local_max_epsvp, op=MPI.MAX)

                # Extract only z-components from interleaved displacement storage:
                # index pattern [2, 5, 8, ...] = u_z at each local vector dof tuple.
                z_components = u.x.array[2::3]
                local_min_z = float(np.min(z_components)) if z_components.size > 0 else np.inf
                global_min_z = comm.allreduce(local_min_z, op=MPI.MIN)

                max_ps_arr = materials.max_principal_stress.x.array
                local_max_ps = float(np.max(max_ps_arr)) if max_ps_arr.size > 0 else 0.0
                global_max_ps = comm.allreduce(local_max_ps, op=MPI.MAX)

                total_newton_iters += int(n_iter)
                total_ksp_its_all += newton_stats.get("total_ksp_its", 0)
                total_perzyna_steps += int(n_sub)
                total_time_newton += float(time_newton)
                total_time_perzyna += float(time_perzyna)
                total_time_proj += float(time_proj)

                final_global_max_u = float(global_max_u)
                final_global_min_z = float(global_min_z)
                final_global_max_vm = float(global_max_vm)
                final_global_max_ps = float(global_max_ps)
                final_global_yielding = int(global_yielding)

                if debug_output_io:
                    print(
                        f"[DBG][rank {comm.rank}] step {step} entering vtx_disp.write(t={t_val:.4f})",
                        flush=True,
                    )
                    if collective_debug_barrier:
                        comm.Barrier()

                t_io_start = time.time()
                vtx_disp.write(t_val)
                time_io = time.time() - t_io_start

                if debug_output_io:
                    print(
                        f"[DBG][rank {comm.rank}] step {step} finished vtx_disp.write",
                        flush=True,
                    )
                    if collective_debug_barrier:
                        comm.Barrier()

                if debug_output_io:
                    print(
                        f"[DBG][rank {comm.rank}] step {step} entering vtx_cell.write(t={t_val:.4f})",
                        flush=True,
                    )
                    if collective_debug_barrier:
                        comm.Barrier()

                t_io_start = time.time()
                vtx_cell.write(t_val)
                time_io += time.time() - t_io_start

                is_final_step = step == (n_steps - 1)
                should_save_checkpoint = (
                    (step % checkpoint_save_every == 0) or is_final_step
                )
                if should_save_checkpoint:
                    t_io_start = time.time()
                    save_checkpoint(step, t_val)
                    time_io += time.time() - t_io_start

                total_time_io += float(time_io)

                if debug_output_io:
                    print(
                        f"[DBG][rank {comm.rank}] step {step} finished vtx_cell.write",
                        flush=True,
                    )
                    if collective_debug_barrier:
                        comm.Barrier()

                if comm.rank == 0:
                    elapsed_step = time.time() - t0
                    log_message(f"\n  RESULT: {msg} in {n_iter} iters ({elapsed_step:.2f}s)")
                    log_message(f"  Step Summary (t = {t_val:.4f} s):")
                    log_message("  ├─ Timings:")
                    log_message(f"  │    Newton Solver:   {time_newton:.2f} s")
                    log_message(f"  │    Perzyna Update:  {time_perzyna:.2f} s")
                    log_message(f"  │    DG0 Projection:  {time_proj:.2f} s")
                    log_message(f"  │    File I/O:        {time_io:.2f} s")
                    log_message("  └─ Global Statistics:")
                    log_message(
                        f"       Yielding Cells:  {global_yielding} (Max f_trial: {global_max_f:.2e} MPa)"
                    )
                    log_message(
                        f"       Max Disp (|u|):  {global_max_u:.2e} mm (Z-sag: {global_min_z:.2e} mm)"
                    )
                    log_message(f"       Max VM Stress:   {global_max_vm:.2e} MPa")
                    log_message(
                        f"       Max Principal:   {global_max_ps:.2e} MPa (Tension)"
                    )
                    log_message(f"       Max Plastic Str: {global_max_epsvp:.2e}")
                    if dt_eff > 0.0 and np.isfinite(dt_stable_limit):
                        log_message(
                            f"       Explicit limit:  dt_sub <= {dt_stable_limit:.2e} s (eta/E)"
                        )

                # Write CSV row (rank 0 only; allreduce is collective).
                local_active_count = int(np.sum(
                    birth_times_dolfinx[:num_owned_cells] <= t_val
                ))
                n_active_csv = comm.allreduce(local_active_count, op=MPI.SUM)
                n_total_csv = comm.allreduce(int(num_owned_cells), op=MPI.SUM)
                # Ranks that own at least one active cell.
                rank_has_active = 1 if local_active_count > 0 else 0
                n_active_ranks = comm.allreduce(rank_has_active, op=MPI.SUM)
                if csv_writer is not None:
                    elapsed_step = time.time() - t0
                    csv_writer.writerow({
                        "step": step,
                        "time_s": f"{t_val:.6f}",
                        "dt_s": f"{dt:.6e}",
                        "mpi_ranks_total": comm.size,
                        "mpi_ranks_active": n_active_ranks,
                        "active_cells": n_active_csv,
                        "total_cells": n_total_csv,
                        "active_dofs": newton_stats.get("n_active_dofs", ""),
                        "inactive_dofs": newton_stats.get("n_inactive_dofs", ""),
                        "newton_iters": n_iter,
                        "converged": int(converged),
                        "newton_msg": msg,
                        "res0": f"{newton_stats.get('res0', float('nan')):.6e}",
                        "res_final": f"{newton_stats.get('res_final', float('nan')):.6e}",
                        "rel_res_final": f"{newton_stats.get('rel_res_final', float('nan')):.6e}",
                        "total_ksp_its": newton_stats.get("total_ksp_its", 0),
                        "total_ls_its": newton_stats.get("total_ls_its", 0),
                        "final_omega": f"{newton_stats.get('final_omega', 1.0):.4f}",
                        "time_newton_s": f"{time_newton:.4f}",
                        "time_perzyna_s": f"{time_perzyna:.4f}",
                        "time_proj_s": f"{time_proj:.4f}",
                        "time_io_s": f"{time_io:.4f}",
                        "time_step_total_s": f"{elapsed_step:.4f}",
                        "n_sub_steps": n_sub,
                        "yielding_cells": global_yielding,
                        "max_yield_f_MPa": f"{global_max_f:.6e}",
                        "max_disp_mm": f"{global_max_u:.6e}",
                        "z_sag_mm": f"{global_min_z:.6e}",
                        "max_von_mises_MPa": f"{global_max_vm:.6e}",
                        "max_principal_MPa": f"{global_max_ps:.6e}",
                        "max_plastic_strain": f"{global_max_epsvp:.6e}",
                        "dt_stable_limit_s": f"{dt_stable_limit:.6e}" if np.isfinite(dt_stable_limit) else "",
                        "rss_GiB": f"{_current_rss_gib():.3f}",
                        "cumul_newton_iters": total_newton_iters,
                        "cumul_ksp_its": total_ksp_its_all,
                        "cumul_wall_s": f"{time.time() - loop_start_time:.2f}",
                    })
                    csv_file.flush()

                t_prev = t_val

            total_loop_time = time.time() - loop_start_time
            if simulation_start_time is None:
                total_simulation_time = total_loop_time
            else:
                # Full wall-clock runtime from main orchestration start.
                total_simulation_time = time.time() - float(simulation_start_time)
            total_time_overhead = total_loop_time - (
                total_time_newton + total_time_perzyna + total_time_proj + total_time_io
            )
            if comm.rank == 0:
                log_message("")
                log_message("  ========================================================================")
                log_message("  SIMULATION COMPLETE")
                log_message("  ========================================================================")
                log_message("  Final Structural State:")
                log_message(
                    f"    Max Disp (|u|):      {final_global_max_u:.2e} mm (Z-sag: {final_global_min_z:.2e} mm)"
                )
                log_message(f"    Max VM Stress:       {final_global_max_vm:.2e} MPa")
                log_message(f"    Max Principal:       {final_global_max_ps:.2e} MPa")
                log_message(f"    Yielding Cells:      {final_global_yielding}")
                log_message("  ")
                log_message("  Computational Effort:")
                log_message(f"    Total Time Steps:    {n_steps}")
                log_message(f"    Total Newton Iters:  {total_newton_iters}")
                log_message(f"    Total Perzyna Steps: {total_perzyna_steps}")
                log_message("  ")
                log_message("  Timing Breakdown:")
                log_message(f"    Total Simulation Time:  {total_simulation_time:.2f} s")
                log_message(f"    Total Loop Time:     {total_loop_time:.2f} s")
                log_message(f"    ├─ Newton Solver:    {total_time_newton:.2f} s")
                log_message(f"    ├─ Perzyna Update:   {total_time_perzyna:.2f} s")
                log_message(f"    ├─ DG0 Projection:   {total_time_proj:.2f} s")
                log_message(f"    ├─ File I/O:         {total_time_io:.2f} s")
                log_message(f"    └─ Overhead (MPI/Misc):  {total_time_overhead:.2f} s")
                log_message("  ========================================================================")
                log_message(f"  Log saved to: {log_path}")
                log_message(f"  CSV metrics saved to: {csv_path}")
            # Close CSV file after loop completes (before finally).
            if csv_file is not None:
                csv_file.close()
    finally:
        if newton_workspace is not None:
            newton_workspace.destroy()
        A_proj.destroy()
        b_strain.destroy()
        inv_diag_proj.destroy()
        diag_proj.destroy()
