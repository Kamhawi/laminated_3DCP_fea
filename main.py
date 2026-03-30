# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""High-level orchestrator for the 3D concrete printing EVP simulation.

This module intentionally avoids low-level math and time-loop internals.
Its role is to:
1. Load configuration.
2. Build and map the custom barrel-vault mesh into DOLFINx.
3. Initialize state fields and tagging data.
4. Build the nonlinear weak form.
5. Hand execution to the simulation time stepper.
"""

import os
import sys
import time
import shutil

from mpi4py import MPI
from dolfinx import fem

from config.config_utils import (
    build_output_paths,
    build_run_tag,
    get_checkpoint_dir,
    load_config,
    save_run_config_snapshot,
)
from materials.material_state import MaterialStateManager
from mesh import (
    BarrelVaultVolumetricMesh,
    build_partitioned_dolfinx_mesh,
    compute_cell_permutation,
    configure_streaming_stdio,
    evaluate_mesh_quality,
    reorder_cell_data,
    tag_interfaces_and_boundaries,
)
from physics.weak_form import build_evp_cohesive_weak_form
from solver.time_stepper import build_cell_to_dofs_and_support, run_simulation


class _TeeStream:
    """Mirror writes to both original stream and log file handle."""

    def __init__(self, original_stream, log_file):
        self._original_stream = original_stream
        self._log_file = log_file

    def write(self, data):
        self._original_stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self):
        self._original_stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._original_stream.isatty()

    @property
    def encoding(self):
        return getattr(self._original_stream, "encoding", "utf-8")

    def fileno(self):
        return self._original_stream.fileno()


def main():
    """Run the full simulation workflow from setup to output.

    Args:
        None.

    Returns:
        None.

    Raises:
        Exception: Propagates uncaught setup/build errors from downstream
            modules. Cell-reordering failures trigger ``comm.Abort(1)``.

    Notes:
        The numerically intensive logic (Newton loop, constitutive updates,
        cohesive laws, and DG0 projections) is delegated to specialized modules.
    """
    # Configure stdout/stderr for immediate MPI log streaming.
    configure_streaming_stdio()
    simulation_start_time = time.time()
    comm = MPI.COMM_WORLD

    # Step 1: Load user configuration and extract geometry/mesh sections.
    cfg = load_config()
    run_tag = build_run_tag() if comm.rank == 0 else None
    run_tag = comm.bcast(run_tag, root=0)
    checkpoint_defaults = {
        "save_every": 10,
        "resume_enabled": False,
        "directory": "checkpoints",
    }
    checkpoint_cfg = dict(checkpoint_defaults)
    checkpoint_cfg.update(cfg.get("checkpoint", {}))
    cfg["checkpoint"] = checkpoint_cfg

    checkpoint_dir = get_checkpoint_dir(cfg)
    if comm.rank == 0:
        # If resuming is disabled, wipe the existing checkpoint directory to prevent stray files
        if not checkpoint_cfg["resume_enabled"] and checkpoint_dir.exists():
            print(
                f"  [Cleanup] 'resume_enabled' is false. Wiping old checkpoints in: {checkpoint_dir}",
                flush=True,
            )
            shutil.rmtree(checkpoint_dir)

        # Ensure the directory exists (whether it was just wiped or we are keeping it)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Force all ranks to wait until Rank 0 is completely done deleting/creating the directory
    comm.Barrier()

    # Resolve output/log paths early so logging can start before mesh generation.
    disp_path, cell_path, log_path = build_output_paths(cfg, run_tag=run_tag)
    run_output_dir = disp_path.parent
    geom = cfg["geometry"]
    mesh_cfg = cfg["mesh"]

    # Tee rank-0 stdout/stderr to log file so all rank-0 prints are persisted.
    log_file = None
    original_stdout = None
    original_stderr = None
    if comm.rank == 0:
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        # Used by newton._live_print (os.write) to append those lines as well.
        os.environ["SIM_LOG_PATH"] = str(log_path)
        os.environ["SIM_TEE_ACTIVE"] = "1"
        settings_snapshot_path = save_run_config_snapshot(cfg, run_output_dir)
        print(f"  Run output directory: {run_output_dir}", flush=True)
        print(f"  Config snapshot: {settings_snapshot_path}", flush=True)

    try:
        # Boundary-condition defaults for support faces.
        bc_dir_face_type = "springing_facing"
        bc_dir_tag = 0
        bc_intrados_upto_layer = cfg.get("boundary_conditions", {}).get(
            "intrados_dirichlet_upto_layer",
            mesh_cfg["n_length"] - 1,
        )

        # Step 2: Build the custom barrel-vault parametric mesh.
        vault_mesh = BarrelVaultVolumetricMesh(
            span=geom["span"],
            length=geom["length"],
            rise=geom["rise"],
            thickness=geom["thickness"],
            n_length=mesh_cfg["n_length"],
            n_span=mesh_cfg["n_span"],
            n_thickness=mesh_cfg["n_thickness"],
        )

        # Apply boundary labels/values on the custom mesh prior to DOLFINx mapping.
        vault_mesh.assign_dirichlet_boundary_conditions({bc_dir_face_type: bc_dir_tag})
        vault_mesh.assign_intrados_dirichlet_up_to_layer(
            layer_number=bc_intrados_upto_layer,
            value=0.0,
        )

        # Step 3: Compute cell birth times from toolpath speed.
        activation_cfg = cfg["activation"]
        birth_times_array = vault_mesh.compute_birth_times(
            tcp_speed=activation_cfg["tcp_speed"]
        )

        # Step 4: Partition and create the distributed DOLFINx mesh.
        partitioner_mode = mesh_cfg.get("partitioner", "strip")
        msh, cells_lst, cells_layers = build_partitioned_dolfinx_mesh(
            vault_mesh, comm, partitioner_mode=partitioner_mode,
        )

        # Step 4b: Evaluate mesh quality on the DOLFINx mesh.
        quality_report = evaluate_mesh_quality(msh, cfg, comm)
        if not quality_report.is_valid:
            if comm.rank == 0:
                print(
                    "  FATAL: Mesh contains invalid elements. Aborting.",
                    flush=True,
                )
            comm.Abort(1)

        # Step 5: Reorder cell-wise arrays into DOLFINx local+ghost ordering.
        try:
            perm = compute_cell_permutation(msh, vault_mesh)
            birth_times_dolfinx = reorder_cell_data(birth_times_array, perm)
            cells_layers_dolfinx = reorder_cell_data(cells_layers, perm)
        except Exception as e:
            print(f"[Rank {comm.rank}] Fatal error in Step 5: {str(e)}", flush=True)
            comm.Abort(1)

        # Step 6: Create all DG0 and DG0-tensor material/state fields.
        materials = MaterialStateManager(
            msh,
            cfg["material"],
            cfg["hardening"],
            birth_times_dolfinx,
        )

        # Exportable cell-wise metadata fields.
        birth_time_func = fem.Function(materials.V_DG0, name="birth_time")
        birth_time_func.x.array[:] = birth_times_dolfinx
        # Synchronize DG0 metadata field on ghost cells before I/O/form usage.
        birth_time_func.x.scatter_forward()

        cells_layers_func = fem.Function(materials.V_DG0, name="layer_id")
        cells_layers_func.x.array[:] = cells_layers_dolfinx
        # Ghost synchronization keeps per-cell layer ids consistent across ranks.
        cells_layers_func.x.scatter_forward()

        is_active_func = fem.Function(materials.V_DG0, name="is_active")
        is_active_func.x.array[:] = 0.0
        # Initialize and scatter once so all ranks start from identical indicator state.
        is_active_func.x.scatter_forward()

        mpi_rank_func = fem.Function(materials.V_DG0, name="mpi_rank")
        num_owned_cells = msh.topology.index_map(msh.topology.dim).size_local
        mpi_rank_func.x.array[:num_owned_cells] = float(comm.rank)
        mpi_rank_func.x.scatter_forward()

        # Step 7: Tag inter-/intra-layer interior facets and Dirichlet boundaries.
        (
            interlayer_facet_indices,
            intralayer_facet_indices,
            interior_facet_tags,
            dirichlet_tags,
        ) = tag_interfaces_and_boundaries(msh, vault_mesh)

        if comm.rank == 0:
            print(
                f"  Interlayer facets (cohesive): {len(interlayer_facet_indices)}",
                flush=True,
            )
            print(
                f"  Intralayer facets (bonded): {len(intralayer_facet_indices)}",
                flush=True,
            )

        # Ensure required facet-to-cell connectivity exists before form assembly.
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

        # Step 8: Displacement space and unknown (DG1 vector).
        if comm.rank == 0:
            print("Creating DG function space...", flush=True)
        V = fem.functionspace(msh, ("DG", 1, (3,)))
        if comm.rank == 0:
            print(
                f"  Global DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}",
                flush=True,
            )

        u = fem.Function(V, name="displacement")
        materials.u = u

        # Step 9: Build cell-to-DOF and support mapping for activation initialization.
        cell_to_dofs, support, support_missing = build_cell_to_dofs_and_support(
            msh,
            V,
            vault_mesh,
            cells_lst,
            perm,
        )
        if support_missing > 0 and comm.rank == 0:
            print(
                f"  WARNING: {support_missing} interlayer supports could not be mapped.",
                flush=True,
            )

        # Step 10: Build bulk EVP + cohesive interface weak forms.
        if comm.rank == 0:
            print(
                "Building nonlinear weak form with EVP bulk and mixed-mode cohesive law...",
                flush=True,
            )

        F_form, J_form = build_evp_cohesive_weak_form(
            msh,
            V,
            materials,
            birth_time_func,
            interior_facet_tags,
            dirichlet_tags,
            cfg,
        )

        # Step 11: Launch the simulation time loop.
        run_simulation(
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
            interior_facet_tags=interior_facet_tags,
            simulation_start_time=simulation_start_time,
        )

        if comm.rank == 0:
            print("\nDone!", flush=True)
    finally:
        if comm.rank == 0:
            os.environ.pop("SIM_TEE_ACTIVE", None)
            os.environ.pop("SIM_LOG_PATH", None)
            if original_stdout is not None:
                sys.stdout = original_stdout
            if original_stderr is not None:
                sys.stderr = original_stderr
            if log_file is not None:
                log_file.close()


if __name__ == "__main__":
    main()
