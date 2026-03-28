"""Shared setup for Phase 1 Jacobian diagnostics.

Builds the complete simulation state (mesh, materials, function spaces,
weak forms) using the same initialization as main.py but without launching
the time-stepping loop.  Allows mesh-size overrides for fast testing.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from config.config_utils import load_config
from materials.material_state import MaterialStateManager
from mesh import (
    BarrelVaultVolumetricMesh,
    build_partitioned_dolfinx_mesh,
    compute_cell_permutation,
    reorder_cell_data,
    tag_interfaces_and_boundaries,
)
from physics.weak_form import build_evp_cohesive_weak_form
from solver.time_stepper import build_cell_to_dofs_and_support
from solver.newton import get_inactive_dofs


@dataclass
class DiagnosticState:
    """Container for all objects needed by Jacobian diagnostics."""

    cfg: dict
    msh: object
    V: object
    u: object
    materials: object
    birth_times_dolfinx: np.ndarray
    birth_time_func: object
    interior_facet_tags: object
    dirichlet_tags: object
    cell_to_dofs: np.ndarray
    support: np.ndarray
    F_form: object
    J_form: object
    vault_mesh: object
    cells_lst: object
    perm: np.ndarray
    comm: object = field(default=None)
    # Set during prepare_state_at_time:
    t_val: float = 0.0
    inactive_dofs: Optional[np.ndarray] = None


def build_diagnostic_state(config_path=None, mesh_overrides=None, comm=None):
    """Build the full simulation state for diagnostic testing.

    Replicates the initialization from main.py (steps 1-10) without
    output paths, logging, checkpoints, or the simulation loop.

    Args:
        config_path: Optional path to config YAML.
        mesh_overrides: Optional dict, e.g. ``{"n_length": 4, "n_span": 4,
            "n_thickness": 2}`` for a fast 32-cell mesh.
        comm: MPI communicator.  Defaults to ``MPI.COMM_WORLD``.

    Returns:
        DiagnosticState with all fields populated.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    # Step 1: Load config, apply mesh overrides.
    cfg = load_config(config_path)
    if mesh_overrides:
        for key, val in mesh_overrides.items():
            cfg["mesh"][key] = val

    geom = cfg["geometry"]
    mesh_cfg = cfg["mesh"]
    activation_cfg = cfg["activation"]

    # Clamp intrados_dirichlet_upto_layer to the (possibly reduced) mesh.
    bc_intrados_upto_layer = cfg.get("boundary_conditions", {}).get(
        "intrados_dirichlet_upto_layer",
        mesh_cfg["n_length"] - 1,
    )
    bc_intrados_upto_layer = min(bc_intrados_upto_layer, mesh_cfg["n_length"] - 1)

    # Step 2: Build parametric barrel-vault mesh.
    vault_mesh = BarrelVaultVolumetricMesh(
        span=geom["span"],
        length=geom["length"],
        rise=geom["rise"],
        thickness=geom["thickness"],
        n_length=mesh_cfg["n_length"],
        n_span=mesh_cfg["n_span"],
        n_thickness=mesh_cfg["n_thickness"],
    )
    vault_mesh.assign_dirichlet_boundary_conditions({"springing_facing": 0})
    vault_mesh.assign_intrados_dirichlet_up_to_layer(
        layer_number=bc_intrados_upto_layer, value=0.0,
    )

    # Step 3: Birth times from toolpath speed.
    birth_times_array = vault_mesh.compute_birth_times(
        tcp_speed=activation_cfg["tcp_speed"],
    )

    # Step 4: Partition into DOLFINx mesh.
    msh, cells_lst, cells_layers = build_partitioned_dolfinx_mesh(vault_mesh, comm)

    # Step 5: Reorder cell-wise arrays into DOLFINx ordering.
    perm = compute_cell_permutation(msh, vault_mesh)
    birth_times_dolfinx = reorder_cell_data(birth_times_array, perm)
    cells_layers_dolfinx = reorder_cell_data(cells_layers, perm)

    # Step 6: Material state manager with DG0 fields.
    materials = MaterialStateManager(
        msh, cfg["material"], cfg["hardening"], birth_times_dolfinx,
    )

    # DG0 metadata fields.
    birth_time_func = fem.Function(materials.V_DG0, name="birth_time")
    birth_time_func.x.array[:] = birth_times_dolfinx
    birth_time_func.x.scatter_forward()

    cells_layers_func = fem.Function(materials.V_DG0, name="layer_id")
    cells_layers_func.x.array[:] = cells_layers_dolfinx
    cells_layers_func.x.scatter_forward()

    is_active_func = fem.Function(materials.V_DG0, name="is_active")
    is_active_func.x.array[:] = 0.0
    is_active_func.x.scatter_forward()

    # Step 7: Tag inter-/intra-layer facets and Dirichlet boundaries.
    (
        _interlayer_facet_indices,
        _intralayer_facet_indices,
        interior_facet_tags,
        dirichlet_tags,
    ) = tag_interfaces_and_boundaries(msh, vault_mesh)

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    # Step 8: DG1 vector function space + displacement unknown.
    V = fem.functionspace(msh, ("DG", 1, (3,)))
    u = fem.Function(V, name="displacement")
    materials.u = u

    # Step 9: Cell-to-DOF and support mapping.
    cell_to_dofs, support, _support_missing = build_cell_to_dofs_and_support(
        msh, V, vault_mesh, cells_lst, perm,
    )

    # Step 10: Build weak forms.
    F_form, J_form = build_evp_cohesive_weak_form(
        msh, V, materials, birth_time_func,
        interior_facet_tags, dirichlet_tags, cfg,
    )

    if comm.rank == 0:
        n_cells = msh.topology.index_map(msh.topology.dim).size_local
        n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        print(f"  Diagnostic state built: {n_cells} local cells, "
              f"{n_dofs} global DOFs", flush=True)

    return DiagnosticState(
        cfg=cfg,
        msh=msh,
        V=V,
        u=u,
        materials=materials,
        birth_times_dolfinx=birth_times_dolfinx,
        birth_time_func=birth_time_func,
        interior_facet_tags=interior_facet_tags,
        dirichlet_tags=dirichlet_tags,
        cell_to_dofs=cell_to_dofs,
        support=support,
        F_form=F_form,
        J_form=J_form,
        vault_mesh=vault_mesh,
        cells_lst=cells_lst,
        perm=perm,
        comm=comm,
    )


def prepare_state_at_time(state, t_val, u_mode="random_small", seed=42):
    """Set the simulation state to a specific time and displacement.

    Args:
        state: DiagnosticState from ``build_diagnostic_state``.
        t_val: Simulation time.
        u_mode: ``"zero"`` or ``"random_small"``.
        seed: Random seed for MPI-consistent perturbation generation.
    """
    state.t_val = float(t_val)

    # Update the time Constant captured by the UFL alpha expression.
    state.materials.t_current.value = float(t_val)

    # Update age-dependent material properties at this time.
    state.materials.update_properties(float(t_val))

    # Compute inactive DOFs for this time.
    inactive_dofs_all = get_inactive_dofs(
        float(t_val), state.birth_times_dolfinx, state.cell_to_dofs,
    )
    num_owned = len(state.u.x.array)
    state.inactive_dofs = inactive_dofs_all[inactive_dofs_all < num_owned].astype(
        np.int32, copy=False,
    )

    if u_mode == "zero":
        state.u.x.array[:] = 0.0
    elif u_mode == "random_small":
        state.u.x.array[:] = _make_mpi_consistent_perturbation(
            state, scale=1.0e-4, seed=seed,
        )
    else:
        raise ValueError(f"Unknown u_mode: {u_mode!r}")

    # Zero inactive DOFs.
    if state.inactive_dofs.size > 0:
        state.u.x.array[state.inactive_dofs] = 0.0
    state.u.x.scatter_forward()


def _make_mpi_consistent_perturbation(state, scale=1.0e-4, seed=42):
    """Generate a perturbation vector deterministic w.r.t. global DOF index.

    Each DOF gets ``rng(seed + global_dof_index)`` so the physical
    perturbation is identical regardless of MPI partition count.

    Args:
        state: DiagnosticState.
        scale: Amplitude of the perturbation.
        seed: Base random seed.

    Returns:
        np.ndarray: Local perturbation array (same shape as ``u.x.array``).
    """
    bs = state.V.dofmap.index_map_bs
    imap = state.V.dofmap.index_map
    local_size = imap.size_local
    ghost_size = imap.num_ghosts
    global_offset = imap.local_range[0]

    # Build global DOF indices for owned + ghost entries.
    # Owned DOFs are contiguous: [global_offset*bs .. (global_offset+local_size)*bs)
    n_local_entries = (local_size + ghost_size) * bs
    global_indices = np.empty(n_local_entries, dtype=np.int64)

    # Owned portion.
    n_owned_entries = local_size * bs
    global_indices[:n_owned_entries] = np.arange(
        global_offset * bs, (global_offset + local_size) * bs, dtype=np.int64,
    )

    # Ghost portion: use the global indices from the index map.
    if ghost_size > 0:
        ghost_global = imap.ghosts  # global node indices for ghosts
        for i, g in enumerate(ghost_global):
            start = n_owned_entries + i * bs
            global_indices[start : start + bs] = np.arange(
                g * bs, (g + 1) * bs, dtype=np.int64,
            )

    # Deterministic per-DOF random values.
    du = np.empty(n_local_entries, dtype=np.float64)
    for i, gi in enumerate(global_indices):
        rng = np.random.RandomState(seed + int(gi))
        du[i] = rng.randn()

    du *= scale
    return du
