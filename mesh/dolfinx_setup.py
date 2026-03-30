# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Utilities for preparing DOLFINx mesh objects and facet tags.

This module isolates orchestration setup steps used by `main.py`:
- stdio streaming setup for MPI runs,
- custom-mesh -> DOLFINx conversion with partitioning,
- interface/boundary facet tagging and interior facet classification.
"""

import sys
import time

import numpy as np
from basix.ufl import element
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import GhostMode, create_cell_partitioner, create_mesh, meshtags

from .dolfinx_mapping import build_custom_node_lookup, tag_boundary_faces, tag_interfaces
from .mesh_core import FaceLabel


def _make_strip_partitioner(n_span, n_thickness, n_cells_per_layer):
    """Build a partitioner that assigns cells to ranks by span-direction strips.

    Each rank gets a contiguous range of span indices across all layers,
    so that when a new layer activates every rank receives new cells
    simultaneously.  DOLFINx then handles cell redistribution and ghost
    cell creation via ``GhostMode.shared_facet``.

    When the ``ghosted`` flag is set, the partitioner also includes
    neighbouring-rank destinations for cells at strip boundaries so that
    DOLFINx can create the ghost layer required for interior-facet
    integrals (DG SIP / cohesive terms).

    Args:
        n_span: Number of span cells per layer.
        n_thickness: Number of thickness cells per layer.
        n_cells_per_layer: ``n_span * n_thickness``.

    Returns:
        Callable: Partitioner function compatible with
        ``create_cell_partitioner``.
    """

    def _partitioner(comm_w, nparts, adj, ghosted):
        n_cells = len(adj)

        # Map every GLOBAL cell index to its destination rank.
        # Collect all global cell ids referenced in the local adjacency.
        all_ids = set()
        for c in range(n_cells):
            all_ids.add(c)
            for nb in adj.links(c):
                all_ids.add(int(nb))
        max_id = max(all_ids) + 1

        gid_to_rank = np.zeros(max_id, dtype=np.int32)
        for gid in range(max_id):
            span_idx = (gid % n_cells_per_layer) % n_span
            gid_to_rank[gid] = min(span_idx * nparts // n_span, nparts - 1)

        # Build per-cell destination list: primary rank first, then ghost
        # ranks (neighbours assigned to a different rank).
        dest_data = []
        dest_offsets = [0]
        for c in range(n_cells):
            primary = int(gid_to_rank[c])
            ranks = [primary]
            if ghosted:
                for nb in adj.links(c):
                    nb_rank = int(gid_to_rank[int(nb)])
                    if nb_rank != primary and nb_rank not in ranks:
                        ranks.append(nb_rank)
            dest_data.extend(ranks)
            dest_offsets.append(len(dest_data))

        return AdjacencyList_int32(
            np.array(dest_data, dtype=np.int32),
            np.array(dest_offsets, dtype=np.int32),
        )

    return _partitioner


def configure_streaming_stdio():
    """Force line-buffered streams for low-latency MPI logging.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass


def build_partitioned_dolfinx_mesh(vault_mesh, comm, partitioner_mode="strip"):
    """Create a distributed DOLFINx hexahedral mesh from the custom mesh.

    Args:
        vault_mesh: Custom hexahedral mesh object.
        comm: MPI communicator.

    Returns:
        tuple:
            ``(msh, cells_lst, cells_layers)`` where ``cells_layers`` stores
            original layer ids per custom cell.

    Raises:
        None.
    """
    cells_lst = vault_mesh.get_all_cells()

    all_cells = np.ascontiguousarray(
        np.array([cell.node_indices for cell in cells_lst]), dtype=np.int64
    )
    x = np.ascontiguousarray(vault_mesh.nodes, dtype=np.float64)

    # Strip-based per-layer partitioning: divide the span direction into
    # uniform strips so every rank owns cells across ALL layers.  When a
    # new layer activates, all ranks receive new active cells simultaneously,
    # eliminating the load imbalance that caused idle ranks with the old
    # contiguous-block scheme.
    #
    # Rank 0 provides all cells; the custom strip partitioner assigns each
    # cell to a rank based on its span index.  DOLFINx then redistributes
    # cells and creates proper ghost cells for ``shared_facet`` mode.
    n_span = vault_mesh.n_span
    n_thickness = vault_mesh.n_thickness
    n_cells_per_layer = n_span * n_thickness

    # Each rank provides a disjoint block of cells as input to create_mesh.
    # The strip partitioner then redistributes cells so that each rank
    # owns a span-direction strip across all layers.  Because cells
    # actually move between ranks, DOLFINx properly creates ghost cells
    # for shared_facet mode.
    n_total_cells = len(all_cells)
    base_chunk = n_total_cells // comm.size
    remainder = n_total_cells % comm.size
    if comm.rank < remainder:
        start_idx = comm.rank * (base_chunk + 1)
        end_idx = start_idx + base_chunk + 1
    else:
        start_idx = comm.rank * base_chunk + remainder
        end_idx = start_idx + base_chunk
    cells = np.ascontiguousarray(all_cells[start_idx:end_idx, :])
    cells.flags.writeable = False
    x.flags.writeable = False

    cells_layers = np.array([cell.layer.layer_id for cell in cells_lst], dtype=np.float64)
    coord_elem = element("Lagrange", "hexahedron", 1, shape=(3,))

    if comm.rank == 0:
        n_layers = len(vault_mesh.layers)
        cells_per_rank = len(all_cells) // max(comm.size, 1)
        print(
            f"Strip partitioning: {n_span} span cells across {comm.size} ranks "
            f"(~{cells_per_rank} cells/rank, {n_layers} layers)",
            flush=True,
        )
        print("Creating DOLFINx mesh and partitioning across cores...", flush=True)
    t0 = time.time()

    if partitioner_mode == "strip":
        # Custom strip partitioner assigns cells by span index; DOLFINx
        # handles redistribution and ``GhostMode.shared_facet`` ghost cells
        # for interior-facet integrals and cohesive interface terms.
        strip_part = _make_strip_partitioner(n_span, n_thickness, n_cells_per_layer)
        part = create_cell_partitioner(strip_part, GhostMode.shared_facet)
    else:
        # Default DOLFINx partitioner (ParMETIS/SCOTCH graph partitioning).
        part = create_cell_partitioner(GhostMode.shared_facet)
    msh = create_mesh(comm=comm, cells=cells, x=x, e=coord_elem, partitioner=part)

    if comm.rank == 0:
        print(f"  Mesh created and partitioned in {time.time() - t0:.2f}s", flush=True)
        print(f"  Global cells: {msh.topology.index_map(3).size_global}", flush=True)

    return msh, cells_lst, cells_layers


def tag_interfaces_and_boundaries(msh, vault_mesh):
    """Build interior/boundary meshtags from custom interface definitions.

    Tag convention for interior facets:
        0: generic interior
        1: inter-layer cohesive interface
        2: intra-layer bonded interface

    Args:
        msh: Distributed DOLFINx mesh.
        vault_mesh: Custom barrel-vault mesh with interface/boundary labels.

    Returns:
        tuple:
            ``(interlayer_facet_indices, intralayer_facet_indices,
            interior_facet_tags, dirichlet_tags)``.

    Raises:
        None.
    """
    custom_lookup, custom_lookup_collisions = build_custom_node_lookup(vault_mesh, decimals=10)

    interlayer_facet_indices, _ = tag_interfaces(
        msh,
        vault_mesh,
        "inter_layer",
        custom_lookup=custom_lookup,
        custom_lookup_collisions=custom_lookup_collisions,
    )
    intralayer_facet_indices, _ = tag_interfaces(
        msh,
        vault_mesh,
        "intra_layer",
        custom_lookup=custom_lookup,
        custom_lookup_collisions=custom_lookup_collisions,
    )

    fdim = msh.topology.dim - 1
    tdim = msh.topology.dim
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    num_facets_local = msh.topology.index_map(fdim).size_local

    all_interior_facets = np.array(
        [f for f in range(num_facets_local) if len(f_to_c.links(f)) == 2], dtype=np.int32
    )
    all_interior_tags = np.zeros_like(all_interior_facets, dtype=np.int32)

    facet_pos = np.full(num_facets_local, -1, dtype=np.int32)
    facet_pos[all_interior_facets] = np.arange(all_interior_facets.size, dtype=np.int32)

    valid_inter = facet_pos[np.asarray(interlayer_facet_indices, dtype=np.int64)]
    valid_inter = valid_inter[valid_inter >= 0]
    all_interior_tags[valid_inter] = 1

    valid_intra = facet_pos[np.asarray(intralayer_facet_indices, dtype=np.int64)]
    valid_intra = valid_intra[valid_intra >= 0]
    # Preserve cohesive precedence: facets already tagged as inter-layer (1)
    # are not overwritten by intra-layer tag (2).
    free_intra = valid_intra[all_interior_tags[valid_intra] != 1]
    all_interior_tags[free_intra] = 2

    interior_facet_tags = meshtags(msh, fdim, all_interior_facets, all_interior_tags)

    _, dirichlet_tags = tag_boundary_faces(
        msh,
        vault_mesh,
        FaceLabel.DIRICHLET,
        custom_lookup=custom_lookup,
        custom_lookup_collisions=custom_lookup_collisions,
    )

    return interlayer_facet_indices, intralayer_facet_indices, interior_facet_tags, dirichlet_tags
