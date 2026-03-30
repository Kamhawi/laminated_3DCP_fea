# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Mesh quality evaluation for hexahedral FEA meshes.

Computes standard element quality metrics (aspect ratio, Jacobian,
skewness, warping, volume) and issues warnings when mesh properties
may cause slow convergence or divergence in the Newton solver.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Warning classes
# ---------------------------------------------------------------------------


class MeshQualityWarning(UserWarning):
    """Base class for mesh quality warnings."""

    pass


class HighAspectRatioWarning(MeshQualityWarning):
    """Element aspect ratios exceed threshold."""

    pass


class NegativeJacobianWarning(MeshQualityWarning):
    """Jacobian determinant is negative or near-zero at evaluation points."""

    pass


class VolumetricLockingWarning(MeshQualityWarning):
    """Poisson ratio combined with element type risks volumetric locking."""

    pass


class PoorConditioningWarning(MeshQualityWarning):
    """Element size jumps or penalty parameters may cause poor conditioning."""

    pass


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_DEFAULT_ASPECT_RATIO_THRESHOLD = 10.0
_DEFAULT_MIN_SCALED_JACOBIAN = 0.1
_DEFAULT_JACOBIAN_NEAR_ZERO_TOL = 1.0e-10
_DEFAULT_VOLUME_VARIATION_THRESHOLD = 20.0
_DEFAULT_MAX_SKEWNESS = 0.5
_DEFAULT_MAX_WARPING = 0.1
_DEFAULT_FAIL_ON_NEGATIVE_JACOBIAN = True

# ---------------------------------------------------------------------------
# Hex element topology constants
# ---------------------------------------------------------------------------

# 12 edges of the hex element (pairs of local node indices).
# Node ordering: 0=(i,j,k) 1=(i+1,j,k) 2=(i,j+1,k) 3=(i+1,j+1,k)
#                4=(i,j,k+1) 5=(i+1,j,k+1) 6=(i,j+1,k+1) 7=(i+1,j+1,k+1)
_EDGES = np.array(
    [
        [0, 1], [2, 3], [4, 5], [6, 7],  # xi-direction
        [0, 2], [1, 3], [4, 6], [5, 7],  # eta-direction
        [0, 4], [1, 5], [2, 6], [3, 7],  # zeta-direction
    ],
    dtype=int,
)

# Reference coordinates for the 8-node trilinear hexahedron.
_REF_COORDS = np.array(
    [
        [-1, -1, -1],  # node 0
        [+1, -1, -1],  # node 1
        [-1, +1, -1],  # node 2
        [+1, +1, -1],  # node 3
        [-1, -1, +1],  # node 4
        [+1, -1, +1],  # node 5
        [-1, +1, +1],  # node 6
        [+1, +1, +1],  # node 7
    ],
    dtype=float,
)

# Evaluation points: 8 corners + centre.
_EVAL_POINTS = np.vstack([_REF_COORDS, [[0.0, 0.0, 0.0]]])

# 2x2x2 Gauss quadrature points (weights all = 1).
_GP = 1.0 / np.sqrt(3.0)
_GAUSS_PTS = np.array(
    [
        [-_GP, -_GP, -_GP], [+_GP, -_GP, -_GP],
        [-_GP, +_GP, -_GP], [+_GP, +_GP, -_GP],
        [-_GP, -_GP, +_GP], [+_GP, -_GP, +_GP],
        [-_GP, +_GP, +_GP], [+_GP, +_GP, +_GP],
    ],
    dtype=float,
)

# Face node indices (from HexahedronCell.FACE_NODE_MAP).
_FACE_NODES = np.array(
    [
        [0, 1, 5, 4],  # BOTTOM
        [2, 6, 7, 3],  # TOP
        [0, 2, 3, 1],  # FRONT
        [4, 5, 7, 6],  # BACK
        [0, 4, 6, 2],  # LEFT
        [1, 3, 7, 5],  # RIGHT
    ],
    dtype=int,
)

# Opposing face pairs (normals should be anti-parallel for a perfect hex).
_OPPOSING_FACES = [(0, 1), (2, 3), (4, 5)]

# Three edges emanating from each corner node (for scaled Jacobian).
_CORNER_NEIGHBOURS = [
    (0, [1, 2, 4]),
    (1, [0, 3, 5]),
    (2, [0, 3, 6]),
    (3, [1, 2, 7]),
    (4, [0, 5, 6]),
    (5, [1, 4, 7]),
    (6, [2, 4, 7]),
    (7, [3, 5, 6]),
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ElementQuality:
    """Per-element quality metrics (arrays of shape ``(n_cells,)``)."""

    aspect_ratio: np.ndarray
    min_jacobian_det: np.ndarray
    max_jacobian_det: np.ndarray
    scaled_jacobian: np.ndarray
    skewness: np.ndarray
    max_warping: np.ndarray
    volume: np.ndarray
    min_edge_length: np.ndarray
    max_edge_length: np.ndarray


@dataclass
class MeshQualityReport:
    """Full mesh quality evaluation report."""

    n_cells: int
    n_nodes: int
    element_quality: ElementQuality
    aspect_ratio_stats: Dict[str, float]
    volume_stats: Dict[str, float]
    edge_length_stats: Dict[str, float]
    jacobian_stats: Dict[str, float]
    scaled_jacobian_stats: Dict[str, float]
    skewness_stats: Dict[str, float]
    warping_stats: Dict[str, float]
    high_aspect_ratio_cells: np.ndarray
    negative_jacobian_cells: np.ndarray
    near_zero_jacobian_cells: np.ndarray
    high_skewness_cells: np.ndarray
    high_warping_cells: np.ndarray
    large_volume_variation_cells: np.ndarray
    warnings_issued: List[str] = field(default_factory=list)
    is_valid: bool = True


# ---------------------------------------------------------------------------
# Shape function helpers (precomputed at module level)
# ---------------------------------------------------------------------------


def _shape_function_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Compute trilinear shape function derivatives at a single point.

    Returns:
        dN: ``(8, 3)`` array where ``dN[i, j]`` = dN_i / d(xi_j).
    """
    dN = np.empty((8, 3))
    for i in range(8):
        xi_i, eta_i, zeta_i = _REF_COORDS[i]
        dN[i, 0] = 0.125 * xi_i * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta)
        dN[i, 1] = 0.125 * (1.0 + xi_i * xi) * eta_i * (1.0 + zeta_i * zeta)
        dN[i, 2] = 0.125 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * zeta_i
    return dN


# Precompute derivatives at evaluation points (9) and Gauss points (8).
_dN_EVAL = np.array([_shape_function_derivatives(*pt) for pt in _EVAL_POINTS])
_dN_GAUSS = np.array([_shape_function_derivatives(*pt) for pt in _GAUSS_PTS])


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------


def _extract_cell_coordinates(msh) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build a ``(n_local_cells, 8, 3)`` coordinate array from a DOLFINx mesh.

    Only owned (non-ghost) cells are included so that MPI reductions
    do not double-count shared cells.

    Returns:
        cell_coords: ``(n_local, 8, 3)`` node coordinates per cell.
        cell_ids: ``(n_local,)`` local cell indices.
        n_cells_global: total number of cells across all ranks.
        n_nodes_global: total number of geometry nodes across all ranks.
    """
    tdim = msh.topology.dim
    map_c = msh.topology.index_map(tdim)
    n_local = map_c.size_local
    n_cells_global = map_c.size_global

    geom_dofmap = msh.geometry.dofmap  # (n_cells_local+ghost, 8)
    coords_all = msh.geometry.x  # (n_nodes_local+ghost, 3)

    node_idx = np.asarray(geom_dofmap[:n_local], dtype=np.int64)
    cell_coords = coords_all[node_idx]  # (n_local, 8, 3)
    cell_ids = np.arange(n_local, dtype=np.int64)

    map_g = msh.geometry.index_map()
    n_nodes_global = map_g.size_global

    return cell_coords, cell_ids, n_cells_global, n_nodes_global


def _compute_edge_metrics(
    cell_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute edge-based metrics for all cells.

    Returns:
        aspect_ratio, min_edge, max_edge, mean_edge — each ``(n_cells,)``.
    """
    edge_vecs = cell_coords[:, _EDGES[:, 1], :] - cell_coords[:, _EDGES[:, 0], :]
    edge_lengths = np.linalg.norm(edge_vecs, axis=2)  # (n_cells, 12)
    min_edge = edge_lengths.min(axis=1)
    max_edge = edge_lengths.max(axis=1)
    mean_edge = edge_lengths.mean(axis=1)
    aspect_ratio = np.where(min_edge > 1.0e-15, max_edge / min_edge, np.inf)
    return aspect_ratio, min_edge, max_edge, mean_edge


def _compute_jacobian_metrics(
    cell_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Compute Jacobian determinant and scaled Jacobian at evaluation points.

    The node ordering convention may produce a consistently negative
    Jacobian determinant (left-handed mapping). This is not element
    inversion — it is a handedness convention. This function detects
    the dominant orientation sign and reports quality relative to it.

    Returns:
        min_abs_det_J: ``(n_cells,)`` minimum ``|det(J)|`` across eval points.
        max_abs_det_J: ``(n_cells,)`` maximum ``|det(J)|`` across eval points.
        scaled_jacobian: ``(n_cells,)`` minimum ``|det(J)| / prod(edge_norms)``
            across corner eval points.
        orientation_sign: ``+1`` or ``-1`` — the dominant orientation.
        inverted_mask: ``(n_cells,)`` bool array — ``True`` for elements
            whose orientation differs from the majority (truly inverted).
    """
    n_cells = cell_coords.shape[0]

    # J[p,n] = dN_EVAL[p]^T @ X[n] via einsum -> (n_eval, n_cells, 3, 3)
    J_all = np.einsum("pki,nkj->pnij", _dN_EVAL, cell_coords)
    det_J = np.linalg.det(J_all)  # (n_eval, n_cells)

    # Determine the dominant orientation from the centre eval point (last).
    centre_det = det_J[-1]  # (n_cells,)
    n_positive = np.sum(centre_det > 0)
    n_negative = np.sum(centre_det < 0)
    orientation_sign = 1 if n_positive >= n_negative else -1

    # An element is truly inverted if its centre det has opposite sign.
    inverted_mask = (centre_det * orientation_sign) < 0

    # Use orientation-adjusted determinants for quality metrics.
    oriented_det = det_J * orientation_sign  # positive for correctly oriented
    min_oriented_det = oriented_det.min(axis=0)
    max_oriented_det = oriented_det.max(axis=0)

    # Scaled Jacobian at the 8 corner eval points (orientation-adjusted).
    scaled_J_corners = np.empty((8, n_cells))
    for c_idx, (node, neighbours) in enumerate(_CORNER_NEIGHBOURS):
        edge_vecs = (
            cell_coords[:, neighbours, :] - cell_coords[:, node : node + 1, :]
        )  # (n_cells, 3, 3)
        edge_norms = np.linalg.norm(edge_vecs, axis=2)  # (n_cells, 3)
        prod_norms = np.prod(edge_norms, axis=1)  # (n_cells,)
        scaled_J_corners[c_idx] = np.where(
            prod_norms > 1.0e-30,
            oriented_det[c_idx] / prod_norms,
            0.0,
        )

    scaled_jacobian = scaled_J_corners.min(axis=0)
    return min_oriented_det, max_oriented_det, scaled_jacobian, orientation_sign, inverted_mask


def _compute_volume(cell_coords: np.ndarray) -> np.ndarray:
    """Compute element volumes using 2x2x2 Gauss quadrature.

    Uses absolute value of the Jacobian determinant so that volumes
    are always positive regardless of the node ordering handedness.

    Returns:
        volumes: ``(n_cells,)``
    """
    J_gauss = np.einsum("pki,nkj->pnij", _dN_GAUSS, cell_coords)
    det_J_gauss = np.linalg.det(J_gauss)  # (8, n_cells)
    # All Gauss weights are 1.0 for 2x2x2 quadrature.
    volumes = np.abs(det_J_gauss).sum(axis=0)
    return volumes


def _compute_skewness(cell_coords: np.ndarray) -> np.ndarray:
    """Compute element skewness from opposing face normal alignment.

    For a perfect hex, opposing face normals are anti-parallel (dot = -1).
    Skewness = max over opposing pairs of ``|1 + dot(n_a, n_b)|``.

    Returns:
        skewness: ``(n_cells,)`` in range [0, 2].
    """
    n_cells = cell_coords.shape[0]

    # Compute unit normals for all 6 faces via diagonal cross products.
    face_normals = np.empty((n_cells, 6, 3))
    for f_idx, fn in enumerate(_FACE_NODES):
        p0 = cell_coords[:, fn[0]]
        p1 = cell_coords[:, fn[1]]
        p2 = cell_coords[:, fn[2]]
        p3 = cell_coords[:, fn[3]]
        normal = np.cross(p2 - p0, p3 - p1)
        norms = np.linalg.norm(normal, axis=1, keepdims=True)
        norms = np.where(norms < 1.0e-15, 1.0, norms)
        face_normals[:, f_idx] = normal / norms

    skew = np.zeros(n_cells)
    for fa, fb in _OPPOSING_FACES:
        dots = np.einsum("ij,ij->i", face_normals[:, fa], face_normals[:, fb])
        skew = np.maximum(skew, np.abs(1.0 + dots))

    return skew


def _compute_warping(cell_coords: np.ndarray) -> np.ndarray:
    """Compute face warping (non-planarity) for all elements.

    Warping is the maximum distance of any face node from the best-fit
    plane through the face, normalised by the face characteristic length.

    Returns:
        max_warping: ``(n_cells,)``
    """
    n_cells = cell_coords.shape[0]
    max_warp = np.zeros(n_cells)

    for fn in _FACE_NODES:
        face_pts = cell_coords[:, fn]  # (n_cells, 4, 3)
        centroid = face_pts.mean(axis=1, keepdims=True)  # (n_cells, 1, 3)
        centered = face_pts - centroid  # (n_cells, 4, 3)

        diag1 = face_pts[:, 2] - face_pts[:, 0]
        diag2 = face_pts[:, 3] - face_pts[:, 1]
        normal = np.cross(diag1, diag2)
        norm_len = np.linalg.norm(normal, axis=1, keepdims=True)
        norm_len = np.where(norm_len < 1.0e-15, 1.0, norm_len)
        normal = normal / norm_len  # (n_cells, 3)

        dists = np.abs(np.einsum("ijk,ik->ij", centered, normal))  # (n_cells, 4)
        max_face_warp = dists.max(axis=1)

        char_len = 0.5 * (
            np.linalg.norm(diag1, axis=1) + np.linalg.norm(diag2, axis=1)
        )
        char_len = np.where(char_len < 1.0e-15, 1.0, char_len)

        max_warp = np.maximum(max_warp, max_face_warp / char_len)

    return max_warp


# ---------------------------------------------------------------------------
# Summary statistics helper
# ---------------------------------------------------------------------------


def _array_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute min, max, mean, median, and 95th percentile of an array."""
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _array_stats_global(
    arr: np.ndarray, comm: MPI.Comm,
) -> Dict[str, float]:
    """Compute global min, max, mean, median, and P95 across MPI ranks.

    Each rank contributes its local array; statistics are computed on
    the gathered (concatenated) array on rank 0 and broadcast back.
    """
    all_arrs = comm.gather(arr, root=0)
    if comm.rank == 0:
        combined = np.concatenate(all_arrs)
        stats = _array_stats(combined)
    else:
        stats = None
    stats = comm.bcast(stats, root=0)
    return stats


# ---------------------------------------------------------------------------
# Physics-aware checks
# ---------------------------------------------------------------------------


def _check_volumetric_locking(
    cfg: dict,
    warnings_list: List[str],
) -> None:
    """Warn if Poisson ratio + linear hex risks volumetric locking."""
    nu_fresh = float(cfg.get("hardening", {}).get("nu_fresh", 0.3))
    if nu_fresh > 0.45:
        msg = (
            f"nu_fresh={nu_fresh} with linear hexahedral (H8) elements risks "
            f"volumetric locking. The DG1 formulation provides partial relief, "
            f"but monitor pressure oscillations carefully."
        )
        warnings.warn(msg, VolumetricLockingWarning, stacklevel=3)
        warnings_list.append(msg)


def _check_penalty_adequacy(
    cfg: dict,
    edge_stats: Dict[str, float],
    warnings_list: List[str],
) -> None:
    """Check if SIP / Nitsche penalty multipliers are reasonable."""
    interface_cfg = cfg.get("interface", {})
    gamma_bonded = float(interface_cfg.get("gamma_bonded_mult", 40.0))

    if gamma_bonded < 10:
        msg = (
            f"SIP penalty multiplier gamma_bonded_mult={gamma_bonded} "
            f"may be too low for stability."
        )
        warnings.warn(msg, PoorConditioningWarning, stacklevel=3)
        warnings_list.append(msg)

    if gamma_bonded > 500:
        msg = (
            f"SIP penalty multiplier gamma_bonded_mult={gamma_bonded} "
            f"is very high and may cause ill-conditioning."
        )
        warnings.warn(msg, PoorConditioningWarning, stacklevel=3)
        warnings_list.append(msg)

    gamma_boundary = float(interface_cfg.get("gamma_boundary_mult", 300.0))
    if gamma_boundary > 1000:
        msg = (
            f"Boundary penalty multiplier gamma_boundary_mult={gamma_boundary} "
            f"is very high and may cause ill-conditioning."
        )
        warnings.warn(msg, PoorConditioningWarning, stacklevel=3)
        warnings_list.append(msg)


def _check_size_uniformity(
    volumes: np.ndarray,
    cell_ids: np.ndarray,
    warnings_list: List[str],
    threshold: float,
) -> np.ndarray:
    """Flag cells where the global volume ratio exceeds *threshold*."""
    vol_min = volumes.min()
    vol_ratio = volumes.max() / max(vol_min, 1.0e-30)
    flagged = np.array([], dtype=int)
    if vol_ratio > threshold:
        msg = (
            f"Volume ratio (max/min) = {vol_ratio:.1f} exceeds threshold "
            f"{threshold:.1f}. Large element size jumps may cause "
            f"conditioning issues."
        )
        warnings.warn(msg, PoorConditioningWarning, stacklevel=3)
        warnings_list.append(msg)
        median_vol = np.median(volumes)
        flagged = cell_ids[volumes > threshold * median_vol]
    return flagged


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary_table(report: MeshQualityReport) -> None:
    """Print a formatted mesh quality summary to stdout."""
    sep = "=" * 68
    thin = "-" * 68

    print(f"\n{sep}")
    print("  Mesh Quality Report")
    print(sep)
    print(f"  Cells: {report.n_cells}    Nodes: {report.n_nodes}")
    print()

    header = (
        f"  {'Metric':<22s}{'Min':>10s}{'Max':>10s}"
        f"{'Mean':>10s}{'Median':>10s}{'P95':>10s}"
    )
    print(header)
    print(f"  {thin[2:]}")

    def _row(label: str, stats: Dict[str, float], fmt: str = ".4f") -> str:
        return (
            f"  {label:<22s}"
            f"{stats['min']:>10{fmt}}"
            f"{stats['max']:>10{fmt}}"
            f"{stats['mean']:>10{fmt}}"
            f"{stats['median']:>10{fmt}}"
            f"{stats['p95']:>10{fmt}}"
        )

    print(_row("Aspect ratio", report.aspect_ratio_stats, ".2f"))
    print(_row("Scaled Jacobian", report.scaled_jacobian_stats))
    print(_row("Min Jacobian det", report.jacobian_stats))
    print(_row("Skewness", report.skewness_stats))
    print(_row("Warping", report.warping_stats, ".6f"))
    print(_row("Volume", report.volume_stats, ".4f"))
    print(_row("Min edge length", report.edge_length_stats, ".4f"))

    print()
    print("  Flagged elements:")

    ar_thresh = report.aspect_ratio_stats.get("_threshold", _DEFAULT_ASPECT_RATIO_THRESHOLD)
    n_ar = len(report.high_aspect_ratio_cells)
    pct_ar = 100.0 * n_ar / max(report.n_cells, 1)
    print(f"    Aspect ratio > {ar_thresh:.0f}:     {n_ar:>6d} cells ({pct_ar:.1f}%)")

    n_neg = len(report.negative_jacobian_cells)
    print(f"    Inverted elements:       {n_neg:>6d} cells")

    n_nz = len(report.near_zero_jacobian_cells)
    print(f"    Near-zero Jacobian:      {n_nz:>6d} cells")

    n_sk = len(report.high_skewness_cells)
    print(f"    High skewness:           {n_sk:>6d} cells")

    n_wp = len(report.high_warping_cells)
    print(f"    High warping:            {n_wp:>6d} cells")

    n_vol = len(report.large_volume_variation_cells)
    print(f"    Volume variation:        {n_vol:>6d} cells")

    if report.warnings_issued:
        print()
        print("  Warnings:")
        for w in report.warnings_issued:
            print(f"    [!] {w}")

    status = "PASS" if report.is_valid else "FAIL"
    print()
    print(f"  Overall: {status}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_mesh_quality(
    msh,
    cfg: dict,
    comm: Optional[MPI.Comm] = None,
) -> MeshQualityReport:
    """Evaluate mesh quality on the DOLFINx mesh and issue warnings.

    This function should be called *after* the DOLFINx mesh has been
    created and partitioned.  Each rank evaluates its owned cells; global
    statistics are reduced across all ranks.

    Args:
        msh: A ``dolfinx.mesh.Mesh`` (hexahedral).
        cfg: Full simulation configuration dictionary.
        comm: MPI communicator. If ``None``, uses ``msh.comm``.

    Returns:
        A :class:`MeshQualityReport` containing all metrics, flagged
        elements, and a pass/fail verdict.
    """
    if comm is None:
        comm = msh.comm
    rank = comm.rank
    mq_cfg = cfg.get("mesh_quality", {})

    if not mq_cfg.get("enabled", True):
        empty = np.array([], dtype=int)
        dummy = ElementQuality(
            aspect_ratio=np.array([]),
            min_jacobian_det=np.array([]),
            max_jacobian_det=np.array([]),
            scaled_jacobian=np.array([]),
            skewness=np.array([]),
            max_warping=np.array([]),
            volume=np.array([]),
            min_edge_length=np.array([]),
            max_edge_length=np.array([]),
        )
        zero_stats: Dict[str, float] = {
            "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0,
        }
        return MeshQualityReport(
            n_cells=0,
            n_nodes=0,
            element_quality=dummy,
            aspect_ratio_stats=zero_stats,
            volume_stats=zero_stats,
            edge_length_stats=zero_stats,
            jacobian_stats=zero_stats,
            scaled_jacobian_stats=zero_stats,
            skewness_stats=zero_stats,
            warping_stats=zero_stats,
            high_aspect_ratio_cells=empty,
            negative_jacobian_cells=empty,
            near_zero_jacobian_cells=empty,
            high_skewness_cells=empty,
            high_warping_cells=empty,
            large_volume_variation_cells=empty,
        )

    # --- Thresholds ---
    ar_thresh = float(mq_cfg.get("aspect_ratio_threshold", _DEFAULT_ASPECT_RATIO_THRESHOLD))
    sj_thresh = float(mq_cfg.get("min_scaled_jacobian", _DEFAULT_MIN_SCALED_JACOBIAN))
    jnz_tol = float(mq_cfg.get("jacobian_near_zero_tol", _DEFAULT_JACOBIAN_NEAR_ZERO_TOL))
    vol_thresh = float(mq_cfg.get("volume_variation_threshold", _DEFAULT_VOLUME_VARIATION_THRESHOLD))
    skew_thresh = float(mq_cfg.get("max_skewness", _DEFAULT_MAX_SKEWNESS))
    warp_thresh = float(mq_cfg.get("max_warping", _DEFAULT_MAX_WARPING))
    fail_neg_j = mq_cfg.get("fail_on_negative_jacobian", _DEFAULT_FAIL_ON_NEGATIVE_JACOBIAN)

    # --- Extract coordinates (owned cells only) ---
    cell_coords, cell_ids, n_cells_global, n_nodes_global = (
        _extract_cell_coordinates(msh)
    )
    n_local = cell_coords.shape[0]

    # --- Compute metrics on local partition ---
    aspect_ratio, min_edge, max_edge, mean_edge = _compute_edge_metrics(cell_coords)
    min_det_J, max_det_J, scaled_jacobian, orient_sign, inverted = (
        _compute_jacobian_metrics(cell_coords)
    )
    volumes = _compute_volume(cell_coords)
    skewness = _compute_skewness(cell_coords)
    max_warping = _compute_warping(cell_coords)

    eq = ElementQuality(
        aspect_ratio=aspect_ratio,
        min_jacobian_det=min_det_J,
        max_jacobian_det=max_det_J,
        scaled_jacobian=scaled_jacobian,
        skewness=skewness,
        max_warping=max_warping,
        volume=volumes,
        min_edge_length=min_edge,
        max_edge_length=max_edge,
    )

    # --- Global summary statistics (MPI reduction) ---
    ar_stats = _array_stats_global(aspect_ratio, comm)
    ar_stats["_threshold"] = ar_thresh
    vol_stats = _array_stats_global(volumes, comm)
    edge_stats = _array_stats_global(min_edge, comm)
    jac_stats = _array_stats_global(min_det_J, comm)
    sj_stats = _array_stats_global(scaled_jacobian, comm)
    skew_stats = _array_stats_global(skewness, comm)
    warp_stats = _array_stats_global(max_warping, comm)

    # --- Flag elements (local counts, then reduce) ---
    high_ar = cell_ids[aspect_ratio > ar_thresh]
    neg_jac = cell_ids[inverted]
    nz_jac = cell_ids[(~inverted) & (min_det_J >= 0.0) & (min_det_J < jnz_tol)]
    high_skew = cell_ids[skewness > skew_thresh]
    high_warp = cell_ids[max_warping > warp_thresh]

    # Global counts for warning messages.
    n_high_ar = comm.allreduce(len(high_ar), op=MPI.SUM)
    n_neg_jac = comm.allreduce(len(neg_jac), op=MPI.SUM)
    n_nz_jac = comm.allreduce(len(nz_jac), op=MPI.SUM)
    n_high_skew = comm.allreduce(len(high_skew), op=MPI.SUM)
    n_high_warp = comm.allreduce(len(high_warp), op=MPI.SUM)

    # Agree on orientation sign across ranks.
    local_pos = int(np.sum(~inverted))
    global_pos = comm.allreduce(local_pos, op=MPI.SUM)
    orient_sign_global = 1 if global_pos >= (n_cells_global - global_pos) else -1

    # --- Warnings ---
    warn_list: List[str] = []

    if orient_sign_global == -1:
        info_msg = (
            "Node ordering produces a left-handed mapping "
            "(consistent negative raw Jacobian). This is a convention "
            "issue, not element inversion \u2014 quality metrics use the "
            "orientation-adjusted determinant."
        )
        warn_list.append(info_msg)

    if n_high_ar > 0:
        pct = 100.0 * n_high_ar / n_cells_global
        msg = (
            f"{n_high_ar} elements ({pct:.1f}%) have aspect ratio "
            f"> {ar_thresh:.0f}. This may slow Newton convergence."
        )
        warnings.warn(msg, HighAspectRatioWarning, stacklevel=2)
        warn_list.append(msg)

    if n_neg_jac > 0:
        msg = (
            f"{n_neg_jac} elements have inverted orientation "
            f"(Jacobian sign differs from the {n_cells_global - n_neg_jac} "
            f"majority). These are truly invalid elements."
        )
        warnings.warn(msg, NegativeJacobianWarning, stacklevel=2)
        warn_list.append(msg)

    if n_nz_jac > 0:
        msg = (
            f"{n_nz_jac} elements have near-zero Jacobian determinant "
            f"(< {jnz_tol:.1e}). These are nearly degenerate."
        )
        warnings.warn(msg, NegativeJacobianWarning, stacklevel=2)
        warn_list.append(msg)

    if n_high_skew > 0:
        pct = 100.0 * n_high_skew / n_cells_global
        msg = (
            f"{n_high_skew} elements ({pct:.1f}%) have skewness "
            f"> {skew_thresh:.2f}."
        )
        warnings.warn(msg, MeshQualityWarning, stacklevel=2)
        warn_list.append(msg)

    if n_high_warp > 0:
        pct = 100.0 * n_high_warp / n_cells_global
        msg = (
            f"{n_high_warp} elements ({pct:.1f}%) have warping "
            f"> {warp_thresh:.2f}."
        )
        warnings.warn(msg, MeshQualityWarning, stacklevel=2)
        warn_list.append(msg)

    # Physics-aware checks.
    _check_volumetric_locking(cfg, warn_list)
    _check_penalty_adequacy(cfg, edge_stats, warn_list)
    vol_flagged = _check_size_uniformity(
        volumes, cell_ids, warn_list, vol_thresh,
    )

    # --- Validity (global agreement) ---
    is_valid = True
    if fail_neg_j and n_neg_jac > 0:
        is_valid = False

    report = MeshQualityReport(
        n_cells=n_cells_global,
        n_nodes=n_nodes_global,
        element_quality=eq,
        aspect_ratio_stats=ar_stats,
        volume_stats=vol_stats,
        edge_length_stats=edge_stats,
        jacobian_stats=jac_stats,
        scaled_jacobian_stats=sj_stats,
        skewness_stats=skew_stats,
        warping_stats=warp_stats,
        high_aspect_ratio_cells=high_ar,
        negative_jacobian_cells=neg_jac,
        near_zero_jacobian_cells=nz_jac,
        high_skewness_cells=high_skew,
        high_warping_cells=high_warp,
        large_volume_variation_cells=vol_flagged,
        warnings_issued=warn_list,
        is_valid=is_valid,
    )

    if rank == 0:
        _print_summary_table(report)

    return report
