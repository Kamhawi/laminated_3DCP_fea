"""Barrel-vault output helpers for normal production runs.

This module provides:
- cell/facet metadata builders for barrel-vault meshes,
- per-step inter-layer opening/damage summaries,
- CSV/JSON writers for run outputs,
- figure generation for a completed run directory.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


REGION_UNKNOWN = -1
REGION_SPRINGER = 0
REGION_HAUNCH = 1
REGION_CROWN = 2

REGION_LABELS = {
    REGION_UNKNOWN: "unknown",
    REGION_SPRINGER: "springer",
    REGION_HAUNCH: "haunch",
    REGION_CROWN: "crown",
}


@dataclass
class CellOutputData:
    """Cell metadata in DOLFINx local+ghost ordering."""

    span_index: np.ndarray
    thickness_index: np.ndarray
    length_index: np.ndarray
    region_code: np.ndarray
    centroid_xyz: np.ndarray


@dataclass
class InterlayerOutputData:
    """Inter-layer facet metadata in local facet ownership ordering."""

    facet_key: np.ndarray
    lower_cell: np.ndarray
    upper_cell: np.ndarray
    normal_xyz: np.ndarray
    centroid_xyz: np.ndarray
    span_index: np.ndarray
    thickness_index: np.ndarray
    lower_layer: np.ndarray
    upper_layer: np.ndarray
    region_code: np.ndarray
    junction_flag: np.ndarray


def get_barrel_vault_output_config(cfg: dict) -> dict:
    """Return normalized barrel-vault output configuration."""
    defaults = {
        "enabled": True,
        "generate_figures": True,
        "write_facet_history": False,
        "write_span_profiles": True,
        "sample_every_steps": 1,
        "damage_threshold": 0.05,
        "gap_threshold": 0.50,
        "front_window_time_mult": 1.0,
    }
    user_cfg = dict(cfg.get("barrel_vault_output", {}))
    merged = dict(defaults)
    merged.update(user_cfg)
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["generate_figures"] = bool(merged.get("generate_figures", True))
    merged["write_facet_history"] = bool(merged.get("write_facet_history", False))
    merged["write_span_profiles"] = bool(merged.get("write_span_profiles", True))
    merged["sample_every_steps"] = max(1, int(merged.get("sample_every_steps", 1)))
    merged["damage_threshold"] = float(merged.get("damage_threshold", 0.05))
    merged["gap_threshold"] = float(merged.get("gap_threshold", 0.50))
    merged["front_window_time_mult"] = max(
        0.0, float(merged.get("front_window_time_mult", 1.0))
    )
    return merged


def build_cell_output_data(vault_mesh, cells_lst, perm) -> CellOutputData:
    """Build cell metadata arrays in DOLFINx local+ghost ordering."""
    nodes = vault_mesh.nodes
    n_cells = len(perm)

    span_index = np.empty(n_cells, dtype=np.int32)
    thickness_index = np.empty(n_cells, dtype=np.int32)
    length_index = np.empty(n_cells, dtype=np.int32)
    region_code = np.empty(n_cells, dtype=np.int8)
    centroid_xyz = np.empty((n_cells, 3), dtype=np.float64)

    for dolfinx_idx, original_idx in enumerate(np.asarray(perm, dtype=np.int64)):
        cell = cells_lst[int(original_idx)]
        span_index[dolfinx_idx] = int(getattr(cell, "span_index", -1))
        thickness_index[dolfinx_idx] = int(getattr(cell, "thickness_index", -1))
        layer = getattr(cell, "layer", None)
        length_index[dolfinx_idx] = int(
            getattr(layer, "layer_id", getattr(cell, "length_index", -1))
        )
        centroid_xyz[dolfinx_idx, :] = np.asarray(
            cell.compute_centroid(nodes), dtype=np.float64
        )
        region_name = getattr(getattr(cell, "vault_cell_type", None), "value", None)
        if region_name == "springer":
            region_code[dolfinx_idx] = REGION_SPRINGER
        elif region_name == "haunch":
            region_code[dolfinx_idx] = REGION_HAUNCH
        elif region_name == "crown":
            region_code[dolfinx_idx] = REGION_CROWN
        else:
            region_code[dolfinx_idx] = REGION_UNKNOWN

    return CellOutputData(
        span_index=span_index,
        thickness_index=thickness_index,
        length_index=length_index,
        region_code=region_code,
        centroid_xyz=centroid_xyz,
    )


def build_interlayer_output_data(
    msh,
    interior_facet_tags,
    cell_data: CellOutputData,
    cfg: dict,
) -> InterlayerOutputData:
    """Build inter-layer facet metadata in local facet ownership ordering."""
    empty = InterlayerOutputData(
        facet_key=np.empty(0, dtype=np.int64),
        lower_cell=np.empty(0, dtype=np.int32),
        upper_cell=np.empty(0, dtype=np.int32),
        normal_xyz=np.empty((0, 3), dtype=np.float64),
        centroid_xyz=np.empty((0, 3), dtype=np.float64),
        span_index=np.empty(0, dtype=np.int32),
        thickness_index=np.empty(0, dtype=np.int32),
        lower_layer=np.empty(0, dtype=np.int32),
        upper_layer=np.empty(0, dtype=np.int32),
        region_code=np.empty(0, dtype=np.int8),
        junction_flag=np.empty(0, dtype=bool),
    )
    if interior_facet_tags is None:
        return empty

    il_mask = np.asarray(interior_facet_tags.values == 1, dtype=bool)
    il_facets = np.asarray(interior_facet_tags.indices[il_mask], dtype=np.int32)
    n_il = int(il_facets.size)
    if n_il == 0:
        return empty

    n_span = int(cfg["mesh"]["n_span"])
    n_thickness = int(cfg["mesh"]["n_thickness"])
    support_upto = int(cfg["boundary_conditions"]["intrados_dirichlet_upto_layer"])

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(fdim, 0)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    f_to_v = msh.topology.connectivity(fdim, 0)

    from dolfinx.mesh import entities_to_geometry

    geom_x = msh.geometry.x

    facet_key = np.empty(n_il, dtype=np.int64)
    lower_cell = np.empty(n_il, dtype=np.int32)
    upper_cell = np.empty(n_il, dtype=np.int32)
    normal_xyz = np.empty((n_il, 3), dtype=np.float64)
    centroid_xyz = np.empty((n_il, 3), dtype=np.float64)
    span_index = np.empty(n_il, dtype=np.int32)
    thickness_index = np.empty(n_il, dtype=np.int32)
    lower_layer = np.empty(n_il, dtype=np.int32)
    upper_layer = np.empty(n_il, dtype=np.int32)
    region_code = np.empty(n_il, dtype=np.int8)
    junction_flag = np.empty(n_il, dtype=bool)

    for idx, facet in enumerate(il_facets):
        cells = np.asarray(f_to_c.links(int(facet)), dtype=np.int32)
        if cells.size == 0:
            continue
        if cells.size == 1:
            c0 = c1 = int(cells[0])
        else:
            c0 = int(cells[0])
            c1 = int(cells[1])

        l0 = int(cell_data.length_index[c0])
        l1 = int(cell_data.length_index[c1])
        if l0 <= l1:
            lo = c0
            up = c1
        else:
            lo = c1
            up = c0

        lower_cell[idx] = lo
        upper_cell[idx] = up
        lower_layer[idx] = int(cell_data.length_index[lo])
        upper_layer[idx] = int(cell_data.length_index[up])
        span_index[idx] = int(cell_data.span_index[lo])
        thickness_index[idx] = int(cell_data.thickness_index[lo])
        region_code[idx] = int(cell_data.region_code[lo])
        junction_flag[idx] = lower_layer[idx] == support_upto
        facet_key[idx] = (
            int(lower_layer[idx]) * n_span * n_thickness
            + int(thickness_index[idx]) * n_span
            + int(span_index[idx])
        )

        verts = np.asarray(f_to_v.links(int(facet)), dtype=np.int32)
        geom_dofs = entities_to_geometry(msh, 0, verts, False).reshape(-1)
        coords = np.asarray(geom_x[geom_dofs], dtype=np.float64)
        centroid_xyz[idx, :] = np.mean(coords, axis=0)
        if coords.shape[0] >= 4:
            normal = np.cross(coords[1] - coords[0], coords[3] - coords[0])
        elif coords.shape[0] >= 3:
            normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        orient = cell_data.centroid_xyz[up] - cell_data.centroid_xyz[lo]
        if np.dot(normal, orient) < 0.0:
            normal *= -1.0
        norm = float(np.linalg.norm(normal))
        if norm > 1.0e-12:
            normal /= norm
        else:
            normal[:] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        normal_xyz[idx, :] = normal

    return InterlayerOutputData(
        facet_key=facet_key,
        lower_cell=lower_cell,
        upper_cell=upper_cell,
        normal_xyz=normal_xyz,
        centroid_xyz=centroid_xyz,
        span_index=span_index,
        thickness_index=thickness_index,
        lower_layer=lower_layer,
        upper_layer=upper_layer,
        region_code=region_code,
        junction_flag=junction_flag,
    )


def mean_birth_interval(birth_times: np.ndarray) -> float:
    """Return the mean positive birth-time increment."""
    uniq = np.unique(np.asarray(birth_times, dtype=np.float64))
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 1.0e-12]
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs))


def safe_ratio(numerator: float, denominator: float, large: float = 1.0e12) -> float:
    """Return a bounded ratio that stays finite when the denominator vanishes."""
    num = float(numerator)
    den = float(denominator)
    if abs(den) > 1.0e-15:
        return num / den
    if abs(num) > 1.0e-15:
        return large if num > 0.0 else -large
    return 0.0


def _max_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
    if values.size == 0 or not np.any(mask):
        return 0.0
    return float(np.max(values[mask]))


def _pick_event_index(score: np.ndarray, mask: np.ndarray) -> Optional[int]:
    if score.size == 0 or not np.any(mask):
        return None
    candidates = np.flatnonzero(mask)
    return int(candidates[np.argmax(score[candidates])])


def summarize_region(
    region_code: int,
    junction_flag: bool,
    front_band_flag: bool,
) -> str:
    """Return a coarse region summary used in event JSON output."""
    if bool(front_band_flag):
        return "deposition_front"
    if bool(junction_flag):
        return "junction"
    return REGION_LABELS.get(int(region_code), "unknown")


def _format_interface_label(
    lower_layer_zero_based: int, upper_layer_zero_based: int
) -> Dict[str, str]:
    return {
        "interface_zero_based": (
            f"{int(lower_layer_zero_based)}-{int(upper_layer_zero_based)}"
        ),
        "interface_one_based": (
            f"{int(lower_layer_zero_based) + 1}-{int(upper_layer_zero_based) + 1}"
        ),
    }


def _empty_step_summary() -> dict:
    return {
        "max_opening_mm": 0.0,
        "max_damage": 0.0,
        "junction_opening_mm": 0.0,
        "junction_damage": 0.0,
        "crown_opening_mm": 0.0,
        "crown_damage": 0.0,
        "front_opening_mm": 0.0,
        "front_damage": 0.0,
        "front_active_facets": 0,
        "front_localization_ratio": 0.0,
        "front_vs_crown_opening_ratio": 0.0,
        "crown_vs_junction_opening_ratio": 0.0,
        "junction_opening_ratio": 0.0,
        "junction_damage_ratio": 0.0,
        "cantilever_sag_mm": 0.0,
    }


def compute_step_output_summary(
    u,
    materials,
    cell_to_dofs: np.ndarray,
    birth_times: np.ndarray,
    cfg: dict,
    output_cfg: dict,
    cell_data: CellOutputData,
    interlayer_data: InterlayerOutputData,
    t_val: float,
    active_layers: int,
    front_window_time: float,
) -> dict:
    """Compute inter-layer opening and damage summaries for one step."""
    n_il = int(interlayer_data.facet_key.size)
    if n_il == 0:
        return {
            "summary": _empty_step_summary(),
            "active_rows": {
                "facet_key": np.empty(0, dtype=np.int64),
                "span_index": np.empty(0, dtype=np.int32),
                "thickness_index": np.empty(0, dtype=np.int32),
                "lower_layer_zero_based": np.empty(0, dtype=np.int32),
                "upper_layer_zero_based": np.empty(0, dtype=np.int32),
                "region_code": np.empty(0, dtype=np.int8),
                "junction_flag": np.empty(0, dtype=bool),
                "front_band_flag": np.empty(0, dtype=bool),
                "centroid_x_mm": np.empty(0, dtype=np.float64),
                "centroid_y_mm": np.empty(0, dtype=np.float64),
                "centroid_z_mm": np.empty(0, dtype=np.float64),
                "jump_n_mm": np.empty(0, dtype=np.float64),
                "jump_n_open_mm": np.empty(0, dtype=np.float64),
                "jump_t_mm": np.empty(0, dtype=np.float64),
                "mode_i_drive": np.empty(0, dtype=np.float64),
                "mode_ii_drive": np.empty(0, dtype=np.float64),
                "facet_damage": np.empty(0, dtype=np.float64),
                "damage_max_pair": np.empty(0, dtype=np.float64),
            },
            "events": {"damage_idx": None, "gap_idx": None, "junction_idx": None},
        }

    lower = interlayer_data.lower_cell
    upper = interlayer_data.upper_cell
    normals = interlayer_data.normal_xyz

    u_arr = u.x.array
    dofs_lower = cell_to_dofs[lower]
    dofs_upper = cell_to_dofs[upper]
    u_lower = u_arr[dofs_lower].reshape(n_il, -1, 3).mean(axis=1)
    u_upper = u_arr[dofs_upper].reshape(n_il, -1, 3).mean(axis=1)
    jump_vec = u_upper - u_lower

    jump_n = np.sum(jump_vec * normals, axis=1)
    jump_n_open = np.maximum(jump_n, 0.0)
    jump_t_vec = jump_vec - jump_n[:, None] * normals
    jump_t_mag = np.linalg.norm(jump_t_vec, axis=1)

    active_mask = (birth_times[lower] <= t_val) & (birth_times[upper] <= t_val)
    recent_birth = np.maximum(birth_times[lower], birth_times[upper])
    front_band_flag = (
        active_mask
        & (interlayer_data.upper_layer == max(active_layers - 1, -1))
        & ((t_val - recent_birth) >= -1.0e-12)
        & ((t_val - recent_birth) <= front_window_time + 1.0e-12)
    )

    sigma_y_arr = materials.sigma_y.x.array
    tau_y_arr = materials.tau_y.x.array
    e_arr = materials.E.x.array
    dmg_arr = materials.damage_max.x.array

    sigma_y_avg = 0.5 * (sigma_y_arr[upper] + sigma_y_arr[lower])
    tau_y_avg = 0.5 * (tau_y_arr[upper] + tau_y_arr[lower])
    e_avg = 0.5 * (e_arr[upper] + e_arr[lower])

    tau_0_pa = float(cfg["material"]["tau_0"])
    a_thix_pa_s = float(cfg["material"]["A_thix"])
    nozzle_pressure_pa = float(cfg["interface"]["nozzle_pressure"])
    g_i_c = float(cfg["interface"]["G_Ic"])
    g_ii_c = float(cfg["interface"]["G_IIc"])
    k_min_mult = float(cfg["interface"].get("K_min_mult", 0.1))

    t_open = np.abs(birth_times[upper] - birth_times[lower])
    tau_sub = tau_0_pa + a_thix_pa_s * t_open
    phi = nozzle_pressure_pa / np.maximum(tau_sub, 1.0e-12)
    beta = np.clip(1.0 - np.exp(-phi), 1.0e-8, 1.0)

    g_i_c_eff = np.maximum(beta * g_i_c, 1.0e-12)
    g_ii_c_eff = np.maximum(beta * g_ii_c, 1.0e-12)

    h_upper = np.linalg.norm(
        cell_data.centroid_xyz[upper] - interlayer_data.centroid_xyz, axis=1
    )
    h_lower = np.linalg.norm(
        cell_data.centroid_xyz[lower] - interlayer_data.centroid_xyz, axis=1
    )
    h_avg = np.maximum(h_upper + h_lower, 1.0e-6)

    k_n = beta * sigma_y_avg ** 2 / (2.0 * g_i_c_eff)
    k_t = beta * tau_y_avg ** 2 / (2.0 * g_ii_c_eff)
    k_min = k_min_mult * e_avg / h_avg
    k_n = np.maximum(k_n, k_min)
    k_t = np.maximum(k_t, k_min)

    mode_i_drive = jump_n_open ** 2 / np.maximum(
        2.0 * g_i_c_eff / np.maximum(k_n, 1.0e-12), 1.0e-12
    )
    mode_ii_drive = jump_t_mag ** 2 / np.maximum(
        2.0 * g_ii_c_eff / np.maximum(k_t, 1.0e-12), 1.0e-12
    )
    facet_damage = np.clip(1.0 - np.exp(-(mode_i_drive + mode_ii_drive)), 0.0, 1.0)
    damage_max_pair = np.maximum(dmg_arr[upper], dmg_arr[lower])

    jump_n = np.where(active_mask, jump_n, 0.0)
    jump_n_open = np.where(active_mask, jump_n_open, 0.0)
    jump_t_mag = np.where(active_mask, jump_t_mag, 0.0)
    mode_i_drive = np.where(active_mask, mode_i_drive, 0.0)
    mode_ii_drive = np.where(active_mask, mode_ii_drive, 0.0)
    facet_damage = np.where(active_mask, facet_damage, 0.0)
    damage_max_pair = np.where(active_mask, damage_max_pair, 0.0)
    front_band_flag = np.where(active_mask, front_band_flag, False)

    region_is_crown = interlayer_data.region_code == REGION_CROWN
    junction_flag = interlayer_data.junction_flag

    summary = {
        "max_opening_mm": _max_or_zero(jump_n_open, active_mask),
        "max_damage": _max_or_zero(facet_damage, active_mask),
        "junction_opening_mm": _max_or_zero(jump_n_open, active_mask & junction_flag),
        "junction_damage": _max_or_zero(facet_damage, active_mask & junction_flag),
        "crown_opening_mm": _max_or_zero(jump_n_open, active_mask & region_is_crown),
        "crown_damage": _max_or_zero(facet_damage, active_mask & region_is_crown),
        "front_opening_mm": _max_or_zero(jump_n_open, front_band_flag),
        "front_damage": _max_or_zero(facet_damage, front_band_flag),
        "front_active_facets": int(np.sum(front_band_flag)),
    }
    summary["front_localization_ratio"] = safe_ratio(
        summary["front_opening_mm"], summary["crown_opening_mm"]
    )
    summary["front_vs_crown_opening_ratio"] = summary["front_localization_ratio"]
    summary["crown_vs_junction_opening_ratio"] = safe_ratio(
        summary["crown_opening_mm"], summary["junction_opening_mm"]
    )
    summary["junction_opening_ratio"] = safe_ratio(
        summary["junction_opening_mm"], summary["max_opening_mm"]
    )
    summary["junction_damage_ratio"] = safe_ratio(
        summary["junction_damage"], summary["max_damage"]
    )

    cell_active_mask = birth_times <= t_val
    unsupported_mask = cell_active_mask & (
        cell_data.length_index
        > int(cfg["boundary_conditions"]["intrados_dirichlet_upto_layer"])
    )
    if np.any(unsupported_mask):
        cell_u_z = u_arr[cell_to_dofs].reshape(cell_to_dofs.shape[0], -1, 3).mean(
            axis=1
        )[:, 2]
        summary["cantilever_sag_mm"] = float(np.min(cell_u_z[unsupported_mask]))
    else:
        summary["cantilever_sag_mm"] = 0.0

    damage_idx = _pick_event_index(
        facet_damage,
        active_mask & (facet_damage >= float(output_cfg["damage_threshold"])),
    )
    gap_idx = _pick_event_index(
        jump_n_open,
        active_mask & (jump_n_open >= float(output_cfg["gap_threshold"])),
    )
    junction_idx = _pick_event_index(
        jump_n_open,
        active_mask & junction_flag & (jump_n_open > 0.0),
    )

    active_rows_mask = active_mask
    active_rows = {
        "facet_key": interlayer_data.facet_key[active_rows_mask].copy(),
        "span_index": interlayer_data.span_index[active_rows_mask].copy(),
        "thickness_index": interlayer_data.thickness_index[active_rows_mask].copy(),
        "lower_layer_zero_based": interlayer_data.lower_layer[active_rows_mask].copy(),
        "upper_layer_zero_based": interlayer_data.upper_layer[active_rows_mask].copy(),
        "region_code": interlayer_data.region_code[active_rows_mask].copy(),
        "junction_flag": interlayer_data.junction_flag[active_rows_mask].copy(),
        "front_band_flag": front_band_flag[active_rows_mask].copy(),
        "centroid_x_mm": interlayer_data.centroid_xyz[active_rows_mask, 0].copy(),
        "centroid_y_mm": interlayer_data.centroid_xyz[active_rows_mask, 1].copy(),
        "centroid_z_mm": interlayer_data.centroid_xyz[active_rows_mask, 2].copy(),
        "jump_n_mm": jump_n[active_rows_mask].copy(),
        "jump_n_open_mm": jump_n_open[active_rows_mask].copy(),
        "jump_t_mm": jump_t_mag[active_rows_mask].copy(),
        "mode_i_drive": mode_i_drive[active_rows_mask].copy(),
        "mode_ii_drive": mode_ii_drive[active_rows_mask].copy(),
        "facet_damage": facet_damage[active_rows_mask].copy(),
        "damage_max_pair": damage_max_pair[active_rows_mask].copy(),
    }

    return {
        "summary": summary,
        "active_rows": active_rows,
        "events": {
            "damage_idx": damage_idx,
            "gap_idx": gap_idx,
            "junction_idx": junction_idx,
        },
    }


def build_event_record(
    root_rows: dict, row_index: int, step: int, time_s: float, active_layers: int
) -> dict:
    """Build a JSON-serializable event record from gathered facet data."""
    region_code = int(root_rows["region_code"][row_index])
    junction_flag = bool(root_rows["junction_flag"][row_index])
    front_band_flag = bool(root_rows["front_band_flag"][row_index])
    lower_layer = int(root_rows["lower_layer_zero_based"][row_index])
    upper_layer = int(root_rows["upper_layer_zero_based"][row_index])
    record = {
        "step": int(step),
        "time_s": float(time_s),
        "active_layers": int(active_layers),
        "facet_key": int(root_rows["facet_key"][row_index]),
        "summary_region": summarize_region(region_code, junction_flag, front_band_flag),
        "region_label": REGION_LABELS.get(region_code, "unknown"),
        "front_band_flag": front_band_flag,
        "junction_flag": junction_flag,
        "span_index": int(root_rows["span_index"][row_index]),
        "thickness_index": int(root_rows["thickness_index"][row_index]),
        "lower_layer_zero_based": lower_layer,
        "upper_layer_zero_based": upper_layer,
        "centroid_x_mm": float(root_rows["centroid_x_mm"][row_index]),
        "centroid_y_mm": float(root_rows["centroid_y_mm"][row_index]),
        "centroid_z_mm": float(root_rows["centroid_z_mm"][row_index]),
        "jump_n_mm": float(root_rows["jump_n_mm"][row_index]),
        "jump_n_open_mm": float(root_rows["jump_n_open_mm"][row_index]),
        "jump_t_mm": float(root_rows["jump_t_mm"][row_index]),
        "mode_i_drive": float(root_rows["mode_i_drive"][row_index]),
        "mode_ii_drive": float(root_rows["mode_ii_drive"][row_index]),
        "facet_damage": float(root_rows["facet_damage"][row_index]),
        "damage_max_pair": float(root_rows["damage_max_pair"][row_index]),
    }
    record.update(_format_interface_label(lower_layer, upper_layer))
    return record


def concat_gathered_rows(gathered_rows: Iterable[dict]) -> dict:
    """Concatenate gathered rank-local active facet arrays on rank 0."""
    gathered_rows = list(gathered_rows)
    if not gathered_rows:
        return {}
    keys = list(gathered_rows[0].keys())
    out = {}
    for key in keys:
        parts = [np.asarray(row[key]) for row in gathered_rows if len(row[key]) > 0]
        if parts:
            out[key] = np.concatenate(parts)
        else:
            sample = np.asarray(gathered_rows[0][key])
            out[key] = np.empty(
                0,
                dtype=sample.dtype if sample.dtype != object else np.float64,
            )
    return out


def write_facet_history_rows(
    writer: csv.DictWriter,
    root_rows: dict,
    step: int,
    time_s: float,
    active_layers: int,
) -> None:
    """Append one sampled step of active-facet history rows."""
    n_rows = int(root_rows.get("facet_key", np.empty(0)).size)
    for idx in range(n_rows):
        writer.writerow(
            {
                "step": step,
                "time_s": f"{time_s:.6f}",
                "active_layers": active_layers,
                "facet_key": int(root_rows["facet_key"][idx]),
                "span_index": int(root_rows["span_index"][idx]),
                "thickness_index": int(root_rows["thickness_index"][idx]),
                "lower_layer_zero_based": int(root_rows["lower_layer_zero_based"][idx]),
                "upper_layer_zero_based": int(root_rows["upper_layer_zero_based"][idx]),
                "region_label": REGION_LABELS.get(
                    int(root_rows["region_code"][idx]), "unknown"
                ),
                "junction_flag": int(bool(root_rows["junction_flag"][idx])),
                "front_band_flag": int(bool(root_rows["front_band_flag"][idx])),
                "centroid_x_mm": f"{float(root_rows['centroid_x_mm'][idx]):.6e}",
                "centroid_y_mm": f"{float(root_rows['centroid_y_mm'][idx]):.6e}",
                "centroid_z_mm": f"{float(root_rows['centroid_z_mm'][idx]):.6e}",
                "jump_n_mm": f"{float(root_rows['jump_n_mm'][idx]):.6e}",
                "jump_n_open_mm": f"{float(root_rows['jump_n_open_mm'][idx]):.6e}",
                "jump_t_mm": f"{float(root_rows['jump_t_mm'][idx]):.6e}",
                "mode_i_drive": f"{float(root_rows['mode_i_drive'][idx]):.6e}",
                "mode_ii_drive": f"{float(root_rows['mode_ii_drive'][idx]):.6e}",
                "facet_damage": f"{float(root_rows['facet_damage'][idx]):.6e}",
                "damage_max_pair": f"{float(root_rows['damage_max_pair'][idx]):.6e}",
            }
        )


def write_span_profile_rows(
    writer: csv.DictWriter,
    root_rows: dict,
    step: int,
    time_s: float,
    active_layers: int,
) -> None:
    """Append span-wise, thickness-collapsed profile rows for one sampled step."""
    n_rows = int(root_rows.get("facet_key", np.empty(0)).size)
    if n_rows == 0:
        return

    lower_layer = np.asarray(root_rows["lower_layer_zero_based"], dtype=np.int32)
    span_index = np.asarray(root_rows["span_index"], dtype=np.int32)
    group_key = lower_layer.astype(np.int64) * 1_000_000 + span_index.astype(np.int64)
    order = np.argsort(group_key, kind="mergesort")
    group_key = group_key[order]

    _, starts = np.unique(group_key, return_index=True)
    ends = np.concatenate([starts[1:], np.array([group_key.size])])

    jump_open = np.asarray(root_rows["jump_n_open_mm"], dtype=np.float64)[order]
    facet_damage = np.asarray(root_rows["facet_damage"], dtype=np.float64)[order]
    region_code = np.asarray(root_rows["region_code"], dtype=np.int8)[order]
    junction_flag = np.asarray(root_rows["junction_flag"], dtype=bool)[order]
    lower_layer_sorted = lower_layer[order]
    span_index_sorted = span_index[order]

    for start, end in zip(starts, ends):
        writer.writerow(
            {
                "step": step,
                "time_s": f"{time_s:.6f}",
                "active_layers": active_layers,
                "lower_layer_zero_based": int(lower_layer_sorted[start]),
                "span_index": int(span_index_sorted[start]),
                "region_label": REGION_LABELS.get(int(region_code[start]), "unknown"),
                "junction_flag": int(bool(np.any(junction_flag[start:end]))),
                "max_opening_mm": f"{float(np.max(jump_open[start:end])):.6e}",
                "max_damage": f"{float(np.max(facet_damage[start:end])):.6e}",
                "mean_opening_mm": f"{float(np.mean(jump_open[start:end])):.6e}",
                "mean_damage": f"{float(np.mean(facet_damage[start:end])):.6e}",
            }
        )


def finalize_barrel_vault_results(
    run_dir: Path,
    cfg: dict,
    output_cfg: dict,
    event_state: dict,
    peak_state: dict,
    final_step_summary: dict,
) -> Path:
    """Write the run-level barrel-vault results JSON."""
    first_damage = event_state.get("first_damage")
    first_visible_gap = event_state.get("first_visible_gap")
    first_junction_opening = event_state.get("first_junction_opening")

    hypotheses = {
        "front_localizes_but_gap_forms_at_crown": bool(
            peak_state.get("front_localization_ratio", 0.0) > 1.0
            and first_visible_gap is not None
            and first_visible_gap.get("summary_region") == "crown"
        ),
        "damage_initiates_at_deposition_front": bool(
            first_damage is not None and first_damage.get("front_band_flag", False)
        ),
        "junction_opens_before_crown": bool(
            first_junction_opening is not None
            and first_visible_gap is not None
            and int(first_junction_opening["step"]) <= int(first_visible_gap["step"])
        ),
    }

    results = {
        "output_version": 1,
        "events": {
            "first_damage": first_damage,
            "first_visible_gap": first_visible_gap,
            "first_junction_opening": first_junction_opening,
        },
        "derived": {
            "first_damage_region": (
                None if first_damage is None else first_damage.get("summary_region")
            ),
            "front_localization_ratio": float(
                peak_state.get("front_localization_ratio", 0.0)
            ),
            "junction_damage_ratio": float(
                peak_state.get("junction_damage_ratio", 0.0)
            ),
            "junction_opening_ratio": float(
                peak_state.get("junction_opening_ratio", 0.0)
            ),
            "front_vs_crown_opening_ratio": float(
                peak_state.get("front_vs_crown_opening_ratio", 0.0)
            ),
            "crown_vs_junction_opening_ratio": float(
                peak_state.get("crown_vs_junction_opening_ratio", 0.0)
            ),
        },
        "hypotheses": hypotheses,
        "summary": {
            "final": dict(final_step_summary),
        },
        "config": {
            "boundary_conditions": {
                "intrados_dirichlet_upto_layer": int(
                    cfg["boundary_conditions"]["intrados_dirichlet_upto_layer"]
                )
            },
            "barrel_vault_output": dict(output_cfg),
        },
    }

    results_path = Path(run_dir) / "barrel_vault_results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results_path


def _load_csv_rows(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = int(value)
                    continue
                except (TypeError, ValueError):
                    pass
                try:
                    parsed[key] = float(value)
                    continue
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _pick_heatmap_steps(results: dict, rows: List[dict]) -> List[tuple]:
    final_step = int(rows[-1]["step"]) if rows else 0
    selections = []
    first_damage = results.get("events", {}).get("first_damage")
    first_gap = results.get("events", {}).get("first_visible_gap")
    if first_damage is not None:
        selections.append(("First Damage", int(first_damage["step"])))
    if first_gap is not None:
        gap_step = int(first_gap["step"])
        if not selections or selections[-1][1] != gap_step:
            selections.append(("First Visible Gap", gap_step))
    if not selections or selections[-1][1] != final_step:
        selections.append(("Final State", final_step))
    return selections[:2] if len(selections) > 2 else selections


def _build_heatmap_matrix(
    profile_rows: List[dict], step: int, value_key: str
) -> np.ndarray:
    step_rows = [row for row in profile_rows if int(row["step"]) == int(step)]
    if not step_rows:
        return np.zeros((1, 1), dtype=np.float64)
    max_layer = max(int(row["lower_layer_zero_based"]) for row in step_rows)
    max_span = max(int(row["span_index"]) for row in step_rows)
    matrix = np.zeros((max_layer + 1, max_span + 1), dtype=np.float64)
    for row in step_rows:
        matrix[int(row["lower_layer_zero_based"]), int(row["span_index"])] = float(
            row[value_key]
        )
    return matrix


def generate_barrel_vault_figures(
    run_dir: Path, cfg: Optional[dict] = None
) -> List[Path]:
    """Generate barrel-vault figures for a completed run directory."""
    run_dir = Path(run_dir)
    step_metrics_path = run_dir / "step_metrics.csv"
    span_profiles_path = run_dir / "span_profiles.csv"
    results_path = run_dir / "barrel_vault_results.json"
    if not step_metrics_path.exists() or not span_profiles_path.exists() or not results_path.exists():
        return []

    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(run_dir / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    rows = _load_csv_rows(step_metrics_path)
    profile_rows = _load_csv_rows(span_profiles_path)
    with open(results_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)

    if not rows:
        return []

    x_vals = np.array(
        [row.get("active_layers", row.get("step", 0)) for row in rows], dtype=float
    )
    max_disp = np.array([float(row.get("max_disp_mm", 0.0)) for row in rows], dtype=float)
    cantilever_sag = np.array(
        [float(row.get("cantilever_sag_mm", 0.0)) for row in rows], dtype=float
    )
    max_open = np.array([float(row.get("max_opening_mm", 0.0)) for row in rows], dtype=float)
    junction_open = np.array(
        [float(row.get("junction_opening_mm", 0.0)) for row in rows], dtype=float
    )
    crown_open = np.array(
        [float(row.get("crown_opening_mm", 0.0)) for row in rows], dtype=float
    )
    front_open = np.array(
        [float(row.get("front_opening_mm", 0.0)) for row in rows], dtype=float
    )
    max_damage = np.array([float(row.get("max_damage", 0.0)) for row in rows], dtype=float)
    junction_damage = np.array(
        [float(row.get("junction_damage", 0.0)) for row in rows], dtype=float
    )
    crown_damage = np.array(
        [float(row.get("crown_damage", 0.0)) for row in rows], dtype=float
    )
    front_damage = np.array(
        [float(row.get("front_damage", 0.0)) for row in rows], dtype=float
    )
    front_ratio = np.array(
        [float(row.get("front_vs_crown_opening_ratio", 0.0)) for row in rows], dtype=float
    )
    junction_ratio = np.array(
        [float(row.get("junction_opening_ratio", 0.0)) for row in rows], dtype=float
    )

    event_damage = results.get("events", {}).get("first_damage")
    event_gap = results.get("events", {}).get("first_visible_gap")
    event_junction = results.get("events", {}).get("first_junction_opening")

    out_paths: List[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x_vals, max_disp, lw=2.0, color="#29465B", label="Max displacement")
    ax.plot(x_vals, cantilever_sag, lw=2.0, color="#8F3B1B", label="Cantilever sag")
    ax.set_xlabel("Active layers")
    ax.set_ylabel("Displacement [mm]")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x_vals, max_open, lw=2.0, color="#29465B", label="Max opening")
    ax.plot(x_vals, crown_open, lw=1.8, color="#A23B2A", label="Crown opening")
    ax.plot(x_vals, junction_open, lw=1.8, color="#1B6A4A", label="Junction opening")
    ax.plot(x_vals, front_open, lw=1.8, color="#7B5EA7", label="Front opening")
    ax.set_xlabel("Active layers")
    ax.set_ylabel("Opening [mm]")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x_vals, max_damage, lw=2.0, color="#29465B", label="Max damage")
    ax.plot(x_vals, crown_damage, lw=1.8, color="#A23B2A", label="Crown damage")
    ax.plot(x_vals, junction_damage, lw=1.8, color="#1B6A4A", label="Junction damage")
    ax.plot(x_vals, front_damage, lw=1.8, color="#7B5EA7", label="Front damage")
    ax.set_xlabel("Active layers")
    ax.set_ylabel("Damage [-]")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_vals, front_ratio, lw=2.0, color="#7B5EA7", label="Front / Crown opening")
    ax.plot(x_vals, junction_ratio, lw=2.0, color="#1B6A4A", label="Junction / Max opening")
    ax.set_xlabel("Active layers")
    ax.set_ylabel("Ratio [-]")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    for event in (event_junction, event_damage, event_gap):
        if event is None:
            continue
        x_event = float(event.get("active_layers", event.get("step", 0)))
        for ax in axes.ravel():
            ax.axvline(x_event, color="#444444", lw=0.8, ls="--", alpha=0.35)

    fig.suptitle("Barrel-Vault Output Summary", fontsize=14)
    progress_pdf = run_dir / "barrel_vault_progress.pdf"
    progress_png = run_dir / "barrel_vault_progress.png"
    fig.savefig(progress_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(progress_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_paths.extend([progress_pdf, progress_png])

    selections = _pick_heatmap_steps(results, rows)
    if selections:
        fig, axes = plt.subplots(
            2,
            len(selections),
            figsize=(6.0 * len(selections), 8.0),
            constrained_layout=True,
        )
        if len(selections) == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for col, (title, step) in enumerate(selections):
            open_matrix = _build_heatmap_matrix(profile_rows, step, "max_opening_mm")
            dmg_matrix = _build_heatmap_matrix(profile_rows, step, "max_damage")

            im0 = axes[0, col].imshow(
                open_matrix, origin="lower", aspect="auto", cmap="magma"
            )
            axes[0, col].set_title(f"{title}\nOpening")
            axes[0, col].set_xlabel("Span index")
            axes[0, col].set_ylabel("Lower layer")
            fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

            im1 = axes[1, col].imshow(
                dmg_matrix, origin="lower", aspect="auto", cmap="viridis"
            )
            axes[1, col].set_title(f"{title}\nDamage")
            axes[1, col].set_xlabel("Span index")
            axes[1, col].set_ylabel("Lower layer")
            fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        fig.suptitle("Barrel-Vault Opening/Damage Heatmaps", fontsize=14)
        heatmap_pdf = run_dir / "barrel_vault_heatmaps.pdf"
        heatmap_png = run_dir / "barrel_vault_heatmaps.png"
        fig.savefig(heatmap_pdf, dpi=300, bbox_inches="tight")
        fig.savefig(heatmap_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        out_paths.extend([heatmap_pdf, heatmap_png])

    return out_paths


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for figure regeneration on an existing run directory."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run directory")
    args = parser.parse_args(argv)

    out_paths = generate_barrel_vault_figures(args.run_dir)
    if out_paths:
        for path in out_paths:
            print(path)
        return 0
    print("No figures generated.")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI convenience.
    raise SystemExit(main())
