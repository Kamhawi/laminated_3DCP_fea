"""Validate Fix 2 (K_min_mult reduction) under amplified gravity.

Tests Newton convergence and interface damage activation on the small mesh
with amplified gravity loading. The gravity amplification forces measurable
inter-layer opening to validate that the cohesive law functions correctly
with the reduced K_min floor.

Usage::

    python -m diagnostics.validate_interface_opening [--config PATH]
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import json

import numpy as np
from mpi4py import MPI

from diagnostics.setup import build_diagnostic_state, prepare_state_at_time
from diagnostics.stiffness_hierarchy import (
    extract_interlayer_facet_data,
    _get_facet_cell_pairs,
    compute_cell_diameters,
)
from diagnostics.run_phase1 import _TeeWriter, _results_to_serializable
from solver.newton import solve_newton


# ---------------------------------------------------------------------------
# Post-solve interface analysis
# ---------------------------------------------------------------------------

def _extract_interface_jumps(state):
    """Compute jump displacements on inter-layer facets from current u.

    Returns dict with per-facet normal jump, tangential jump magnitude,
    and damage computed from the TSL formula.
    """
    msh = state.msh
    tags = state.interior_facet_tags
    cfg = state.cfg
    interface_cfg = cfg["interface"]
    material_cfg = cfg["material"]

    mask = tags.values == 1
    facet_indices = tags.indices[mask]
    if len(facet_indices) == 0:
        return {"n_facets": 0}

    cell_plus, cell_minus = _get_facet_cell_pairs(msh, facet_indices)

    # Get displacement values at facet quadrature points.
    # For DG1 elements, the jump is the difference of cell-averaged values
    # evaluated at the facet. For a simple diagnostic, use cell DOF averages.
    V = state.V
    bs = V.dofmap.index_map_bs  # = 3
    u_arr = state.u.x.array

    # Each cell has dofs_per_cell DOFs. Average displacement per cell.
    dofmap = V.dofmap
    n_dofs_per_cell = len(dofmap.cell_dofs(0))

    def cell_avg_u(cell_idx):
        """Average displacement vector for a cell."""
        dofs = dofmap.cell_dofs(cell_idx)
        # dofs is array of DOF indices; each DOF has bs components.
        u_cell = np.zeros(3)
        for d in dofs:
            u_cell += u_arr[d * bs : (d + 1) * bs]
        u_cell /= len(dofs)
        return u_cell

    n_facets = len(facet_indices)
    jump_n = np.zeros(n_facets)
    jump_t_mag = np.zeros(n_facets)

    # Facet normals: approximate as z-direction for inter-layer (horizontal layers).
    # More precise: use mesh geometry. For the barrel vault, inter-layer facets
    # have normals roughly in the thickness direction. For simplicity, use the
    # actual facet normal from mesh topology.
    # For diagnostic purposes, just use the jump magnitude decomposition.
    for i in range(n_facets):
        u_plus = cell_avg_u(int(cell_plus[i]))
        u_minus = cell_avg_u(int(cell_minus[i]))
        jump = u_plus - u_minus
        # Approximate normal as unit vector in the direction of the jump
        # For inter-layer facets in a barrel vault, the normal is roughly
        # along the local thickness direction. Use z-component as proxy.
        jump_n[i] = jump[2]  # Normal jump (z-direction for flat layers)
        jump_t_mag[i] = np.sqrt(jump[0]**2 + jump[1]**2)

    # Compute damage from the TSL formula.
    interlayer_data = extract_interlayer_facet_data(state, state.t_val)
    K_n = interlayer_data["K_n"]
    K_t = interlayer_data["K_t"]
    G_Ic_eff = interlayer_data.get("g_i_c_eff", None)
    G_IIc_eff = interlayer_data.get("g_ii_c_eff", None)

    # Reconstruct G_Ic_eff from config if not in interlayer_data.
    beta = interlayer_data["beta_bond"]
    g_i_c = interface_cfg["G_Ic"]
    g_ii_c = interface_cfg["G_IIc"]
    G_Ic_eff = np.maximum(beta * g_i_c, 1e-12)
    G_IIc_eff = np.maximum(beta * g_ii_c, 1e-12)

    # Opening only (positive normal jump).
    delta_n_open = np.maximum(jump_n, 0.0)
    delta_n_sq = delta_n_open ** 2
    delta_t_sq = jump_t_mag ** 2

    damage_arg_n = delta_n_sq / np.maximum(2.0 * G_Ic_eff / np.maximum(K_n, 1e-12), 1e-12)
    damage_arg_t = delta_t_sq / np.maximum(2.0 * G_IIc_eff / np.maximum(K_t, 1e-12), 1e-12)
    damage = 1.0 - np.exp(-(damage_arg_n + damage_arg_t))
    damage = np.clip(damage, 0.0, 1.0)

    # Enforce damage irreversibility from history variable (mirrors weak_form.py line 143).
    if hasattr(state.materials, "damage_max"):
        dmg_hist = state.materials.damage_max.x.array
        dmg_floor = np.maximum(dmg_hist[cell_plus], dmg_hist[cell_minus])
        damage = np.maximum(damage, dmg_floor)

    # Traction magnitude.
    r_s = interface_cfg.get("residual_stiffness_ratio", 0.05)
    T_n = ((1.0 - damage) * K_n + damage * r_s * K_n) * delta_n_open
    T_t = ((1.0 - damage) * K_t + damage * r_s * K_t) * np.sqrt(delta_t_sq)

    return {
        "n_facets": n_facets,
        "jump_n": jump_n,
        "jump_t_mag": jump_t_mag,
        "delta_n_open": delta_n_open,
        "damage": damage,
        "T_n": T_n,
        "T_t": T_t,
        "K_n": K_n,
        "K_t": K_t,
        "K_min": interlayer_data["K_min"],
        "beta_bond": beta,
    }


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Fix 2: interface opening under amplified gravity",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    mesh_overrides = {"n_length": 4, "n_span": 4, "n_thickness": 2}

    tee = None
    if comm.rank == 0:
        tee = _TeeWriter(sys.stdout)
        sys.stdout = tee

    t0 = time.time()

    if comm.rank == 0:
        print("=" * 72)
        print("  FIX 2 VALIDATION: INTERFACE OPENING UNDER AMPLIFIED GRAVITY")
        print("=" * 72)
        print("  Mesh: SMALL (4x4x2 = 32 cells)")
        print(flush=True)

    state = build_diagnostic_state(
        config_path=args.config, mesh_overrides=mesh_overrides, comm=comm,
    )

    bt = state.birth_times_dolfinx
    t_max = float(bt.max())
    t_min = float(bt.min())
    t_midpoint = 0.5 * (t_min + t_max)
    t_all_active = t_max * 1.1

    # Time points to test.
    t_points = [
        t_min + 0.2 * (t_max - t_min),  # Early: ~20% active
        t_midpoint,                        # Mid: ~50% active
        t_all_active,                      # All active
    ]

    g_factors = [1, 10, 50, 100]
    g_base = state.materials.g_val  # Original gravity value

    results = {}

    for g_factor in g_factors:
        g_amplified = g_base * g_factor

        if comm.rank == 0:
            print(f"\n{'='*72}")
            print(f"  Gravity amplification: {g_factor}x  (g = {g_amplified:.0f} mm/s²)")
            print(f"{'='*72}")

        # Override gravity constant (no form rebuild needed).
        state.materials.g_const.value = g_amplified

        factor_results = []

        for t_val in t_points:
            if comm.rank == 0:
                print(f"\n  t = {t_val:.2f} s", end="", flush=True)

            # Reset state and solve.
            prepare_state_at_time(state, t_val, u_mode="zero")

            n_iter, converged, msg = solve_newton(
                state.u, state.V, state.msh, state.F_form, state.J_form,
                t_val=t_val,
                birth_times=state.birth_times_dolfinx,
                cell_to_dofs=state.cell_to_dofs,
                max_iter=50,
                workspace=None,
                debug=False,
            )

            # Max displacement.
            max_disp_local = float(np.max(np.abs(state.u.x.array)))
            max_disp = comm.allreduce(max_disp_local, op=MPI.MAX)

            # Extract interface data.
            iface = _extract_interface_jumps(state)

            entry = {
                "t_val": float(t_val),
                "g_factor": g_factor,
                "converged": bool(converged),
                "n_iter": int(n_iter),
                "message": msg,
                "max_displacement": float(max_disp),
            }

            if iface["n_facets"] > 0:
                entry.update({
                    "max_jump_n": float(np.max(np.abs(iface["jump_n"]))),
                    "max_opening": float(np.max(iface["delta_n_open"])),
                    "max_damage": float(np.max(iface["damage"])),
                    "mean_damage": float(np.mean(iface["damage"])),
                    "n_damaged_facets": int(np.sum(iface["damage"] > 0.01)),
                    "max_T_n": float(np.max(iface["T_n"])),
                    "max_T_t": float(np.max(iface["T_t"])),
                    "K_n_range": [float(np.min(iface["K_n"])), float(np.max(iface["K_n"]))],
                    "K_min_range": [float(np.min(iface["K_min"])), float(np.max(iface["K_min"]))],
                })
            else:
                entry["n_facets"] = 0

            factor_results.append(entry)

            if comm.rank == 0:
                status = "CONV" if converged else "FAIL"
                dmg = entry.get("max_damage", 0)
                opening = entry.get("max_opening", 0)
                print(f"  {status} in {n_iter} iters  "
                      f"|u|_max={max_disp:.2e}  "
                      f"opening={opening:.2e}  "
                      f"damage={dmg:.4f}  "
                      f"n_damaged={entry.get('n_damaged_facets', 0)}")

        results[f"g_{g_factor}x"] = factor_results

    # Restore original gravity.
    state.materials.g_const.value = g_base

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - t0

    if comm.rank == 0:
        print(f"\n{'='*72}")
        print("  VALIDATION SUMMARY")
        print(f"{'='*72}")
        print(f"\n  {'g_factor':>8s}  {'t_val':>8s}  {'status':>6s}  {'iters':>5s}  "
              f"{'|u|_max':>10s}  {'opening':>10s}  {'max_dmg':>8s}  {'n_dmg':>5s}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*5}  "
              f"{'-'*10}  {'-'*10}  {'-'*8}  {'-'*5}")

        all_converged = True
        any_damage = False
        for key in sorted(results.keys()):
            for entry in results[key]:
                status = "CONV" if entry["converged"] else "FAIL"
                if not entry["converged"]:
                    all_converged = False
                dmg = entry.get("max_damage", 0)
                if dmg > 0.01:
                    any_damage = True
                opening = entry.get("max_opening", 0)
                n_dmg = entry.get("n_damaged_facets", 0)
                print(f"  {entry['g_factor']:8d}  {entry['t_val']:8.2f}  {status:>6s}  "
                      f"{entry['n_iter']:5d}  {entry['max_displacement']:10.2e}  "
                      f"{opening:10.2e}  {dmg:8.4f}  {n_dmg:5d}")

        print(f"\n  All converged: {'YES' if all_converged else 'NO'}")
        print(f"  Damage activated: {'YES' if any_damage else 'NO'}")

        # Cohesive zone length note.
        print(f"\n  Note: Small mesh elements ~384mm.")
        print(f"  Cohesive zone l_cz = E*G_Ic/sigma_bond^2:")
        print(f"    Early ages (l_cz >> h): smooth damage expected.")
        print(f"    Late ages (l_cz < h): abrupt single-facet damage (mesh artifact, not instability).")

        print(f"\n  Total time: {elapsed:.1f} s")
        print("=" * 72)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    if comm.rank == 0:
        if tee is not None:
            sys.stdout = tee._original

        output_dir = Path("diagnostics") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_path = output_dir / f"validate_opening_{timestamp}.log"
        log_path.write_text(tee.get_log() if tee else "")

        json_path = output_dir / f"validate_opening_{timestamp}.json"
        json_data = {
            "timestamp": timestamp,
            "elapsed_s": elapsed,
            "g_base": g_base,
            "g_factors": g_factors,
            "results": _results_to_serializable(results),
        }
        json_path.write_text(json.dumps(json_data, indent=2))

        print(f"\n  Output saved to:")
        print(f"    Log:  {log_path}")
        print(f"    JSON: {json_path}")


if __name__ == "__main__":
    main()
