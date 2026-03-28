# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Newton solver utilities with MPI-safe inactive-DOF pinning.

The solver keeps MPI collective safety checks from the original code and uses:
- Jacobian assembly each Newton iteration,
- line search with Armijo-like sufficient decrease,
- explicit pinning of inactive cell DOFs.
"""

import os
import time as _time
import gc
import resource
import sys

import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


class NewtonLinearWorkspace:
    """Persistent PETSc objects for repeated Newton solves."""

    def __init__(self, V, msh, F_form, solver_cfg=None):
        self.comm = msh.comm
        self.msh = msh
        self.solver_cfg = solver_cfg or {}
        self.b = create_vector(F_form)
        self.du = fem.Function(V)
        self.ksp = PETSc.KSP().create(self.comm)
        self.pc = None
        self.linear_solver_mode = self._resolve_linear_solver_mode(
            self.solver_cfg.get("linear_solver", "direct")
        )
        self._configure_linear_solver()
        self.A = None
        # 0: assemble_matrix(form, mat=A), 1: assemble_matrix(A, form), 2: assemble_matrix(form, A)
        self.assemble_mode = 0

    @staticmethod
    def _resolve_linear_solver_mode(mode):
        """Normalize configured linear solver mode."""
        mode_str = str(mode).strip().lower()
        aliases = {
            "direct": "direct",
            "lu": "direct",
            "mumps": "direct",
            "iterative": "iterative",
            "itterative": "iterative",
            "krylov": "iterative",
        }
        if mode_str not in aliases:
            raise ValueError(
                "Unsupported solver.linear_solver="
                f"{mode!r}. Expected one of: direct, iterative."
            )
        return aliases[mode_str]

    def _configure_linear_solver(self):
        """Configure linear solver backend selected in ``solver_cfg``."""
        if self.linear_solver_mode == "direct":
            self._configure_direct_solver()
            return
        self._configure_iterative_solver()

    def _configure_iterative_solver_on_ksp(self, target_ksp):
        """Apply iterative solver config to an external KSP (for subsystem solves)."""
        # Temporarily swap self.ksp, configure, then restore.
        orig_ksp = self.ksp
        orig_pc = self.pc
        self.ksp = target_ksp
        self._configure_iterative_solver()
        self.ksp = orig_ksp
        self.pc = orig_pc

    def ensure_jacobian(self, J_form):
        """Allocate Jacobian once so sparsity is reused across all steps."""
        if self.A is None:
            self.A = assemble_matrix(J_form)
            self.A.assemble()
            self.assemble_mode = 0

    def _configure_direct_solver(self):
        """Configure KSP/PC for direct LU+MUMPS solves.

        Sets MUMPS ICNTL(14) to 200% extra workspace.  By default MUMPS
        uses 20%, which causes repeated malloc/realloc/free cycles during
        numeric factorization as frontal buffers grow.  macOS's allocator
        does not return these freed pages to the OS, producing ~20-30 MB
        RSS growth per ksp.solve() call.  A generous pre-allocation lets
        MUMPS reuse a single contiguous block for all temporary workspace.
        """
        direct_cfg = self.solver_cfg.get("direct", {})
        mumps_icntl_14 = int(
            direct_cfg.get(
                "mumps_icntl_14",
                self.solver_cfg.get("mumps_icntl_14", 200),
            )
        )

        self.ksp.setType("preonly")
        self.pc = self.ksp.getPC()
        self.pc.setType("lu")
        self.pc.setFactorSolverType("mumps")

        # Must be set via options database before the first factorization.
        opts = PETSc.Options()
        opts.setValue("-mat_mumps_icntl_14", str(mumps_icntl_14))
        if direct_cfg.get("out_of_core", False):
            opts.setValue("-mat_mumps_icntl_22", "1")
        for key, val in direct_cfg.get("petsc_options", {}).items():
            opts.setValue(f"-{key}", str(val))
        self.ksp.setFromOptions()

        # Cached reference to avoid creating new Python wrappers per call.
        self._factor_mat = None

    def _configure_iterative_solver(self):
        """Configure KSP/PC for iterative solves on the active-DOF subsystem.

        The default configuration uses ``preonly`` + ``lu``: a direct solve
        on the extracted active-DOF subsystem (typically 1K-22K DOFs instead
        of the full 432K).  On multi-rank, MUMPS is used for distributed LU.

        Alternative PC types (hypre, bjacobi, asm) are supported via config
        for experimentation.
        """
        iterative_cfg = self.solver_cfg.get("iterative", {})
        ksp_type = str(
            iterative_cfg.get(
                "ksp_type",
                self.solver_cfg.get("iterative_ksp_type", "gmres"),
            )
        ).strip().lower()
        pc_type = str(
            iterative_cfg.get(
                "pc_type",
                self.solver_cfg.get("iterative_pc_type", "bjacobi"),
            )
        ).strip().lower()
        sub_pc_type = str(
            iterative_cfg.get("sub_pc_type", "lu")
        ).strip().lower()
        ksp_rtol = float(
            iterative_cfg.get(
                "rtol",
                self.solver_cfg.get("iterative_rtol", 1.0e-8),
            )
        )
        ksp_atol = float(
            iterative_cfg.get(
                "atol",
                self.solver_cfg.get("iterative_atol", 1.0e-12),
            )
        )
        ksp_max_it = int(
            iterative_cfg.get(
                "max_it",
                self.solver_cfg.get("iterative_max_it", 500),
            )
        )
        gmres_restart = int(
            iterative_cfg.get(
                "gmres_restart",
                self.solver_cfg.get("iterative_gmres_restart", 200),
            )
        )

        self.ksp.setType(ksp_type)
        self.pc = self.ksp.getPC()
        self.pc.setType(pc_type)
        self.ksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)
        if ksp_type in ("gmres", "fgmres") and gmres_restart > 0:
            self.ksp.setGMRESRestart(gmres_restart)

        opts = PETSc.Options()
        if pc_type == "lu":
            opts.setValue("-pc_factor_shift_type", "nonzero")
            opts.setValue("-pc_factor_shift_amount", "1e-10")
        elif pc_type == "hypre":
            # BoomerAMG: robust algebraic multigrid for 3D problems.
            opts.setValue("-pc_hypre_type", "boomeramg")
            opts.setValue("-pc_hypre_boomeramg_max_levels", "10")
            opts.setValue("-pc_hypre_boomeramg_strong_threshold", "0.5")
            opts.setValue("-pc_hypre_boomeramg_coarsen_type", "HMIS")
            opts.setValue("-pc_hypre_boomeramg_interp_type", "ext+i")
            opts.setValue("-pc_hypre_boomeramg_agg_nl", "2")
        elif pc_type == "bjacobi":
            # Cell-level block Jacobi: DG DOFs are contiguous per cell,
            # so n_blocks = n_cells gives one 24×24 block per cell.
            sub_pc_type = str(iterative_cfg.get("sub_pc_type", "lu")).strip().lower()
            map_c = self.msh.topology.index_map(self.msh.topology.dim)
            n_cells = map_c.size_local + map_c.num_ghosts
            opts.setValue("-pc_bjacobi_blocks", str(n_cells))
            opts.setValue("-sub_ksp_type", "preonly")
            opts.setValue("-sub_pc_type", sub_pc_type)
            if sub_pc_type == "lu":
                opts.setValue("-sub_pc_factor_shift_type", "nonzero")
                opts.setValue("-sub_pc_factor_shift_amount", "1e-10")
        elif pc_type == "asm":
            sub_pc_type = str(iterative_cfg.get("sub_pc_type", "ilu")).strip().lower()
            ilu_levels = int(iterative_cfg.get("ilu_levels", 0))
            opts.setValue("-sub_ksp_type", "preonly")
            opts.setValue("-sub_pc_type", sub_pc_type)
            if sub_pc_type == "ilu" and ilu_levels >= 0:
                opts.setValue("-sub_pc_factor_levels", str(ilu_levels))
            opts.setValue("-sub_pc_factor_shift_type", "nonzero")
            opts.setValue("-sub_pc_factor_shift_amount", "1e-4")
            opts.setValue("-sub_pc_factor_zeropivot", "1e-6")
            opts.setValue("-sub_pc_factor_mat_ordering_type", "rcm")
            overlap = int(iterative_cfg.get("asm_overlap", 1))
            opts.setValue("-pc_asm_overlap", str(overlap))
        for key, val in iterative_cfg.get("petsc_options", {}).items():
            opts.setValue(f"-{key}", str(val))
        self.ksp.setFromOptions()

        # Cached reference to avoid creating new Python wrappers per call.
        self._factor_mat = None

    def destroy(self):
        """Release PETSc objects held by this workspace."""
        self._factor_mat = None
        try:
            self.pc.reset()
        except Exception:
            pass
        try:
            self.ksp.reset()
        except Exception:
            pass
        if self.A is not None:
            self.A.destroy()
            self.A = None
        self.ksp.destroy()
        self.b.destroy()
        self.du.x.petsc_vec.destroy()


def get_inactive_dofs(t_val, birth_times, cell_to_dofs):
    """Return sorted DOF ids belonging to cells not active at ``t_val``.

    Args:
        t_val: Current physical time.
        birth_times: Cell birth-time array in DOLFINx ordering.
        cell_to_dofs: Cell-to-vector-DOF map.

    Returns:
        np.ndarray: Sorted inactive DOF ids as ``int32``.

    Raises:
        None.
    """
    inactive_cells = np.where(birth_times > t_val)[0]
    if inactive_cells.size == 0:
        return np.array([], dtype=np.int32)
    inactive_dofs = np.unique(cell_to_dofs[inactive_cells].reshape(-1))
    return inactive_dofs.astype(np.int32, copy=False)


def pin_inactive_dofs(A, b, inactive_dofs):
    """Apply identity-row pinning for inactive DOFs.

    Effect:
        - Matrix rows/columns for inactive DOFs are zeroed except diagonal=1.
        - Residual entries at those DOFs are forced to zero.

    Args:
        A: PETSc Jacobian matrix.
        b: PETSc residual vector.
        inactive_dofs: Owned inactive DOF indices on this rank.

    Returns:
        None.

    Raises:
        None.
    """
    # A.zeroRowsLocal is an MPI COLLECTIVE operation. All ranks MUST call it simultaneously,
    # even if they are passing an empty array. Returning early here causes an MPI Deadlock.
    A.zeroRowsColumnsLocal(inactive_dofs, diag=1.0)

    if len(inactive_dofs) > 0:
        b.array[inactive_dofs] = 0.0


def _probe(comm, enabled, barrier, label):
    """Emit optional rank-aware debug probe around collective hotspots.

    Args:
        comm: MPI communicator.
        enabled: Whether probe output is enabled.
        barrier: Whether to enforce a debug barrier after printing.
        label: Probe label text.

    Returns:
        None.

    Raises:
        None.
    """
    if not enabled:
        return
    print(f"[DBG][rank {comm.rank}] {label}", flush=True)
    if barrier:
        comm.Barrier()


def _live_print(msg):
    """Write directly to stdout file descriptor to minimize MPI buffering.

    Args:
        msg: Message to print.

    Returns:
        None.

    Raises:
        None.
    """
    tee_active = os.environ.get("SIM_TEE_ACTIVE") == "1"
    if tee_active:
        # In tee mode, route through Python stdout so rank-0 stream mirroring
        # captures Newton table lines in the log file.
        print(msg, flush=True)
        return

    wrote_via_fd = False
    try:
        os.write(1, f"{msg}\n".encode("utf-8", "replace"))
        wrote_via_fd = True
    except Exception:
        print(msg, flush=True)

    log_path = os.environ.get("SIM_LOG_PATH")
    if not log_path:
        return
    # In tee mode, regular print() is already mirrored; only os.write output
    # needs explicit append. Without tee mode, always append.
    if tee_active and (not wrote_via_fd):
        return
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{msg}\n")
    except Exception:
        pass


def _peak_rss_bytes():
    """Return process peak resident memory in bytes."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS reports bytes.
        return int(ru_maxrss)
    # Linux reports kilobytes.
    return int(ru_maxrss * 1024)


def _current_rss_bytes():
    """Return process current resident memory in bytes when available."""
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                resident_pages = int(f.read().split()[1])
            return int(resident_pages * resource.getpagesize())
        except Exception:
            pass

    # Fallback when current RSS is not directly accessible.
    return _peak_rss_bytes()


def _bytes_to_gib(num_bytes):
    """Convert bytes to GiB."""
    return float(num_bytes) / float(1024**3)


def _mat_info_value(A, key, default=0.0):
    """Safely read selected PETSc Mat info values."""
    if A is None:
        return float(default)

    try:
        info = A.getInfo()
    except Exception:
        return float(default)

    if isinstance(info, dict):
        value = info.get(key, default)
    else:
        value = getattr(info, key, default)

    try:
        return float(value)
    except Exception:
        return float(default)


def _mumps_infog(pc, field, workspace=None):
    """Return a MUMPS INFOG entry when available; otherwise ``None``.

    When *workspace* is provided, the factor matrix Python wrapper is cached
    on ``workspace._factor_mat`` to avoid creating (and leaking) a new
    petsc4py Mat object on every call.
    """
    if pc is None:
        return None
    try:
        if workspace is not None and workspace._factor_mat is not None:
            factor_matrix = workspace._factor_mat
        else:
            factor_matrix = pc.getFactorMatrix()
            if workspace is not None:
                workspace._factor_mat = factor_matrix
    except Exception:
        return None
    if factor_matrix is None:
        return None
    getter = getattr(factor_matrix, "getMumpsInfog", None)
    if getter is None:
        return None
    try:
        return float(getter(int(field)))
    except Exception:
        return None


def solve_newton(
    u,
    V,
    msh,
    F_form,
    J_form,
    t_val,
    birth_times,
    cell_to_dofs,
    max_iter=50,
    rtol=1e-6,
    atol=1e-10,
    debug=True,
    collective_debug=False,
    collective_debug_max_iter=2,
    collective_debug_barrier=False,
    workspace=None,
    memory_tracking=False,
    memory_tracking_every_iter=1,
    memory_tracking_collect_garbage=False,
    memory_tracking_mumps=False,
):
    """Solve the nonlinear system with Newton + backtracking.

    Args:
        u: Displacement unknown updated in-place.
        V: Displacement function space.
        msh: DOLFINx mesh (for communicator).
        F_form: Compiled nonlinear residual form.
        J_form: Compiled nonlinear Jacobian form.
        t_val: Current physical time.
        birth_times: Cell birth-time array.
        cell_to_dofs: Map from cell id to its vector DOF ids.
        max_iter: Maximum Newton iterations.
        rtol: Relative residual tolerance.
        atol: Absolute residual tolerance.
        debug: Print iteration table on rank 0.
        collective_debug: Enable probes around MPI-collective hotspots.
        collective_debug_max_iter: Probe only first ``N`` Newton iterations.
        collective_debug_barrier: Insert MPI barriers for deterministic tracing.
        workspace: Optional persistent ``NewtonLinearWorkspace`` reused across
            time steps. If omitted, a temporary workspace is created and
            destroyed inside this call.
        memory_tracking: Enable Newton-iteration memory diagnostics.
        memory_tracking_every_iter: Emit memory diagnostics every ``N`` iterations.
        memory_tracking_collect_garbage: Run local ``gc.collect()`` before memory
            snapshots (debug only).
        memory_tracking_mumps: Include MUMPS INFOG memory metrics in tracker.

    Returns:
        tuple[int, bool, str]:
            ``(n_iter, converged, message)``.

    Raises:
        None.

    Math:
        Solve F(u)=0 with Newton increments du from J du = F and update
        u <- u - omega*du. The step length omega is reduced (0.5 factors) until
        an Armijo-like residual decrease criterion is met.
    """
    owns_workspace = workspace is None
    if workspace is None:
        workspace = NewtonLinearWorkspace(V, msh, F_form)

    b = workspace.b
    du = workspace.du
    collective_debug_max_iter = max(1, int(collective_debug_max_iter))
    # Armijo sufficient-decrease parameter.
    armijo_c1 = 1.0e-4
    omega_min = 0.02

    # Find the boundary of locally owned DOFs vs Ghost DOFs
    num_owned_dofs = len(b.array)

    # Pre-compute inactive DOFs for this time step
    inactive_dofs_all = get_inactive_dofs(t_val, birth_times, cell_to_dofs)

    # FILTER: Each core is only allowed to pin its OWNED DOFs.
    inactive_dofs = inactive_dofs_all[inactive_dofs_all < num_owned_dofs]

    # Calculate LOCAL counts
    n_inactive_local = len(inactive_dofs)
    n_total_local = num_owned_dofs
    n_active_local = n_total_local - n_inactive_local

    # Calculate GLOBAL counts across all MPI ranks
    n_inactive = msh.comm.allreduce(n_inactive_local, op=MPI.SUM)
    n_total = msh.comm.allreduce(n_total_local, op=MPI.SUM)
    n_active = msh.comm.allreduce(n_active_local, op=MPI.SUM)

    if debug:
        _live_print(
            f"  DOFs: {n_active} active / {n_inactive} pinned / {n_total} total (global)"
        )
        _live_print(
            "  ┌─────┬──────────────┬──────────────┬──────────┬───────┬──────────────┐"
        )
        _live_print(
            "  │ Iter│  ||R||       │  ||R||/||R0|| │  ||du||  │  ω    │  Status      │"
        )
        _live_print(
            "  ├─────┼──────────────┼──────────────┼──────────┼───────┼──────────────┤"
        )

    res_norm_0 = None
    res_norm = np.nan
    rel_res = np.nan

    # Reuse solver and Jacobian structure when workspace is persistent.
    ksp = workspace.ksp
    pc = workspace.pc
    A = workspace.A
    if A is None:
        workspace.ensure_jacobian(J_form)
        A = workspace.A
    assemble_mode = workspace.assemble_mode
    mem_every = max(1, int(memory_tracking_every_iter))
    mem_header_printed = False
    last_rss_max = None

    def _memory_snapshot(iteration, stage):
        """Emit rank-aware memory diagnostics for the current Newton stage."""
        nonlocal mem_header_printed, last_rss_max
        if not memory_tracking:
            return
        if iteration >= 0 and (iteration % mem_every) != 0:
            return

        if memory_tracking_collect_garbage:
            gc.collect()

        rss_local = _current_rss_bytes()
        peak_local = _peak_rss_bytes()
        rss_max = msh.comm.allreduce(rss_local, op=MPI.MAX)
        rss_sum = msh.comm.allreduce(rss_local, op=MPI.SUM)
        peak_max = msh.comm.allreduce(peak_local, op=MPI.MAX)
        rank_rss_pairs = msh.comm.gather((int(rss_local), msh.comm.rank), root=0)

        nz_used_local = _mat_info_value(A, "nz_used", 0.0)

        nz_used = msh.comm.allreduce(nz_used_local, op=MPI.SUM)

        if msh.comm.rank == 0:
            rss_avg = rss_sum / max(msh.comm.size, 1)
            max_pair = max(rank_rss_pairs, key=lambda item: item[0])
            stage_short = {
                "iter-start": "start",
                "post-residual": "residual",
                "post-jacobian": "jacobian",
                "pre-ksp-solve": "solve-pre",
                "post-ksp-solve": "solve-post",
                "post-linesearch": "linesearch",
                "cleanup-pre": "cleanup-pre",
                "cleanup-post": "cleanup-post",
            }.get(stage, stage[:10])
            iter_label = "END" if iteration < 0 else f"{iteration:03d}"
            rss_max_gib = _bytes_to_gib(rss_max)
            rss_avg_gib = _bytes_to_gib(rss_avg)
            peak_max_gib = _bytes_to_gib(peak_max)
            delta_gib = (
                0.0 if last_rss_max is None else _bytes_to_gib(rss_max - last_rss_max)
            )
            last_rss_max = rss_max
            nnz_used_m = nz_used / 1.0e6

            if not mem_header_printed:
                _live_print(
                    "  [MEM]  it   stage       rss_max   d_max   rss_avg   peak_max  nnz(M)  max_rk"
                )
                mem_header_printed = True

            _live_print(
                f"  [MEM]  {iter_label:>3}  {stage_short:<10} "
                f"{rss_max_gib:7.3f}G {delta_gib:+7.3f}G "
                f"{rss_avg_gib:7.3f}G {peak_max_gib:8.3f}G "
                f"{nnz_used_m:7.2f} {max_pair[1]:7d}"
            )

        if memory_tracking_mumps and stage in {
            "post-ksp-solve",
            "cleanup-pre",
            "cleanup-post",
        }:
            m16_local = _mumps_infog(pc, 16, workspace)
            m26_local = _mumps_infog(pc, 26, workspace)
            m29_local = _mumps_infog(pc, 29, workspace)
            m16_max = msh.comm.allreduce(
                float(m16_local) if m16_local is not None else -1.0, op=MPI.MAX
            )
            m26_max = msh.comm.allreduce(
                float(m26_local) if m26_local is not None else -1.0, op=MPI.MAX
            )
            m29_max = msh.comm.allreduce(
                float(m29_local) if m29_local is not None else -1.0, op=MPI.MAX
            )
            if msh.comm.rank == 0 and (m16_max >= 0.0 or m26_max >= 0.0):
                fac_nz_m = (m29_max / 1.0e6) if m29_max >= 0.0 else np.nan
                _live_print(
                    "  [MEM]       mumps:"
                    f" info16={m16_max:8.1f} MB,"
                    f" info26={m26_max:8.1f} MB,"
                    f" info29={fac_nz_m:8.2f} M"
                )

    def _cleanup(destroy_workspace):
        nonlocal A_sub, b_sub, du_sub, active_is, ksp_sub, pc_sub
        _memory_snapshot(-1, "cleanup-pre")
        workspace.A = A
        workspace.assemble_mode = assemble_mode
        # Destroy subsystem objects.
        if A_sub is not None:
            A_sub.destroy()
            A_sub = None
        if b_sub is not None:
            b_sub.destroy()
            b_sub = None
        if du_sub is not None:
            du_sub.destroy()
            du_sub = None
        if active_is is not None:
            active_is.destroy()
            active_is = None
        if ksp_sub is not None:
            ksp_sub.destroy()
            ksp_sub = None
            pc_sub = None
        if destroy_workspace:
            workspace.destroy()
            workspace.A = None
            try:
                PETSc.garbage_cleanup(msh.comm)
            except Exception:
                pass
            gc.collect()
        _memory_snapshot(-1, "cleanup-post")

    # Solver path selection:
    # - 1 rank + iterative: subsystem extraction + LU on active DOFs only
    # - direct: pin inactive DOFs + MUMPS on full system (unchanged)
    is_iterative = workspace.linear_solver_mode == "iterative"
    n_ranks = msh.comm.size
    use_subsystem = is_iterative and n_ranks == 1 and n_active_local > 0
    A_sub = None
    b_sub = None
    du_sub = None
    active_is = None
    active_dofs_local = None
    ksp_sub = None
    pc_sub = None
    first_ksp_its = 0
    pc_lagged = False

    if use_subsystem:
        # Single-rank fast path: extract active-DOF subsystem.
        active_dofs_local = np.setdiff1d(
            np.arange(num_owned_dofs, dtype=np.int32), inactive_dofs
        )
        active_is = PETSc.IS().createGeneral(active_dofs_local, comm=PETSc.COMM_SELF)
        ksp_sub = PETSc.KSP().create(msh.comm)
        workspace._configure_iterative_solver_on_ksp(ksp_sub)
        pc_sub = ksp_sub.getPC()
    else:
        # Direct solver path (or multi-rank — falls back to pin+solve).
        ksp.setOperators(A)

    for iteration in range(max_iter):
        probe_iter = collective_debug and (iteration < collective_debug_max_iter)
        _memory_snapshot(iteration, "iter-start")

        # --- Assemble residual ---
        with b.localForm() as b_local:
            b_local.set(0.0)
        _probe(
            msh.comm,
            probe_iter,
            collective_debug_barrier,
            f"iter {iteration}: before assemble_vector(residual)",
        )
        assemble_vector(b, F_form)
        _probe(
            msh.comm,
            probe_iter,
            collective_debug_barrier,
            f"iter {iteration}: before b.ghostUpdate(residual)",
        )
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        _probe(
            msh.comm,
            probe_iter,
            collective_debug_barrier,
            f"iter {iteration}: after b.ghostUpdate(residual)",
        )

        # Pin inactive DOFs in residual
        if n_inactive > 0:
            b.array[inactive_dofs] = 0.0

        _probe(
            msh.comm,
            probe_iter,
            collective_debug_barrier,
            f"iter {iteration}: before b.norm(residual)",
        )
        res_norm = b.norm()
        _probe(
            msh.comm,
            probe_iter,
            collective_debug_barrier,
            f"iter {iteration}: after b.norm(residual)",
        )
        _memory_snapshot(iteration, "post-residual")

        if not np.isfinite(res_norm):
            if debug:
                _live_print(
                    f"  │ {iteration:3d} │  NaN/Inf     │              │          │       │ ABORT: bad R │"
                )
                _live_print(
                    "  └─────┴──────────────┴──────────────┴──────────┴───────┴──────────────┘"
                )
            _cleanup(owns_workspace)
            return iteration, False, "non-finite residual"

        if iteration == 0:
            res_norm_0 = res_norm if res_norm > 1e-14 else 1.0

        rel_res = res_norm / res_norm_0

        if res_norm < atol or rel_res < rtol:
            if debug:
                _live_print(
                    f"  │ {iteration:3d} │ {res_norm:12.4e} │ {rel_res:12.4e} │    —     │   —   │ CONVERGED    │"
                )
                _live_print(
                    "  └─────┴──────────────┴──────────────┴──────────┴───────┴──────────────┘"
                )
            _cleanup(owns_workspace)
            return iteration, True, "converged"

        # --- Assemble Jacobian ---
        jac_t0 = _time.time()
        _probe(
            msh.comm,
            probe_iter,
            collective_debug_barrier,
            f"iter {iteration}: before assemble_matrix(J)",
        )
        if A is None:
            A = assemble_matrix(J_form)
        else:
            A.zeroEntries()
            assembled = False
            if assemble_mode == 0:
                try:
                    assemble_matrix(J_form, mat=A)
                    assembled = True
                except TypeError:
                    assemble_mode = 1
            if (not assembled) and assemble_mode == 1:
                try:
                    assemble_matrix(A, J_form)
                    assembled = True
                except TypeError:
                    assemble_mode = 2
            if not assembled:
                assemble_matrix(J_form, A)
        _probe(
            msh.comm,
            probe_iter,
            collective_debug_barrier,
            f"iter {iteration}: before A.assemble()",
        )
        A.assemble()
        _probe(
            msh.comm,
            probe_iter,
            collective_debug_barrier,
            f"iter {iteration}: after A.assemble()",
        )

        jac_time_asm = _time.time() - jac_t0

        if is_iterative and active_is is not None:
            # --- Active-DOF subsystem path (iterative solver) ---
            # Extract the active submatrix from the full Jacobian.
            # Only active DOF rows/columns are included; inactive DOFs
            # (identity rows) are eliminated, shrinking the system by 90%+.
            if A_sub is None:
                A_sub = A.createSubMatrix(active_is, active_is)
            else:
                A.createSubMatrix(active_is, active_is, submat=A_sub)
            A_sub.assemble()

            # Extract active portion of residual (safe even if this rank has 0 active DOFs).
            if b_sub is None:
                b_sub = A_sub.createVecRight()
                du_sub = A_sub.createVecRight()
            if len(active_dofs_local) > 0:
                b_sub.array[:] = b.array[active_dofs_local]

            # Set operators and manage PC lagging.
            ksp_sub.setOperators(A_sub)
            if iteration == 0:
                pc_sub.setReusePreconditioner(False)
                pc_lagged = False
            elif iteration == 1 and not pc_lagged:
                pc_sub.setReusePreconditioner(True)
                pc_lagged = True
        else:
            # --- Full-system path (direct solver only) ---
            pin_inactive_dofs(A, b, inactive_dofs)

        jac_time = _time.time() - jac_t0
        _memory_snapshot(iteration, "post-jacobian")

        if debug and iteration == 0:
            _live_print(f"  │  Jacobian: asm={jac_time_asm:.2f}s total={jac_time:.2f}s")

        # --- Solve linear system ---
        du.x.array[:] = 0.0

        _memory_snapshot(iteration, "pre-ksp-solve")
        if is_iterative and active_is is not None:
            du_sub.set(0.0)
            ksp_sub.solve(b_sub, du_sub)
            # Scatter solution back to full du vector.
            if len(active_dofs_local) > 0:
                du.x.array[active_dofs_local] = du_sub.array
            ksp_reason = ksp_sub.getConvergedReason()
        else:
            ksp.solve(b, du.x.petsc_vec)
            ksp_reason = ksp.getConvergedReason()
        _memory_snapshot(iteration, "post-ksp-solve")
        ksp_reason_global = msh.comm.allreduce(int(ksp_reason), op=MPI.MIN)

        if ksp_reason_global < 0:
            # Near-convergence guard: if the nonlinear residual is already
            # substantially reduced, accept the solution despite KSP failure.
            # Threshold: within 1% of initial residual, or 10x rtol.
            ksp_near_tol = max(rtol * 10, 1e-2)
            if rel_res < ksp_near_tol:
                if debug:
                    _live_print(
                        f"  │ {iteration:3d} │ {res_norm:12.4e} │ {rel_res:12.4e} │          │   —   │ CONV(KSP-nr) │"
                    )
                    _live_print(
                        "  └─────┴──────────────┴──────────────┴──────────┴───────┴──────────────┘"
                    )
                _cleanup(owns_workspace)
                return iteration, True, "converged (KSP-near)"
            if debug:
                _live_print(
                    f"  │ {iteration:3d} │ {res_norm:12.4e} │ {rel_res:12.4e} │          │   —   │ ABORT: KSP {ksp_reason_global:2d} │"
                )
                _live_print(
                    "  └─────┴──────────────┴──────────────┴──────────┴───────┴──────────────┘"
                )
            _cleanup(owns_workspace)
            return iteration, False, f"KSP diverged ({ksp_reason_global})"

        # After KSP writes owned entries, synchronize to ghost DOFs before any
        # norm checks/line-search evaluations that read the full local vector.
        du.x.scatter_forward()

        du_bad_local = int(not np.all(np.isfinite(du.x.array)))
        du_bad_any = msh.comm.allreduce(du_bad_local, op=MPI.MAX)
        if du_bad_any:
            if debug:
                _live_print(
                    f"  │ {iteration:3d} │ {res_norm:12.4e} │ {rel_res:12.4e} │ NaN/Inf  │   —   │ ABORT: bad du│"
                )
                _live_print(
                    "  └─────┴──────────────┴──────────────┴──────────┴───────┴──────────────┘"
                )
            _cleanup(owns_workspace)
            return iteration, False, "non-finite linear solution"

        du_norm = np.linalg.norm(du.x.array)

        # Newton stagnation guard: if KSP returns a (near-)zero update, the
        # residual is at the solver noise floor.  Declare converged rather
        # than looping with zero progress.
        u_norm = np.linalg.norm(u.x.array)
        du_rel = du_norm / max(u_norm, 1.0)
        if du_rel < 1e-14:
            if debug:
                _live_print(
                    f"  │ {iteration:3d} │ {res_norm:12.4e} │ {rel_res:12.4e} │ {du_norm:8.2e} │   —   │ CONV(stag)   │"
                )
                _live_print(
                    "  └─────┴──────────────┴──────────────┴──────────┴───────┴──────────────┘"
                )
            _cleanup(owns_workspace)
            return iteration, True, "converged (stagnation)"

        # Track linear solver iteration count for iterative KSP types.
        _ksp_ref = ksp_sub if (is_iterative and ksp_sub is not None) else ksp
        ksp_its = _ksp_ref.getIterationNumber()

        # Record first-iteration KSP count for PC lagging safety valve.
        if iteration == 0:
            first_ksp_its = max(ksp_its, 1)
        # Safety valve: if lagged PC degrades badly, refresh it.
        if is_iterative and pc_lagged and pc_sub is not None and ksp_its > 3 * first_ksp_its:
            pc_sub.setReusePreconditioner(False)
            pc_lagged = False

        # --- Backtracking line search ---
        # Trial updates use u <- u - omega*du with omega in (omega_min, 1].
        omega = 1.0
        prev_omega = 0.0
        new_res = np.inf

        for ls_iter in range(10):
            step_scale = omega - prev_omega
            u.x.array[:] -= step_scale * du.x.array
            # Line-search trial state must be ghost-consistent before residual assembly.
            u.x.scatter_forward()

            # Re-evaluate residual at trial point
            with b.localForm() as b_local:
                b_local.set(0.0)
            _probe(
                msh.comm,
                probe_iter,
                collective_debug_barrier,
                f"iter {iteration}: ls {ls_iter} before assemble_vector(trial)",
            )
            assemble_vector(b, F_form)
            _probe(
                msh.comm,
                probe_iter,
                collective_debug_barrier,
                f"iter {iteration}: ls {ls_iter} before b.ghostUpdate(trial)",
            )
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            _probe(
                msh.comm,
                probe_iter,
                collective_debug_barrier,
                f"iter {iteration}: ls {ls_iter} after b.ghostUpdate(trial)",
            )

            # Pin inactive in trial residual too
            if n_inactive > 0:
                b.array[inactive_dofs] = 0.0

            _probe(
                msh.comm,
                probe_iter,
                collective_debug_barrier,
                f"iter {iteration}: ls {ls_iter} before b.norm(trial)",
            )
            new_res = b.norm()
            _probe(
                msh.comm,
                probe_iter,
                collective_debug_barrier,
                f"iter {iteration}: ls {ls_iter} after b.norm(trial)",
            )

            if not np.isfinite(new_res):
                prev_omega = omega
                omega *= 0.5
                continue

            armijo_target = (1.0 - armijo_c1 * omega) * res_norm
            if new_res <= armijo_target or omega <= omega_min:
                break
            prev_omega = omega
            omega *= 0.5

        ls_iters = ls_iter + 1
        _memory_snapshot(iteration, "post-linesearch")

        if debug:
            status = f"ω={omega:.3f} ls={ls_iters}"
            if ksp_its > 1:
                status += f" ki={ksp_its}"
            if ksp_reason < 0:
                status += f" KSP:{ksp_reason}"
            _live_print(
                f"  │ {iteration:3d} │ {res_norm:12.4e} │ {rel_res:12.4e} │ {du_norm:8.2e} │ {omega:5.3f} │ {status:<12s} │"
            )

            if iteration > 0 and iteration % 10 == 0:
                u_max = np.max(np.abs(u.x.array))
                _live_print(f"  │      max|u|={u_max:.4e}, new_res={new_res:.4e}")

    if debug:
        _live_print(
            f"  │ MAX │ {res_norm:12.4e} │ {rel_res:12.4e} │          │       │ NOT CONVERGED│"
        )
        _live_print(
            "  └─────┴──────────────┴──────────────┴──────────┴───────┴──────────────┘"
        )
        _live_print(
            f"  STALL: ||R||={res_norm:.4e}, ||R||/||R0||={rel_res:.4e}, max|u|={np.max(np.abs(u.x.array)):.4e}"
        )

    _cleanup(owns_workspace)
    return max_iter, False, "max iterations"