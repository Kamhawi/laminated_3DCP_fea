# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Configuration loading and output/checkpoint path helpers.

This module provides:
- YAML configuration loading,
- recursive numeric-string normalization,
- output/checkpoint file path construction for simulation writers.
"""

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union

import yaml


def _coerce_numeric_strings(obj: Any) -> Any:
    """Recursively convert numeric-like strings to native numeric types.

    Args:
        obj: Arbitrary nested Python object (dict/list/scalar).

    Returns:
        Any: Object with numeric-looking strings converted to ``int`` or
        ``float`` when possible.

    Raises:
        None.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numeric_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric_strings(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        try:
            if re.fullmatch(r"[+-]?\d+", s):
                return int(s)
            return float(s)
        except ValueError:
            return obj
    return obj


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load YAML config and normalize numeric strings.

    Args:
        path: Optional explicit config path. If omitted, loads
            `config/config.yaml` in the same package directory.

    Returns:
        Dict[str, Any]: Parsed and normalized configuration dictionary.

    Raises:
        OSError: If the configuration file cannot be opened.
        yaml.YAMLError: If YAML parsing fails.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _coerce_numeric_strings(raw)


def build_run_tag() -> str:
    """Build a compact timestamp tag for run-associated filenames.

    Returns:
        str: Timestamp tag in ``YYMMDD_HHMMSS`` format.
    """
    return datetime.now().strftime("%y%m%d_%H%M%S")


def _resolve_relative_to(base_dir: Path, configured_path: Union[str, Path]) -> Path:
    """Resolve a configured path; keep absolute paths unchanged."""
    p = Path(configured_path)
    return p if p.is_absolute() else (base_dir / p)


def build_output_paths(
    cfg: Dict[str, Any],
    run_tag: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Create run-scoped output/log paths under the configured output directory.

    Args:
        cfg: Full configuration dictionary containing an ``output`` section.
        run_tag: Optional run tag used in the run-folder name.

    Returns:
        Tuple[Path, Path, Path]:
            ``(disp_path, cell_path, log_path)`` output file paths.

    Raises:
        OSError: If directory creation fails.
    """
    output_cfg = cfg.get("output", {})
    tag = run_tag or build_run_tag()
    out_dir = Path(output_cfg.get("directory", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = out_dir / f"run_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep user overrides for filenames, but force them to stay inside run_dir.
    disp_name = Path(str(output_cfg.get("displacement_file") or "disp.bp")).name
    cell_name = Path(str(output_cfg.get("cell_data_file") or "cell_data.bp")).name
    log_name = Path(str(output_cfg.get("log_file") or "sim_run.log")).name

    disp_path = run_dir / disp_name
    cell_path = run_dir / cell_name
    log_path = run_dir / log_name
    return disp_path, cell_path, log_path


def save_run_config_snapshot(
    cfg: Dict[str, Any],
    run_dir: Union[str, Path],
    filename: str = "settings_used.yaml",
) -> Path:
    """Write the effective configuration used by this run into ``run_dir``.

    Args:
        cfg: Effective configuration dictionary (after default merging).
        run_dir: Run-specific output directory.
        filename: Snapshot filename placed in ``run_dir``.

    Returns:
        Path: Absolute/relative path written for the settings snapshot.
    """
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir_path / filename
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return snapshot_path


def get_checkpoint_dir(cfg: Dict[str, Any]) -> Path:
    """Resolve and create the checkpoint directory.

    The directory is taken from ``checkpoint.directory`` and defaults to
    ``"checkpoints"``. Relative paths are resolved against the project root.
    """
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_cfg = cfg.get("checkpoint", {})
    checkpoint_dir_cfg = checkpoint_cfg.get("directory", "checkpoints")
    checkpoint_dir = _resolve_relative_to(project_root, checkpoint_dir_cfg)
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass  # Race condition with MPI ranks or macOS xattr quirk
    return checkpoint_dir
