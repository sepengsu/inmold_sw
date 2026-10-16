"""YAML loading and strict, side-effect-free preprocessing configuration."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ._validation import integer
from .clahe import validate_clahe
from .tiling import validate_tiling

Config = dict[str, Any]


def _section(config: Mapping[str, Any], name: str, defaults: Config) -> Config:
    """Merge one section while rejecting misspelled keys and invalid mappings."""
    values = config.get(name, {})
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown {name} option(s): {', '.join(map(str, unknown))}")
    return {**defaults, **values}


def validate_config(config: Mapping[str, Any], *, base_dir: Path | None = None) -> Config:
    """Return a normalized copy; relative output paths use base_dir or the cwd."""
    if not isinstance(config, Mapping):
        raise ValueError("Preprocess configuration must be a mapping")
    unknown = config.keys() - {"roi", "auto_roi", "clahe", "tiling", "output"}
    if unknown:
        raise ValueError(f"Unknown configuration section(s): {', '.join(map(str, unknown))}")
    if "roi" not in config:
        raise ValueError("A roi section with x, y, width, height is required")
    roi = _section(config, "roi", {"x": None, "y": None, "width": None, "height": None})
    roi = {key: integer(value, f"roi.{key}", 0 if key in ("x", "y") else 1)
           for key, value in roi.items()}
    auto_roi = _section(config, "auto_roi", {"enabled": False, "boundary_config": None})
    if not isinstance(auto_roi["enabled"], bool):
        raise ValueError("auto_roi.enabled must be a boolean")
    boundary_path = auto_roi["boundary_config"]
    if boundary_path is not None:
        if not isinstance(boundary_path, (str, Path)) or not str(boundary_path).strip():
            raise ValueError("auto_roi.boundary_config must be a non-empty path or null")
        boundary_path = Path(boundary_path).expanduser()
        if not boundary_path.is_absolute():
            boundary_path = (base_dir if base_dir is not None else Path.cwd()) / boundary_path
        auto_roi["boundary_config"] = boundary_path.resolve()
    clahe = _section(config, "clahe", {
        "enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8],
    })
    if not isinstance(clahe["enabled"], bool):
        raise ValueError("clahe.enabled must be a boolean")
    clahe["clip_limit"], clahe["tile_grid_size"] = validate_clahe(
        clahe["clip_limit"], clahe["tile_grid_size"],
    )
    tiling = _section(config, "tiling", {"patch_size": 256, "stride": 128})
    tiling["patch_size"], tiling["stride"] = validate_tiling(**tiling)
    output = _section(config, "output", {
        "root_dir": "data", "save_roi": True, "save_clahe": True,
        "save_tiles": False, "save_metadata": True,
    })
    for key in ("save_roi", "save_clahe", "save_tiles", "save_metadata"):
        if not isinstance(output[key], bool):
            raise ValueError(f"output.{key} must be a boolean")
    root = output["root_dir"]
    if not isinstance(root, (str, Path)) or not str(root).strip():
        raise ValueError("output.root_dir must be a non-empty path")
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (base_dir if base_dir is not None else Path.cwd()) / root
    output["root_dir"] = root.resolve()
    return {"roi": roi, "auto_roi": auto_roi, "clahe": clahe, "tiling": tiling, "output": output}


def load_config(path: str | Path) -> Config:
    """Load YAML; root_dir is resolved relative to the YAML file's directory."""
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8-sig") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    return validate_config(config, base_dir=path.parent)
