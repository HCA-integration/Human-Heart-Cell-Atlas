"""Locations inside the repository: configuration files and pipeline outputs."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def results_dir(*parts: str) -> str:
    """Path under the results folder: ``results/`` in the repository root unless ``HHCA_RESULTS`` is set."""
    root = os.environ.get("HHCA_RESULTS", str(REPO / "results"))
    return os.path.join(root, *parts)


def config_path(name: str) -> str:
    """Path of ``config/<name>.yaml``."""
    return str(REPO / "config" / f"{name}.yaml")
