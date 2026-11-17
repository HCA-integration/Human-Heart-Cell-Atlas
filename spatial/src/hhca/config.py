"""Reading the pipeline configuration files in ``config/``.

Every config has a ``roots`` block with the data folders; path templates elsewhere in the file refer to them
as ``{root_name}``. A root can be overridden for one run with the environment variable ``HHCA_<ROOT_NAME>``.
"""

from __future__ import annotations

import os

import yaml

from hhca.paths import config_path


def load_config(config) -> dict:
    """Parse a config file given by path, or by pipeline name (``"vascular"`` for ``config/vascular.yaml``)."""
    config = os.fspath(config)
    path = config if config.endswith((".yaml", ".yml")) else config_path(config)
    with open(path) as handle:
        return yaml.safe_load(handle)


def resolve_roots(cfg: dict) -> dict[str, str]:
    """Data roots of a config with the ``HHCA_<ROOT>`` overrides applied and ``~`` expanded."""
    return {
        name: os.path.expanduser(os.environ.get(f"HHCA_{name.upper()}", path)) for name, path in cfg["roots"].items()
    }


def fill(template, roots: dict[str, str], **values) -> str:
    """Fill the ``{root}`` names and any further ``{placeholders}`` of a path template."""
    return os.path.expanduser(str(template).format(**roots, **values))


def get(cfg: dict, dotted_key: str):
    """Value of a dotted key such as ``substrate.decode``."""
    value = cfg
    for key in dotted_key.split("."):
        value = value[key]
    return value


def section_path(cfg: dict, key: str, slide: str | None = None) -> str:
    """Path of `key` in a pipeline config, for one slide if the template has a ``{slide}`` (or ``{s}``) placeholder.

    `key` is looked up in the ``substrate`` block (outputs of pipeline ``substrate``, relative to the repository
    root), then in the ``paths`` block, then at the top level. ``{base}`` in a template is the slide id without
    its ``VS_`` prefix. Slides listed under ``slides.ah`` read ``<key>_ah`` when the config has it next to `key`.
    """
    roots = resolve_roots(cfg)
    values = {}
    if slide is not None:
        values = {"slide": slide, "s": slide, "base": slide[3:] if slide.startswith("VS_") else slide}
    block = next((cfg[name] for name in ("substrate", "paths") if key in cfg.get(name, {})), cfg)
    if f"{key}_ah" in block and slide in cfg.get("slides", {}).get("ah", []):
        key = f"{key}_ah"
    return fill(block[key], roots, **values)
