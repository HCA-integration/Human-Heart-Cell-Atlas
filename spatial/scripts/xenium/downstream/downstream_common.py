"""Shared helpers of the Xenium downstream scripts: config access, output folders and the rejection cohort as AnnData.

The analysis steps themselves are in hhca.tl (src/hhca/tl/_xenium.py, _rejection.py); the scripts of this folder
load the cohort, call them and write the tables.

Usage: imported by the scripts in this folder (`import downstream_common as common`).
"""

from __future__ import annotations

import os

import pandas as pd

import hhca.config
import hhca.io
import hhca.tl
from hhca import REPO


def load_config() -> dict:
    """Settings of config/xenium_downstream.yaml."""
    return hhca.config.load_config("xenium_downstream")


def repo_path(path: str) -> str:
    """Relative config paths are relative to the repository root."""
    return path if os.path.isabs(path) else str(REPO / path)


def out_dir(cfg: dict, *sub: str) -> str:
    """Output sub-folder under config outputs.root (created if missing)."""
    folder = os.path.join(repo_path(cfg["outputs"]["root"]), *sub)
    os.makedirs(folder, exist_ok=True)
    return folder


def extracted(cfg: dict) -> str:
    """Extraction folder of the GEO object (GSE290577_heart_spatial.rds, scripts/xenium/extract_mtx.R)."""
    return repo_path(cfg["inputs"]["extracted"])


def read_lines(path: str) -> list[str]:
    """Lines of a text file (genes.txt, barcodes.txt) without surrounding whitespace."""
    with open(path) as handle:
        return [line.strip() for line in handle]


def read_meta(meta_path: str) -> pd.DataFrame:
    """GEO clinical metadata per core, with the derived columns used downstream (rej, grade, pamr, bxid, ...)."""
    return hhca.tl.clinical_metadata(pd.read_csv(meta_path, sep="\t"))


def label_paths(cfg: dict) -> dict:
    """Label tables written by label_tables.py."""
    folder = out_dir(cfg, "labels")
    return {"labels": f"{folder}/xenium_atlas_labels.csv", "lymphoid": f"{folder}/xenium_lymphoid8_annotation.csv"}


def add_label_arguments(parser) -> None:
    """The arguments that locate the cohort: label tables and extraction folder."""
    parser.add_argument("--labels", help="per-cell label table (default: labels/xenium_atlas_labels.csv)")
    parser.add_argument("--lymphoid", help="lymphoid Level-4 table (default: labels/xenium_lymphoid8_annotation.csv)")
    parser.add_argument("--extracted", help="extraction folder (default: config inputs.extracted)")


def load_cohort(cfg: dict, args, counts: bool = False):
    """Cells of the rejection cohort as AnnData with the labels of label_tables.py in obs.

    obs: author metadata, atlas_L3, atlas_immune and label8 (lymphoid Level 4); obsm["spatial"]: centroids;
    X: raw counts when `counts` is set.
    """
    paths = label_paths(cfg)
    adata = hhca.io.read_xenium_cohort(args.extracted or extracted(cfg), counts=counts)
    labels = pd.read_csv(args.labels or paths["labels"], usecols=["cell_id", "atlas_L3", "atlas_immune"])
    labels = labels.set_index("cell_id").reindex(adata.obs_names)
    level4 = pd.read_csv(args.lymphoid or paths["lymphoid"], usecols=["cell_id", "label8"])
    level4 = level4.set_index("cell_id").reindex(adata.obs_names)
    adata.obs["atlas_L3"] = labels["atlas_L3"].to_numpy()
    adata.obs["atlas_immune"] = labels["atlas_immune"].astype(bool).to_numpy()
    adata.obs["label8"] = level4["label8"].to_numpy()
    return adata


def classify(adata, cfg: dict, classes: str, key_added: str = "ct") -> None:
    """Add the cell classes of one analysis (config classes.<classes>): lymphoid Level 4 + Level-3 classes."""
    labels = cfg["labels"]
    hhca.tl.cell_classes(
        adata, cfg["classes"][classes], labels["lymphoid"], exclude=labels.get("exclude_l3", []), key_added=key_added
    )
