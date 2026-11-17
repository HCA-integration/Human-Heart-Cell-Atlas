#!/usr/bin/env python
"""Decode-row AnnData of one Visium HD slide: one obs per row of the FICTURE 2 um Level-3 decode.

The decode is read in chunks with compact dtypes. Categories follow the factor order. An existing readable
output is kept unless --force is given.

Inputs: the pixel decode table and the Level-3 factor order.
Outputs: --out, an h5ad with
  obs   X_um, Y_um (float32), L3_type (top factor), L3_type_noAtrial (top factor other than atrial
        cardiomyocytes), K1..K5_type (categorical) and P1..P5 (float32 posteriors)
  X     n x 0 (no expression matrix)
  obsm  spatial = [X_um, Y_um]
  uns   description, TYPES_L3, slide

Usage: python build_pixel_h5ad.py --slide SLIDE --decode PIXEL.tsv.gz --factor-order JSON --out OUT.h5ad [--force]
"""

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from hhca.log import get_logger

CHUNKSIZE = 4_000_000
ATRIAL_TYPE = "Atrial Cardiomyocytes"

log = get_logger("pixels")


def load_types(factor_order: Path):
    """Level-3 type names in factor order."""
    types = json.loads(factor_order.read_text(encoding="utf-8"))
    assert isinstance(types, list) and all(isinstance(t, str) for t in types)
    return types


def decode_columns(path: Path):
    """Names of the factor (K1, K2, ...) and posterior (P1, P2, ...) columns of a decode table, by rank."""
    columns = list(pd.read_csv(path, sep="\t", nrows=5).columns)
    factor_columns = [c for c in columns if c.lower().startswith("k") and c[1:].isdigit()]
    posterior_columns = [c for c in columns if c.lower().startswith("p") and c[1:].isdigit()]
    factor_columns.sort(key=lambda c: int(c[1:]))
    posterior_columns.sort(key=lambda c: int(c[1:]))
    return factor_columns, posterior_columns


def read_decode_chunked(path: Path, factor_columns, posterior_columns):
    """Read the decode table in chunks: X, Y float32; K* int8 (factor index); P* float32."""
    dtype = {"X": np.float32, "Y": np.float32}
    for column in factor_columns:
        dtype[column] = np.int16  # parsed as int16, stored as int8
    for column in posterior_columns:
        dtype[column] = np.float32

    xs, ys = [], []
    factors = [[] for _ in factor_columns]
    posteriors = [[] for _ in posterior_columns]
    n_rows = 0
    for chunk in pd.read_csv(
        path, sep="\t", usecols=["X", "Y"] + factor_columns + posterior_columns, dtype=dtype, chunksize=CHUNKSIZE
    ):
        xs.append(chunk["X"].to_numpy(np.float32, copy=True))
        ys.append(chunk["Y"].to_numpy(np.float32, copy=True))
        for i, column in enumerate(factor_columns):
            factors[i].append(chunk[column].to_numpy(np.int16, copy=True).astype(np.int8))
        for i, column in enumerate(posterior_columns):
            posteriors[i].append(chunk[column].to_numpy(np.float32, copy=True))
        n_rows += len(chunk)
        log(f"  read {n_rows} rows")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    K = np.vstack([np.concatenate(parts) for parts in factors]).T  # (n, number of K columns) int8
    P = np.vstack([np.concatenate(parts) for parts in posteriors]).T  # (n, number of P columns) float32
    return x, y, K, P


def is_complete(out: Path, slide: str):
    """Number of obs of an existing output that can be read and has the expected fields, otherwise None."""
    try:
        adata = ad.read_h5ad(out, backed="r")
        n_obs = adata.n_obs
        complete = n_obs > 0 and "L3_type" in adata.obs and "spatial" in adata.obsm
        adata.file.close()
    except (OSError, KeyError, ValueError, TypeError) as error:
        log(f"{slide}: existing output cannot be read ({error}), rebuilding")
        return None
    if complete:
        return n_obs
    log(f"{slide}: existing output is incomplete, rebuilding")
    return None


def top_factor_without_atrial(K, atrial):
    """Per row, the first of its ranked factors that is not the atrial cardiomyocyte factor."""
    top = K[:, 0].astype(np.int64)
    is_atrial = K[:, 0] == atrial
    if is_atrial.any():
        for rank in range(1, K.shape[1]):
            pending = is_atrial & (top == atrial)
            if not pending.any():
                break
            candidate = K[:, rank].astype(np.int64)
            replace = pending & (candidate != atrial)
            top[replace] = candidate[replace]
    return top


def build_slide(slide: str, types, force: bool, decode: Path, out: Path):
    """Write the h5ad of one slide; returns the output path, the number of obs and "built" or "skipped"."""
    if not decode.exists():
        raise FileNotFoundError(f"decode missing for {slide}: {decode}")

    if out.exists() and not force:
        n_obs = is_complete(out, slide)
        if n_obs is not None:
            log(f"{slide}: output exists ({n_obs:,} obs), skip")
            return out, n_obs, "skipped"

    log(f"{slide}: reading header of {decode.name}")
    factor_columns, posterior_columns = decode_columns(decode)
    log(f"{slide}: K cols {factor_columns}  P cols {posterior_columns}")

    x, y, K, P = read_decode_chunked(decode, factor_columns, posterior_columns)
    n_obs = x.shape[0]
    log(f"{slide}: {n_obs:,} rows loaded; K {K.shape}, P {P.shape}")

    if ATRIAL_TYPE not in types:
        raise ValueError(f"factor order missing '{ATRIAL_TYPE}'")
    categories = list(types)

    obs = pd.DataFrame(index=pd.Index([f"px{i}" for i in range(n_obs)], name=None))
    obs["X_um"] = x
    obs["Y_um"] = y
    obs["L3_type"] = pd.Categorical([types[k] for k in K[:, 0].astype(np.int64)], categories=categories)

    log(f"{slide}: top factor without atrial cardiomyocytes")
    without_atrial = top_factor_without_atrial(K, types.index(ATRIAL_TYPE))
    obs["L3_type_noAtrial"] = pd.Categorical([types[k] for k in without_atrial], categories=categories)

    for i in range(K.shape[1]):
        obs[f"K{i + 1}_type"] = pd.Categorical([types[k] for k in K[:, i].astype(np.int64)], categories=categories)
        obs[f"P{i + 1}"] = P[:, i].astype(np.float32)

    log(f"{slide}: building AnnData")
    adata = ad.AnnData(X=np.zeros((n_obs, 0), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = np.c_[x, y].astype(np.float32)
    adata.uns["description"] = (
        f"Full-slide 2um FICTURE-L3 pixels, {slide}. obs: X/Y(um), "
        "L3_type(argmax), L3_type_noAtrial(atrial factor removed, re-argmax), "
        "top-5 factor names K1-5_type + posteriors P1-5. No expression (X is n x 0). "
        "microns frame (CytAssist; um/4.6146 = hires px)."
    )
    adata.uns["TYPES_L3"] = list(types)
    adata.uns["slide"] = slide

    partial = out.with_suffix(".h5ad.tmp")
    log(f"{slide}: writing {out}")
    adata.write_h5ad(partial, compression="gzip")
    os.replace(partial, out)
    log(f"{slide}: wrote {os.path.getsize(out) / 1e6:.1f} MB")
    return out, n_obs, "built"


def check_schema(out: Path, types):
    """Assert the layout of the written h5ad; return the number of types present and the five most frequent."""
    adata = ad.read_h5ad(out)
    assert adata.n_obs > 0, "n_obs == 0"
    assert "L3_type" in adata.obs.columns, "L3_type missing"
    categories = list(adata.obs["L3_type"].cat.categories)
    assert len(categories) > 0, "L3_type categories empty"
    assert categories == list(types), "L3_type categories do not match factor order"
    assert "spatial" in adata.obsm, "obsm spatial missing"
    assert adata.obsm["spatial"].shape == (adata.n_obs, 2), "spatial shape wrong"
    counts = adata.obs["L3_type"].value_counts()
    return int((counts > 0).sum()), counts.head(5)


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide", required=True, help="slide id")
    parser.add_argument("--decode", required=True, help="atlas_L3.decode.prj_12.r_4.0_5.0.pixel.tsv.gz of the slide")
    parser.add_argument("--factor-order", required=True, help="atlas_L3_factor_order.json")
    parser.add_argument("--out", required=True, help="output h5ad")
    parser.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    return parser.parse_args()


def main():
    """Build the decode-row h5ad of one slide and check its layout."""
    args = parse_args()
    types = load_types(Path(args.factor_order))
    log(f"factor order: {len(types)} types")
    out, n_obs, _ = build_slide(args.slide, types, args.force, Path(args.decode), Path(args.out))
    n_used, top = check_schema(out, types)
    size_mb = os.path.getsize(out) / 1e6
    log(f"{args.slide}: n_obs={n_obs:,} types_used={n_used} size={size_mb:.1f}MB\n" + top.to_string())


if __name__ == "__main__":
    main()
