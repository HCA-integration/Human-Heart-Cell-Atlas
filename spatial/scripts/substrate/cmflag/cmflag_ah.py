#!/usr/bin/env python
"""CMFLAG for the three AH Visium HD slides (VS_AH9_LV_FW_epi, VS_AH10_LV_mid_endo, VS_AH10_LV_mid_epi).

Same algorithm as cmflag_tif.py. Everything that does not depend on the H&E source comes from hhca.celltyping:
marker densities and depth into whitespace (marker_depth), the spillover rule (spillover_flags, share 0.80)
with the same guards per type, the haematoxylin nucleus gate (fraction 0.35) and the lumen of large vessels
(vessel_lumen). The microscopy path uses its registration (register_he), offset field and per-bin sampling
(Microscopy, sample_he). Thresholds and marker lists: config/substrate_cmflag.yaml.

What differs for the AH slides:
  * Bins and assignment: one row per decoded 2 um bin of the 2 um h5ad, assign = L3_label (top factor; the AH
    slides have no certainty score), recorded as "argmax_L3_h5ad".
  * Bins -> Space Ranger hires H&E: pure scale, offset 0, pixel = um / CMPP. The AH 2 um matrices come from
    feature_slice.h5 and have no tissue_hires_scalef, so hs = (hires max dim) / (microscope max dim) and
    CMPP = microns_per_pixel / hs; the microscope dimensions come from the CZI mosaic box (AH10) or from the
    AH9 OME-TIFF header (--micro-dims). The hires aspect ratio must equal that of the microscope image.
  * --he-source czi (AH10): the Zeiss CZI is rendered at an effective 0.7 um/px (mosaic read, channel order
    matched to the hires H&E) and white-referenced per channel to its glass (raw CZI glass is ~184 grey; the
    Berlin TIFFs have glass at ~250-255); then the microscopy path of cmflag_tif.py with its fixed cuts
    (grey > 235 or eosin OD < 0.008, lumen grey > 220).
  * --he-source hires (AH9): the full microscopy image is not available (truncated OME-TIFF), so the Space
    Ranger hires H&E (microscope image downscaled to 6000 px, ~2 um/px) is sampled per bin. Its canvas is
    grey (~144), so the cuts adapt to the image: grey cut = tissue + 0.75 (background - tissue), eosin cut =
    0.35 x median tissue eosin OD, lumen cut = tissue + 0.55 (background - tissue), with background and tissue
    the median grey outside and inside the tissue mask of the hires image. Bins outside the hires image have
    no H&E and are never whitespace.

Inputs: the 2 um h5ad, Space Ranger outs (binned_outputs/square_002um), hires H&E, CZI (AH10), factor order,
certainty quantiles, per-type thresholds, decay fits, vessel tables.
Outputs: <outdir>/<slide>_clean_bins.parquet (flags per bin), <slide>_counts.json and, for --he-source czi, the
registration (--reg-cache) and the CZI rendering (--render-dir).

Usage: python cmflag_ah.py SLIDE --he-source {czi,hires} --h5ad SLIDE.h5ad --spaceranger OUTS [--hires PNG]
           [--czi HE.czi] [--micro-dims W,H] --factor-order JSON --quantiles JSON --tau JSON --cm-fit JSON
           --vsmc-fit JSON [--structs CSV --cells PARQUET] --reg-cache NPZ [--render-dir DIR] --outdir DIR
           [--force] [--force-reg]
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.ndimage import map_coordinates
from skimage.color import rgb2hed

import hires_mask
from hhca import celltyping, thread_env
from hhca.celltyping import local_density, luminance
from hhca.config import load_config
from hhca.grid import SPILLOVER_FLAGS, bin_key
from hhca.log import get_logger

log = get_logger("cmflag")

# four threads for the libraries loaded from here on (numba, with scanpy) unless the caller sets these
os.environ.update(thread_env(4, keep_existing=True))

Image.MAX_IMAGE_PIXELS = None

# cuts of the hires path, as fractions of the way from tissue grey to background grey
GREY_TO_BG_FRAC = 0.75
EOD_FRAC = 0.35
LUMEN_TO_BG_FRAC = 0.55
CZI_TARGET_MPP = 0.70


def parse_args(argv=None):
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slide", help="slide id")
    parser.add_argument(
        "--he-source", choices=["czi", "hires"], required=True, help="H&E image the bins are sampled from"
    )
    parser.add_argument("--h5ad", required=True, help="2 um h5ad of the slide (obs: L3_label, x_um, y_um)")
    parser.add_argument("--spaceranger", required=True, help="<slide>/outs with binned_outputs/square_002um")
    parser.add_argument(
        "--hires", default="", help="hires H&E png (default: square_002um/spatial/tissue_hires_image.png)"
    )
    parser.add_argument("--czi", default="", help="Zeiss CZI microscopy H&E (--he-source czi)")
    parser.add_argument(
        "--micro-dims", default="", help="W,H of the microscope (full-resolution) image; read from the CZI if not given"
    )
    parser.add_argument("--factor-order", required=True, help="atlas_L3_factor_order.json")
    parser.add_argument("--quantiles", required=True, help="normalization_quantiles.json of the certainty score")
    parser.add_argument("--tau", required=True, help="pertype_tau_recommended.json")
    parser.add_argument(
        "--cm-fit", required=True, help="decay fit of cardiomyocyte markers into whitespace (A, lam, b)"
    )
    parser.add_argument("--vsmc-fit", required=True, help="decay fit of VSMC markers into whitespace (A, lam, b)")
    parser.add_argument("--structs", default="", help="vessel structures <slide>_structs.csv (for the lumen)")
    parser.add_argument("--cells", default="", help="member cells of the vessel structures <slide>_cells.parquet")
    parser.add_argument(
        "--reg-cache", required=True, help="registration .npz of the CZI path (read if present, else written)"
    )
    parser.add_argument("--render-dir", default="", help="folder for the 0.7 um/px CZI rendering (default $TMPDIR)")
    parser.add_argument("--outdir", required=True, help="folder for <slide>_clean_bins.parquet and <slide>_counts.json")
    parser.add_argument("--force", action="store_true", help="overwrite existing flags")
    parser.add_argument("--force-reg", action="store_true", help="recompute the registration")
    return parser.parse_args(argv)


def read_json(path):
    """Content of a JSON file."""
    with open(path) as handle:
        return json.load(handle)


def load_bins(h5ad, order):
    """x, y (um) and Level-3 type index of every bin of the 2 um h5ad (-1 for labels outside the factor order)."""
    import anndata as ad

    adata = ad.read_h5ad(h5ad, backed="r")
    obs = adata.obs[["L3_label", "x_um", "y_um"]].copy()
    adata.file.close()
    lab = obs["L3_label"]
    if not isinstance(lab.dtype, pd.CategoricalDtype):
        lab = lab.astype("category")
    index = {name: i for i, name in enumerate(order)}
    lut = np.array([index.get(str(c), -1) for c in lab.cat.categories] + [-1], np.int64)
    codes = lab.cat.codes.to_numpy()
    assign = lut[np.where(codes >= 0, codes, len(lut) - 1)]
    unknown = sorted(set(map(str, lab.cat.categories)) - set(order))
    X = obs["x_um"].to_numpy(np.float64)
    Y = obs["y_um"].to_numpy(np.float64)
    log(
        f"bins from h5ad n={len(X):,} unassigned%={100 * (assign < 0).mean():.3f} labels not in factor order: {unknown}"
    )
    return X, Y, assign


def czi_dims(czi_path):
    """Width and height of the CZI mosaic in pixels."""
    from aicspylibczi import CziFile

    bb = CziFile(czi_path).get_mosaic_bounding_box()
    return int(bb.w), int(bb.h)


def hires_scale(hires_wh, micro_wh, mpp):
    """Scale of the hires image relative to the microscope image, and its um per pixel."""
    hs = max(hires_wh) / max(micro_wh)
    ar_h = hires_wh[0] / hires_wh[1]
    ar_m = micro_wh[0] / micro_wh[1]
    rel = abs(ar_h / ar_m - 1)
    log(
        f"hires {hires_wh} vs microscope {micro_wh}: aspect {ar_h:.5f} vs {ar_m:.5f} (rel diff {rel:.2e}); "
        f"hs={hs:.6f} CMPP={mpp / hs:.4f}"
    )
    if rel > 2e-3:
        raise SystemExit("hires aspect ratio does not match the microscope image: hires is not a downscale of it")
    return hs, mpp / hs, rel


def read_czi_rgb(czi, scale):
    """Read the CZI mosaic at `scale` as an RGB uint8 array, in the channel order of the file."""
    try:
        nC = czi.get_dims_shape()[0].get("C", [0, 1])[1]
    except (IndexError, KeyError, TypeError, RuntimeError):  # channel dimension not readable: one (RGB) channel
        nC = 1
    log(f"CZI nC={nC}; reading mosaic at scale {scale:.5f}")

    def read_ch(c):
        return np.squeeze(np.asarray(czi.read_mosaic(scale_factor=scale, C=c)))

    if nC >= 3:
        rgb = np.stack([read_ch(c) for c in range(3)], axis=-1)
    else:
        m = read_ch(0)
        if m.ndim == 3 and m.shape[-1] in (3, 4):
            rgb = m[..., :3]
        elif m.ndim == 3 and m.shape[0] in (3, 4):
            rgb = np.moveaxis(m[:3], 0, -1)
        else:
            rgb = np.repeat(m[..., None], 3, axis=-1)
    if rgb.dtype != np.uint8:
        mx = float(rgb.max()) or 1.0
        rgb = (np.clip(rgb / mx, 0, 1) * 255).astype(np.uint8)
    return rgb


def red_minus_blue(a):
    """Mean red minus mean blue over the tissue of a thumbnail of the image."""
    a = np.asarray(a[:: max(1, a.shape[0] // 600), :: max(1, a.shape[1] // 600), :3], np.float32)
    tis = a.mean(2) < 220
    return float(a[..., 0][tis].mean() - a[..., 2][tis].mean())


def white_reference(rgb):
    """Scale each channel of `rgb` in place so that the median glass pixel becomes 255.

    Raw CZI intensities are not white-balanced (glass ~184 grey, ~250 in the Berlin TIFFs), and the fixed cuts
    and optical densities assume I/I0 with I0 = glass. Glass pixels have low saturation and are not the black
    canvas outside the scan. Returns the glass intensity and gain per channel and the number of glass pixels.
    """
    small = np.asarray(rgb[::16, ::16, :3], np.float32)
    glass = ((small.max(2) - small.min(2)) < 12) & (small.min(2) > 60)
    I0 = np.array([float(np.median(small[..., c][glass])) for c in range(3)])
    gain = 255.0 / I0
    for r0 in range(0, rgb.shape[0], 2048):
        blk = rgb[r0 : r0 + 2048].astype(np.float32) * gain
        rgb[r0 : r0 + 2048] = np.clip(np.round(blk), 0, 255).astype(np.uint8)
    log(
        f"white reference: glass I0 (R,G,B) = {np.round(I0, 1).tolist()} from {int(glass.sum()):,} px "
        f"-> gains {np.round(gain, 4).tolist()}"
    )
    return I0, gain, int(glass.sum())


def render_czi(czi_path, out_tif, native_mpp, hires_rgb, target_mpp=CZI_TARGET_MPP):
    """Render the CZI mosaic as an RGB uint8 TIFF at about target_mpp, white-referenced to its glass."""
    from aicspylibczi import CziFile

    czi = CziFile(czi_path)
    bb = czi.get_mosaic_bounding_box()
    raw_W, raw_H = int(bb.w), int(bb.h)
    scale = native_mpp / target_mpp
    log(f"CZI raw {raw_W}x{raw_H}")
    rgb = read_czi_rgb(czi, scale)
    # Zeiss brightfield is stored BGR; the channel order is matched to the hires H&E, which Space Ranger
    # wrote from the same microscope image in RGB order.
    rb_hires = red_minus_blue(hires_rgb)
    rb_raw = red_minus_blue(rgb)
    swap = np.sign(rb_raw) != np.sign(rb_hires)
    if swap:
        rgb = np.ascontiguousarray(rgb[:, :, ::-1])
    log(f"channel order: hires R-B={rb_hires:.1f}, CZI as read R-B={rb_raw:.1f} -> BGR->RGB swap={bool(swap)}")
    I0, gain, n_glass = white_reference(rgb)
    H, W = rgb.shape[:2]
    eff_mpp = native_mpp * raw_W / W
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    tifffile.imwrite(out_tif, rgb)
    log(f"rendered {W}x{H} at {eff_mpp:.4f} um/px -> {out_tif}")
    return eff_mpp, {
        "czi_raw_WH": [raw_W, raw_H],
        "render_WH": [W, H],
        "scale_factor": scale,
        "bgr_to_rgb_swap": bool(swap),
        "rb_hires": rb_hires,
        "rb_czi_raw": rb_raw,
        "white_reference_I0": I0.tolist(),
        "white_gain": gain.tolist(),
        "n_glass_px": n_glass,
    }


def registration_qc(square_002um, X, Y, grey_bins, cov):
    """Agreement of the bin positions with the hires H&E.

    raw: correlation of hires grey at the bin with log1p UMI of the bin (whitespace is bright and has few
    transcripts); hp: the same after subtracting 25 um local means from both (fine structure only). The hp
    correlation drops to ~0 when the map is shifted by 4 hires pixels.
    """
    import h5py

    with h5py.File(f"{square_002um}/filtered_feature_bc_matrix.h5", "r") as f:
        ip = f["matrix/indptr"][:]
        bc = f["matrix/barcodes"][:].astype(str)
        data = f["matrix/data"][:]
    tot = np.bincount(
        np.repeat(np.arange(len(ip) - 1), np.diff(ip)), weights=data.astype(np.float64), minlength=len(ip) - 1
    )
    del data
    mpp = read_json(f"{square_002um}/spatial/scalefactors_json.json")["microns_per_pixel"]
    tp = pd.read_parquet(
        f"{square_002um}/spatial/tissue_positions.parquet",
        columns=["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    )
    tp = tp.set_index("barcode").loc[bc]
    s = pd.Series(
        np.log1p(tot), index=bin_key(tp.pxl_col_in_fullres.to_numpy() * mpp, tp.pxl_row_in_fullres.to_numpy() * mpp)
    )
    s = s[~s.index.duplicated()]
    lu = s.reindex(bin_key(X, Y)).to_numpy()
    ok = cov & np.isfinite(lu)
    if ok.sum() <= 100:
        return {"raw": float("nan"), "hp": float("nan"), "n": int(ok.sum())}
    r = float(np.corrcoef(grey_bins[ok], lu[ok])[0, 1])
    luf = np.where(np.isfinite(lu), lu, 0).astype(np.float32)
    gf = grey_bins.astype(np.float32)
    lhp = luf - local_density(X, Y, luf, 25.0)
    ghp = gf - local_density(X, Y, gf, 25.0)
    rhp = float(np.corrcoef(ghp[ok], lhp[ok])[0, 1])
    log(
        f"registration QC: corr(hires grey, log1p UMI) raw = {r:.4f}, high-pass = {rhp:.4f} over {int(ok.sum()):,} bins"
    )
    return {"raw": r, "hp": rhp, "n": int(ok.sum())}


def sample_hires(hires_rgb, X, Y, CMPP):
    """Bilinear RGB per bin from the hires H&E at pixel = um / CMPP; haematoxylin and eosin OD from rgb2hed."""
    Hh, Wh = hires_rgb.shape[:2]
    px = X / CMPP
    py = Y / CMPP
    cov = (px >= 0) & (px <= Wh - 1) & (py >= 0) & (py <= Hh - 1)
    rgb = np.full((len(X), 3), 255.0, np.float32)
    for ch in range(3):
        rgb[cov, ch] = map_coordinates(hires_rgb[:, :, ch], [py[cov], px[cov]], order=1, mode="nearest")
    grey = luminance(rgb).astype(np.float32)
    hed = rgb2hed((rgb / 255.0).reshape(-1, 1, 3))
    hem = hed[:, 0, 0].astype(np.float32)
    eod = hed[:, 0, 1].astype(np.float32)
    return grey, eod, hem, cov


def adaptive_cuts(hires_rgb):
    """Whitespace, eosin and lumen cuts from the grey of tissue and background in the hires image."""
    small = hires_rgb[::4, ::4]
    tis = hires_mask.tissue_mask(small)
    g = luminance(small).astype(np.float32)
    eod_img = rgb2hed(small / 255.0)[:, :, 1]
    bg = float(np.median(g[~tis]))
    tg = float(np.median(g[tis]))
    em = float(np.median(eod_img[tis]))
    cuts = {
        "bg_grey": bg,
        "tissue_grey": tg,
        "grey_cut": tg + GREY_TO_BG_FRAC * (bg - tg),
        "eod_median": em,
        "eod_cut": EOD_FRAC * em,
        "lumen_grey_cut": tg + LUMEN_TO_BG_FRAC * (bg - tg),
        "GREY_TO_BG_FRAC": GREY_TO_BG_FRAC,
        "EOD_FRAC": EOD_FRAC,
        "LUMEN_TO_BG_FRAC": LUMEN_TO_BG_FRAC,
        "tissue_mask_frac": float(tis.mean()),
    }
    log(f"adaptive cuts: {json.dumps({k: round(v, 4) for k, v in cuts.items()})}")
    return cuts


def he_from_czi(args, mpp, hires_path, hires_rgb, CMPP, X, Y, cuts):
    """H&E per bin from the rendered CZI, registered to the hires image; whitespace by the fixed cuts."""
    render_tif = os.path.join(
        args.render_dir or os.environ.get("TMPDIR", args.outdir), f"{args.slide}_czi_{CZI_TARGET_MPP:.2f}umpx.tif"
    )
    eff_mpp, render = render_czi(args.czi, render_tif, mpp, hires_rgb)
    # the registration downsamples the rendering by round(CMPP / its um per pixel)
    M0, t0, micro_to_cyt = celltyping.register_he(
        args.slide, render_tif, hires_path, CMPP, args.reg_cache, args.force_reg, micro_mpp=eff_mpp
    )
    mic = celltyping.Microscopy(
        tifffile.memmap(render_tif, mode="r"), np.asarray(Image.open(hires_path).convert("RGB")), (M0, t0), CMPP
    )
    he = celltyping.sample_he(mic, X, Y)
    eod, hema, cov = he["eosin_od"].to_numpy(), he["hema_od"].to_numpy(), he["covered"].to_numpy()
    gcyt, grey_eff = he["grey_cyt"].to_numpy(), he["grey"].to_numpy()
    return {
        "grey_eff": grey_eff,
        "eod": eod,
        "hema": hema,
        "cov": cov,
        "white_bin": (grey_eff > cuts["grey"]) | (cov & (eod < cuts["eosin_od"])),
        "lumen_white": grey_eff > cuts["lumen_grey"],
        "qc_grey": gcyt,
        "cuts": {"WHITE_CUT": cuts["grey"], "EOD_MIN": cuts["eosin_od"], "WHITE_CUT_LUMEN": cuts["lumen_grey"]},
        "field": {"field_kept": he.attrs["field"]["nkeep"], "field_cells": he.attrs["field"]["ncells"]},
        "render": render,
        "registration": {"micro_to_cyt": micro_to_cyt, "micro_um_per_px": eff_mpp},
    }


def he_from_hires(hires_rgb, CMPP, X, Y):
    """H&E per bin from the hires image; whitespace by cuts adapted to its grey canvas."""
    cuts = adaptive_cuts(hires_rgb)
    grey, eod, hema, cov = sample_hires(hires_rgb, X, Y, CMPP)
    grey_eff = np.where(cov, grey, cuts["bg_grey"]).astype(np.float32)
    eod = np.where(cov, eod, 0.0).astype(np.float32)
    hema = np.where(cov, hema, 0.0).astype(np.float32)
    return {
        "grey_eff": grey_eff,
        "eod": eod,
        "hema": hema,
        "cov": cov,
        "white_bin": cov & ((grey_eff > cuts["grey_cut"]) | (eod < cuts["eod_cut"])),
        "lumen_white": cov & (grey_eff > cuts["lumen_grey_cut"]),
        "qc_grey": grey,
        "cuts": cuts,
        "field": {},
        "render": {},
        "registration": {},
    }


def main():
    """Sample the H&E of one AH slide per bin, flag its bins and write the flag table and counts."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[cmflag] %(message)s")
    cfg = load_config("substrate_cmflag")
    order = read_json(args.factor_order)
    os.makedirs(args.outdir, exist_ok=True)
    slide = args.slide
    flags_path = f"{args.outdir}/{slide}_clean_bins.parquet"
    if os.path.exists(flags_path) and not args.force:
        log(f"skip {slide}: flags exist")
        return
    start = time.time()
    square_002um = f"{args.spaceranger}/binned_outputs/square_002um"
    hires_path = args.hires or f"{square_002um}/spatial/tissue_hires_image.png"
    scalefactors = f"{square_002um}/spatial/scalefactors_json.json"
    inputs = {
        "h5": f"{square_002um}/filtered_feature_bc_matrix.h5",
        "tp": f"{square_002um}/spatial/tissue_positions.parquet",
        "sf2": scalefactors,
        "cyt": hires_path,
        "h5ad": args.h5ad,
    }
    for name, path in inputs.items():
        if not os.path.exists(path):
            log(f"missing input {name}: {path}")
    mpp = float(read_json(scalefactors)["microns_per_pixel"])
    hires_rgb = np.asarray(Image.open(hires_path).convert("RGB"))
    micro_wh = tuple(int(v) for v in args.micro_dims.split(",")) if args.micro_dims else czi_dims(args.czi)
    hs, CMPP, aspect_rel_diff = hires_scale((hires_rgb.shape[1], hires_rgb.shape[0]), micro_wh, mpp)
    log(f"=== {slide}  he_source={args.he_source}  hires um/px={CMPP:.4f}  mpp={mpp:.6f} ===")

    X, Y, assign = load_bins(args.h5ad, order)
    from_czi = args.he_source == "czi"
    if from_czi:
        he = he_from_czi(args, mpp, hires_path, hires_rgb, CMPP, X, Y, cfg["whitespace"])
    else:
        he = he_from_hires(hires_rgb, CMPP, X, Y)
    grey_eff, eod, hema, cov, white_bin = he["grey_eff"], he["eod"], he["hema"], he["cov"], he["white_bin"]
    reg = {
        "method": "pure_scale_offset0",
        "hs": hs,
        "CMPP_hires": CMPP,
        "microscope_WH": list(micro_wh),
        "hires_WH": [int(hires_rgb.shape[1]), int(hires_rgb.shape[0])],
        "aspect_rel_diff": aspect_rel_diff,
        **he["registration"],
        "qc_corr_grey_logumi": registration_qc(square_002um, X, Y, he["qc_grey"], cov),
    }

    resident, gate = celltyping.nucleus_gate(hema, cov, white_bin, cfg["nucleus_fraction"])
    marker = celltyping.marker_depth(celltyping.read_bins(square_002um), X, Y, white_bin, cfg["markers"])
    fits = {"cm": read_json(args.cm_fit), "vsmc": read_json(args.vsmc_fit)}
    flags = celltyping.spillover_flags(
        assign, order, marker, white_bin, resident, cfg["flags"], fits, cfg["min_spillover_share"]
    )
    lumen = celltyping.vessel_lumen(X, Y, he["lumen_white"], args.structs, args.cells)
    table = celltyping.flag_table(X, Y, assign, flags, lumen, resident, grey_eff, eod, hema, cov)
    table.to_parquet(flags_path, index=False)
    n = len(X)
    counts = {
        "slide": slide,
        "tier": ("full_microscopy_HE" if from_czi else "hires_HE"),
        "he_source": ("microscopy_czi" if from_czi else "spaceranger_hires"),
        "n_bins": int(n),
        "assign_source": "argmax_L3_h5ad",
        "gcert_available": False,
        "whitespace_source": ("microscopy_czi_fixed_cuts" if from_czi else "hires_adaptive_cuts"),
        "cuts": he["cuts"],
        "czi_render": he["render"],
        "registration": reg,
        **gate,
        "micro_coverage": float(cov.mean()),
        "real_white_frac": float(white_bin.mean()),
        **he["field"],
        "pct_excluded_total": celltyping.percent(table["exclude_spillover"].to_numpy().sum(), n),
        "pct_flagged_any": celltyping.percent(table[SPILLOVER_FLAGS].to_numpy().any(axis=1).sum(), n),
        **celltyping.flag_counts(flags, lumen),
        "cm_lambda": fits["cm"]["lam"],
        "vsmc_lambda": fits["vsmc"]["lam"],
        "F_HI": cfg["min_spillover_share"],
    }
    with open(f"{args.outdir}/{slide}_counts.json", "w") as handle:
        json.dump(counts, handle, indent=2, default=float)
    log(f"wrote {flags_path} and counts.json  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
