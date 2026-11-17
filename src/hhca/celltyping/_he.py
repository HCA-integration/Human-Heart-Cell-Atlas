"""Registration of the microscopy H&E to the CytAssist image of a Visium HD section, and H&E values per 2 um bin.

registration   register_he    SIFT + RANSAC affine, ECC refinement (microscopy -> CytAssist image); gives the map
                              from um to microscopy pixels
offset field   offset_field   phase correlation between the two images on a 300 um lattice
H&E per bin    sample_he      grey, eosin and haematoxylin optical density, with the offset field interpolated per
                              bin (CytAssist grey outside the microscopy image)

OpenCV, scikit-image and Pillow are imported inside the functions that need them.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, generic_filter, map_coordinates

log = logging.getLogger("hhca.celltyping").info

MICROSCOPY_UM_PER_PX = 0.3867850693862723  # one scanner for all sections; SIFT absorbs the residual

# offset field: lattice spacing, half width of the correlated tile, largest accepted error and offset (um)
SPACING = 300.0
TILE_HALF = 160.0
ERR_MAX = 0.45
OFF_CAP = 110.0
SAMPLE_TILE = 500.0  # bins are sampled from the microscopy image in tiles of this size (um)


def luminance(rgb):
    """Grey value of RGB (last axis)."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def cytassist_scale(scalefactors) -> tuple[float, float]:
    """Um per pixel of the CytAssist hires image and of the full-resolution image, from a Space Ranger
    ``scalefactors_json.json`` (path or parsed content).
    """
    if not isinstance(scalefactors, dict):
        import json

        with open(scalefactors) as handle:
            path, scalefactors = scalefactors, json.load(handle)
    else:
        path = "scale factors"
    microns_per_pixel = scalefactors.get("microns_per_pixel")
    hires_scale = scalefactors.get("tissue_hires_scalef")
    if microns_per_pixel and hires_scale:
        return float(microns_per_pixel) / float(hires_scale), float(microns_per_pixel)
    raise RuntimeError(f"cannot derive the hires scale from {path}: keys={list(scalefactors)}")


def _clahe_grey(rgb):
    """Contrast-equalised grey image."""
    import cv2

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)


def _sift_ecc_affine(micro_ds_rgb, cyt_rgb):
    """Affine map (3 x 3) from the downsampled microscopy image to the CytAssist image.

    SIFT matches with a RANSAC affine fit, refined by ECC; the refinement is kept if it raises the correlation
    of the warped images. Also returns that correlation, the method used and the numbers of RANSAC inliers and
    of matches.
    """
    import cv2

    CH, CW = cyt_rgb.shape[:2]
    gM = _clahe_grey(micro_ds_rgb)
    gC = _clahe_grey(cyt_rgb)
    # 60,000 keypoints are plenty for one affine map and keep the descriptor sets small
    sift = cv2.SIFT_create(nfeatures=60000, contrastThreshold=0.02)
    kM, dM = sift.detectAndCompute(gM, None)
    kC, dC = sift.detectAndCompute(gC, None)
    log(f"SIFT keypoints micro_ds={len(kM)} cyt={len(kC)}")
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    raw = flann.knnMatch(np.asarray(dM, np.float32), np.asarray(dC, np.float32), k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    ptsM = np.float32([kM[m.queryIdx].pt for m in good])
    ptsC = np.float32([kC[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffine2D(
        ptsM, ptsC, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999
    )
    ninl = int(inl.sum()) if inl is not None else 0
    log(f"RANSAC good={len(good)} inliers={ninl}")

    def to3(A):
        H = np.eye(3)
        H[:2] = A
        return H

    def warp_ncc(Hds2cyt):
        Wm = cv2.warpAffine(gM, Hds2cyt[:2], (CW, CH), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.ones_like(gM), Hds2cyt[:2], (CW, CH), flags=cv2.INTER_NEAREST) > 0
        a = Wm[mask].astype(np.float32)
        b = gC[mask].astype(np.float32)
        if a.std() < 1e-6 or b.std() < 1e-6:
            return -1.0
        return float(np.corrcoef(a, b)[0, 1])

    H_sift = to3(M)
    ncc_sift = warp_ncc(H_sift)
    log(f"NCC SIFT = {ncc_sift:.4f}")
    homography, ncc, used = H_sift, ncc_sift, "SIFT"
    try:
        warped = cv2.warpAffine(gM, H_sift[:2], (CW, CH), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        tmpl = gC.astype(np.float32) / 255.0
        mask_u = (cv2.warpAffine(np.ones_like(gM), H_sift[:2], (CW, CH), flags=cv2.INTER_NEAREST) * 255).astype(
            np.uint8
        )
        Rwarp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
        cc, Rwarp = cv2.findTransformECC(tmpl, warped, Rwarp, cv2.MOTION_AFFINE, crit, mask_u, 5)
        H_ecc = np.linalg.inv(to3(Rwarp)) @ H_sift
        ncc_ecc = warp_ncc(H_ecc)
        log(f"ECC cc={cc:.4f} -> NCC {ncc_ecc:.4f}")
        if ncc_ecc > ncc_sift:
            homography, ncc, used = H_ecc, ncc_ecc, "SIFT+ECC"
    except cv2.error as e:
        log(f"ECC failed ({str(e)[:70]}); keep SIFT")
    return homography, ncc, used, ninl, len(good)


def register_he(section, tif, cyt, cytassist_um_per_px, cache, force=False, micro_mpp=MICROSCOPY_UM_PER_PX):
    """Register the microscopy H&E of a section to its CytAssist image.

    Parameters
    ----------
    tif
        Microscopy image (TIFF) at `micro_mpp` um per pixel.
    cyt
        CytAssist hires image at `cytassist_um_per_px` um per pixel.
    cache
        ``.npz`` file of the registration: read if it exists (unless `force`), written otherwise.

    Returns
    -------
    ``M0``, ``t0`` (pixel = ``M0 @ um + t0``) and a dictionary with the correlation of the registered images.
    """
    import tifffile
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    if os.path.exists(cache) and not force:
        Z = np.load(cache)
        log(f"registration cache {cache}")
        return Z["M0"], Z["t0"], {"cached": True, "ncc": float(Z["ncc"]) if "ncc" in Z else None}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    mm = tifffile.memmap(tif, mode="r")
    RAW_H, RAW_W = mm.shape[:2]
    step = max(1, int(round(cytassist_um_per_px / micro_mpp)))
    micro_ds_rgb = np.asarray(mm[::step, ::step, :3]).astype(np.uint8)
    DS_H, DS_W = micro_ds_rgb.shape[:2]
    sx = DS_W / RAW_W
    sy = DS_H / RAW_H
    log(f"micro raw {RAW_W}x{RAW_H} -> ds {DS_W}x{DS_H} (step={step}, sx={sx:.6f})")
    cyt_rgb = np.asarray(Image.open(cyt).convert("RGB"))
    homography, ncc, used, ninl, ngood = _sift_ecc_affine(micro_ds_rgb, cyt_rgb)
    Hc2ds = np.linalg.inv(homography)
    Dum = np.diag([1 / cytassist_um_per_px, 1 / cytassist_um_per_px, 1.0])
    Dds2raw = np.diag([1 / sx, 1 / sy, 1.0])
    H_um2raw = Dds2raw @ Hc2ds @ Dum
    M0 = H_um2raw[:2, :2].copy()
    t0 = H_um2raw[:2, 2].copy()
    np.savez_compressed(
        cache,
        M0=M0,
        t0=t0,
        H_um2raw=H_um2raw,
        ncc=ncc,
        used=used,
        ninl=ninl,
        ngood=ngood,
        sx=sx,
        sy=sy,
        CMPP=cytassist_um_per_px,
    )
    log(f"registered {section}: {used} NCC={ncc:.4f} inliers={ninl}/{ngood} -> {cache}")
    return M0, t0, {"cached": False, "ncc": float(ncc), "used": used, "ninl": ninl, "ngood": ngood}


class Microscopy:
    """Microscopy H&E and CytAssist image of a section, sampled at coordinates in um.

    Parameters
    ----------
    image
        Microscopy image (rows x columns x channels); a memory-mapped image can be passed.
    cytassist
        CytAssist hires image (RGB).
    um_to_px
        (matrix, offset) of the affine map from um to microscopy pixels (``M0``, ``t0`` of the registration).
    cytassist_um_per_px
        Scale of the CytAssist hires image.
    """

    def __init__(self, image, cytassist, um_to_px, cytassist_um_per_px):
        self.image = image
        self.height, self.width = self.image.shape[:2]
        self.M0, self.t0 = um_to_px
        self.CMPP = cytassist_um_per_px
        self.cyt_grey = luminance(np.asarray(cytassist)).astype(np.float32)

    def um_to_px(self, xum, yum, dx=0.0, dy=0.0):
        """Microscopy pixel (x, y) of positions in um, optionally shifted by (dx, dy) um."""
        r = (
            np.column_stack([np.asarray(xum, float), np.asarray(yum, float)]) + np.array([dx, dy])
        ) @ self.M0.T + self.t0
        return r[:, 0], r[:, 1]

    def covered(self, px, py):
        """Whether the pixel positions lie inside the microscopy image."""
        return (px >= 0) & (px <= self.width - 1) & (py >= 0) & (py <= self.height - 1)

    def sample_rgb(self, px, py):
        """Bilinear RGB at pixel positions (white outside the image) and whether each position is inside."""
        out = np.full((len(px), 3), 255.0, np.float32)
        cov = self.covered(px, py)
        if cov.any():
            bx0 = max(0, int(px[cov].min()) - 2)
            bx1 = min(self.width, int(px[cov].max()) + 2)
            by0 = max(0, int(py[cov].min()) - 2)
            by1 = min(self.height, int(py[cov].max()) + 2)
            blk = np.asarray(self.image[by0:by1, bx0:bx1, :3])
            for ch in range(3):
                out[cov, ch] = map_coordinates(blk[:, :, ch], [py[cov] - by0, px[cov] - bx0], order=1, mode="nearest")
        return out, cov

    def micro_gray(self, gx, gy, dx=0.0, dy=0.0):
        """Microscopy grey on the grid gx x gy (um)."""
        GX, GY = np.meshgrid(gx, gy)
        px, py = self.um_to_px(GX.ravel(), GY.ravel(), dx, dy)
        rgb, _ = self.sample_rgb(px, py)
        return luminance(rgb.reshape(len(gy), len(gx), 3))

    def cyt_gray(self, gx, gy):
        """CytAssist grey on the grid gx x gy (um)."""
        GX, GY = np.meshgrid(gx, gy)
        cx = GX.ravel() / self.CMPP
        cy = GY.ravel() / self.CMPP
        return map_coordinates(self.cyt_grey, [cy, cx], order=1, mode="nearest").reshape(len(gy), len(gx))

    def auto_offset(self, cx, cy, half, res=2.0):
        """Offset (um) of the microscopy against the CytAssist image in the tile around (cx, cy), and its error."""
        from skimage.registration import phase_cross_correlation

        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
        gx = np.arange(x0, x1, res)
        gy = np.arange(y0, y1, res)
        mg = self.micro_gray(gx, gy)
        cg = self.cyt_gray(gx, gy)
        shift, err, _ = phase_cross_correlation(cg, mg, upsample_factor=10, normalization=None)
        return -float(shift[1] * res), -float(shift[0] * res), float(err)

    def on_tissue(self, cx, cy, half, coarse=40):
        """Fractions of the tile around (cx, cy) that lie inside the microscopy image and on tissue."""
        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
        gx = x0 + (np.arange(coarse) + 0.5) / coarse * (x1 - x0)
        gy = y0 + (np.arange(coarse) + 0.5) / coarse * (y1 - y0)
        GX, GY = np.meshgrid(gx, gy)
        px, py = self.um_to_px(GX.ravel(), GY.ravel())
        cov = self.covered(px, py)
        rgb, _ = self.sample_rgb(px, py)
        g = rgb.mean(1)
        tis = cov & (g < 220)
        return float(cov.mean()), float(tis.mean())


def _nan_median(a):
    """Median of the finite values (nan if there are none)."""
    a = a[np.isfinite(a)]
    return np.nan if a.size == 0 else np.median(a)


def _fill_nearest(a):
    """Replace non-finite cells by the nearest finite one (zeros if there is none)."""
    m = ~np.isfinite(a)
    if not m.any():
        return a
    if m.all():
        return np.zeros_like(a)
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def offset_field(mic, X, Y) -> dict:
    """Offset (um) between microscopy and CytAssist image on a 300 um lattice, median-filtered and gap-filled.

    The affine registration leaves local shifts of a few um. In every lattice tile that lies on tissue the two
    images are aligned by phase correlation; offsets with a large error are dropped and filled from their
    neighbours.

    Returns
    -------
    Dictionary with the lattice (``xs``, ``ys``), the offsets (``OFX``, ``OFY``) and the numbers of lattice
    tiles kept (``nkeep``) and in total (``ncells``).
    """
    xs = np.arange(X.min() + TILE_HALF, X.max() - TILE_HALF, SPACING)
    ys = np.arange(Y.min() + TILE_HALF, Y.max() - TILE_HALF, SPACING)
    OFX = np.full((len(ys), len(xs)), np.nan)
    OFY = np.full((len(ys), len(xs)), np.nan)
    nkeep = 0
    for iy, cy in enumerate(ys):
        for ix, cx in enumerate(xs):
            cov, tis = mic.on_tissue(cx, cy, TILE_HALF)
            if cov < 0.97 or tis < 0.55:
                continue
            dX, dY, err = mic.auto_offset(cx, cy, TILE_HALF)
            if err > ERR_MAX or abs(dX) > OFF_CAP or abs(dY) > OFF_CAP:
                continue
            OFX[iy, ix] = dX
            OFY[iy, ix] = dY
            nkeep += 1
        if (iy % 5) == 0:
            log(f"  field row {iy + 1}/{len(ys)} kept {nkeep}")
    OFXm = generic_filter(np.where(np.isfinite(OFX), OFX, np.nan), _nan_median, size=3, mode="nearest")
    OFYm = generic_filter(np.where(np.isfinite(OFY), OFY, np.nan), _nan_median, size=3, mode="nearest")
    OFXm = np.where(np.isfinite(OFXm), OFXm, OFX)
    OFYm = np.where(np.isfinite(OFYm), OFYm, OFY)
    OFXf = _fill_nearest(OFXm)
    OFYf = _fill_nearest(OFYm)
    log(f"field lattice {OFX.shape} kept {nkeep}/{OFX.size}")
    return {"xs": xs, "ys": ys, "OFX": OFXf, "OFY": OFYf, "nkeep": int(nkeep), "ncells": int(OFX.size)}


def _field_offset(F, X, Y):
    """Bilinear offset of the field at each bin."""
    xs, ys, OFX, OFY = F["xs"], F["ys"], F["OFX"], F["OFY"]
    fx = np.clip((X - xs[0]) / (xs[1] - xs[0]), 0, len(xs) - 1.001) if len(xs) > 1 else np.zeros_like(X)
    fy = np.clip((Y - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1.001) if len(ys) > 1 else np.zeros_like(Y)
    ix = fx.astype(int)
    iy = fy.astype(int)
    tx = fx - ix
    ty = fy - iy

    def bil(A):
        return (
            A[iy, ix] * (1 - tx) * (1 - ty)
            + A[iy, ix + 1] * tx * (1 - ty)
            + A[iy + 1, ix] * (1 - tx) * ty
            + A[iy + 1, ix + 1] * tx * ty
        )

    return bil(OFX).astype(np.float32), bil(OFY).astype(np.float32)


def sample_he(mic, X, Y, field=None) -> pd.DataFrame:
    """H&E values at the position of every decode row of a section.

    The microscopy image is sampled at the registered position of each row, shifted by the local offset of
    :func:`offset_field` (computed here unless `field` is given), and separated into haematoxylin and eosin
    optical density.

    Parameters
    ----------
    mic
        The registered images (:class:`Microscopy`).
    X, Y
        Coordinates of the decode rows in um.

    Returns
    -------
    One row per decode row: ``eosin_od``, ``hema_od``, ``covered`` (inside the microscopy image), ``grey_cyt``
    (CytAssist grey) and ``grey`` (microscopy grey; CytAssist grey where the microscopy image does not cover the
    row). ``attrs["field"]`` holds the numbers of lattice tiles of the offset field kept and in total.
    """
    from skimage.color import rgb2hed

    F = offset_field(mic, X, Y) if field is None else field
    n = len(X)
    DXf, DYf = _field_offset(F, X, Y)
    grey = np.full(n, 255.0, np.float32)
    eod = np.zeros(n, np.float32)
    hem = np.zeros(n, np.float32)
    cov = np.zeros(n, bool)
    gcyt = np.full(n, 255.0, np.float32)
    M0 = mic.M0
    xs = np.arange(X.min(), X.max() + SAMPLE_TILE, SAMPLE_TILE)
    ys = np.arange(Y.min(), Y.max() + SAMPLE_TILE, SAMPLE_TILE)
    nt = 0
    for cy in ys:
        for cx in xs:
            m = (X >= cx) & (X < cx + SAMPLE_TILE) & (Y >= cy) & (Y < cy + SAMPLE_TILE)
            if not m.any():
                continue
            idx = np.where(m)[0]
            px, py = mic.um_to_px(X[idx], Y[idx], 0.0, 0.0)
            px = px + (DXf[idx] * M0[0, 0] + DYf[idx] * M0[0, 1])
            py = py + (DXf[idx] * M0[1, 0] + DYf[idx] * M0[1, 1])
            rgb, c = mic.sample_rgb(px, py)
            grey[idx] = luminance(rgb)
            cov[idx] = c
            hed = rgb2hed((rgb / 255.0).reshape(-1, 1, 3))
            hem[idx] = hed[:, 0, 0].astype(np.float32)  # haematoxylin OD
            eod[idx] = hed[:, 0, 1].astype(np.float32)  # eosin OD
            gcyt[idx] = map_coordinates(mic.cyt_grey, [Y[idx] / mic.CMPP, X[idx] / mic.CMPP], order=1, mode="nearest")
            nt += 1
    log(f"sampled {nt} tiles; micro coverage={cov.mean():.3f}")
    he = pd.DataFrame(
        {
            "eosin_od": eod,
            "hema_od": hem,
            "covered": cov,
            "grey_cyt": gcyt,
            "grey": np.where(cov, grey, gcyt).astype(np.float32),
        }
    )
    he.attrs["field"] = {"nkeep": F["nkeep"], "ncells": F["ncells"]}
    return he
