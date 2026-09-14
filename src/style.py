"""HHCA manuscript unified style — the authoritative Level-3 palette (user-provided).
Import in every panel script so the whole manuscript shares one palette + style.

    from hhca import style as H
    from hhca import style as H
    H.apply()                          # submission-ready rcParams (muted / professional)
    ... color=H.ct("Fibroblasts") ...  cmap=H.DIVERGING ...  color=H.SEX["female"] ...
    H.stars(p); H.save(fig, "stem")    # stem.pdf + stem.png (300 dpi, TrueType)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  # registers rocket / mako / flare / crest / vlag / icefire colormaps

# ---- categorical: the paper's Level-3 palette (exact, user-provided; softer/muted) ----
CT = {
    "Adipocytes": "#d7b699",
    "Atrial Cardiomyocytes": "#eea2a4",
    "Ventricular Cardiomyocytes": "#eabcc5",
    "Endocardial Endothelial Cells": "#7ac4bd",
    "Lymphatic Endothelial Cells": "#26386a",
    "Arterial Endothelial cells": "#4292a6",
    "Capillary Endothelial cells": "#96dcdd",
    "Venous Endothelial cells": "#6b9acb",
    "Epicardium": "#cdc6a8",
    "B cells": "#de91b3",
    "Plasma Cells": "#b471ee",
    "Dendritic cells": "#c2eec1",
    "Macrophages": "#90ba80",
    "Mast cells": "#62a65e",
    "Monocytes": "#4da79d",
    "ILC": "#a565bc",
    "Lymphoid Cells (Proliferating)": "#e9bd96",
    "NK cells": "#d19faf",
    "T cells": "#d587a4",
    "Neural Cells": "#f3e5b8",
    "Fibroblasts": "#e7af83",
    "Pericytes": "#dbcff0",
    "Vascular Smooth Muscle cells": "#8161bc",
}
_ALIAS = {"Ventricular CM": "Ventricular Cardiomyocytes", "Atrial CM": "Atrial Cardiomyocytes",
          "VSMC": "Vascular Smooth Muscle cells", "Arterial EC": "Arterial Endothelial cells",
          "Capillary EC": "Capillary Endothelial cells", "Venous EC": "Venous Endothelial cells",
          "Endocardial EC": "Endocardial Endothelial Cells", "Lymphatic EC": "Lymphatic Endothelial Cells",
          "Macrophage": "Macrophages", "T cell": "T cells", "NK cell": "NK cells",
          "Lymphoid Proliferating": "Lymphoid Cells (Proliferating)"}
def ct(name, default="#B9B9B9"):
    if name in CT: return CT[name]
    if name in _ALIAS: return CT[_ALIAS[name]]
    return default

# ---- two-group (sex): red female / blue male, muted professional tones (ColorBrewer RdBu ends) ----
SEX = {"female": "#B2182B", "male": "#2166AC"}
# ---- continuous colormaps ----
DIVERGING  = "RdBu_r"    # z-scores, log fold-change (red=high/+, blue=low/-)
SEQUENTIAL = "rocket_r"  # expression on UMAP / spatial (light -> crimson -> black)  [Option B]
CONTINUOUS = "mako"      # physical scores (stress, density)  (black -> teal -> mint) [Option B]
GREY = "#DCDCDC"; INK = "#2B2B2B"; MUTED = "#7A7A7A"

FLOOR = 7          # never render text below this (pt); labels must survive shrinking to small print
PANEL = 2.6        # default authored panel width (in) ≈ final placement size, so no shrink < FLOOR

def apply():
    """Minimal-text, print-final rcParams. Panels are authored at ~PANEL inches so the
    6-7pt fonts here ARE the final print sizes (>= FLOOR). Keep on-panel text to axis
    labels + short tick/category labels + significance stars; put all prose in the legend."""
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.titleweight": "regular", "axes.labelcolor": INK, "text.color": INK,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
        "lines.linewidth": 1.4, "lines.markersize": 4, "patch.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.5,
        "legend.borderpad": 0.2, "legend.labelspacing": 0.3,
        "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": False,
    })

def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

def save(fig, stem, dpi=300):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    return f"{stem}.pdf"
