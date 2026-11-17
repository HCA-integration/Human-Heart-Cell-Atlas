"""Writing figure panels."""

from __future__ import annotations

import os

from hhca.paths import results_dir


def figure_dir(figure: str) -> str:
    """Folder of a figure's panels, ``results/figures/<figure>``, created on first use."""
    folder = results_dir("figures", figure)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_panel(fig, figure: str, panel: str, dpi: int | None = None, **savefig_kwargs) -> str:
    """Write a panel as ``panel_<panel>.png`` and ``.pdf`` in the figure's folder and return the PNG path.

    `dpi` defaults to the resolution of the panel (:class:`hhca.pl.PanelFigure`), or 300 for other figures.
    """
    folder = figure_dir(figure)
    dpi = dpi or getattr(fig, "save_dpi", 300)
    savefig_kwargs.setdefault("facecolor", "white")
    for extension in ("png", "pdf"):
        fig.savefig(f"{folder}/panel_{panel}.{extension}", dpi=dpi, **savefig_kwargs)
    return f"{folder}/panel_{panel}.png"


def save_figure(fig, stem: str, dpi: int = 300) -> str:
    """Write a figure as ``<stem>.pdf`` and ``<stem>.png`` with a tight bounding box and return the PDF path."""
    for extension in ("pdf", "png"):
        fig.savefig(f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")
    return f"{stem}.pdf"
