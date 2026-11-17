"""Figures of publication panels: drawn at their final size with their own matplotlib settings."""

from __future__ import annotations

import io
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

PANEL_DPI = 300


@contextmanager
def panel_rc(rc: dict):
    """Matplotlib defaults plus the settings of a panel, whatever style is active outside."""
    with plt.style.context("default"), mpl.rc_context(rc):
        yield


class PanelFigure(Figure):
    """Figure of a publication panel.

    The panel keeps its matplotlib settings: it is built, drawn and saved with them, so it looks the same in a
    script and in a notebook that uses another style. In a notebook it is shown as it is saved, at the full
    figure size and the resolution `save_dpi`.
    """

    def __init__(self, *args, rc: dict | None = None, save_dpi: int = PANEL_DPI, **kwargs):
        self.rc = dict(rc or {})
        self.save_dpi = save_dpi
        with panel_rc(self.rc):
            super().__init__(*args, **kwargs)

    def draw(self, renderer):
        """Draw with the settings of the panel."""
        with panel_rc(self.rc):
            super().draw(renderer)

    def savefig(self, fname, **kwargs):
        """Save with the settings of the panel."""
        with panel_rc(self.rc):
            super().savefig(fname, **kwargs)

    def _ipython_display_(self):
        from IPython.display import Image, display

        buffer = io.BytesIO()
        self.savefig(buffer, format="png", dpi=self.save_dpi, facecolor="white")
        display(Image(buffer.getvalue(), width=round(self.get_figwidth() * 96)))


@contextmanager
def panel(figsize, rc: dict, save_dpi: int = PANEL_DPI):
    """Open a :class:`PanelFigure` of `figsize` inches; everything added inside uses the settings `rc`.

    Settings such as ``savefig.bbox`` apply when the panel is saved; `save_dpi` is the resolution of the saved PNG.
    """
    with panel_rc(rc):
        yield plt.figure(figsize=figsize, FigureClass=PanelFigure, rc=rc, save_dpi=save_dpi)
