"""Helpers shared by the HHCA analysis scripts, figure panels and notebooks.

Importing the package does not import numpy, so :func:`thread_env` can be used to set the BLAS thread
variables first::

    import os

    from hhca import thread_env

    os.environ.update(thread_env(1))

    import numpy as np
"""

import importlib

from hhca._threads import thread_env
from hhca.paths import REPO, config_path, results_dir

__all__ = ["REPO", "config_path", "results_dir", "thread_env"]
__version__ = "0.1.0"

# analysis (hhca.tl), panels (hhca.pl) and the other submodules are imported on first use
_SUBMODULES = ("config", "grid", "io", "pl", "stats", "tl")


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"hhca.{name}")
    raise AttributeError(f"module 'hhca' has no attribute {name!r}")
