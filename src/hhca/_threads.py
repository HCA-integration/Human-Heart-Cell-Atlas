"""Thread settings of the numerical libraries."""

from __future__ import annotations

import os

BLAS_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
NUMBA_VARIABLES = ("NUMBA_NUM_THREADS",)


def thread_env(n_threads: int = 1, *, numba: bool = True, keep_existing: bool = False) -> dict[str, str]:
    """Environment variables that limit BLAS, OpenMP, numexpr and numba to `n_threads` threads.

    The variables are read when numpy is imported, so apply them before that import::

        os.environ.update(thread_env(1))

    Parameters
    ----------
    n_threads
        Number of threads per library.
    numba
        Also set ``NUMBA_NUM_THREADS``.
    keep_existing
        Leave out variables that are already set, e.g. by the SLURM job.

    Returns
    -------
    Mapping of variable name to value, ready for ``os.environ.update``.
    """
    names = BLAS_VARIABLES + (NUMBA_VARIABLES if numba else ())
    return {name: str(n_threads) for name in names if not (keep_existing and name in os.environ)}
