"""Progress messages of the pipeline scripts."""

from __future__ import annotations

import time
from typing import Callable


def get_logger(tag: str = "") -> Callable[..., None]:
    """Return a ``print``-like function that prefixes each message with `tag` and the seconds since this call.

    >>> log = get_logger("niche")
    >>> log("reading", 3, "slides")  # doctest: +SKIP
    [niche    0.0s] reading 3 slides
    """
    start = time.time()
    prefix = f"{tag} " if tag else ""

    def log(*message) -> None:
        print(f"[{prefix}{time.time() - start:7.1f}s]", *message, flush=True)

    return log
