"""Chunked-array backend detection and shared helpers.

Provides utilities to detect whether an array is backed by dask, cubed, or
is a plain numpy array, plus backend-agnostic helpers shared by the dask
and cubed graph builders.
"""

from __future__ import annotations

import numpy as np


def detect_backend(data) -> str | None:
    """Return ``'dask'``, ``'cubed'``, or ``None`` for the array's chunked backend."""
    if hasattr(data, "dask"):
        return "dask"
    try:
        import cubed

        if isinstance(data, cubed.Array):
            return "cubed"
    except ImportError:
        pass
    return None


def is_chunked(data) -> bool:
    """Return ``True`` if *data* is backed by any chunked backend."""
    return detect_backend(data) is not None


def stack_arrays(arrays, *, axis: int, backend: str):
    """Stack a sequence of chunked arrays along *axis*.

    Dispatches to the correct backend's ``stack`` implementation.
    """
    if backend == "dask":
        import dask.array

        return dask.array.stack(arrays, axis=axis)
    if backend == "cubed":
        import cubed

        return cubed.stack(arrays, axis=axis)
    raise ValueError(f"Unknown backend: {backend!r}")


def _derive_batch_size(
    dtype: np.dtype,
    dst_chunks: tuple[int, int],
    batch_size: int | None,
    max_task_bytes: int,
) -> int:
    """Return actual batch size: explicit if given, else derived from *max_task_bytes*.

    *dst_chunks* is used as a proxy for output tile memory. Pass explicit *dst_chunks*
    for accuracy when source and destination pixel sizes differ substantially.
    """
    if batch_size is not None:
        return max(1, batch_size)
    chunk_bytes = dst_chunks[0] * dst_chunks[1] * dtype.itemsize
    return max(1, max_task_bytes // chunk_bytes)


def _compute_chunk_sizes(total: int, chunk: int) -> tuple[int, ...]:
    """Return chunk-sizes tuple for one dimension."""
    n_full = total // chunk
    remainder = total % chunk
    if remainder:
        return (chunk,) * n_full + (remainder,)
    return (chunk,) * n_full
