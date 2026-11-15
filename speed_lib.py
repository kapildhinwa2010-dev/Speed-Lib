# Speed-Lib - Reusable timing and latency-measurement library
# Part of the Kapildhinwa2010-dev Toolkit.

"""Speed-Lib - timing utilities for compression-style workloads.

Provides timed(), a best-of-N wall-clock timer, and method_timings(),
which times the four toolkit codecs (SVD, quantize, Fourier, hybrid) on
a matrix so their relative cost can be compared honestly.

Honest limitation: wall-clock timings on a shared machine are noisy;
always report min-of-N, never a single run.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["timed", "method_timings", "svd_compress", "quantize",
           "fourier_compress", "hybrid_compress", "fourier_basis"]


def fourier_basis(n_points: int, num_coeffs: int = 4) -> np.ndarray:
    """Design matrix [1, sin t, cos t, sin 2t, cos 2t, ...] over one period."""
    if n_points < 1 or num_coeffs < 1:
        raise ValueError("n_points and num_coeffs must be >= 1")
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    cols = [np.ones(n_points)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _check(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return matrix


def svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    matrix = _check(matrix)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if num_coeffs < 1:
        raise ValueError("num_coeffs must be >= 1")
    basis = fourier_basis(matrix.shape[1], num_coeffs)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def hybrid_compress(matrix: np.ndarray, rank: int = 4, bits: int = 8) -> np.ndarray:
    """SVD low-rank followed by quantization."""
    return quantize(svd_compress(matrix, rank=rank), bits=bits)


def timed(func, repeats: int = 5) -> dict:
    """Run func() repeats times; return min/avg/max wall-clock seconds.

    min_s is the honest number to report; avg and max show noise.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)
    return {
        "repeats": repeats,
        "min_s": min(samples),
        "avg_s": sum(samples) / len(samples),
        "max_s": max(samples),
    }


def method_timings(matrix: np.ndarray, repeats: int = 3) -> list:
    """Time all four codecs on matrix; list of dicts sorted fastest first."""
    matrix = _check(matrix)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    methods = {
        "svd_rank4": lambda: svd_compress(matrix, rank=4),
        "quantize8": lambda: quantize(matrix, bits=8),
        "fourier4": lambda: fourier_compress(matrix, num_coeffs=4),
        "hybrid": lambda: hybrid_compress(matrix, rank=4, bits=8),
    }
    rows = []
    for name, func in methods.items():
        stats = timed(func, repeats=repeats)
        rows.append({"method": name, "min_s": stats["min_s"],
                     "avg_s": stats["avg_s"]})
    rows.sort(key=lambda row: row["min_s"])
    return rows
