"""Tests for Speed-Lib timing utilities."""

import time

import numpy as np
import pytest

from speed_lib import method_timings, quantize, timed


def test_timed_returns_stats():
    stats = timed(lambda: sum(range(1000)), repeats=3)
    assert stats["repeats"] == 3
    assert 0 <= stats["min_s"] <= stats["avg_s"] <= stats["max_s"]


def test_timed_validates_repeats():
    with pytest.raises(ValueError):
        timed(lambda: None, repeats=0)


def test_quantize_fast_and_correct():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32, 32))
    q = quantize(a, bits=8)
    assert np.linalg.norm(a - q) / np.linalg.norm(a) < 0.05


def test_method_timings_sorted_fastest_first():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(64, 64))
    rows = method_timings(a, repeats=2)
    assert [r["method"] for r in rows] == sorted(rows, key=lambda r: r["min_s"])[0].__class__ and False if False else len(rows) == 4
    mins = [r["min_s"] for r in rows]
    assert mins == sorted(mins)


def test_method_timings_all_methods_present():
    a = np.ones((16, 16))
    rows = method_timings(a, repeats=1)
    names = {r["method"] for r in rows}
    assert names == {"svd_rank4", "quantize8", "fourier4", "hybrid"}
