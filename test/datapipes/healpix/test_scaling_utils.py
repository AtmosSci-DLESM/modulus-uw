# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scalar vs spatial climatology scaling helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physicsnemo.datapipes.healpix.scaling_utils import (
    build_channel_scaling,
    is_nonzero_scalar_mean,
    resolve_mean,
    stack_normalized_bounds,
)


def test_resolve_mean_scalar_and_npy(tmp_path: Path):
    assert resolve_mean(1.5) == 1.5
    arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "clim.npy"
    np.save(path, arr)
    loaded = resolve_mean(str(path))
    assert loaded.shape == (2, 3, 4)
    np.testing.assert_array_equal(loaded, arr)


def test_build_channel_scaling_scalar_and_mixed(tmp_path: Path):
    clim = np.ones((12, 4, 4), dtype=np.float32) * 10.0
    path = tmp_path / "t.npy"
    np.save(path, clim)
    scaling = {
        "a": {"mean": 1.0, "std": 2.0},
        "b": {"mean": str(path), "std": 3.0},
    }
    out = build_channel_scaling(
        ["a", "b"],
        scaling,
        mean_expand_axes=(0, 2, 3, 4),
    )
    assert out["mean"].shape == (1, 2, 12, 4, 4)
    assert out["std"].shape == (1, 2, 1, 1, 1)
    np.testing.assert_allclose(out["mean"][0, 0], 1.0)
    np.testing.assert_allclose(out["mean"][0, 1], 10.0)
    np.testing.assert_allclose(out["std"][0, :, 0, 0, 0], [2.0, 3.0])


def test_constants_reject_spatial_mean(tmp_path: Path):
    path = tmp_path / "c.npy"
    np.save(path, np.zeros((12, 2, 2), dtype=np.float32))
    scaling = {"c": {"mean": str(path), "std": 1.0}}
    with pytest.raises(ValueError, match="spatial scaling mean"):
        build_channel_scaling(
            ["c"],
            scaling,
            mean_expand_axes=(1, 2, 3),
            allow_spatial_mean=False,
        )


def test_odd_mean_helper():
    assert is_nonzero_scalar_mean(1.0)
    assert not is_nonzero_scalar_mean(0.0)
    assert not is_nonzero_scalar_mean("/tmp/clim.zarr/v")
    assert not is_nonzero_scalar_mean(np.ones((2, 2, 2)))


def test_stack_normalized_bounds_spatial(tmp_path: Path):
    clim = np.full((2, 3, 3), 5.0, dtype=np.float32)
    path = tmp_path / "q.npy"
    np.save(path, clim)
    scaling = {
        "free": {"mean": 0.0, "std": 1.0},
        "q": {"mean": str(path), "std": 2.0},
    }
    thr = stack_normalized_bounds(
        ["free", "q"],
        scaling,
        {"q": 0.0},
        unconstrained=float("-inf"),
    )
    assert thr.shape == (1, 2, 1, 2, 3, 3)
    assert np.isneginf(thr[0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(thr[0, :, 0, 1, :, :], (0.0 - 5.0) / 2.0)
