# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained regression tests for HEALPix couplers."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from physicsnemo.datapipes.healpix.couplers import (  # noqa: E402
    ConstantCoupler,
    TrailingAverageCoupler,
)

_FACE, _HEIGHT, _WIDTH = 2, 3, 4
_N_CHANNELS = 2


def _make_coupler_dataset(channel_in, n_time=8):
    return xr.Dataset(
        data_vars={
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                np.zeros(
                    (n_time, len(channel_in), _FACE, _HEIGHT, _WIDTH),
                    dtype="float32",
                ),
            )
        },
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq="3h"),
            "channel_in": list(channel_in),
            "face": np.arange(_FACE),
            "height": np.arange(_HEIGHT),
            "width": np.arange(_WIDTH),
        },
    )


def _make_coupler(batch_size, output_time_dim=1):
    coupler = ConstantCoupler(
        dataset=_make_coupler_dataset(["c0", "c1", "x"]),
        batch_size=batch_size,
        variables=["c0", "c1"],
        input_times=["0h"],
        input_time_dim=1,
        output_time_dim=output_time_dim,
        presteps=0,
    )
    # Normally assigned by setup_coupling().
    coupler.coupled_channel_indices = [0, 1]
    return coupler


def _coupled_fields(batch, timedim=3):
    """Return fields in [B, F, T, C, H, W] layout."""
    return torch.rand(batch, _FACE, timedim, _N_CHANNELS, _HEIGHT, _WIDTH)


def test_base_coupler_invalid_dataset_type():
    with pytest.raises(
        TypeError,
        match=("Coupler only supports xarray Datasets or zarr Groups"),
    ):
        ConstantCoupler(
            dataset={"inputs": np.zeros((4, 12, 1, 4, 4))},
            batch_size=1,
            variables=["z500"],
        )


def test_set_coupled_fields_adapts_to_provided_batch_size():
    """CRPS-style batch expansion must not be rejected or truncated."""
    configured_batch_size = 2
    provided_batch = configured_batch_size * 2
    coupler = _make_coupler(batch_size=configured_batch_size)

    # timedim deliberately differs from batch to catch axis mixups.
    coupler.set_coupled_fields(_coupled_fields(provided_batch, timedim=3))

    assert coupler.coupled_mode
    assert coupler.preset_coupled_fields.shape[1] == provided_batch


def test_set_coupled_fields_matches_configured_batch_size():
    coupler = _make_coupler(batch_size=3)
    coupler.set_coupled_fields(_coupled_fields(3, timedim=5))

    assert coupler.preset_coupled_fields.shape[1] == 3


def test_set_coupled_fields_broadcasts_all_channels_from_first_time():
    """Constant coupling must preserve every channel, not only the last one."""
    integration_steps = 3
    batch, timedim = 2, 4
    coupler = _make_coupler(
        batch_size=batch,
        output_time_dim=integration_steps,
    )
    fields = torch.zeros(batch, _FACE, timedim, _N_CHANNELS, _HEIGHT, _WIDTH)
    fields[:, :, 0, 0, :, :] = 1.0
    fields[:, :, 0, 1, :, :] = 2.0
    fields[:, :, 1:, :, :, :] = 99.0

    coupler.set_coupled_fields(fields)
    out = coupler.construct_integrated_couplings()

    assert out.shape == (
        integration_steps,
        batch,
        _N_CHANNELS,
        _FACE,
        _HEIGHT,
        _WIDTH,
    )
    assert torch.equal(out[:, :, 0], torch.ones_like(out[:, :, 0]))
    assert torch.equal(out[:, :, 1], torch.full_like(out[:, :, 1], 2.0))


def test_reset_coupler_requires_batch_and_bsize():
    coupler = _make_coupler(batch_size=2)
    coupler.set_coupled_fields(_coupled_fields(2, timedim=3))
    coupler.reset_coupler()

    with pytest.raises(
        ValueError,
        match=("batch and bsize must be provided when not in coupled_mode"),
    ):
        coupler.construct_integrated_couplings()


def test_set_scaling_missing_variable_raises():
    coupler = _make_coupler(batch_size=1)
    scaling_da = (
        pd.DataFrame({"mean": [0.0], "std": [1.0]}, index=["c0"])
        .rename_axis("index")
        .to_xarray()
        .astype("float32")
    )

    with pytest.raises(
        KeyError,
        match=("Coupled variable\\(s\\) not found in scaling values"),
    ):
        coupler.set_scaling(scaling_da)


def test_zarr_variable_order_matches_requested_variables(tmp_path):
    """Zarr path must honor `variables` order, not native channel_in order."""
    native_channels = ["a", "b", "c"]
    requested = ["c", "a"]
    n_time = 4
    data = np.arange(
        n_time * len(native_channels) * _FACE * _HEIGHT * _WIDTH,
        dtype="float32",
    ).reshape(n_time, len(native_channels), _FACE, _HEIGHT, _WIDTH)

    ds = xr.Dataset(
        data_vars={
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                data,
            )
        },
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq="3h"),
            "channel_in": native_channels,
            "face": np.arange(_FACE),
            "height": np.arange(_HEIGHT),
            "width": np.arange(_WIDTH),
        },
    )
    dataset_path = tmp_path / "order.zarr"
    ds.to_zarr(dataset_path)

    xr_ds = xr.open_zarr(dataset_path)
    zarr_ds = zarr.open(str(dataset_path))
    batch_size = 2
    batch = {"time": slice(0, 2)}

    coupler_xr = ConstantCoupler(
        dataset=xr_ds,
        batch_size=batch_size,
        variables=requested,
        input_times=["0h"],
        input_time_dim=1,
        output_time_dim=1,
    )
    coupler_zarr = ConstantCoupler(
        dataset=zarr_ds,
        batch_size=batch_size,
        variables=requested,
        input_times=["0h"],
        input_time_dim=1,
        output_time_dim=1,
    )
    coupler_xr.compute_coupled_indices(interval=1, data_time_step="3h")
    coupler_zarr.compute_coupled_indices(interval=1, data_time_step="3h")

    coupled_xr = coupler_xr.construct_integrated_couplings(
        batch=batch, bsize=batch_size
    )
    coupled_zarr = coupler_zarr.construct_integrated_couplings(
        batch=batch, bsize=batch_size
    )

    input_indices = [native_channels.index(v) for v in requested]
    expected = zarr_ds["inputs"][:2][:, input_indices]
    assert np.array_equal(expected, coupled_xr[0])
    assert np.array_equal(expected, coupled_zarr[0])
    assert coupler_zarr.ds_variable_indices == input_indices

    for i, var in enumerate(requested):
        expected_var = zarr_ds["inputs"][:2][:, native_channels.index(var)]
        assert np.array_equal(expected_var, coupled_zarr[0][:, i])
        assert np.array_equal(expected_var, coupled_xr[0][:, i])


def test_trailing_average_preserves_all_coupled_variables():
    """TrailingAverageCoupler (opt-in mode) must keep every coupled variable
    through averaging, and select the correct raw indices per period."""
    variables = ["c0", "c1", "c2"]
    input_times = ["6h", "12h"]
    batch_size = 2
    coupler = TrailingAverageCoupler(
        dataset=_make_coupler_dataset(variables + ["x"], n_time=16),
        batch_size=batch_size,
        variables=variables,
        presteps=0,
        averaging_window="6h",
        input_times=input_times,
        input_time_dim=2,
        output_time_dim=2,
        use_inclusive_trailing_average=True,
    )
    mock_coupled_module = SimpleNamespace(output_variables=variables, time_step="3h")
    coupler.setup_coupling(mock_coupled_module)
    assert coupler.coupled_channel_indices == [0, 1, 2]

    channel_bases = [10.0, 100.0, 1000.0]
    boundary_offsets = [-5.0, -50.0, -500.0]
    boundary = torch.empty(
        batch_size, coupler.spatial_dims[0], 1, len(variables),
        coupler.spatial_dims[1], coupler.spatial_dims[2],
    )
    for i, (base, offset) in enumerate(zip(channel_bases, boundary_offsets)):
        boundary[:, :, 0, i, :, :] = base + offset
    coupler.seed_boundary_state(boundary)

    coupled_fields = torch.empty(
        batch_size,
        coupler.spatial_dims[0],
        4,
        len(variables),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    for i, base in enumerate(channel_bases):
        for t in range(4):
            coupled_fields[:, :, t, i, :, :] = base + t

    coupler.set_coupled_fields(coupled_fields)
    result = coupler.construct_integrated_couplings()
    assert list(result.shape) == [
        coupler.coupled_integration_dim,
        batch_size,
        coupler.timevar_dim,
    ] + list(coupler.spatial_dims)

    # dt=3h, averaging_window=6h -> window_steps=2; input_times=[6h,12h] -> r=[2,4].
    # Post-prepend buffer is [boundary, v0, v1, v2, v3].
    # period0 (r=2) selects post-prepend indices [0,1,2] = [boundary, v0, v1].
    # period1 (r=4) selects post-prepend indices [2,3,4] = [v1, v2, v3].
    for var_idx, (base, offset) in enumerate(zip(channel_bases, boundary_offsets)):
        v = [base + t for t in range(4)]
        expected_period0 = ((base + offset) + v[0] + v[1]) / 3
        expected_period1 = (v[1] + v[2] + v[3]) / 3
        for period, expected in enumerate([expected_period0, expected_period1]):
            timevar_idx = period * len(variables) + var_idx
            slice_result = result[:, :, timevar_idx, :, :, :]
            assert torch.allclose(
                slice_result, torch.full_like(slice_result, expected)
            ), f"channel {var_idx} period {period}: expected {expected}"


def test_trailing_average_unseeded_raises():
    variables = ["c0", "c1"]
    coupler = TrailingAverageCoupler(
        dataset=_make_coupler_dataset(variables + ["x"], n_time=16),
        batch_size=1,
        variables=variables,
        presteps=0,
        averaging_window="6h",
        input_times=["6h", "12h"],
        input_time_dim=2,
        output_time_dim=2,
        use_inclusive_trailing_average=True,
    )
    mock_coupled_module = SimpleNamespace(output_variables=variables, time_step="3h")
    coupler.setup_coupling(mock_coupled_module)

    coupled_fields = torch.rand(
        1, coupler.spatial_dims[0], 4, len(variables),
        coupler.spatial_dims[1], coupler.spatial_dims[2],
    )
    with pytest.raises(RuntimeError, match="no boundary state has been"):
        coupler.set_coupled_fields(coupled_fields)


def test_trailing_average_legacy_default_and_opt_in_guards():
    """Default (opt-in flag unset) uses the legacy disjoint-block formula, and
    the opt-in-only APIs (seed_boundary_state, variable_strides) guard against
    misuse on a legacy coupler."""
    variables = ["c0", "c1"]
    input_times = ["6h", "12h"]
    batch_size = 2
    coupler = TrailingAverageCoupler(
        dataset=_make_coupler_dataset(variables + ["x"], n_time=16),
        batch_size=batch_size,
        variables=variables,
        presteps=0,
        averaging_window="6h",
        input_times=input_times,
        input_time_dim=2,
        output_time_dim=2,
    )
    assert coupler.use_inclusive_trailing_average is False

    data_time_step = "3h"
    dt = pd.Timedelta(data_time_step)
    averaging_window_max_indices = [pd.Timedelta(t) // dt for t in input_times]
    di = averaging_window_max_indices[0]
    # Original/legacy disjoint-block formula (no boundary prepend).
    averaging_slices = []
    for j in range(coupler.coupled_integration_dim):
        averaging_slices.append([])
        for i, r in enumerate(averaging_window_max_indices):
            averaging_slices[j].append(
                slice(
                    coupler.input_time_dim * j * di + i * di,
                    coupler.input_time_dim * j * di + r,
                )
            )
    coupler.averaging_slices = averaging_slices
    coupler.coupled_channel_indices = list(range(len(variables)))

    coupled_fields = torch.rand(
        batch_size,
        coupler.spatial_dims[0],
        4,
        len(variables),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    coupler.set_coupled_fields(coupled_fields)
    assert coupler.coupled_mode

    with pytest.raises(RuntimeError, match="use_inclusive_trailing_average=True"):
        coupler.seed_boundary_state(coupled_fields)

    with pytest.raises(ValueError, match="use_inclusive_trailing_average=True"):
        TrailingAverageCoupler(
            dataset=_make_coupler_dataset(variables + ["x"], n_time=16),
            batch_size=batch_size,
            variables=variables,
            presteps=0,
            averaging_window="6h",
            input_times=input_times,
            input_time_dim=2,
            output_time_dim=2,
            variable_strides={"c0": "24h"},
        )


def test_trailing_average_variable_strides_subsample_independently():
    """Per-variable `variable_strides` must select each channel's own raw
    indices within a shared `averaging_window`, matching
    `compute_trailing_mean.py`'s per-variable `coupled_dt` semantics."""
    variables = ["z1000", "ttr"]
    coupler = TrailingAverageCoupler(
        dataset=_make_coupler_dataset(variables + ["x"], n_time=32),
        batch_size=1,
        variables=variables,
        presteps=0,
        averaging_window="96h",
        input_times=["96h"],
        input_time_dim=2,
        output_time_dim=2,
        use_inclusive_trailing_average=True,
        variable_strides={"z1000": "24h"},  # ttr defaults to coupled_module.time_step (6h)
    )
    mock_coupled_module = SimpleNamespace(output_variables=variables, time_step="6h")
    coupler.setup_coupling(mock_coupled_module)
    assert coupler.coupled_channel_indices == [0, 1]

    n_raw = 16  # 96h / 6h
    z1000_boundary, ttr_boundary = 500.0, 2500.0
    boundary = torch.empty(
        1, coupler.spatial_dims[0], 1, len(variables),
        coupler.spatial_dims[1], coupler.spatial_dims[2],
    )
    boundary[:, :, 0, 0, :, :] = z1000_boundary
    boundary[:, :, 0, 1, :, :] = ttr_boundary
    coupler.seed_boundary_state(boundary)

    coupled_fields = torch.empty(
        1,
        coupler.spatial_dims[0],
        n_raw,
        len(variables),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    z1000_raw = [1000.0 + 10.0 * t for t in range(n_raw)]
    ttr_raw = [5000.0 + 7.0 * t for t in range(n_raw)]
    for t in range(n_raw):
        coupled_fields[:, :, t, 0, :, :] = z1000_raw[t]
        coupled_fields[:, :, t, 1, :, :] = ttr_raw[t]

    coupler.set_coupled_fields(coupled_fields)
    result = coupler.construct_integrated_couplings()

    # window_steps = 96h/6h = 16. z1000 stride ratio = 24h/6h = 4 -> indices
    # [0, 4, 8, 12, 16] into the boundary-prepended buffer ([boundary] + raw[0:16]).
    # ttr stride ratio = 1 -> every index [0..16].
    z1000_selected = [z1000_boundary, z1000_raw[3], z1000_raw[7], z1000_raw[11], z1000_raw[15]]
    expected_z1000 = sum(z1000_selected) / len(z1000_selected)
    ttr_selected = [ttr_boundary] + ttr_raw
    expected_ttr = sum(ttr_selected) / len(ttr_selected)

    got_z1000 = result[0, :, 0, :, :, :]
    got_ttr = result[0, :, 1, :, :, :]
    assert torch.allclose(got_z1000, torch.full_like(got_z1000, expected_z1000), atol=1e-3)
    assert torch.allclose(got_ttr, torch.full_like(got_ttr, expected_ttr), atol=1e-3)
