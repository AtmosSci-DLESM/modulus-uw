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
    """TrailingAverageCoupler must keep every coupled variable through averaging."""
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
    )
    coupler.coupled_channel_indices = list(range(len(variables)))

    data_time_step = "3h"
    averaging_window_max_indices = [
        pd.Timedelta(t) // pd.Timedelta(data_time_step) for t in input_times
    ]
    di = averaging_window_max_indices[0]
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

    channel_values = [100.0, 200.0, 300.0]
    coupled_fields = torch.empty(
        batch_size,
        coupler.spatial_dims[0],
        4,
        len(variables),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    for i, value in enumerate(channel_values):
        coupled_fields[:, :, :, i, :, :] = value

    coupler.set_coupled_fields(coupled_fields)
    result = coupler.construct_integrated_couplings()
    assert list(result.shape) == [
        coupler.coupled_integration_dim,
        batch_size,
        coupler.timevar_dim,
    ] + list(coupler.spatial_dims)

    for period in range(len(input_times)):
        for var_idx, value in enumerate(channel_values):
            timevar_idx = period * len(variables) + var_idx
            slice_result = result[:, :, timevar_idx, :, :, :]
            assert torch.allclose(slice_result, torch.full_like(slice_result, value))
