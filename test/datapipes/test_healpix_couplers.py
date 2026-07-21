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

from physicsnemo.datapipes.healpix.couplers import ConstantCoupler

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
