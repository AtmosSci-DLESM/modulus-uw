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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from physicsnemo.datapipes.healpix.couplers import ConstantCoupler


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("times", [1, 2, 4])
@pytest.mark.parametrize("scaled", [False, True])
def test_final_state_preserves_channels_scaling_feedback_and_gradients(
    batch_size, times, scaled
):
    """Only the final source time reaches every recipient slot and its gradient."""
    dataset = xr.Dataset(
        {
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                np.zeros((2, 3, 2, 1, 2), dtype=np.float32),
            )
        },
        coords={"channel_in": ["sic", "unused", "sst"]},
    )
    coupler = ConstantCoupler(
        dataset,
        batch_size,
        ["sst", "sic"],
        input_time_dim=1,
        output_time_dim=32,
        input_times=["0h"],
        presteps=1,
    )
    coupler.setup_coupling(SimpleNamespace(output_variables=["sic", "unused", "sst"]))
    if scaled:
        coupler.set_scaling(
            xr.Dataset(
                {"mean": ("index", [280.0, 0.1]), "std": ("index", [10.0, 0.5])},
                coords={"index": ["sst", "sic"]},
            )
        )
        coupler.set_coupled_scaling(
            {"sst": {"mean": 270.0, "std": 2.0}, "sic": {"mean": 0.2, "std": 0.1}}
        )
    state = torch.arange(batch_size * 2 * times * 3 * 2, dtype=torch.float32)
    state = state.reshape(batch_size, 2, times, 3, 1, 2).requires_grad_()
    before = state.detach().clone()
    coupler.set_coupled_fields(state)
    expected = before[:, :, -1:, [2, 0]]
    if scaled:
        expected = (
            expected * torch.tensor([2.0, 0.1]).reshape(1, 1, 1, 2, 1, 1)
            + torch.tensor([270.0, 0.2]).reshape(1, 1, 1, 2, 1, 1)
            - torch.tensor([280.0, 0.1]).reshape(1, 1, 1, 2, 1, 1)
        )
        expected = expected / torch.tensor([10.0, 0.5]).reshape(1, 1, 1, 2, 1, 1)
    expected = expected.expand(-1, -1, 33, -1, -1, -1).permute(2, 0, 3, 1, 4, 5)
    actual = coupler.construct_integrated_couplings()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, before)
    actual.sum().backward()
    assert torch.count_nonzero(state.grad[:, :, :-1]) == 0
    assert torch.count_nonzero(state.grad[:, :, :, 1]) == 0
    assert torch.all(state.grad[:, :, -1, [0, 2]] > 0)
    with torch.no_grad():
        state.add_(1000)
    torch.testing.assert_close(actual, expected)
