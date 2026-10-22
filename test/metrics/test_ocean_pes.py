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

"""Ocean masking preserves AS PES while excluding invalid patch components."""
from types import SimpleNamespace

import pytest
import torch

from physicsnemo.metrics.climate.healpix_loss import PatchedEnergyScoreLoss


def make_loss(members=2, patch=3, mode="karlbauer", sigma=None, device="cpu"):
    """Construct the real PES with small native HEALPix faces."""
    loss = PatchedEnergyScoreLoss(
        weights=[1.0, 1.0],
        n_members=members,
        patch_size=patch,
        hpx_padding_mode=mode,
        enable_nhwc=False,
        nside=8,
        patch_weight_sigma=sigma,
    )
    loss.setup(SimpleNamespace(output_variables=["a", "b"], device=device))
    return loss


@pytest.mark.parametrize("members", [2, 3])
@pytest.mark.parametrize("mode", ["karlbauer", "isolatitude"])
@pytest.mark.parametrize("sigma", [None, 1.0])
def test_all_ocean_matches_as_value_and_gradient(members, mode, sigma):
    torch.manual_seed(7)
    loss = make_loss(members, mode=mode, sigma=sigma)
    target = torch.randn(2, 12, 2, 2, 8, 8)
    prediction = torch.randn(members * 2, 12, 2, 2, 8, 8, requires_grad=True)
    sums, counts = loss.masked_channel_sums(prediction, target, torch.ones_like(target))
    actual = (sums / counts).mean()
    expected = loss(prediction, target)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_missing_targets_and_land_do_not_change_score_or_receive_gradients():
    loss = make_loss()
    target = torch.zeros(1, 12, 1, 2, 8, 8)
    mask = torch.ones_like(target)
    mask[..., :3, :3] = 0
    target[..., 4, 4] = float("nan")
    prediction = torch.ones(2, 12, 1, 2, 8, 8, requires_grad=True)
    with torch.no_grad():
        prediction[..., :3, :3] = float("nan")
        prediction[..., 4, 4] = float("inf")
    sums, counts = loss.masked_channel_sums(prediction, target, mask)
    actual = (sums / counts).mean()
    # A constant unit error still has a full 3x3 norm of 3 at coastal patches.
    torch.testing.assert_close(actual, torch.tensor(1.5))
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert not prediction.grad[..., :3, :3].any()
    assert not prediction.grad[..., 4, 4].any()
    assert prediction.grad[..., 5, 5].abs().sum() > 0


def test_fully_invalid_channel_zero_counts_and_finite_zero_gradient():
    loss = make_loss()
    target = torch.full((1, 12, 1, 2, 8, 8), float("nan"))
    prediction = torch.ones(2, 12, 1, 2, 8, 8, requires_grad=True)
    sums, counts = loss.masked_channel_sums(prediction, target, torch.ones_like(target))
    assert not counts.any() and not sums.any()
    sums.sum().backward()
    assert torch.isfinite(prediction.grad).all() and not prediction.grad.any()


def test_nonfinite_prediction_on_valid_ocean_fails():
    loss = make_loss()
    target = torch.zeros(1, 12, 1, 2, 8, 8)
    prediction = torch.ones(2, 12, 1, 2, 8, 8)
    prediction[..., 4, 4] = float("nan")
    with pytest.raises(FloatingPointError, match="valid ocean"):
        loss.masked_channel_sums(prediction, target, torch.ones_like(target))


def test_partitioned_batches_pool_counts_before_ratios():
    loss = make_loss()
    torch.manual_seed(13)
    target = torch.randn(3, 12, 1, 2, 8, 8)
    target[1, ..., :4, :] = float("nan")
    prediction = torch.randn(2, *target.shape)
    whole = loss.masked_channel_sums(
        prediction.flatten(0, 1), target, torch.ones_like(target)
    )
    pieces = [
        loss.masked_channel_sums(
            prediction[:, i : i + 1].flatten(0, 1),
            target[i : i + 1],
            torch.ones_like(target[i : i + 1]),
        )
        for i in range(3)
    ]
    torch.testing.assert_close(whole[0], sum(p[0] for p in pieces))
    torch.testing.assert_close(whole[1], sum(p[1] for p in pieces))
