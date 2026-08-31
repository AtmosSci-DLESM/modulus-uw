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

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail, nfsdata_or_fail

from physicsnemo.metrics.climate.healpix_loss import (
    BaseMSE,
    OceanMSE,
    WeightedMSE,
    WeightedOceanMSE,
    WeightedCRPSLoss,
    PatchedEnergyScoreLoss,
    GlobalEnergyScoreLoss,
    WeightedCompositeLoss,
)

xr = pytest.importorskip("xarray")


@pytest.fixture
def test_data():
    # create dummy data

    # We'll pretend h,w are a lat lon grid instead of healpix
    # so the test works the same
    # Set lat/lon in terms of degrees (for use with _compute_lat_weights)
    def generate_test_data(channels=2, img_shape=(768, 768)):
        x = np.linspace(-180, 180, img_shape[1], dtype=np.float32)
        y = np.linspace(-90, 90, img_shape[0], dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        pred_tensor_np = np.cos(2 * np.pi * yv / (180))
        targ_tensor_np = np.cos(np.pi * yv / (180))

        return channels, pred_tensor_np, targ_tensor_np

    return generate_test_data


@dataclass
class trainer_helper:
    """helper class for setup with the MSE testers"""

    output_variables: Sequence
    device: str


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_BaseMSE(device, test_data, rtol: float = 1e-3, atol: float = 1e-3):
    mse_func = BaseMSE()
    mse_func.setup(None)  # for coverage
    channels, pred_tensor_np, targ_tensor_np = test_data()

    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    # test for insufficient dimensions
    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(torch.zeros((10,), device=device), targ_tensor)

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(targ_tensor, torch.zeros((10,), device=device))

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(torch.zeros((10,), device=device), torch.zeros((10,), device=device))

    # test for 0 loss
    error = mse_func(targ_tensor, targ_tensor)
    assert torch.allclose(
        error,
        torch.zeros([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # int( cos(x)^2 - cos(2x)^2 )dx, x = 0...2*pi = pi/4
    # So MSE should be pi/4 / (pi) = 0.25
    error = mse_func(pred_tensor**2, targ_tensor**2)
    assert torch.allclose(
        error,
        0.25 * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # test for non averaged channesl
    # make the last channel of prediction and target the same
    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[0, 0, 0, -1, ...] = ones[...]
    targ_tensor[0, 0, 0, -1, ...] = ones[...]

    error = mse_func(pred_tensor**2, targ_tensor**2, average_channels=False)

    expected_err = 0.25 * torch.ones(error.shape, dtype=torch.float32, device=device)
    expected_err[-1] = 0

    assert torch.allclose(
        error,
        expected_err,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_WeightedMSE(device, test_data, rtol: float = 1e-3, atol: float = 1e-3):
    num_channels = 3
    channels, pred_tensor_np, targ_tensor_np = test_data(channels=num_channels)

    # first two channels will be as BaseMSE above, the last channel will be 0 loss
    # so per channel loss will be [0.25, 0.25, 0]
    # Giving the last channel half the weight results in a per loss of:
    # [0.25*0.5, 0.25*0.25, 0.0*0.25] == [0.125,0.0625,0]
    # and an average weighted loss of 0.0625
    channel_weights = [0.5, 0.25, 0.25]
    channel_weighted_mse = torch.Tensor([0.125, 0.0625, 0]).to(device)
    mean_weighted_mse = channel_weighted_mse.mean()
    weighted_mse_func = WeightedMSE(channel_weights)

    trainer = trainer_helper(
        output_variables=["a", "b", "ones"],
        device=device,
    )
    weighted_mse_func.setup(trainer)

    # test setup fail case if number of variables doesn't match number of weights
    trainer = trainer_helper(
        output_variables=["a", "b"],
        device=device,
    )
    with pytest.raises(
        ValueError, match="Length of outputs and loss_weights is not the same!"
    ):
        weighted_mse_func.setup(trainer)

    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    # make the last channel of prediction and target the same
    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[-1, ...] = ones[...]
    targ_tensor[-1, ...] = ones[...]

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    # test for insufficient dimensions
    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(torch.zeros((10,), device=device), targ_tensor)

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(targ_tensor, torch.zeros((10,), device=device))

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(
            torch.zeros((10,), device=device), torch.zeros((10,), device=device)
        )

    # test for 0 loss
    error = weighted_mse_func(pred_tensor**2, pred_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        torch.zeros(1).to(device),
        rtol=rtol,
        atol=atol,
    )

    # test for individual channel loss
    error = weighted_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=False
    )
    assert torch.allclose(
        error,
        channel_weighted_mse,
        rtol=rtol,
        atol=atol,
    )

    # test with mean across channels
    error = weighted_mse_func(pred_tensor**2, targ_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        mean_weighted_mse,
        rtol=rtol,
        atol=atol,
    )


@pytest.fixture
def data_dir():
    path = "/data/nfs/modulus-data/datasets/healpix/"
    return path


@pytest.fixture
def dataset_name():
    name = "healpix"
    return name


@nfsdata_or_fail
@import_or_fail("xarray")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_OceanMSE(
    data_dir,
    dataset_name,
    device,
    test_data,
    pytestconfig,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    num_channels = 3
    channels, pred_tensor_np, targ_tensor_np = test_data(
        channels=num_channels, img_shape=(32, 32)
    )
    ds_path = Path(data_dir, dataset_name + ".zarr")

    lsm_ds = xr.open_dataset(ds_path, engine="zarr").constants.sel({"channel_c": "lsm"})

    channel_ocean_mse = torch.Tensor([0.2706, 0.2706, 0]).to(device)
    mean_ocean_mse = channel_ocean_mse.mean()
    ocean_mse_func = OceanMSE(ds_path)

    trainer = trainer_helper(
        output_variables=["a", "b", "ones"],
        device=device,
    )
    lsm_tensor = 1 - torch.tensor(np.expand_dims(lsm_ds.values, (0, 2, 3))).to(
        trainer.device
    )
    ocean_mse_func.setup(trainer)

    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    # make the last channel of prediction and target the same
    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[-1, ...] = ones[...]
    targ_tensor[-1, ...] = ones[...]

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    pred_tensor = pred_tensor.expand(1, lsm_tensor.shape[1], 1, -1, -1, -1)
    targ_tensor = targ_tensor.expand(1, lsm_tensor.shape[1], 1, -1, -1, -1)

    # test for 0 loss
    error = ocean_mse_func(pred_tensor**2, pred_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        torch.zeros(1).to(device),
        rtol=rtol,
        atol=atol,
    )

    # test for individual channels
    error = ocean_mse_func(pred_tensor**2, targ_tensor**2, average_channels=False)
    assert torch.allclose(
        error,
        channel_ocean_mse,
        rtol=rtol,
        atol=atol,
    )

    # test for mean across channels
    error = ocean_mse_func(pred_tensor**2, targ_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        mean_ocean_mse,
        rtol=rtol,
        atol=atol,
    )


@nfsdata_or_fail
@import_or_fail("xarray")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_WeightedOceanMSE(
    data_dir,
    dataset_name,
    device,
    test_data,
    pytestconfig,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    num_channels = 3
    identity_weights = [1, 1, 1]  # same as OceanMSE
    test_weights = [2.0, 0.5, 1]  # Check positive and negative weighing factors
    test_weights_tensor = torch.Tensor(test_weights).to(device)
    channels, pred_tensor_np, targ_tensor_np = test_data(
        channels=num_channels, img_shape=(32, 32)
    )
    ds_path = Path(data_dir, dataset_name + ".zarr")

    lsm_ds = xr.open_dataset(ds_path, engine="zarr").constants.sel({"channel_c": "lsm"})

    # setup target weights
    atmos_channel_mse = torch.Tensor([0.2706, 0.2706, 0]).to(device)
    mean_atmos_channel_mse = atmos_channel_mse.mean()
    weighted_atmos_channel_mse = atmos_channel_mse * test_weights_tensor
    mean_weighted_atmos_channel_mse = weighted_atmos_channel_mse.mean()

    trainer = trainer_helper(
        output_variables=["a", "b", "ones"],
        device=device,
    )
    lsm_tensor = 1 - torch.tensor(np.expand_dims(lsm_ds.values, (0, 2, 3))).to(
        trainer.device
    )

    # expand to 4 dims C, F, H, W
    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    # make the last channel of prediction and target the same
    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[-1, ...] = ones[...]
    targ_tensor[-1, ...] = ones[...]

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    # fit to LSM field size
    pred_tensor = pred_tensor.expand(1, lsm_tensor.shape[1], 1, -1, -1, -1)
    targ_tensor = targ_tensor.expand(1, lsm_tensor.shape[1], 1, -1, -1, -1)

    # Test mismatch between weights and number of variables
    weighted_ocean_mse_func = WeightedOceanMSE(ds_path)
    with pytest.raises(
        ValueError, match="Length of outputs and loss_weights is not the same!"
    ):
        weighted_ocean_mse_func.setup(trainer)

    # Test with identity weights, same as OceanMSE
    weighted_ocean_mse_func = WeightedOceanMSE(ds_path, weights=identity_weights)
    weighted_ocean_mse_func.setup(trainer)

    # test for 0 loss
    error = weighted_ocean_mse_func(
        pred_tensor**2, pred_tensor**2, average_channels=True
    )
    assert torch.allclose(
        error,
        torch.zeros(1).to(device),
        rtol=rtol,
        atol=atol,
    )

    # test identity on individual channels
    error = weighted_ocean_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=False
    )
    assert torch.allclose(
        error,
        atmos_channel_mse,
        rtol=rtol,
        atol=atol,
    )

    # test identity on mean across channels
    error = weighted_ocean_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=True
    )
    assert torch.allclose(
        error,
        mean_atmos_channel_mse,
        rtol=rtol,
        atol=atol,
    )

    # Test with different weights
    weighted_ocean_mse_func = WeightedOceanMSE(ds_path, weights=test_weights)
    weighted_ocean_mse_func.setup(trainer)

    # test identity on individual channels
    error = weighted_ocean_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=False
    )
    assert torch.allclose(
        error,
        weighted_atmos_channel_mse,
        rtol=rtol,
        atol=atol,
    )

    # test identity on mean across channels
    error = weighted_ocean_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=True
    )
    assert torch.allclose(
        error,
        mean_weighted_atmos_channel_mse,
        rtol=rtol,
        atol=atol,
    )


def test_WeightedCRPSLoss_rejects_zero_members():
    with pytest.raises(ValueError, match="n_members must be at least 1"):
        WeightedCRPSLoss(weights=[1.0], n_members=0)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_WeightedCRPSLoss_one_member_zero_and_matches_mae(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 2, 3, 16, 16
    n_members = 1
    weights = [1.0, 2.0, 0.5]

    loss_fn = WeightedCRPSLoss(weights=weights, n_members=n_members)
    trainer = trainer_helper(
        output_variables=[f"var{i}" for i in range(c)], device=device
    )
    loss_fn.setup(trainer)

    # Perfect forecast → 0
    target = torch.randn(b, f, t, c, h, w, device=device)
    prediction = target.clone()
    loss_perfect = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss_perfect, torch.tensor(0.0, device=device), atol=1e-6)

    # Random pred: mean |w * (pred - target)| with no spread term
    torch.manual_seed(1)
    target = torch.randn(b, f, t, c, h, w, device=device)
    pred = torch.randn(n_members, b, f, t, c, h, w, device=device)
    # Clone: WeightedCRPSLoss multiplies channel weights into tensors in-place
    prediction = pred.reshape(n_members * b, f, t, c, h, w).clone()
    target_in = target.clone()

    loss = loss_fn(prediction, target_in, average_channels=True)

    w = torch.tensor(weights, device=device, dtype=torch.float32)
    expected = (
        (pred.squeeze(0) * w[None, None, None, :, None, None]
         - target * w[None, None, None, :, None, None])
        .abs()
        .mean()
    )
    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("patch_size", [3, 5])
@pytest.mark.parametrize("hpx_padding_mode", ["earth2grid", "karlbauer", "isolatitude"])
@pytest.mark.parametrize("enable_nhwc", [True, False])
@pytest.mark.parametrize("patch_weight_sigma", [None, 1.0])
def test_PatchedEnergyScoreLoss_two_members_zero_and_symmetry(device, patch_size, hpx_padding_mode, enable_nhwc, patch_weight_sigma):
    if hpx_padding_mode == "earth2grid" and (not torch.cuda.is_available() or enable_nhwc):
        pytest.skip(
            f"hpx_padding_mode=earth2grid requires CUDA, but CUDA is not available or enable_nhwc is True. "
            f"Got hpx_padding_mode={hpx_padding_mode}, enable_nhwc={enable_nhwc}"
        )
    
    # Small toy data: B=1,F=1,T=1,C=2,H=4,W=4
    b, f, t, c, h, w = 2, 12, 4, 4, 64, 64
    n_members = 2
    weights = [1.0] * c

    loss_fn = PatchedEnergyScoreLoss(
        weights=weights,
        n_members=n_members,
        patch_size=patch_size,
        hpx_padding_mode=hpx_padding_mode,
        enable_nhwc=enable_nhwc,
        nside=h,
        patch_weight_sigma=patch_weight_sigma,
    )
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)
    loss_fn.setup(trainer)

    base = torch.arange(h * w, dtype=torch.float32, device=device).reshape(h, w)
    target = base.repeat(b, f, t, c, 1, 1)  # [B,F,T,C,H,W]

    # Perfect ensemble: both members equal to target
    pred_members = target.repeat(n_members, 1, 1, 1, 1, 1)  # [Cond,B,F,T,C,H,W]
    prediction = pred_members.reshape(n_members * b, f, t, c, h, w)

    print(prediction.shape, target.shape)

    loss = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("patch_size", [3])
@pytest.mark.parametrize("hpx_padding_mode", ["karlbauer"])
@pytest.mark.parametrize("enable_nhwc", [False])
def test_PatchedEnergyScoreLoss_two_member_fast_matches_pairwise(
    device, patch_size, hpx_padding_mode, enable_nhwc
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 2, 2, 64, 64
    n_members = 2
    loss_fn = PatchedEnergyScoreLoss(
        weights=[1.0] * c,
        n_members=n_members,
        alpha=0.95,
        patch_size=patch_size,
        hpx_padding_mode=hpx_padding_mode,
        enable_nhwc=enable_nhwc,
        nside=h,
    )
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)
    loss_fn.setup(trainer)

    torch.manual_seed(0)
    target = torch.randn(b, f, t, c, h, w, device=device)
    pred = torch.randn(n_members, b, f, t, c, h, w, device=device)

    fast = loss_fn._energy_score_field(pred, target, n_members, b, f, t, c, h, w).mean()

    with torch.no_grad():
        tar_unfold = loss_fn._unfold_target(target)
    pred_unfold = loss_fn._unfold_prediction_members(pred)
    diff_to_target = loss_fn._reshape_member_norms(
        loss_fn._weighted_patch_norm(pred_unfold - tar_unfold.unsqueeze(0), patch_dim=-2),
        b, f, t, c, h, w,
    )
    diff_i = diff_to_target
    pred_i = pred_unfold.unsqueeze(1)
    pred_j = pred_unfold.unsqueeze(0)
    dist_ensemble = loss_fn._weighted_patch_norm(pred_i - pred_j, patch_dim=-2)
    dist_ensemble = dist_ensemble.view(n_members, n_members, b, t, f, c, h, w).permute(
        0, 1, 2, 4, 3, 5, 6, 7
    )
    mask = loss_fn.diag_mask[:, :, None, None, None, None, None, None]
    diff_terms = mask * (diff_i.unsqueeze(0) + diff_i.unsqueeze(1))
    dist_terms = mask * dist_ensemble
    pairwise = (
        loss_fn.averaging_coeff * (diff_terms - loss_fn.coeff_eps * dist_terms).sum(dim=(0, 1))
    ).mean()

    assert torch.isclose(fast, pairwise, rtol=1e-5, atol=1e-5)


def test_PatchedEnergyScoreLoss_rejects_zero_members():
    with pytest.raises(ValueError, match="n_members must be at least 1"):
        PatchedEnergyScoreLoss(weights=[1.0], n_members=0)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("patch_weight_sigma", [None, 1.0])
@pytest.mark.parametrize("n_members", [1, 2])
def test_PatchedEnergyScoreLoss_zero_residual_finite_grad(
    device, patch_weight_sigma, n_members
):
    """vector_norm(v⊙√w) keeps finite zero grads at identical-member residuals."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 1, 2, 16, 16
    loss_fn = PatchedEnergyScoreLoss(
        weights=[1.0] * c,
        n_members=n_members,
        patch_size=3,
        hpx_padding_mode="karlbauer",
        enable_nhwc=False,
        nside=h,
        patch_weight_sigma=patch_weight_sigma,
    )
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)
    loss_fn.setup(trainer)

    # Perfect / identical-member forecast → exact 0 loss, finite grads
    target = torch.randn(b, f, t, c, h, w, device=device)
    pred_members = target.unsqueeze(0).expand(n_members, -1, -1, -1, -1, -1, -1).contiguous()
    prediction = pred_members.reshape(n_members * b, f, t, c, h, w).detach().requires_grad_(True)
    loss = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)
    loss.backward()
    assert prediction.grad is not None
    assert torch.all(torch.isfinite(prediction.grad))

    # Direct zero patch residual through the norm helper
    diff = torch.zeros(2, 9, 8, device=device, requires_grad=True)
    norms = loss_fn._weighted_patch_norm(diff, patch_dim=-2)
    norms.sum().backward()
    assert diff.grad is not None
    assert torch.all(torch.isfinite(diff.grad))
    assert torch.allclose(diff.grad, torch.zeros_like(diff.grad), atol=1e-6)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("patch_size", [3])
@pytest.mark.parametrize("hpx_padding_mode", ["karlbauer", "isolatitude"])
@pytest.mark.parametrize("enable_nhwc", [False])
@pytest.mark.parametrize("patch_weight_sigma", [None, 1.0])
def test_PatchedEnergyScoreLoss_one_member_zero_and_matches_patch_mae(
    device, patch_size, hpx_padding_mode, enable_nhwc, patch_weight_sigma
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 2, 2, 64, 64
    n_members = 1
    weights = [1.0] * c

    loss_fn = PatchedEnergyScoreLoss(
        weights=weights,
        n_members=n_members,
        patch_size=patch_size,
        hpx_padding_mode=hpx_padding_mode,
        enable_nhwc=enable_nhwc,
        nside=h,
        patch_weight_sigma=patch_weight_sigma,
    )
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)
    loss_fn.setup(trainer)

    # Perfect forecast → 0
    target = torch.randn(b, f, t, c, h, w, device=device)
    prediction = target.clone()
    loss_perfect = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss_perfect, torch.tensor(0.0, device=device), atol=1e-6)

    # Random pred: equals mean of weighted patch norms vs target (no spread term)
    torch.manual_seed(1)
    target = torch.randn(b, f, t, c, h, w, device=device)
    pred = torch.randn(n_members, b, f, t, c, h, w, device=device)
    prediction = pred.reshape(n_members * b, f, t, c, h, w)

    loss = loss_fn(prediction, target, average_channels=True)

    with torch.no_grad():
        tar_unfold = loss_fn._unfold_target(target)
    pred_unfold = loss_fn._unfold_prediction_members(pred)
    expected = loss_fn._reshape_member_norms(
        loss_fn._weighted_patch_norm(
            pred_unfold - tar_unfold.unsqueeze(0), patch_dim=-2
        ),
        b,
        f,
        t,
        c,
        h,
        w,
    ).squeeze(0).mean()

    assert torch.isclose(loss, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("patch_size", [3, 5])
def test_PatchedEnergyScoreLoss_default_uniform_weights_mae_scale(device, patch_size):
    """Default (no sigma) uses uniform 1/D weights → flat residual scores |ε| like MAE."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 1, 12, 1, 1, 16, 16
    D = patch_size ** 2
    loss_fn = PatchedEnergyScoreLoss(
        weights=[1.0],
        n_members=1,
        patch_size=patch_size,
        hpx_padding_mode="karlbauer",
        enable_nhwc=False,
        nside=h,
        patch_weight_sigma=None,
    )
    trainer = trainer_helper(output_variables=["var0"], device=device)
    loss_fn.setup(trainer)

    assert loss_fn.patch_weights is not None
    assert torch.allclose(
        loss_fn.patch_weights.sum(), torch.tensor(1.0, device=device), atol=1e-6
    )
    assert torch.allclose(
        loss_fn.patch_weights,
        torch.full_like(loss_fn.patch_weights, 1.0 / D),
        atol=1e-6,
    )

    eps = 0.5
    target = torch.zeros(b, f, t, c, h, w, device=device)
    prediction = torch.full_like(target, eps)
    loss = loss_fn(prediction, target, average_channels=True)
    # Uniform 1/D: sqrt(mean(r^2)) = |eps|; also equals ||r||_2 / sqrt(D)
    assert torch.isclose(
        loss, torch.tensor(eps, device=device, dtype=loss.dtype), rtol=1e-5, atol=1e-5
    )

    diff = torch.full((1, D, 4), eps, device=device)
    weighted = loss_fn._weighted_patch_norm(diff, patch_dim=-2)
    unnorm = torch.linalg.vector_norm(diff, dim=-2)
    assert torch.allclose(weighted, unnorm / (D ** 0.5), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("patch_size", [3, 5])
@pytest.mark.parametrize("hpx_padding_mode", ["earth2grid", "karlbauer", "isolatitude"])
@pytest.mark.parametrize("enable_nhwc", [True, False])
@pytest.mark.parametrize("patch_weight_sigma", [None, 1.0])
def test_PatchedEnergyScoreLoss_three_members_zero(device, patch_size, hpx_padding_mode, enable_nhwc, patch_weight_sigma):
    if hpx_padding_mode == "earth2grid" and (not torch.cuda.is_available() or enable_nhwc):
        pytest.skip(
            f"hpx_padding_mode=earth2grid requires CUDA, but CUDA is not available or enable_nhwc is True. "
            f"Got hpx_padding_mode={hpx_padding_mode}, enable_nhwc={enable_nhwc}"
        )

    # Ensure n_members>2 path executes and yields ~0 for perfect forecasts
    b, f, t, c, h, w = 2, 12, 4, 4, 64, 64
    n_members = 3
    weights = [1.0] * c

    loss_fn = PatchedEnergyScoreLoss(
        weights=weights,
        n_members=n_members,
        patch_size=patch_size,
        hpx_padding_mode=hpx_padding_mode,
        enable_nhwc=enable_nhwc,
        nside=h,
        patch_weight_sigma=patch_weight_sigma,
    )
    trainer = trainer_helper(output_variables=[f'var{i}' for i in range(c)], device=device)
    loss_fn.setup(trainer)

    base = torch.randn(h, w, dtype=torch.float32, device=device)
    target = base.view(1, 1, 1, 1, h, w).repeat(b, f, t, c, 1, 1)

    pred_members = target.repeat(n_members, 1, 1, 1, 1, 1)
    prediction = pred_members.reshape(n_members * b, f, t, c, h, w)

    loss = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_GlobalEnergyScoreLoss_two_members_zero(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 4, 4, 64, 64
    n_members = 2
    weights = [1.0] * c

    loss_fn = GlobalEnergyScoreLoss(weights=weights, n_members=n_members)
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)
    loss_fn.setup(trainer)

    base = torch.arange(h * w, dtype=torch.float32, device=device).reshape(h, w)
    target = base.repeat(b, f, t, c, 1, 1)

    pred_members = target.unsqueeze(0).expand(n_members, -1, -1, -1, -1, -1, -1)
    prediction = pred_members.reshape(n_members * b, f, t, c, h, w)

    loss = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_GlobalEnergyScoreLoss_three_members_zero(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 4, 4, 64, 64
    n_members = 3
    weights = [1.0] * c

    loss_fn = GlobalEnergyScoreLoss(weights=weights, n_members=n_members)
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)
    loss_fn.setup(trainer)

    base = torch.randn(h, w, dtype=torch.float32, device=device)
    target = base.view(1, 1, 1, 1, h, w).repeat(b, f, t, c, 1, 1)
    pred_members = target.unsqueeze(0).expand(n_members, -1, -1, -1, -1, -1, -1)
    prediction = pred_members.reshape(n_members * b, f, t, c, h, w)

    loss = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_GlobalEnergyScoreLoss_two_member_fast_matches_pairwise(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 2, 2, 64, 64
    n_members = 2
    loss_fn = GlobalEnergyScoreLoss(
        weights=[1.0] * c,
        n_members=n_members,
        alpha=0.95,
    )
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)
    loss_fn.setup(trainer)

    torch.manual_seed(0)
    target = torch.randn(b, f, t, c, h, w, device=device)
    pred = torch.randn(n_members, b, f, t, c, h, w, device=device)

    fast = loss_fn._energy_score_channels(pred, target, n_members, b, f, t, c, h, w).mean()

    pred_flat = pred.permute(0, 1, 3, 4, 2, 5, 6).reshape(n_members, b * t * c, f * h * w)
    tar_flat = target.permute(0, 2, 3, 1, 4, 5).reshape(b * t * c, f * h * w)
    diff_to_target = torch.linalg.vector_norm(
        pred_flat - tar_flat.unsqueeze(0), dim=-1
    )
    explicit = loss_fn.averaging_coeff * (
        diff_to_target.sum(dim=0)
        - loss_fn.coeff_eps
        * torch.linalg.vector_norm(pred_flat[0] - pred_flat[1], dim=-1)
    ).mean()

    assert torch.isclose(fast, explicit, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_GlobalEnergyScoreLoss_channel_chunk_matches_full(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 2, 6, 64, 64
    n_members = 2
    torch.manual_seed(1)
    target = torch.randn(b, f, t, c, h, w, device=device)
    pred = torch.randn(n_members, b, f, t, c, h, w, device=device)
    prediction = pred.reshape(n_members * b, f, t, c, h, w)

    full_fn = GlobalEnergyScoreLoss(weights=[1.0] * c, n_members=n_members)
    chunk_fn = GlobalEnergyScoreLoss(
        weights=[1.0] * c, n_members=n_members, channel_chunk_size=2
    )
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)
    full_fn.setup(trainer)
    chunk_fn.setup(trainer)

    full_loss = full_fn(prediction, target, average_channels=True)
    chunk_loss = chunk_fn(prediction, target, average_channels=True)
    assert torch.isclose(full_loss, chunk_loss, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_WeightedCompositeLoss_weighted_sum(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b, f, t, c, h, w = 2, 12, 2, 4, 64, 64
    n_members = 2
    weights = [1.0] * c
    trainer = trainer_helper(output_variables=[f"var{i}" for i in range(c)], device=device)

    pes = PatchedEnergyScoreLoss(
        weights=weights,
        n_members=n_members,
        patch_size=3,
        use_earth2grid_padding=False,
        enable_nhwc=False,
    )
    ges = GlobalEnergyScoreLoss(weights=weights, n_members=n_members)
    composite = WeightedCompositeLoss(weights=[0.9, 0.1], losses=[pes, ges])
    composite.setup(trainer)

    torch.manual_seed(2)
    target = torch.randn(b, f, t, c, h, w, device=device)
    pred = torch.randn(n_members, b, f, t, c, h, w, device=device)
    prediction = pred.reshape(n_members * b, f, t, c, h, w)

    combined = composite(prediction, target, average_channels=True)
    expected = 0.9 * pes(prediction, target, average_channels=True) + 0.1 * ges(
        prediction, target, average_channels=True
    )
    assert torch.isclose(combined, expected, rtol=1e-5, atol=1e-5)

    per_var = composite(prediction, target, average_channels=False)
    expected_per_var = 0.9 * pes(prediction, target, average_channels=False) + 0.1 * ges(
        prediction, target, average_channels=False
    )
    assert torch.allclose(per_var, expected_per_var, rtol=1e-5, atol=1e-5)