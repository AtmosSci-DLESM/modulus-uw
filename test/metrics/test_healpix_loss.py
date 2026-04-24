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
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_func
import torch.multiprocessing as mp
from pytest_utils import import_or_fail, nfsdata_or_fail

from physicsnemo.metrics.climate.healpix_loss import (
    BaseMSE,
    OceanMSE,
    PatchedEnergyScoreLoss,
    WeightedCRPSLoss,
    WeightedCRPSLossSpectral,
    WeightedMSE,
    WeightedOceanMSE,
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
    distributed_ensemble_loss: bool = False
    ensemble_group_size: int = 1
    ensemble_group: object = None
    ensemble_sharding_enabled: bool = False


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

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("patch_size", [3, 5])
@pytest.mark.parametrize("use_earth2grid_padding", [True, False])
@pytest.mark.parametrize("enable_nhwc", [True, False])
@pytest.mark.parametrize("patch_weight_sigma", [None, 1.0])
def test_PatchedEnergyScoreLoss_two_members_zero_and_symmetry(device, patch_size, use_earth2grid_padding, enable_nhwc, patch_weight_sigma):
    # Small toy data: B=1,F=12,T=1,C=2,H=4,W=4
    b, f, t, c, h, w = 2, 12, 4, 4, 64, 64
    n_members = 2
    weights = [1.0] * c

    loss_fn = PatchedEnergyScoreLoss(
        weights=weights,
        n_members=n_members,
        patch_size=patch_size,
        use_earth2grid_padding=use_earth2grid_padding,
        enable_nhwc=enable_nhwc,
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
@pytest.mark.parametrize("patch_size", [3, 5])
@pytest.mark.parametrize("use_earth2grid_padding", [True, False])
@pytest.mark.parametrize("enable_nhwc", [True, False])
@pytest.mark.parametrize("patch_weight_sigma", [None, 1.0])
def test_PatchedEnergyScoreLoss_three_members_zero(device, patch_size, use_earth2grid_padding, enable_nhwc, patch_weight_sigma):
    # Ensure n_members>2 path executes and yields ~0 for perfect forecasts
    b, f, t, c, h, w = 2, 12, 4, 4, 64, 64
    n_members = 3
    weights = [1.0] * c

    loss_fn = PatchedEnergyScoreLoss(
        weights=weights,
        n_members=n_members,
        patch_size=patch_size,
        use_earth2grid_padding=use_earth2grid_padding,
        enable_nhwc=enable_nhwc,
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


def _small_hpx_case(device: str = "cpu", n_members: int = 2):
    b, f, t, c, h, w = 2, 12, 2, 2, 64, 64
    target = torch.randn(b, f, t, c, h, w, device=device, dtype=torch.float32)
    prediction = target.repeat(n_members, 1, 1, 1, 1, 1)
    return prediction.reshape(n_members * b, f, t, c, h, w), target


def test_WeightedCRPSLoss_zero_for_perfect_forecast_cpu():
    pred, tar = _small_hpx_case(device="cpu", n_members=3)
    loss_fn = WeightedCRPSLoss(weights=[1.0, 1.0], n_members=3)
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    loss_fn.setup(trainer)
    loss = loss_fn(pred, tar, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    per_channel = loss_fn(pred, tar, average_channels=False)
    assert per_channel.shape == (2,)


def test_WeightedCRPSLoss_shape_checks():
    pred, tar = _small_hpx_case(device="cpu", n_members=2)
    loss_fn = WeightedCRPSLoss(weights=[1.0, 1.0], n_members=2)
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    loss_fn.setup(trainer)
    with pytest.raises((ValueError, RuntimeError)):
        loss_fn(pred[:, :, :, :, :-1, :], tar, average_channels=True)
    loss_fn_bad_members = WeightedCRPSLoss(weights=[1.0, 1.0], n_members=3)
    loss_fn_bad_members.setup(trainer)
    with pytest.raises(ValueError):
        loss_fn_bad_members(pred, tar, average_channels=True)


def test_WeightedCRPSLossSpectral_apply_sht_face_dim_validation_cpu():
    pred, _ = _small_hpx_case(device="cpu", n_members=2)
    loss_fn = WeightedCRPSLossSpectral(
        weights=[1.0, 1.0],
        n_members=2,
        lambda_spec=0.0,
        nside=64,
        lmax=3 * 64 - 1,
        mmax=3 * 64 - 1,
    )
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    try:
        loss_fn.setup(trainer)
    except Exception as exc:
        pytest.skip(f"Spectral setup unavailable in current env: {exc}")
    x = pred.view(2, 2, 12, 2, 2, 64, 64)
    with pytest.raises(ValueError, match="Shape of input tensor should be"):
        loss_fn._apply_sht(x, face_dim=3, return_abs=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="lambda_spec>0 spectral CRPS needs CUDA/SHTCUDA",
)
def test_WeightedCRPSLossSpectral_lambda_nonzero_cuda_smoke():
    # Basic GPU smoke test for lambda_spec > 0 branch (spectral term active).
    device = "cuda:0"
    b, f, t, c, h, w = 1, 12, 1, 2, 32, 32
    n_members = 2
    nside = 32
    lmax = mmax = 3 * nside - 1

    target = torch.randn(b, f, t, c, h, w, device=device, dtype=torch.float32)
    pred_members = torch.randn(
        n_members, b, f, t, c, h, w, device=device, dtype=torch.float32
    )
    prediction = pred_members.reshape(n_members * b, f, t, c, h, w)

    loss_fn_lambda0 = WeightedCRPSLossSpectral(
        weights=[1.0, 1.0],
        n_members=n_members,
        lambda_spec=0.0,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
    )
    loss_fn_lambda_nonzero = WeightedCRPSLossSpectral(
        weights=[1.0, 1.0],
        n_members=n_members,
        lambda_spec=0.3,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
    )

    trainer = trainer_helper(output_variables=["v0", "v1"], device=device)
    loss_fn_lambda0.setup(trainer)
    loss_fn_lambda_nonzero.setup(trainer)

    out0 = loss_fn_lambda0(prediction, target, average_channels=True)
    out1 = loss_fn_lambda_nonzero(prediction, target, average_channels=True)

    assert torch.isfinite(out0)
    assert torch.isfinite(out1)
    # Non-zero lambda_spec should change the total loss compared to lambda_spec == 0
    assert not torch.allclose(out0, out1)


def test_WeightedCRPSLossSpectral_two_members_zero_lambda0_cpu():
    b, f, t, c, h, w = 2, 12, 2, 2, 32, 32
    n_members = 2
    nside = 32
    lmax = mmax = 3 * nside - 1

    target = torch.randn(b, f, t, c, h, w, dtype=torch.float32)
    prediction = target.repeat(n_members, 1, 1, 1, 1, 1).reshape(
        n_members * b, f, t, c, h, w
    )

    loss_fn = WeightedCRPSLossSpectral(
        weights=[1.0, 1.0],
        n_members=n_members,
        lambda_spec=0.0,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
    )
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    loss_fn.setup(trainer)

    loss = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_WeightedCRPSLossSpectral_three_members_zero_lambda0_cpu():
    b, f, t, c, h, w = 2, 12, 2, 2, 32, 32
    n_members = 3
    nside = 32
    lmax = mmax = 3 * nside - 1

    target = torch.randn(b, f, t, c, h, w, dtype=torch.float32)
    prediction = target.repeat(n_members, 1, 1, 1, 1, 1).reshape(
        n_members * b, f, t, c, h, w
    )

    loss_fn = WeightedCRPSLossSpectral(
        weights=[1.0, 1.0],
        n_members=n_members,
        lambda_spec=0.0,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
    )
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    loss_fn.setup(trainer)

    loss = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="lambda_spec>0 spectral CRPS needs CUDA/SHTCUDA",
)
def test_WeightedCRPSLossSpectral_two_members_zero_lambda_nonzero_cuda():
    device = "cuda:0"
    b, f, t, c, h, w = 2, 12, 2, 2, 32, 32
    n_members = 2
    nside = 32
    lmax = mmax = 3 * nside - 1

    target = torch.randn(b, f, t, c, h, w, device=device, dtype=torch.float32)
    prediction = target.repeat(n_members, 1, 1, 1, 1, 1).reshape(
        n_members * b, f, t, c, h, w
    )

    loss_fn = WeightedCRPSLossSpectral(
        weights=[1.0, 1.0],
        n_members=n_members,
        lambda_spec=0.3,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
    )
    trainer = trainer_helper(output_variables=["v0", "v1"], device=device)
    loss_fn.setup(trainer)

    loss = loss_fn(prediction, target, average_channels=True)
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)


def test_WeightedCRPSLossSpectral_lambda0_perfect_forecast_smoke():
    pred, tar = _small_hpx_case(device="cpu", n_members=2)
    loss_fn = WeightedCRPSLossSpectral(
        weights=[1.0, 1.0],
        n_members=2,
        lambda_spec=0.0,
        nside=64,
        lmax=3 * 64 - 1,
        mmax=3 * 64 - 1,
    )
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    try:
        loss_fn.setup(trainer)
    except Exception as exc:
        pytest.skip(f"Spectral setup unavailable in current env: {exc}")
    out = loss_fn(pred, tar, average_channels=True)
    assert torch.isfinite(out)


# --- Ensemble / distributed loss equivalence (multi-GPU) ---------------------------------
#
# We compare three operational modes. Outputs (forward tests) and parameter updates
# (weight-update tests) should agree across them where applicable.
#
# Mode A — Unsharded: all ensemble members on one process/GPU; loss is computed locally
#   in non-distributed mode (full prediction tensor, ``distributed_ensemble_loss=False``).
#
# Mode B — Sharded, gathered loss: ensemble members are split across processes; each rank
#   holds a local shard. Predictions are all_gather'd along the member dimension, then
#   the same non-distributed loss runs on the full tensor (simulates "gather then one-GPU
#   loss" without changing the loss code path to distributed). To ensure that gradient
#   updates are the same whether we use ensemble sharding or not, parameter gradients are
#   all-reduced with ``ReduceOp.AVG`` over the world group (same as ``loss / world_size``
#   then ``ReduceOp.SUM`` on gradients since loss is the same for all ranks).
#
# Mode C — Sharded, distributed loss: each rank keeps its shard; the loss uses
#   ``distributed_ensemble_loss=True`` and ensemble-group collectives (including a ring
#   exchange) so the result matches the global ensemble objective.


def _require_distributed_cuda(world_size: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for distributed probabilistic GPU tests")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"Need at least {world_size} GPUs, found {torch.cuda.device_count()}"
        )


def _init_cuda_process_group(rank: int, world_size: int, init_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _build_shared_random_tensors(rank: int, shape_target, shape_pred):
    if rank == 0:
        target = torch.randn(*shape_target, dtype=torch.float32, device=f"cuda:{rank}")
        pred_all = torch.randn(*shape_pred, dtype=torch.float32, device=f"cuda:{rank}")
    else:
        target = torch.zeros(*shape_target, dtype=torch.float32, device=f"cuda:{rank}")
        pred_all = torch.zeros(*shape_pred, dtype=torch.float32, device=f"cuda:{rank}")
    dist.broadcast(target, src=0, group=dist.group.WORLD)
    dist.broadcast(pred_all, src=0, group=dist.group.WORLD)
    return target, pred_all


def _crps_dist_matches_gathered_worker(
    rank: int,
    world_size: int,
    n_members: int,
    init_file: str,
    average_channels: bool,
    batch_size: int,
):
    _init_cuda_process_group(rank, world_size, init_file)
    device = f"cuda:{rank}"
    try:
        torch.manual_seed(2025)
        b = batch_size
        f, t, c, h, w = 12, 2, 2, 32, 32
        target, pred_all = _build_shared_random_tensors(
            rank,
            (b, f, t, c, h, w),
            (n_members, b, f, t, c, h, w),
        )

        loss_fn = WeightedCRPSLoss(weights=[1.0, 1.0], n_members=n_members)
        trainer = trainer_helper(
            output_variables=["v0", "v1"],
            device=device,
            distributed_ensemble_loss=True,
            ensemble_group_size=world_size,
            ensemble_group=dist.group.WORLD,
            ensemble_sharding_enabled=True,
        )
        loss_fn.setup(trainer)

        local_members = n_members // world_size
        start = rank * local_members
        stop = start + local_members
        local = pred_all[start:stop].reshape(local_members * b, f, t, c, h, w)
        # Mode C: sharded members, distributed loss (ring / ensemble collectives)
        dist_loss = loss_fn(
            local,
            target,
            average_channels=average_channels,
        )
        # Mode B: sharded members, all_gather full prediction, then non-distributed loss
        gathered = torch.cat(
            list(dist_nn_func.all_gather(local, group=dist.group.WORLD)), dim=0
        )
        loss_fn.distributed_ensemble_loss = False
        gather_loss = loss_fn(
            gathered,
            target,
            average_channels=average_channels,
        )
        # Mode A: unsharded (full member dim on one rank; reference)
        if rank == 0:
            pred_full = pred_all.reshape(n_members * b, f, t, c, h, w)
            ref = loss_fn(
                pred_full,
                target,
                average_channels=average_channels,
            )
        else:
            ref = torch.zeros_like(dist_loss)
        dist.broadcast(ref, src=0, group=dist.group.WORLD)
        assert torch.allclose(
            gather_loss,
            ref,
            rtol=1e-4,
            atol=1e-4,
        ), f"rank {rank}: gather_loss={gather_loss} ref={ref}"
        assert torch.allclose(
            dist_loss,
            ref,
            rtol=1e-4,
            atol=1e-4,
        ), f"rank {rank}: dist_loss={dist_loss} ref={ref}"
    finally:
        dist.destroy_process_group()


def _spectral_crps_dist_matches_gathered_worker(
    rank: int,
    world_size: int,
    n_members: int,
    init_file: str,
    average_channels: bool,
    batch_size: int,
    lambda_spec: float,
):
    _init_cuda_process_group(rank, world_size, init_file)
    device = f"cuda:{rank}"
    try:
        torch.manual_seed(2025)
        b = batch_size
        f, t, c, h, w = 12, 2, 2, 32, 32
        nside = 32
        lmax = mmax = 3 * nside - 1
        target, pred_all = _build_shared_random_tensors(
            rank,
            (b, f, t, c, h, w),
            (n_members, b, f, t, c, h, w),
        )

        loss_fn = WeightedCRPSLossSpectral(
            weights=[1.0, 1.0],
            n_members=n_members,
            lambda_spec=lambda_spec,
            nside=nside,
            lmax=lmax,
            mmax=mmax,
        )
        trainer = trainer_helper(
            output_variables=["v0", "v1"],
            device=device,
            distributed_ensemble_loss=True,
            ensemble_group_size=world_size,
            ensemble_group=dist.group.WORLD,
            ensemble_sharding_enabled=True,
        )
        loss_fn.setup(trainer)

        local_members = n_members // world_size
        start = rank * local_members
        stop = start + local_members
        local = pred_all[start:stop].reshape(local_members * b, f, t, c, h, w)
        # Mode C: sharded, distributed loss
        dist_loss = loss_fn(
            local,
            target,
            average_channels=average_channels,
        )
        # Mode B: sharded, gather then non-distributed loss
        gathered = torch.cat(
            list(dist_nn_func.all_gather(local, group=dist.group.WORLD)), dim=0
        )
        loss_fn.distributed_ensemble_loss = False
        gather_loss = loss_fn(
            gathered,
            target,
            average_channels=average_channels,
        )
        # Mode A: unsharded reference
        if rank == 0:
            pred_full = pred_all.reshape(n_members * b, f, t, c, h, w)
            ref = loss_fn(
                pred_full,
                target,
                average_channels=average_channels,
            )
        else:
            ref = torch.zeros_like(dist_loss)
        dist.broadcast(ref, src=0, group=dist.group.WORLD)
        assert torch.allclose(
            gather_loss,
            ref,
            rtol=1e-4,
            atol=1e-4,
        ), f"rank {rank}: gather_loss={gather_loss} ref={ref}"
        assert torch.allclose(
            dist_loss,
            ref,
            rtol=1e-4,
            atol=1e-4,
        ), f"rank {rank}: dist_loss={dist_loss} ref={ref}"
    finally:
        dist.destroy_process_group()


def _energy_dist_matches_gathered_worker(
    rank: int,
    world_size: int,
    n_members: int,
    init_file: str,
    average_channels: bool,
    batch_size: int,
):
    _init_cuda_process_group(rank, world_size, init_file)
    device = f"cuda:{rank}"
    try:
        torch.manual_seed(4242)
        b = batch_size
        f, t, c, h, w = 12, 2, 2, 32, 32
        target, pred_all = _build_shared_random_tensors(
            rank,
            (b, f, t, c, h, w),
            (n_members, b, f, t, c, h, w),
        )

        loss_fn = PatchedEnergyScoreLoss(
            weights=[1.0, 1.0],
            n_members=n_members,
            patch_size=3,
            enable_nhwc=False,
        )
        trainer = trainer_helper(
            output_variables=["v0", "v1"],
            device=device,
            distributed_ensemble_loss=True,
            ensemble_group_size=world_size,
            ensemble_group=dist.group.WORLD,
            ensemble_sharding_enabled=True,
        )
        loss_fn.setup(trainer)

        local_members = n_members // world_size
        start = rank * local_members
        stop = start + local_members
        local = pred_all[start:stop].reshape(local_members * b, f, t, c, h, w)
        # Mode C: sharded, distributed loss
        dist_loss = loss_fn(
            local,
            target,
            average_channels=average_channels,
        )
        # Mode B: sharded, gather then non-distributed loss
        gathered = torch.cat(
            list(dist_nn_func.all_gather(local, group=dist.group.WORLD)), dim=0
        )
        loss_fn.distributed_ensemble_loss = False
        gather_loss = loss_fn(
            gathered,
            target,
            average_channels=average_channels,
        )
        # Mode A: unsharded reference
        if rank == 0:
            pred_full = pred_all.reshape(n_members * b, f, t, c, h, w)
            ref = loss_fn(
                pred_full,
                target,
                average_channels=average_channels,
            )
        else:
            ref = torch.zeros_like(dist_loss)
        dist.broadcast(ref, src=0, group=dist.group.WORLD)
        assert torch.allclose(
            gather_loss,
            ref,
            rtol=1e-4,
            atol=1e-4,
        ), f"rank {rank}: gather_loss={gather_loss} ref={ref}"
        assert torch.allclose(
            dist_loss,
            ref,
            rtol=1e-4,
            atol=1e-4,
        ), f"rank {rank}: dist_loss={dist_loss} ref={ref}"
    finally:
        dist.destroy_process_group()


class _NoisyEnsembleModel(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(channels))
        self.bias = torch.nn.Parameter(torch.randn(channels))
        self.noise_scale = torch.nn.Parameter(torch.randn(channels))

    def forward(self, base: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # base: [B,F,T,C,H,W], noise: [M,B,C]
        base_term = (
            base.unsqueeze(0) * self.scale[None, None, None, None, :, None, None]
            + self.bias[None, None, None, None, :, None, None]
        )
        noise_term = noise[:, :, None, None, :, None, None] * self.noise_scale[
            None, None, None, None, :, None, None
        ]
        return base_term + noise_term


def _build_shared_model_inputs(rank: int, n_members: int, b: int, c: int, device: str):
    f, t, h, w = 12, 2, 32, 32
    if rank == 0:
        base = torch.randn(b, f, t, c, h, w, dtype=torch.float32, device=device)
        target = torch.randn(b, f, t, c, h, w, dtype=torch.float32, device=device)
        noise = torch.randn(n_members, b, c, dtype=torch.float32, device=device)
    else:
        base = torch.zeros(b, f, t, c, h, w, dtype=torch.float32, device=device)
        target = torch.zeros(b, f, t, c, h, w, dtype=torch.float32, device=device)
        noise = torch.zeros(n_members, b, c, dtype=torch.float32, device=device)
    dist.broadcast(base, src=0, group=dist.group.WORLD)
    dist.broadcast(target, src=0, group=dist.group.WORLD)
    dist.broadcast(noise, src=0, group=dist.group.WORLD)
    return base, target, noise


def _assert_model_params_close(
    model_ref, model_test, rank: int, rtol=1e-5, atol=1e-6, reduce_test_params: bool = False
):
    for (name_ref, p_ref), (name_test, p_test) in zip(
        model_ref.named_parameters(), model_test.named_parameters()
    ):
        assert name_ref == name_test
        test_param = p_test.detach().clone()
        if reduce_test_params:
            dist.all_reduce(test_param, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        assert torch.allclose(p_ref.detach(), test_param, rtol=rtol, atol=atol), (
            f"rank {rank}: parameter update mismatch for {name_ref}, "
            f"max_abs_diff={(p_ref.detach() - test_param).abs().max().item()}"
        )


def _optimizer_step_all_modes(
    model_ref,
    model_gather,
    model_dist,
    loss_fn,
    base: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor,
    start: int,
    stop: int,
    world_size: int,
):
    """One optimizer step each for Mode A, Mode B, and Mode C; always ``average_channels=True``."""
    b = base.shape[0]
    c = base.shape[3]
    local_members = stop - start

    opt_ref = torch.optim.SGD(model_ref.parameters(), lr=1e-2)
    opt_gather = torch.optim.SGD(model_gather.parameters(), lr=1e-2)
    opt_dist = torch.optim.SGD(model_dist.parameters(), lr=1e-2)

    opt_ref.zero_grad(set_to_none=True)
    pred_ref = model_ref(base, noise).reshape(noise.shape[0] * b, 12, 2, c, 32, 32)
    loss_fn.distributed_ensemble_loss = False
    ref_loss = loss_fn(
        pred_ref,
        target,
        average_channels=True,
    )
    ref_loss.sum().backward()
    opt_ref.step()

    opt_gather.zero_grad(set_to_none=True)
    pred_loc_gather = model_gather(base, noise[start:stop]).reshape(
        local_members * b, 12, 2, c, 32, 32
    )
    pred_gather = torch.cat(
        list(dist_nn_func.all_gather(pred_loc_gather, group=dist.group.WORLD)), dim=0
    )
    loss_fn.distributed_ensemble_loss = False
    gather_loss = loss_fn(
        pred_gather,
        target,
        average_channels=True,
    )
    assert torch.allclose(
        ref_loss.detach(),
        gather_loss.detach(),
        rtol=1e-4,
        atol=1e-4,
    ), (
        f"rank {dist.get_rank()}: Gathered loss does not match reference "
        f"loss: {ref_loss.detach()} vs {gather_loss.detach()}"
    )
    gather_loss.sum().backward()
    for param in model_gather.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dist.group.WORLD)
    opt_gather.step()

    opt_dist.zero_grad(set_to_none=True)
    pred_loc_dist = model_dist(base, noise[start:stop]).reshape(
        local_members * b, 12, 2, c, 32, 32
    )
    loss_fn.distributed_ensemble_loss = True
    dist_loss = loss_fn(
        pred_loc_dist,
        target,
        average_channels=True,
    )
    assert torch.allclose(
        ref_loss.detach(),
        dist_loss.detach(),
        rtol=1e-4,
        atol=1e-4,
    ), (
        f"rank {dist.get_rank()}: Loss from distributed ensemble does not match reference "
        f"loss: {ref_loss.detach()} vs {dist_loss.detach()}"
    )
    dist_loss.sum().backward()
    for param in model_dist.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    opt_dist.step()


def _crps_weight_update_worker(
    rank: int,
    world_size: int,
    n_members: int,
    init_file: str,
    batch_size: int,
):
    _init_cuda_process_group(rank, world_size, init_file)
    device = f"cuda:{rank}"
    try:
        torch.manual_seed(3001)
        b, c = batch_size, 2
        local_members = n_members // world_size
        start = rank * local_members
        stop = start + local_members
        base, target, noise = _build_shared_model_inputs(rank, n_members, b, c, device)

        torch.manual_seed(991)
        model_ref = _NoisyEnsembleModel(channels=c).to(device)
        torch.manual_seed(991)
        model_gather = _NoisyEnsembleModel(channels=c).to(device)
        torch.manual_seed(991)
        model_dist = _NoisyEnsembleModel(channels=c).to(device)

        loss_fn = WeightedCRPSLoss(weights=[1.0, 1.0], n_members=n_members)
        trainer = trainer_helper(
            output_variables=["v0", "v1"],
            device=device,
            distributed_ensemble_loss=True,
            ensemble_group_size=world_size,
            ensemble_group=dist.group.WORLD,
            ensemble_sharding_enabled=True,
        )
        loss_fn.setup(trainer)

        _optimizer_step_all_modes(
            model_ref=model_ref,
            model_gather=model_gather,
            model_dist=model_dist,
            loss_fn=loss_fn,
            base=base,
            target=target,
            noise=noise,
            start=start,
            stop=stop,
            world_size=world_size,
        )
        _assert_model_params_close(
            model_ref, model_gather, rank=rank, reduce_test_params=False
        )
        _assert_model_params_close(model_ref, model_dist, rank=rank)
    finally:
        dist.destroy_process_group()


def _spectral_weight_update_worker(
    rank: int,
    world_size: int,
    n_members: int,
    init_file: str,
    batch_size: int,
):
    _init_cuda_process_group(rank, world_size, init_file)
    device = f"cuda:{rank}"
    try:
        torch.manual_seed(3002)
        b, c = batch_size, 2
        nside = 32
        local_members = n_members // world_size
        start = rank * local_members
        stop = start + local_members
        base, target, noise = _build_shared_model_inputs(rank, n_members, b, c, device)

        torch.manual_seed(992)
        model_ref = _NoisyEnsembleModel(channels=c).to(device)
        torch.manual_seed(992)
        model_gather = _NoisyEnsembleModel(channels=c).to(device)
        torch.manual_seed(992)
        model_dist = _NoisyEnsembleModel(channels=c).to(device)

        loss_fn = WeightedCRPSLossSpectral(
            weights=[1.0, 1.0],
            n_members=n_members,
            lambda_spec=0.3,
            nside=nside,
            lmax=3 * nside - 1,
            mmax=3 * nside - 1,
        )
        trainer = trainer_helper(
            output_variables=["v0", "v1"],
            device=device,
            distributed_ensemble_loss=True,
            ensemble_group_size=world_size,
            ensemble_group=dist.group.WORLD,
            ensemble_sharding_enabled=True,
        )
        loss_fn.setup(trainer)

        _optimizer_step_all_modes(
            model_ref=model_ref,
            model_gather=model_gather,
            model_dist=model_dist,
            loss_fn=loss_fn,
            base=base,
            target=target,
            noise=noise,
            start=start,
            stop=stop,
            world_size=world_size,
        )
        _assert_model_params_close(
            model_ref, model_gather, rank=rank, reduce_test_params=False
        )
        _assert_model_params_close(model_ref, model_dist, rank=rank)
    finally:
        dist.destroy_process_group()


def _energy_weight_update_worker(
    rank: int,
    world_size: int,
    n_members: int,
    init_file: str,
    batch_size: int,
):
    _init_cuda_process_group(rank, world_size, init_file)
    device = f"cuda:{rank}"
    try:
        torch.manual_seed(3003)
        b, c = batch_size, 2
        local_members = n_members // world_size
        start = rank * local_members
        stop = start + local_members
        base, target, noise = _build_shared_model_inputs(rank, n_members, b, c, device)

        torch.manual_seed(993)
        model_ref = _NoisyEnsembleModel(channels=c).to(device)
        torch.manual_seed(993)
        model_gather = _NoisyEnsembleModel(channels=c).to(device)
        torch.manual_seed(993)
        model_dist = _NoisyEnsembleModel(channels=c).to(device)

        loss_fn = PatchedEnergyScoreLoss(
            weights=[1.0, 1.0], n_members=n_members, patch_size=3, enable_nhwc=False
        )
        trainer = trainer_helper(
            output_variables=["v0", "v1"],
            device=device,
            distributed_ensemble_loss=True,
            ensemble_group_size=world_size,
            ensemble_group=dist.group.WORLD,
            ensemble_sharding_enabled=True,
        )
        loss_fn.setup(trainer)

        _optimizer_step_all_modes(
            model_ref=model_ref,
            model_gather=model_gather,
            model_dist=model_dist,
            loss_fn=loss_fn,
            base=base,
            target=target,
            noise=noise,
            start=start,
            stop=stop,
            world_size=world_size,
        )
        _assert_model_params_close(
            model_ref, model_gather, rank=rank, reduce_test_params=False
        )
        _assert_model_params_close(model_ref, model_dist, rank=rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("n_members", [4, 8])
def test_WeightedCRPSLoss_model_weight_updates_match_across_ensemble_modes(
    batch_size: int, world_size: int, n_members: int
):
    """Mode A vs B vs C: parameter updates match; loss always ``average_channels=True`` (scalar training objective). See module docstring above for mode definitions."""
    if n_members % world_size != 0:
        pytest.skip("n_members must be divisible by world_size")
    _require_distributed_cuda(world_size)
    mp.set_start_method("spawn", force=True)
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        mp.spawn(
            _crps_weight_update_worker,
            args=(world_size, n_members, tmp.name, batch_size),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("n_members", [4, 8])
def test_WeightedCRPSLossSpectral_model_weight_updates_match_across_ensemble_modes(
    batch_size: int, world_size: int, n_members: int
):
    """Mode A vs B vs C: parameter updates match; loss always ``average_channels=True``. See module docstring for modes."""
    if n_members % world_size != 0:
        pytest.skip("n_members must be divisible by world_size")
    _require_distributed_cuda(world_size)
    mp.set_start_method("spawn", force=True)
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        mp.spawn(
            _spectral_weight_update_worker,
            args=(world_size, n_members, tmp.name, batch_size),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("n_members", [4, 8])
def test_PatchedEnergyScoreLoss_model_weight_updates_match_across_ensemble_modes(
    batch_size: int, world_size: int, n_members: int
):
    """Mode A vs B vs C: parameter updates match; loss always ``average_channels=True``. See module docstring for modes."""
    if n_members % world_size != 0:
        pytest.skip("n_members must be divisible by world_size")
    _require_distributed_cuda(world_size)
    mp.set_start_method("spawn", force=True)
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        mp.spawn(
            _energy_weight_update_worker,
            args=(world_size, n_members, tmp.name, batch_size),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("average_channels", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("n_members", [4, 8])
def test_WeightedCRPSLoss_forward_matches_across_ensemble_modes(
    average_channels: bool,
    batch_size: int,
    world_size: int,
    n_members: int,
):
    """Mode A, B, C forward outputs match. Parametrized on ``average_channels`` (True: scalar
    like training loss, False: per-channel like validation logging). See module docstring."""
    if n_members % world_size != 0:
        pytest.skip("n_members must be divisible by world_size")
    _require_distributed_cuda(world_size)
    mp.set_start_method("spawn", force=True)
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        mp.spawn(
            _crps_dist_matches_gathered_worker,
            args=(world_size, n_members, tmp.name, average_channels, batch_size),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("average_channels", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("n_members", [4, 8])
@pytest.mark.parametrize("lambda_spec", [0.0, 0.3])
def test_WeightedCRPSLossSpectral_forward_matches_across_ensemble_modes(
    average_channels: bool,
    batch_size: int,
    world_size: int,
    n_members: int,
    lambda_spec: float,
):
    """Mode A, B, C forward outputs match; both ``average_channels`` values. See module docstring."""
    if n_members % world_size != 0:
        pytest.skip("n_members must be divisible by world_size")
    _require_distributed_cuda(world_size)
    mp.set_start_method("spawn", force=True)
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        mp.spawn(
            _spectral_crps_dist_matches_gathered_worker,
            args=(world_size, n_members, tmp.name, average_channels, batch_size, lambda_spec),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("average_channels", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("n_members", [4, 8])
def test_PatchedEnergyScoreLoss_forward_matches_across_ensemble_modes(
    average_channels: bool,
    batch_size: int,
    world_size: int,
    n_members: int,
):
    """Mode A, B, C forward outputs match; both ``average_channels`` values. See module docstring."""
    if n_members % world_size != 0:
        pytest.skip("n_members must be divisible by world_size")
    _require_distributed_cuda(world_size)
    mp.set_start_method("spawn", force=True)
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        mp.spawn(
            _energy_dist_matches_gathered_worker,
            args=(world_size, n_members, tmp.name, average_channels, batch_size),
            nprocs=world_size,
            join=True,
        )

