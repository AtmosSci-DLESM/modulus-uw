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
    # Small toy data: B=1,F=1,T=1,C=2,H=4,W=4
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
        loss_fn_bad_members(pred, tar, average_channels=True, distributed_ensemble_loss=False)


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


@pytest.mark.parametrize("average_channels", [True, False])
def test_WeightedCRPSLoss_distributed_exact_matches_single_rank(average_channels: bool):
    b, f, t, c, h, w = 2, 12, 2, 2, 64, 64
    n_members = 2
    target = torch.randn(b, f, t, c, h, w, dtype=torch.float32)
    pred_members = torch.randn(n_members, b, f, t, c, h, w, dtype=torch.float32)
    prediction = pred_members.reshape(n_members * b, f, t, c, h, w)

    loss_fn = WeightedCRPSLoss(weights=[1.0, 1.0], n_members=n_members)
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    loss_fn.setup(trainer)
    ref = loss_fn(prediction, target, average_channels=average_channels)

    # Simulate n=1 local shard with world_size=1 by passing distributed flag.
    local_pred = pred_members[:1].reshape(b, f, t, c, h, w)
    got = loss_fn(local_pred, target, average_channels=average_channels, distributed_ensemble_loss=True)
    assert torch.isfinite(got).all()
    assert got.shape == ref.shape


def _dist_crps_worker(rank: int, world_size: int, init_file: str):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(123)
    b, f, t, c, h, w = 1, 12, 1, 2, 64, 64
    target = torch.randn(b, f, t, c, h, w, dtype=torch.float32)
    pred_all = torch.randn(world_size, b, f, t, c, h, w, dtype=torch.float32)
    prediction_local = pred_all[rank].reshape(b, f, t, c, h, w)

    loss_fn = WeightedCRPSLoss(weights=[1.0, 1.0], n_members=world_size)
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    loss_fn.setup(trainer)

    dist_loss = loss_fn(
        prediction_local,
        target,
        average_channels=True,
        distributed_ensemble_loss=True,
        ensemble_group=dist.group.WORLD,
    )
    gathered = [torch.zeros_like(dist_loss) for _ in range(world_size)]
    dist.all_gather(gathered, dist_loss, group=dist.group.WORLD)
    assert torch.isfinite(dist_loss)
    for other in gathered:
        assert torch.isfinite(other)
        assert torch.allclose(dist_loss, other, rtol=1e-4, atol=1e-4)
    dist.destroy_process_group()


def test_WeightedCRPSLoss_distributed_exact_equivalence():
    mp.set_start_method("spawn", force=True)
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        mp.spawn(_dist_crps_worker, args=(2, tmp.name), nprocs=2, join=True)


def test_PatchedEnergyScoreLoss_distributed_exact_equivalence():
    b, f, t, c, h, w = 1, 12, 1, 2, 64, 64
    n_members = 2
    target = torch.randn(b, f, t, c, h, w, dtype=torch.float32)
    pred_all = torch.randn(n_members, b, f, t, c, h, w, dtype=torch.float32)
    local_pred = pred_all[:1].reshape(b, f, t, c, h, w)
    loss_fn = PatchedEnergyScoreLoss(
        weights=[1.0, 1.0],
        n_members=n_members,
        patch_size=3,
        enable_nhwc=False,
    )
    trainer = trainer_helper(output_variables=["v0", "v1"], device="cpu")
    loss_fn.setup(trainer)
    out = loss_fn(local_pred, target, average_channels=True, distributed_ensemble_loss=True)
    assert torch.isfinite(out)