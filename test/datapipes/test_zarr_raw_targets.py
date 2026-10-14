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

"""CPU checks for optional raw targets on the shared native Zarr path."""
import numpy as np
import pytest
import torch
import xarray as xr
import zarr
from omegaconf import OmegaConf
from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import TimeSeriesDatasetZarr
from physicsnemo.datapipes.healpix.data_modules_zarr import (
    CoupledTimeSeriesDataModuleZarr,
)


@pytest.fixture
def source(tmp_path):
    dates = np.arange(np.datetime64("1993-01-01"), np.datetime64("1993-02-15")).astype(
        "datetime64[ns]"
    )
    names = ["sst", "q", "f1", "f2", "f3"]
    values = (
        np.broadcast_to(
            np.arange(len(dates))[:, None, None, None, None], (len(dates), 5, 12, 2, 2)
        )
        .copy()
        .astype("f4")
    )
    values += np.arange(5, dtype="f4")[None, :, None, None, None] * 100
    targets = np.full((len(dates), 2, 12, 2, 2), 1000, dtype="f4")
    targets += np.arange(len(dates), dtype="f4")[:, None, None, None, None]
    targets[8, 0, 0, 0, 1] = np.nan
    constants = np.zeros((1, 12, 2, 2), dtype="f4")
    constants[0, 0, 0, 0] = 1
    constants[0, 0, 1, 0] = 0.25
    ds = xr.Dataset(
        {
            "inputs": (("time", "channel_in", "face", "height", "width"), values),
            "targets": (("time", "channel_out", "face", "height", "width"), targets),
            "constants": (("channel_c", "face", "height", "width"), constants),
            "lat": (("face", "height", "width"), np.zeros((12, 2, 2))),
            "lon": (("face", "height", "width"), np.zeros((12, 2, 2))),
        },
        coords=dict(
            time=dates,
            channel_in=names,
            channel_out=["q", "sst"],
            channel_c=["land_sea_mask"],
            face=np.arange(12),
            height=np.arange(2),
            width=np.arange(2),
        ),
    )
    path = tmp_path / "source.zarr"
    ds.to_zarr(path, zarr_format=2)
    group = zarr.open_group(path, mode="r+")
    del group["time"]
    array = group.create_array("time", data=dates)
    array.attrs.update(
        _ARRAY_DIMENSIONS=["time"],
        calendar=["gregorian"],
        units=["hours since 1900-01-01"],
    )
    zarr.consolidate_metadata(path)
    scaling = {name: dict(mean=1.0, std=2.0) for name in ["sst", "q"]}
    scaling.update({name: dict(mean=3.0, std=4.0) for name in ["f1", "f2", "f3"]})
    scaling["land_sea_mask"] = dict(mean=0.0, std=1.0)
    return path, scaling


def dataset_kwargs(source):
    path, scaling = source
    return dict(
        dataset_path=path,
        scaling=OmegaConf.create(scaling),
        input_variables=["sst", "q"],
        output_variables=["q", "sst"],
        constant_variables=["land_sea_mask"],
        input_time_dim=2,
        output_time_dim=2,
        data_time_step="24h",
        time_step="96h",
        batch_size=1,
        start_date="1993-01-01",
        end_date="1993-02-14",
        drop_last=False,
    )


def test_raw_targets_preserve_nan_source_order_and_atmos_default(source):
    kwargs = dataset_kwargs(source)
    default = TimeSeriesDatasetZarr(**kwargs)
    ordinary_inputs, ordinary_targets = default[0]
    assert ordinary_targets[0, 0, 0, 0, 0, 1] == (108 - 1) / 2
    raw = TimeSeriesDatasetZarr(
        **kwargs,
        raw_target_mode=True,
        target_mask_constant="land_sea_mask",
        preserve_land_channels=["sst"]
    )
    inputs, targets = raw[0]
    assert np.isnan(targets[0, 0, 0, 0, 0, 1])
    assert targets[0, 0, 1, 0, 0, 1] == (1012 - 1) / 2
    assert inputs[0][0, 0, 0, 0, 0, 0] == -0.5  # finite historical SST land retained
    assert inputs[0][0, 0, 0, 1, 0, 0] == 0  # other land neutral
    assert inputs[1][0, 0, 1, 0] == 0.25  # actual fractional LSM
    assert ordinary_inputs[0][0, 0, 0, 1, 0, 0] != 0
    assert len((inputs, targets)) == 2


def test_raw_targets_subset_last_partial_batch_bounds(source):
    kwargs = dataset_kwargs(source)
    kwargs.update(start_date="1993-01-11", end_date="1993-02-09", batch_size=4)
    dataset = TimeSeriesDatasetZarr(**kwargs, raw_target_mode=True)
    sizes = [dataset[index][1].shape[0] for index in range(len(dataset))]
    assert sizes == [4, 4, 4, 4, 2]
    assert sum(sizes) == 18


def test_actual_module_forecast_coupler_datetime_and_next_integration(source):
    path, scaling = source
    couplings = [
        dict(
            coupler="TrailingAverageCoupler",
            params=dict(
                variables=["f1", "f2", "f3"],
                batch_size=1,
                presteps=0,
                input_time_dim=2,
                output_time_dim=2,
                input_times=["96h", "192h"],
                averaging_window="96h",
                prepared_coupled_data=True,
            ),
        )
    ]
    from hydra.utils import instantiate

    module = instantiate(
        {
            "_target_": "physicsnemo.datapipes.healpix.data_modules_zarr.CoupledTimeSeriesDataModuleZarr",
            "_recursive_": False,
        },
        dataset_path=str(path),
        batch_size=1,
        dataloader_batch_size=None,
        input_variables=["sst", "q"],
        output_variables=["q", "sst"],
        constant_variables=["land_sea_mask"],
        scaling=OmegaConf.create(scaling),
        input_time_dim=2,
        output_time_dim=2,
        data_time_step="24h",
        time_step="96h",
        forecast_init_times=["1993-01-05"],
        couplings=couplings,
        num_workers=0,
        pin_memory=False,
        raw_target_mode=True,
        target_mask_constant="land_sea_mask",
        preserve_land_channels=["sst"],
        _convert_="all",
    )
    loader, sampler = module.test_dataloader()
    inputs = next(iter(loader))
    assert sampler is None and len(inputs) == 3
    assert inputs[0].shape == (1, 12, 2, 2, 2, 2)
    assert inputs[2].shape == (1, 1, 6, 12, 2, 2)
    expected = [
        (208 - 3) / 4,
        (308 - 3) / 4,
        (408 - 3) / 4,
        (212 - 3) / 4,
        (312 - 3) / 4,
        (412 - 3) / 4,
    ]
    np.testing.assert_allclose(inputs[2][0, 0, :, 0, 0, 1], expected)
    output = torch.full((1, 12, 2, 2, 2, 2), 7.0)
    following = module.test_dataset.next_integration(output, inputs[1])
    assert torch.equal(following[0], output)
    expected = [
        (216 - 3) / 4,
        (316 - 3) / 4,
        (416 - 3) / 4,
        (220 - 3) / 4,
        (320 - 3) / 4,
        (420 - 3) / 4,
    ]
    np.testing.assert_allclose(following[2][0, 0, :, 0, 0, 1], expected)
    assert torch.equal(following[1], inputs[1])
