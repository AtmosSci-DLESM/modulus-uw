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

"""Unit tests for persistent per-worker Zarr IO pools in TimeSeriesDataModuleZarr."""

from __future__ import annotations

import pytest
from torch.utils.data import DataLoader, Dataset

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from physicsnemo.datapipes.healpix import zarr_layout
from physicsnemo.datapipes.healpix.data_modules_zarr import TimeSeriesDataModuleZarr
from physicsnemo.datapipes.healpix.zarr_layout import load_channel_data


def _make_named_store(path, *, t: int = 4, f: int = 12, h: int = 4, w: int = 4):
    dynamic = ["t2m", "u10m", "v10m"]
    data_vars = {}
    for i, name in enumerate(dynamic):
        data_vars[name] = (
            ("time", "face", "height", "width"),
            np.full((t, f, h, w), float(i + 1), dtype=np.float32),
        )
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": np.arange(t),
            "face": np.arange(f),
            "height": np.arange(h),
            "width": np.arange(w),
        },
        attrs={"layout": "named_arrays_healpix"},
    )
    ds.to_zarr(path, mode="w")
    return zarr.open_group(str(path), mode="r")


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        return index


def _bare_datamodule(*, num_workers: int, dataloader_io_threads: int) -> TimeSeriesDataModuleZarr:
    dm = TimeSeriesDataModuleZarr.__new__(TimeSeriesDataModuleZarr)
    dm.dataloader_batch_size = 1
    dm.drop_last = False
    dm.pin_memory = False
    dm.num_workers = num_workers
    dm.persistent_workers = False
    dm.prefetch_factor = None
    dm.in_order = None
    dm.dataloader_io_threads = dataloader_io_threads
    dm.collate_fn = None
    return dm


def test_init_worker_pool_is_persistent_until_shutdown():
    zarr_layout.shutdown_worker_pool()
    try:
        assert zarr_layout.worker_pool_active() is False
        zarr_layout.init_worker_pool(2)
        assert zarr_layout.worker_pool_active() is True
        zarr_layout.init_worker_pool(2)
        assert zarr_layout.worker_pool_active() is True
    finally:
        zarr_layout.shutdown_worker_pool()
        assert zarr_layout.worker_pool_active() is False


def test_load_channel_data_reuses_persistent_pool(tmp_path):
    ds = _make_named_store(tmp_path / "named")
    zarr_layout.shutdown_worker_pool()
    try:
        zarr_layout.init_worker_pool(2)
        loaded = load_channel_data(ds, slice(0, 2), ["t2m", "u10m", "v10m"], n_threads=2)
        assert loaded.shape == (2, 3, 12, 4, 4)
    finally:
        zarr_layout.shutdown_worker_pool()


def test_dataloader_worker_init_fn_installed_when_io_threads_enabled():
    dm = _bare_datamodule(num_workers=2, dataloader_io_threads=8)
    loader, _ = dm._base_dataloader(dataset=_TinyDataset(), drop_last=False)
    assert isinstance(loader, DataLoader)
    assert loader.worker_init_fn is not None


def test_dataloader_worker_init_fn_omitted_for_single_thread():
    dm = _bare_datamodule(num_workers=2, dataloader_io_threads=1)
    loader, _ = dm._base_dataloader(dataset=_TinyDataset(), drop_last=False)
    assert loader.worker_init_fn is None


def test_dataloader_worker_init_fn_omitted_without_workers():
    dm = _bare_datamodule(num_workers=0, dataloader_io_threads=8)
    loader, _ = dm._base_dataloader(dataset=_TinyDataset(), drop_last=False)
    assert loader.worker_init_fn is None
