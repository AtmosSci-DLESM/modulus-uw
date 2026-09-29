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

"""DataLoader worker start method for TimeSeriesDataModuleZarr."""

from __future__ import annotations

import sys

import pytest
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("numpy")

from physicsnemo.datapipes.healpix.data_modules_zarr import TimeSeriesDataModuleZarr


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        return index


def _bare_datamodule(*, num_workers: int) -> TimeSeriesDataModuleZarr:
    dm = TimeSeriesDataModuleZarr.__new__(TimeSeriesDataModuleZarr)
    dm.dataloader_batch_size = 1
    dm.drop_last = False
    dm.pin_memory = False
    dm.num_workers = num_workers
    dm.persistent_workers = False
    dm.prefetch_factor = None
    dm.in_order = None
    dm.mp_sharing_strategy = None
    dm.dataloader_multiprocessing_context = "spawn" if num_workers > 0 else None
    dm.collate_fn = None
    return dm


def test_dataloader_spawns_workers_when_requested():
    dm = _bare_datamodule(num_workers=2)
    loader, _ = dm._base_dataloader(dataset=_TinyDataset(), drop_last=False)
    assert isinstance(loader, DataLoader)
    assert loader.worker_init_fn is None
    if sys.platform not in ("win32", "darwin"):
        assert loader.multiprocessing_context is not None


def test_worker_init_applies_mp_sharing_strategy():
    import torch

    dm = _bare_datamodule(num_workers=2)
    dm.mp_sharing_strategy = "file_system"
    loader, _ = dm._base_dataloader(dataset=_TinyDataset(), drop_last=False)
    assert loader.worker_init_fn is not None
    previous = torch.multiprocessing.get_sharing_strategy()
    try:
        loader.worker_init_fn(0)
        assert torch.multiprocessing.get_sharing_strategy() == "file_system"
    finally:
        torch.multiprocessing.set_sharing_strategy(previous)


def test_dataloader_does_not_spawn_without_workers():
    dm = _bare_datamodule(num_workers=0)
    loader, _ = dm._base_dataloader(dataset=_TinyDataset(), drop_last=False)
    assert loader.worker_init_fn is None
    assert loader.multiprocessing_context is None
