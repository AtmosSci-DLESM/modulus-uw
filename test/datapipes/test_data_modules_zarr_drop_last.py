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

"""Unit tests for DataLoader drop_last vs dataset-level batching.

These configs (e.g. dlesym-aq) set ``drop_last=True`` with
``dataloader_batch_size=None`` so the dataset forms batches. PyTorch forbids
``drop_last=True`` when ``batch_size=None``; the datamodule must not pass that
combination through to ``DataLoader``.
"""

from __future__ import annotations

import pytest
from torch.utils.data import DataLoader, Dataset

from physicsnemo.datapipes.healpix.data_modules_zarr import TimeSeriesDataModuleZarr


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> int:
        return index


def _bare_datamodule(
    *,
    dataloader_batch_size: int | None,
    drop_last: bool,
) -> TimeSeriesDataModuleZarr:
    """Minimal instance for exercising ``_base_dataloader`` only."""
    dm = TimeSeriesDataModuleZarr.__new__(TimeSeriesDataModuleZarr)
    dm.dataloader_batch_size = dataloader_batch_size
    dm.drop_last = drop_last
    dm.pin_memory = False
    dm.num_workers = 0
    dm.persistent_workers = False
    dm.prefetch_factor = None
    dm.in_order = None
    dm.collate_fn = None
    return dm


def test_pytorch_rejects_drop_last_with_batch_size_none():
    """Document the PyTorch constraint this datamodule must work around."""
    with pytest.raises(ValueError, match="mutually exclusive with drop_last"):
        DataLoader(_TinyDataset(), batch_size=None, drop_last=True)


def test_drop_last_false_on_dataloader_when_batch_size_none():
    """Dataset-level batching: DataLoader must construct and drop_last is off."""
    dm = _bare_datamodule(dataloader_batch_size=None, drop_last=True)
    loader, sampler = dm._base_dataloader(
        dataset=_TinyDataset(),
        drop_last=True,
    )
    assert sampler is None
    assert isinstance(loader, DataLoader)
    assert loader.batch_size is None
    assert loader.drop_last is False


def test_drop_last_honored_when_dataloader_auto_batches():
    """When the DataLoader auto-batches, drop_last must still apply."""
    dm = _bare_datamodule(dataloader_batch_size=2, drop_last=True)
    loader, sampler = dm._base_dataloader(
        dataset=_TinyDataset(),
        drop_last=True,
    )
    assert sampler is None
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2
    assert loader.drop_last is True
    # Incomplete final batch of size 1 is dropped (5 samples, batch_size 2).
    assert len(loader) == 2
