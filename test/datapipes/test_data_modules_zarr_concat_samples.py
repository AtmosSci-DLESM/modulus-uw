"""Random-start batching: with ``dataloader_batch_size`` the DataLoader draws dataset items
independently and ``concat_samples`` joins them along the batch axis, giving the same
tensors a dataset with that batch size would give for the same samples."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from physicsnemo.datapipes.healpix.data_modules_zarr import (
    CoupledTimeSeriesDataModuleZarr,
    TimeSeriesDataModuleZarr,
    concat_samples,
)

_N, _F, _T, _C, _H, _W, _SLOTS = 12, 2, 3, 4, 3, 3, 5


class _Samples(Dataset):
    """Items of batch_size consecutive samples from fixed per-sample arrays, laid out as the
    healpix zarr datasets lay them out: state, insolation, targets [B, F, T, C, H, W],
    constants [F, C, H, W] without a batch axis, couplings time-first [T, B, C, F, H, W]."""

    def __init__(self, batch_size, coupled=True, forecast=False):
        rng = np.random.default_rng(0)
        self.state = rng.standard_normal((_N, _F, _T, _C, _H, _W), dtype=np.float32)
        self.insol = rng.standard_normal((_N, _F, _T, 1, _H, _W), dtype=np.float32)
        self.coupled_fields = rng.standard_normal((_N, _SLOTS, 2, _F, _H, _W), dtype=np.float32)
        self.targets = rng.standard_normal((_N, _F, _T, _C, _H, _W), dtype=np.float32)
        self.constants = rng.standard_normal((_F, 2, _H, _W), dtype=np.float32)
        self.batch_size, self.coupled, self.forecast = batch_size, coupled, forecast

    def __len__(self):
        return _N - self.batch_size + 1

    def __getitem__(self, item):
        s = slice(item, item + self.batch_size)
        inputs = [self.state[s], self.insol[s], self.constants]
        if self.coupled:
            inputs.append(self.coupled_fields[s].transpose(1, 0, 2, 3, 4, 5))
        return inputs if self.forecast else (inputs, self.targets[s])


def _assert_same(a, b):
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    else:
        assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))


@pytest.mark.parametrize("coupled", [True, False])
@pytest.mark.parametrize("forecast", [False, True])
def test_concat_of_single_samples_equals_a_batched_item(coupled, forecast):
    singles = _Samples(1, coupled, forecast)
    batched = _Samples(3, coupled, forecast)
    starts = [4, 5, 6]
    got = concat_samples([singles[i] for i in starts], coupled=coupled)
    _assert_same(got, batched[4])


def test_concat_keeps_the_items_order_for_random_starts():
    singles = _Samples(1)
    starts = [7, 0, 3]
    (inputs, targets) = concat_samples([singles[i] for i in starts], coupled=True)
    assert torch.equal(inputs[0], torch.as_tensor(singles.state[starts]))
    assert torch.equal(inputs[3], torch.as_tensor(singles.coupled_fields[starts].transpose(1, 0, 2, 3, 4, 5)))
    assert torch.equal(targets, torch.as_tensor(singles.targets[starts]))


def _datamodule(cls, monkeypatch, **kwargs):
    monkeypatch.setattr(cls, "setup", lambda self: None)
    return cls(dataset_path="/nonexistent", **kwargs)


def test_collate_is_installed_only_with_dataloader_batch_size(monkeypatch):
    dm = _datamodule(TimeSeriesDataModuleZarr, monkeypatch, batch_size=4)
    assert dm.collate_fn is None
    dm = _datamodule(TimeSeriesDataModuleZarr, monkeypatch, batch_size=1, dataloader_batch_size=8)
    assert dm.collate_fn.func is concat_samples and dm.collate_fn.keywords == {"coupled": False}
    dm = _datamodule(
        CoupledTimeSeriesDataModuleZarr, monkeypatch, batch_size=1, dataloader_batch_size=8,
        couplings=[{"coupler": "ConstantCoupler", "params": {}}],
    )
    assert dm.collate_fn.keywords == {"coupled": True}


def test_distributed_loader_draws_independent_starts(monkeypatch):
    dm = _datamodule(
        CoupledTimeSeriesDataModuleZarr, monkeypatch, batch_size=1, dataloader_batch_size=3,
        couplings=[{"coupler": "ConstantCoupler", "params": {}}], num_workers=0,
    )
    singles = _Samples(1)
    orders = []
    for rank in range(2):
        loader, sampler = dm._base_dataloader(
            dataset=singles, num_shards=2, shard_id=rank, shuffle=True, drop_last=True
        )
        sampler.set_epoch(0)
        order = list(iter(sampler))
        for k, (inputs, targets) in enumerate(loader):
            idx = order[3 * k: 3 * k + 3]
            assert inputs[0].shape == (3, _F, _T, _C, _H, _W)
            assert torch.equal(inputs[0], torch.as_tensor(singles.state[idx]))
            assert torch.equal(inputs[3], torch.as_tensor(singles.coupled_fields[idx].transpose(1, 0, 2, 3, 4, 5)))
        orders.append(order)
    assert not set(orders[0]) & set(orders[1])  # ranks get disjoint samples
    assert all(o != sorted(o) for o in orders)  # shuffled, not consecutive starts


def test_test_loader_keeps_one_dataset_item_per_batch(monkeypatch):
    # Forecast datasets keep per-item rollout state, so the test loader never joins items.
    dm = _datamodule(
        CoupledTimeSeriesDataModuleZarr, monkeypatch, batch_size=1, dataloader_batch_size=3,
        couplings=[{"coupler": "ConstantCoupler", "params": {}}], num_workers=0,
    )
    dm.test_dataset = _Samples(1, forecast=True)
    loader, _ = dm.test_dataloader()
    assert loader.batch_size is None and loader.collate_fn is not concat_samples
    first = next(iter(loader))
    assert first[0].shape == (1, _F, _T, _C, _H, _W)
