"""Multi-store training: a list of dataset paths concatenates one dataset per store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import DictConfig, ListConfig

from physicsnemo.datapipes.healpix.data_modules_zarr import TimeSeriesDataModuleZarr
from physicsnemo.datapipes.healpix.multi_store_dataset import MultiStoreDataset

CHANNELS = ["z500", "t2m0"]
SPLITS = {
    "train_date_start": "1979-01-01",
    "train_date_end": "1979-01-04T18:00",
    "val_date_start": "1979-01-05",
    "val_date_end": "1979-01-06T18:00",
    "test_date_start": "1979-01-05",
    "test_date_end": "1979-01-06T18:00",
}


def _store(path, value, n_time=24):
    """Tiny monolithic HEALPix zarr with every value equal to ``value``."""
    shape = (n_time, len(CHANNELS), 1, 2, 2)
    xr.Dataset(
        data_vars={
            "inputs": (("time", "channel_in", "face", "height", "width"), np.full(shape, value, "float32")),
            "targets": (("time", "channel_out", "face", "height", "width"), np.full(shape, value, "float32")),
            "lat": (("face", "height", "width"), np.zeros((1, 2, 2))),
            "lon": (("face", "height", "width"), np.zeros((1, 2, 2))),
        },
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq="6h"),
            "channel_in": CHANNELS,
            "channel_out": CHANNELS,
            "face": [0],
            "height": [0, 1],
            "width": [0, 1],
        },
    ).to_zarr(path)
    return str(path)


def _datamodule(dataset_path, **kw):
    return TimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        batch_size=2,
        input_variables=CHANNELS,
        scaling=DictConfig({c: {"mean": 0.0, "std": 1.0} for c in CHANNELS}),
        splits=DictConfig(SPLITS),
        data_time_step="6h",
        time_step="6h",
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        **kw,
    )


def _value(item):
    return float(np.asarray(item[0]).flat[0])


def test_single_path_unchanged(tmp_path):
    dm = _datamodule(_store(tmp_path / "a.zarr", 1.0))
    assert not isinstance(dm.train_dataset, MultiStoreDataset)
    assert dm.dataset_paths == [dm.dataset_path]


def test_multi_store_concat_same_splits(tmp_path):
    a = _store(tmp_path / "a.zarr", 1.0)
    b = _store(tmp_path / "b.zarr", 2.0)
    dm = _datamodule(ListConfig([a, b]))
    for split in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
        assert isinstance(split, MultiStoreDataset)
        first, second = split.datasets
        assert len(first) == len(second) > 0
        assert first.start_index == second.start_index
        assert len(split) == 2 * len(first)
    train = dm.train_dataset
    n0 = len(train.datasets[0])
    assert _value(train[0]) == 1.0
    assert _value(train[n0]) == 2.0
    # Per-store attributes and the common kwargs path resolve to store 0.
    assert train.time_da is train.datasets[0].time_da
    assert str(dm._get_common_dataset_kwargs()["dataset_path"]) == a
    train.curr_item = 3
    assert train.datasets[0].curr_item == 3
    loader, _ = dm.train_dataloader()
    assert len(loader) == len(train)
    assert {_value(batch) for batch in loader} == {1.0, 2.0}


def test_multi_store_errors(tmp_path):
    a = _store(tmp_path / "a.zarr", 1.0)
    with pytest.raises(FileNotFoundError, match="missing.zarr"):
        _datamodule([a, str(tmp_path / "missing.zarr")])
    with pytest.raises(ValueError, match="single dataset_path"):
        TimeSeriesDataModuleZarr(
            dataset_path=[a, a],
            input_variables=CHANNELS,
            scaling=DictConfig({c: {"mean": 0.0, "std": 1.0} for c in CHANNELS}),
            forecast_init_times=pd.date_range("1979-01-01", periods=2, freq="6h"),
        )
