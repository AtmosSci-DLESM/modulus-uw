"""ConstantCoupler refresh period: which coupled time each slot of a window gets."""

import numpy as np
import pandas as pd
import pytest
import torch

xr = pytest.importorskip("xarray")

from physicsnemo.datapipes.healpix.couplers import ConstantCoupler  # noqa: E402
from physicsnemo.datapipes.healpix.data_modules_zarr import (  # noqa: E402
    CoupledTimeSeriesDataModuleZarr,
)

_FACE, _HEIGHT, _WIDTH = 2, 3, 4


def _time_valued_dataset(n_time=40):
    """6-hourly dataset whose coupled values equal their time index."""
    data = np.broadcast_to(
        np.arange(n_time, dtype="float32")[:, None, None, None, None],
        (n_time, 2, _FACE, _HEIGHT, _WIDTH),
    ).copy()
    return xr.Dataset(
        data_vars={"inputs": (("time", "channel_in", "face", "height", "width"), data)},
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq="6h"),
            "channel_in": ["sst", "sic"],
            "face": np.arange(_FACE),
            "height": np.arange(_HEIGHT),
            "width": np.arange(_WIDTH),
        },
    )


def _coupler(output_time_dim, refresh_period, batch_size=2):
    c = ConstantCoupler(
        dataset=_time_valued_dataset(),
        batch_size=batch_size,
        variables=["sst", "sic"],
        input_times=["0h"],
        input_time_dim=1,
        output_time_dim=output_time_dim,
        presteps=1,
        refresh_period=refresh_period,
    )
    c.compute_coupled_indices(interval=1, data_time_step=pd.Timedelta("6h"))
    c.coupled_channel_indices = [0, 1]
    return c


def test_refresh_slots_follow_periods_of_the_window():
    # 16 steps of 6 h, refresh 24 h. Sample b's first frame (the prestep input,
    # t0 - 6 h) is index start + b, so t0 is start + b + 1.
    c = _coupler(output_time_dim=16, refresh_period="24h")
    start = 10
    out = c.construct_integrated_couplings(batch={"time": slice(start, start + 20)}, bsize=2)
    assert out.shape == (17, 2, 2, _FACE, _HEIGHT, _WIDTH)
    # prestep: t0 - 24 h; steps 0-3: t0; 4-7: t0 + 24 h; 8-11: t0 + 48 h; 12-15: t0 + 72 h
    rel_t0 = np.array([-4] + [0] * 4 + [4] * 4 + [8] * 4 + [12] * 4)
    for b in range(2):
        expected = start + b + 1 + rel_t0
        np.testing.assert_array_equal(out[:, b, 0, 0, 0, 0], expected)
        np.testing.assert_array_equal(out[:, b, 1, 0, 0, 0], expected)
    np.testing.assert_array_equal(c._slot_periods, [-1] + [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)


def test_no_refresh_keeps_one_field_per_window():
    c = _coupler(output_time_dim=4, refresh_period=None)
    out = c.construct_integrated_couplings(batch={"time": slice(10, 16)}, bsize=2)
    # every slot holds the sample's first frame (t0 - 6 h), as before refresh periods
    for b in range(2):
        np.testing.assert_array_equal(out[:, b, 0, 0, 0, 0], [10 + b] * 5)


def test_refresh_before_dataset_start_raises():
    c = _coupler(output_time_dim=4, refresh_period="24h")
    with pytest.raises(ValueError, match="before the start of the dataset"):
        c.construct_integrated_couplings(batch={"time": slice(0, 6)}, bsize=2)


def test_refresh_must_be_a_multiple_of_the_step():
    with pytest.raises(ValueError, match="multiple of the model step"):
        _coupler(output_time_dim=4, refresh_period="9h")


def test_coupled_mode_prestep_keeps_the_previous_window_field():
    # Inference: window 0 comes from the dataset, later windows from set_coupled_fields.
    c = _coupler(output_time_dim=4, refresh_period="24h", batch_size=1)
    start = 10
    c.construct_integrated_couplings(batch={"time": slice(start, start + 6)}, bsize=1)
    first = torch.full((1, _FACE, 1, 2, _HEIGHT, _WIDTH), 100.0)
    out = c.set_coupled_fields(first) or c.preset_coupled_fields  # [T, B, C, F, H, W]
    # the prestep slot keeps window 0's period field (its t0, index start + 1)
    assert torch.all(out[0] == start + 1)
    assert torch.all(out[1:] == 100.0)
    second = torch.full((1, _FACE, 1, 2, _HEIGHT, _WIDTH), 200.0)
    c.set_coupled_fields(second)
    out = c.preset_coupled_fields
    assert torch.all(out[0] == 100.0)
    assert torch.all(out[1:] == 200.0)
    c.reset_coupler()
    assert c._previous_fields is None and c._period0_source is None


def test_train_split_gets_longer_window_couplings():
    dm = CoupledTimeSeriesDataModuleZarr.__new__(CoupledTimeSeriesDataModuleZarr)
    dm.output_time_dim, dm.train_output_time_dim = 4, 16
    dm.couplings = [{"coupler": "ConstantCoupler", "params": {"output_time_dim": 4, "refresh_period": "24h"}}]
    common = {"output_time_dim": 4, "couplings": dm.couplings}
    train = dm._train_kwargs(common)
    assert train["output_time_dim"] == 16
    assert train["couplings"][0]["params"]["output_time_dim"] == 16
    assert train["couplings"][0]["params"]["refresh_period"] == "24h"
    # val/test kwargs and the module's own coupling configs are untouched
    assert common["output_time_dim"] == 4 and dm.couplings[0]["params"]["output_time_dim"] == 4
    # setup() wraps the dataset class: only the train split (it passes add_train_noise) gets the window
    seen = []
    build = dm._with_train_window(lambda **kw: seen.append(kw))
    build(add_train_noise=False, **common)
    build(drop_last=False, **common)
    assert [kw["output_time_dim"] for kw in seen] == [16, 4]
    assert seen[0]["couplings"][0]["params"]["output_time_dim"] == 16
    assert seen[1]["couplings"][0]["params"]["output_time_dim"] == 4
    dm.train_output_time_dim = 4
    assert dm._train_kwargs(common) is common
