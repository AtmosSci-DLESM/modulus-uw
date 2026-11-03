"""WindowMeanCoupler: trapezoidal window means of the coupled fields, in training (from the
dataset) and in coupled inference (from a sender's outputs)."""

import numpy as np
import pandas as pd
import pytest
import torch

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from physicsnemo.datapipes.healpix.couplers import WindowMeanCoupler  # noqa: E402

_FACE, _HEIGHT, _WIDTH = 2, 3, 4
_N_TIME = 60


def _values(t):
    """Coupled values at 6-hourly time index t: nonlinear in t so that the trapezoid
    weights matter, and different per channel."""
    t = np.asarray(t, dtype="float64")
    return np.stack([t**2, 100.0 + 3.0 * t**2], axis=-1)  # [..., channel]


def _dataset(values_of_t=_values):
    data = values_of_t(np.arange(_N_TIME))  # [time, channel]
    data = np.broadcast_to(
        data[:, :, None, None, None], (_N_TIME, 2, _FACE, _HEIGHT, _WIDTH)
    ).astype("float32")
    return xr.Dataset(
        data_vars={"inputs": (("time", "channel_in", "face", "height", "width"), data.copy())},
        coords={
            "time": pd.date_range("1979-01-01", periods=_N_TIME, freq="6h"),
            "channel_in": ["u1", "t1"],
            "face": np.arange(_FACE),
            "height": np.arange(_HEIGHT),
            "width": np.arange(_WIDTH),
        },
    )


def _coupler(output_time_dim=4, presteps=1, batch_size=2, dataset=None, window="24h"):
    c = WindowMeanCoupler(
        dataset=_dataset() if dataset is None else dataset,
        batch_size=batch_size,
        variables=["u1", "t1"],
        presteps=presteps,
        input_time_dim=1,
        output_time_dim=output_time_dim,
        input_times=[window],
    )
    # 24 h model step on 6-hourly data
    c.compute_coupled_indices(interval=4, data_time_step=pd.Timedelta("6h"))
    c.coupled_channel_indices = [0, 1]
    return c


def _trapezoid(t_start):
    """Reference: trapezoidal mean of the 6-hourly samples t_start .. t_start + 4."""
    w = np.array([1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 8])
    return (w[:, None] * _values(np.arange(t_start, t_start + 5))).sum(0)  # [channel]


def test_training_slots_are_trapezoid_means_of_their_windows():
    # A sample's first frame (the prestep input, t0 - 24 h) is index start + b, so the
    # prestep averages [t0 - 24 h, t0] and step k averages [t0 + 24 k, t0 + 24 (k + 1)].
    c = _coupler(batch_size=4)
    start = 10
    out = c.construct_integrated_couplings(batch={"time": slice(start, start + 24)}, bsize=4)
    assert out.shape == (5, 4, 2, _FACE, _HEIGHT, _WIDTH)
    for b in range(4):  # four consecutive starts: all four synoptic hours
        for i in range(5):
            expected = _trapezoid(start + b + 4 * i)
            np.testing.assert_allclose(out[i, b, :, 0, 0, 0], expected, rtol=1e-6)
            assert np.all(out[i, b] == out[i, b, :, :1, :1, :1])  # spatially uniform input stays uniform


def test_daily_cycle_averages_out_for_every_start_hour():
    # A 24 h-periodic signal has a zero trapezoidal mean over any 24 h window, so the
    # forcing carries no trace of the sample's start hour.
    def diurnal(t):
        s = np.sin(2 * np.pi * np.asarray(t, dtype="float64") / 4 + 0.3)
        return np.stack([s, 2 * s], axis=-1)

    c = _coupler(batch_size=4, dataset=_dataset(diurnal))
    out = c.construct_integrated_couplings(batch={"time": slice(7, 31)}, bsize=4)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_window_must_equal_the_model_step():
    with pytest.raises(ValueError, match="must equal the model time step"):
        _coupler(window="12h")


def test_only_one_input_time_per_step():
    with pytest.raises(ValueError, match="input_time_dim 1"):
        WindowMeanCoupler(dataset=_dataset(), batch_size=1, variables=["u1"], input_time_dim=2)
    with pytest.raises(ValueError, match="one input time"):
        WindowMeanCoupler(dataset=_dataset(), batch_size=1, variables=["u1"], input_times=["24h", "48h"])


def test_zarr_store_matches_xarray_in_requested_variable_order(tmp_path):
    # Training reads a zarr store; channels follow the coupler's variable order.
    path = tmp_path / "coupled.zarr"
    _dataset().to_zarr(path)
    batch = {"time": slice(10, 32)}
    from_xr = _coupler(batch_size=2).construct_integrated_couplings(batch=batch, bsize=2)
    c = WindowMeanCoupler(
        dataset=zarr.open(str(path)), batch_size=2, variables=["t1", "u1"], presteps=1,
        input_time_dim=1, output_time_dim=4, input_times=["24h"],
    )
    c.compute_coupled_indices(interval=4, data_time_step=pd.Timedelta("6h"))
    from_zarr = c.construct_integrated_couplings(batch=batch, bsize=2)
    np.testing.assert_allclose(from_zarr, from_xr[:, :, ::-1], rtol=1e-6)


def test_means_are_normalized_with_the_coupled_scaling_and_float32():
    # The base class's normalization applies before averaging (identical to after,
    # with the same mean/std, since it is linear).
    c = _coupler(batch_size=1)
    scaling = xr.Dataset(
        {"mean": ("index", [5.0, 7.0]), "std": ("index", [2.0, 4.0])},
        coords={"index": ["u1", "t1"]},
    )
    c.set_scaling(scaling)
    out = c.construct_integrated_couplings(batch={"time": slice(10, 31)}, bsize=1)
    assert out.dtype == np.float32
    for i in range(5):
        expected = (_trapezoid(10 + 4 * i) - np.array([5.0, 7.0])) / np.array([2.0, 4.0])
        np.testing.assert_allclose(out[i, 0, :, 0, 0, 0], expected, rtol=1e-5)


def _sender_outputs(t_first, n, batch_size=1):
    """Truth as a sender would output it: n 6-hourly fields from index t_first, [B, F, T, C, H, W]."""
    v = torch.as_tensor(_values(np.arange(t_first, t_first + n)), dtype=torch.float64)  # [T, C]
    return v[None, None, :, :, None, None].expand(batch_size, _FACE, n, 2, _HEIGHT, _WIDTH).clone()


def _started(output_time_dim, start):
    """Coupler seeded for inference from window 0 of the dataset, as the inferencer seeds it."""
    c = _coupler(output_time_dim=output_time_dim, batch_size=1)
    first = c.construct_integrated_couplings(batch={"time": slice(start, start + 4 * output_time_dim + 5)}, bsize=1)
    c.start_rollout(torch.as_tensor(first), _sender_outputs(start + 4, 1))  # sender state at t0
    return c


def test_inference_on_truth_matches_training_windows():
    # Inference: one 24 h step per coupler window (output_time_dim 1); each window's
    # slots come from the sender's 4 outputs plus its state at the window start.
    start = 10  # t0 = start + 4
    c = _started(1, start)
    for k in range(3):
        t0 = start + 4 + 4 * k
        c.set_coupled_fields(_sender_outputs(t0 + 1, 4))
        out = c.construct_integrated_couplings()  # [slots, B, C, F, H, W]
        assert out.shape == (2, 1, 2, _FACE, _HEIGHT, _WIDTH)
        assert out.dtype == torch.float32 and out.device.type == "cpu"
        # prestep: the previous window [t0 - 24 h, t0]; step: [t0, t0 + 24 h]
        np.testing.assert_allclose(out[0, 0, :, 0, 0, 0].numpy(), _trapezoid(t0 - 4), rtol=1e-6)
        np.testing.assert_allclose(out[1, 0, :, 0, 0, 0].numpy(), _trapezoid(t0), rtol=1e-6)


def test_inference_splits_a_multi_step_window():
    # Two 24 h steps per coupler window: 8 sender outputs give two step slots.
    start = 10
    c = _started(2, start)
    c.set_coupled_fields(_sender_outputs(start + 5, 8))
    out = c.construct_integrated_couplings()
    for i, t in enumerate([start, start + 4, start + 8]):
        np.testing.assert_allclose(out[i, 0, :, 0, 0, 0].numpy(), _trapezoid(t), rtol=1e-6)


def test_inference_rejects_a_sender_at_another_resolution():
    # 12-hourly sender outputs would make a 2-point mean, not the 5-point mean trained on.
    c = _started(1, 10)
    with pytest.raises(ValueError, match="as in training"):
        c.set_coupled_fields(_sender_outputs(15, 2))


def test_inference_needs_start_rollout_and_reset_clears():
    c = _coupler(output_time_dim=1, batch_size=1)
    with pytest.raises(RuntimeError, match="start_rollout"):
        c.set_coupled_fields(_sender_outputs(5, 4))
    c = _started(1, 10)
    c.set_coupled_fields(_sender_outputs(15, 4))
    c.reset_coupler()
    assert c._sender_last is None and c._previous_means is None
    assert not c.coupled_mode
