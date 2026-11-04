"""Coupling one receiver to several senders (setup_coupling_sources / gather_sources), e.g. a troposphere model fed
SST/SIC by a prescribed ocean and its top layer by a stratosphere model."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

xr = pytest.importorskip("xarray")

from physicsnemo.datapipes.healpix.couplers import ConstantCoupler  # noqa: E402

_FACE, _HEIGHT, _WIDTH = 2, 3, 4
_VARS = ["sst", "sic", "t0", "u0"]


def _dataset(n_time=40):
    data = np.broadcast_to(
        np.arange(n_time, dtype="float32")[:, None, None, None, None], (n_time, len(_VARS), _FACE, _HEIGHT, _WIDTH)
    ).copy()
    return xr.Dataset(
        data_vars={"inputs": (("time", "channel_in", "face", "height", "width"), data)},
        coords={"time": pd.date_range("1979-01-01", periods=n_time, freq="6h"), "channel_in": _VARS,
                "face": np.arange(_FACE), "height": np.arange(_HEIGHT), "width": np.arange(_WIDTH)},
    )


def _coupler(variables=_VARS):
    c = ConstantCoupler(dataset=_dataset(), batch_size=1, variables=variables, input_times=["0h"], input_time_dim=1,
                        output_time_dim=4, presteps=1, refresh_period="24h")
    c.compute_coupled_indices(interval=1, data_time_step=pd.Timedelta("6h"))
    return c


def _field(values):
    """[B=1, F, T=1, C, H, W] whose channels hold the given constants."""
    v = torch.as_tensor(values, dtype=torch.float32)
    return v[None, None, None, :, None, None].expand(1, _FACE, 1, len(values), _HEIGHT, _WIDTH).clone()


def test_legacy_setup_coupling_takes_channels_in_sender_order():
    # Pins existing behaviour: indices follow the sender's output order, not the coupler's
    # variable order (the single-sender configs list both in the same order).
    c = _coupler(["sst", "sic"])
    c.setup_coupling(SimpleNamespace(output_variables=["sic", "other", "sst"]))
    assert c.coupled_channel_indices == [0, 2]


def test_gather_follows_the_coupler_variable_order():
    c = _coupler()
    c.setup_coupling_sources(
        {"strat": ["u0", "q0", "t0"], "ocean": ["sic", "sst"]},
        {"sst": "ocean", "sic": "ocean", "t0": "strat", "u0": "strat"},
    )
    fields = {"strat": _field([30.0, 31.0, 32.0]), "ocean": _field([10.0, 11.0])}
    got = c.gather_sources(fields)
    # sst, sic, t0, u0 <- ocean[1], ocean[0], strat[2], strat[0]
    assert torch.equal(got, _field([11.0, 10.0, 32.0, 30.0]))
    assert c.coupled_channel_indices == [0, 1, 2, 3]


def test_mapping_input_equals_the_concatenated_tensor_across_windows():
    # Refresh coupler: the prestep slot keeps the previous window's field either way.
    sources = {"strat": ["t0", "u0"], "ocean": ["sst", "sic"]}
    source_of = {"sst": "ocean", "sic": "ocean", "t0": "strat", "u0": "strat"}
    a, b = _coupler(), _coupler()
    for c in (a, b):
        c.construct_integrated_couplings(batch={"time": slice(10, 16)}, bsize=1)
    a.setup_coupling_sources(sources, source_of)
    b.coupled_channel_indices = [0, 1, 2, 3]
    for k in range(2):
        ocean, strat = _field([100.0 + k, 200.0 + k]), _field([300.0 + k, 400.0 + k])
        a.set_coupled_fields({"ocean": ocean, "strat": strat})
        b.set_coupled_fields(torch.cat([ocean, strat], dim=3))
        assert torch.equal(a.construct_integrated_couplings(), b.construct_integrated_couplings())


def test_missing_sender_or_variable_raises():
    c = _coupler()
    with pytest.raises(ValueError, match="No sender"):
        c.setup_coupling_sources({"ocean": ["sst", "sic"]}, {"sst": "ocean", "sic": "ocean"})
    with pytest.raises(ValueError, match="does not output"):
        c.setup_coupling_sources({"ocean": ["sst", "sic"], "strat": ["t0"]},
                                 {"sst": "ocean", "sic": "ocean", "t0": "strat", "u0": "strat"})


def test_input_kind_must_match_the_setup_and_senders_may_differ_in_device_and_dtype():
    c = _coupler()
    c.construct_integrated_couplings(batch={"time": slice(10, 16)}, bsize=1)
    with pytest.raises(ValueError, match="after setup_coupling_sources"):
        c.set_coupled_fields({"ocean": _field([1.0, 2.0])})  # no multi-source setup
    c.setup_coupling_sources({"ocean": ["sst", "sic"], "strat": ["t0", "u0"]},
                             {"sst": "ocean", "sic": "ocean", "t0": "strat", "u0": "strat"})
    with pytest.raises(ValueError, match="tensor after setup_coupling"):
        c.set_coupled_fields(_field([1.0, 2.0, 3.0, 4.0]))  # plain tensor after multi-source setup
    c.set_coupled_fields({"ocean": _field([1.0, 2.0]).double(), "strat": _field([3.0, 4.0])})
    out = c.construct_integrated_couplings()
    assert out.dtype == torch.float32 and torch.equal(out[1:, 0, :, 0, 0, 0], torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 4))
    # a plain setup_coupling afterwards returns to tensor input
    c.setup_coupling(SimpleNamespace(output_variables=list(_VARS)))
    c.set_coupled_fields(_field([5.0, 6.0, 7.0, 8.0]))


def test_unknown_sender_name_raises_value_error():
    c = _coupler()
    with pytest.raises(ValueError, match="not among"):
        c.setup_coupling_sources({"strat": ["t0", "u0"], "ocean": ["sst", "sic"]},
                                 {"sst": "ocean", "sic": "ocean", "t0": "stratosphere", "u0": "strat"})
