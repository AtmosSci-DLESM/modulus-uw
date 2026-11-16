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

"""Tests for coupler ``coupled_dataset_path`` (a coupler reading its own zarr).

These use small synthetic datasets, so they need no external data.
"""

import types

import pytest

omegaconf = pytest.importorskip("omegaconf")
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

N_FACE, N_PIX = 12, 4
MAIN_VARS = ["a", "b"]  # prognostic variables of the main dataset
COUPLED_VARS = ["c1", "c2"]  # variables only present in the coupled zarr
TIME_STEP = np.timedelta64(3, "h")


def make_times(start="2020-01-01T00", n=16, step=TIME_STEP):
    return np.datetime64(start) + step * np.arange(n)


def make_dataset(times, variables, constants=("lsm",), seed=0, value=None):
    """Small HEALPix-layout dataset with inputs/targets/constants.

    If ``value`` is given every ``inputs`` entry is the integer index of its time
    in ``times`` offset by ``value`` (handy for checking which sample was read).
    """
    rng = np.random.default_rng(seed)
    shape = (len(times), len(variables), N_FACE, N_PIX, N_PIX)
    if value is None:
        data = rng.standard_normal(shape).astype("float32")
    else:
        data = np.broadcast_to(
            (np.arange(len(times)) + value).astype("float32")[:, None, None, None, None],
            shape,
        ).copy()
    return xr.Dataset(
        {
            "inputs": (("time", "channel_in", "face", "height", "width"), data),
            "targets": (
                ("time", "channel_out", "face", "height", "width"),
                data.copy(),
            ),
            "constants": (
                ("channel_c", "face", "height", "width"),
                rng.standard_normal((len(constants), N_FACE, N_PIX, N_PIX)).astype(
                    "float32"
                ),
            ),
        },
        coords={
            "time": times,
            "channel_in": list(variables),
            "channel_out": list(variables),
            "channel_c": list(constants),
            "face": np.arange(N_FACE),
            "height": np.arange(N_PIX),
            "width": np.arange(N_PIX),
            "lat": (("face", "height", "width"), rng.random((N_FACE, N_PIX, N_PIX))),
            "lon": (("face", "height", "width"), rng.random((N_FACE, N_PIX, N_PIX))),
        },
    )


def write_zarr(ds, path):
    ds.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def coupler_config(**params):
    base = {
        "batch_size": 1,
        "variables": COUPLED_VARS,
        "input_times": ["0h"],
        "input_time_dim": 1,
        "output_time_dim": 1,
        "presteps": 0,
        "prepared_coupled_data": True,
    }
    base.update(params)
    return [{"coupler": "ConstantCoupler", "params": base}]


def scaling_config(variables):
    return omegaconf.DictConfig(
        {v: {"mean": 0.0, "std": 1.0} for v in list(variables) + ["lsm"]}
    )


# --------------------------------------------------------------------------
# _coupled_index_offset (zarr dataset time-axis alignment)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# BaseCoupler zarr read with an index offset
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# _get_coupled_vars in the data modules
# --------------------------------------------------------------------------


def _stub(couplings):
    return types.SimpleNamespace(couplings=couplings)


def _couplings():
    return [
        {"coupler": "ConstantCoupler", "params": {"variables": ["x", "y"]}},
        {
            "coupler": "ConstantCoupler",
            "params": {"variables": ["lsm_c"], "coupled_dataset_path": "/some/other.zarr"},
        },
    ]


def test_get_coupled_vars_skips_couplers_with_own_path():
    from physicsnemo.datapipes.healpix.data_modules import CoupledTimeSeriesDataModule

    got = CoupledTimeSeriesDataModule._get_coupled_vars(_stub(_couplings()))
    assert got == ["x", "y"]


# --------------------------------------------------------------------------
# CoupledTimeSeriesDataset (xarray): coupled_dataset_path is reindexed
# --------------------------------------------------------------------------


def build_xarray_dataset(main, coupled_path):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset import (
        CoupledTimeSeriesDataset,
    )

    return CoupledTimeSeriesDataset(
        dataset=main,
        input_variables=MAIN_VARS,
        scaling=scaling_config(MAIN_VARS + COUPLED_VARS),
        data_time_step="3h",
        time_step="3h",
        batch_size=1,
        couplings=coupler_config(coupled_dataset_path=coupled_path),
    )


@pytest.fixture
def main_times():
    return make_times("2020-01-01T00", 16)


def test_xarray_identical_time_axis_is_not_reindexed(tmp_path, main_times, recwarn):
    main = make_dataset(main_times, MAIN_VARS)
    path = write_zarr(make_dataset(main_times, COUPLED_VARS, value=0), tmp_path / "c.zarr")
    ds = build_xarray_dataset(main, path)

    coupled = ds.couplings[0].ds
    assert list(coupled.channel_in.values) == COUPLED_VARS  # read from its own zarr
    assert not np.isnan(coupled.inputs.values).any()
    assert not [w for w in recwarn if "different time axis" in str(w.message)]


def test_xarray_coupled_starting_later_is_nan_padded_and_warns(tmp_path, main_times):
    main = make_dataset(main_times, MAIN_VARS)
    path = write_zarr(
        make_dataset(main_times[3:], COUPLED_VARS, value=0), tmp_path / "c.zarr"
    )
    with pytest.warns(UserWarning, match="3 dataset times are missing") as record:
        ds = build_xarray_dataset(main, path)
    assert any("will be NaN" in str(w.message) for w in record)

    coupled = ds.couplings[0].ds
    np.testing.assert_array_equal(coupled.time.values, main_times)
    values = coupled.inputs.values[:, 0, 0, 0, 0]
    assert np.isnan(values[:3]).all()
    # the coupled sample for main time k is the coupled zarr's sample k - 3
    np.testing.assert_array_equal(values[3:], np.arange(len(main_times) - 3))


def test_xarray_longer_coupled_zarr_is_trimmed_and_still_warns(tmp_path, main_times):
    main = make_dataset(main_times[4:12], MAIN_VARS)
    path = write_zarr(make_dataset(main_times, COUPLED_VARS, value=0), tmp_path / "c.zarr")
    with pytest.warns(UserWarning, match="different time axis") as record:
        ds = build_xarray_dataset(main, path)
    assert not any("will be NaN" in str(w.message) for w in record)

    coupled = ds.couplings[0].ds
    np.testing.assert_array_equal(coupled.time.values, main_times[4:12])
    np.testing.assert_array_equal(
        coupled.inputs.values[:, 0, 0, 0, 0], np.arange(4, 12)
    )


# --------------------------------------------------------------------------
# CoupledTimeSeriesDatasetZarr: coupled_dataset_path with a position offset
# --------------------------------------------------------------------------


# The dataset's final index yields an empty batch (existing behavior, unrelated to
# couplings), so only compare the samples before it.


# --------------------------------------------------------------------------
# CoupledTimeSeriesDataset (xarray): scaling is applied per variable name
# --------------------------------------------------------------------------


def channel_valued_dataset(times, variables, channel_values):
    """Dataset whose inputs/targets hold a constant per channel."""
    ds = make_dataset(times, variables, value=0)
    for name, value in channel_values.items():
        for key, dim in (("inputs", "channel_in"), ("targets", "channel_out")):
            ds[key].loc[{dim: name}] = value
    return ds


SCALING = omegaconf.DictConfig(
    {
        "a": {"mean": 1.0, "std": 2.0},
        "b": {"mean": 10.0, "std": 5.0},
        "c1": {"mean": 0.0, "std": 1.0},
        "c2": {"mean": 0.0, "std": 1.0},
        "lsm": {"mean": 0.0, "std": 1.0},
    }
)
RAW = {"a": 101.0, "b": 210.0}
NORMALIZED = {v: (RAW[v] - SCALING[v]["mean"]) / SCALING[v]["std"] for v in RAW}


def first_sample(ds):
    sample = ds[0]
    inputs = sample[0][0]  # prognostics: [B, F, T, C, H, W]
    targets = sample[1]
    return np.asarray(inputs), np.asarray(targets)


def make_scaled_dataset(main, input_variables, output_variables, couplings):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset import (
        CoupledTimeSeriesDataset,
    )

    return CoupledTimeSeriesDataset(
        dataset=main,
        input_variables=input_variables,
        output_variables=output_variables,
        scaling=SCALING,
        data_time_step="3h",
        time_step="3h",
        batch_size=1,
        couplings=couplings,
    )


def test_xarray_getitem_with_coupled_dataset_path(tmp_path, main_times):
    """Coupled variables are not in the main channel_in, so nothing may be trimmed
    from the scaling."""
    main = channel_valued_dataset(main_times, MAIN_VARS, RAW)
    path = write_zarr(make_dataset(main_times, COUPLED_VARS, value=0), tmp_path / "c.zarr")
    ds = make_scaled_dataset(
        main, MAIN_VARS, MAIN_VARS, coupler_config(coupled_dataset_path=path)
    )
    inputs, targets = first_sample(ds)
    assert inputs.shape[-3] == 2
    np.testing.assert_allclose(inputs[0, 0, 0, :, 0, 0], [NORMALIZED["a"], NORMALIZED["b"]])
    np.testing.assert_allclose(targets[0, 0, 0, :, 0, 0], [NORMALIZED["a"], NORMALIZED["b"]])


# --------------------------------------------------------------------------
# CoupledTimeSeriesDatasetZarr: scaling is applied per variable name
# (the zarr dataset already did this; these guard against regressions)
# --------------------------------------------------------------------------


