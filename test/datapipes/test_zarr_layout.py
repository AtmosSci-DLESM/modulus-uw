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

"""Unit tests for healpix Zarr layout helpers."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from physicsnemo.datapipes.healpix.zarr_layout import (
    available_field_names,
    enable_zarrs_pipeline,
    is_monolithic_layout,
    is_named_arrays_layout,
    load_channel_data,
    load_constant_fields,
    load_windowed_channel_data,
    resolve_mask_field,
)


def _spatial_coords(*, f: int = 12, h: int = 4, w: int = 4) -> dict:
    return {
        "face": np.arange(f),
        "height": np.arange(h),
        "width": np.arange(w),
    }


def _make_monolithic_store(path, *, t: int = 4, c: int = 3, f: int = 12, h: int = 4, w: int = 4):
    channels = ["t2m", "u10m", "v10m"][:c]
    data = np.arange(t * c * f * h * w, dtype=np.float32).reshape(t, c, f, h, w)
    ds = xr.Dataset(
        {"inputs": (("time", "channel_c", "face", "height", "width"), data)},
        coords={
            "time": np.arange(t),
            "channel_c": channels,
            "channel_in": channels,
            **_spatial_coords(f=f, h=h, w=w),
        },
    )
    ds.to_zarr(path, mode="w")
    return zarr.open_group(str(path), mode="r")


def _make_named_store(path, *, t: int = 4, f: int = 12, h: int = 4, w: int = 4):
    dynamic = ["t2m", "u10m", "v10m"]
    constants = ["lsm", "z"]
    data_vars = {}
    for i, name in enumerate(dynamic):
        data_vars[name] = (
            ("time", "face", "height", "width"),
            np.full((t, f, h, w), float(i + 1), dtype=np.float32),
        )
    for i, name in enumerate(constants):
        data_vars[name] = (
            ("face", "height", "width"),
            np.full((f, h, w), float(10 + i), dtype=np.float32),
        )
    ds = xr.Dataset(
        data_vars,
        coords={"time": np.arange(t), **_spatial_coords(f=f, h=h, w=w)},
        attrs={"layout": "named_arrays_healpix"},
    )
    ds.to_zarr(path, mode="w")
    return zarr.open_group(str(path), mode="r")


def test_layout_detection(tmp_path):
    mono = _make_monolithic_store(tmp_path / "mono")
    named = _make_named_store(tmp_path / "named")
    assert is_monolithic_layout(mono) is True
    assert is_named_arrays_layout(named) is True
    assert is_monolithic_layout(named) is False


def test_available_field_names_named_store(tmp_path):
    named = _make_named_store(tmp_path / "named")
    assert available_field_names(named) == {"t2m", "u10m", "v10m", "lsm", "z"}


def test_load_channel_data_monolithic_by_name(tmp_path):
    ds = _make_monolithic_store(tmp_path / "mono")
    time_sl = slice(0, 2)
    expected = np.asarray(ds["inputs"][time_sl, [0, 2]])
    loaded = load_channel_data(ds, time_sl, ["t2m", "v10m"], n_threads=1)
    np.testing.assert_array_equal(loaded, expected)


def test_load_channel_data_named_store(tmp_path):
    ds = _make_named_store(tmp_path / "named")
    time_sl = slice(0, 2)
    expected = np.stack(
        [np.asarray(ds["v10m"][time_sl]), np.asarray(ds["t2m"][time_sl])], axis=1
    )
    loaded = load_channel_data(ds, time_sl, ["v10m", "t2m"], n_threads=1)
    np.testing.assert_array_equal(loaded, expected)


def test_load_constant_fields_named_store(tmp_path):
    ds = _make_named_store(tmp_path / "named")
    loaded = load_constant_fields(ds, ["lsm", "z"], n_threads=1)
    expected = np.stack([np.asarray(ds["lsm"]), np.asarray(ds["z"])], axis=0)
    np.testing.assert_array_equal(loaded, expected)


def test_load_channel_data_empty_raises_on_named(tmp_path):
    ds = _make_named_store(tmp_path / "named")
    with pytest.raises(ValueError, match="empty field name list"):
        load_channel_data(ds, slice(0, 1), [], n_threads=1)


def test_load_channel_data_named_store_with_scaling(tmp_path):
    ds = _make_named_store(tmp_path / "named")
    time_sl = slice(0, 2)
    raw = load_channel_data(ds, time_sl, ["v10m", "t2m"], n_threads=1)
    scaling = {
        "mean": np.expand_dims(np.array([1.0, 2.0], dtype=np.float32), (0, 2, 3, 4)),
        "std": np.expand_dims(np.array([2.0, 4.0], dtype=np.float32), (0, 2, 3, 4)),
    }
    scaled = load_channel_data(
        ds, time_sl, ["v10m", "t2m"], n_threads=1, scaling=scaling
    )
    expected = raw.copy()
    expected -= scaling["mean"]
    expected /= scaling["std"]
    np.testing.assert_array_equal(scaled, expected)


def test_load_windowed_channel_data_named_store(tmp_path):
    """Option A: direct window fill matches staging[:, c][time_idx] gather."""
    ds = _make_named_store(tmp_path / "named", t=8)
    time_sl = slice(0, 6)
    input_names = ["t2m", "v10m"]
    output_names = ["u10m", "t2m"]  # t2m shared with inputs
    input_time_idx = np.asarray([[0, 2], [1, 3]], dtype=np.intp)
    output_time_idx = np.asarray([[2, 4], [3, 5]], dtype=np.intp)

    staging = load_channel_data(
        ds, time_sl, ["t2m", "v10m", "u10m"], n_threads=1
    )
    # staging channel order: t2m=0, v10m=1, u10m=2
    exp_in = staging[input_time_idx[:, :, None], np.asarray([0, 1])[None, None, :]]
    exp_out = staging[output_time_idx[:, :, None], np.asarray([2, 0])[None, None, :]]

    got_in, got_out, got_ic = load_windowed_channel_data(
        ds,
        time_sl,
        input_names=input_names,
        input_time_idx=input_time_idx,
        output_names=output_names,
        output_time_idx=output_time_idx,
        n_threads=2,
    )
    np.testing.assert_array_equal(got_in, exp_in)
    np.testing.assert_array_equal(got_out, exp_out)
    assert got_ic is None


def test_enable_zarrs_pipeline_when_installed():
    assert enable_zarrs_pipeline() is True


def test_resolve_mask_field_named_arrays_and_coupled_stem_loader(tmp_path):
    """named_arrays_healpix LSM feeds coupled_partial_conv.load_spatial_mask."""
    from physicsnemo.models.dlwp_healpix_layers.coupled_partial_conv import (
        load_spatial_mask,
    )

    f, h, w = 12, 4, 4
    ocean = np.zeros((f, h, w), dtype=np.float32)
    ocean[0::2] = 1.0
    land = 1.0 - ocean
    ds = xr.Dataset(
        {"lsm": (("face", "height", "width"), land)},
        coords=_spatial_coords(f=f, h=h, w=w),
        attrs={"layout": "named_arrays_healpix"},
    )
    path = tmp_path / "mask_named.zarr"
    ds.to_zarr(path, mode="w")

    opened = xr.open_zarr(path)
    field = resolve_mask_field(opened, "constants", {"channel_c": "lsm"})
    np.testing.assert_allclose(field.values, land)
    opened.close()

    soft = load_spatial_mask(
        str(path),
        data_var="constants",
        selection_dict={"channel_c": "lsm"},
        invert=True,
        threshold=None,
    )
    np.testing.assert_allclose(soft.numpy(), ocean)
