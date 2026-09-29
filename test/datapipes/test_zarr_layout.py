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
    constants_are_stacked,
    enable_zarrs_pipeline,
    is_per_variable_layout,
    is_stacked_layout,
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


def _make_stacked_store(path, *, t: int = 4, c: int = 3, f: int = 12, h: int = 4, w: int = 4):
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


def _make_per_variable_store(
    path, *, t: int = 4, f: int = 12, h: int = 4, w: int = 4, layout: str = "per_variable"
):
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
        attrs={"layout": layout},
    )
    ds.to_zarr(path, mode="w")
    return zarr.open_group(str(path), mode="r")


def test_layout_detection(tmp_path):
    stacked = _make_stacked_store(tmp_path / "stacked")
    per_var = _make_per_variable_store(tmp_path / "per_var")
    legacy = _make_per_variable_store(
        tmp_path / "legacy", layout="named_arrays_healpix"
    )
    assert is_stacked_layout(stacked) is True
    assert is_per_variable_layout(stacked) is False
    assert constants_are_stacked(stacked) is True
    assert is_per_variable_layout(per_var) is True
    assert is_stacked_layout(per_var) is False
    assert constants_are_stacked(per_var) is False
    assert is_per_variable_layout(legacy) is True
    assert constants_are_stacked(legacy) is False


def test_available_field_names_per_variable_store(tmp_path):
    named = _make_per_variable_store(tmp_path / "per_var")
    assert available_field_names(named) == {"t2m", "u10m", "v10m", "lsm", "z"}


def test_load_channel_data_stacked_by_name(tmp_path):
    ds = _make_stacked_store(tmp_path / "mono")
    time_sl = slice(0, 2)
    expected = np.asarray(ds["inputs"][time_sl, [0, 2]])
    loaded = load_channel_data(ds, time_sl, ["t2m", "v10m"])
    np.testing.assert_array_equal(loaded, expected)


def test_load_channel_data_per_variable_store(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named")
    time_sl = slice(0, 2)
    expected = np.stack(
        [np.asarray(ds["v10m"][time_sl]), np.asarray(ds["t2m"][time_sl])], axis=1
    )
    loaded = load_channel_data(ds, time_sl, ["v10m", "t2m"])
    np.testing.assert_array_equal(loaded, expected)


def test_load_constant_fields_per_variable_store(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named")
    loaded = load_constant_fields(ds, ["lsm", "z"])
    expected = np.stack([np.asarray(ds["lsm"]), np.asarray(ds["z"])], axis=0)
    np.testing.assert_array_equal(loaded, expected)


def test_load_channel_data_empty_raises_on_per_variable(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named")
    with pytest.raises(ValueError, match="empty field name list"):
        load_channel_data(ds, slice(0, 1), [])


def test_load_channel_data_per_variable_store_with_scaling(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named")
    time_sl = slice(0, 2)
    raw = load_channel_data(ds, time_sl, ["v10m", "t2m"])
    scaling = {
        "mean": np.expand_dims(np.array([1.0, 2.0], dtype=np.float32), (0, 2, 3, 4)),
        "std": np.expand_dims(np.array([2.0, 4.0], dtype=np.float32), (0, 2, 3, 4)),
    }
    scaled = load_channel_data(
        ds, time_sl, ["v10m", "t2m"], scaling=scaling
    )
    expected = raw.copy()
    expected -= scaling["mean"]
    expected /= scaling["std"]
    np.testing.assert_array_equal(scaled, expected)


def test_load_windowed_channel_data_per_variable_store(tmp_path):
    """Option A: direct window fill matches staging[:, c][time_idx] gather."""
    ds = _make_per_variable_store(tmp_path / "named", t=8)
    time_sl = slice(0, 6)
    input_names = ["t2m", "v10m"]
    output_names = ["u10m", "t2m"]  # t2m shared with inputs
    input_time_idx = np.asarray([[0, 2], [1, 3]], dtype=np.intp)
    output_time_idx = np.asarray([[2, 4], [3, 5]], dtype=np.intp)

    staging = load_channel_data(
        ds, time_sl, ["t2m", "v10m", "u10m"]    )
    # staging channel order: t2m=0, v10m=1, u10m=2
    exp_in = staging[input_time_idx[:, :, None], np.asarray([0, 1])[None, None, :]]
    exp_out = staging[output_time_idx[:, :, None], np.asarray([2, 0])[None, None, :]]

    got_in, got_out = load_windowed_channel_data(
        ds,
        time_sl,
        input_names=input_names,
        input_time_idx=input_time_idx,
        output_names=output_names,
        output_time_idx=output_time_idx,
    )
    np.testing.assert_array_equal(got_in, exp_in)
    np.testing.assert_array_equal(got_out, exp_out)


def test_load_windowed_channel_data_stacked_matches_per_variable(tmp_path):
    """Stacked ``inputs`` and per-variable stores yield the same windowed tensors."""
    t, f, h, w = 8, 12, 4, 4
    channels = ["t2m", "u10m", "v10m"]
    stacked_ds = _make_stacked_store(tmp_path / "stacked", t=t, c=3, f=f, h=h, w=w)
    # Same channel planes as the stacked store, one array per field.
    data_vars = {
        name: (
            ("time", "face", "height", "width"),
            np.asarray(stacked_ds["inputs"][:, i]),
        )
        for i, name in enumerate(channels)
    }
    per_var_xr = xr.Dataset(
        data_vars,
        coords={"time": np.arange(t), **_spatial_coords(f=f, h=h, w=w)},
        attrs={"layout": "per_variable"},
    )
    per_var_path = tmp_path / "per_variable"
    per_var_xr.to_zarr(per_var_path, mode="w")
    per_var_ds = zarr.open_group(str(per_var_path), mode="r")

    assert is_stacked_layout(stacked_ds) is True
    assert is_per_variable_layout(per_var_ds) is True

    time_sl = slice(0, 6)
    input_names = ["t2m", "v10m"]
    output_names = ["u10m", "t2m"]
    input_time_idx = np.asarray([[0, 2], [1, 3]], dtype=np.intp)
    output_time_idx = np.asarray([[2, 4], [3, 5]], dtype=np.intp)
    kwargs = dict(
        time_sl=time_sl,
        input_names=input_names,
        input_time_idx=input_time_idx,
        output_names=output_names,
        output_time_idx=output_time_idx,
    )
    stacked_in, stacked_out = load_windowed_channel_data(stacked_ds, **kwargs)
    per_var_in, per_var_out = load_windowed_channel_data(per_var_ds, **kwargs)
    np.testing.assert_allclose(stacked_in, per_var_in)
    np.testing.assert_allclose(stacked_out, per_var_out)


def test_enable_zarrs_pipeline_when_installed():
    assert enable_zarrs_pipeline() is True


def test_resolve_mask_field_per_variable_legacy_layout_attr(tmp_path):
    """``channel_c`` selects a top-level array on a per-variable store.

    The legacy ``named_arrays_healpix`` attribute is what catalogs already
    on disk declare. The coupled stem calls this helper when it is installed.
    """
    f, h, w = 12, 4, 4
    land = np.zeros((f, h, w), dtype=np.float32)
    land[0::2] = 1.0
    ds = xr.Dataset(
        {"lsm": (("face", "height", "width"), land)},
        coords=_spatial_coords(f=f, h=h, w=w),
        attrs={"layout": "named_arrays_healpix"},
    )
    path = tmp_path / "mask_per_variable.zarr"
    ds.to_zarr(path, mode="w")

    opened = xr.open_zarr(path)
    field = resolve_mask_field(opened, "constants", {"channel_c": "lsm"})
    np.testing.assert_allclose(field.values, land)
    opened.close()


def test_missing_layout_attr_is_not_per_variable(tmp_path):
    bare = _make_per_variable_store(tmp_path / "bare", layout="")
    assert is_stacked_layout(bare) is False
    assert is_per_variable_layout(bare) is False
    with pytest.raises(ValueError, match="Unrecognized healpix Zarr layout"):
        load_channel_data(bare, slice(0, 1), ["t2m"])


def test_stacked_getitem_matches_joint_inputs_scale(tmp_path):
    """Stacked batches match a joint ``inputs`` read, in-place scale, and float32."""
    omegaconf = pytest.importorskip("omegaconf")
    pd = pytest.importorskip("pandas")
    from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import (
        TimeSeriesDatasetZarr,
    )

    channels = ["t2m", "u10m", "v10m"]
    input_variables = ["v10m", "t2m"]
    output_variables = ["u10m", "t2m"]
    n_time, face, height, width = 16, 1, 2, 2
    times = pd.date_range("2021-03-01", periods=n_time, freq="6h")
    data = np.arange(
        n_time * len(channels) * face * height * width, dtype=np.float64
    ).reshape(n_time, len(channels), face, height, width)
    path = tmp_path / "stacked.zarr"
    xr.Dataset(
        data_vars={
            "inputs": (("time", "channel_in", "face", "height", "width"), data),
            "lat": (("face", "height", "width"), np.zeros((face, height, width))),
            "lon": (("face", "height", "width"), np.zeros((face, height, width))),
        },
        coords={
            "time": times,
            "channel_in": channels,
            "face": np.arange(face),
            "height": np.arange(height),
            "width": np.arange(width),
        },
    ).to_zarr(path)

    means = {"t2m": 1.5, "u10m": -2.0, "v10m": 0.25}
    stds = {"t2m": 2.0, "u10m": 4.0, "v10m": 0.5}
    dataset = TimeSeriesDatasetZarr(
        dataset_path=str(path),
        data_time_step="6h",
        time_step="6h",
        gap="6h",
        scaling=omegaconf.DictConfig(
            {name: {"mean": means[name], "std": stds[name]} for name in channels}
        ),
        input_variables=input_variables,
        output_variables=output_variables,
        start_date="2021-03-01",
        end_date="2021-03-04",
        batch_size=2,
        input_time_dim=2,
        output_time_dim=1,
    )

    raw = np.asarray(zarr.open_group(str(path), mode="r")["inputs"][:])
    name_to_i = {name: i for i, name in enumerate(channels)}
    time_index, this_batch = dataset._get_time_index(0)
    staging = np.array(raw[slice(*time_index)])
    for i, name in enumerate(channels):
        staging[:, i] -= np.asarray(means[name], dtype=staging.dtype)
        staging[:, i] /= np.asarray(stds[name], dtype=staging.dtype)

    def _windows(names, index_lists):
        selected = staging[:, [name_to_i[name] for name in names]]
        out = np.empty(
            (this_batch, len(index_lists[0]), len(names), face, height, width),
            dtype=np.float32,
        )
        for sample in range(this_batch):
            out[sample] = selected[index_lists[sample]]
        return np.transpose(out, (0, 3, 1, 2, 4, 5))

    inputs, targets = dataset[0]
    np.testing.assert_array_equal(
        inputs[0], _windows(input_variables, dataset._input_indices)
    )
    np.testing.assert_array_equal(
        targets, _windows(output_variables, dataset._output_indices)
    )
    assert inputs[0].dtype == np.float32
    assert targets.dtype == np.float32


def _make_sharded_field(path, *, t: int = 20, shard_time: int = 8, f: int = 2, h: int = 2, w: int = 2):
    """Per-variable array with inner time chunk 1 and a time shard, zstd level 0."""
    from zarr.codecs import ZstdCodec

    data = np.arange(t * f * h * w, dtype=np.float32).reshape(t, f, h, w)
    data[3] = np.nan
    zarr.create_array(
        store=str(path),
        name="t2m",
        shape=data.shape,
        chunks=(1, f, h, w),
        shards=(shard_time, f, h, w),
        dtype="float32",
        zarr_format=3,
        compressors=[ZstdCodec(level=0)],
        fill_value=np.nan,
        dimension_names=["time", "face", "height", "width"],
    )
    array = zarr.open_array(f"{path}/t2m", mode="a")
    array[:] = data
    group = zarr.open_group(str(path), mode="a")
    group.attrs["layout"] = "per_variable"
    return data


def test_read_sharded_time_slice_matches_array(tmp_path):
    from physicsnemo.datapipes.healpix.zarr_shard_read import read_sharded_time_slice

    data = _make_sharded_field(tmp_path / "shard")
    array = zarr.open_array(f"{tmp_path / 'shard'}/t2m", mode="r")
    sl = slice(2, 12)
    got = read_sharded_time_slice(array, sl)
    assert got is not None
    ref = np.asarray(array[sl])
    assert got.shape == ref.shape
    assert np.array_equal(got, ref, equal_nan=True)
    loaded = load_channel_data(zarr.open_group(str(tmp_path / "shard"), mode="r"), sl, ["t2m"])
    assert np.array_equal(loaded[:, 0], ref, equal_nan=True)
    assert np.array_equal(loaded[:, 0], data[sl], equal_nan=True)


def test_sharded_window_matches_arrays(tmp_path):
    from zarr.codecs import ZstdCodec

    path = tmp_path / "multi"
    t, f, h, w, shard_time = 20, 2, 2, 2, 8
    stored = {}
    for i, name in enumerate(("t2m", "u10m")):
        data = (np.arange(t * f * h * w, dtype=np.float32) + i * 1000).reshape(t, f, h, w)
        data[3] = np.nan
        stored[name] = data
        zarr.create_array(
            store=str(path),
            name=name,
            shape=data.shape,
            chunks=(1, f, h, w),
            shards=(shard_time, f, h, w),
            dtype="float32",
            zarr_format=3,
            compressors=[ZstdCodec(level=0)],
            fill_value=np.nan,
            dimension_names=["time", "face", "height", "width"],
        )
        zarr.open_array(f"{path}/{name}", mode="a")[:] = data
    group = zarr.open_group(str(path), mode="a")
    group.attrs["layout"] = "per_variable"
    group = zarr.open_group(str(path), mode="r")
    sl = slice(2, 12)
    loaded = load_channel_data(group, sl, ["t2m", "u10m"])
    for c, name in enumerate(("t2m", "u10m")):
        assert np.array_equal(loaded[:, c], stored[name][sl], equal_nan=True)
    idx_in = np.array([[0, 1], [2, 3]])
    idx_out = np.array([[4, 6], [5, 7]])
    inputs, targets = load_windowed_channel_data(
        group, sl, ["t2m", "u10m"], idx_in, ["u10m", "t2m"], idx_out
    )
    for c, name in enumerate(("t2m", "u10m")):
        assert np.array_equal(inputs[:, :, c], stored[name][sl][idx_in], equal_nan=True)
    for c, name in enumerate(("u10m", "t2m")):
        assert np.array_equal(targets[:, :, c], stored[name][sl][idx_out], equal_nan=True)


def test_stored_shard_index_skips_tail_read(tmp_path, monkeypatch):
    """A ``_shard_index/<field>`` table is (n_shards, chunks_per_shard, 2) uint64."""
    import physicsnemo.datapipes.healpix.zarr_shard_read as shard_read

    path = tmp_path / "indexed"
    _make_sharded_field(path)
    group = zarr.open_group(str(path), mode="a")
    layout = shard_read._field_table(group)["t2m"]
    rows = np.stack([layout.offsets_for(i) for i in range(layout.n_shards)]).astype("<u8")
    assert rows.shape == (layout.n_shards, layout.chunks_per_shard_t, 2)
    index_arr = zarr.create_array(
        store=group.store,
        name="_shard_index/t2m",
        shape=rows.shape,
        chunks=rows.shape,
        dtype="<u8",
        zarr_format=3,
        overwrite=True,
        compressors=[],
    )
    index_arr[:] = rows
    shard_read.clear_shard_cache()

    def _no_tail(*_args, **_kwargs):
        raise AssertionError("stored shard index should replace the per-shard tail read")

    monkeypatch.setattr(shard_read, "_read_index_tail", _no_tail)
    group = zarr.open_group(str(path), mode="r")
    sl = slice(2, 12)
    loaded = load_channel_data(group, sl, ["t2m"])
    assert np.array_equal(loaded[:, 0], np.asarray(group["t2m"][sl]), equal_nan=True)
    assert "_shard_index" not in available_field_names(group)


def test_direct_read_falls_back_when_o_direct_fails(tmp_path, monkeypatch):
    import physicsnemo.datapipes.healpix.zarr_shard_read as shard_read

    def _no_direct(*_args, **_kwargs):
        raise OSError("O_DIRECT refused")

    monkeypatch.setattr(shard_read, "_pread_direct", _no_direct)
    data = _make_sharded_field(tmp_path / "shard")
    group = zarr.open_group(str(tmp_path / "shard"), mode="r")
    sl = slice(2, 12)
    loaded = load_channel_data(group, sl, ["t2m"])
    assert np.array_equal(loaded[:, 0], data[sl], equal_nan=True)


def test_load_windowed_channel_data_ic_diagnostics(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named", t=8)
    time_sl = slice(0, 6)
    input_time_idx = np.asarray([[0], [1]], dtype=np.intp)
    output_time_idx = np.asarray([[2], [3]], dtype=np.intp)
    output_scaling = {
        "mean": np.zeros((1, 3, 1, 1, 1), dtype=np.float32),
        "std": np.ones((1, 3, 1, 1, 1), dtype=np.float32),
    }
    inputs, targets, ic_diag = load_windowed_channel_data(
        ds,
        time_sl,
        input_names=["t2m"],
        input_time_idx=input_time_idx,
        output_names=["t2m", "u10m", "v10m"],
        output_time_idx=output_time_idx,
        output_scaling=output_scaling,
        ic_diagnostic_names=["u10m", "v10m"],
    )
    assert ic_diag.shape[2] == 2
    staging = load_channel_data(ds, time_sl, ["t2m", "u10m", "v10m"])
    exp_ic = staging[input_time_idx[:, :, None], np.asarray([1, 2])[None, None, :]]
    np.testing.assert_array_equal(ic_diag, exp_ic)
    assert inputs.shape[2] == 1
    assert targets.shape[2] == 3


def test_TimeSeriesDataset_return_ic_diagnostics(tmp_path):
    omegaconf = pytest.importorskip("omegaconf")
    pd = pytest.importorskip("pandas")
    from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import (
        TimeSeriesDatasetZarr,
    )

    output_variables = ["tcwv", "msl", "sp"]
    dataset_path = tmp_path / "ic_diag.zarr"
    n_time, face, height, width = 12, 1, 2, 2
    data = np.zeros((n_time, 3, face, height, width), dtype=np.float32)
    for i in range(3):
        data[:, i] = float(i + 1)
    xr.Dataset(
        data_vars={
            "inputs": (("time", "channel_in", "face", "height", "width"), data),
            "lat": (("face", "height", "width"), np.zeros((face, height, width))),
            "lon": (("face", "height", "width"), np.zeros((face, height, width))),
        },
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq="6h"),
            "channel_in": output_variables,
            "face": np.arange(face),
            "height": np.arange(height),
            "width": np.arange(width),
        },
    ).to_zarr(dataset_path)
    scaling = omegaconf.DictConfig(
        {name: {"mean": 0.0, "std": 1.0} for name in output_variables}
    )
    common = dict(
        dataset_path=str(dataset_path),
        data_time_step="6h",
        time_step="6h",
        gap="6h",
        scaling=scaling,
        input_variables=["tcwv"],
        output_variables=output_variables,
        start_date="1979-01-01",
        end_date="1979-01-03",
        batch_size=1,
        input_time_dim=1,
        output_time_dim=1,
    )
    dataset = TimeSeriesDatasetZarr(**common, return_ic_diagnostics=True)
    assert dataset.ic_diagnostic_variables == ["msl", "sp"]
    inputs, targets, ic_diag = dataset[0]
    assert ic_diag.shape[3] == 2
    assert float(ic_diag[0, 0, -1, 0].mean()) == pytest.approx(2.0)
    assert float(ic_diag[0, 0, -1, 1].mean()) == pytest.approx(3.0)
    assert inputs[0].dtype == np.float32
    assert ic_diag.dtype == np.float32
    batch = TimeSeriesDatasetZarr(**common, return_ic_diagnostics=False)[0]
    assert len(batch) == 2
