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

import os

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from physicsnemo.datapipes.healpix.zarr_layout import (
    _decode_shard_index,
    _read_index_tail,
    available_field_names,
    build_shard_index_table,
    constants_are_stacked,
    enable_zarrs_pipeline,
    is_per_variable_layout,
    is_stacked_layout,
    load_channel_data,
    load_constant_fields,
    load_windowed_channel_data,
    read_sharded_time_slice,
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
    loaded = load_channel_data(ds, time_sl, ["t2m", "v10m"], n_threads=1)
    np.testing.assert_array_equal(loaded, expected)


def test_load_channel_data_per_variable_store(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named")
    time_sl = slice(0, 2)
    expected = np.stack(
        [np.asarray(ds["v10m"][time_sl]), np.asarray(ds["t2m"][time_sl])], axis=1
    )
    loaded = load_channel_data(ds, time_sl, ["v10m", "t2m"], n_threads=1)
    np.testing.assert_array_equal(loaded, expected)


def test_load_constant_fields_per_variable_store(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named")
    loaded = load_constant_fields(ds, ["lsm", "z"], n_threads=1)
    expected = np.stack([np.asarray(ds["lsm"]), np.asarray(ds["z"])], axis=0)
    np.testing.assert_array_equal(loaded, expected)


def test_load_channel_data_empty_raises_on_per_variable(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named")
    with pytest.raises(ValueError, match="empty field name list"):
        load_channel_data(ds, slice(0, 1), [], n_threads=1)


def test_load_channel_data_per_variable_store_with_scaling(tmp_path):
    ds = _make_per_variable_store(tmp_path / "named")
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


def test_load_windowed_channel_data_per_variable_store(tmp_path):
    """Option A: direct window fill matches staging[:, c][time_idx] gather."""
    ds = _make_per_variable_store(tmp_path / "named", t=8)
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
        n_threads=1,
    )
    stacked_in, stacked_out, stacked_ic = load_windowed_channel_data(
        stacked_ds, **kwargs
    )
    per_var_in, per_var_out, per_var_ic = load_windowed_channel_data(
        per_var_ds, **kwargs
    )
    np.testing.assert_allclose(stacked_in, per_var_in)
    np.testing.assert_allclose(stacked_out, per_var_out)
    assert stacked_ic is None and per_var_ic is None


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


def test_load_windowed_channel_data_ic_diagnostics(tmp_path):
    """IC diagnostics are output-only channels gathered at input times."""
    ds = _make_per_variable_store(tmp_path / "named", t=8)
    time_sl = slice(0, 6)
    input_names = ["t2m"]
    output_names = ["t2m", "u10m", "v10m"]
    ic_diag_names = ["u10m", "v10m"]
    input_time_idx = np.asarray([[0], [1]], dtype=np.intp)
    output_time_idx = np.asarray([[2], [3]], dtype=np.intp)
    output_scaling = {
        "mean": np.zeros((1, 3, 1, 1, 1), dtype=np.float32),
        "std": np.ones((1, 3, 1, 1, 1), dtype=np.float32),
    }
    inputs, targets, ic_diag = load_windowed_channel_data(
        ds,
        time_sl,
        input_names=input_names,
        input_time_idx=input_time_idx,
        output_names=output_names,
        output_time_idx=output_time_idx,
        output_scaling=output_scaling,
        ic_diagnostic_names=ic_diag_names,
    )
    assert ic_diag is not None
    assert ic_diag.shape[2] == 2
    staging = load_channel_data(ds, time_sl, ["t2m", "u10m", "v10m"], n_threads=1)
    exp_ic = staging[input_time_idx[:, :, None], np.asarray([1, 2])[None, None, :]]
    np.testing.assert_array_equal(ic_diag, exp_ic)
    assert inputs.shape[2] == 1
    assert targets.shape[2] == 3


def test_TimeSeriesDataset_return_ic_diagnostics(tmp_path):
    """Train mode can return output-only channels at input times for soft constraints."""
    omegaconf = pytest.importorskip("omegaconf")
    pd = pytest.importorskip("pandas")
    from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import (
        TimeSeriesDatasetZarr,
    )

    input_variables = ["tcwv"]
    output_variables = ["tcwv", "msl", "sp"]
    dataset_path = tmp_path / "ic_diag.zarr"
    n_time = 12
    face, height, width = 1, 2, 2
    times = pd.date_range("1979-01-01", periods=n_time, freq="6h")
    n_chan = len(output_variables)
    data = np.zeros((n_time, n_chan, face, height, width), dtype=np.float32)
    for i in range(n_chan):
        data[:, i] = float(i + 1)
    ds = xr.Dataset(
        data_vars={
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                data,
            ),
            "targets": (
                ("time", "channel_out", "face", "height", "width"),
                data.copy(),
            ),
            "lat": (("face", "height", "width"), np.zeros((face, height, width))),
            "lon": (("face", "height", "width"), np.zeros((face, height, width))),
            "face": ("face", np.arange(face)),
            "height": ("height", np.arange(height)),
            "width": ("width", np.arange(width)),
        },
        coords={
            "time": times,
            "channel_in": output_variables,
            "channel_out": output_variables,
        },
    )
    ds.to_zarr(dataset_path)

    scaling = omegaconf.DictConfig(
        {
            "tcwv": {"mean": 0.0, "std": 1.0},
            "msl": {"mean": 0.0, "std": 1.0},
            "sp": {"mean": 0.0, "std": 1.0},
        }
    )
    dataset = TimeSeriesDatasetZarr(
        dataset_path=str(dataset_path),
        data_time_step="6h",
        time_step="6h",
        gap="6h",
        scaling=scaling,
        input_variables=input_variables,
        output_variables=output_variables,
        start_date="1979-01-01",
        end_date="1979-01-03",
        batch_size=1,
        input_time_dim=1,
        output_time_dim=1,
        return_ic_diagnostics=True,
    )
    assert dataset.ic_diagnostic_variables == ["msl", "sp"]
    inputs, targets, ic_diag = dataset[0]
    assert ic_diag is not None
    assert ic_diag.shape[3] == 2
    assert float(ic_diag[0, 0, -1, 0].mean()) == pytest.approx(2.0)
    assert float(ic_diag[0, 0, -1, 1].mean()) == pytest.approx(3.0)
    assert inputs[0].shape[3] == 1
    assert float(inputs[0][0, 0, -1, 0].mean()) == pytest.approx(1.0)

    batch = TimeSeriesDatasetZarr(
        dataset_path=str(dataset_path),
        data_time_step="6h",
        time_step="6h",
        gap="6h",
        scaling=scaling,
        input_variables=input_variables,
        output_variables=output_variables,
        start_date="1979-01-01",
        end_date="1979-01-03",
        batch_size=1,
        input_time_dim=1,
        output_time_dim=1,
        return_ic_diagnostics=False,
    )[0]
    assert len(batch) == 2


def _make_sharded_field(path, *, t: int = 20, shard_time: int = 8, f: int = 2, h: int = 2, w: int = 2):
    """Per-variable array with inner time chunk 1 and a time shard, zstd level 0."""
    from zarr.codecs import ZstdCodec

    data = np.arange(t * f * h * w, dtype=np.float32).reshape(t, f, h, w)
    # A NaN chunk forces an empty slot so a window can straddle a gap.
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
    return data


def test_read_sharded_time_slice_matches_array(tmp_path):
    data = _make_sharded_field(tmp_path / "shard")
    array = zarr.open_array(f"{tmp_path / 'shard'}/t2m", mode="r")
    # Crosses the shard boundary at t=8 and includes the NaN sample at t=3.
    sl = slice(2, 12)
    got = read_sharded_time_slice(array, sl)
    assert got is not None
    ref = np.asarray(array[sl])
    assert got.shape == ref.shape
    assert np.array_equal(got, ref, equal_nan=True)
    loaded = load_channel_data(zarr.open_group(str(tmp_path / "shard"), mode="r"), sl, ["t2m"])
    assert np.array_equal(loaded[:, 0], ref, equal_nan=True)


def test_sharded_gather_window_matches_arrays(tmp_path, monkeypatch):
    """Every field of a sharded window is loaded together and matches array slices."""
    import physicsnemo.datapipes.healpix.zarr_layout as zarr_layout

    def _forbid_thread_pool(*_args, **_kwargs):
        raise AssertionError("sharded fields must use the direct reader, not the small thread pool")

    monkeypatch.setattr(zarr_layout, "_run_loaders_parallel", _forbid_thread_pool)
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
    group = zarr.open_group(str(path), mode="r")
    sl = slice(2, 12)
    loaded = load_channel_data(group, sl, ["t2m", "u10m"])
    for c, name in enumerate(("t2m", "u10m")):
        assert np.array_equal(loaded[:, c], stored[name][sl], equal_nan=True)
    idx_in = np.array([[0, 1], [2, 3]])
    idx_out = np.array([[4, 6], [5, 7]])
    inputs, targets, _ic = load_windowed_channel_data(
        group, sl, ["t2m", "u10m"], idx_in, ["u10m", "t2m"], idx_out
    )
    for c, name in enumerate(("t2m", "u10m")):
        assert np.array_equal(inputs[:, :, c], stored[name][sl][idx_in], equal_nan=True)
    for c, name in enumerate(("u10m", "t2m")):
        assert np.array_equal(targets[:, :, c], stored[name][sl][idx_out], equal_nan=True)


def _shard_index_nbytes(array) -> int:
    shard_t = int(tuple(array.shards)[0])
    inner_t = int(tuple(array.chunks)[0])
    return (shard_t // inner_t) * 16 + 4


def test_shard_index_table_matches_decoded_indexes(tmp_path):
    path = tmp_path / "multi"
    _make_sharded_field(path, t=20, shard_time=8)
    from zarr.codecs import ZstdCodec

    data = (np.arange(20 * 2 * 2 * 2, dtype=np.float32) + 1000).reshape(20, 2, 2, 2)
    data[3] = np.nan
    zarr.create_array(
        store=str(path),
        name="u10m",
        shape=data.shape,
        chunks=(1, 2, 2, 2),
        shards=(8, 2, 2, 2),
        dtype="float32",
        zarr_format=3,
        compressors=[ZstdCodec(level=0)],
        fill_value=np.nan,
        dimension_names=["time", "face", "height", "width"],
    )
    zarr.open_array(f"{path}/u10m", mode="a")[:] = data
    group = zarr.open_group(str(path), mode="a")
    written = build_shard_index_table(group, workers=2)
    assert set(written) == {"t2m", "u10m"}
    # Second pass: the store is now consolidated, which is how a real catalog is opened.
    written_again = build_shard_index_table(zarr.open_group(str(path), mode="a"), workers=2)
    assert set(written_again) == {"t2m", "u10m"}
    assert "_shard_index" not in available_field_names(group)
    group = zarr.open_group(str(path), mode="r")
    for name in ("t2m", "u10m"):
        array = group[name]
        table = np.asarray(group["_shard_index"][name][:])
        n_shards = int(np.ceil(array.shape[0] / array.shards[0]))
        cps = int(array.shards[0] // array.chunks[0])
        assert table.shape == (n_shards, cps, 2)
        nbytes = _shard_index_nbytes(array)
        root = str(array.store.root)
        for shard_i in range(n_shards):
            shard_path = os.path.join(root, name, "c", str(shard_i), "0", "0", "0")
            decoded = _decode_shard_index(_read_index_tail(shard_path, nbytes), cps)
            assert np.array_equal(table[shard_i], decoded)
    sl = slice(2, 12)
    loaded = load_channel_data(group, sl, ["t2m", "u10m"])
    for c, name in enumerate(("t2m", "u10m")):
        assert np.array_equal(loaded[:, c], np.asarray(group[name][sl]), equal_nan=True)


def test_direct_read_falls_back_when_o_direct_fails(tmp_path, monkeypatch):
    import physicsnemo.datapipes.healpix.zarr_layout as zarr_layout

    def _no_direct(*_args, **_kwargs):
        raise OSError("O_DIRECT refused")

    monkeypatch.setattr(zarr_layout, "_pread_direct", _no_direct)
    data = _make_sharded_field(tmp_path / "shard")
    group = zarr.open_group(str(tmp_path / "shard"), mode="r")
    sl = slice(2, 12)
    loaded = load_channel_data(group, sl, ["t2m"])
    assert np.array_equal(loaded[:, 0], data[sl], equal_nan=True)
