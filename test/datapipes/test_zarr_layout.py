"""Unit tests for healpix Zarr layout helpers."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from physicsnemo.datapipes.healpix.zarr_layout import (
    available_field_names,
    constants_are_stacked,
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
    path, *, t: int = 4, f: int = 12, h: int = 4, w: int = 4, attrs: dict | None = None
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
        attrs=attrs or {},
    )
    ds.to_zarr(path, mode="w")
    return zarr.open_group(str(path), mode="r")


def test_layout_detection(tmp_path):
    """The layout comes from the store's arrays; root attributes are ignored."""
    stacked = _make_stacked_store(tmp_path / "stacked")
    per_var = _make_per_variable_store(tmp_path / "per_var")
    # Catalogs carry assorted attributes, including an unrelated ``layout``.
    tagged = _make_per_variable_store(
        tmp_path / "tagged", attrs={"layout": "named_arrays_healpix", "source": "x"}
    )
    assert is_stacked_layout(stacked) is True
    assert is_per_variable_layout(stacked) is False
    assert constants_are_stacked(stacked) is True
    for ds in (per_var, tagged):
        assert is_per_variable_layout(ds) is True
        assert is_stacked_layout(ds) is False
        assert constants_are_stacked(ds) is False


def test_available_field_names_per_variable_store(tmp_path):
    named = _make_per_variable_store(tmp_path / "per_var")
    assert available_field_names(named) == {"t2m", "u10m", "v10m", "lsm", "z"}


def test_load_channel_data_stacked_by_name(tmp_path):
    ds = _make_stacked_store(tmp_path / "stacked")
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


def _make_sharded_store(path, names=("t2m", "u10m"), *, t=20, shard_time=8, f=2, h=2, w=2):
    """Per-variable fields with one-step inner chunks in time shards (the training store layout)."""
    from zarr.codecs import ZstdCodec

    stored = {}
    for i, name in enumerate(names):
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
    return zarr.open_group(str(path), mode="r"), stored


def test_sharded_per_variable_windows_match_arrays(tmp_path):
    group, stored = _make_sharded_store(tmp_path / "sharded")
    # Inside one shard, across a shard boundary, and the final partial shard.
    for sl in (slice(1, 6), slice(2, 12), slice(14, 20)):
        loaded = load_channel_data(group, sl, ["t2m", "u10m"])
        for c, name in enumerate(("t2m", "u10m")):
            assert np.array_equal(loaded[:, c], stored[name][sl], equal_nan=True)
    sl = slice(2, 12)
    idx_in = np.array([[0, 1], [2, 3]])
    idx_out = np.array([[4, 6], [5, 7]])
    inputs, targets = load_windowed_channel_data(
        group, sl, ["t2m", "u10m"], idx_in, ["u10m", "t2m"], idx_out
    )
    for c, name in enumerate(("t2m", "u10m")):
        assert np.array_equal(inputs[:, :, c], stored[name][sl][idx_in], equal_nan=True)
    for c, name in enumerate(("u10m", "t2m")):
        assert np.array_equal(targets[:, :, c], stored[name][sl][idx_out], equal_nan=True)


def test_open_healpix_zarr_positional_reads_match(tmp_path):
    from physicsnemo.datapipes.healpix.zarr_layout import open_healpix_zarr

    _, stored = _make_sharded_store(tmp_path / "sharded")
    group = open_healpix_zarr(str(tmp_path / "sharded"))
    assert type(group.store).__name__ == "_PositionalReadLocalStore"
    for sl in (slice(1, 6), slice(2, 12), slice(14, 20)):
        loaded = load_channel_data(group, sl, ["t2m", "u10m"])
        for c, name in enumerate(("t2m", "u10m")):
            assert np.array_equal(loaded[:, c], stored[name][sl], equal_nan=True)


def test_dropping_read_pages_drops_only_inside_the_block(tmp_path, monkeypatch):
    import os

    from physicsnemo.datapipes.healpix import zarr_layout

    _, stored = _make_sharded_store(tmp_path / "sharded")
    group = zarr_layout.open_healpix_zarr(str(tmp_path / "sharded"))
    advice = []
    real = os.posix_fadvise
    monkeypatch.setattr(
        zarr_layout.os,
        "posix_fadvise",
        lambda fd, offset, length, flag: (advice.append(flag), real(fd, offset, length, flag)),
    )
    load_channel_data(group, slice(2, 12), ["t2m"])
    assert advice == []
    with zarr_layout.dropping_read_pages():
        loaded = load_channel_data(group, slice(2, 12), ["t2m"])
    assert np.array_equal(loaded[:, 0], stored["t2m"][2:12], equal_nan=True)
    assert advice and set(advice) == {os.POSIX_FADV_RANDOM, os.POSIX_FADV_DONTNEED}
    n = len(advice)
    load_channel_data(group, slice(2, 12), ["t2m"])
    assert len(advice) == n  # back to cached reads after the block


def test_per_variable_arrays_use_fused_pipeline(tmp_path):
    """Per-variable arrays decode through FusedCodecPipeline on zarr >= 3.3.

    Without it reads fall back to the active pipeline, which is correct but
    slower; this fails if a zarr release renames or drops the pipeline.
    """
    from packaging.version import Version

    from physicsnemo.datapipes.healpix.zarr_layout import _async_array

    if Version(zarr.__version__) < Version("3.3"):
        pytest.skip("FusedCodecPipeline needs zarr >= 3.3")
    group, _ = _make_sharded_store(tmp_path / "sharded")
    pipeline = type(_async_array(group, "t2m").codec_pipeline).__name__
    assert pipeline == "FusedCodecPipeline", f"per-variable reads use {pipeline}"


def _read_in_child(queue, path):
    group = zarr.open_group(path, mode="r")
    queue.put(float(np.nansum(load_channel_data(group, slice(8, 16), ["t2m", "u10m"]))))


def test_per_variable_reads_work_after_fork(tmp_path):
    """DataLoader workers fork after the parent has read (constants, coupled fields)."""
    import multiprocessing as mp

    group, stored = _make_sharded_store(tmp_path / "sharded")
    load_channel_data(group, slice(0, 8), ["t2m", "u10m"])  # parent reads first
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_read_in_child, args=(queue, str(tmp_path / "sharded")))
    child.start()
    child.join(60)
    hung = child.is_alive()
    if hung:
        child.kill()
    assert not hung, "per-variable read hung in a forked child"
    expected = float(np.nansum(stored["t2m"][8:16]) + np.nansum(stored["u10m"][8:16]))
    assert queue.get(timeout=5) == expected


def test_resolve_mask_field_per_variable_store(tmp_path):
    """``channel_c`` selects a top-level array on a per-variable store.

    The coupled stem calls this helper when it is installed.
    """
    f, h, w = 12, 4, 4
    land = np.zeros((f, h, w), dtype=np.float32)
    land[0::2] = 1.0
    ds = xr.Dataset(
        {"lsm": (("face", "height", "width"), land)},
        coords=_spatial_coords(f=f, h=h, w=w),
    )
    path = tmp_path / "mask_per_variable.zarr"
    ds.to_zarr(path, mode="w")

    opened = xr.open_zarr(path)
    field = resolve_mask_field(opened, "constants", {"channel_c": "lsm"})
    np.testing.assert_allclose(field.values, land)
    opened.close()


def test_per_variable_detected_without_layout_attr(tmp_path):
    bare = _make_per_variable_store(tmp_path / "bare")
    assert dict(bare.attrs) == {}
    assert is_per_variable_layout(bare) is True
    np.testing.assert_array_equal(
        load_channel_data(bare, slice(0, 2), ["u10m"]), np.full((2, 1, 12, 4, 4), 2.0)
    )


def test_warns_once_when_fused_pipeline_is_missing(tmp_path, monkeypatch):
    """Per-variable reads without FusedCodecPipeline (zarr < 3.3) warn once per process."""
    import warnings

    from physicsnemo.datapipes.healpix import zarr_layout

    def slow_read_warnings() -> list:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_channel_data(ds, slice(0, 2), ["t2m"])
        return [w for w in caught if "zarr >= 3.3" in str(w.message)]

    ds = _make_per_variable_store(tmp_path / "per_var")
    monkeypatch.setattr(zarr_layout, "_warned_slow_reads", False)
    assert slow_read_warnings() == []  # FusedCodecPipeline available

    monkeypatch.setattr(zarr_layout, "_READ_CONFIG", {})
    assert len(slow_read_warnings()) == 1
    assert slow_read_warnings() == []  # once per process


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

