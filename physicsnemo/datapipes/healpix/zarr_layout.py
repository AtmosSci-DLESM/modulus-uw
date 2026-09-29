"""Helpers for healpix Zarr layouts (stacked vs per-variable).

- **stacked** — prognostic fields packed on a channel axis in ``inputs`` /
  ``constants``, with names in ``channel_in`` / ``channel_c``.
  Detection: ``is_stacked_layout``.
- **per-variable** — one Zarr array per field, root attribute
  ``layout=per_variable``. Detection: ``is_per_variable_layout``.

Catalogs already written with ``layout=named_arrays_healpix`` still load.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

COORD_KEYS = frozenset({"time", "face", "height", "width", "lat", "lon"})

# Root attribute written by current converters.
PER_VARIABLE_LAYOUT_ATTR = "per_variable"
# Written by earlier converters and the b24 builder.
_LEGACY_PER_VARIABLE_LAYOUT_ATTR = "named_arrays_healpix"

_zarrs_pipeline_enabled = False
_zarrs_import_warned = False


def enable_zarrs_pipeline() -> bool:
    """Select the zarrs Rust codec pipeline when the optional package is installed.

    zarr.config is per-process. Call from per-variable read helpers so a
    num_workers=0 load still uses zarrs. Do not call this on stacked reads, and
    do not enable it in the parent before forking DataLoader workers: fork
    inherits zarrs state and can deadlock the first batch fetch.
    """
    global _zarrs_pipeline_enabled, _zarrs_import_warned
    if _zarrs_pipeline_enabled:
        return True
    try:
        import zarr

        import zarrs  # noqa: F401

        zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
        _zarrs_pipeline_enabled = True
        return True
    except ImportError:
        if not _zarrs_import_warned:
            logger.warning(
                "zarrs is not installed; per-variable Zarr reads use the default "
                "Python codec pipeline (slower). Install with: pip install zarrs"
            )
            _zarrs_import_warned = True
        return False


def _layout_attr(ds) -> str:
    attrs = getattr(ds, "attrs", None) or {}
    if not hasattr(attrs, "get"):
        return ""
    return str(attrs.get("layout", "") or "")


def _has_per_variable_layout_attr(ds) -> bool:
    layout = _layout_attr(ds)
    return (
        layout == PER_VARIABLE_LAYOUT_ATTR
        or layout == _LEGACY_PER_VARIABLE_LAYOUT_ATTR
        or layout.startswith("per_variable")
    )


def is_stacked_layout(ds) -> bool:
    """True when prognostic fields are packed on a channel axis in ``inputs``."""
    return "inputs" in ds


def is_per_variable_layout(ds) -> bool:
    """True when the root ``layout`` attribute marks one array per field.

    A store without ``inputs`` is not per-variable unless it declares
    ``layout=per_variable`` (or the legacy ``named_arrays_healpix`` value).
    """
    if is_stacked_layout(ds):
        return False
    return _has_per_variable_layout_attr(ds)


def constants_are_stacked(ds) -> bool:
    """True when constants are a ``constants`` array indexed by ``channel_c``.

    Per-variable catalogs (``layout=per_variable`` or the older
    ``named_arrays_healpix``) store each constant as a top-level array.
    A store that still has a ``constants`` array and does not declare a
    per-variable layout keeps the stacked read.
    """
    if is_stacked_layout(ds):
        return True
    if _has_per_variable_layout_attr(ds):
        return False
    return "constants" in ds


def resolve_mask_field(
    ds,
    data_var: str,
    selection_dict: Mapping[str, Any] | None = None,
):
    """Resolve a spatial mask field from stacked or per-variable stores.

    Stacked stores use ``data_var`` (e.g. ``constants``) with optional
    ``selection_dict`` (e.g. ``channel_c``). Per-variable stores expose each
    constant as a top-level array; configs still use ``data_var: constants`` and
    ``selection_dict.channel_c`` to pick the field name.
    """
    sel = dict(selection_dict or {})
    if data_var in ds.data_vars:
        field = ds[data_var]
        if sel:
            field = field.sel(**sel)
        return field

    if is_per_variable_layout(ds) and data_var == "constants" and "channel_c" in sel:
        field_name = str(sel.pop("channel_c"))
        if field_name not in ds.data_vars:
            raise KeyError(
                f"Mask field {field_name!r} not found in dataset; "
                "per-variable stores expose constants as top-level arrays."
            )
        field = ds[field_name]
        if sel:
            field = field.sel(**sel)
        return field

    raise KeyError(
        f"No variable named {data_var!r} in dataset"
        + (f" (available: {list(ds.data_vars)!r})" if hasattr(ds, "data_vars") else "")
    )


def available_field_names(ds) -> set[str]:
    """Field names loadable from this store (prognostic + constant)."""
    if is_stacked_layout(ds):
        names = {str(x) for x in np.asarray(ds["channel_in"][:])}
        if "channel_out" in ds:
            names.update(str(x) for x in np.asarray(ds["channel_out"][:]))
        if "channel_c" in ds:
            names.update(str(x) for x in np.asarray(ds["channel_c"][:]))
        return names
    if not is_per_variable_layout(ds):
        return set()
    # ``_shard_index`` is loader metadata, not a prognostic field.
    return {
        str(k)
        for k in ds.keys()
        if k not in COORD_KEYS and k != "constants" and not str(k).startswith("_")
    }


def _stacked_channel_indices(ds, field_names: Sequence[str]) -> list[int]:
    cin = [str(x) for x in np.asarray(ds["channel_in"][:])]
    return [cin.index(n) for n in field_names]


def _slice_length(dim_size: int, time_sl) -> int:
    if isinstance(time_sl, slice):
        start = 0 if time_sl.start is None else time_sl.start
        stop = dim_size if time_sl.stop is None else time_sl.stop
        step = 1 if time_sl.step is None else time_sl.step
        return len(range(start, stop, step))
    return len(time_sl)


def _apply_channel_scaling(block: np.ndarray, scaling: Mapping, channel: int) -> np.ndarray:
    """In-place ``(x - mean) / std`` on one field, keeping ``block.dtype``.

    ``scaling`` mean/std are shaped ``(1, C, 1, 1, 1)`` in channel order.
    Casting the stats to ``block.dtype`` matches the stacked in-place scale
    used by the previous joint ``inputs`` read.
    """
    mean = np.asarray(scaling["mean"][0, channel], dtype=block.dtype)
    std = np.asarray(scaling["std"][0, channel], dtype=block.dtype)
    block -= mean
    block /= std
    return block


def _run_parallel(fns: list):
    """Run one callable per field. A single field runs on this thread."""
    if len(fns) <= 1:
        return [fn() for fn in fns]
    with ThreadPoolExecutor(max_workers=len(fns)) as ex:
        return list(ex.map(lambda fn: fn(), fns))


def _require_layout(ds) -> None:
    if is_stacked_layout(ds) or is_per_variable_layout(ds):
        return
    raise ValueError(
        "Unrecognized healpix Zarr layout. Stacked stores have an 'inputs' "
        "array; per-variable stores set the root attribute layout='per_variable'."
    )


def load_channel_data(
    ds,
    time_sl,
    field_names: Sequence[str],
    scaling: Mapping | None = None,
) -> np.ndarray:
    """Load selected prognostic fields for a time window as (T, C, F, H, W)."""
    names = list(field_names)
    _require_layout(ds)
    if len(names) == 0:
        if is_per_variable_layout(ds):
            raise ValueError("empty field name list")
        return np.asarray(ds["inputs"][time_sl])[:, []]

    if is_stacked_layout(ds):
        indices = _stacked_channel_indices(ds, names)
        out = np.array(np.asarray(ds["inputs"][time_sl, indices]))
        if scaling is not None:
            for i in range(len(names)):
                _apply_channel_scaling(out[:, i], scaling, i)
        return out

    enable_zarrs_pipeline()
    ref = ds[names[0]]
    tlen = _slice_length(ref.shape[0], time_sl)
    spatial = ref.shape[1:]
    out = np.empty((tlen, len(names)) + spatial, dtype=ref.dtype)

    def _fill(i: int, n: str) -> None:
        block = np.asarray(ds[n][time_sl])
        if scaling is not None:
            block = _apply_channel_scaling(block, scaling, i)
        out[:, i] = block

    _run_parallel([lambda i=i, n=n: _fill(i, n) for i, n in enumerate(names)])
    return out


def _scale_stacked_staging(
    staging: np.ndarray,
    names: Sequence[str],
    input_names: Sequence[str],
    output_names: Sequence[str],
    input_scaling: Mapping | None,
    output_scaling: Mapping | None,
) -> None:
    """Scale a stacked ``(T, C, F, H, W)`` staging array in place.

    Stats stay in ``staging.dtype`` so a float32 store matches the previous
    in-place ``inputs`` scale. A field that is both an input and an output
    uses the input stats; both come from the same scaling dict.
    """
    if input_scaling is None and output_scaling is None:
        return
    in_pos = {n: i for i, n in enumerate(input_names)}
    out_pos = {n: i for i, n in enumerate(output_names)}
    for i, name in enumerate(names):
        if name in in_pos and input_scaling is not None:
            _apply_channel_scaling(staging[:, i], input_scaling, in_pos[name])
        elif name in out_pos and output_scaling is not None:
            _apply_channel_scaling(staging[:, i], output_scaling, out_pos[name])


def load_windowed_channel_data(
    ds,
    time_sl,
    input_names: Sequence[str],
    input_time_idx: np.ndarray,
    output_names: Sequence[str] | None = None,
    output_time_idx: np.ndarray | None = None,
    input_scaling: Mapping | None = None,
    output_scaling: Mapping | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load and scale fields directly into (B, T, C, F, H, W) sample windows.

    For per-variable stores, each unique field is decoded and scaled once on
    a worker thread, then scattered into channel-first scratch buffers. For
    stacked ``inputs`` stores, channels are read jointly, scaled on that
    ``(T, C, ...)`` array, then gathered into the same window layout.

    Parameters
    ----------
    ds :
        Open Zarr group.
    time_sl :
        Time slice covering the full batch window (indices in ``*_time_idx``
        are relative to this slice).
    input_names, output_names :
        Channel order for the returned buffers.
    input_time_idx, output_time_idx :
        Integer index arrays of shape ``(B, T)`` into the loaded time window.
    input_scaling, output_scaling :
        Optional ``{"mean", "std"}`` arrays indexed by channel position in
        ``input_names`` / ``output_names``. Mean/std are shaped
        ``(1, C, 1, 1, 1)``.

    Returns
    -------
    inputs, targets
        ``inputs`` has shape ``(B, T_in, C_in, ...)``. ``targets`` is ``None``
        when ``output_names`` is omitted.
    """
    input_names = list(input_names)
    _require_layout(ds)
    if len(input_names) == 0:
        raise ValueError("empty input field name list")
    if input_time_idx.ndim != 2:
        raise ValueError(
            f"input_time_idx must be (B, T), got shape {input_time_idx.shape}"
        )

    if is_stacked_layout(ds):
        # Scale on (T, C, F, H, W) before the window gather. Mean/std shaped
        # (1, C, 1, 1, 1) align with that axis; they do not align with
        # (B, T, C, F, H, W).
        names = list(dict.fromkeys(list(input_names) + list(output_names or [])))
        staging = np.array(load_channel_data(ds, time_sl, names, scaling=None))
        _scale_stacked_staging(
            staging,
            names,
            input_names,
            list(output_names or []),
            input_scaling,
            output_scaling,
        )
        name_to_i = {n: i for i, n in enumerate(names)}
        in_c = np.asarray([name_to_i[n] for n in input_names], dtype=np.intp)
        inputs = staging[
            input_time_idx[:, :, np.newaxis], in_c[np.newaxis, np.newaxis, :]
        ]
        targets = None
        if output_names is not None:
            if output_time_idx is None:
                raise ValueError("output_time_idx required when output_names is set")
            out_c = np.asarray(
                [name_to_i[n] for n in output_names], dtype=np.intp
            )
            targets = staging[
                output_time_idx[:, :, np.newaxis], out_c[np.newaxis, np.newaxis, :]
            ]
        return inputs, targets

    enable_zarrs_pipeline()
    batch_size, t_in = input_time_idx.shape
    ref = ds[input_names[0]]
    spatial = ref.shape[1:]
    dtype = ref.dtype
    # Channel-first scratch buffers so each per-variable scatter write is a
    # single contiguous (B, T, F, H, W) store rather than a strided channel
    # slice of a (B, T, C, ...) array.
    inputs_cf = np.empty((len(input_names), batch_size, t_in) + spatial, dtype=dtype)

    targets_cf = None
    out_names: list[str] = []
    if output_names is not None:
        if output_time_idx is None:
            raise ValueError("output_time_idx required when output_names is set")
        out_names = list(output_names)
        batch_size_out, t_out = output_time_idx.shape
        if batch_size_out != batch_size:
            raise ValueError(
                f"input/output batch mismatch: {batch_size} vs {batch_size_out}"
            )
        targets_cf = np.empty((len(out_names), batch_size, t_out) + spatial, dtype=dtype)

    # Unique fields → destinations in input and/or target channel axes.
    slots: dict[str, tuple[int | None, int | None]] = {}
    for c, name in enumerate(input_names):
        in_c, out_c = slots.get(name, (None, None))
        slots[name] = (c, out_c)
    for c, name in enumerate(out_names):
        in_c, out_c = slots.get(name, (None, None))
        slots[name] = (in_c, c)

    def _place(block: np.ndarray, in_c: int | None, out_c: int | None) -> None:
        # Scale once. Shared input/output vars use the same physical mean/std;
        # prefer input_scaling when the field appears in both buffers.
        if in_c is not None and input_scaling is not None:
            block = _apply_channel_scaling(block, input_scaling, in_c)
        elif out_c is not None and output_scaling is not None:
            block = _apply_channel_scaling(block, output_scaling, out_c)
        if in_c is not None:
            inputs_cf[in_c] = block[input_time_idx]
        if out_c is not None:
            targets_cf[out_c] = block[output_time_idx]

    def _fill(name: str, in_c: int | None, out_c: int | None) -> None:
        block = np.asarray(ds[name][time_sl])
        _place(block, in_c, out_c)

    _run_parallel(
        [lambda n=n, ic=ic, oc=oc: _fill(n, ic, oc) for n, (ic, oc) in slots.items()]
    )
    # (C, B, T, ...) -> (B, T, C, ...); view, no copy.
    inputs = np.transpose(inputs_cf, (1, 2, 0, 3, 4, 5))
    targets = None if targets_cf is None else np.transpose(targets_cf, (1, 2, 0, 3, 4, 5))
    return inputs, targets


def load_constant_fields(ds, field_names: Sequence[str]) -> np.ndarray:
    """Load constant fields as (C, F, H, W).

    Constants are read once during dataset setup in the parent process; keep the
    default codec pipeline here so zarrs is not enabled before DataLoader fork.
    """
    names = list(field_names)
    if len(names) == 0:
        raise ValueError("empty constant field name list")

    if constants_are_stacked(ds):
        cc = [str(x) for x in np.asarray(ds["channel_c"][:])]
        indices = [cc.index(n) for n in names]
        return np.asarray(ds["constants"][indices])

    if not is_per_variable_layout(ds):
        _require_layout(ds)
    parts = _run_parallel([lambda n=n: np.asarray(ds[n]) for n in names])
    return np.stack(parts, axis=0)
