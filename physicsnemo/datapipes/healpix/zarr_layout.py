"""Helpers for healpix Zarr layouts (monolithic inputs vs per-variable arrays)."""

from __future__ import annotations

import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

COORD_KEYS = frozenset({"time", "face", "height", "width", "lat", "lon"})

# Process-local pool for DataLoader workers (set via init_worker_pool in worker_init_fn).
_worker_pool: ThreadPoolExecutor | None = None
_worker_pool_n_threads: int = 0
_worker_pool_lock = threading.Lock()

_zarrs_pipeline_enabled = False
_zarrs_import_warned = False

def enable_zarrs_pipeline() -> bool:
    """Select the zarrs Rust codec pipeline when the optional package is installed.

    zarr.config is per-process. Call from DataLoader worker_init_fn and from read
    helpers (load_channel_data) so num_workers=0 still gets zarrs. Do not enable
    in the parent process before forking DataLoader workers: fork inherits zarrs
    state and can deadlock the first batch fetch.
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


def maybe_collect_worker_gc() -> None:
    """No-op kept for call-site compatibility.

    DataLoader workers disable cyclic GC in ``worker_init_fn`` and rely on
    refcounting for large numpy buffers. A previous periodic ``gc.collect()``
    here caused multi-hundred-ms stalls every few dozen batches; do not
    reintroduce full collections on the getitem path.
    """
    return


def init_worker_pool(n_threads: int = 8) -> None:
    """Create a persistent thread pool in the current process (DataLoader worker)."""
    global _worker_pool, _worker_pool_n_threads
    if n_threads <= 1:
        return
    with _worker_pool_lock:
        if _worker_pool is not None:
            if _worker_pool_n_threads == n_threads:
                return
            _worker_pool.shutdown(wait=False)
            _worker_pool = None
        _worker_pool = ThreadPoolExecutor(max_workers=n_threads)
        _worker_pool_n_threads = n_threads


def shutdown_worker_pool(wait: bool = True) -> None:
    """Tear down the process-local pool (tests / worker shutdown)."""
    global _worker_pool, _worker_pool_n_threads
    with _worker_pool_lock:
        if _worker_pool is not None:
            _worker_pool.shutdown(wait=wait)
            _worker_pool = None
            _worker_pool_n_threads = 0


def worker_pool_active() -> bool:
    """True when a persistent pool is installed in this process."""
    return _worker_pool is not None


def _atexit_shutdown() -> None:
    shutdown_worker_pool(wait=False)


atexit.register(_atexit_shutdown)


def is_monolithic_layout(ds) -> bool:
    """True when store uses a single ``inputs`` array."""
    return "inputs" in ds


def is_named_arrays_layout(ds) -> bool:
    """True for name-only per-field stores (no channel_* metadata arrays)."""
    attrs = getattr(ds, "attrs", None) or {}
    layout = attrs.get("layout", "") if hasattr(attrs, "get") else ""
    return layout == "named_arrays_healpix"


def is_per_variable_layout(ds) -> bool:
    """True when prognostic fields are stored as separate arrays (not monolithic)."""
    if is_monolithic_layout(ds):
        return False
    if is_named_arrays_layout(ds):
        return True
    attrs = getattr(ds, "attrs", None) or {}
    layout = attrs.get("layout", "") if hasattr(attrs, "get") else ""
    if layout == "per_variable_healpix" or str(layout).startswith("per_variable"):
        return True
    return True


def resolve_mask_field(
    ds,
    data_var: str,
    selection_dict: Mapping[str, Any] | None = None,
):
    """Resolve a spatial mask field from monolithic or named_arrays_healpix stores.

    Monolithic stores use ``data_var`` (e.g. ``constants``) with optional
    ``selection_dict`` (e.g. ``channel_c``). Named-array stores expose each
    constant as a top-level array; configs still use ``data_var: constants`` and
    ``selection_dict.channel_c`` to pick the field name.
    """
    sel = dict(selection_dict or {})
    if data_var in ds.data_vars:
        field = ds[data_var]
        if sel:
            field = field.sel(**sel)
        return field

    if is_named_arrays_layout(ds) and data_var == "constants" and "channel_c" in sel:
        field_name = str(sel.pop("channel_c"))
        if field_name not in ds.data_vars:
            raise KeyError(
                f"Mask field {field_name!r} not found in dataset; "
                "named_arrays_healpix stores expose constants as top-level arrays."
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
    if is_monolithic_layout(ds):
        names = {str(x) for x in np.asarray(ds["channel_in"][:])}
        if "channel_out" in ds:
            names.update(str(x) for x in np.asarray(ds["channel_out"][:]))
        if "channel_c" in ds:
            names.update(str(x) for x in np.asarray(ds["channel_c"][:]))
        return names
    return {str(k) for k in ds.keys() if k not in COORD_KEYS and k != "constants"}


def _monolithic_channel_indices(ds, field_names: Sequence[str]) -> list[int]:
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
    block -= scaling["mean"][0, channel]
    block /= scaling["std"][0, channel]
    return block


def _run_loaders_parallel(loaders: list, n_threads: int) -> None:
    if not loaders:
        return
    if n_threads <= 1:
        for fn in loaders:
            fn()
        return
    if _worker_pool is not None:
        list(_worker_pool.map(lambda fn: fn(), loaders))
        return
    with ThreadPoolExecutor(max_workers=min(n_threads, len(loaders))) as ex:
        list(ex.map(lambda fn: fn(), loaders))


def _load_fields_parallel(
    loaders: list,
    n_threads: int,
) -> list[np.ndarray]:
    if n_threads <= 1:
        return [fn() for fn in loaders]
    if _worker_pool is not None:
        return list(_worker_pool.map(lambda fn: fn(), loaders))
    with ThreadPoolExecutor(max_workers=min(n_threads, len(loaders))) as ex:
        return list(ex.map(lambda fn: fn(), loaders))


def load_channel_data(
    ds,
    time_sl,
    field_names: Sequence[str],
    n_threads: int = 8,
    scaling: Mapping | None = None,
) -> np.ndarray:
    """Load selected prognostic fields for a time window as (T, C, F, H, W)."""
    enable_zarrs_pipeline()
    names = list(field_names)
    if len(names) == 0:
        if is_per_variable_layout(ds):
            raise ValueError("empty field name list")
        return np.asarray(ds["inputs"][time_sl])[:, []]

    if is_monolithic_layout(ds):
        indices = _monolithic_channel_indices(ds, names)
        out = np.asarray(ds["inputs"][time_sl, indices])
        if scaling is not None:
            out -= scaling["mean"]
            out /= scaling["std"]
        return out

    ref = ds[names[0]]
    tlen = _slice_length(ref.shape[0], time_sl)
    spatial = ref.shape[1:]
    out = np.empty((tlen, len(names)) + spatial, dtype=ref.dtype)

    def _fill(i: int, n: str) -> None:
        block = np.asarray(ds[n][time_sl])
        if scaling is not None:
            block = _apply_channel_scaling(block, scaling, i)
        out[:, i] = block

    loaders = [lambda i=i, n=n: _fill(i, n) for i, n in enumerate(names)]
    _run_loaders_parallel(loaders, n_threads)
    return out


def load_windowed_channel_data(
    ds,
    time_sl,
    input_names: Sequence[str],
    input_time_idx: np.ndarray,
    output_names: Sequence[str] | None = None,
    output_time_idx: np.ndarray | None = None,
    n_threads: int = 8,
    input_scaling: Mapping | None = None,
    output_scaling: Mapping | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load and scale fields directly into (B, T, C, F, H, W) sample windows.

    For per-variable / named-array stores, each unique field is decoded and
    scaled once on a worker thread, then scattered into channel-first scratch
    buffers. For monolithic ``inputs`` stores, channels are read jointly and
    gathered into the same window layout.

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
        ``input_names`` / ``output_names`` (same layout as dataset scaling).

    Returns
    -------
    inputs, targets
        ``inputs`` has shape ``(B, T_in, C_in, ...)``. ``targets`` is ``None``
        when ``output_names`` is omitted.
    """
    enable_zarrs_pipeline()
    input_names = list(input_names)
    if len(input_names) == 0:
        raise ValueError("empty input field name list")
    if input_time_idx.ndim != 2:
        raise ValueError(
            f"input_time_idx must be (B, T), got shape {input_time_idx.shape}"
        )

    if is_monolithic_layout(ds):
        # Monolithic stores are already a joint array; stage once then gather.
        names = list(dict.fromkeys(list(input_names) + list(output_names or [])))
        staging = load_channel_data(
            ds, time_sl, names, n_threads=n_threads, scaling=None
        )
        name_to_i = {n: i for i, n in enumerate(names)}
        in_c = np.asarray([name_to_i[n] for n in input_names], dtype=np.intp)
        inputs = staging[
            input_time_idx[:, :, np.newaxis], in_c[np.newaxis, np.newaxis, :]
        ]
        if input_scaling is not None:
            inputs = (inputs - input_scaling["mean"]) / input_scaling["std"]
        targets = None
        if output_names is not None:
            if output_time_idx is None:
                raise ValueError("output_time_idx required when output_names is set")
            out_c = np.asarray([name_to_i[n] for n in output_names], dtype=np.intp)
            targets = staging[
                output_time_idx[:, :, np.newaxis], out_c[np.newaxis, np.newaxis, :]
            ]
            if output_scaling is not None:
                targets = (targets - output_scaling["mean"]) / output_scaling["std"]
        return inputs, targets

    ref = ds[input_names[0]]
    spatial = ref.shape[1:]
    dtype = ref.dtype
    batch_size, t_in = input_time_idx.shape
    # Channel-first scratch buffers so each per-variable scatter write is a
    # single contiguous (B, T, F, H, W) store rather than a strided channel
    # slice of a (B, T, C, ...) array.
    inputs_cf = np.empty(
        (len(input_names), batch_size, t_in) + spatial, dtype=dtype
    )

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
        targets_cf = np.empty(
            (len(out_names), batch_size, t_out) + spatial, dtype=dtype
        )

    # Unique fields → destinations in input and/or target channel axes.
    slots: dict[str, tuple[int | None, int | None]] = {}
    for c, name in enumerate(input_names):
        in_c, out_c = slots.get(name, (None, None))
        slots[name] = (c, out_c)
    for c, name in enumerate(out_names):
        in_c, out_c = slots.get(name, (None, None))
        slots[name] = (in_c, c)

    def _fill(name: str, in_c: int | None, out_c: int | None) -> None:
        block = np.asarray(ds[name][time_sl])
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

    loaders = [
        lambda n=n, ic=ic, oc=oc: _fill(n, ic, oc) for n, (ic, oc) in slots.items()
    ]
    _run_loaders_parallel(loaders, n_threads)
    # (C, B, T, ...) -> (B, T, C, ...); view, no copy.
    inputs = np.transpose(inputs_cf, (1, 2, 0, 3, 4, 5))
    targets = (
        None if targets_cf is None else np.transpose(targets_cf, (1, 2, 0, 3, 4, 5))
    )
    return inputs, targets


def load_constant_fields(
    ds, field_names: Sequence[str], n_threads: int = 8
) -> np.ndarray:
    """Load constant fields as (C, F, H, W).

    Constants are read once during dataset setup in the parent process; keep the
    default codec pipeline here so zarrs is not enabled before DataLoader fork.
    """
    names = list(field_names)
    if len(names) == 0:
        raise ValueError("empty constant field name list")

    if is_monolithic_layout(ds) or (
        not is_named_arrays_layout(ds) and "constants" in ds
    ):
        cc = [str(x) for x in np.asarray(ds["channel_c"][:])]
        indices = [cc.index(n) for n in names]
        return np.asarray(ds["constants"][indices])

    loaders = [lambda n=n: np.asarray(ds[n]) for n in names]
    parts = _load_fields_parallel(loaders, n_threads)
    return np.stack(parts, axis=0)
