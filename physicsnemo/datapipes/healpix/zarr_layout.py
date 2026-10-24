"""Helpers for healpix Zarr layouts (stacked vs per-variable).

- **stacked** — prognostic fields packed on a channel axis in ``inputs`` /
  ``constants``, with names in ``channel_in`` / ``channel_c``.
  Detection: ``is_stacked_layout``.
- **per-variable** — one top-level Zarr array per field: any store without an
  ``inputs`` array. Detection: ``is_per_variable_layout``.

The layout is read from the store's arrays, not from root attributes: catalogs
carry different attributes, and forecast output stores use ``layout`` for
something else.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import warnings
from typing import Any, Mapping, Sequence

import numpy as np

try:  # zarr >= 3; zarr 2 has no async API (see _read_fields)
    from zarr.abc.store import RangeByteRequest
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync
    from zarr.storage import LocalStore
except ImportError:
    LocalStore = sync = None

COORD_KEYS = frozenset({"time", "face", "height", "width", "lat", "lon"})
# Arrays that belong to the stacked layout, never fields of a per-variable store.
_STACKED_KEYS = frozenset(
    {"inputs", "targets", "constants", "channel_in", "channel_out", "channel_c"}
)

# A one-off bulk read (a rollout's multi-year truth, read once per run) brings
# hundreds of GB through the page cache that nothing reads again. Under a job
# memory limit those pages fill the cgroup, and every process in the job then
# stalls in reclaim. See ``dropping_read_pages``.
_DROP_READ_PAGES = False


@contextlib.contextmanager
def dropping_read_pages():
    """Within this block, local byte-range reads leave their pages out of the page cache.

    Readahead is switched off for each read, so only the requested range enters
    the cache, and that range is dropped right after it is read. The flag is
    process-wide: use it around a bulk read that no other thread of the process
    overlaps with reads that should stay cached.
    """
    global _DROP_READ_PAGES
    previous, _DROP_READ_PAGES = _DROP_READ_PAGES, True
    try:
        yield
    finally:
        _DROP_READ_PAGES = previous


if LocalStore is not None:

    class _PositionalReadLocalStore(LocalStore):
        """``LocalStore`` whose byte-range reads are one ``os.pread``.

        zarr's ``LocalStore`` opens the file and seeks to its end before every
        ranged read to learn the size. On Lustre that size lookup is a remote
        call per read; it cost about 7% of training throughput on Perlmutter.
        A positional read needs no size. Shard-index (suffix) and whole-key
        reads keep the base implementation.
        """

        def get_sync(self, key, *, prototype=None, byte_range=None):
            if not isinstance(byte_range, RangeByteRequest):
                return super().get_sync(key, prototype=prototype, byte_range=byte_range)
            try:
                fd = os.open(self.root / key, os.O_RDONLY)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return None
            try:
                length = byte_range.end - byte_range.start
                drop = _DROP_READ_PAGES
                if drop:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # no readahead
                data = os.pread(fd, length, byte_range.start)
                if drop:
                    os.posix_fadvise(fd, byte_range.start, length, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            if len(data) != length:  # past EOF: let zarr's own read decide
                return super().get_sync(key, prototype=prototype, byte_range=byte_range)
            return (prototype or default_buffer_prototype()).buffer.from_bytes(data)

        async def get(self, key, prototype=None, byte_range=None):
            if not isinstance(byte_range, RangeByteRequest):
                return await super().get(key, prototype, byte_range)
            return await asyncio.to_thread(
                self.get_sync, key, prototype=prototype, byte_range=byte_range
            )


def open_healpix_zarr(path):
    """``zarr.open(path)``; a local directory store reads through ``_PositionalReadLocalStore``."""
    import zarr

    if LocalStore is not None and isinstance(path, (str, os.PathLike)) and os.path.isdir(path):
        return zarr.open(_PositionalReadLocalStore(path))
    return zarr.open(path)


def _per_variable_read_config() -> dict:
    """zarr config for per-variable field reads: FusedCodecPipeline, one worker.

    Byte-range coalescing does not depend on this config: from zarr 3.3 every
    codec pipeline reads the inner chunks a window needs from a shard as
    coalesced byte ranges (one read per shard for a training window).
    FusedCodecPipeline (opt-in and experimental, zarr >= 3.3; see
    https://zarr.readthedocs.io/en/stable/user-guide/experimental/) changes how
    those chunks are decoded. The default pipeline gives each inner chunk its
    own coroutine and ``asyncio.to_thread`` hop; Fused runs a whole read, IO and
    decode, in one hop. Time-sharded fields hold one time step per inner chunk,
    so a training window is a few dozen inner chunks per field, and on a local
    store that per-chunk scheduling costs more than the decode itself.

    ``max_workers=1`` keeps that decode in the read's own thread. Fused's pool
    runs one task per shard of a read and decodes a shard's inner chunks in
    order; a window touches one shard per field (two at a shard boundary), so
    the pool has nothing to parallelize. A pool sized to the CPU count in every
    DataLoader worker would also oversubscribe the node; reads are concurrent
    across fields instead (``_read_fields``). Through zarr 3.4.1 the pool is also
    not reset after ``fork`` (fixed upstream in zarr-python#4483), so using it
    before DataLoader workers start can hang them.

    Older zarr has no FusedCodecPipeline; reads then use the active pipeline
    (correct, slower).
    """
    try:
        from zarr.core.codec_pipeline import FusedCodecPipeline  # noqa: F401
    except ImportError:
        return {}
    return {
        "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
        "codec_pipeline.max_workers": 1,
    }


_READ_CONFIG = _per_variable_read_config()
_async_arrays: dict[tuple[str, str], Any] = {}
# Set once the slow-reader warning has been shown; forked DataLoader workers
# inherit it, so the warning appears once per job, not once per worker.
_warned_slow_reads = False
# Reads that run at the same time share one zarr.config.set(_READ_CONFIG); see
# _read_config.
_read_config_lock = threading.Lock()
_read_config_users = 0
_read_config_stack = contextlib.ExitStack()


@contextlib.contextmanager
def _read_config():
    """``zarr.config.set(_READ_CONFIG)``, shared by reads that run at the same time.

    zarr's config is global to the process, and Fused reads
    ``codec_pipeline.max_workers`` when each read starts. If every read entered
    and left its own ``zarr.config.set``, a read finishing in one thread could
    restore zarr's defaults while a read in another thread still runs. Here the
    first read sets the config and the last one to finish restores it.
    """
    global _read_config_users
    import zarr

    with _read_config_lock:
        if _read_config_users == 0:
            _read_config_stack.enter_context(zarr.config.set(_READ_CONFIG))
        _read_config_users += 1
    try:
        yield
    finally:
        with _read_config_lock:
            _read_config_users -= 1
            if _read_config_users == 0:
                _read_config_stack.close()


def _warn_slow_per_variable_reads() -> None:
    """Warn once per process when per-variable reads cannot use FusedCodecPipeline.

    Without it (zarr < 3.3, or a zarr that renamed it), per-variable reads fall
    back to the active pipeline; zarr < 3.3 also fetches every inner chunk of a
    shard with its own store read. Both make reads much slower.
    """
    global _warned_slow_reads
    if _READ_CONFIG or _warned_slow_reads:
        return
    _warned_slow_reads = True
    import zarr

    warnings.warn(
        f"Per-variable Zarr reads need zarr >= 3.3 to be fast. zarr {zarr.__version__} "
        "has no FusedCodecPipeline (and before 3.3 reads every inner chunk of a "
        "shard separately), so these reads are much slower. Install zarr >= 3.3.",
        stacklevel=3,
    )


def _async_array(group, name: str):
    """``AsyncArray`` for one field, opened with ``_READ_CONFIG``.

    zarr fixes an array's codec pipeline when the array object is created, so it
    is created under that config; caching also skips re-reading ``zarr.json`` for
    every batch.
    """
    key = (str(group.store_path), name)
    array = _async_arrays.get(key)
    if array is None:
        with _read_config():
            array = group[name].async_array
        _async_arrays[key] = array
    return array


def _read_fields(ds, names: Sequence[str], selection, on_block=None) -> list[np.ndarray]:
    """``ds[name][selection]`` for every name, read concurrently.

    For a zarr group each field is one coroutine on zarr's own event loop (the
    loop every synchronous zarr call runs on; zarr resets it after ``fork``).
    ``on_block(i, block)`` runs in a worker thread as each field arrives, so
    scaling and scattering overlap the remaining reads. Other inputs (an xarray
    Dataset, or any group under zarr 2, which has no async API) are read in order.
    """
    import zarr

    _warn_slow_per_variable_reads()
    if sync is None or not isinstance(ds, zarr.Group):
        blocks = [np.asarray(ds[n][selection]) for n in names]
        if on_block is not None:
            for i, block in enumerate(blocks):
                on_block(i, block)
        return blocks

    arrays = [_async_array(ds, n) for n in names]

    async def _one(i: int, array) -> np.ndarray:
        block = np.asarray(await array.getitem(selection))
        if on_block is not None:
            await asyncio.to_thread(on_block, i, block)
        return block

    async def _all() -> list[np.ndarray]:
        return list(await asyncio.gather(*(_one(i, a) for i, a in enumerate(arrays))))

    # max_workers is read at read time, so the config applies to the reads too.
    with _read_config():
        return sync(_all())


def is_stacked_layout(ds) -> bool:
    """True when prognostic fields are packed on a channel axis in ``inputs``."""
    return "inputs" in ds


def is_per_variable_layout(ds) -> bool:
    """True when each field is its own top-level array (no ``inputs`` array)."""
    return not is_stacked_layout(ds)


def constants_are_stacked(ds) -> bool:
    """True when constants are a ``constants`` array indexed by ``channel_c``.

    That holds for every stacked store and for a constants-only stacked store;
    per-variable stores keep each constant as a top-level array.
    """
    return is_stacked_layout(ds) or ("constants" in ds and "channel_c" in ds)


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
    # A leading underscore marks loader metadata (e.g. ``_shard_index``), not a field.
    return {
        str(k)
        for k in ds.keys()
        if k not in COORD_KEYS and k not in _STACKED_KEYS and not str(k).startswith("_")
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


def load_channel_data(
    ds,
    time_sl,
    field_names: Sequence[str],
    scaling: Mapping | None = None,
) -> np.ndarray:
    """Load selected prognostic fields for a time window as (T, C, F, H, W)."""
    names = list(field_names)
    if len(names) == 0:
        if is_per_variable_layout(ds):
            raise ValueError("empty field name list")
        return np.asarray(ds["inputs"][time_sl])[:, []]

    if is_stacked_layout(ds):
        indices = _stacked_channel_indices(ds, names)
        # zarr returns a new array; copy only if it is read-only (scaled in place).
        out = np.require(ds["inputs"][time_sl, indices], requirements="W")
        if scaling is not None:
            for i in range(len(names)):
                _apply_channel_scaling(out[:, i], scaling, i)
        return out

    ref = ds[names[0]]
    tlen = _slice_length(ref.shape[0], time_sl)
    spatial = ref.shape[1:]
    out = np.empty((tlen, len(names)) + spatial, dtype=ref.dtype)

    def _fill(i: int, block: np.ndarray) -> None:
        if scaling is not None:
            block = _apply_channel_scaling(block, scaling, i)
        out[:, i] = block

    _read_fields(ds, names, time_sl, on_block=_fill)
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
    ic_diagnostic_names: Sequence[str] | None = None,
) -> (
    tuple[np.ndarray, np.ndarray | None]
    | tuple[np.ndarray, np.ndarray | None, np.ndarray]
):
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
    ic_diagnostic_names :
        Optional output-only channel names gathered at ``input_time_idx`` and
        scaled with ``output_scaling``. Empty or omitted yields ``None``.

    Returns
    -------
    inputs, targets
        ``inputs`` has shape ``(B, T_in, C_in, ...)``. ``targets`` is ``None``
        when ``output_names`` is omitted. When ``ic_diagnostic_names`` is
        non-empty the return is ``(inputs, targets, ic_diagnostics)`` with
        ``ic_diagnostics`` shaped ``(B, T_in, C_diag, ...)``.
    """
    input_names = list(input_names)
    ic_diag_names = list(ic_diagnostic_names or [])
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
        names = list(dict.fromkeys(
            list(input_names) + list(output_names or []) + ic_diag_names
        ))
        staging = load_channel_data(ds, time_sl, names, scaling=None)
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
        if not ic_diag_names:
            return inputs, targets
        ic_c = np.asarray([name_to_i[n] for n in ic_diag_names], dtype=np.intp)
        ic_diagnostics = staging[
            input_time_idx[:, :, np.newaxis], ic_c[np.newaxis, np.newaxis, :]
        ]
        return inputs, targets, ic_diagnostics

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

    ic_diag_cf = None
    ic_pos: dict[str, int] = {}
    if ic_diag_names:
        ic_diag_cf = np.empty(
            (len(ic_diag_names), batch_size, t_in) + spatial, dtype=dtype
        )
        ic_pos = {n: i for i, n in enumerate(ic_diag_names)}

    # Unique fields → destinations in input and/or target channel axes.
    slots: dict[str, tuple[int | None, int | None]] = {}
    for c, name in enumerate(input_names):
        in_c, out_c = slots.get(name, (None, None))
        slots[name] = (c, out_c)
    for c, name in enumerate(out_names):
        in_c, out_c = slots.get(name, (None, None))
        slots[name] = (in_c, c)
    for name in ic_diag_names:
        slots.setdefault(name, (None, None))

    def _place(block: np.ndarray, in_c: int | None, out_c: int | None, name: str) -> None:
        # Scale once. Shared input/output vars use the same physical mean/std;
        # prefer input_scaling when the field appears in both buffers.
        if in_c is not None and input_scaling is not None:
            block = _apply_channel_scaling(block, input_scaling, in_c)
        elif out_c is not None and output_scaling is not None:
            block = _apply_channel_scaling(block, output_scaling, out_c)
        elif name in ic_pos and output_scaling is not None and out_names:
            block = _apply_channel_scaling(block, output_scaling, out_names.index(name))
        if in_c is not None:
            inputs_cf[in_c] = block[input_time_idx]
        if out_c is not None:
            targets_cf[out_c] = block[output_time_idx]
        if name in ic_pos:
            ic_diag_cf[ic_pos[name]] = block[input_time_idx]

    fields = list(slots.items())
    _read_fields(
        ds,
        [name for name, _ in fields],
        time_sl,
        on_block=lambda i, block: _place(block, *fields[i][1], fields[i][0]),
    )
    # (C, B, T, ...) -> (B, T, C, ...); view, no copy.
    inputs = np.transpose(inputs_cf, (1, 2, 0, 3, 4, 5))
    targets = None if targets_cf is None else np.transpose(targets_cf, (1, 2, 0, 3, 4, 5))
    if not ic_diag_names:
        return inputs, targets
    ic_diagnostics = np.transpose(ic_diag_cf, (1, 2, 0, 3, 4, 5))
    return inputs, targets, ic_diagnostics


def load_constant_fields(ds, field_names: Sequence[str]) -> np.ndarray:
    """Load constant fields as (C, F, H, W)."""
    names = list(field_names)
    if len(names) == 0:
        raise ValueError("empty constant field name list")

    if constants_are_stacked(ds):
        cc = [str(x) for x in np.asarray(ds["channel_c"][:])]
        indices = [cc.index(n) for n in names]
        return np.asarray(ds["constants"][indices])

    return np.stack(_read_fields(ds, names, slice(None)), axis=0)
