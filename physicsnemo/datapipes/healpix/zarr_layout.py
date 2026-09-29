"""Helpers for healpix Zarr layouts (stacked vs per-variable).

- **stacked** — prognostic fields packed on a channel axis in ``inputs`` /
  ``constants``, with names in ``channel_in`` / ``channel_c``.
  Detection: ``is_stacked_layout``.
- **per-variable** — one Zarr array per field. Detection:
  ``is_per_variable_layout``.

New catalogs set the root attribute ``layout`` to ``per_variable``.
Catalogs already written with ``layout=named_arrays_healpix`` still load.
"""

from __future__ import annotations

import atexit
import logging
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

COORD_KEYS = frozenset({"time", "face", "height", "width", "lat", "lon"})

# Root attribute written by current converters.
PER_VARIABLE_LAYOUT_ATTR = "per_variable"
# Written by earlier converters and the b24 builder.
_LEGACY_PER_VARIABLE_LAYOUT_ATTR = "named_arrays_healpix"

# Process-local pool for DataLoader workers (set via init_worker_pool in worker_init_fn).
_worker_pool: ThreadPoolExecutor | None = None
_worker_pool_n_threads: int = 0
_worker_pool_lock = threading.Lock()
# One thread per shard read of a window. A persistent 4–8 pool leaves most
# fields waiting; CFS/DVS charges a round trip per syscall, so every field of
# the window has to be in flight together.
_io_pool: ThreadPoolExecutor | None = None
_io_pool_n: int = 0

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
    global _worker_pool, _worker_pool_n_threads, _io_pool, _io_pool_n
    with _worker_pool_lock:
        if _worker_pool is not None:
            _worker_pool.shutdown(wait=wait)
            _worker_pool = None
            _worker_pool_n_threads = 0
        if _io_pool is not None:
            _io_pool.shutdown(wait=wait)
            _io_pool = None
            _io_pool_n = 0


def worker_pool_active() -> bool:
    """True when a persistent pool is installed in this process."""
    return _worker_pool is not None


def _atexit_shutdown() -> None:
    # wait=True so pool threads are joined before native codec runtimes tear down.
    # wait=False aborts the worker (std::terminate) when zarrs is still live.
    shutdown_worker_pool(wait=True)


atexit.register(_atexit_shutdown)


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
    """True when each field is stored as its own array."""
    return not is_stacked_layout(ds)


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
    block -= scaling["mean"][0, channel]
    block /= scaling["std"][0, channel]
    return block


# On-disk shard index tables live here. Not a prognostic field.
SHARD_INDEX_GROUP = "_shard_index"
# O_DIRECT on this CFS/DVS mount requires 4 KiB alignment. The logical block
# size advertised by statvfs is 16 MiB; 4 KiB is what actually succeeds.
_DIRECT_ALIGN = 4096
_MAX_U64 = np.uint64(2**64 - 1)

# Layouts and per-store field tables. Built once per process; the per-window
# path does not reopen arrays.
_layout_cache: dict[tuple[str, str], Any] = {}
_field_tables: dict[str, dict[str, Any]] = {}


def _coalesced_shard_codec(array):
    """Return the sharding codec when one pread per time-shard is valid.

    Healpix per-variable stores shard only along time, with inner chunks packed
    in time order (morton order collapses to time order when the other axes
    have one chunk per shard). Other layouts return None and keep the zarr path.
    """
    meta = getattr(array, "metadata", None)
    if meta is None or getattr(meta, "zarr_format", None) != 3:
        return None
    codecs = tuple(getattr(meta, "codecs", ()) or ())
    if len(codecs) != 1:
        return None
    codec = codecs[0]
    try:
        from zarr.codecs.bytes import BytesCodec
        from zarr.codecs.crc32c_ import Crc32cCodec
        from zarr.codecs.sharding import ShardingCodec
        from zarr.codecs.zstd import ZstdCodec
    except ImportError:
        return None
    if not isinstance(codec, ShardingCodec):
        return None
    # ``ShardingCodecIndexLocation.end`` is deprecated; compare the string value.
    location = getattr(codec.index_location, "value", codec.index_location)
    if location != "end":
        return None
    index_codecs = tuple(codec.index_codecs)
    if len(index_codecs) != 2 or not isinstance(index_codecs[0], BytesCodec):
        return None
    if not isinstance(index_codecs[1], Crc32cCodec):
        return None
    inner_codecs = tuple(codec.codecs)
    if len(inner_codecs) != 2 or not isinstance(inner_codecs[0], BytesCodec):
        return None
    if not isinstance(inner_codecs[1], ZstdCodec):
        return None
    endian = getattr(inner_codecs[0], "endian", None)
    if getattr(endian, "value", endian) != "little":
        return None
    shards = getattr(array, "shards", None)
    inner = tuple(int(x) for x in codec.chunk_shape)
    if shards is None or len(inner) != len(tuple(shards)):
        return None
    cps = tuple(int(s) // c for s, c in zip(tuple(shards), inner, strict=False))
    if not cps or cps[0] < 1 or any(c != 1 for c in cps[1:]):
        return None
    if not hasattr(getattr(array, "store", None), "root"):
        return None
    return codec


def _pread_exact(fd: int, nbytes: int, offset: int) -> bytes:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = os.pread(fd, nbytes - len(buf), offset + len(buf))
        if not chunk:
            raise OSError(f"short shard read at offset {offset}")
        buf += chunk
    return bytes(buf)


def _decode_shard_index(raw: bytes, n_chunks: int) -> np.ndarray:
    """Decode a ShardingCodec index (little-endian uint64 pairs + crc32c)."""
    import google_crc32c

    payload = raw[:-4]
    stored = raw[-4:]
    computed = np.uint32(google_crc32c.value(payload)).tobytes()
    if computed != stored:
        raise ValueError("shard index checksum does not match")
    offsets = np.frombuffer(payload, dtype="<u8").reshape(n_chunks, 2)
    return offsets.copy()


def _read_index_tail(path: str, index_nbytes: int) -> bytes:
    """Buffered read of the shard index at EOF. Used when no index table is stored."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        return _pread_exact(fd, index_nbytes, size - index_nbytes)
    finally:
        os.close(fd)


def _pread_direct(path: str, offset: int, length: int) -> bytes:
    """One aligned O_DIRECT read. Raises OSError so the caller can fall back."""
    if not hasattr(os, "O_DIRECT"):
        raise OSError("O_DIRECT is not available")
    aligned = _DIRECT_ALIGN
    start = (offset // aligned) * aligned
    end = ((offset + length + aligned - 1) // aligned) * aligned
    buf = mmap.mmap(-1, end - start)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            nread = os.preadv(fd, [memoryview(buf)], start)
        finally:
            os.close(fd)
        need = (offset - start) + length
        if nread < need:
            raise OSError(f"short O_DIRECT read of {path}")
        rel = offset - start
        return bytes(memoryview(buf)[rel : rel + length])
    finally:
        buf.close()


def _pread_span(path: str, offset: int, length: int) -> bytes:
    """Read ``length`` bytes at ``offset``. Prefer one O_DIRECT syscall."""
    if length <= 0:
        return b""
    try:
        return _pread_direct(path, offset, length)
    except OSError:
        fd = os.open(path, os.O_RDONLY)
        try:
            return _pread_exact(fd, length, offset)
        finally:
            os.close(fd)


def _time_slice_bounds(n_time: int, time_sl: slice) -> tuple[int, int] | None:
    if not isinstance(time_sl, slice):
        return None
    step = 1 if time_sl.step is None else time_sl.step
    if step != 1:
        return None
    start = 0 if time_sl.start is None else time_sl.start
    stop = n_time if time_sl.stop is None else time_sl.stop
    if start < 0 or stop < 0 or start > stop or stop > n_time:
        return None
    return start, stop


class _FieldLayout:
    """Per-field shard geometry. Built once; reused for every window."""

    def __init__(
        self,
        *,
        root: str,
        prefix: str,
        n_time: int,
        inner: tuple[int, ...],
        shard_t: int,
        chunks_per_shard_t: int,
        dtype: np.dtype,
        fill,
        zstd_level: int,
        zstd_checksum: bool,
    ) -> None:
        self.root = root
        self.prefix = prefix
        self.n_time = n_time
        self.inner = inner
        self.shard_t = shard_t
        self.chunks_per_shard_t = chunks_per_shard_t
        self.spatial = inner[1:]
        self.dtype = np.dtype(dtype)
        self.fill = fill
        self.zstd_level = zstd_level
        self.zstd_checksum = zstd_checksum
        self.ndim = len(inner)
        self.index_nbytes = chunks_per_shard_t * 16 + 4
        self.n_shards = (n_time + shard_t - 1) // shard_t
        # (n_shards, chunks_per_shard, 2) when ``_shard_index/<name>`` is present.
        self.index: np.ndarray | None = None
        self._lazy: dict[int, np.ndarray] = {}
        self._lazy_lock = threading.Lock()

    def shard_path(self, shard_i: int) -> str:
        tail = os.path.join("c", str(shard_i), *("0",) * (self.ndim - 1))
        if self.prefix:
            return os.path.join(self.root, self.prefix, tail)
        return os.path.join(self.root, tail)

    def offsets_for(self, shard_i: int) -> np.ndarray:
        if self.index is not None:
            return self.index[shard_i]
        with self._lazy_lock:
            cached = self._lazy.get(shard_i)
        if cached is not None:
            return cached
        decoded = _decode_shard_index(
            _read_index_tail(self.shard_path(shard_i), self.index_nbytes),
            self.chunks_per_shard_t,
        )
        with self._lazy_lock:
            self._lazy[shard_i] = decoded
        return decoded


def _array_prefix(array) -> str:
    return str(getattr(getattr(array, "store_path", None), "path", "") or "").strip("/")


def _layout_from_array(array) -> _FieldLayout | None:
    """Geometry for a time-sharded field, or None when the zarr path should be used."""
    codec = _coalesced_shard_codec(array)
    if codec is None:
        return None
    root = str(array.store.root)
    prefix = _array_prefix(array)
    # Chunk keys must be ``c/<shard>/0/0/0``. Anything else keeps the zarr path.
    ndim = len(tuple(codec.chunk_shape))
    expected = "c/" + "/".join(["0"] * ndim)
    if array.metadata.encode_chunk_key((0,) * ndim) != expected:
        return None
    key = (root, prefix)
    cached = _layout_cache.get(key)
    if cached is not None:
        return cached
    inner = tuple(int(x) for x in codec.chunk_shape)
    shard_t = int(tuple(array.shards)[0])
    zstd_codec = codec.codecs[1]
    layout = _FieldLayout(
        root=root,
        prefix=prefix,
        n_time=int(array.shape[0]),
        inner=inner,
        shard_t=shard_t,
        chunks_per_shard_t=shard_t // inner[0],
        dtype=np.dtype(array.dtype),
        fill=array.fill_value,
        zstd_level=int(zstd_codec.level),
        zstd_checksum=bool(zstd_codec.checksum),
    )
    _layout_cache[key] = layout
    return layout


def _drop_store_cache(root: str) -> None:
    _field_tables.pop(root, None)
    for key in [k for k in _layout_cache if k[0] == root]:
        del _layout_cache[key]


def _store_root(ds) -> str | None:
    store = getattr(ds, "store", None)
    root = getattr(store, "root", None)
    return None if root is None else str(root)


def _read_layout_window(layout: _FieldLayout, start: int, stop: int) -> np.ndarray:
    """One pread per shard, decoded straight into a contiguous (T, *spatial) block."""
    from numcodecs.zstd import Zstd

    out = np.empty((stop - start,) + layout.spatial, dtype=layout.dtype)
    if start == stop:
        return out
    decoder = Zstd(level=layout.zstd_level, checksum=layout.zstd_checksum)
    inner_t = layout.inner[0]
    cps = layout.chunks_per_shard_t
    chunk_i = start // inner_t
    last_chunk = (stop - 1) // inner_t
    while chunk_i <= last_chunk:
        shard_i = chunk_i // cps
        local_i = chunk_i - shard_i * cps
        run_last = min(last_chunk, (shard_i + 1) * cps - 1)
        locals_in_run = range(local_i, local_i + (run_last - chunk_i) + 1)
        offsets = layout.offsets_for(shard_i)
        present = [i for i in locals_in_run if offsets[i, 0] != _MAX_U64]
        blob = b""
        span_start = 0
        if present:
            span_start = min(int(offsets[i, 0]) for i in present)
            span_end = max(int(offsets[i, 0] + offsets[i, 1]) for i in present)
            blob = _pread_span(layout.shard_path(shard_i), span_start, span_end - span_start)
        for i in locals_in_run:
            chunk_time0 = (shard_i * cps + i) * inner_t
            dest0 = max(0, chunk_time0 - start)
            dest1 = min(out.shape[0], chunk_time0 + inner_t - start)
            src0 = dest0 - (chunk_time0 - start)
            src1 = src0 + (dest1 - dest0)
            if offsets[i, 0] == _MAX_U64:
                out[dest0:dest1] = layout.fill
                continue
            rel = int(offsets[i, 0]) - span_start
            raw = blob[rel : rel + int(offsets[i, 1])]
            view = out[dest0:dest1]
            if (
                src0 == 0
                and src1 == inner_t
                and view.shape == layout.inner
                and view.flags.c_contiguous
            ):
                decoder.decode(raw, out=view)
            else:
                decoded = np.frombuffer(decoder.decode(raw), dtype=out.dtype).reshape(layout.inner)
                out[dest0:dest1] = decoded[src0:src1]
        chunk_i = run_last + 1
    return out


def read_sharded_time_slice(array, time_sl: slice) -> np.ndarray | None:
    """Read a contiguous time window from a time-sharded per-variable array.

    One pread per shard. Returns None when the array is not that layout;
    callers then use the zarr codec pipeline.
    """
    layout = _layout_from_array(array)
    if layout is None:
        return None
    bounds = _time_slice_bounds(layout.n_time, time_sl)
    if bounds is None:
        return None
    return _read_layout_window(layout, *bounds)


def _load_field_time(array, time_sl):
    """Time window for one per-variable field. Coalesce sharded reads when possible."""
    block = read_sharded_time_slice(array, time_sl)
    if block is None:
        block = np.asarray(array[time_sl])
    return block


def _attach_stored_indexes(group, table: dict[str, _FieldLayout]) -> None:
    if SHARD_INDEX_GROUP not in group or not table:
        return
    index_group = group[SHARD_INDEX_GROUP]
    names = [name for name in table if name in index_group]

    def _load(name: str):
        return name, np.asarray(index_group[name][:], dtype=np.uint64)

    workers = min(32, len(names))
    if workers <= 1:
        loaded = [_load(name) for name in names]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            loaded = list(ex.map(_load, names))
    for name, arr in loaded:
        layout = table[name]
        if arr.shape == (layout.n_shards, layout.chunks_per_shard_t, 2):
            layout.index = arr


def _field_table(group) -> dict[str, _FieldLayout]:
    """Sharded fields in this store. Metadata is read once per process."""
    root = _store_root(group)
    if root is None:
        return {}
    cached = _field_tables.get(root)
    if cached is not None:
        return cached
    table: dict[str, _FieldLayout] = {}
    for name in list(group.keys()):
        if str(name).startswith("_") or name in COORD_KEYS:
            continue
        array = group[name]
        if not hasattr(array, "shape"):
            continue
        layout = _layout_from_array(array)
        if layout is not None:
            table[str(name)] = layout
    _attach_stored_indexes(group, table)
    _field_tables[root] = table
    return table


def _layouts_for(group, names: Sequence[str]) -> dict[str, _FieldLayout] | None:
    if not names:
        return None
    table = _field_table(group)
    layouts: dict[str, _FieldLayout] = {}
    for name in names:
        layout = table.get(name)
        if layout is None:
            return None
        layouts[name] = layout
    return layouts


def _run_concurrent(fns: list) -> None:
    """Run every field read at once. The pool stays at the largest window seen."""
    global _io_pool, _io_pool_n
    if not fns:
        return
    if len(fns) == 1:
        fns[0]()
        return
    if _io_pool is None or _io_pool_n < len(fns):
        if _io_pool is not None:
            _io_pool.shutdown(wait=True)
        _io_pool = ThreadPoolExecutor(max_workers=len(fns))
        _io_pool_n = len(fns)
    list(_io_pool.map(lambda fn: fn(), fns))


def _scatter_time(block: np.ndarray, time_idx: np.ndarray, dest: np.ndarray) -> None:
    """Copy window times into ``dest[b, t]`` without a fancy-index temporary."""
    for bi in range(time_idx.shape[0]):
        row = time_idx[bi]
        for ti in range(row.shape[0]):
            dest[bi, ti] = block[int(row[ti])]


def _window_timer_on() -> bool:
    return os.environ.get("PER_VARIABLE_WINDOW_TIMER") == "1"


def _log_window(t0: float | None, n_fields: int) -> None:
    if t0 is None:
        return
    print(
        f"per_variable_window pid={os.getpid()} fields={n_fields} "
        f"{time.perf_counter() - t0:.3f}s",
        flush=True,
    )


def build_shard_index_table(group, *, workers: int = 64) -> list[str]:
    """Write ``_shard_index/<field>`` and consolidate metadata.

    Each array is ``(n_shards, chunks_per_shard, 2)`` uint64 offset/length
    pairs, one chunk, uncompressed. Chunk bytes are not rewritten. Fields
    that are not time-sharded are skipped. Returns the field names written.
    """
    import zarr

    root = _store_root(group)
    names = [str(name) for name in list(group.keys()) if not str(name).startswith("_")]
    written: list[str] = []
    pool_n = max(1, workers)
    for name in names:
        array = group[name]
        layout = _layout_from_array(array)
        if layout is None:
            continue
        table = np.empty((layout.n_shards, layout.chunks_per_shard_t, 2), dtype="<u8")

        def _read_shard(shard_i: int, layout: _FieldLayout = layout) -> tuple[int, np.ndarray]:
            raw = _read_index_tail(layout.shard_path(shard_i), layout.index_nbytes)
            return shard_i, _decode_shard_index(raw, layout.chunks_per_shard_t)

        with ThreadPoolExecutor(max_workers=min(pool_n, layout.n_shards)) as ex:
            for shard_i, offsets in ex.map(_read_shard, range(layout.n_shards)):
                table[shard_i] = offsets
        # Write through the returned array. group[path] on an already
        # consolidated store only sees arrays listed in that metadata.
        index_arr = zarr.create_array(
            store=group.store,
            name=f"{SHARD_INDEX_GROUP}/{name}",
            shape=table.shape,
            chunks=table.shape,
            dtype="<u8",
            zarr_format=3,
            overwrite=True,
            compressors=[],
        )
        index_arr[:] = table
        written.append(name)
    zarr.consolidate_metadata(group.store, zarr_format=3)
    if root is not None:
        _drop_store_cache(root)
    return written


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

    if is_stacked_layout(ds):
        indices = _stacked_channel_indices(ds, names)
        out = np.asarray(ds["inputs"][time_sl, indices])
        if scaling is not None:
            out -= scaling["mean"]
            out /= scaling["std"]
        return out

    layouts = _layouts_for(ds, names)
    bounds = (
        None
        if layouts is None
        else _time_slice_bounds(layouts[names[0]].n_time, time_sl)
    )
    if layouts is not None and bounds is not None:
        start, stop = bounds
        layout0 = layouts[names[0]]
        out = np.empty((stop - start, len(names)) + layout0.spatial, dtype=layout0.dtype)
        t0 = time.perf_counter() if _window_timer_on() else None

        def _fill_direct(i: int) -> None:
            block = _read_layout_window(layouts[names[i]], start, stop)
            if scaling is not None:
                block = _apply_channel_scaling(block, scaling, i)
            out[:, i] = block

        _run_concurrent([lambda i=i: _fill_direct(i) for i in range(len(names))])
        _log_window(t0, len(names))
        return out

    ref = ds[names[0]]
    tlen = _slice_length(ref.shape[0], time_sl)
    spatial = ref.shape[1:]
    out = np.empty((tlen, len(names)) + spatial, dtype=ref.dtype)

    def _fill(i: int, n: str) -> None:
        block = _load_field_time(ds[n], time_sl)
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
    ic_diagnostic_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load and scale fields directly into (B, T, C, F, H, W) sample windows.

    For per-variable stores, each unique field is decoded and scaled once on
    a worker thread, then scattered into channel-first scratch buffers. For
    stacked ``inputs`` stores, channels are read jointly and gathered into
    the same window layout.

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
    ic_diagnostic_names :
        Optional output-only channel names to also gather at ``input_time_idx``
        (scaled with ``output_scaling``). Used for soft-constraint IC anchors.

    Returns
    -------
    inputs, targets, ic_diagnostics
        ``inputs`` has shape ``(B, T_in, C_in, ...)``. ``targets`` is ``None``
        when ``output_names`` is omitted. ``ic_diagnostics`` is ``None`` when
        ``ic_diagnostic_names`` is empty/omitted; otherwise
        ``(B, T_in, C_diag, ...)``.
    """
    enable_zarrs_pipeline()
    input_names = list(input_names)
    if len(input_names) == 0:
        raise ValueError("empty input field name list")
    if input_time_idx.ndim != 2:
        raise ValueError(
            f"input_time_idx must be (B, T), got shape {input_time_idx.shape}"
        )
    ic_diag_names = list(ic_diagnostic_names or [])

    if is_stacked_layout(ds):
        # Stacked stores are already a joint array; stage once then gather.
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
        ic_diagnostics = None
        if ic_diag_names:
            if output_names is None or output_scaling is None:
                raise ValueError(
                    "ic_diagnostic_names requires output_names and output_scaling"
                )
            ic_c = np.asarray([name_to_i[n] for n in ic_diag_names], dtype=np.intp)
            ic_diagnostics = staging[
                input_time_idx[:, :, np.newaxis], ic_c[np.newaxis, np.newaxis, :]
            ]
            out_idx = [list(output_names).index(n) for n in ic_diag_names]
            # output_scaling mean/std are broadcastable on channel axis (index 1
            # after expand to match (B,T,C,...)); select diagnostic channels.
            mean = output_scaling["mean"]
            std = output_scaling["std"]
            # Shapes are typically (1, C, 1, 1, 1) after dataset expand_dims.
            mean_c = np.take(mean, out_idx, axis=1)
            std_c = np.take(std, out_idx, axis=1)
            ic_diagnostics = (ic_diagnostics - mean_c) / std_c
        return inputs, targets, ic_diagnostics

    batch_size, t_in = input_time_idx.shape
    # Slots are known before any read, so the sharded path can skip per-field
    # array opens. Non-sharded stores still take shape from the first array.
    slot_names = list(dict.fromkeys(list(input_names) + list(output_names or []) + ic_diag_names))
    layouts = _layouts_for(ds, slot_names)
    direct_bounds = (
        None
        if layouts is None
        else _time_slice_bounds(layouts[slot_names[0]].n_time, time_sl)
    )
    use_direct = layouts is not None and direct_bounds is not None
    if use_direct:
        layout0 = layouts[slot_names[0]]
        spatial = layout0.spatial
        dtype = layout0.dtype
    else:
        ref = ds[input_names[0]]
        spatial = ref.shape[1:]
        dtype = ref.dtype
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

    ic_diag_cf = None
    ic_name_to_c: dict[str, int] = {}
    if ic_diag_names:
        ic_diag_cf = np.empty(
            (len(ic_diag_names), batch_size, t_in) + spatial, dtype=dtype
        )
        ic_name_to_c = {n: i for i, n in enumerate(ic_diag_names)}

    # Unique fields → destinations in input and/or target channel axes.
    slots: dict[str, tuple[int | None, int | None]] = {}
    for c, name in enumerate(input_names):
        in_c, out_c = slots.get(name, (None, None))
        slots[name] = (c, out_c)
    for c, name in enumerate(out_names):
        in_c, out_c = slots.get(name, (None, None))
        slots[name] = (in_c, c)
    # Ensure output-only IC diagnostics are loaded even if somehow omitted
    # from output_names (should not happen when wired from the dataset).
    for name in ic_diag_names:
        if name not in slots:
            slots[name] = (None, None)

    def _place(block: np.ndarray, in_c: int | None, out_c: int | None, name: str) -> None:
        # Scale once. Shared input/output vars use the same physical mean/std;
        # prefer input_scaling when the field appears in both buffers.
        if in_c is not None and input_scaling is not None:
            block = _apply_channel_scaling(block, input_scaling, in_c)
        elif out_c is not None and output_scaling is not None:
            block = _apply_channel_scaling(block, output_scaling, out_c)
        elif name in ic_name_to_c and output_scaling is not None and out_names:
            block = _apply_channel_scaling(
                block, output_scaling, out_names.index(name)
            )
        if use_direct:
            if in_c is not None:
                _scatter_time(block, input_time_idx, inputs_cf[in_c])
            if out_c is not None:
                _scatter_time(block, output_time_idx, targets_cf[out_c])
            if name in ic_name_to_c:
                _scatter_time(block, input_time_idx, ic_diag_cf[ic_name_to_c[name]])
            return
        if in_c is not None:
            inputs_cf[in_c] = block[input_time_idx]
        if out_c is not None:
            targets_cf[out_c] = block[output_time_idx]
        if name in ic_name_to_c:
            # Same scaled block; gather at input times for soft-constraint ICs.
            ic_diag_cf[ic_name_to_c[name]] = block[input_time_idx]

    def _fill(name: str, in_c: int | None, out_c: int | None) -> None:
        if use_direct:
            block = _read_layout_window(layouts[name], *direct_bounds)
        else:
            block = _load_field_time(ds[name], time_sl)
        _place(block, in_c, out_c, name)

    loaders = [
        lambda n=n, ic=ic, oc=oc: _fill(n, ic, oc) for n, (ic, oc) in slots.items()
    ]
    t0 = time.perf_counter() if use_direct and _window_timer_on() else None
    if use_direct:
        _run_concurrent(loaders)
        _log_window(t0, len(loaders))
    else:
        _run_loaders_parallel(loaders, n_threads)
    # (C, B, T, ...) -> (B, T, C, ...); view, no copy.
    inputs = np.transpose(inputs_cf, (1, 2, 0, 3, 4, 5))
    targets = (
        None if targets_cf is None else np.transpose(targets_cf, (1, 2, 0, 3, 4, 5))
    )
    ic_diagnostics = (
        None if ic_diag_cf is None else np.transpose(ic_diag_cf, (1, 2, 0, 3, 4, 5))
    )
    return inputs, targets, ic_diagnostics


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

    if constants_are_stacked(ds):
        cc = [str(x) for x in np.asarray(ds["channel_c"][:])]
        indices = [cc.index(n) for n in names]
        return np.asarray(ds["constants"][indices])

    loaders = [lambda n=n: np.asarray(ds[n]) for n in names]
    parts = _load_fields_parallel(loaders, n_threads)
    return np.stack(parts, axis=0)
