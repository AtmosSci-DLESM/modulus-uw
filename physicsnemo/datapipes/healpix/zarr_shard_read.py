"""One ``pread`` per time shard for per-variable healpix fields.

Used when a field is time-sharded and inner chunks are packed in time order.
The offset table at the end of each shard is read once and cached. Other
layouts fall back to the zarr codec path. Stacked ``inputs`` reads never come here.
"""

from __future__ import annotations

import mmap
import os
import threading

import numpy as np

_DIRECT_ALIGN = 4096
_MAX_U64 = np.uint64(2**64 - 1)
_COORD_KEYS = frozenset({"time", "face", "height", "width", "lat", "lon"})

_layout_cache: dict[tuple[str, str], "_FieldLayout"] = {}
_field_tables: dict[str, dict[str, "_FieldLayout"]] = {}


def clear_shard_cache() -> None:
    """Drop cached shard geometry. Tests use this between stores."""
    _layout_cache.clear()
    _field_tables.clear()


def _coalesced_shard_codec(array):
    """Return the sharding codec when one pread per time-shard is valid.

    Healpix per-variable stores shard only along time, with inner chunks packed
    in time order. Other layouts return None and keep the zarr path.
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
        self._lazy: dict[int, np.ndarray] = {}
        self._lazy_lock = threading.Lock()

    def shard_path(self, shard_i: int) -> str:
        tail = os.path.join("c", str(shard_i), *("0",) * (self.ndim - 1))
        if self.prefix:
            return os.path.join(self.root, self.prefix, tail)
        return os.path.join(self.root, tail)

    def offsets_for(self, shard_i: int) -> np.ndarray:
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
        if str(name).startswith("_") or name in _COORD_KEYS:
            continue
        array = group[name]
        if not hasattr(array, "shape"):
            continue
        layout = _layout_from_array(array)
        if layout is not None:
            table[str(name)] = layout
    _field_tables[root] = table
    return table


def read_field_time(group, name: str, time_sl) -> np.ndarray:
    """Time window for one per-variable field.

    Uses one pread per shard when the field is time-sharded. Otherwise reads
    through the zarr codec pipeline.
    """
    layout = _field_table(group).get(name)
    if layout is not None:
        bounds = _time_slice_bounds(layout.n_time, time_sl)
        if bounds is not None:
            return _read_layout_window(layout, *bounds)
    from physicsnemo.datapipes.healpix.zarr_layout import enable_zarrs_pipeline

    enable_zarrs_pipeline()
    return np.asarray(group[name][time_sl])
