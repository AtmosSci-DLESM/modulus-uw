# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for Hydra ``data.scaling`` with scalar or spatial (climatology) means.

``mean`` may be a float (broadcast over space) or a filesystem path to a
``(face, height, width)`` array (zarr array path or ``.npy``). ``std`` is always
a positive scalar. Normalization is ``(x - mean) / std``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def is_path_like_mean(mean: Any) -> bool:
    """True if ``mean`` is a non-numeric string/path referring to a spatial map."""
    if isinstance(mean, (str, Path)):
        try:
            float(mean)
            return False
        except (TypeError, ValueError):
            return True
    return False


def is_nonzero_scalar_mean(mean: Any) -> bool:
    """True only for a numeric scalar mean that is not (approximately) zero.

    Used by reflectional-equivariance checks: odd channels may use a spatial
    (path/array) clim mean, but not a nonzero scalar mean.
    """
    if is_path_like_mean(mean) or isinstance(mean, np.ndarray):
        return False
    try:
        return abs(float(mean)) > 1e-12
    except (TypeError, ValueError):
        return False


def resolve_mean(mean: Any) -> float | np.ndarray:
    """Load a scaling mean as float32 scalar or ``(F, H, W)`` float32 array."""
    if isinstance(mean, np.ndarray):
        arr = np.asarray(mean, dtype=np.float32)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim != 3:
            raise ValueError(
                f"Spatial scaling mean must have shape (F, H, W); got {arr.shape}"
            )
        return arr

    if is_path_like_mean(mean):
        path = Path(str(mean))
        if path.suffix == ".npy":
            arr = np.load(path)
        else:
            import zarr

            arr = np.asarray(zarr.open_array(str(path), mode="r")[:])
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(
                f"Spatial scaling mean at {path} must have shape (F, H, W); "
                f"got {arr.shape}"
            )
        return arr

    return float(mean)


def resolve_std(std: Any) -> float:
    """Return a positive float32 std."""
    value = float(std)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"scaling std must be a positive finite float; got {std!r}")
    return value


def _entry(scaling: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in scaling:
        raise KeyError(f"Variable {name!r} not found in scaling config")
    entry = scaling[name]
    if "mean" not in entry or "std" not in entry:
        raise KeyError(f"scaling[{name!r}] must define 'mean' and 'std'")
    return entry


def build_channel_scaling(
    names: Sequence[str],
    scaling: Mapping[str, Any],
    *,
    mean_expand_axes: Sequence[int],
    std_expand_axes: Sequence[int] | None = None,
    allow_spatial_mean: bool = True,
) -> dict[str, np.ndarray]:
    """Stack per-channel means/stds into broadcastable float32 arrays.

    Parameters
    ----------
    names:
        Channel names in tensor order.
    scaling:
        Hydra ``data.scaling`` mapping.
    mean_expand_axes / std_expand_axes:
        Axes inserted around a scalar mean/std (same convention as the legacy
        ``np.expand_dims(..., (0, 2, 3, 4))`` path). Ignored for spatial means
        except that a leading batch axis of size 1 is prepended when
        ``0 in mean_expand_axes``.
    allow_spatial_mean:
        If False, path/array means raise (used for constant fields).
    """
    if std_expand_axes is None:
        std_expand_axes = mean_expand_axes

    means: list[float | np.ndarray] = []
    stds: list[float] = []
    spatial_shape: tuple[int, int, int] | None = None

    for name in names:
        entry = _entry(scaling, name)
        mean = resolve_mean(entry["mean"])
        std = resolve_std(entry["std"])
        if isinstance(mean, np.ndarray):
            if not allow_spatial_mean:
                raise ValueError(
                    f"Variable {name!r} uses a spatial scaling mean, but this "
                    "scaling group only allows scalar means"
                )
            if spatial_shape is None:
                spatial_shape = tuple(mean.shape)
            elif mean.shape != spatial_shape:
                raise ValueError(
                    f"Spatial mean shape mismatch for {name!r}: {mean.shape} vs "
                    f"{spatial_shape}"
                )
        means.append(mean)
        stds.append(std)

    if spatial_shape is None:
        mean_arr = np.asarray(means, dtype=np.float32)
        for axis in sorted(mean_expand_axes):
            mean_arr = np.expand_dims(mean_arr, axis)
    else:
        # (C, F, H, W); prepend batch axis when legacy expand used axis 0.
        stacked = []
        for mean in means:
            if isinstance(mean, np.ndarray):
                stacked.append(mean)
            else:
                stacked.append(
                    np.full(spatial_shape, float(mean), dtype=np.float32)
                )
        mean_arr = np.stack(stacked, axis=0).astype(np.float32, copy=False)
        if 0 in mean_expand_axes:
            mean_arr = mean_arr[np.newaxis, ...]

    std_arr = np.asarray(stds, dtype=np.float32)
    for axis in sorted(std_expand_axes):
        std_arr = np.expand_dims(std_arr, axis)

    return {"mean": mean_arr, "std": std_arr}


def physical_bound_to_normalized(
    phys_bound: float,
    mean: float | np.ndarray,
    std: float,
) -> float | np.ndarray:
    """Map a physical clamp bound into normalized space."""
    return (float(phys_bound) - mean) / float(std)


def stack_normalized_bounds(
    channels: Sequence[str],
    scaling: Mapping[str, Any],
    bounds: Mapping[str, float],
    *,
    unconstrained: float,
) -> np.ndarray:
    """Build per-channel normalized bounds for ``[B, F, T, C, H, W]`` tensors.

    Returns
    -------
    np.ndarray
        Shape ``(1, 1, 1, C, 1, 1)`` if all relevant means are scalar, else
        ``(1, F, 1, C, H, W)`` when any constrained channel uses a spatial mean.
    """
    constrained = [name for name in channels if name in bounds]
    spatial_shape: tuple[int, int, int] | None = None
    resolved: dict[str, float | np.ndarray] = {}

    for name in constrained:
        entry = _entry(scaling, name)
        mean = resolve_mean(entry["mean"])
        std = resolve_std(entry["std"])
        if isinstance(mean, np.ndarray):
            spatial_shape = tuple(mean.shape)
        resolved[name] = physical_bound_to_normalized(bounds[name], mean, std)

    if spatial_shape is None:
        per_channel = [
            float(resolved[name]) if name in resolved else unconstrained
            for name in channels
        ]
        return np.asarray(per_channel, dtype=np.float32).reshape(1, 1, 1, -1, 1, 1)

    f, h, w = spatial_shape
    c = len(channels)
    out = np.full((1, f, 1, c, h, w), unconstrained, dtype=np.float32)
    for i, name in enumerate(channels):
        if name not in resolved:
            continue
        value = resolved[name]
        if isinstance(value, np.ndarray):
            out[0, :, 0, i, :, :] = value
        else:
            out[0, :, 0, i, :, :] = float(value)
    return out
