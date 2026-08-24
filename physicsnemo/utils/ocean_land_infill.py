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

"""
Ocean-over-land infill utilities.

Infills ocean variables over land with a fixed value (e.g. standardized -1) so that
land pixels do not evolve during autoregressive steps. Used in coupled atmosphere-ocean
models for both the ocean state and the atmosphere's coupling (ocean) inputs.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch as th
import xarray as xr


def load_land_mask(
    lsm_file: str,
    open_dict: Optional[Dict[str, Any]] = None,
    selection_dict: Optional[Dict[str, str]] = None,
    device: Optional[th.device] = None,
) -> th.Tensor:
    """
    Load land mask from the same dataset convention as healpix_loss (LSM file).

    Parameters
    ----------
    lsm_file : str
        Path to the land-sea mask file (e.g. zarr or netcdf with constants).
    open_dict : dict, optional
        Kwargs for xarray.open_dataset (e.g. {"engine": "zarr"}). Default: {"engine": "zarr"}.
    selection_dict : dict, optional
        Channel selection for the land_sea_mask (e.g. {"channel_c": "land_sea_mask"}).
        Default: {"channel_c": "land_sea_mask"}.
    device : torch.device, optional
        Device for the returned tensor. If None, returns CPU tensor.

    Returns
    -------
    torch.Tensor
        Land mask with 1 = land, 0 = ocean. Shape [1, 1, 1, 1, H, W] (or [1, 1, H, W]
        if grid has no face dim), broadcastable to state/coupling spatial dims.
    """
    if open_dict is None:
        open_dict = {"engine": "zarr"}
    if selection_dict is None:
        selection_dict = {"channel_c": "land_sea_mask"}
    # Accept either land_sea_mask or lsm as the constant name
    _channel_names = ("land_sea_mask", "lsm")
    dim_name = next(iter(selection_dict.keys())) if selection_dict else "channel_c"
    names_to_try = [selection_dict[dim_name]] if selection_dict else list(_channel_names)
    if names_to_try[0] not in _channel_names:
        for alt in _channel_names:
            if alt not in names_to_try:
                names_to_try.append(alt)
    else:
        for alt in _channel_names:
            if alt != names_to_try[0] and alt not in names_to_try:
                names_to_try.append(alt)
    ds = xr.open_dataset(lsm_file, **open_dict)
    lsm_da = None
    last_err = None
    for ch_name in names_to_try:
        try:
            sel = {dim_name: ch_name}
            lsm_da = ds.constants.sel(sel)
            break
        except KeyError as e:
            last_err = e
            continue
    if lsm_da is None:
        raise RuntimeError(
            f"Land mask not found: tried {dim_name} in {names_to_try}. Last error: {last_err}"
        ) from last_err
    # Convention: dataset typically has 1 = land, 0 = ocean (land fraction)
    land_np = np.asarray(lsm_da.values, dtype=np.float32)
    # Ensure shape [1, F, 1, 1, H, W] for broadcasting to [B, F, T, C, H, W] (F = faces if HEALPix)
    if land_np.ndim == 3:
        # (n_faces, H, W) -> [1, n_faces, 1, 1, H, W] so 12 aligns with F, not C
        n_faces, h, w = land_np.shape
        land_np = land_np.reshape(1, n_faces, 1, 1, h, w)
    else:
        while land_np.ndim < 6:
            land_np = np.expand_dims(land_np, axis=0)
    land = th.tensor(land_np, dtype=th.float32)
    if device is not None:
        land = land.to(device=device)
    return land


def land_mask_from_constants(
    constants_arr: np.ndarray,
    constants_config: Any,
    device: Optional[th.device] = None,
) -> th.Tensor:
    """
    Extract land-sea mask from the data module's constants array using the constants config.

    Use when LSM is provided in dataset constants (e.g. atmosphere training without
    criterion.lsm_file). constants_arr has shape [F, C, H, W]; constants_config is a dict
    mapping logical names to dataset channel names (e.g. {"land_sea_mask": "lsm"}).

    Returns a tensor with shape [1, F, 1, 1, H, W], 1 = land, 0 = ocean.
    """
    if constants_arr is None or constants_config is None:
        raise ValueError("constants_arr and constants_config must be provided")
    constants_config = (
        dict(constants_config) if not hasattr(constants_config, "items") else constants_config
    )
    if hasattr(constants_config, "keys"):
        keys = list(constants_config.keys())
        values = list(constants_config.values())
    else:
        keys = values = []
    # Find channel index for LSM (config key "land_sea_mask" or value "lsm")
    _LSM_NAMES = ("land_sea_mask", "lsm")
    lsm_idx = None
    for i, (k, v) in enumerate(zip(keys, values)):
        if k in _LSM_NAMES or (isinstance(v, str) and v in _LSM_NAMES):
            lsm_idx = i
            break
    if lsm_idx is None:
        raise ValueError(
            f"Land mask channel not found in constants config (tried keys/values {_LSM_NAMES}); "
            f"config keys={list(keys)}, values={list(values)}"
        )
    # constants_arr: [F, C, H, W] -> take channel lsm_idx -> [F, H, W]
    land_np = np.asarray(constants_arr[:, lsm_idx, :, :], dtype=np.float32)
    # [F, H, W] -> [1, F, 1, 1, H, W]
    n_faces, h, w = land_np.shape
    land_np = land_np.reshape(1, n_faces, 1, 1, h, w)
    land = th.tensor(land_np, dtype=th.float32)
    if device is not None:
        land = land.to(device=device)
    return land


def infill_ocean_over_land(
    tensor: th.Tensor,
    land_mask: th.Tensor,
    fill_standardized: th.Tensor,
    channel_dim: int,
    land_threshold: float = 0.0,
) -> None:
    """
    In-place infill of ocean variables over land with per-channel standardized values.

    Sets tensor at all land pixels (land_mask >= land_threshold) to the corresponding
    channel's fill value. Default land_threshold=0.0 is interpreted as 1.0 (only 100%%
    land cells are infilled; mixed/ocean cells keep their value and are weighted in loss).

    Parameters
    ----------
    tensor : torch.Tensor
        State [B, F, T, C, H, W] (channel_dim=3) or coupling [T, B, C, F, H, W]
        (channel_dim=2). Modified in-place.
    land_mask : torch.Tensor
        Land mask with 1 = land, 0 = ocean (land fraction). Must broadcast to the
        spatial dimensions (last two dims) of tensor; e.g. shape [1, 1, 1, 1, H, W].
    fill_standardized : torch.Tensor
        1D tensor of length equal to number of channels to infill (same as
        tensor.shape[channel_dim]). Fill value in standardized space per channel.
    channel_dim : int
        Index of the channel dimension in tensor (3 for state, 2 for coupling).
    land_threshold : float, optional
        Minimum land fraction to treat as land (infill). Pixels with land_mask >=
        land_threshold are infilled. Default 0.0 is interpreted as 1.0 (only 100%%
        land cells infilled). Use e.g. 0.5 for "any land" (previous behavior).
    """
    if tensor.device != land_mask.device:
        land_mask = land_mask.to(device=tensor.device)
    if tensor.device != fill_standardized.device:
        fill_standardized = fill_standardized.to(device=tensor.device)
    if land_mask.dtype != tensor.dtype:
        land_mask = land_mask.to(dtype=tensor.dtype)
    if fill_standardized.dtype != tensor.dtype:
        fill_standardized = fill_standardized.to(dtype=tensor.dtype)

    # Land mask is stored as [1, F, 1, 1, H, W] (state layout). For coupling [T, B, C, F, H, W]
    # we need [1, 1, 1, F, H, W] so expand_as works; permute face dim from 1 to 3.
    if channel_dim == 2 and land_mask.dim() == 6 and land_mask.shape[1] != 1:
        land_mask = land_mask.permute(0, 2, 3, 1, 4, 5)

    # Spatial dims are always last two (H, W). Expand land_mask to tensor shape.
    # Default 0.0 -> 1.0 so only 100% land cells are infilled.
    effective_threshold = 1.0 if land_threshold == 0.0 else land_threshold
    land_expanded = land_mask.expand_as(tensor).to(device=tensor.device, dtype=tensor.dtype)
    land_bool = land_expanded >= effective_threshold

    n_channels = tensor.shape[channel_dim]
    n_fill = fill_standardized.numel()
    if n_fill != n_channels:
        if n_channels % n_fill == 0:
            # Repeat fill per channel block (e.g. 3 vars × 2 times = 6 channels, 3 fill values)
            fill_standardized = fill_standardized.to(device=tensor.device, dtype=tensor.dtype)
            fill_standardized = fill_standardized.repeat(n_channels // n_fill)
        else:
            raise ValueError(
                f"fill_standardized length ({n_fill}) must match tensor channels at dim {channel_dim} "
                f"({n_channels}) or divide them"
            )

    for c in range(n_channels):
        if channel_dim == 2:
            # coupling [T, B, C, F, H, W]
            slice_c = (slice(None), slice(None), c, slice(None), slice(None), slice(None))
        elif channel_dim == 3:
            # state [B, F, T, C, H, W]
            slice_c = (slice(None), slice(None), slice(None), c, slice(None), slice(None))
        else:
            raise ValueError(f"channel_dim must be 2 or 3 for HEALPix state/coupling, got {channel_dim}")
        view_c = tensor[slice_c]  # [..., H, W]
        mask_c = land_bool[slice_c]
        view_c.masked_fill_(mask_c, fill_standardized[c].item())


def verify_infill(
    tensor: th.Tensor,
    land_mask: th.Tensor,
    fill_standardized: th.Tensor,
    channel_dim: int,
    land_threshold: float = 0.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict:
    """
    Verify that land pixels in `tensor` have the expected infill values (for debugging).

    land_threshold: same as infill_ocean_over_land (default 0.0 -> 1.0, only 100%% land).

    Returns a dict with keys: ok (bool), land_frac (float), n_land_checked (int),
    max_abs_err_land (float), sample_land_vals (list), sample_fill_vals (list).
    """
    if tensor.device != land_mask.device:
        land_mask = land_mask.to(device=tensor.device)
    land_mask = land_mask.to(dtype=tensor.dtype)
    # Land mask is stored as [1, F, 1, 1, H, W] (state layout). For coupling [T, B, C, F, H, W]
    # we need [1, 1, 1, F, H, W]; permute face dim from 1 to 3.
    if channel_dim == 2 and land_mask.dim() == 6 and land_mask.shape[1] != 1:
        land_mask = land_mask.permute(0, 2, 3, 1, 4, 5)
    effective_threshold = 1.0 if land_threshold == 0.0 else land_threshold
    land_expanded = land_mask.expand_as(tensor)
    land_bool = land_expanded >= effective_threshold
    n_land = land_bool.sum().item()
    n_total = land_bool.numel()
    land_frac = n_land / n_total if n_total > 0 else 0.0

    n_channels = tensor.shape[channel_dim]
    n_fill = fill_standardized.numel()
    fill = (
        fill_standardized.repeat(n_channels // n_fill).to(device=tensor.device, dtype=tensor.dtype)
        if n_channels != n_fill and n_channels % n_fill == 0
        else fill_standardized.to(device=tensor.device, dtype=tensor.dtype)
    )

    max_err = 0.0
    sample_land_vals = []
    sample_fill_vals = []
    n_checked = 0
    for c in range(n_channels):
        if channel_dim == 2:
            slice_c = (slice(None), slice(None), c, slice(None), slice(None), slice(None))
        else:
            slice_c = (slice(None), slice(None), slice(None), c, slice(None), slice(None))
        view_c = tensor[slice_c]
        mask_c = land_bool[slice_c]
        fill_c = fill[c].item()
        vals_at_land = view_c[mask_c]
        if vals_at_land.numel() > 0:
            err = (vals_at_land - fill_c).abs()
            max_err = max(max_err, err.max().item())
            n_checked += vals_at_land.numel()
            if len(sample_land_vals) < 5:
                sample_land_vals.extend(vals_at_land.flatten()[: 5 - len(sample_land_vals)].tolist())
                sample_fill_vals.extend([fill_c] * min(5 - len(sample_fill_vals), vals_at_land.numel()))

    ok = max_err <= atol + rtol * (abs(fill[0].item()) + 1e-8)
    return {
        "ok": ok,
        "land_frac": land_frac,
        "n_land_checked": n_checked,
        "max_abs_err_land": max_err,
        "sample_land_vals": sample_land_vals[:5],
        "sample_fill_vals": sample_fill_vals[:5],
    }
