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

"""Partial convolution stem for coupled HEALPix inputs (e.g. SST/SIC).

Soft masks (continuous ocean fraction) *downweight* filled/invalid cells; they do
not make the stem independent of the fill strategy on fractional cells. Hard
threshold masks set ``M in {0, 1}`` so invalid fills do not contribute to the
forward pass or gradients.

Padding uses the same ``HEALPixLayer`` / ``make_hpx_padding_layer`` path as the
rest of the atmosphere model.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from .healpix_layers import HEALPixLayer
from .healpix_paddings import HEALPixFoldFaces, HEALPixUnfoldFaces


def coupled_variable_channel_names(couplings: Sequence[Mapping[str, Any]]) -> list[str]:
    """Expand coupling configs to the per-channel variable name list.

    Order matches the coupler / ``_reshape_inputs`` concatenation:
    for each coupling, for each variable, for each ``input_times`` entry.
    """
    names: list[str] = []
    for c in couplings:
        params = c["params"]
        for v in params["variables"]:
            for _ in params["input_times"]:
                names.append(str(v))
    return names


def _open_mask_dataset(dataset_path: str) -> xr.Dataset:
    path = str(dataset_path)
    if path.rstrip("/").endswith(".zarr"):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def load_spatial_mask(
    dataset_path: str,
    data_var: str = "constants",
    selection_dict: Mapping[str, Any] | None = None,
    invert: bool = True,
    threshold: float | None = None,
) -> th.Tensor:
    """Load a single spatial mask as ``[F, H, W]`` float tensor in ``[0, 1]``.

    Parameters
    ----------
    dataset_path:
        Path to a netCDF or zarr store.
    data_var:
        Dataset variable containing the mask field (e.g. ``constants``).
    selection_dict:
        Optional ``.sel`` kwargs (e.g. ``{"channel_c": "lsm"}``).
    invert:
        If True, use ``1 - values`` (LSM → ocean fraction).
    threshold:
        If set, binarize after invert: ``M = (M > threshold)``.
        If ``None``, keep a soft (continuous) mask.
    """
    ds = _open_mask_dataset(dataset_path)
    try:
        field = ds[data_var]
        if selection_dict:
            field = field.sel(**dict(selection_dict))
        values = np.asarray(field.values, dtype=np.float32)
    finally:
        ds.close()

    values = np.squeeze(values)
    if values.ndim != 3:
        raise ValueError(
            f"Expected mask field with 3 dimensions [F, H, W] after squeeze; "
            f"got shape {values.shape}"
        )

    mask = th.from_numpy(values)
    if invert:
        mask = 1.0 - mask
    if threshold is not None:
        mask = (mask > float(threshold)).to(dtype=th.float32)
    else:
        mask = mask.to(dtype=th.float32)
    return mask


def _as_plain_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


class PartialHEALPixConv2d(nn.Module):
    """Depthwise Liu-style partial convolution on folded HEALPix faces.

    For each channel independently:

    ``Y = Conv(X ⊙ M) * |K| / (Conv(1, M) + eps) + b``

    ``X`` and ``M`` are padded with the same HEALPix padding as other model layers.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        eps: float = 1.0e-8,
        hpx_padding_mode: str | None = None,
        nside: int | None = None,
        compile_padding: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.eps = float(eps)
        self.kernel_numel = float(self.kernel_size * self.kernel_size)

        common = dict(
            in_channels=channels,
            out_channels=channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=channels,
            bias=False,
            hpx_padding_mode=hpx_padding_mode,
            nside=nside,
            compile_padding=compile_padding,
        )
        self.feature_conv = HEALPixLayer(layer=th.nn.Conv2d, **common)
        self.mask_sum_conv = HEALPixLayer(layer=th.nn.Conv2d, **common)

        # All-ones kernel for local valid-mass; never trained.
        mask_weight = self._conv_weight(self.mask_sum_conv)
        with th.no_grad():
            mask_weight.fill_(1.0)
        mask_weight.requires_grad_(False)

        if bias:
            self.bias = nn.Parameter(th.zeros(channels))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _conv_weight(hpx_layer: HEALPixLayer) -> th.Tensor:
        # HEALPixLayer is Sequential(optional_pad, Conv2d)
        for module in hpx_layer.layers:
            if isinstance(module, th.nn.Conv2d):
                return module.weight
        raise RuntimeError("HEALPixLayer has no Conv2d submodule")

    def forward(self, x: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        x:
            Folded features ``[B*F, C, H, W]``.
        mask:
            Validity mask broadcastable to ``x`` (typically ``[1, C, H, W]`` or
            ``[B*F, C, H, W]``) with values in ``[0, 1]``.
        """
        if mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"mask spatial shape {mask.shape[-2:]} must match x {x.shape[-2:]}"
            )
        if mask.shape[1] != x.shape[1] and mask.shape[1] != 1:
            raise ValueError(
                f"mask channels {mask.shape[1]} incompatible with x channels {x.shape[1]}"
            )

        x_masked = x * mask
        y = self.feature_conv(x_masked)
        # mask_sum_conv pads M the same way as feature_conv pads X
        valid_mass = self.mask_sum_conv(mask.expand_as(x) if mask.shape[0] == 1 else mask)
        y = y * (self.kernel_numel / (valid_mass + self.eps))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class CoupledPartialConvStem(nn.Module):
    """Apply a depthwise partial-conv stem to coupled inputs ``[B, F, C, H, W]``.

    Channel count is preserved so encoder wiring and reflection channel layouts
    stay unchanged.
    """

    def __init__(
        self,
        channel_masks: th.Tensor,
        kernel_size: int = 3,
        eps: float = 1.0e-8,
        hpx_padding_mode: str | None = None,
        nside: int | None = None,
        compile_padding: bool = False,
    ):
        """
        Parameters
        ----------
        channel_masks:
            Tensor ``[C, F, H, W]`` — one spatial mask per coupled channel.
        """
        super().__init__()
        if channel_masks.ndim != 4:
            raise ValueError(
                f"channel_masks must be [C, F, H, W], got shape {tuple(channel_masks.shape)}"
            )
        channels, n_faces, _, _ = channel_masks.shape
        self.n_faces = int(n_faces)
        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=self.n_faces)
        # Stored as [1, F, C, H, W] for easy expand over batch, then fold to [B*F, C, H, W]
        self.register_buffer(
            "channel_masks",
            channel_masks.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
            persistent=True,
        )
        self.pconv = PartialHEALPixConv2d(
            channels=channels,
            kernel_size=kernel_size,
            eps=eps,
            hpx_padding_mode=hpx_padding_mode,
            nside=nside,
            compile_padding=compile_padding,
            bias=True,
        )

    def forward(self, coupled: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        coupled:
            Coupled forcing tensor ``[B, F, C, H, W]`` (after time-first permute).
        """
        if coupled.ndim != 5:
            raise ValueError(
                f"coupled inputs must be [B, F, C, H, W], got shape {tuple(coupled.shape)}"
            )
        if coupled.shape[1] != self.n_faces:
            raise ValueError(
                f"coupled face count {coupled.shape[1]} != mask faces {self.n_faces}"
            )
        if coupled.shape[2] != self.channel_masks.shape[2]:
            raise ValueError(
                f"coupled channels {coupled.shape[2]} != mask channels "
                f"{self.channel_masks.shape[2]}"
            )

        x = self.fold(coupled)
        # channel_masks: [1, F, C, H, W] -> [B, F, C, H, W] -> fold
        mask = self.fold(self.channel_masks.expand(coupled.shape[0], -1, -1, -1, -1))
        y = self.pconv(x, mask)
        return self.unfold(y)


def build_coupled_partial_conv_stem(
    config: Any,
    couplings: Sequence[Mapping[str, Any]],
    hpx_padding_mode: str | None = None,
    nside: int | None = None,
    compile_padding: bool = False,
) -> CoupledPartialConvStem | None:
    """Build a stem from a Hydra/dict config, or return ``None`` if disabled.

    Expected config::

        kernel_size: 3
        eps: 1.0e-8
        masks:
          sst:
            dataset_path: ...
            data_var: constants
            selection_dict: {channel_c: lsm}
            invert: true
            threshold: null   # soft; or e.g. 0.5 for hard
          sic: ...
    """
    if config is None:
        return None

    cfg = _as_plain_dict(config)
    masks_cfg = cfg.get("masks")
    if not masks_cfg:
        raise ValueError("coupled_partial_conv config requires a non-empty 'masks' mapping")

    channel_names = coupled_variable_channel_names(couplings)
    if not channel_names:
        raise ValueError("coupled_partial_conv requires at least one coupled variable")

    missing = sorted({n for n in channel_names if n not in masks_cfg})
    if missing:
        raise ValueError(
            "coupled_partial_conv.masks is missing entries for coupled variables: "
            + ", ".join(missing)
        )

    per_channel = []
    for name in channel_names:
        mcfg = _as_plain_dict(masks_cfg[name])
        if "dataset_path" not in mcfg:
            raise ValueError(f"coupled_partial_conv.masks.{name} requires dataset_path")
        per_channel.append(
            load_spatial_mask(
                dataset_path=mcfg["dataset_path"],
                data_var=mcfg.get("data_var", "constants"),
                selection_dict=mcfg.get("selection_dict"),
                invert=bool(mcfg.get("invert", True)),
                threshold=mcfg.get("threshold", None),
            )
        )

    channel_masks = th.stack(per_channel, dim=0)  # [C, F, H, W]
    return CoupledPartialConvStem(
        channel_masks=channel_masks,
        kernel_size=int(cfg.get("kernel_size", 3)),
        eps=float(cfg.get("eps", 1.0e-8)),
        hpx_padding_mode=hpx_padding_mode,
        nside=nside,
        compile_padding=compile_padding,
    )
