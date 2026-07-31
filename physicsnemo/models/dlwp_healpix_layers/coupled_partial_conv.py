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

Depthwise feature kernels are always projected onto the even-under-``R`` subspace
(``K = R(K)``) with gradient hooks so Adam stays on-manifold. That is correct for
both even and odd coupled channels (depthwise odd→odd also needs an even kernel).
Odd channels additionally force bias to zero. Channel parity is supplied via
``channel_is_odd`` / RecUNet ``odd_coupled_variables``. Full stem intertwining
``stem(ρx) = ρ stem(x)`` additionally needs a ρ-symmetric mask ``M ≈ ρM``
(B24 ocean is fine; real geographic LSM breaks that).

The stem itself is Hydra-instantiable via ``coupled_partial_conv.stem`` (default
:class:`CoupledPartialConvStem`). Custom stems must accept ``channel_masks`` plus
the HEALPix padding kwargs and map ``[B, F, C, H, W] → [B, F, C, H, W]`` (channel
count preserved for encoder / reflection layouts).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch as th
import torch.nn as nn
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .healpix_layers import HEALPixLayer
from .healpix_paddings import HEALPixFoldFaces, HEALPixUnfoldFaces
from .reflection_ops import project_even_kernel

logger = logging.getLogger(__name__)

DEFAULT_COUPLED_PARTIAL_CONV_STEM_TARGET = (
    "physicsnemo.models.dlwp_healpix_layers.coupled_partial_conv.CoupledPartialConvStem"
)


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


def _normalize_channel_is_odd(
    channels: int,
    channel_is_odd: Optional[Union[Sequence[bool], th.Tensor]],
) -> th.Tensor:
    """Return a length-``channels`` bool tensor; ``None`` means all-even."""
    if channel_is_odd is None:
        return th.zeros(channels, dtype=th.bool)
    if isinstance(channel_is_odd, th.Tensor):
        odd = channel_is_odd.detach().to(dtype=th.bool).reshape(-1).cpu()
    else:
        odd = th.tensor([bool(x) for x in channel_is_odd], dtype=th.bool)
    if odd.numel() != channels:
        raise ValueError(
            f"channel_is_odd length {odd.numel()} must match channels={channels}"
        )
    return odd


class PartialHEALPixConv2d(nn.Module):
    """Depthwise Liu-style partial convolution on folded HEALPix faces.

    For each channel independently:

    ``Y = Conv(X ⊙ M) * |K| / (Conv(1, M) + eps) + b``

    ``X`` and ``M`` are padded with the same HEALPix padding as other model layers.

    Feature kernels are constrained to the even-under-``R`` subspace
    (``K = R(K)``) via init/``eval()`` projection and gradient hooks — required
    for both even and odd depthwise channels. Even channels may have a free
    bias; odd channels (see ``channel_is_odd``) force bias to zero.
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
        enable_nhwc: bool = False,
        channel_is_odd: Optional[Union[Sequence[bool], th.Tensor]] = None,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.eps = float(eps)
        self.kernel_numel = float(self.kernel_size * self.kernel_size)
        self.register_buffer(
            "channel_is_odd",
            _normalize_channel_is_odd(channels, channel_is_odd),
            persistent=False,
        )

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
            enable_nhwc=enable_nhwc,
        )
        self.feature_conv = HEALPixLayer(layer=th.nn.Conv2d, **common)
        self.mask_sum_conv = HEALPixLayer(layer=th.nn.Conv2d, **common)

        # All-ones kernel for local valid-mass; never trained (already even under R).
        mask_weight = self._conv_weight(self.mask_sum_conv)
        with th.no_grad():
            mask_weight.fill_(1.0)
        mask_weight.requires_grad_(False)

        # Even-kernel manifold (depthwise even→even and odd→odd): project init and
        # keep Adam on-manifold via grad hooks (same pattern as ReflectionSteerable).
        feat_weight = self._conv_weight(self.feature_conv)
        with th.no_grad():
            feat_weight.copy_(project_even_kernel(feat_weight))
        feat_weight.register_hook(lambda g: project_even_kernel(g))

        if bias:
            self.bias = nn.Parameter(th.zeros(channels))
            if bool(self.channel_is_odd.any()):
                with th.no_grad():
                    self.bias[self.channel_is_odd] = 0
                odd_mask = self.channel_is_odd

                def _odd_bias_hook(grad, odd_mask=odd_mask):
                    g = grad.clone()
                    g[odd_mask] = 0
                    return g

                self.bias.register_hook(_odd_bias_hook)
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _conv_weight(hpx_layer: HEALPixLayer) -> th.Tensor:
        # HEALPixLayer is Sequential(optional_pad, Conv2d)
        for module in hpx_layer.layers:
            if isinstance(module, th.nn.Conv2d):
                return module.weight
        raise RuntimeError("HEALPixLayer has no Conv2d submodule")

    def train(self, mode: bool = True):
        was_training = self.training
        out = super().train(mode)
        # Re-snap weights / odd bias when entering eval so checkpoints / inference are exact.
        if was_training and not mode:
            self._project_parity_params_()
        return out

    def _project_feature_kernel_(self) -> None:
        """Backward-compatible alias for kernel-only projection."""
        w = self._conv_weight(self.feature_conv)
        with th.no_grad():
            w.copy_(project_even_kernel(w))

    def _project_parity_params_(self) -> None:
        self._project_feature_kernel_()
        if self.bias is not None and bool(self.channel_is_odd.any()):
            with th.no_grad():
                self.bias[self.channel_is_odd] = 0

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
    """Default depthwise partial-conv stem for coupled inputs ``[B, F, C, H, W]``.

    Channel count is preserved so encoder wiring and reflection channel layouts
    stay unchanged. Swap via Hydra ``coupled_partial_conv.stem._target_``.

    ``channel_is_odd`` marks coupled channels that flip under equatorial reflection
    (wired from RecUNet/UNet ``odd_coupled_variables``); those channels keep an
    even kernel but zero bias.
    """

    def __init__(
        self,
        channel_masks: th.Tensor,
        kernel_size: int = 3,
        dilation: int = 1,
        eps: float = 1.0e-8,
        bias: bool = True,
        hpx_padding_mode: str | None = None,
        nside: int | None = None,
        compile_padding: bool = False,
        enable_nhwc: bool = False,
        channel_is_odd: Optional[Union[Sequence[bool], th.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        channel_masks:
            Tensor ``[C, F, H, W]`` — one spatial mask per coupled channel.
        kernel_size, dilation, eps, bias:
            Forwarded to :class:`PartialHEALPixConv2d`.
        hpx_padding_mode, nside, compile_padding, enable_nhwc:
            HEALPix padding / memory-format options; normally injected by the
            model builder to match the parent RecUNet/UNet.
        channel_is_odd:
            Per-channel odd/even flags (length ``C``). ``None`` means all even.
        """
        super().__init__()
        if channel_masks.ndim != 4:
            raise ValueError(
                f"channel_masks must be [C, F, H, W], got shape {tuple(channel_masks.shape)}"
            )
        channels, n_faces, _, _ = channel_masks.shape
        self.n_faces = int(n_faces)
        self.fold = HEALPixFoldFaces(enable_nhwc=enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=self.n_faces, enable_nhwc=enable_nhwc)
        # Store as [F, C, H, W] (rank 4) so model.to(channels_last) succeeds when
        # enable_nhwc is on. Repeat over batch in forward, then fold to [B*F, C, H, W].
        self.register_buffer(
            "channel_masks",
            channel_masks.permute(1, 0, 2, 3).contiguous(),
            persistent=True,
        )
        self.pconv = PartialHEALPixConv2d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            eps=eps,
            hpx_padding_mode=hpx_padding_mode,
            nside=nside,
            compile_padding=compile_padding,
            bias=bias,
            enable_nhwc=enable_nhwc,
            channel_is_odd=channel_is_odd,
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
        if coupled.shape[2] != self.channel_masks.shape[1]:
            raise ValueError(
                f"coupled channels {coupled.shape[2]} != mask channels "
                f"{self.channel_masks.shape[1]}"
            )

        x = self.fold(coupled)
        # channel_masks: [F, C, H, W] -> [B, F, C, H, W] -> fold
        mask = self.fold(
            self.channel_masks.unsqueeze(0).expand(coupled.shape[0], -1, -1, -1, -1)
        )
        y = self.pconv(x, mask)
        return self.unfold(y)


def _resolve_stem_config(cfg: dict) -> Any:
    """Build a Hydra stem config, defaulting to :class:`CoupledPartialConvStem`.

    Preferred form::

        stem:
          _target_: ...CoupledPartialConvStem
          kernel_size: 3
          eps: 1.0e-8

    Legacy flat keys ``kernel_size`` / ``eps`` / ``dilation`` / ``bias`` at the
    ``coupled_partial_conv`` root are still accepted when ``stem`` is omitted.

    Returns a ``DictConfig`` for string targets, or a plain dict when ``_target_``
    is already a class object (OmegaConf cannot store class values).
    """
    stem_cfg = cfg.get("stem")
    if stem_cfg is None:
        return OmegaConf.create(
            {
                "_target_": DEFAULT_COUPLED_PARTIAL_CONV_STEM_TARGET,
                "kernel_size": cfg.get("kernel_size", 3),
                "dilation": cfg.get("dilation", 1),
                "eps": cfg.get("eps", 1.0e-8),
                "bias": cfg.get("bias", True),
            }
        )

    if isinstance(stem_cfg, DictConfig):
        if "_target_" not in stem_cfg:
            with OmegaConf.set_struct(stem_cfg, False):
                stem_cfg._target_ = DEFAULT_COUPLED_PARTIAL_CONV_STEM_TARGET
        return stem_cfg

    stem_cfg = dict(stem_cfg)
    target = stem_cfg.get("_target_", DEFAULT_COUPLED_PARTIAL_CONV_STEM_TARGET)
    stem_cfg["_target_"] = target
    # Class objects cannot round-trip through OmegaConf.create.
    if isinstance(target, type):
        return stem_cfg
    return OmegaConf.create(stem_cfg)


def build_coupled_partial_conv_stem(
    config: Any,
    couplings: Sequence[Mapping[str, Any]],
    hpx_padding_mode: str | None = None,
    nside: int | None = None,
    compile_padding: bool = False,
    enable_nhwc: bool = False,
    odd_coupled_variables: Optional[Sequence[str]] = None,
) -> nn.Module | None:
    """Build a stem from a Hydra/dict config, or return ``None`` if disabled.

    Expected config::

        masks:
          sst:
            dataset_path: ...
            data_var: constants
            selection_dict: {channel_c: lsm}
            invert: true
            threshold: null   # soft; or e.g. 0.5 for hard
          sic: ...
        stem:
          _target_: physicsnemo.models.dlwp_healpix_layers.coupled_partial_conv.CoupledPartialConvStem
          kernel_size: 3
          dilation: 1
          eps: 1.0e-8
          bias: true

    ``odd_coupled_variables`` names channels that flip under equatorial reflection;
    they are expanded against ``coupled_variable_channel_names(couplings)`` into
    ``channel_is_odd`` for the stem. Names that never appear in couplings are ignored.

    ``stem`` is Hydra-instantiated with ``channel_masks`` and the parent model's
    HEALPix padding kwargs injected. Custom ``_target_`` modules must preserve
    coupled channel count in ``forward``.
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

    odd_set = {str(v) for v in (odd_coupled_variables or [])}
    unused_odd = sorted(odd_set - set(channel_names))
    if unused_odd:
        logger.warning(
            "odd_coupled_variables not present in couplings (ignored for stem parity): %s",
            ", ".join(unused_odd),
        )
    channel_is_odd = [name in odd_set for name in channel_names]

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
    stem_cfg = _resolve_stem_config(cfg)
    return instantiate(
        stem_cfg,
        channel_masks=channel_masks,
        hpx_padding_mode=hpx_padding_mode,
        nside=nside,
        compile_padding=compile_padding,
        enable_nhwc=enable_nhwc,
        channel_is_odd=channel_is_odd,
    )
