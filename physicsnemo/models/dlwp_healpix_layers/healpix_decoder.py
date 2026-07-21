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

from typing import Sequence

import torch as th
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from torch.utils.checkpoint import checkpoint

from .healpix_paddings import warn_deprecated_enable_healpixpad
from .reflection_ops import bank_sizes, strip_nested_odd_fraction
from .reflection_steerable_blocks import parity_skip_concat


class UNetDecoder(th.nn.Module):
    """Generic UNetDecoder that can be applied to arbitrary meshes."""

    def __init__(
        self,
        conv_block: DictConfig,
        up_sampling_block: DictConfig,
        output_layer: DictConfig,
        recurrent_block: DictConfig = None,
        n_channels: Sequence = (64, 32, 16),
        n_layers: Sequence = (1, 2, 2),
        output_channels: int = 1,
        dilations: list = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: str | None = None,
        compile_padding: bool = False,
        nside: Sequence[int] = (64, 32, 16),
        per_level_cln: list[bool] = None,
        per_level_checkpointing: list[bool] = None,
        enable_healpixpad: bool | None = None,
        structural_output_even: int | None = None,
        odd_fraction: float = 0.25,
    ):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        up_sampling_blockoder: DictConfig
            dictionary of instantiable parameters for the upsampling block
        output_layer: DictConfig
            dictionary of instantiable parameters for the output layer
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        n_channels: Sequence, optional
            The number of channels in each decoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
        output_channels: int, optional
            Number of output channels
        dilations: list, optional
            List of dialtions to use for the the convolutional blocks
        enable_nhwc: bool, optional
            If channel last format should be used
        hpx_padding_mode: str, optional
            Passed through to HEALPix blocks. ``None`` (omitted) defaults to ``earth2grid``
            unless deprecated ``enable_healpixpad`` is set without an explicit mode.
        compile_padding: bool, optional
            If True, apply torch compile to the padding module.
        nside: Sequence[int], optional
            Per-level face size from full resolution (``nside[0]``) to deepest (``nside[-1]``),
            same convention as the encoder. ``len(nside)`` must equal ``len(n_channels)``.
            Default ``(64, 32, 16)`` for the default three-level decoder.
        per_level_cln: list[bool], optional
            If the CLN should be applied to each level of the decoder
            If None, the CLN will based on the conv_block.conditional_layer_norm attribute
        per_level_checkpointing: list[bool], optional
            If the checkpointing should be applied to each level of the decoder
            If None, the checkpointing will not be applied
        enable_healpixpad: bool, optional
            Deprecated; see ``hpx_padding_mode`` (legacy mapping when mode omitted).
        odd_fraction: float, optional
            Global even|odd bank split for ReflectionSteerable blocks. Set on
            ``HEALPixRecUNet`` and propagated here; do not set per block.
        """
        super().__init__()
        hpx_padding_mode = warn_deprecated_enable_healpixpad(enable_healpixpad, hpx_padding_mode)
        self.odd_fraction = float(odd_fraction)
        self.channel_dim = 1  # 1 in previous layout
        if len(nside) != len(n_channels):
            raise ValueError(
                f"nside must have the same length as n_channels; got {len(nside)} "
                f"for nside and {len(n_channels)} for n_channels"
            )

        if per_level_cln is not None and len(per_level_cln) != len(n_channels):
            raise ValueError(
                "per_level_cln must be a list of booleans of the same length as n_channels"
                f"Got {len(per_level_cln)} for per_level_cln and {len(n_channels)} for n_channels"
            )
        per_level_cln = per_level_cln if per_level_cln is not None else [True] * len(n_channels)

        if per_level_checkpointing is not None and len(per_level_checkpointing) != len(n_channels):
            raise ValueError(
                "per_level_checkpointing must be a list of booleans of the same length as n_channels"
                f"Got {len(per_level_checkpointing)} for per_level_checkpointing and {len(n_channels)} for n_channels"
            )
        # Generate the per_level_checkpointing list, simplifies forward logic
        self.per_level_checkpointing = per_level_checkpointing if per_level_checkpointing is not None else [False] * len(n_channels)

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        self.decoder = []
        for n, curr_channel in enumerate(n_channels):
            # Second half of the synoptic layer does not need an upsampling module
            if n == 0:
                up_sample_module = None
            else:
                up_config = strip_nested_odd_fraction(
                    up_sampling_block, self.odd_fraction, label="decoder.up_sampling_block"
                )
                up_extra = {}
                up_target = str(OmegaConf.select(up_config, "_target_", default=""))
                if "ReflectionSteerable" in up_target:
                    up_extra["odd_fraction"] = self.odd_fraction
                up_sample_module = instantiate(
                    config=up_config,
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    compile_padding=compile_padding,
                    nside=nside[len(n_channels) - n],
                    **up_extra,
                )

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            # apply conditional layer norm if enabled for this level
            block_config = strip_nested_odd_fraction(
                conv_block, self.odd_fraction, label="decoder.conv_block"
            )
            if "conditional_layer_norm" in block_config and block_config.conditional_layer_norm is not None:
                if not per_level_cln[n]:
                    block_config.conditional_layer_norm = None

            target = str(OmegaConf.select(block_config, "_target_", default=""))
            is_reflection_steerable = "ReflectionSteerable" in target
            conv_extra = {}
            if is_reflection_steerable:
                conv_extra["odd_fraction"] = self.odd_fraction
                if n > 0:
                    ne, _ = bank_sizes(curr_channel, self.odd_fraction)
                    conv_extra["in_even"] = 2 * ne
                ne_out, _ = bank_sizes(next_channel, self.odd_fraction)
                conv_extra["out_even"] = ne_out

            conv_module = instantiate(
                config=block_config,
                in_channels=curr_channel * 2
                if n > 0
                else curr_channel,  # Considering skip connection
                latent_channels=curr_channel,
                out_channels=next_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                hpx_padding_mode=hpx_padding_mode,
                compile_padding=compile_padding,
                nside=nside[len(n_channels) - 1 - n],
                **conv_extra,
            )

            # Recurrent module
            if recurrent_block is not None:
                rec_config = strip_nested_odd_fraction(
                    recurrent_block, self.odd_fraction, label="decoder.recurrent_block"
                )
                rec_extra = {}
                rec_target = str(OmegaConf.select(rec_config, "_target_", default=""))
                if "ReflectionSteerable" in rec_target:
                    rec_extra["odd_fraction"] = self.odd_fraction
                rec_module = instantiate(
                    config=rec_config,
                    in_channels=next_channel,
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    compile_padding=compile_padding,
                    nside=nside[len(n_channels) - 1 - n],
                    **rec_extra,
                )
            else:
                rec_module = None

            self.decoder.append(
                th.nn.ModuleDict(
                    {
                        "upsamp": up_sample_module,
                        "conv": conv_module,
                        "recurrent": rec_module,
                    }
                )
            )

        self.decoder = th.nn.ModuleList(self.decoder)
        # (Linear) Output layer
        out_config = strip_nested_odd_fraction(
            output_layer, self.odd_fraction, label="decoder.output_layer"
        )
        out_extra = {}
        out_target = str(OmegaConf.select(out_config, "_target_", default=""))
        if "ReflectionSteerable" in out_target:
            out_extra["odd_fraction"] = self.odd_fraction
            if structural_output_even is not None:
                out_extra["out_even"] = structural_output_even
            ne_in, _ = bank_sizes(curr_channel, self.odd_fraction)
            out_extra["in_even"] = ne_in
        self.output_layer = instantiate(
            config=out_config,
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            enable_nhwc=enable_nhwc,
            hpx_padding_mode=hpx_padding_mode,
            compile_padding=compile_padding,
            nside=nside[0],
            **out_extra,
        )
        # Detect ReflectionSteerable Hydra targets (vs baseline SymmetricConvNeXt blocks).
        self.reflection_steerable = "ReflectionSteerable" in str(
            OmegaConf.select(conv_block, "_target_", default="")
        )

    def _forward_layer_pass(
        self,
        layer: th.nn.Module,
        x: th.Tensor,
        skip_connection: th.Tensor = None,
        conditions_cln: th.Tensor = None,
        sin_lat_gate: th.Tensor = None,
    ) -> th.Tensor:
        """
        Forward pass of a single layer of the decoder
        Handled seperately to allow for checkpointing of the layer
        """

        if layer["upsamp"] is not None:
            up = layer["upsamp"](x)
            if self.reflection_steerable:
                x = parity_skip_concat(up, skip_connection, self.odd_fraction)
            else:
                x = th.cat([up, skip_connection], dim=self.channel_dim)
        if hasattr(layer["conv"], "cln_enabled") and layer["conv"].cln_enabled:
            if conditions_cln is not None:
                if getattr(layer["conv"], "reflection_steerable", False):
                    x = layer["conv"](x, conditions_cln=conditions_cln, sin_lat_gate=sin_lat_gate)
                else:
                    x = layer["conv"](x, conditions_cln=conditions_cln)
            else:
                raise ValueError("Conditional inputs are required for layers with cln_enabled=True")
        elif getattr(layer["conv"], "reflection_steerable", False):
            x = layer["conv"](x, sin_lat_gate=sin_lat_gate)
        else:
            x = layer["conv"](x)

        return x

    def forward(self, inputs: Sequence, conditions_cln: Sequence = None, sin_lat_gate=None) -> th.Tensor:
        """
        Forward pass of the HEALPix Unet decoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to decode
        conditions_cln: Sequence, optional
            The conditional inputs for the normalization layers.
        sin_lat_gate: optional
            Odd geometric gate (sin_lat). Sequence uses one tensor per decoder level
            (deepest-first), or a single tensor for all levels.
        """
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            skip_connection = inputs[-1 - n] if layer["upsamp"] is not None else None
            gate_n = None
            if sin_lat_gate is not None:
                gate_n = sin_lat_gate[n] if isinstance(sin_lat_gate, (list, tuple)) else sin_lat_gate
            if self.per_level_checkpointing[n]:
                x = checkpoint(
                    self._forward_layer_pass,
                    layer,
                    x,
                    skip_connection,
                    conditions_cln,
                    gate_n,
                    use_reentrant=False,
                )
            else:
                x = self._forward_layer_pass(layer, x, skip_connection, conditions_cln, gate_n)

            # apply the recurrent block if it exists
            # NOTE: this should be done after the checkpointing to avoid issues with
            # the recurrent block changing during reinitialization
            if layer["recurrent"] is not None:
                if getattr(layer["recurrent"], "reflection_steerable", False):
                    x = layer["recurrent"](x, sin_lat_gate=gate_n)
                else:
                    x = layer["recurrent"](x)

        if getattr(self.output_layer, "reflection_steerable", False):
            gate_out = None
            if sin_lat_gate is not None:
                gate_out = sin_lat_gate[-1] if isinstance(sin_lat_gate, (list, tuple)) else sin_lat_gate
            return self.output_layer(x, sin_lat_gate=gate_out)
        return self.output_layer(x)

    def reset(self):
        """Resets the state of the decoder layers"""
        for layer in self.decoder:
            if layer["recurrent"] is not None:
                layer["recurrent"].reset()
