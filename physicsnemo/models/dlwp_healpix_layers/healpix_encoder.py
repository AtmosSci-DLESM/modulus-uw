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


class UNetEncoder(th.nn.Module):
    """Generic UNetEncoder that can be applied to arbitrary meshes."""

    def __init__(
        self,
        conv_block: DictConfig,
        down_sampling_block: DictConfig,
        recurrent_block: DictConfig = None,
        input_channels: int = 3,
        n_channels: Sequence = (16, 32, 64),
        n_layers: Sequence = (2, 2, 1),
        dilations: list = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: str | None = None,
        compile_padding: bool = False,
        nside: Sequence[int] = (64, 32, 16),
        per_level_cln: Sequence[bool] = None,
        per_level_checkpointing: Sequence[bool] = None,
        enable_healpixpad: bool | None = None,
        structural_input_even: int | None = None,
        odd_fraction: float = 0.25,
    ):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        down_sampling_block: DictConfig
            dictionary of instantiable parameters for the downsample block
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        input_channels: int, optional
            Number of input channels
        n_channels: Sequence, optional
            The number of channels in each encoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
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
            Native HEALPix face height/width (``H == W``) per encoder level, shallowest
            (full resolution) to deepest. ``len(nside)`` must equal ``len(n_channels)``.
            Default ``(64, 32, 16)`` for the default three-level encoder.
        per_level_cln: list[bool] | None, optional
            If the CLN should be applied to each level of the encoder
            If None, the CLN will based on the conv_block.conditional_layer_norm attribute
        per_level_checkpointing: list[bool] | None, optional
            If the checkpointing should be applied to each level of the encoder
            If None, the checkpointing will not be applied
        enable_healpixpad: bool, optional
            Deprecated; see ``hpx_padding_mode`` (legacy mapping when mode omitted).
        odd_fraction: float, optional
            Global even|odd bank split for ReflectionSteerable blocks. Set on
            ``HEALPixRecUNet`` and propagated here; do not set per conv block.
        """
        super().__init__()
        hpx_padding_mode = warn_deprecated_enable_healpixpad(enable_healpixpad, hpx_padding_mode)
        self.odd_fraction = float(odd_fraction)
        if len(nside) != len(n_channels):
            raise ValueError(
                f"nside must have the same length as n_channels ({len(n_channels)}), "
                f"got {len(nside)}: {nside!r}"
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

        # Build encoder
        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                modules.append(
                    instantiate(
                        config=down_sampling_block,
                        in_channels=old_channels,
                        enable_nhwc=enable_nhwc,
                        hpx_padding_mode=hpx_padding_mode,
                        compile_padding=compile_padding,
                        nside=nside[n - 1],
                    )
                )

            # apply conditional layer norm if enabled for this level
            block_config = strip_nested_odd_fraction(
                conv_block, self.odd_fraction, label="encoder.conv_block"
            )
            if "conditional_layer_norm" in block_config and block_config.conditional_layer_norm is not None:
                if not per_level_cln[n]:
                    block_config.conditional_layer_norm = None

            target = str(OmegaConf.select(block_config, "_target_", default=""))
            is_reflection_steerable = "ReflectionSteerable" in target
            conv_extra = {}
            if is_reflection_steerable:
                conv_extra["odd_fraction"] = self.odd_fraction
                if n == 0 and structural_input_even is not None:
                    conv_extra["in_even"] = structural_input_even
                elif n > 0:
                    conv_extra["in_even"] = bank_sizes(old_channels, self.odd_fraction)[0]
                conv_extra["out_even"] = bank_sizes(curr_channel, self.odd_fraction)[0]

            modules.append(
                instantiate(
                    config=block_config,
                    in_channels=old_channels,
                    latent_channels=curr_channel,
                    out_channels=curr_channel,
                    dilation=dilations[n],
                    n_layers=n_layers[n],
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    compile_padding=compile_padding,
                    nside=nside[n],
                    **conv_extra,
                )
            )
            old_channels = curr_channel

            self.encoder.append(th.nn.Sequential(*modules))

        self.encoder = th.nn.ModuleList(self.encoder)


    def _forward_layer_pass(self, layer_group: th.nn.Module, inp: th.Tensor, conditions_cln: th.Tensor=None, sin_lat_gate: th.Tensor=None) -> th.Tensor:
        """
        Forward pass of single layer of the encoder
        Handled seperately to allow for checkpointing of the layer

        Parameters
        ----------
        inputs: Sequence
            The inputs to encode
        conditions_cln: th.Tensor: optional
            The conditional inputs for the normalization layers.
        sin_lat_gate: th.Tensor, optional
            Odd geometric field (``sin(lat)``) for optional cross-parity 1×1
            paths in ReflectionSteerable blocks.

        Returns
        -------
        Sequence: The encoded values
        """
        interim_output = inp
        for layer in layer_group:
            # check if class accepts cln inputs
            if getattr(layer, 'cln_enabled', False):
                if conditions_cln is None:
                    raise ValueError("Conditional inputs are required for layers with cln_enabled=True")
                if getattr(layer, "reflection_steerable", False):
                    interim_output = layer(
                        interim_output, conditions_cln=conditions_cln, sin_lat_gate=sin_lat_gate
                    )
                else:
                    interim_output = layer(interim_output, conditions_cln=conditions_cln)
            elif getattr(layer, "reflection_steerable", False):
                interim_output = layer(interim_output, sin_lat_gate=sin_lat_gate)
            else:
                interim_output = layer(interim_output)
            
        # Return the outputs of the last layer
        return interim_output

    def forward(self, inputs: Sequence, conditions_cln: th.Tensor=None, sin_lat_gate: th.Tensor=None) -> Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to encode
        conditions_cln: th.Tensor: optional
            The conditional inputs for the normalization layers.
        sin_lat_gate: th.Tensor, optional
            Odd geometric field (``sin(lat)``) at the current encoder resolution.
            If a sequence is provided, ``sin_lat_gate[n]`` is used at level ``n``.

        Returns
        -------
        Sequence: The encoded values
        """
        outputs = []
        for n, layer_group in enumerate(self.encoder):
            interim_output = inputs
            gate_n = None
            if sin_lat_gate is not None:
                gate_n = sin_lat_gate[n] if isinstance(sin_lat_gate, (list, tuple)) else sin_lat_gate
            if self.per_level_checkpointing[n]:
                interim_output = checkpoint(
                    self._forward_layer_pass,
                    layer_group,
                    interim_output,
                    conditions_cln,
                    gate_n,
                    use_reentrant=False,
                )
            else:
                interim_output = self._forward_layer_pass(
                    layer_group, interim_output, conditions_cln, gate_n
                )
            outputs.append(interim_output)
            inputs = outputs[-1]
        # Return the outputs of the last layer
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass
