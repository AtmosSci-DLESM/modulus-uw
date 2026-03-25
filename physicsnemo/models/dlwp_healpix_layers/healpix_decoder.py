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
from omegaconf import DictConfig

from torch.utils.checkpoint import checkpoint

from .healpix_paddings import warn_deprecated_enable_healpixpad


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
        hpx_padding_mode: str = 'earth2grid',
        enable_healpixpad: bool | None = None,
        nside: int = 64,
        per_level_cln: list[bool] = None,
        per_level_checkpointing: list[bool] = None,
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
            Passed through to HEALPix blocks (e.g. ``earth2grid`` for fast CUDA padding).
        enable_healpixpad: bool, optional
            Deprecated; ignored. Use ``hpx_padding_mode`` instead.
        nside: int, optional
            Native HEALPix face height/width. Passed to blocks for isolatitude padding.
            Default 64.
        per_level_cln: list[bool], optional
            If the CLN should be applied to each level of the decoder
            If None, the CLN will based on the conv_block.conditional_layer_norm attribute
        per_level_checkpointing: list[bool], optional
            If the checkpointing should be applied to each level of the decoder
            If None, the checkpointing will not be applied
        """
        super().__init__()
        warn_deprecated_enable_healpixpad(enable_healpixpad)
        self.channel_dim = 1  # 1 in previous layout

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
                up_sample_module = instantiate(
                    config=up_sampling_block,
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    nside=nside,
                )

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            # apply conditional layer norm if enabled for this level
            block_config = conv_block.copy()
            if "conditional_layer_norm" in block_config and block_config.conditional_layer_norm is not None:
                if not per_level_cln[n]:
                    block_config.conditional_layer_norm = None

            conv_module = instantiate(
                config=conv_block,
                in_channels=curr_channel * 2
                if n > 0
                else curr_channel,  # Considering skip connection
                latent_channels=curr_channel,
                out_channels=next_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
                enable_nhwc=enable_nhwc,
                hpx_padding_mode=hpx_padding_mode,
                nside=nside,
            )

            # Recurrent module
            if recurrent_block is not None:
                rec_module = instantiate(
                    config=recurrent_block,
                    in_channels=next_channel,
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    nside=nside,
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
        self.output_layer = instantiate(
            config=output_layer,
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            enable_nhwc=enable_nhwc,
            hpx_padding_mode=hpx_padding_mode,
            nside=nside,
        )

    def _forward_layer_pass(self, layer: th.nn.Module, x: th.Tensor, skip_connection: th.Tensor=None, conditions_cln: th.Tensor=None) -> th.Tensor:
        """
        Forward pass of a single layer of the decoder
        Handled seperately to allow for checkpointing of the layer
        """
        
        if layer["upsamp"] is not None:
            up = layer["upsamp"](x)
            x = th.cat([up, skip_connection], dim=self.channel_dim)
        # apply the conv block, check if the layer accepts conditional inputs
        if hasattr(layer["conv"], "cln_enabled") and layer["conv"].cln_enabled:
            if conditions_cln is not None:
                x = layer["conv"](x, conditions_cln=conditions_cln)
            else:
                raise ValueError("Conditional inputs are required for layers with cln_enabled=True")
        else:
            x = layer["conv"](x)

        return x
    def forward(self, inputs: Sequence, conditions_cln: Sequence = None) -> th.Tensor:
        """
        Forward pass of the HEALPix Unet decoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to decode
        conditions_cln: Sequence, optional
            The conditional inputs for the normalization layers.

        Returns
        -------
        torch.Tensor: The decoded values
        """
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            skip_connection = inputs[-1 - n] if layer["upsamp"] is not None else None
            if self.per_level_checkpointing[n]:
                x = checkpoint(self._forward_layer_pass, layer, x, skip_connection, conditions_cln, use_reentrant=False)
            else:
                x = self._forward_layer_pass(layer, x, skip_connection, conditions_cln)

            # apply the recurrent block if it exists
            # NOTE: this should be done after the checkpointing to avoid issues with
            # the recurrent block changing during reinitialization
            if layer["recurrent"] is not None:
                x = layer["recurrent"](x)

        return self.output_layer(x)

    def reset(self):
        """Resets the state of the decoder layers"""
        for layer in self.decoder:
            if layer["recurrent"] is not None:
                layer["recurrent"].reset()
