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
from torch.utils.checkpoint import checkpoint as ckpt_fn


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
        enable_healpixpad: bool = False,
        cln_per_level: Sequence = None,
        act_ckpt_levels: Sequence = None,
        hpx_padding_mode: str = "karlbauer",
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
        enable_healpixpad, bool, optional
            If the healpixpad library should be used (if installed)
        cln_per_level: Sequence, optional
            Per-level flag for conditional layer norm. If provided, length must equal
            len(n_channels). Where cln_per_level[n] is False, that level gets
            conditional_layer_norm=None. If None (default), all levels use conv_block as-is (backward compatible).
        act_ckpt_levels: Sequence, optional
            Per-level boolean flags for activation checkpointing. If provided,
            length must equal len(n_channels). Levels marked True will have
            their forward pass wrapped with torch.utils.checkpoint to trade
            compute for memory. If None (default), no checkpointing is used.
        """
        super().__init__()
        self.n_channels = n_channels

        if act_ckpt_levels is not None and len(act_ckpt_levels) != len(n_channels):
            raise ValueError(
                f"act_ckpt_levels length ({len(act_ckpt_levels)}) must equal "
                f"number of encoder levels ({len(n_channels)})"
            )
        self.act_ckpt_levels = act_ckpt_levels

        if cln_per_level is not None and len(cln_per_level) != len(n_channels):
            raise ValueError(
                f"cln_per_level length ({len(cln_per_level)}) must equal number of encoder levels ({len(n_channels)})"
            )

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
                        enable_nhwc=enable_nhwc,
                        enable_healpixpad=enable_healpixpad,
                        hpx_padding_mode=hpx_padding_mode,
                    )
                )
            if cln_per_level is not None and not cln_per_level[n]:
                modules.append(
                    instantiate(
                        config=conv_block,
                        in_channels=old_channels,
                        latent_channels=curr_channel,
                        out_channels=curr_channel,
                        dilation=dilations[n],
                        n_layers=n_layers[n],
                        enable_nhwc=enable_nhwc,
                        enable_healpixpad=enable_healpixpad,
                        conditional_layer_norm=None,
                        hpx_padding_mode=hpx_padding_mode,
                    )
                )
            else:
                modules.append(
                    instantiate(
                        config=conv_block,
                        in_channels=old_channels,
                        latent_channels=curr_channel,
                        out_channels=curr_channel,
                        dilation=dilations[n],
                        n_layers=n_layers[n],
                        enable_nhwc=enable_nhwc,
                        enable_healpixpad=enable_healpixpad,
                        hpx_padding_mode=hpx_padding_mode,
                    )
                )
            old_channels = curr_channel

            self.encoder.append(th.nn.Sequential(*modules))

        self.encoder = th.nn.ModuleList(self.encoder)

    def _forward_level(self, layer_group, x, conditions_cln):
        """Run one encoder level (downsample + conv)."""
        for layer in layer_group:
            if getattr(layer, 'cln_enabled', False):
                if conditions_cln is None:
                    raise ValueError("Conditional inputs are required for layers with cln_enabled=True")
                x = layer(x, conditions_cln=conditions_cln)
            else:
                x = layer(x)
        return x

    def forward(self, inputs: Sequence, conditions_cln: th.Tensor=None) -> Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to encode
        conditions_cln: th.Tensor: optional
            The conditional inputs for the normalization layers.

        Returns
        -------
        Sequence: The encoded values
        """
        outputs = []
        for n, layer_group in enumerate(self.encoder):
            if self.act_ckpt_levels is not None and self.act_ckpt_levels[n]:
                interim_output = ckpt_fn(
                    self._forward_level, layer_group, inputs, conditions_cln,
                    use_reentrant=False,
                )
            else:
                interim_output = self._forward_level(layer_group, inputs, conditions_cln)
            outputs.append(interim_output)
            inputs = outputs[-1]
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass
