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

from typing import List, Optional, Sequence, Tuple

import torch as th
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint as ckpt_fn

from .normalization import AdditiveSpatialNoise


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
        enable_healpixpad: bool = False,
        cln_per_level: Sequence = None,
        act_ckpt_levels: Sequence = None,
        asn_per_level: Sequence = None,
        asn_spatial_sizes: Optional[Sequence[Tuple[int, int]]] = None,
        asn_alpha: float = 0.1,
        asn_noise_dim: int = 32,
        asn_mlp_hidden_dims: Optional[Sequence[int]] = None,
        asn_num_faces: int = 12,
        hpx_padding_mode: str = "karlbauer",
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
        enable_healpixpad, bool, optional
            If the healpixpad library should be used if installed
        cln_per_level: Sequence, optional
            Per-level flag for conditional layer norm. If provided, length must equal
            len(n_channels). Where cln_per_level[n] is False, that level gets
            conditional_layer_norm=None. If None (default), all levels use conv_block as-is (backward compatible).
        act_ckpt_levels: Sequence, optional
            Per-level boolean flags for activation checkpointing. If provided,
            length must equal len(n_channels). Levels marked True will have
            their upsample+conv forward pass wrapped with
            torch.utils.checkpoint to trade compute for memory. Recurrent
            blocks are always run outside the checkpoint boundary to protect
            hidden state. If None (default), no checkpointing is used.
        asn_per_level: Sequence, optional
            Per-level flag for AdditiveSpatialNoise (SDL) in pre-skip position (after
            upsample, before skip concat). If provided, length must equal len(n_channels).
            Where asn_per_level[n] is True (and n > 0), that level gets ASN (Z -> MLP -> S,
            then Fout = Fin + alpha * R ⊙ S ⊙ M) applied before concatenating with skip.
            Z is sampled per-batch (global style). If None (default), no ASN is used (backward compatible).
        asn_spatial_sizes: Sequence[Tuple[int, int]], optional
            Per-level (H, W) for ASN. Required when any asn_per_level is True. Length must equal
            len(n_channels). asn_spatial_sizes[n] is the spatial size of the decoder feature map
            at level n (after upsample for that level). Used to initialize M in __init__ (no lazy init).
        asn_alpha: float, optional
            Fixed scalar scaling the additive spatial noise magnitude. Default 0.1.
        asn_noise_dim: int, optional
            Dimension of latent Z for the SDL style vector. Default 32.
        asn_mlp_hidden_dims: Sequence[int], optional
            Hidden sizes for MLP mapping Z -> S. If None, defaults to [64, 64].
        asn_num_faces: int, optional
            Number of HEALPix faces for ASN unfold/refold (default 12). Must match fold convention.
        """
        super().__init__()
        self.channel_dim = 1  # 1 in previous layout

        if act_ckpt_levels is not None and len(act_ckpt_levels) != len(n_channels):
            raise ValueError(
                f"act_ckpt_levels length ({len(act_ckpt_levels)}) must equal "
                f"number of decoder levels ({len(n_channels)})"
            )
        self.act_ckpt_levels = act_ckpt_levels

        if cln_per_level is not None and len(cln_per_level) != len(n_channels):
            raise ValueError(
                f"cln_per_level length ({len(cln_per_level)}) must equal number of decoder levels ({len(n_channels)})"
            )

        if asn_per_level is not None and len(asn_per_level) != len(n_channels):
            raise ValueError(
                f"asn_per_level length ({len(asn_per_level)}) must equal number of decoder levels ({len(n_channels)})"
            )
        if asn_per_level is not None and any(asn_per_level):
            if asn_spatial_sizes is None or len(asn_spatial_sizes) != len(n_channels):
                raise ValueError(
                    "asn_spatial_sizes must be provided and have length len(n_channels) when any asn_per_level is True"
                )

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
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            if cln_per_level is not None and not cln_per_level[n]:
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
                    enable_healpixpad=enable_healpixpad,
                    conditional_layer_norm=None,
                    hpx_padding_mode=hpx_padding_mode,
                )
            else:
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
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )

            # Recurrent module
            if recurrent_block is not None:
                rec_module = instantiate(
                    config=recurrent_block,
                    in_channels=next_channel,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            else:
                rec_module = None

            layer_dict = {
                "upsamp": up_sample_module,
                "conv": conv_module,
                "recurrent": rec_module,
            }
            # Pre-skip ASN / SDL (after upsample, before skip concat). No CLN; S from Z via MLP.
            if asn_per_level is not None and n > 0 and asn_per_level[n]:
                mlp_dims = (
                    list(asn_mlp_hidden_dims)
                    if asn_mlp_hidden_dims is not None
                    else None
                )
                H_lvl, W_lvl = asn_spatial_sizes[n]
                asn_module = AdditiveSpatialNoise(
                    channel_depth=curr_channel,
                    noise_dim=asn_noise_dim,
                    spatial_size=(H_lvl, W_lvl),
                    alpha=asn_alpha,
                    mlp_hidden_dims=mlp_dims,
                    num_faces=asn_num_faces,
                )
                layer_dict["pre_skip"] = th.nn.ModuleDict({"asn": asn_module})
            self.decoder.append(th.nn.ModuleDict(layer_dict))

        self.decoder = th.nn.ModuleList(self.decoder)
        # (Linear) Output layer
        self.output_layer = instantiate(
            config=output_layer,
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            hpx_padding_mode=hpx_padding_mode,
        )

    def _forward_level(self, layer, x, skip, conditions_cln):
        """Run one decoder level (upsample + conv), excluding recurrent."""
        if layer["upsamp"] is not None:
            x = layer["upsamp"](x)
            if "pre_skip" in layer:
                x = layer["pre_skip"]["asn"](x)
            x = th.cat([x, skip], dim=self.channel_dim)
        if hasattr(layer["conv"], "cln_enabled") and layer["conv"].cln_enabled:
            if conditions_cln is None:
                raise ValueError("Conditional inputs are required for layers with cln_enabled=True")
            x = layer["conv"](x, conditions_cln=conditions_cln)
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
            skip = inputs[-1 - n] if layer["upsamp"] is not None else None
            if self.act_ckpt_levels is not None and self.act_ckpt_levels[n]:
                x = ckpt_fn(
                    self._forward_level, layer, x, skip, conditions_cln,
                    use_reentrant=False,
                )
            else:
                x = self._forward_level(layer, x, skip, conditions_cln)
            if layer["recurrent"] is not None:
                x = layer["recurrent"](x)
        return self.output_layer(x)

    def reset(self):
        """Resets the state of the decoder layers"""
        for layer in self.decoder:
            if layer["recurrent"] is not None:
                layer["recurrent"].reset()
