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

from .healpix_layers import check_for_rng_consumers
from .healpix_paddings import warn_deprecated_enable_healpixpad


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
        spatial_noise_dim: int | None = None,
        spatial_noise_levels: Sequence[int] = None,
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
        spatial_noise_dim: int | None, optional
            Channels of the spatial noise fed to each injected level; None disables injection.
        spatial_noise_levels: Sequence[int], optional
            Encoder levels that receive spatial noise, in the order of ``forward(spatial_noise=...)``.
        """
        super().__init__()
        hpx_padding_mode = warn_deprecated_enable_healpixpad(enable_healpixpad, hpx_padding_mode)
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

        # All layers are identical, so we can just set the flag once
        self.preserve_rng_state = False

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
            block_config = conv_block.copy()
            if "conditional_layer_norm" in block_config and block_config.conditional_layer_norm is not None:
                if not per_level_cln[n]:
                    block_config.conditional_layer_norm = None

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
                )
            )
            old_channels = curr_channel

            self.encoder.append(th.nn.Sequential(*modules))

        # The checkpointed regions use preserve_rng_state=False to avoid a 30% performance penalty
        # (see the helper's docstring). The forward and backward passes are exact only while the
        # region draws no RNG, so check if the region draws RNG and if enable preserve_rng_state accordingly.
        for n, enabled in enumerate(self.per_level_checkpointing):
            if enabled:
                # all layers are identical, so we check the result from the last layer for simplicity
                self.preserve_rng_state = check_for_rng_consumers(
                    self.encoder[n], f"UNetEncoder level {n}"
                )

        self.encoder = th.nn.ModuleList(self.encoder)

        # Zero-initialised 1x1 conv per noise level: adds learned-amplitude spatial noise to that
        # level's features and is a no-op until trained.
        self.spatial_noise_dim = spatial_noise_dim
        self.spatial_noise_levels = list(spatial_noise_levels) if spatial_noise_levels else []
        if self.spatial_noise_dim is not None and self.spatial_noise_levels:
            self.spatial_noise_injectors = th.nn.ModuleDict()
            for lvl in self.spatial_noise_levels:
                if lvl < 0 or lvl >= len(n_channels):
                    raise ValueError(f"spatial_noise_levels entry {lvl} out of range [0, {len(n_channels)})")
                inj = th.nn.Conv2d(self.spatial_noise_dim, n_channels[lvl], kernel_size=1, bias=True)
                th.nn.init.zeros_(inj.weight)
                th.nn.init.zeros_(inj.bias)
                self.spatial_noise_injectors[str(lvl)] = inj
        else:
            self.spatial_noise_injectors = None


    def _forward_layer_pass(self, layer_group: th.nn.Module, inp: th.Tensor, conditions_cln: th.Tensor=None) -> th.Tensor:
        """
        Forward pass of single layer of the encoder
        Handled seperately to allow for checkpointing of the layer

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
        interim_output = inp
        for layer in layer_group:
            # check if class accepts cln inputs
            if getattr(layer, 'cln_enabled', False):
                if conditions_cln is None:
                    raise ValueError("Conditional inputs are required for layers with cln_enabled=True")
                interim_output = layer(interim_output, conditions_cln=conditions_cln)
            else:
                interim_output = layer(interim_output)
            
        # Return the outputs of the last layer
        return interim_output

    def forward(self, inputs: Sequence, conditions_cln: th.Tensor=None, spatial_noise: Sequence[th.Tensor]=None) -> Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to encode
        conditions_cln: th.Tensor: optional
            The conditional inputs for the normalization layers.
        spatial_noise: Sequence[th.Tensor]: optional
            One pre-drawn noise tensor per level that has an injector, shape
            ``(B*n_cond, noise_dim, F, H_level, W_level)``. Positions in the sequence
            must align with ``spatial_noise_levels`` passed to ``__init__``. Each
            tensor is folded to ``(B*n_cond*F, noise_dim, H_level, W_level)`` and
            passed through the per-level 1x1 conv injector, then added to that
            level's output feature map. None (default) skips injection entirely.

        Returns
        -------
        Sequence: The encoded values
        """
        emit_nvtx = not th.compiler.is_compiling()
        if emit_nvtx:
            th.cuda.nvtx.range_push("dlesym_encoder")
        outputs = []
        noise_pos = {lvl: i for i, lvl in enumerate(self.spatial_noise_levels)} if self.spatial_noise_injectors is not None else {}
        for n, layer_group in enumerate(self.encoder):
            if emit_nvtx:
                th.cuda.nvtx.range_push("dlesym_encoder:layer")
            interim_output = inputs
            if self.per_level_checkpointing[n]:
                # Disable reentrancy and rng preservation conditionally to allow for cuda graph capture with checkpointing
                interim_output = checkpoint(
                    self._forward_layer_pass,
                    layer_group,
                    interim_output,
                    conditions_cln,
                    use_reentrant=False,
                    preserve_rng_state=self.preserve_rng_state
                )
            else:
                interim_output = self._forward_layer_pass(layer_group, interim_output, conditions_cln)
            if spatial_noise is not None and n in noise_pos:
                z = spatial_noise[noise_pos[n]]
                bsz, ndim, F, H, W = z.shape
                z_folded = z.permute(0, 2, 1, 3, 4).reshape(bsz * F, ndim, H, W).to(interim_output.dtype)
                interim_output = interim_output + self.spatial_noise_injectors[str(n)](z_folded)
            outputs.append(interim_output)
            inputs = outputs[-1]
            if emit_nvtx:
                th.cuda.nvtx.range_pop() # dlesym_encoder:layer
        # Return the outputs of the last layer
        if emit_nvtx:
            th.cuda.nvtx.range_pop()
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass
