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

from typing import Sequence, Tuple, Union, Callable

import torch
import torch as th
from .healpix_layers import HEALPixLayer
from .normalization import ConditionalLayerNorm, AdaLNZero

from hydra.utils import instantiate
from omegaconf import DictConfig

#
# Helper: standard LayerNorm over channel dimension for (B, C, H, W)
#


class _LayerNormOverChannels(th.nn.Module):
    """Applies nn.LayerNorm over the channel dimension for (B, C, H, W) tensors."""

    def __init__(self, channel_depth: int, eps: float = 1e-5):
        super().__init__()
        self.norm = th.nn.LayerNorm(channel_depth, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


#
# RECURRENT BLOCKS
#


class ConvGRUBlock(th.nn.Module):
    """Class that implements a Convolutional GRU
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        kernel_size: int = 1,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        kernel_size: int, optional
            Size of the convolutioonal kernel
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        self.channels = in_channels
        self.conv_gates = geometry_layer(
            layer=torch.nn.Conv2d,
            in_channels=in_channels + self.channels,
            out_channels=2 * self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            hpx_padding_mode=hpx_padding_mode,
        )
        self.conv_can = geometry_layer(
            layer=torch.nn.Conv2d,
            in_channels=in_channels + self.channels,
            out_channels=self.channels,  # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            hpx_padding_mode=hpx_padding_mode,
        )
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs: Sequence) -> Sequence:
        """Forward pass of the ConvGRUBlock

        Parameters
        ----------
        inputs: Sequence
            Input to the forward pass

        Returns
        -------
        Sequence
            Result of the forward pass
        """
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        combined = th.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = th.split(combined_conv, self.channels, dim=1)
        reset_gate = th.sigmoid(gamma)
        update_gate = th.sigmoid(beta)

        combined = th.cat([inputs, reset_gate * self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = th.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        """Reset the update gates"""
        self.h = th.zeros_like(self.h)


#
# CONV BLOCKS
#


class BasicConvBlock(th.nn.Module):
    """Convolution block consisting of n subsequent convolutions and activations"""

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        latent_channels: int = None,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernel
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        n_layers:
            Number of convolutional layers
        latent_channels:
            Number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(
                geometry_layer(
                    layer=torch.nn.Conv2d,
                    in_channels=in_channels if n == 0 else latent_channels,
                    out_channels=out_channels if n == n_layers - 1 else latent_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if activation is not None:
                convblock.append(activation)
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the BasicConvBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        return self.convblock(x)


class ConvNeXtBlock(th.nn.Module):
    """Class implementing a modified ConvNeXt network as described in https://arxiv.org/pdf/2201.03545.pdf
    and shown in figure 4
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        latent_channels: int, optional
            Number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        # Convolution block
        convblock = []
        # 3x3 convolution increasing channels
        convblock.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # 3x3 convolution maintaining increased channels
        convblock.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # Linear postprocessing
        convblock.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the ConvNextBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        return self.skip_module(x) + self.convblock(x)


class DoubleConvNeXtBlock(th.nn.Module):
    """Modification of ConvNeXtBlock block this time putting two sequentially
    in a single block with the number of channels in the middle being the
    number of latent channels
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        latent_channels: int = 1,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters:
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        latent_channels: int, optional
            Number of latent channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        if in_channels == int(latent_channels):
            self.skip_module1 = (
                lambda x: x
            )  # Identity-function required in forward pass
        else:
            self.skip_module1 = geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        if out_channels == int(latent_channels):
            self.skip_module2 = (
                lambda x: x
            )  # Identity-function required in forward pass
        else:
            self.skip_module2 = geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )

        # 1st ConvNeXt block, the output of this one remains internal
        convblock1 = []
        # 3x3 convolution establishing latent channels channels
        convblock1.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            convblock1.append(activation)
        # 1x1 convolution establishing increased channels
        convblock1.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            convblock1.append(activation)
        # 1x1 convolution returning to latent channels
        convblock1.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            convblock1.append(activation)
        self.convblock1 = th.nn.Sequential(*convblock1)

        # 2nd ConNeXt block, takes the output of the first convnext block
        convblock2 = []
        # 3x3 convolution establishing latent channels channels
        convblock2.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            convblock2.append(activation)
        # 1x1 convolution establishing increased channels
        convblock2.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            convblock2.append(activation)
        # 1x1 convolution reducing to output channels
        convblock2.append(
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            convblock2.append(activation)
        self.convblock2 = th.nn.Sequential(*convblock2)

    def forward(self, x):
        """Forward pass of the DoubleConvNextBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        # internal convnext result
        x1 = self.skip_module1(x) + self.convblock1(x)
        # return second convnext result
        return self.skip_module2(x1) + self.convblock2(x1)

class Multi_SymmetricConvNeXtBlock(th.nn.Module):
    """
    Wrapper for SymmetricConvNeXtBlock that allows serial linking of blocks. 
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        n_layers: int = 1,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
        conditional_layer_norm: Callable = None,
        cln_once_per_block: bool = False,
        use_initial_one_conv: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        n_layers: int, optional
            The number of SymmetricConvNeXt Blocks
        conditional_layer_norm: Callable, optional
            Callable for physicsnemo.models.dlwp_healpix_layers.normalization.ConditionalLayerNorm. 
            Callable can be passed in by setting _partial_ to True in hydra config. If None,
            conditional layer normalization is not applied.
        cln_once_per_block: bool, optional
            If True, use AdaLN-style (CLN once at block entry, LayerNorm in middle).
            If False (default), keep current structure (3 CLN positions). Backward compatible.
        use_initial_one_conv: bool, optional
            If True, each block inserts an optional 1x1 conv before the first 3x3 conv
            (independent of cln_once_per_block).
        """
        super().__init__()

        # Create a ModuleList to store complete blocks
        self.blocks = th.nn.ModuleList()
        # flag for conditional layer normalization
        self.cln_enabled = conditional_layer_norm is not None

        for i in range(n_layers):
            curr_in = in_channels if i == 0 else out_channels
            self.blocks.append(
                SymmetricConvNeXtBlock(
                    geometry_layer=geometry_layer,
                    in_channels=curr_in,
                    latent_channels=latent_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    upscale_factor=upscale_factor,
                    activation=activation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    conditional_layer_norm=conditional_layer_norm if conditional_layer_norm is not None else None,
                    cln_once_per_block=cln_once_per_block,
                    use_initial_one_conv=use_initial_one_conv,
                    hpx_padding_mode=hpx_padding_mode,
                ),
            )

    def forward(self, x, conditions_cln=None):
        out = x
        for block in self.blocks:
            out = block(out, conditions_cln=conditions_cln)
        return out


class InceptionDWConv2d(th.nn.Module):
    """
    Inception-style depthwise spatial mixer for HEALPix data.

    Splits channels into four groups:
      - Group 1: square_kernel_size x square_kernel_size convolution
      - Group 2: 1 x band_kernel_size convolution (zonal / horizontal)
      - Group 3: band_kernel_size x 1 convolution (meridional / vertical)
      - Group 4: identity passthrough

    All internal convolutions are wrapped by the provided geometry_layer.
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        dilation: int = 1,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        InceptionDWConv2d: Efficient depthwise convolution with channel splitting.
        
        Note: band_kernel_size should be chosen carefully. With HEALPixPadding, the padding
        is calculated as (band_kernel_size - 1) // 2. This padding must be <= spatial dimensions
        in all layers. For networks with downsampling, use smaller band_kernel_size (e.g., 7 or 5)
        to avoid padding errors in deeper layers with smaller feature maps.
        """
        super().__init__()

        self.in_channels = in_channels

        # Compute channel splits: roughly quartered, identity branch takes the remainder.
        g = max(in_channels // 4, 0)
        c1 = g
        c2 = g
        c3 = g
        c4 = in_channels - (c1 + c2 + c3)
        self.splits = (c1, c2, c3, c4)

        # Some branches may be empty for very small in_channels; guard accordingly.
        convs = {}
        if c1 > 0:
            convs["square"] = geometry_layer(
                layer=th.nn.Conv2d,
                in_channels=c1,
                out_channels=c1,
                kernel_size=square_kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        if c2 > 0:
            convs["horiz"] = geometry_layer(
                layer=th.nn.Conv2d,
                in_channels=c2,
                out_channels=c2,
                kernel_size=(1, band_kernel_size),
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        if c3 > 0:
            convs["vert"] = geometry_layer(
                layer=th.nn.Conv2d,
                in_channels=c3,
                out_channels=c3,
                kernel_size=(band_kernel_size, 1),
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )

        self.square_conv = convs.get("square", None)
        self.horiz_conv = convs.get("horiz", None)
        self.vert_conv = convs.get("vert", None)

    def forward(self, x: th.Tensor) -> th.Tensor:
        c1, c2, c3, c4 = self.splits

        # Fast path: no splitting needed
        if c1 == 0 and c2 == 0 and c3 == 0:
            return x

        x_chunks = th.split(x, [c for c in self.splits if c > 0], dim=1)

        idx = 0
        y_chunks = []
        
        # Get target spatial dimensions from input (or from square conv output)
        target_h, target_w = x.shape[-2], x.shape[-1]

        # Group 1: square conv
        if c1 > 0:
            x1 = x_chunks[idx]
            idx += 1
            y1 = self.square_conv(x1) if self.square_conv is not None else x1
            # Update target dimensions from square conv output (it should preserve spatial dims)
            target_h, target_w = y1.shape[-2], y1.shape[-1]
            y_chunks.append(y1)

        # Group 2: horizontal band (1, band_kernel_size)
        # Uniform padding pads both H and W, but conv only reduces W dimension
        # So we need to crop the H dimension back to target_h
        if c2 > 0:
            x2 = x_chunks[idx]
            idx += 1
            y2 = self.horiz_conv(x2) if self.horiz_conv is not None else x2
            # Crop H dimension if needed (remove extra padding from uniform padding)
            if y2.shape[-2] != target_h:
                pad_h = (y2.shape[-2] - target_h) // 2
                y2 = y2[..., pad_h:pad_h + target_h, :]
            y_chunks.append(y2)

        # Group 3: vertical band (band_kernel_size, 1)
        # Uniform padding pads both H and W, but conv only reduces H dimension
        # So we need to crop the W dimension back to target_w
        if c3 > 0:
            x3 = x_chunks[idx]
            idx += 1
            y3 = self.vert_conv(x3) if self.vert_conv is not None else x3
            # Crop W dimension if needed (remove extra padding from uniform padding)
            if y3.shape[-1] != target_w:
                pad_w = (y3.shape[-1] - target_w) // 2
                y3 = y3[..., :, pad_w:pad_w + target_w]
            y_chunks.append(y3)

        # Group 4: identity
        if c4 > 0:
            x4 = x_chunks[idx]
            y_chunks.append(x4)

        return th.cat(y_chunks, dim=1)


class SymmetricConvNeXtBlock(th.nn.Module):
    """Another modification of ConvNeXtBlock block this time using 4 layers and adding
    a layer that instead of going from in_channels to latent*upscale channesl goes to
    latent channels first
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        use_block_skip_connection: bool = True,
        enable_healpixpad: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
        conditional_layer_norm: th.nn.Module = None,
        cln_once_per_block: bool = False,
        use_initial_one_conv: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        latent_channels: int, optional
            Number of latent channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        use_block_skip_connection: bool, optional
            Whether or not to use block-level skip connection
        batch_norm: bool, optional
            Whether or not to use batch normalization after the first convolution
        dropout: float, optional
            Dropout probability to apply after the first convolution
        conditional_layer_norm: th.nn.Module, optional
            conditional layer normalization. If None,
            no conditional layer normalization is applied.
        cln_once_per_block: bool, optional
            If True, AdaLN-style: one CLN at block entry, LayerNorm in middle.
            If False (default), current structure (3 CLN positions). Backward compatible.
        use_initial_one_conv: bool, optional
            If True, insert a 1x1 conv (in_channels → in_channels) before the first 3x3 conv.
            When cln_once_per_block is True this is after entry norm; otherwise it is the first
            layer in the main path. Independent of cln_once_per_block. Default False (current behavior).
        """

        super().__init__()

        self.use_block_skip_connection = use_block_skip_connection
        self.activation = activation
        self.dropout = dropout > 0.0
        self.cln_enabled = conditional_layer_norm is not None
        self.cln_once_per_block = cln_once_per_block
        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_block_skip_connection:
            if in_channels == int(out_channels):
                self.skip_module = lambda x: x
            else:
                self.skip_module = geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )

        # Collect conv->norm->activation->dropout operations in list for sequential execution
        convblock = []

        if cln_once_per_block:
            # AdaLN-style: one norm at block entry (CLN or LayerNorm), then conv chain with LayerNorm in middle only
            if conditional_layer_norm is not None:
                # For AdaLNZero, gate should match output channels while normalization matches input channels
                import functools
                is_adaln_zero = False
                if isinstance(conditional_layer_norm, functools.partial):
                    # Check if the partial's function is AdaLNZero
                    is_adaln_zero = conditional_layer_norm.func is AdaLNZero
                
                if is_adaln_zero and in_channels != out_channels:
                    # Create AdaLNZero with separate gate channel depth
                    new_kwargs = dict(conditional_layer_norm.keywords) if isinstance(conditional_layer_norm, functools.partial) else {}
                    new_kwargs['channel_depth'] = in_channels
                    new_kwargs['gate_channel_depth'] = out_channels
                    if isinstance(conditional_layer_norm, functools.partial):
                        self.entry_norm = conditional_layer_norm.func(**new_kwargs)
                    else:
                        self.entry_norm = conditional_layer_norm(channel_depth=in_channels, gate_channel_depth=out_channels)
                else:
                    self.entry_norm = conditional_layer_norm(channel_depth=in_channels)
            else:
                self.entry_norm = _LayerNormOverChannels(in_channels)
            if use_initial_one_conv:
                convblock.append(
                    geometry_layer(
                        layer=th.nn.Conv2d,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=1,
                        dilation=dilation,
                        enable_nhwc=enable_nhwc,
                        enable_healpixpad=enable_healpixpad,
                        hpx_padding_mode=hpx_padding_mode,
                    )
                )
            # Conv (3x3 in → latent), Act, [Dropout]
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=in_channels,
                    out_channels=int(latent_channels),
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))
            # Conv (1x1 latent → upscale), LayerNorm, Act, [Dropout]
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels),
                    out_channels=int(latent_channels * upscale_factor),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            convblock.append(_LayerNormOverChannels(int(latent_channels * upscale_factor)))
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))
            # Conv (1x1 upscale → latent), LayerNorm, Act, [Dropout]
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels * upscale_factor),
                    out_channels=int(latent_channels),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            convblock.append(_LayerNormOverChannels(int(latent_channels)))
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))
            # Conv (3x3 latent → out), Act, [Dropout]
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels),
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))
            self.convblock = th.nn.ModuleList(convblock)
        else:
            # Current structure: Conv -> [BN] -> [CLN] -> Act -> ... (3 CLN positions when conditional_layer_norm set)
            if use_initial_one_conv:
                convblock.append(
                    geometry_layer(
                        layer=th.nn.Conv2d,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=1,
                        dilation=dilation,
                        enable_nhwc=enable_nhwc,
                        enable_healpixpad=enable_healpixpad,
                        hpx_padding_mode=hpx_padding_mode,
                    )
                )
            # 3x3: in → latent
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=in_channels,
                    out_channels=int(latent_channels),
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if batch_norm:
                convblock.append(th.nn.BatchNorm2d(int(latent_channels), track_running_stats=False, affine=False))
            # Conditional Layer Normalization
            if conditional_layer_norm is not None:
                # resolve context-dependent parameters (channel depth)
                cln = conditional_layer_norm(channel_depth=int(latent_channels))
                convblock.append(cln)
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # 1x1: latent → latent * upscale
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels),
                    out_channels=int(latent_channels * upscale_factor),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if batch_norm:
                convblock.append(th.nn.BatchNorm2d(int(latent_channels * upscale_factor), track_running_stats=False, affine=False))
            if conditional_layer_norm is not None:
                cln = conditional_layer_norm(channel_depth=int(latent_channels * upscale_factor))
                convblock.append(cln)
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # 1x1: upscale → latent
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels * upscale_factor),
                    out_channels=int(latent_channels),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if batch_norm:
                convblock.append(th.nn.BatchNorm2d(int(latent_channels), track_running_stats=False, affine=False))
            if conditional_layer_norm is not None:
                cln = conditional_layer_norm(channel_depth=int(latent_channels))
                convblock.append(cln)
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # 3x3: latent → out (no norm on this one, following convnext)
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels),
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            self.convblock = th.nn.ModuleList(convblock)
            self.entry_norm = None  # no entry norm in current structure


    def forward(self, x, conditions_cln=None):
        """
        Forward pass of the SymmetricConvNextBlock, broken into steps for support of conditional layer normalization
        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass
        conditions_cln: torch.Tensor, optional
            Condition for the conditional layer normalization, if applicable
        Returns
        -------
        torch.Tensor
            result of the forward pass
        """

        # Save residual
        residual = self.skip_module(x) if self.use_block_skip_connection else 0
        gate = None

        # AdaLN-style: apply entry norm first (CLN or LayerNorm or AdaLNZero)
        if self.entry_norm is not None:
            if isinstance(self.entry_norm, AdaLNZero):
                if conditions_cln is None:
                    raise ValueError("AdaLNZero requires non-None conditions_cln.")
                x, gate = self.entry_norm(x, conditions=conditions_cln)
            elif isinstance(self.entry_norm, ConditionalLayerNorm):
                x = self.entry_norm(x, conditions=conditions_cln)
            else:
                x = self.entry_norm(x)

        for layer in self.convblock:
            if isinstance(layer, ConditionalLayerNorm):
                x = layer(x, conditions=conditions_cln)
            else:
                x = layer(x)

        if gate is not None:
            x = x * gate

        return x + residual


class InceptionNeXtBlock(th.nn.Module):
    """
    Classical ConvNeXt-style block with InceptionNeXt-inspired depthwise 7x7
    spatial mixing followed by an inverted bottleneck MLP:

        DwConv 7x7 -> LayerNorm -> 1x1 Conv (expand 4x) -> GELU -> 1x1 Conv (shrink)

    Supports optional AdaLN-Zero / CLN applied once at block entry (cln_once_per_block).
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 3,
        n_layers: int = 1,  # unused, for hydra compatibility
        mlp_ratio: int = 4,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        conditional_layer_norm: Callable = None,
        cln_once_per_block: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        super().__init__()

        self.cln_enabled = conditional_layer_norm is not None
        self.cln_once_per_block = cln_once_per_block
        self.activation = activation if activation is not None else th.nn.GELU()

        # Skip / projection
        if in_channels == out_channels:
            self.skip_module = lambda x: x
        else:
            self.skip_module = geometry_layer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )

        hidden_channels = in_channels * mlp_ratio

        # Entry norm: can be AdaLNZero, ConditionalLayerNorm, or simple LayerNorm.
        if cln_once_per_block:
            if conditional_layer_norm is not None:
                # For AdaLNZero, gate should match output channels while normalization matches input channels
                import functools
                is_adaln_zero = False
                if isinstance(conditional_layer_norm, functools.partial):
                    is_adaln_zero = conditional_layer_norm.func is AdaLNZero
                
                if is_adaln_zero and in_channels != out_channels:
                    new_kwargs = dict(conditional_layer_norm.keywords) if isinstance(conditional_layer_norm, functools.partial) else {}
                    new_kwargs['channel_depth'] = in_channels
                    new_kwargs['gate_channel_depth'] = out_channels
                    if isinstance(conditional_layer_norm, functools.partial):
                        self.entry_norm = conditional_layer_norm.func(**new_kwargs)
                    else:
                        self.entry_norm = conditional_layer_norm(channel_depth=in_channels, gate_channel_depth=out_channels)
                else:
                    self.entry_norm = conditional_layer_norm(channel_depth=in_channels)
            else:
                self.entry_norm = _LayerNormOverChannels(in_channels)
        else:
            self.entry_norm = None

        # Core ConvNeXt-style body: DwConv -> LN -> 1x1 expand -> GELU -> 1x1 shrink
        self.dwconv = geometry_layer(
            layer=th.nn.Conv2d,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            groups=in_channels,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            hpx_padding_mode=hpx_padding_mode,
        )
        self.mid_norm = _LayerNormOverChannels(in_channels)
        self.pw_expand = geometry_layer(
            layer=th.nn.Conv2d,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            hpx_padding_mode=hpx_padding_mode,
        )
        self.pw_shrink = geometry_layer(
            layer=th.nn.Conv2d,
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            hpx_padding_mode=hpx_padding_mode,
        )

    def forward(self, x: th.Tensor, conditions_cln: th.Tensor = None) -> th.Tensor:
        residual = self.skip_module(x)
        gate = None

        if self.entry_norm is not None:
            if isinstance(self.entry_norm, AdaLNZero):
                if conditions_cln is None:
                    raise ValueError("AdaLNZero requires non-None conditions_cln.")
                x, gate = self.entry_norm(x, conditions=conditions_cln)
            elif isinstance(self.entry_norm, ConditionalLayerNorm):
                x = self.entry_norm(x, conditions=conditions_cln)
            else:
                x = self.entry_norm(x)

        x = self.dwconv(x)
        x = self.mid_norm(x)
        x = self.pw_expand(x)
        x = self.activation(x)
        x = self.pw_shrink(x)

        if gate is not None:
            x = x * gate

        return residual + x


class SymmetricInceptionNeXtBlock(th.nn.Module):
    """
    Symmetric ConvNeXt-style block that replaces the initial 3x3 spatial mixer
    with an Inception-style depthwise module (InceptionDWConv2d).

    The overall structure mirrors SymmetricConvNeXtBlock:
        [InceptionDW] -> [1x1 expand] -> [1x1 shrink] -> [3x3 out]

    When cln_once_per_block=True, supports AdaLNZero / CLN entry norm plus
    mid-block LayerNorms, and applies AdaLN gating to the block output.
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # unused, hydra compatibility
        upscale_factor: int = 4,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        use_block_skip_connection: bool = True,
        enable_healpixpad: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
        conditional_layer_norm: Callable = None,
        cln_once_per_block: bool = False,
        inception_kernel_size: int = 11,
        hpx_padding_mode: str = "karlbauer",
    ):
        super().__init__()

        self.use_block_skip_connection = use_block_skip_connection
        self.activation = activation
        self.dropout = dropout > 0.0
        self.cln_enabled = conditional_layer_norm is not None
        self.cln_once_per_block = cln_once_per_block

        if use_block_skip_connection:
            if in_channels == int(out_channels):
                self.skip_module = lambda x: x
            else:
                self.skip_module = geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )

        convblock = []

        if cln_once_per_block:
            # Entry norm (AdaLNZero, CLN, or LayerNorm)
            if conditional_layer_norm is not None:
                # For AdaLNZero, gate should match output channels while normalization matches input channels
                import functools
                is_adaln_zero = False
                if isinstance(conditional_layer_norm, functools.partial):
                    is_adaln_zero = conditional_layer_norm.func is AdaLNZero
                
                if is_adaln_zero and in_channels != out_channels:
                    new_kwargs = dict(conditional_layer_norm.keywords) if isinstance(conditional_layer_norm, functools.partial) else {}
                    new_kwargs['channel_depth'] = in_channels
                    new_kwargs['gate_channel_depth'] = out_channels
                    if isinstance(conditional_layer_norm, functools.partial):
                        self.entry_norm = conditional_layer_norm.func(**new_kwargs)
                    else:
                        self.entry_norm = conditional_layer_norm(channel_depth=in_channels, gate_channel_depth=out_channels)
                else:
                    self.entry_norm = conditional_layer_norm(channel_depth=in_channels)
            else:
                self.entry_norm = _LayerNormOverChannels(in_channels)

            # Inception depthwise mixer on input channels
            convblock.append(
                InceptionDWConv2d(
                    geometry_layer=geometry_layer,
                    in_channels=in_channels,
                    square_kernel_size=kernel_size,
                    band_kernel_size=inception_kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            # 1x1 in_channels -> latent_channels
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=in_channels,
                    out_channels=int(latent_channels),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # 1x1 latent -> latent * upscale, then LayerNorm + activation
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels),
                    out_channels=int(latent_channels * upscale_factor),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            convblock.append(_LayerNormOverChannels(int(latent_channels * upscale_factor)))
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # 1x1 upscale -> latent, LayerNorm, activation
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels * upscale_factor),
                    out_channels=int(latent_channels),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            convblock.append(_LayerNormOverChannels(int(latent_channels)))
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # Final 3x3 latent -> out
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels),
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            self.convblock = th.nn.ModuleList(convblock)
        else:
            # Non-AdaLN layout: preserve original SymmetricConvNeXtBlock ordering,
            # simply replacing the first 3x3 conv with InceptionDWConv2d + 1x1 to latent.
            self.entry_norm = None

            # Inception mixer + 1x1 to latent
            convblock.append(
                InceptionDWConv2d(
                    geometry_layer=geometry_layer,
                    in_channels=in_channels,
                    square_kernel_size=kernel_size,
                    band_kernel_size=inception_kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=in_channels,
                    out_channels=int(latent_channels),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if batch_norm:
                convblock.append(
                    th.nn.BatchNorm2d(
                        int(latent_channels),
                        track_running_stats=False,
                        affine=False,
                    )
                )
            if conditional_layer_norm is not None:
                cln = conditional_layer_norm(channel_depth=int(latent_channels))
                convblock.append(cln)
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # 1x1 latent -> latent * upscale
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels),
                    out_channels=int(latent_channels * upscale_factor),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if batch_norm:
                convblock.append(
                    th.nn.BatchNorm2d(
                        int(latent_channels * upscale_factor),
                        track_running_stats=False,
                        affine=False,
                    )
                )
            if conditional_layer_norm is not None:
                cln = conditional_layer_norm(channel_depth=int(latent_channels * upscale_factor))
                convblock.append(cln)
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # 1x1 upscale -> latent
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels * upscale_factor),
                    out_channels=int(latent_channels),
                    kernel_size=1,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if batch_norm:
                convblock.append(
                    th.nn.BatchNorm2d(
                        int(latent_channels),
                        track_running_stats=False,
                        affine=False,
                    )
                )
            if conditional_layer_norm is not None:
                cln = conditional_layer_norm(channel_depth=int(latent_channels))
                convblock.append(cln)
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            # 3x3 latent -> out
            convblock.append(
                geometry_layer(
                    layer=th.nn.Conv2d,
                    in_channels=int(latent_channels),
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )
            if activation is not None:
                convblock.append(activation)
            if dropout > 0.0:
                convblock.append(th.nn.Dropout2d(p=dropout))

            self.convblock = th.nn.ModuleList(convblock)

    def forward(self, x: th.Tensor, conditions_cln: th.Tensor = None) -> th.Tensor:
        residual = self.skip_module(x) if self.use_block_skip_connection else 0
        gate = None

        if self.entry_norm is not None:
            if isinstance(self.entry_norm, AdaLNZero):
                if conditions_cln is None:
                    raise ValueError("AdaLNZero requires non-None conditions_cln.")
                x, gate = self.entry_norm(x, conditions=conditions_cln)
            elif isinstance(self.entry_norm, ConditionalLayerNorm):
                x = self.entry_norm(x, conditions=conditions_cln)
            else:
                x = self.entry_norm(x)

        for layer in self.convblock:
            if isinstance(layer, ConditionalLayerNorm):
                x = layer(x, conditions=conditions_cln)
            else:
                x = layer(x)

        if gate is not None:
            x = x * gate

        return residual + x


class Multi_SymmetricInceptionNeXtBlock(th.nn.Module):
    """
    Wrapper for SymmetricInceptionNeXtBlock that allows serial linking of blocks,
    mirroring Multi_SymmetricConvNeXtBlock.
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        n_layers: int = 1,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
        conditional_layer_norm: Callable = None,
        cln_once_per_block: bool = False,
        inception_kernel_size: int = 11,
        hpx_padding_mode: str = "karlbauer",
    ):
        super().__init__()

        self.blocks = th.nn.ModuleList()
        self.cln_enabled = conditional_layer_norm is not None

        for i in range(n_layers):
            curr_in = in_channels if i == 0 else out_channels
            self.blocks.append(
                SymmetricInceptionNeXtBlock(
                    geometry_layer=geometry_layer,
                    in_channels=curr_in,
                    latent_channels=latent_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    upscale_factor=upscale_factor,
                    activation=activation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    conditional_layer_norm=conditional_layer_norm if conditional_layer_norm is not None else None,
                    cln_once_per_block=cln_once_per_block,
                    inception_kernel_size=inception_kernel_size,
                    hpx_padding_mode=hpx_padding_mode,
                )
            )

    def forward(self, x: th.Tensor, conditions_cln: th.Tensor = None) -> th.Tensor:
        out = x
        for block in self.blocks:
            out = block(out, conditions_cln=conditions_cln)
        return out


#
# DOWNSAMPLING BLOCKS
#


class MaxPool(th.nn.Module):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the torch.nn.MaxPool2d class.
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2d
        pooling: int, optional
            Pooling kernel size passed to geometry layer
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        self.maxpool = geometry_layer(
            layer=torch.nn.MaxPool2d,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            hpx_padding_mode=hpx_padding_mode,
        )

    def forward(self, x):
        """Forward pass of the MaxPool

        Parameters
        ----------
        x: torch.Tensor
            The values to MaxPool

        Returns
        -------
        torch.Tensor
            The MaxPooled values
        """
        return self.maxpool(x)


class AvgPool(th.nn.Module):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the torch.nn.AvgPool2d class.
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2d
        pooling: int, optional
            Pooling kernel size passed to geometry layer
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        self.avgpool = geometry_layer(
            layer=torch.nn.AvgPool2d,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
            hpx_padding_mode=hpx_padding_mode,
        )

    def forward(self, x):
        """Forward pass of the AvgPool layer

        Parameters
        ----------
        x: torch.Tensor
            The values to average

        Returns
        -------
        torch.Tensor
            The averaged values
        """
        return self.avgpool(x)


#
# UPSAMPLING BLOCKS
#


class TransposedConvUpsample(th.nn.Module):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the torch.nn.ConvTranspose2d class.
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: th.nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2d
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        upsampling: int, optional
            Size used for upsampling
        activation: torch.nn.Module, optional
            Activation function used in upsampling
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(
            geometry_layer(
                layer=torch.nn.ConvTranspose2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        )
        if activation is not None:
            upsampler.append(activation)
        self.upsampler = th.nn.Sequential(*upsampler)

    def forward(self, x):
        """Forward pass of the TransposedConvUpsample layer

        Parameters
        ----------
        x: torch.Tensor
            The values to upsample

        Returns
        -------
        torch.Tensor
            The upsampled values
        """
        return self.upsampler(x)


class SmoothedInterpolateConv(th.nn.Module):
    """
    Class for sequentially interpolating, applying a smoothing filter which
    preserves zonally uniform signals, then applying a simple Conv2d on
    HEALPix tensor data
    """

    def __init__(
        self,
        geometry_layer: th.nn.Module = HEALPixLayer,
        in_channels = 3,
        out_channels = 3,
        kernel_size = 3,
        dilation = 1,
        scale_factor = 2,
        mode = 'nearest',
        activation: th.nn.Module = None,
        enable_nhwc = False,
        enable_healpixpad = True,
        hpx_padding_mode: str = "karlbauer",
    ):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry of the tensor being bassed to this module
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutional kernel
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        scale_factor: int, optional
            Multiplier for spatial size, passed to torch.nn.functional.interpolate
        mode: str, optional
            Algorithm used for upsampling, passed to torch.nn.functional.interpolate
        activation: torch.nn.Module, optional
            Activation function used in upsampling
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        if dilation > 1:
            raise Exception(
                f"dilation > 1 is not currently supported for hpx resize \
                convolutions, received dilation = {dilation}"
            )

        # We pad first before upsampling to prevent edge artifacts at seams
        # between HPX faces. This means that our final upsampled signal will
        # have extra padding which we need to trim before passing to conv. We
        # only require padding=1 before upsampling, so only need to trim 1 row/
        # column from each side of result.
        trim_size = 1 

        block = []
        block += [
            geometry_layer(
                layer=SmoothedInterpolate,
                in_channels=in_channels,
                scale_factor=scale_factor,
                mode=mode,
                trim_size=trim_size,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            ),
            geometry_layer(
                layer=torch.nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                hpx_padding_mode=hpx_padding_mode,
            )
        ]

        if activation is not None:
            block.append(activation)
        self.block = th.nn.Sequential(*block)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass of the ResizeConv layer

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        x = self.block(x)
        return x


#
# Helper classes
#


class Interpolate(th.nn.Module):
    """Helper class that handles interpolation
    This is done as a class so that scale and mode can be stored
    """

    def __init__(self, scale_factor: Union[int, Tuple], mode: str = "nearest"):
        """
        Parameters:
        ----------
        scale_factor: Union[int , Tuple]
            Multiplier for spatial size, passed to torch.nn.functional.interpolate
        mode: str, optional
            Interpolation mode used for upsampling, passed to torch.nn.functional.interpolate
        """
        super().__init__()
        self.interp = th.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs):
        """Forward pass of the Interpolate layer

        Parameters
        ----------
        x: torch.Tensor
            inputs to interpolate

        Returns
        -------
        torch.Tensor
            the interpolated values
        """
        return self.interp(inputs, scale_factor=self.scale_factor, mode=self.mode)


class SmoothedInterpolate(th.nn.Module):
    """
    Helper class for interpolating a HEALPix signal then applying a four point
    smoother which preserves zonal uniformity if the upsampling mode is nearest
    neighbor or bilinear.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        scale_factor: int = 2,
        mode: str = 'nearest',
        trim_size: int = 0,
    ):
        """
        Parameters
        ----------
        in_channels: int, optional
            The number of input channels
        scale_factor: int, optional
            Multiplier for spatial size, passed to torch.nn.functional.interpolate
        mode: str, optional
            Algorithm used for upsampling, passed to torch.nn.functional.interpolate
        trim_size: int, optional
            Amount of padding to trim from final tensor, which is assumed to be
            square
        """
        super().__init__()

        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.mode = mode
        self.trim_size = trim_size
        self.interp = th.nn.functional.interpolate

        # Four point smoother specific to HPX grid. This smooths out the specific
        # type of aliasing that nearest neighbor and bilinear upsampling introduce
        # into zonally uniform signals
        self.smoother_kernel = torch.tensor(
            [[0.,1.,0.],
             [1.,0.,1.],
             [0.,1.,0.]]
        )
        self.smoother_kernel = self.smoother_kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,3,3)
        self.smoother_kernel = self.smoother_kernel.repeat((in_channels,1,1,1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        self.smoother_kernel = self.smoother_kernel.to(device=x.device, dtype=x.dtype)

        # Interpolate, smooth, trim in order
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)

        x = torch.nn.functional.conv2d(
            x,
            self.smoother_kernel,
            padding=0,
            groups=self.in_channels
        ) / 4 # divide by 4 to take average of 4 neighbors

        if self.trim_size > 0:
            x = x[..., self.trim_size:-self.trim_size, self.trim_size:-self.trim_size]

        return x


