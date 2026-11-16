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

from dataclasses import dataclass

import torch as th
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo.models.dlwp_healpix_layers import HEALPixFoldFaces, HEALPixUnfoldFaces
from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
    warn_deprecated_enable_healpixpad,
)
from physicsnemo.models.meta import ModelMetaData

from .HEALPixUNet import HEALPixUNet


@dataclass
class MetaData(ModelMetaData):
    """Metadata for the DLWP HEALPix ResNet Model"""

    name: str = "DLWP_HEALPixResNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class HEALPixResNet(HEALPixUNet):
    """Deep Learning Weather Prediction (DLWP) ResNet on the HEALPix mesh.

    A stack of convolutional blocks at full resolution (no encoder/decoder, no
    downsampling) followed by a linear output layer. It is a HEALPixUNet with the
    encoder/decoder replaced, so input assembly, the autoregressive rollout,
    residual prediction, constraints and the channel bookkeeping all come from
    :class:`HEALPixUNet`.
    """

    def __init__(
        self,
        n_channels: list[int],
        dilations: list[int],
        conv_block: DictConfig,
        output_layer: DictConfig,
        input_channels: int,
        output_channels: int,
        n_constants: int,
        decoder_input_channels: int,
        input_time_dim: int,
        output_time_dim: int,
        enable_nhwc: bool = False,
        couplings: list = [],
        constraints: list[DictConfig] = None,
        hpx_padding_mode: str | None = None,
        compile_padding: bool = False,
        nside: int | None = None,
        enable_healpixpad: bool | None = None,
    ):
        """
        Parameters
        ----------
        n_channels: list[int]
            number of channels in each convolutional block
        dilations: list[int]
            dilation factor for each block. Defaults to 1 for every block if None.
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional blocks
        output_layer: DictConfig
            dictionary of instantiable parameters for the (linear) output layer
        input_channels, output_channels, n_constants, decoder_input_channels,
        input_time_dim, output_time_dim, enable_nhwc, couplings, constraints:
            same meaning as in :class:`HEALPixUNet`. The model always predicts the
            residual of the prognostic channels, and coupled inputs are always in
            time-first format.
        hpx_padding_mode: str, optional
            Padding strategy: ``earth2grid``, ``karlbauer``, or ``isolatitude``;
            see :class:`HEALPixUNet`.
        compile_padding: bool, optional
            If True, apply torch compile to the padding module.
        nside: int, optional
            Face height/width. Only required for ``isolatitude`` padding. The ResNet
            runs at a single resolution, so this is one int rather than a list.
        enable_healpixpad: bool, optional
            Deprecated; see ``hpx_padding_mode``.
        """
        # Skip HEALPixUNet.__init__ (it builds an encoder/decoder); initialize
        # the torch/physicsnemo Module base directly and set the attributes the
        # inherited HEALPixUNet methods rely on.
        super(HEALPixUNet, self).__init__()
        hpx_padding_mode = warn_deprecated_enable_healpixpad(
            enable_healpixpad, hpx_padding_mode
        )

        if n_constants == 0 and decoder_input_channels == 0:
            raise NotImplementedError(
                "support for models with no constant fields and no decoder inputs (TOA insolation) is not available at this time."
            )
        if len(couplings) > 0:
            if n_constants == 0:
                raise NotImplementedError(
                    "support for coupled models with no constant fields is not available at this time."
                )
            if decoder_input_channels == 0:
                raise NotImplementedError(
                    "support for coupled models with no decoder inputs (TOA insolation) is not available at this time."
                )
        if dilations is not None and len(dilations) != len(n_channels):
            raise ValueError (
                "Number of dilations and number of layers must match"
                f", got {len(dilations)} dilations and {len(n_channels)} channels"
            )

        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.train_couplers = None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 2  # [B, F, C*T, H, W]
        self.enable_nhwc = enable_nhwc
        # fixed for the ResNet; read by the inherited HEALPixUNet methods
        self.residual_prediction = True
        self.couplings_time_first = True
        self.hpx_padding_mode = hpx_padding_mode
        self.compile_padding = compile_padding
        self.nside = nside

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        self.fold = HEALPixFoldFaces(enable_nhwc=self.enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=12)

        if dilations is None:
            dilations = [1 for _ in range(len(n_channels))]
        padding_kwargs = dict(
            enable_nhwc=self.enable_nhwc,
            hpx_padding_mode=self.hpx_padding_mode,
            compile_padding=self.compile_padding,
            nside=self.nside,
        )

        # Conv blocks, then a linear output layer
        layers = []
        in_channels = self._compute_input_channels()
        for n_ch, dilation in zip(n_channels, dilations):
            layers.append(
                instantiate(
                    config=conv_block,
                    in_channels=in_channels,
                    latent_channels=n_ch,
                    out_channels=n_ch,
                    dilation=dilation,
                    **padding_kwargs,
                )
            )
            in_channels = n_ch
        layers.append(
            instantiate(
                config=output_layer,
                in_channels=in_channels,
                out_channels=self._compute_output_channels(),
                dilation=1,
                **padding_kwargs,
            )
        )
        self.resnet = th.nn.Sequential(*layers)

        self.constraints = None
        self.set_constraints(constraints)

    def _apply_network(self, input_tensor: th.Tensor, conditions_cln=None) -> th.Tensor:
        """Run the folded input through the conv blocks and output layer."""
        if conditions_cln is not None:
            raise NotImplementedError(
                "HEALPixResNet does not support conditional normalization (conditions_cln)."
            )
        return self.resnet(input_tensor)
