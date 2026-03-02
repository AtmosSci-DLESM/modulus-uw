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

import logging
from dataclasses import dataclass
from typing import Sequence

import torch as th
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo.models.dlwp_healpix_layers import HEALPixFoldFaces, HEALPixUnfoldFaces
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

logger = logging.getLogger(__name__)


@dataclass
class MetaData(ModelMetaData):
    """Metadata for the DLWP HEALPix UNet Model"""

    name: str = "DLWP_HEALPixUNet"
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


class HEALPixUNet(Module):
    """Deep Learning Weather Prediction (DLWP) UNet on the HEALPix mesh."""

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        input_channels: int,
        output_channels: int,
        n_constants: int,
        decoder_input_channels: int,
        input_time_dim: int,
        output_time_dim: int,
        presteps: int = 0,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        couplings: list = [],
        residual_prediction: bool = False,
        couplings_time_first: bool = True,
        couplings_time_first: bool = True,
        constraints: list[DictConfig] = None,
        hpx_padding_mode: str = 'karlbauer',
        enforce_reflectional_equivariance: bool = False,
        odd_prognostic_variables: Sequence[str] = None,
        odd_constants: Sequence[str] = None,
        channels: Sequence[str] = None,
        constants: Sequence[str] = None,
        scaling: dict[str, dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        encoder: DictConfig
            dictionary of instantiable parameters for the U-net encoder
        decoder: DictConfig
            dictionary of instantiable parameters for the U-net decoder
        input_channels: int
            number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        output_channels: int
            number of output channels expected in the output array schema, or output variables
        n_constants: int
            number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        decoder_input_channels: int
            number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        input_time_dim: int
            number of time steps in the input array
        output_time_dim: int
            number of time steps in the output array
        presteps: int, optional
            number of model steps to initialize recurrent states. default: 0
        enable_nhwc: bool, optional
            Model with [N, H, W, C] instead of [N, C, H, W]. default: False
        enable_healpixpad: bool, optional
            Enable CUDA HEALPixPadding if installed. default: False
        couplings: list, optional
            sequence of dictionaries that describe coupling mechanisms
        residual_prediction: bool, optional
            If the model should predict the residual between the input and the output. Default: False
        couplings_time_first: bool, optional
            Whether coupled data is in [T, B, C, F, H, W] rather than [B, F, T, C, H, W] format
        constraints: list[DictConfig], optional
            List of hydra instantiable DictConfigs specifying constraints 
            (e.g., nonnegativity) to be applied to the model outputs
        hpx_padding_mode: str, optional
            Method to use for padding HEALPix faces for convolutions. Options
            are 'karlbauer' (default) and 'isolatitude'.
        """
        super().__init__()

        if len(couplings) > 0:
            if n_constants == 0:
                raise NotImplementedError(
                    "support for coupled models with no constant fields is not available at this time."
                )
            if decoder_input_channels == 0:
                raise NotImplementedError(
                    "support for coupled models with no decoder inputs (TOA insolation) is not available at this time."
                )

        # add coupled fields to input channels for model initialization
        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.train_couplers = None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 2  # Now 2 with [B, F, C*T, H, W]. Was 1 in old data format with [B, T*C, F, H, W]
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad
        self.residual_prediction = residual_prediction
        self.couplings_time_first = couplings_time_first
        self.hpx_padding_mode = hpx_padding_mode
        self.enforce_reflectional_equivariance = enforce_reflectional_equivariance
        self.odd_prognostic_variables = odd_prognostic_variables
        self.odd_constants = odd_constants
        self.channels = channels
        self.constants = constants
        self.scaling = scaling

        # Setting variables which are used for enforcing reflectional equivariance
        self.register_buffer("refl_face_order", th.tensor([8,9,10,11,4,5,6,7,0,1,2,3], dtype=th.long), persistent=False)

        if self.enforce_reflectional_equivariance:

            odd_out_var_idx = th.tensor([self.channels.index(v) for v in self.odd_prognostic_variables], dtype=th.long) \
                if self.odd_prognostic_variables is not None else None
            self.register_buffer("odd_out_var_idx", odd_out_var_idx, persistent=False)

            odd_in_vars = []
            odd_in_var_idx = []
            if self.odd_prognostic_variables is not None:
                odd_in_vars += self.odd_prognostic_variables
                odd_in_var_idx = [self.channels.index(v) for v in odd_in_vars]
            if self.odd_constants is not None:
                odd_in_vars += self.odd_constants
                odd_const_idx = [
                    self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.constants.index(c)
                    for c in self.odd_constants
                ]
                odd_in_var_idx += odd_const_idx

            odd_in_var_idx = th.tensor(odd_in_var_idx, dtype=th.long) if len(odd_in_vars) > 0 else None
            self.register_buffer("odd_in_var_idx", odd_in_var_idx, persistent=False)
            
            odd_in_var_mean = th.tensor([self.scaling[var]['mean'] for var in odd_in_vars]) if len(odd_in_vars) > 0 else None
            for i, mean in enumerate(odd_in_var_mean):
                if mean != 0.0:
                    raise ValueError(
                        f"Reflectional equivariance can only be enforced if all odd variables have zero mean. "
                        f"Odd variable {odd_in_vars[i]} has mean {mean.item()}"
                    )

            if len(odd_in_vars) == 0:
                logger.warning(
                    "Reflectional equivariance is enabled but no odd variables "
                    "were specified. The model will be reflectionally equivariant "
                    "only if all input variables are even scalars."
                )

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        # Build the model layers
        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)
        self.encoder = instantiate(
            config=encoder,
            input_channels=self._compute_input_channels(),
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
            hpx_padding_mode=self.hpx_padding_mode,
        )
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(
            config=decoder,
            output_channels=self._compute_output_channels(),
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
            hpx_padding_mode=self.hpx_padding_mode,
        )

        self.constraints = None
        self.set_constraints(constraints)

    @property
    def integration_steps(self):
        """Number of integration steps"""
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        """Calculate total number of input channels in the model"""
        return (
            self.input_time_dim * (self.input_channels + self.decoder_input_channels)
            + self.n_constants
            + self.coupled_channels
        )

    def _compute_coupled_channels(self, couplings):

        c_channels = 0
        for c in couplings:
            c_channels += len(c["params"]["variables"]) * len(
                c["params"]["input_times"]
            )
        return c_channels

    def _compute_output_channels(self) -> int:
        """Compute the total number of output channels in the model"""
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.

        Parameters
        ----------
        inputs: Sequence
            list of expected input tensors (inputs, decoder_inputs, constants)
        step: int, optional
            step number in the sequence of integration_stepsi. default: 0

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model encoder
        """

        if len(self.couplings) > 0:
            result = [
                inputs[0].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                ),
                inputs[1][
                    :,
                    :,
                    slice(step * self.input_time_dim, (step + 1) * self.input_time_dim),
                    ...,
                ].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                ),  # DI
                inputs[2].expand(
                    *tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])
                ),  # constants
                inputs[3].permute(0, 2, 1, 3, 4) if self.couplings_time_first else inputs[3],  # coupled inputs
            ]
            res = th.cat(result, dim=self.channel_dim)

        else:
            if not (self.n_constants > 0 or self.decoder_input_channels > 0):
                res = inputs[0].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                )
                return self.fold(res)
            if self.n_constants == 0:
                result = [
                    inputs[0].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                    ),  # inputs
                    inputs[1][
                        :,
                        :,
                        slice(
                            step * self.input_time_dim, (step + 1) * self.input_time_dim
                        ),
                        ...,
                    ].flatten(
                        self.channel_dim, self.channel_dim + 1
                    ),  # DI
                ]
                res = th.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)

                return res
            if self.decoder_input_channels == 0:
                result = [
                    inputs[0].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                    ),  # inputs
                    inputs[1].expand(
                        *tuple([inputs[0].shape[0]] + len(inputs[1].shape) * [-1])
                    ),  # constants
                ]
                res = th.cat(result, dim=self.channel_dim)

                # fold faces into batch dim
                res = self.fold(res)

                return res

            result = [
                inputs[0].flatten(
                    start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                ),  # inputs
                inputs[1][
                    :,
                    :,
                    slice(step * self.input_time_dim, (step + 1) * self.input_time_dim),
                    ...,
                ].flatten(
                    self.channel_dim, self.channel_dim + 1
                ),  # DI
                inputs[2].expand(
                    *tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])
                ),  # constants
            ]
            res = th.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)

        return res

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:
        """Returns a maultiple tensors to from the model decoder.
        Splits the time/channel dimensions.

        Parameters
        ----------
        inputs: Sequence
            list of expected input tensors (inputs, decoder_inputs, constants)
        step: int, optional
            step number in the sequence of integration_steps

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model outputs
        """

        # unfold:
        outputs = self.unfold(outputs)

        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = th.reshape(
            outputs,
            shape=(
                shape[0],
                shape[1],
                1 if self.is_diagnostic else self.input_time_dim,
                -1,
                *shape[3:],
            ),
        )

        return res

    def set_constraints(self, constraints: list[DictConfig] = None):
        """
        Sets constraints (e.g., non-negative) to be applied to the model outputs
        Parameters
        ----------
        constraints: list[DictConfig]
            List of hydra instantiable DictConfigs specifying constraints
        """
        if constraints is not None:
            self.constraints = [instantiate(constraints[constraint]) for constraint in constraints]

    def hpx_reflect(
        self,
        x,
        latent_tensor: bool = False,
        includes_constants: bool = False,
    ):
        '''
        Helper function to reflect a HPX tensor across its horizontal axis.
        Assumes x has shape [B*F,C,H,W]
        '''
        # Reflect each face individually
        x = th.rot90(th.flip(x, dims=[3]), dims=(-1,-2))

        # Unfold faces from batch dimension and reorder to swap N/S faces
        x = x.reshape(-1, 12, *x.shape[1:])
        x = th.index_select(x, dim=1, index=self.refl_face_order.to(x.device))

        # Refold faces into batch dimension
        x = x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])

        # Flip sign of odd variables (e.g., v-velocity, f)
        if not latent_tensor:
            
            var_idx = self.odd_in_var_idx if includes_constants else self.odd_out_var_idx

            if var_idx is not None:
                v = th.index_select(x, dim=1, index=var_idx)
                v = -1 * v
                x.index_copy_(
                    1,
                    var_idx,
                    v.to(x.dtype)
                )

        return x

    def forward(self, inputs: Sequence, output_only_last=False, conditions_cln: Sequence=None) -> th.Tensor:
        """
        Forward pass of the HEALPixUnet

        Parameters
        ----------
        inputs: Sequence
            Inputs to the model, of the form [prognostics|TISR|constants]
            [B, F, T, C, H, W] is the format for prognostics and TISR
            [F, C, H, W] is the format for constants
        output_only_last: bool, optional
            If only the last dimension of the outputs should be returned. default: False
        conditions_cln: Sequence, optional
            If the model is using conditional normalization, this is a sequence of tensors that will be used to condition the 
            normalization layers. The shape of the tensors should be [Cond*B, N], where N is the size of the conditions, Cond is the 
            number of conditions, and B is the batch size. It is expected that the inputs have a leading dimension of Cond*B (e.g., data
            for different ensmble members/conditions has been duplicated along this dimension). The sequence should have length equal to
            the model's `n_integration_steps` attribute.
        Returns
        -------
        th.Tensor: Predicted outputs
        """
        outputs = []
        for step in range(self.integration_steps):
            if step == 0:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        list(inputs[0:3]) + [inputs[3][step]], step
                    )
                else:
                    input_tensor = self._reshape_inputs(inputs, step)
            else:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        [outputs[-1][:, :, :, :self.input_channels]]
                        + list(inputs[1:3])
                        + [inputs[3][step]],
                        step
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        [outputs[-1][:, :, :, :self.input_channels]]
                        + list(inputs[1:]),
                        step
                    )

            # Forward through model, with or without conditions
            kwargs = {}
            if conditions_cln is not None:
                kwargs = {"conditions_cln": conditions_cln[step]}
            else:
                kwargs = {}

            encodings = self.encoder(input_tensor, **kwargs)
            decodings = self.decoder(encodings, **kwargs)

            # Forward through model again with reflected input and original hidden states
            if self.enforce_reflectional_equivariance:

                # Forward through model with reflected input
                input_tensor_refl = self.hpx_reflect(input_tensor, includes_constants=True)
                encodings_refl = self.encoder(input_tensor_refl, **kwargs)
                decodings_refl = self.decoder(encodings_refl, **kwargs)               

                # Average of decodings
                decodings = 0.5 * (decodings + self.hpx_reflect(decodings_refl, includes_constants=False))
            
            # Reshape from [B*F, T*C, H, W] to [B, F, T, C, H, W]
            combined = self._reshape_outputs(decodings)
            prognostics = combined[:, :, :, :self.input_channels]
            diagnostics = combined[:, :, :, self.input_channels:]

            # Residual prediction
            orig_input = self._reshape_outputs(
                input_tensor[:, :self.input_channels * self.input_time_dim]
            )
            if self.residual_prediction:
                prognostics += orig_input
            
            # Concat along channel dim, shape is [B, F, T, C, H, W]
            out = th.cat([prognostics, diagnostics], dim=3)

            # Apply constraints
            if self.constraints is not None:
                for constraint in self.constraints:
                    out = constraint(out, orig_input)

            outputs.append(out)

        if output_only_last:
            res = outputs[-1]
        else:
            res = th.cat(outputs, dim=self.channel_dim)

        return res
