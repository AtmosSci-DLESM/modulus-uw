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

import pandas as pd
import torch as th
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo.models.dlwp_healpix_layers import (
    HEALPixFoldFaces,
    HEALPixUnfoldFaces,
    warn_deprecated_enable_healpixpad,
)
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

logger = logging.getLogger(__name__)


@dataclass
class MetaData(ModelMetaData):
    """Metadata for the DLWP HEALPix Model"""

    name: str = "DLWP_HEALPixRec"
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


class HEALPixRecUNet(Module):
    """Deep Learning Weather Prediction (DLWP) recurrent UNet model on the HEALPix mesh."""

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
        delta_time: str = "6h",
        reset_cycle: str = "24h",
        presteps: int = 1,
        enable_nhwc: bool = False,
        couplings: list = [],
        residual_prediction: bool = True,
        couplings_time_first: bool = True,
        constraints: list[DictConfig] = None,
        hpx_padding_mode: str | None = None,
        compile_padding: bool = False,
        nside: Sequence[int] = (64, 32, 16),
        enable_healpixpad: bool | None = None,
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
        delta_time: str, optional
            hours between two consecutive data points
        reset_cycle: str, optional
            hours after which the recurrent states are reset to zero and re-initialized. Set np.infty
            to never reset the hidden states.
        presteps: int, optional
            number of model steps to initialize recurrent states.
        enable_nhwc: bool, optional
            Model with [N, H, W, C] instead of [N, C, H, W]
        couplings: list, optional
            sequence of dictionaries that describe coupling mechanisms
        residual_prediction: bool, optional
            If the model should predict the residual between the input and the output. Default: True
        couplings_time_first: bool, optional
            Whether coupled data is in [T, B, C, F, H, W] rather than [B, F, T, C, H, W] format
        constraints: list[DictConfig], optional
            List of hydra instantiable DictConfigs specifying constraints 
            (e.g., nonnegativity) to be applied to the model outputs
        hpx_padding_mode: str, optional
            Padding strategy: ``earth2grid``, ``karlbauer``, or ``isolatitude``.
            ``None`` (omitted) defaults to ``earth2grid`` unless deprecated ``enable_healpixpad``
            is set without an explicit ``hpx_padding_mode``.
        compile_padding: bool, optional
            If True, apply torch compile to the padding module.
        nside : Sequence[int], optional
            Face height/width per UNet level (shallowest to deepest).
            Length must match the encoder/decoder ``n_channels`` list length.
            Default ``(64, 32, 16)``.
        enable_healpixpad: bool, optional
            Deprecated. When ``hpx_padding_mode`` is omitted, ``False`` maps to ``karlbauer``
            and ``True`` to ``earth2grid`` (legacy configs). Prefer ``hpx_padding_mode``.
        """
        super().__init__()
        hpx_padding_mode = warn_deprecated_enable_healpixpad(enable_healpixpad, hpx_padding_mode)
        self.channel_dim = 2  # Now 2 with [B, F, T*C, H, W]. Was 1 in old data format with [B, T*C, F, H, W]

        self.input_channels = input_channels

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

        # add coupled fields to input channels for model initialization
        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.train_couplers = None
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.delta_t = int(pd.Timedelta(delta_time).total_seconds() // 3600)
        if reset_cycle == float('inf'):
            self.reset_cycle = reset_cycle
        else:
            self.reset_cycle = int(pd.Timedelta(reset_cycle).total_seconds() // 3600)
        self.presteps = presteps
        self.enable_nhwc = enable_nhwc
        self.residual_prediction = residual_prediction
        self.couplings_time_first = couplings_time_first
        self.hpx_padding_mode = hpx_padding_mode
        self.compile_padding = compile_padding
        self.nside = nside

        if len(encoder["n_channels"]) != len(decoder["n_channels"]):
            raise ValueError(
                "encoder and decoder must have the same number of UNet levels; "
                f"got {len(encoder['n_channels'])} for encoder and {len(decoder['n_channels'])} for decoder"
            )
        if len(self.nside) != len(encoder["n_channels"]):
            raise ValueError(
                f"nside must have same length as n_channels; got {len(self.nside)} "
                f"for nside and {len(encoder['n_channels'])} for n_channels"
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
            hpx_padding_mode=self.hpx_padding_mode,
            compile_padding=self.compile_padding,
            nside=self.nside,
        )
        self.decoder = instantiate(
            config=decoder,
            output_channels=self._compute_output_channels(),
            enable_nhwc=self.enable_nhwc,
            hpx_padding_mode=self.hpx_padding_mode,
            compile_padding=self.compile_padding,
            nside=self.nside,
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
        """Get number of coupled channels

        Returns
        -------
        int
            The number of coupled channels
        """
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
            step number in the sequence of integration_steps

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model encoder [F*B, T*C+n_constants+(coupled_channels*coupled_input_times), H, W]
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
            if self.n_constants == 0:
                result = [
                    inputs[0].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
                    ),
                    inputs[1][
                        :,
                        :,
                        slice(
                            step * self.input_time_dim, (step + 1) * self.input_time_dim
                        ),
                        ...,
                    ].flatten(
                        start_dim=self.channel_dim, end_dim=self.channel_dim + 1
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
                    ),
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
            ]
            res = th.cat(result, dim=self.channel_dim)

        # fold faces into batch dim (BF, C, H, W)
        res = self.fold(res)
        if self.enable_nhwc:
            res = res.to(memory_format=th.channels_last)
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

    def _initialize_hidden(
        self, inputs: Sequence, outputs: Sequence, step: int, conditions_cln: Sequence = None
    ) -> None:
        """Initialize the hidden layers

        Parameters
        ----------
        inputs: Sequence
            Inputs to use to initialize the hideen layers
        outputs: Sequence
            Outputs to use to initialize the hideen layers
        step: int
            Current step number of the initialization
        conditions_cln: Sequence, optional
            Conditional inputs for the normalization layers.
        """
        self.reset()
        for prestep in range(self.presteps):
            if step < self.presteps:
                s = step + prestep
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                            ]
                        ]
                        + list(inputs[1:3])
                        + [inputs[3][prestep]],
                        step=step + prestep,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                            ]
                        ]
                        + list(inputs[1:]),
                        step=step + prestep,
                    )
            else:
                s = step - self.presteps + prestep
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[s - 1][:, :, :, :self.input_channels]]
                        + list(inputs[1:3])
                        + [inputs[3][step - (prestep - self.presteps)]],
                        step=s + 1,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[s - 1][:, :, :, :self.input_channels]] + list(inputs[1:]), step=s + 1
                    )
            # Forward the data through the model to initialize hidden states
            self.decoder(self.encoder(input_tensor, conditions_cln=conditions_cln), conditions_cln=conditions_cln)

    def forward(self, inputs: Sequence, output_only_last=False, conditions_cln=None) -> th.Tensor:
        """
        Forward pass of the HEALPixUnet

        Parameters
        ----------
        inputs: Sequence
            Inputs to the model, of the form [prognostics|TISR|constants|coupled inputs].
            [B*Cond, F, T, C, H, W] is the format for prognostics and TISR. Cond is the number of (optional) conditional inputs.
                Note the time dimension in prognostics is for initialization and hidden state priming (input_time_dim*2) while
                the T dimension in TISR is for initialization and hidden state priming as well as roll-out. There are 2 additional 
                time steps provided to TISR that are apparently not used. 
            [F, C, H, W] is the format for constants
            [T, B*Cond, C, F, H, W] is the format for coupled inputs. Here time is for initialization and roll-out (one per model step).
        output_only_last: bool, optional
            If only the last dimension of the outputs should be returned
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
        # Do not call self.reset() at the top of every forward. Finite reset_cycle
        # still re-zeros via _initialize_hidden when hours % reset_cycle == 0
        # (always true at step 0). With reset_cycle=inf, wiping here would discard
        # GRU state between coupled-inference forward() calls (one per coupler window).
        outputs = []
        for step in range(self.integration_steps):
            # th.cuda.nvtx.range_push(f"Integration step: {step}")
            # (Re-)initialize recurrent hidden states
            hours = step * (self.delta_t * self.input_time_dim)
            if self.reset_cycle == float("inf"):
                # 0 % inf == 0, so the finite-cycle modulo would re-prime every
                # forward at step 0. Honor "never reset" after the first prime so
                # hidden state carries across coupler windows when inference sets
                # reset_cycle=inf (disable_*_recurrent_reset).
                need_init = not getattr(self, "_recurrent_hidden_primed", False)
            else:
                need_init = (hours % self.reset_cycle) == 0
            if need_init:
                if conditions_cln is not None:
                    self._initialize_hidden(inputs=inputs, outputs=outputs, step=step, conditions_cln=conditions_cln[step])
                else:
                    self._initialize_hidden(inputs=inputs, outputs=outputs, step=step)
                if self.reset_cycle == float("inf"):
                    self._recurrent_hidden_primed = True

            # Construct concatenated input: [prognostics|TISR|constants]
            if step == 0:
                s = self.presteps
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                            ]
                        ]
                        + list(inputs[1:3])
                        + [inputs[3][s]],
                        step=s,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[
                            inputs[0][
                                :,
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                            ]
                        ]
                        + list(inputs[1:]),
                        step=s,
                    )
            else:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[-1][:, :, :, :self.input_channels]]
                        + list(inputs[1:3])
                        + [inputs[3][self.presteps + step]],
                        step=step + self.presteps,
                    )
                else:
                    input_tensor = self._reshape_inputs(
                        inputs=[outputs[-1][:, :, :, :self.input_channels]] + list(inputs[1:]),
                        step=step + self.presteps,
                    )
            # th.cuda.nvtx.range_pop()

            # Forward through model, with or without conditions
            if conditions_cln is not None:
                kwargs = {"conditions_cln": conditions_cln[step]}
            else:
                kwargs = {}

            # th.cuda.nvtx.range_push("Encoder")
            encodings = self.encoder(input_tensor, **kwargs)
            # th.cuda.nvtx.range_pop()
            # th.cuda.nvtx.range_push("Dencoder")
            decodings = self.decoder(encodings, **kwargs)
            # th.cuda.nvtx.range_pop()

            # Residual prediction
            combined = self._reshape_outputs(decodings)
            prognostics = combined[:, :, :, :self.input_channels]
            orig_input = self._reshape_outputs(input_tensor[:, : self.input_channels * self.input_time_dim])
            if self.residual_prediction:
                prognostics += orig_input
            diagnostics = combined[:, :, :, self.input_channels:]
            out = th.cat([prognostics, diagnostics], dim=3)

            # Apply constraints
            if self.constraints is not None:
                for constraint in self.constraints:
                    out = constraint(out, orig_input)

            outputs.append(out)
            # th.cuda.nvtx.range_pop()

        if output_only_last:
            return outputs[-1]

        return th.cat(outputs, dim=self.channel_dim)

    def reset(self):
        """Resets the state of the network"""
        self.encoder.reset()
        self.decoder.reset()
        # Allow a later forward to re-prime when reset_cycle is inf.
        self._recurrent_hidden_primed = False
