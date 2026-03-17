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
from typing import Optional, Sequence, Tuple, Union

import pandas as pd
import torch as th
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo.models.dlwp_healpix_layers import (
    CoupledEmbedding,
    HEALPixFoldFaces,
    HEALPixUnfoldFaces,
)
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

logger = logging.getLogger(__name__)


@dataclass
class MetaData(ModelMetaData):
    """Metadata for the DLWP HEALPix Model with coupled cross-attention"""

    name: str = "DLWP_HEALPixRecCoupledCrossAttention"
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    onnx: bool = False
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class HEALPixRecUNetCoupledCrossAttention(Module):
    """
    HEALPix recurrent UNet with separate coupled stream and cross-attention.
    Coupled variables are embedded (per-face CNN + face self-attention) and
    used as context in within-face cross-attention at encoder/decoder levels.
    """

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
        enable_healpixpad: bool = False,
        couplings: list = None,
        residual_prediction: bool = True,
        constraints: list = None,
        encoder_attention_levels: Sequence[int] = None,
        decoder_attention_levels: Sequence[int] = None,
        decoder_attention_position: str = "after_recurrent",
        coupled_embed_dim: int = 64,
        cross_attention_heads: int = 4,
        cross_attention_dropout: float = 0.0,
        num_faces: int = 12,
        cross_face_patch_size: Optional[int] = None,
        cross_face_heads: int = 4,
        cross_face_dropout: float = 0.0,
    ):
        super().__init__()
        couplings = couplings or []
        if len(couplings) == 0:
            raise ValueError("HEALPixRecUNetCoupledCrossAttention requires couplings.")
        if n_constants == 0:
            raise NotImplementedError(
                "support for coupled models with no constant fields is not available."
            )
        if decoder_input_channels == 0:
            raise NotImplementedError(
                "support for coupled models with no decoder inputs (TOA insolation) is not available."
            )

        self.channel_dim = 2
        self.input_channels = input_channels
        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.delta_t = int(pd.Timedelta(delta_time).total_seconds() // 3600)
        self.reset_cycle = int(pd.Timedelta(reset_cycle).total_seconds() // 3600)
        self.presteps = presteps
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad
        self.residual_prediction = residual_prediction
        self.num_faces = num_faces
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        encoder_attention_levels = list(encoder_attention_levels or [])
        decoder_attention_levels = list(decoder_attention_levels or [])
        required_levels = sorted(set(encoder_attention_levels) | set(decoder_attention_levels))

        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=num_faces)

        self.coupled_embedding = CoupledEmbedding(
            coupled_channels=self.coupled_channels,
            embed_dim=coupled_embed_dim,
            attention_levels=required_levels,
            num_faces=num_faces,
        )

        self.encoder = instantiate(
            config=encoder,
            input_channels=self._compute_input_channels(),
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
            encoder_attention_levels=encoder_attention_levels,
            coupled_embed_dim=coupled_embed_dim,
            cross_attention_heads=cross_attention_heads,
            cross_attention_dropout=cross_attention_dropout,
            num_faces=num_faces,
            cross_face_patch_size=cross_face_patch_size,
            cross_face_heads=cross_face_heads,
            cross_face_dropout=cross_face_dropout,
        )
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(
            config=decoder,
            output_channels=self._compute_output_channels(),
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
            decoder_attention_levels=decoder_attention_levels,
            decoder_attention_position=decoder_attention_position,
            coupled_embed_dim=coupled_embed_dim,
            cross_attention_heads=cross_attention_heads,
            cross_attention_dropout=cross_attention_dropout,
            num_faces=num_faces,
            cross_face_patch_size=cross_face_patch_size,
            cross_face_heads=cross_face_heads,
            cross_face_dropout=cross_face_dropout,
        )

        self.constraints = None
        self.set_constraints(constraints)

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        """Total input channels excluding coupled (coupled goes through CoupledEmbedding)."""
        return (
            self.input_time_dim * (self.input_channels + self.decoder_input_channels)
            + self.n_constants
        )

    def _compute_coupled_channels(self, couplings):
        c_channels = 0
        for c in couplings:
            c_channels += len(c["params"]["variables"]) * len(c["params"]["input_times"])
        return c_channels

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(
        self, inputs: Sequence, step: int = 0
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Returns (main_tensor, coupled_tensor). Main tensor is [BF, C_main, H, W]
        (prognostics, TISR, constants). Coupled tensor is [BF, C_coupled, H, W].
        """
        # inputs: [prognostics, decoder_inputs, constants, coupled_for_step]
        # coupled_for_step is already the tensor for this step: [B*Cond, C_coupled, F, H, W]
        coupled_raw = inputs[3]  # [B*Cond, C_coupled, F, H, W]
        # Fold coupled to [BF, C_coupled, H, W]
        coupled_tensor = self.fold(coupled_raw.permute(0, 2, 1, 3, 4))

        result = [
            inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1),
            inputs[1][
                :,
                :,
                slice(step * self.input_time_dim, (step + 1) * self.input_time_dim),
                ...,
            ].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1),
            inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])),
        ]
        main = th.cat(result, dim=self.channel_dim)
        main = self.fold(main)
        if self.enable_nhwc:
            main = main.to(memory_format=th.channels_last)
        return main, coupled_tensor

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:
        outputs = self.unfold(outputs)
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

    def set_constraints(self, constraints: list = None):
        if constraints is not None:
            self.constraints = [instantiate(constraints[constraint]) for constraint in constraints]

    def _initialize_hidden(
        self,
        inputs: Sequence,
        outputs: Sequence,
        step: int,
        conditions_cln: Sequence = None,
    ) -> None:
        self.reset()
        for prestep in range(self.presteps):
            if step < self.presteps:
                s = step + prestep
                main_tensor, coupled_tensor = self._reshape_inputs(
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
                s = step - self.presteps + prestep
                main_tensor, coupled_tensor = self._reshape_inputs(
                    inputs=[outputs[s - 1]] + list(inputs[1:3]) + [inputs[3][step - (prestep - self.presteps)]],
                    step=s + 1,
                )
            coupled_embedding = self.coupled_embedding(coupled_tensor)
            kwargs = {"coupled_embedding": coupled_embedding}
            if conditions_cln is not None:
                kwargs["conditions_cln"] = conditions_cln
            self.decoder(self.encoder(main_tensor, **kwargs), **kwargs)

    def forward(
        self,
        inputs: Sequence,
        output_only_last: bool = False,
        conditions_cln: Sequence = None,
    ) -> th.Tensor:
        self.reset()
        outputs = []
        for step in range(self.integration_steps):
            if (step * (self.delta_t * self.input_time_dim)) % self.reset_cycle == 0:
                if conditions_cln is not None:
                    self._initialize_hidden(
                        inputs=inputs,
                        outputs=outputs,
                        step=step,
                        conditions_cln=conditions_cln[step],
                    )
                else:
                    self._initialize_hidden(inputs=inputs, outputs=outputs, step=step)

            if step == 0:
                s = self.presteps
                main_tensor, coupled_tensor = self._reshape_inputs(
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
                main_tensor, coupled_tensor = self._reshape_inputs(
                    inputs=[outputs[-1]] + list(inputs[1:3]) + [inputs[3][self.presteps + step]],
                    step=step + self.presteps,
                )

            coupled_embedding = self.coupled_embedding(coupled_tensor)
            kwargs = {"coupled_embedding": coupled_embedding}
            if conditions_cln is not None:
                kwargs["conditions_cln"] = conditions_cln[step]

            encodings = self.encoder(main_tensor, **kwargs)
            decodings = self.decoder(encodings, **kwargs)

            if self.residual_prediction:
                prediction = main_tensor[:, : self.input_channels * self.input_time_dim] + decodings
            else:
                prediction = decodings

            reshaped = self._reshape_outputs(prediction)
            if self.constraints is not None:
                for constraint in self.constraints:
                    reshaped = constraint(reshaped)
            outputs.append(reshaped)

        if output_only_last:
            return outputs[-1]
        return th.cat(outputs, dim=self.channel_dim)

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()
