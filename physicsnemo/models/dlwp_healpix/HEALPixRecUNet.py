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
import warnings
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
from physicsnemo.models.dlwp_healpix_layers.coupled_partial_conv import (
    build_coupled_partial_conv_stem,
)
from physicsnemo.models.dlwp_healpix_layers.reflection_ops import (
    apply_channel_order,
    bank_sizes,
    compute_sin_lat_faces,
    expand_sin_lat_folded,
    hpx_reflect_typed,
    is_nonzero_scalar_mean,
    reorder_channels_even_odd,
    resolve_reflection_equivariance_mode,
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
        enforce_reflectional_equivariance: bool = False,
        reflection_equivariance_mode: str | None = None,
        odd_prognostic_variables: Sequence[str] = None,
        odd_diagnostic_variables: Sequence[str] = None,
        odd_constants: Sequence[str] = None,
        odd_coupled_variables: Sequence[str] = None,
        odd_fraction: float = 0.25,
        channels: Sequence[str] = None,
        output_channel_names: Sequence[str] = None,
        constants: Sequence[str] = None,
        scaling: dict[str, dict[str, float]] = None,
        enable_healpixpad: bool | None = None,
        coupled_partial_conv: dict | DictConfig | None = None,
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
            Hours after which recurrent states are reset to zero and re-initialized.
            Set ``float("inf")`` (or ``np.infty``) to never reset during ``forward()``:
            hidden state then persists across successive ``forward()`` calls (for
            example coupled-inference coupler windows). Call ``reset()`` between
            independent forecasts or initial conditions so state does not leak from
            one initialization to the next. ``inf`` during training keeps state
            across batches until ``reset()``; a warning is emitted on the first
            training ``forward()`` so that is not accidental.
        presteps: int, optional
            number of model steps to initialize recurrent states.
        enable_nhwc: bool, optional
            If True, use channels-last (NHWC) memory format for folded face
            tensors and HEALPix conv/padding layers. Logical shape stays
            ``[N, C, H, W]``.
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
        reflection_equivariance_mode: str, optional
            Hard Z₂ equatorial reflection equivariance:

            * ``off`` — no hard equivariance (default when omitted).
            * ``structural`` — ReflectionSteerable layers (even/odd banks, constrained
              kernels); one forward per step. Legacy alias: ``steerable``.
            * ``averaged`` — twin-forward Reynolds projector on outputs and GRU
              state. Legacy alias: ``reynolds``. Legacy
              ``enforce_reflectional_equivariance=True`` also selects ``averaged``.
        enforce_reflectional_equivariance: bool, optional
            Legacy flag; ``True`` maps to ``averaged`` when mode is omitted/``off``.
        odd_fraction: float, optional
            Fraction of latent channels typed odd under reflection (structural mode).
        odd_prognostic_variables / odd_constants / odd_coupled_variables:
            Physical channel names that flip sign under equatorial reflection.
        odd_diagnostic_variables: sequence of str, optional
            Output-only (diagnostic) channel names that flip sign under equatorial
            reflection. Diagnostics remain absolute predictions (no residual); only
            their even/odd bank typing changes. Requires ``output_channel_names``.
        output_channel_names: sequence of str, optional
            Names of decoder output channels in tensor order. Defaults to ``channels``
            when omitted (prognostic-only layouts). Required when
            ``odd_diagnostic_variables`` is non-empty.
        coupled_partial_conv: dict or DictConfig, optional
            Opt-in partial-convolution stem for coupled inputs. ``None`` (default)
            leaves coupled fields unchanged. Configure ``masks`` plus an optional
            Hydra-instantiable ``stem`` (default ``CoupledPartialConvStem``). See
            ``physicsnemo.models.dlwp_healpix_layers.coupled_partial_conv``.
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
        self._recurrent_hidden_primed = False
        self._reset_cycle_inf_train_warned = False
        self.presteps = presteps
        self.enable_nhwc = enable_nhwc
        self.residual_prediction = residual_prediction
        self.couplings_time_first = couplings_time_first
        self.hpx_padding_mode = hpx_padding_mode
        self.compile_padding = compile_padding
        self.nside = nside
        self.enforce_reflectional_equivariance = enforce_reflectional_equivariance
        self.odd_prognostic_variables = odd_prognostic_variables
        self.odd_diagnostic_variables = odd_diagnostic_variables
        self.odd_constants = odd_constants
        self.odd_coupled_variables = odd_coupled_variables
        self.odd_fraction = odd_fraction
        self.channels = channels
        self.output_channel_names = output_channel_names
        self.constants = constants
        self.scaling = scaling

        self.reflection_equivariance_mode = resolve_reflection_equivariance_mode(
            reflection_equivariance_mode, enforce_reflectional_equivariance
        )
        # Keep legacy flag aligned with resolved mode for existing Reynolds code paths.
        self.enforce_reflectional_equivariance = self.reflection_equivariance_mode == "averaged"

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

        # Setting variables which are used for enforcing reflectional equivariance
        self.register_buffer("refl_face_order", th.tensor([8,9,10,11,4,5,6,7,0,1,2,3], dtype=th.long), persistent=False)

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        if self.reflection_equivariance_mode in ("averaged", "structural"):
            self._setup_reflection_channel_indices()

        # Build the model layers
        self.fold = HEALPixFoldFaces(enable_nhwc=self.enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=12)

        encoder_kwargs = dict(
            config=encoder,
            input_channels=self._compute_input_channels(),
            enable_nhwc=self.enable_nhwc,
            hpx_padding_mode=self.hpx_padding_mode,
            compile_padding=self.compile_padding,
            nside=self.nside,
        )
        decoder_kwargs = dict(
            config=decoder,
            output_channels=self._compute_output_channels(),
            enable_nhwc=self.enable_nhwc,
            hpx_padding_mode=self.hpx_padding_mode,
            compile_padding=self.compile_padding,
            nside=self.nside,
        )
        if self.reflection_equivariance_mode == "structural":
            encoder_kwargs["structural_input_even"] = int(self.structural_input_n_even)
            decoder_kwargs["structural_output_even"] = int(self.structural_output_n_even)

        encoder_kwargs["odd_fraction"] = self.odd_fraction
        decoder_kwargs["odd_fraction"] = self.odd_fraction

        self.encoder = instantiate(**encoder_kwargs)
        self.decoder = instantiate(**decoder_kwargs)

        # Opt-in stem: partial-conv over coupled SST/SIC (etc.) before channel concat.
        # Default None preserves historical behavior. nside[0] is full-res face size.
        self.coupled_partial_conv_stem = build_coupled_partial_conv_stem(
            coupled_partial_conv,
            couplings=self.couplings,
            hpx_padding_mode=self.hpx_padding_mode,
            nside=int(self.nside[0]) if self.nside is not None else None,
            compile_padding=self.compile_padding,
            enable_nhwc=self.enable_nhwc,
            odd_coupled_variables=self.odd_coupled_variables,
        )

        if self.reflection_equivariance_mode == "structural":
            # Only materialize sin_lat gates when some layer opts into cross-parity 1×1.
            self._structural_needs_sin_lat_gate = any(
                getattr(m, "needs_sin_lat", False)
                for m in list(self.encoder.modules()) + list(self.decoder.modules())
            )
            if self._structural_needs_sin_lat_gate:
                for ns in self.nside:
                    self.register_buffer(
                        f"sin_lat_nside_{ns}",
                        compute_sin_lat_faces(int(ns)),
                        persistent=False,
                    )
        else:
            self._structural_needs_sin_lat_gate = False

        self.constraints = None
        self.set_constraints(constraints)

    @property
    def integration_steps(self):
        """Number of integration steps"""
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _setup_reflection_channel_indices(self) -> None:
        """Build odd-channel index buffers for averaged and/or structural modes."""
        if self.channels is None:
            raise ValueError("channels must be provided when reflection equivariance is enabled")

        odd_prog = list(self.odd_prognostic_variables or [])
        odd_diag = list(self.odd_diagnostic_variables or [])
        out_names = list(
            self.output_channel_names if self.output_channel_names is not None else self.channels
        )

        if odd_diag:
            if self.output_channel_names is None:
                raise ValueError(
                    "output_channel_names must be provided when odd_diagnostic_variables is non-empty"
                )
            channel_set = set(self.channels)
            for v in odd_diag:
                if v in channel_set:
                    raise ValueError(
                        f"Odd diagnostic variable {v!r} is also an input/prognostic channel; "
                        f"list it under odd_prognostic_variables instead"
                    )
                if v not in out_names:
                    raise ValueError(
                        f"Odd diagnostic variable {v!r} not found in output_channel_names {out_names}"
                    )

        # Full concatenated encoder input layout after _reshape_inputs (per face batch):
        # [T * prognostics | T * decoder_inputs | constants | couplings]
        odd_in_vars = []
        odd_in_var_idx = []
        T = self.input_time_dim
        n_prog = self.input_channels
        n_di = self.decoder_input_channels

        for t in range(T):
            for v in odd_prog:
                odd_in_vars.append(v)
                odd_in_var_idx.append(t * n_prog + self.channels.index(v))

        # Decoder inputs (TISR etc.) are even — no indices.
        const_offset = T * (n_prog + n_di)
        if self.odd_constants is not None and self.constants is not None:
            for c in self.odd_constants:
                odd_in_vars.append(c)
                odd_in_var_idx.append(const_offset + self.constants.index(c))

        coup_offset = const_offset + self.n_constants
        if self.odd_coupled_variables and self.couplings:
            # Coupled block is concatenated as provided by coupler; mark listed vars if present
            coup_vars = []
            for c in self.couplings:
                for v in c["params"]["variables"]:
                    for _ in c["params"]["input_times"]:
                        coup_vars.append(v)
            for v in self.odd_coupled_variables:
                for i, name in enumerate(coup_vars):
                    if name == v:
                        odd_in_vars.append(v)
                        odd_in_var_idx.append(coup_offset + i)

        odd_in_var_idx_t = th.tensor(odd_in_var_idx, dtype=th.long) if odd_in_var_idx else None
        self.register_buffer("odd_in_var_idx", odd_in_var_idx_t, persistent=False)

        # Odd channels may use a spatial (path) clim mean that satisfies R(μ)=-μ;
        # only nonzero *scalar* means break normalize∘ρ = ρ∘normalize.
        odd_mean_vars = list(odd_in_vars) + odd_diag
        if self.scaling is not None and odd_mean_vars:
            for var in odd_mean_vars:
                if var in self.scaling and is_nonzero_scalar_mean(
                    self.scaling[var]["mean"]
                ):
                    raise ValueError(
                        f"Reflectional equivariance can only be enforced if all odd variables have zero "
                        f"scalar mean (or a spatial clim mean with R(μ)=-μ). "
                        f"Odd variable {var} has mean {self.scaling[var]['mean']}"
                    )

        if not odd_in_vars and not odd_diag:
            logger.warning(
                "Reflectional equivariance is enabled but no odd variables "
                "were specified. The model will be reflectionally equivariant "
                "only if all input variables are even scalars."
            )

        # Steerable layout: even channels first, then odd
        total_in = self._compute_input_channels()
        odd_set = set(odd_in_var_idx)
        even_idx = [i for i in range(total_in) if i not in odd_set]
        order = even_idx + odd_in_var_idx
        inverse = [0] * total_in
        for new_i, old_i in enumerate(order):
            inverse[old_i] = new_i
        self.register_buffer("structural_in_order", th.tensor(order, dtype=th.long), persistent=False)
        self.register_buffer("structural_in_inverse", th.tensor(inverse, dtype=th.long), persistent=False)
        self.structural_input_n_even = len(even_idx)

        # Output layout from decoder: T * output_channels in out_names order.
        # Prognostics use residual prediction; diagnostics stay absolute — parity typing only.
        out_ch = self._compute_output_channels()
        t_out = 1 if self.is_diagnostic else self.input_time_dim
        per = self.output_channels
        odd_out_names = odd_prog + odd_diag
        odd_out = []
        for t in range(t_out):
            for v in odd_out_names:
                if v not in out_names:
                    continue
                local = out_names.index(v)
                if local < per:
                    odd_out.append(t * per + local)
        # Averaged hpx_reflect on decoder outputs uses the full time-expanded odd indices.
        odd_out_var_idx = th.tensor(odd_out, dtype=th.long) if odd_out else None
        self.register_buffer("odd_out_var_idx", odd_out_var_idx, persistent=False)

        odd_out_set = set(odd_out)
        even_out = [i for i in range(out_ch) if i not in odd_out_set]
        out_order = even_out + odd_out
        out_inverse = [0] * out_ch
        for new_i, old_i in enumerate(out_order):
            out_inverse[old_i] = new_i
        self.register_buffer("structural_out_order", th.tensor(out_order, dtype=th.long), persistent=False)
        self.register_buffer("structural_out_inverse", th.tensor(out_inverse, dtype=th.long), persistent=False)
        self.structural_output_n_even = len(even_out)

    def _sin_lat_gates_for_batch(self, batch_faces: int) -> list | None:
        """sin_lat folded tensors at each encoder nside (shallow → deep).

        Returns ``None`` when no ReflectionSteerable layer sets ``needs_sin_lat``
        (default fast path: same-parity 1×1, no sin_lat buffers).
        """
        if not getattr(self, "_structural_needs_sin_lat_gate", False):
            return None
        gates = []
        for ns in self.nside:
            faces = getattr(self, f"sin_lat_nside_{ns}")
            gates.append(expand_sin_lat_folded(faces, batch_faces))
        return gates

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
            coupled = (
                inputs[3].permute(0, 2, 1, 3, 4)
                if self.couplings_time_first
                else inputs[3]
            )
            if self.coupled_partial_conv_stem is not None:
                # Stem expects [B, F, C, H, W]; preserves C so encoder channel math is unchanged.
                coupled = self.coupled_partial_conv_stem(coupled)
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
                coupled,
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

            # Save initial hidden states
            if self.enforce_reflectional_equivariance:
                orig_hidden_states = [
                    self.decoder.decoder[n].recurrent.h for n in range(len(self.decoder.decoder))
                ]
            
            # Forward the data through the model to initialize hidden states
            if self.reflection_equivariance_mode == "structural":
                input_tensor = apply_channel_order(input_tensor, self.structural_in_order)
                enc_gates = self._sin_lat_gates_for_batch(input_tensor.shape[0])
                dec_gates = list(reversed(enc_gates)) if enc_gates is not None else None
                self.decoder(
                    self.encoder(input_tensor, conditions_cln=conditions_cln, sin_lat_gate=enc_gates),
                    conditions_cln=conditions_cln,
                    sin_lat_gate=dec_gates,
                )
            else:
                self.decoder(self.encoder(input_tensor, conditions_cln=conditions_cln), conditions_cln=conditions_cln)

            if self.enforce_reflectional_equivariance:

                new_hidden_states = [
                    self.decoder.decoder[n].recurrent.h for n in range(len(self.decoder.decoder))
                ]

                # Reset hidden states to original
                for n in range(len(self.decoder.decoder)):
                    self.decoder.decoder[n].recurrent.h = \
                        self.hpx_reflect(orig_hidden_states[n], latent_tensor=True) \
                            if orig_hidden_states[n].shape != (1,1,1,1) \
                                else orig_hidden_states[n]

                # Forward through model with reflected input
                self.decoder(self.encoder(self.hpx_reflect(input_tensor, includes_constants=True), conditions_cln=conditions_cln), conditions_cln=conditions_cln)
                new_hidden_states_refl = [
                    self.decoder.decoder[n].recurrent.h for n in range(len(self.decoder.decoder))
                ]
                
                # Average of new hidden states resulting from the default forward
                # pass and the reflected forward pass
                for n in range(len(self.decoder.decoder)):
                    self.decoder.decoder[n].recurrent.h = 0.5 * (new_hidden_states[n] + self.hpx_reflect(new_hidden_states_refl[n], latent_tensor=True))

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
        if self.reset_cycle == float("inf") and self.training and not self._reset_cycle_inf_train_warned:
            warnings.warn(
                "reset_cycle=inf during training keeps GRU hidden state across "
                "forward() calls until reset() is invoked. Finite reset_cycle is "
                "the usual training setting; call reset() between independent "
                "samples if this persistence is intended.",
                UserWarning,
                stacklevel=2,
            )
            self._reset_cycle_inf_train_warned = True
        outputs = []
        for step in range(self.integration_steps):
            # (Re-)initialize recurrent hidden states
            hours = step * (self.delta_t * self.input_time_dim)
            if self.reset_cycle == float("inf"):
                # 0 % inf == 0, so the finite-cycle modulo would re-prime every
                # forward at step 0. Honor "never reset" after the first prime so
                # hidden state carries across coupler windows when inference sets
                # reset_cycle=inf (disable_*_recurrent_reset).
                need_init = not self._recurrent_hidden_primed
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

            # Save original hidden states for restoration later
            if self.enforce_reflectional_equivariance:
                orig_hidden_states = [
                    self.decoder.decoder[n].recurrent.h for n in range(len(self.decoder.decoder))
                ]

            # Forward through model, with or without conditions
            if conditions_cln is not None:
                kwargs = {"conditions_cln": conditions_cln[step]}
            else:
                kwargs = {}

            structural = self.reflection_equivariance_mode == "structural"
            input_for_residual = input_tensor
            if structural:
                input_tensor = apply_channel_order(input_tensor, self.structural_in_order)
                enc_gates = self._sin_lat_gates_for_batch(input_tensor.shape[0])
                dec_gates = list(reversed(enc_gates)) if enc_gates is not None else None
                encodings = self.encoder(input_tensor, sin_lat_gate=enc_gates, **kwargs)
                decodings = self.decoder(encodings, sin_lat_gate=dec_gates, **kwargs)
                decodings = apply_channel_order(decodings, self.structural_out_inverse)
            else:
                encodings = self.encoder(input_tensor, **kwargs)
                decodings = self.decoder(encodings, **kwargs)

            # Forward through model again with reflected input and original hidden states
            if self.enforce_reflectional_equivariance:

                new_hidden_states = [
                    self.decoder.decoder[n].recurrent.h for n in range(len(self.decoder.decoder))
                ]
                
                # Reset hidden states to original
                for n in range(len(self.decoder.decoder)):
                    self.decoder.decoder[n].recurrent.h = \
                        self.hpx_reflect(orig_hidden_states[n], latent_tensor=True) \
                            if orig_hidden_states[n].shape != (1,1,1,1) \
                                else orig_hidden_states[n]

                # Forward through model with reflected input
                input_tensor_refl = self.hpx_reflect(input_for_residual, includes_constants=True)
                encodings_refl = self.encoder(input_tensor_refl, **kwargs)
                decodings_refl = self.decoder(encodings_refl, **kwargs)               

                new_hidden_states_refl = [
                    self.decoder.decoder[n].recurrent.h for n in range(len(self.decoder.decoder))
                ]

                # Average of new hidden states resulting from the default forward
                # pass and the reflected forward pass
                for n in range(len(self.decoder.decoder)):
                    self.decoder.decoder[n].recurrent.h = 0.5 * (new_hidden_states[n] + self.hpx_reflect(new_hidden_states_refl[n], latent_tensor=True))

                # Average of decodings
                decodings = 0.5 * (decodings + self.hpx_reflect(decodings_refl, includes_constants=False))
            
            # Residual prediction
            combined = self._reshape_outputs(decodings) # [B*F, T*C, H, W] -> [B, F, T, C, H, W]
            prognostics = combined[:, :, :, :self.input_channels]
            orig_input = self._reshape_outputs(input_for_residual[:, : self.input_channels * self.input_time_dim])
            if self.residual_prediction:
                prognostics += orig_input
            diagnostics = combined[:, :, :, self.input_channels:]
            out = th.cat([prognostics, diagnostics], dim=3)

            # Apply constraints
            if self.constraints is not None:
                for constraint in self.constraints:
                    out = constraint(out, orig_input)

            outputs.append(out)

        if output_only_last:
            return outputs[-1]

        return th.cat(outputs, dim=self.channel_dim)

    def reset(self):
        """Reset encoder/decoder recurrent state.

        When ``reset_cycle`` is infinite, also clears the primed flag so the next
        ``forward()`` re-initializes hidden state. Call this between independent
        forecasts so GRU state does not carry from one initial condition to the next.
        """
        self.encoder.reset()
        self.decoder.reset()
        # Allow a later forward to re-prime when reset_cycle is inf.
        self._recurrent_hidden_primed = False
