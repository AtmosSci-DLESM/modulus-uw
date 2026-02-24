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

from typing import Sequence, List, Tuple

import torch as th
from omegaconf import DictConfig

try:
    from apex.normalization import FusedLayerNorm
    _APEX_AVAILABLE = True
except ImportError:
    _APEX_AVAILABLE = False


class ConditionalLayerNorm(th.nn.Module):
    def __init__(
        self,
        condition_shape: int,
        channel_depth: int,
        mlp_hidden_dims: List[int] = [128, 128],
        activation: th.nn.Module = None,
        eps: float = 1e-5,
        n_faces: int = 12,
        norm_op:str = "torch",
        init_cln_to_zero: bool = False,
        scale_center: float = 0.0,
    ):
        """
        Conditional LayerNorm with MLP-based conditioning.

        Parameters
        ----------
        condition_shape : int
            Shape of the conditioning input.
        channel_depth : int
            Number of channels in the input tensor.
        mlp_hidden_dims : List[int]
            Hidden layer sizes for MLPs predicting gamma and beta.
        activation : DictConfig
            Activation function configuration for the MLPs.
        eps : float
            Numerical stability constant.
        n_faces : int
            Number of faces in the Healpix grid, used for reshaping.
        norm_op : str
            "torch" for torch.nn.LayerNorm, "apex" for apex FusedLayerNorm.
        init_cln_to_zero : bool = False
            If True, initialize the last layer of the MLPs to zero.
            At the start of training, the noise will be ignored
        scale_center : float = 0.0
            Center of the scale parameter. Set to 1.0 and use `init_cln_to_zero=True`
            to make CLN behave like standard LayerNorm at initialization.
        """
        super().__init__()
        self.eps = eps
        self.condition_shape = condition_shape
        self.channel_depth = channel_depth
        self.hidden_dims = mlp_hidden_dims
        self.activation = activation if activation is not None else th.nn.Identity()
        self.gamma_mlp = self._make_mlp(self.condition_shape, self.hidden_dims, self.channel_depth, self.activation)
        self.beta_mlp = self._make_mlp(self.condition_shape, self.hidden_dims, self.channel_depth, self.activation)
        self.n_faces = n_faces
        self.scale_center = scale_center

        if init_cln_to_zero:
            self.gamma_mlp[-1].weight.data.zero_()
            self.beta_mlp[-1].weight.data.zero_()
            self.gamma_mlp[-1].bias.data.zero_()
            self.beta_mlp[-1].bias.data.zero_()

        if norm_op == "torch":
            self.norm = th.nn.LayerNorm(channel_depth, elementwise_affine=False)
        elif norm_op == "apex":
            if not _APEX_AVAILABLE:
                raise ImportError("Apex FusedLayerNorm requested but apex is not available, please install it from https://github.com/NVIDIA/apex")
            self.norm = FusedLayerNorm(channel_depth, elementwise_affine=False)

    def _make_mlp(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: th.nn.Module) -> th.nn.Sequential:

        layers = []
        for hdim in hidden_dims:
            layers.append(th.nn.Linear(in_dim, hdim))
            if activation:
                layers.append(activation)
            in_dim = hdim
        layers.append(th.nn.Linear(in_dim, out_dim))
        return th.nn.Sequential(*layers)


    def forward(self, x: th.Tensor, conditions: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        x : th.Tensor
            Input tensor of shape: (B, C, H, W)
        conditions : th.Tensor
            Conditioning tensor of shape (B*n_cond, cond_dim)

        Returns
        -------
        th.Tensor
            Normalized and conditioned tensor of shape: (B, C, H, W)
        """

        # normal layer norm without default scale, bias applied (permute to channels_last)
        x = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x)
    
        # Compute gamma and beta from conditions
        gamma = self.scale_center + self.gamma_mlp(conditions)[:, None, None, :] # (B*n_cond, 1, 1, C)
        beta = self.beta_mlp(conditions)[:, None, None, :] # (B*n_cond, 1, 1, C)

        # Repeat for the number of faces(which has been folded into the batch dimension)
        gamma = gamma.repeat_interleave(self.n_faces, dim=0)
        beta = beta.repeat_interleave(self.n_faces, dim=0)

        # Apply and reshape to channels first
        x = gamma * x_norm + beta
        return x.permute(0, 3, 1, 2)

class AdaLNZero(th.nn.Module):
    def __init__(
        self,
        condition_shape: int,
        channel_depth: int,
        mlp_hidden_dims: List[int] = [128, 128],
        activation: th.nn.Module = None,
        eps: float = 1e-5,
        n_faces: int = 12,
        norm_op: str = "torch",
        scale_center: float = 1.0, # Default to 1 for identity scaling
        gate_channel_depth: int = None, # If None, uses channel_depth (backward compatible)
    ):
        """
        AdaLN-Zero: Adaptive Layer Norm with Zero Initialization.
        
        Predicts scale (gamma), shift (beta), and a gate (alpha).
        Initialized such that gamma=1, beta=0, alpha=0.
        
        Parameters
        ----------
        condition_shape : int
            Shape of the conditioning input.
        channel_depth : int
            Number of channels in the input tensor (for normalization/modulation).
        gate_channel_depth : int, optional
            Number of channels for the gate. If None, uses channel_depth.
            Use this when gate needs to match output channels while normalization
            matches input channels.
        mlp_hidden_dims : List[int]
            Hidden layer sizes for MLPs.
        activation : th.nn.Module
            Activation function configuration for the MLPs.
        eps : float
            Numerical stability constant.
        n_faces : int
            Number of faces in the Healpix grid.
        norm_op : str
            "torch" for torch.nn.LayerNorm, "apex" for apex FusedLayerNorm.
        scale_center : float
            Center of the scale parameter.
        """
        super().__init__()
        self.eps = eps
        self.condition_shape = condition_shape
        self.channel_depth = channel_depth
        self.gate_channel_depth = gate_channel_depth if gate_channel_depth is not None else channel_depth
        self.hidden_dims = mlp_hidden_dims
        self.activation = activation if activation is not None else th.nn.Identity()
        self.n_faces = n_faces
        self.scale_center = scale_center

        # MLP for modulation (scale/shift) - uses input channel depth
        self.adaLN_modulation = self._make_mlp(
            self.condition_shape, 
            self.hidden_dims, 
            2 * self.channel_depth + self.gate_channel_depth, 
            self.activation
        )

        # ZERO INITIALIZATION (Crucial for AdaLN-Zero)
        # We initialize the last layer weights and bias to zero.
        # This results in the MLP outputting exactly 0 at step 0.
        self.adaLN_modulation[-1].weight.data.zero_()
        self.adaLN_modulation[-1].bias.data.zero_()

        if norm_op == "torch":
            self.norm = th.nn.LayerNorm(channel_depth, elementwise_affine=False, eps=eps)
        elif norm_op == "apex":
            if not _APEX_AVAILABLE:
                raise ImportError("Apex requested but not available")
            self.norm = FusedLayerNorm(channel_depth, elementwise_affine=False, eps=eps)

    def _make_mlp(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: th.nn.Module) -> th.nn.Sequential:
        layers = []
        for hdim in hidden_dims:
            layers.append(th.nn.Linear(in_dim, hdim))
            if activation:
                layers.append(activation)
            in_dim = hdim
        layers.append(th.nn.Linear(in_dim, out_dim))
        return th.nn.Sequential(*layers)

    def forward(self, x: th.Tensor, conditions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns
        -------
        x : th.Tensor
            Normalized and shifted input: x = x * gamma + beta
        gate : th.Tensor
            The zero-initialized gate to be applied to the residual block output.
        """
        # 1. Standard Layer Norm (Channel-wise)
        # Permute to (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x)

        # 2. Predict Modulation Parameters (Gamma, Beta, Alpha)
        # Shape: (B*n_cond, 2*input_C + gate_C)
        embedding = self.adaLN_modulation(conditions)
        
        # Reshape to (B*n_cond, 1, 1, 2*input_C + gate_C) for broadcasting
        embedding = embedding[:, None, None, :]
        
        # Split into shift(beta), scale(gamma), and gate(alpha)
        # shift and scale use input channel depth, gate uses gate_channel_depth
        shift = embedding[..., :self.channel_depth]
        scale = embedding[..., self.channel_depth:2*self.channel_depth]
        gate = embedding[..., 2*self.channel_depth:]
        
        # Apply scale center (typically 1.0) so effective scale is 1.0 at init
        # (Since MLP output is 0, scale becomes 0 + 1 = 1)
        scale = scale + self.scale_center

        # 3. Handle Healpix/Grid Dimensions
        # Repeat for the number of faces (folded into batch dim)
        scale = scale.repeat_interleave(self.n_faces, dim=0)
        shift = shift.repeat_interleave(self.n_faces, dim=0)
        gate = gate.repeat_interleave(self.n_faces, dim=0)

        # 4. Modulate
        # Note: We do NOT apply the gate here. The gate is applied *after* the convolution.
        x = x_norm * scale + shift
        
        # Return permuted x (B, C, H, W) AND the gate (B, C, H, W)
        return x.permute(0, 3, 1, 2), gate.permute(0, 3, 1, 2)