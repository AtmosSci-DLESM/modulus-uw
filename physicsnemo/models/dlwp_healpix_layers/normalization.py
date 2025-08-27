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

from typing import Sequence, List

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