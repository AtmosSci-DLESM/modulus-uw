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

class ConditionalLayerNorm(th.nn.Module):
    def __init__(
        self,
        condition_shape: int,
        batch_size: int,
        mlp_hidden_dims: List[int] = [128, 128],
        activation: th.nn.Module = None,
        eps: float = 1e-5,
        n_faces: int = 12,
    ):
        """
        Conditional LayerNorm with MLP-based conditioning.

        Args:
            condition_shape (int): Shape of the conditioning input.
            batch_size (int): Size of the batch.
            mlp_hidden_dims (List[int]): Hidden layer sizes for MLPs predicting gamma and beta.
            activation (DictConfig): Activation function configuration for the MLPs.
            eps (float): Numerical stability constant.
            n_faces (int): Number of faces in the Healpix grid, used for reshaping.
        """
        super().__init__()
        self.eps = eps
        self.condition_shape = condition_shape
        self.hidden_dims = mlp_hidden_dims
        self.batch_size = batch_size
        self.activation = activation if activation is not None else th.nn.Identity()
        self.gamma_mlp = None
        self.beta_mlp = None
        self.gamma = None
        self.beta = None
        self.n_faces = n_faces
        # 
        self.initialized = False

    def setup_cln(self, channel_depth):
        """
        Set up the Conditional Layer Normalization with MLPs. Uses context dependent parameters.

        Args:
            normalized_shape (int): The number of channels (C) to normalize across.
        """
        # Create MLPs to produce gamma and beta from the cond input
        self.gamma_mlp = self._make_mlp(self.condition_shape, self.hidden_dims, channel_depth, self.activation)
        self.beta_mlp = self._make_mlp(self.condition_shape, self.hidden_dims, channel_depth, self.activation)
    
    def _make_mlp(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: th.nn.Module) -> th.nn.Sequential:

        layers = []
        for hdim in hidden_dims:
            layers.append(th.nn.Linear(in_dim, hdim))
            if activation:
                layers.append(activation)
            in_dim = hdim
        layers.append(th.nn.Linear(in_dim, out_dim))
        return th.nn.Sequential(*layers)

    def _initialze(self, device):

        """
        Initialize the Conditional LayerNorm by moving MLPs to the specified device.

        Args:
            device (torch.device): The device to which the MLPs should be moved.
        """
        self.gamma_mlp.to(device)
        self.beta_mlp.to(device)
        self.initialized = True

    def forward(self, x: th.Tensor, conditions: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input tensor of shape: 
            conditions: Conditioning tensor of shape (B, N, cond_dim)
        Returns:
            Normalized and conditioned tensor of shape: 
        """

        # check if the layer is initialized
        if not self.initialized:
            self._initialze(x.device)

        # normal layer norm, assumes channel-last format
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / th.sqrt(var + self.eps)
    
        # dynamically build gamma and beta, don't store in self
        gammas = []
        betas = []

        for i in range(self.batch_size):
            gamma = self.gamma_mlp(conditions[i])
            beta = self.beta_mlp(conditions[i])
            gammas.append(gamma)
            betas.append(beta)

        # add singleton dimensions for H, W, repeat batch dimesnsion n_faces times: these two
        # dimensions are folded together
        gamma = th.stack(gammas, dim=0)[:,None,None,:].repeat_interleave(self.n_faces, dim=0)
        beta = th.stack(betas, dim=0)[:,None,None,:].repeat_interleave(self.n_faces, dim=0)

        # now gamma and beta are clean tensors that participate in autograd safely
        return gamma * x_norm + beta