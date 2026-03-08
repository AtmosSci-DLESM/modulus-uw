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

from typing import List, Optional, Sequence, Tuple

import torch as th
import torch.nn.functional as F
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


class SPADE(th.nn.Module):
    """
    SPADE (Spatially-Adaptive Normalization): conditions on a spatial map Z to produce
    per-pixel scale (gamma) and shift (beta) via convolutions. Operates in 4D (N, C, H, W)
    for LayerNorm and final modulation; Z is unfolded to (B, F, C_cond, H, W) to add a
    learnable per-face embedding so gamma/beta can differ per HEALPix face (no single
    map forced to generalize across all faces). Zero-initialized for identity at init.

    Per-face modulation: face_embed (1, F, C_cond, 1, 1) is added to Z (after broadcast
    if Z is per-batch) so the conv sees different input per face and produces different
    gamma/beta per face. face_embed is initialized to zero so at init behavior is unchanged.

    Conditions contract (important for physics):
        - (B, C_cond, H, W): broadcast to all faces, then add face_embed so gamma/beta
          still differ per face. Use for global maps (e.g. solar); avoid with stochastic
          noise unless you want the same noise pattern tiled.
        - (N, C_cond, H, W) with N = B*n_faces: per-face conditions; face_embed adds
          optional face-specific adjustment on top.
    """

    def __init__(
        self,
        condition_shape: int,
        channel_depth: int,
        mlp_hidden_dims: List[int] = [128, 128],
        activation: th.nn.Module = None,
        eps: float = 1e-5,
        n_faces: int = 12,
        norm_op: str = "torch",
        scale_center: float = 1.0,
        epsilon_bound: Optional[float] = None,
        conv_hidden_channels: Optional[int] = None,
        init_spade_to_zero: bool = True,
    ):
        """
        Parameters
        ----------
        condition_shape : int
            Number of channels in the spatial condition map Z.
        channel_depth : int
            Number of channels in the normalized feature map.
        mlp_hidden_dims : List[int]
            Ignored if conv_hidden_channels is set. Otherwise first element is used as
            hidden channels for the shared conv (kept for config compatibility with CLN).
        conv_hidden_channels : int, optional
            Hidden channels for the shared 3x3 conv. If set, used instead of mlp_hidden_dims[0].
        activation : th.nn.Module
            Activation after the shared convolution.
        eps : float
            Passed to base LayerNorm.
        n_faces : int
            Number of HEALPix faces; used to broadcast Z when provided per-batch.
        norm_op : str
            "torch" or "apex" for base LayerNorm.
        scale_center : float
            Center for gamma (default 1.0 so init is identity).
        epsilon_bound : float, optional
            If set, gamma = scale_center + epsilon_bound * tanh(conv_out) to bound scale.
        init_spade_to_zero : bool
            If True (default), zero-initialize gamma and beta convs so at init the condition
            does not contribute (identity mapping); network learns to use the condition over
            training. Analogous to CLN's init_cln_to_zero. If False, convs use default init.

        A learnable per-face embedding (1, n_faces, condition_shape, 1, 1) is added to Z
        before the conv so gamma/beta can differ per HEALPix face; it is zero-initialized.
        """
        super().__init__()
        self.eps = eps
        self.condition_shape = condition_shape
        self.channel_depth = channel_depth
        if conv_hidden_channels is not None:
            hidden_channels = conv_hidden_channels
        else:
            hidden_channels = mlp_hidden_dims[0] if mlp_hidden_dims else 128
        self.activation = activation if activation is not None else th.nn.Identity()
        self.n_faces = n_faces
        self.scale_center = scale_center
        self.epsilon_bound = epsilon_bound

        self.shared_conv = th.nn.Conv2d(
            condition_shape, hidden_channels, kernel_size=3, padding=1
        )
        self.gamma_conv = th.nn.Conv2d(
            hidden_channels, channel_depth, kernel_size=3, padding=1
        )
        self.beta_conv = th.nn.Conv2d(
            hidden_channels, channel_depth, kernel_size=3, padding=1
        )

        # Optionally zero-init gamma/beta convs so condition does not contribute at init (identity).
        if init_spade_to_zero:
            self.gamma_conv.weight.data.zero_()
            self.gamma_conv.bias.data.zero_()
            self.beta_conv.weight.data.zero_()
            self.beta_conv.bias.data.zero_()

        if norm_op == "torch":
            self.norm = th.nn.LayerNorm(
                channel_depth, elementwise_affine=False, eps=eps
            )
        elif norm_op == "apex":
            if not _APEX_AVAILABLE:
                raise ImportError(
                    "Apex FusedLayerNorm requested but apex is not available"
                )
            self.norm = FusedLayerNorm(
                channel_depth, elementwise_affine=False, eps=eps
            )
        else:
            raise ValueError(f"norm_op must be 'torch' or 'apex', got {norm_op}")

        # Per-face embedding so conv input differs per face -> different gamma/beta per face.
        # Zero init: at init no change; network can learn face-specific modulation.
        self.face_embed = th.nn.Parameter(
            th.zeros(1, n_faces, condition_shape, 1, 1)
        )

    def forward(self, x: th.Tensor, conditions: th.Tensor) -> th.Tensor:
        """
        Normalize x and modulate with spatial gamma/beta from conditions. All in 4D (N, C, H, W).

        Parameters
        ----------
        x : th.Tensor
            (N, C, H, W) with N = B*n_faces (folded HEALPix).
        conditions : th.Tensor
            (B, C_cond, H_ref, W_ref) or (N, C_cond, H, W). If spatial size differs from x,
            Z is interpolated to x's size. Use (B, ...) only for deterministic global maps;
            for per-face stochastic noise pass (N, ...) so each face gets different content.

        Returns
        -------
        th.Tensor
            (N, C, H, W).
        """
        N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # 1. LayerNorm over channels (4D; no unfold needed)
        x = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (N, C, H, W)

        Z = conditions
        if Z.dim() == 4 and (Z.shape[-2] != H or Z.shape[-1] != W):
            Z = F.interpolate(
                Z,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # Broadcast (B, ...) -> (N, ...) when conditions are per-batch.
        if Z.shape[0] != N:
            Z = Z.repeat_interleave(self.n_faces, dim=0)

        # Add per-face embedding so conv sees different input per face -> different gamma/beta per face.
        B = N // self.n_faces
        n_f = self.n_faces
        Z = Z.view(B, n_f, self.condition_shape, H, W) + self.face_embed
        Z = Z.view(N, self.condition_shape, H, W)

        shared = self.activation(self.shared_conv(Z))
        gamma_raw = self.gamma_conv(shared)
        beta = self.beta_conv(shared)

        if self.epsilon_bound is not None:
            gamma = self.scale_center + self.epsilon_bound * th.tanh(gamma_raw)
        else:
            gamma = self.scale_center + gamma_raw

        return gamma * x_norm + beta


class AdditiveSpatialNoise(th.nn.Module):
    """
    Stochastic Decomposition Layer (SDL): Fout = Fin + α · (R ⊙ S ⊙ M).
    Operates in unfolded (B, F, C, H, W) space so M can learn per-face spatial modulation.

    Math (⊙ = element-wise, with broadcasting):
        Unfold: (N, C, H, W) → (B, F, C, H, W)  with N = B*F
        Z ∈ R^(B × noise_dim)   latent, sampled per-batch N(0,I) (global style per ensemble member)
        S = MLP(Z)             S ∈ R^(B × 1 × C × 1 × 1)   broadcast across faces (same globe-wide)
        R ~ N(0,I)             R ∈ R^(B × F × C × H × W)   per-pixel noise in unfolded space
        M (learnable)          M ∈ R^(1 × F × C × H × W)   per-face, per-point modulation (init zero)
        Fout_unf = Fin_unf + α · (R ⊙ S ⊙ M)
        Refold: (B, F, C, H, W) → (N, C, H, W)

    Z is sampled per-batch so the style S is global (no discontinuities at face seams).
    M is initialized in __init__ to spatial_size (H, W); do not re-assign parameters in forward.

    Tensor shapes (this layer works in unfolded 5D space):
        Component    Shape                  Description
        ---------   ----------------------  ------------------------------------------
        x (in)       (N, C, H, W)            Input folded (N = B*F)
        x_unf        (B, F, C, H, W)         Unfolded for SDL
        Z            (B, noise_dim)          Latent per-batch (global style)
        S            (B, 1, C, 1, 1)         Style from MLP(Z), broadcast across faces
        M            (1, F, C, H, W)        Learnable per-face per-point (init zero)
        R            (B, F, C, H, W)        Per-pixel Gaussian in unfolded space
        α            scalar                 Fixed scale
        output       (N, C, H, W)            Refolded, same as input layout
    """

    def __init__(
        self,
        channel_depth: int,
        noise_dim: int,
        spatial_size: Tuple[int, int],
        alpha: float = 0.1,
        mlp_hidden_dims: Optional[List[int]] = None,
        num_faces: int = 12,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        channel_depth : int
            Number of channels (C). S and M have channel dimension C.
        noise_dim : int
            Dimension of latent Z. Z ~ N(0, I) is sampled per-batch each forward and passed through MLP -> S.
        spatial_size : Tuple[int, int]
            (H, W) spatial dimensions for this decoder level. M is initialized to (1, F, C, H, W)
            and must not be re-assigned in forward (optimizer and DDP require fixed parameters).
        alpha : float
            Fixed scalar scaling the noise magnitude (not learned).
        mlp_hidden_dims : list of int, optional
            Hidden layer sizes for MLP mapping Z -> S. Defaults to [64, 64].
        num_faces : int
            Number of HEALPix faces (F). Used to unfold/refold; must match model convention (default 12).
        seed : int, optional
            If set, use a module-local generator for reproducible Z and R. If None, use default RNG.
        """
        super().__init__()
        self.channel_depth = channel_depth
        self.noise_dim = noise_dim
        self.alpha = alpha
        self.num_faces = num_faces
        self._seed = seed
        self._generator = None

        H, W = spatial_size
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [64, 64]
        self.mlp = self._make_mlp(noise_dim, mlp_hidden_dims, channel_depth)
        # M: (1, F, C, H, W) fixed at init so optimizer and DDP keep a valid reference
        self.M = th.nn.Parameter(th.zeros(1, num_faces, channel_depth, H, W))

    def _make_mlp(
        self, in_dim: int, hidden_dims: List[int], out_dim: int
    ) -> th.nn.Module:
        layers = []
        for h in hidden_dims:
            layers.append(th.nn.Linear(in_dim, h))
            layers.append(th.nn.GELU())
            in_dim = h
        layers.append(th.nn.Linear(in_dim, out_dim))
        return th.nn.Sequential(*layers)

    def _get_generator(self, device: th.device) -> Optional[th.Generator]:
        if self._seed is None:
            return None
        if self._generator is None:
            self._generator = th.Generator(device=device)
            self._generator.manual_seed(self._seed)
        elif self._generator.device != device:
            # Generator has no .to(device); create a new one on the target device
            seed = self._generator.initial_seed()
            self._generator = th.Generator(device=device)
            self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        x : th.Tensor
            Input of shape (N, C, H, W) with N = B*F in HEALPix folded format.

        Returns
        -------
        th.Tensor
            Fout = Fin + α · (R ⊙ S ⊙ M), refolded to (N, C, H, W).
        """
        n_f = self.num_faces
        N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        B = N // n_f
        # Unfold: (N, C, H, W) -> (B, n_f, C, H, W) (view only)
        x_unf = x.view(B, n_f, C, H, W)
        M = self.M
        gen = self._get_generator(x.device)
        # Z per-batch (global style) so S is identical across all faces (no seam discontinuities)
        Z = th.randn(B, self.noise_dim, device=x.device, dtype=x.dtype, generator=gen)
        S = self.mlp(Z).view(B, 1, self.channel_depth, 1, 1)  # broadcast across faces
        R = th.randn(
            B, n_f, C, H, W,
            generator=gen, device=x.device, dtype=x.dtype,
        )
        out_unf = x_unf + self.alpha * (R * S * M)
        # Refold: (B, n_f, C, H, W) -> (N, C, H, W) (view only)
        return out_unf.view(N, C, H, W)