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

from typing import Sequence, Optional

import math
import numpy as np
import torch as th
import xarray as xr

import earth2grid
from cuhpx import SHTCUDA, iSHTCUDA
from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder

"""
Custom dlwp compatible loss classes that allow for more sophisticated training optimization.

Each custom loss should inherit all methods of th.nn._Loss base class or subclasses thereof. 
Additionally, custom loss classes should define a setup function which receives the trainer object. 
The setup function should be used to move tensors to appropriate gpus and finalize configuration
of the loss calculation using information about the model (trainer.model) and trainer. Custom
losses should also redefine the forward function to contain a flag indicating whether or not to 
average output channels. This is used in the varible wise logging of validation loss by the trainer. 

"""


class BaseMSE(th.nn.MSELoss):
    """
    Base MSE class offers impementaion for basic MSE loss compatable with dlwp custom loss training
    """

    def __init__(
        self,
    ):
        """Constructer for BaseMSE"""
        super().__init__()
        self.device = None

    def setup(self, trainer):
        """
        Nothing to implement here
        """
        pass

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the base MSE class
        Tensors are expected to be in the shape [N, B, F, C, H, W]

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        if not (prediction.ndim == 6 and target.ndim == 6):
            raise AssertionError("Expected predictions to have 6 dimensions")

        d = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5))
        if average_channels:
            return th.mean(d)
        else:
            return d


class WeightedMSE(th.nn.MSELoss):

    """
    Loss object that allows for user defined weighting of variables when calculating MSE
    """

    def __init__(
        self,
        weights: Sequence = [],
    ):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        self.device = None

    def setup(self, trainer):
        """
        pushes weights to cuda device
        """

        if len(trainer.output_variables) != len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedMSE pass
        Tensors are expected to be in the shape [N, B, F, C, H, W]

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        if not (prediction.ndim == 6 and target.ndim == 6):
            raise AssertionError("Expected predictions to have 6 dimensions")

        d = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5)) * self.loss_weights
        if average_channels:
            return th.mean(d)
        else:
            return d

class ConditionalWeightLoss( th.nn.MSELoss ):
    """
    Conditional loss for precipitation diagnostic model.
    (Total 6hr precipitation is the only output field.)
    """

    def __init__(
        self,
        weight=(0.01,1.0),
        b=None,
        w=1,
        ):
        """
        Parameters
        -----------
        weight: tuple of floats
            weight[0] is used when the target precipitation value is zero
            weight[1] is used for all non-zero precipitation
        b: float
            Exponential scaling factor used to define weighting curve for non-zero precip. weights
        w: float
            Final scaling factor applied to loss.
        """

        super().__init__()
        self.weight_zero = weight[0]
        self.weight_nonzero = weight[1]
        self.b = b
        self.device = None
        self.w = w

    def setup(self, trainer):
        self.b = th.tensor(self.b, device=trainer.device)
        self.w = th.tensor(self.w, device=trainer.device)

    def forward(self, prediction, target):
        """
        Computes the MSE of model prediction and applies weights for zero and non-zero precipitation cases.

        Parameters
        -----------
        prediction: torch.tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        """
        weights_for_zero = th.ones_like(target) * self.weight_zero
        weights_for_nonzero = (th.ones_like(target) * self.weight_nonzero) * th.exp(self.b*target)
        weights = th.where(target > 0, weights_for_nonzero, weights_for_zero)
        loss = (th.mean(weights * (prediction - target) ** 2))*self.w
        return loss

class OceanMSE(th.nn.MSELoss):
    """
    Ocean MSE class offers impementaion for MSE loss weighted by a land-sea-mask field.
    """

    def __init__(
        self,
        lsm_file: str,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "lsm"},
    ):
        """
        Parameters
        ----------
        lsm_file: str
            land-sea-mask file
        open_dict: dict, optional
            dictionary that store land-sea-mask file information
        selection_dict: dict, optional
            dictionary that store channel selection information
        """
        super().__init__()
        self.device = None
        self.lsm_file = lsm_file
        self.lsm_ds = None
        self.open_dict = open_dict
        self.selection_dict = selection_dict
        self.lsm_tensor = None
        self.lsm_sum_calculated = False
        self.lsm_sum = None
        self.lsm_var_sum = None

    def setup(self, trainer):
        """
        reshape lsm and put on device
        """
        self.lsm_ds = xr.open_dataset(self.lsm_file, **self.open_dict).constants.sel(
            self.selection_dict
        )
        # 1-lsm gives the percentage of pixel that has ocean
        self.lsm_tensor = 1 - th.tensor(
            np.expand_dims(self.lsm_ds.values, (0, 2, 3))
        ).to(trainer.device)

    def forward(self, prediction, target, average_channels=True):

        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum(
                dim=(0, 1, 2, 4, 5)
            )
            self.lsm_sum_calculated = True
        # average weighted
        ocean_err = ((target - prediction) ** 2) * self.lsm_tensor
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        if average_channels:
            return th.sum(ocean_mean_err) / self.lsm_sum
        else:
            return ocean_mean_err / self.lsm_var_sum


class WeightedOceanMSE(th.nn.MSELoss):
    """
    Ocean MSE class offers impementaion for MSE loss with:
    1) weighted by a land-sea-mask field.
    2) weighted by channel (e.g. sic more than sst)
    """

    def __init__(
        self,
        lsm_file: str,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "lsm"},
        weights: Sequence = [],
    ):
        """ """
        super().__init__()
        self.device = None
        self.lsm_file = lsm_file
        self.lsm_ds = None
        self.open_dict = open_dict
        self.selection_dict = selection_dict
        self.lsm_tensor = None
        self.lsm_sum_calculated = False
        self.lsm_sum = None
        self.lsm_var_sum = None
        self.loss_weights = th.tensor(weights)

    def setup(self, trainer):
        """
        reshape lsm and put on device
        pushes weights to cuda device
        """
        ### 1. OCEAN PREP ###
        self.lsm_ds = xr.open_dataset(self.lsm_file, **self.open_dict).constants.sel(
            self.selection_dict
        )
        # 1-lsm gives the percentage of pixel that has ocean
        self.lsm_tensor = 1 - th.tensor(
            np.expand_dims(self.lsm_ds.values, (0, 2, 3))
        ).to(trainer.device)

        ### 2. WEIGHTS PREP ###

        if not len(trainer.output_variables) == len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)

    def forward(self, prediction, target, average_channels=True):

        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum(
                dim=(0, 1, 2, 4, 5)
            )
            self.lsm_sum_calculated = True
        # average weighted
        ocean_err = ((target - prediction) ** 2) * self.lsm_tensor
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        ocean_mean_err = ocean_mean_err * self.loss_weights

        if average_channels:
            return th.sum(ocean_mean_err) / self.lsm_sum
        else:
            return ocean_mean_err / self.lsm_var_sum

class WeightedCRPSLoss(th.nn.MSELoss):

    """
    Probabilistic loss function that allows for user defined weighting of variables when calculating CRPS.
    """

    def __init__(
        self,
        weights: Sequence = [],
        n_members: int = 2,
        alpha: float = 0.95,
        mean_penalty: float = 0.0,
        lsm_file: str = None,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "land_sea_mask"},
        lsm_binary_mask: bool = False,
        lsm_binary_threshold: float = 0.5,
        multiscale: float = 0.0,
        masked_pool: bool = False,
        temporal_dt: float = 0.0,
    ):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        n_members: int
            number of ensemble members in the model output
        alpha: float
            hyperparamter for approximating fair CRPS loss. between 0 and 1, 1 corresponds to a fair CRPS loss.
        mean_penalty: float
            weight for the penalty constraining the global mean of the ensemble to be close to the target mean
            if 0, no penalty is applied
        lsm_file: str
            land-sea-mask file. When provided, lsm_tensor weights the loss per grid cell.
        open_dict: dict, optional
            dictionary that store land-sea-mask file information
        selection_dict: dict, optional
            dictionary that store channel selection information
        lsm_binary_mask: bool, optional
            If False (default), lsm_tensor = 1 - land_fraction (continuous mask, loss weighted by ocean fraction).
            If True, use a binary mask: land < lsm_binary_threshold → weight 1, land >= lsm_binary_threshold → weight 0.
            With default threshold 0.5, matches infill logic in ocean_land_infill (land_mask >= 0.5 → land).
        lsm_binary_threshold: float, optional
            Float in [0, 1], default 0.5. When lsm_binary_mask is True, grid cells with land fraction
            < lsm_binary_threshold contribute 1 to the loss; cells with land >= lsm_binary_threshold
            contribute 0. Ignored when lsm_binary_mask is False.
        multiscale: float, optional
            weight for the multiscale CRPS loss. Default is 0, no multiscale loss is applied.
        masked_pool: bool, optional
            if True, spatial pooling uses only ocean pixels (land ignored). When using
            land masking/infilling with multiscale, set masked_pool=True so land and
            infilled values do not contribute to spatial averages.
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        if n_members < 2:
            raise ValueError("n_members must be at least 2 for CRPS loss to be defined")
        else:    
            self.n_members = n_members
        self.device = None
        self.mean_penalty = mean_penalty
        self.multiscale = multiscale
        self.masked_pool = masked_pool
        self.temporal_dt = temporal_dt
        self.scales = [4, 16, 32]

        if lsm_file is not None:
            self.lsm_ds = xr.open_dataset(lsm_file, **open_dict).constants.sel(selection_dict)
            lsm_values = np.expand_dims(self.lsm_ds.values, (0, 2, 3))
            if lsm_binary_mask:
                # Binary mask: 1 where land < threshold (cell contributes to loss), 0 where land >= threshold
                lsm_binary = (lsm_values < lsm_binary_threshold).astype(np.float32)
                self.lsm_tensor = th.tensor(lsm_binary)
            else:
                # Continuous mask: 1 - land_fraction (ocean fraction), loss weighted by ocean
                self.lsm_tensor = 1 - th.tensor(lsm_values.astype(np.float32))
        else:
            self.lsm_tensor = th.ones(1, 1, 1, 1, 1, 1) # Spoof the tensor dimensions for broadcasting

        # Parameters for "almost fair CRPS" loss. See https://arxiv.org/html/2412.15832v1
        self.coeff_eps = 1 - ((1-alpha) / (n_members))
        self.averaging_coeff = 1 / (2* n_members * (n_members - 1))

        # For n>2, will use pairwise distance to copmute [NxN] distance matrix
        # Diagonal elements of (prediciton - target) matrix are zeroed out to avoid double counting
        self.pdist = th.nn.PairwiseDistance(p=1)
        self.diag_mask = th.ones(self.n_members, self.n_members) - th.eye(self.n_members) # Mask to zero out diagonal elements

    def setup(self, trainer):
        """
        pushes constants to cuda device
        """

        if len(trainer.output_variables) != len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)
        self.averaging_coeff = th.tensor(self.averaging_coeff, device=trainer.device)
        self.coeff_eps = th.tensor(self.coeff_eps, device=trainer.device)
        self.pdist = self.pdist.to(device=trainer.device)
        self.diag_mask = self.diag_mask.to(device=trainer.device)   
        self.lsm_tensor = self.lsm_tensor.to(device=trainer.device)

    def _2member_crps(self, prediction, target, lsm_tensor):
        diff_target = th.abs(prediction - target.unsqueeze(0)).sum(dim=0) # [B, F, T, C, H, W]
        diff_ensemble = th.abs(prediction[0] - prediction[1]) # [B, F, T, C, H, W]
        # multiply by 2 to account for the fact that we are using a 2-member ensemble
        crps = self.averaging_coeff*(diff_target - self.coeff_eps * diff_ensemble) # [B, F, T, C, H, W]
        crps *= lsm_tensor
        return crps

    def _pool(self, tensor, scale):
        shape = tensor.shape
        h, w = shape[-2:]
        pooled = th.nn.functional.avg_pool2d(tensor.reshape(shape[0], -1, h, w), scale, scale)
        return pooled.reshape(*shape[:-2], h//scale, w//scale)
    
    def _masked_pool(self, tensor, mask, scale):
        """
        Pools a tensor while ignoring masked values (land).
        Returns:
            valid_avg: The average of only the VALID pixels in the window.
            pooled_mask: The fraction of valid pixels in the window (used for weighting).
        """
        # 1. Zero out invalid (land) pixels so they don't corrupt the sum
        masked_tensor = tensor * mask
        
        # 2. Pool the values (Calculate: Sum / Total_Pixels)
        num = self._pool(masked_tensor, scale)
        
        # 3. Pool the mask (Calculate: Valid_Pixels / Total_Pixels)
        denom = self._pool(mask, scale)
        
        # 4. Divide to get true average: Sum / Valid_Pixels
        # We add epsilon to avoid division by zero in fully land blocks
        valid_avg = num / (denom + 1e-6)
        
        return valid_avg, denom
    
    def _calculate_dt_loss(self, prediction, target, average_channels=True):
        """
        Calculates the CRPS of the temporal gradient (X_t+1 - X_t).
        Expects prediction and target to be already weighted.
        """
        if target.shape[2] < 2:
            return th.tensor(0.0, device=prediction.device)

        # 1. Calculate gradients: X(t+1) - X(t)
        # Slicing [1:2] keeps the T dim as 1 for broadcasting with lsm_tensor
        pred_dt = prediction[:, :, :, 1:2, ...] - prediction[:, :, :, 0:1, ...] 
        tar_dt = target[:, :, 1:2, ...] - target[:, :, 0:1, ...]

        # 2. Calculate CRPS on the delta
        # We pass self.lsm_tensor to mask land values in the gradient calculation
        if self.n_members == 2:
            crps_dt = self._2member_crps(pred_dt, tar_dt, self.lsm_tensor)
            
            if average_channels:
                return crps_dt.mean()
            else:
                return crps_dt.mean(dim=(0, 1, 2, 4, 5))
        
        else:
            # Fallback for N > 2 (Pairwise distance)
            # We reuse the logic but applied to the difference tensors
            b, f, t, c, h, w = tar_dt.shape
            
            if not average_channels:
                # Permute to [C, N, B, F, T, H, W]
                p_dt = pred_dt.permute(4, 0, 1, 2, 3, 5, 6).reshape(c, self.n_members, -1)
                t_dt = tar_dt.permute(3, 0, 1, 2, 4, 5).unsqueeze(1).reshape(c, 1, -1)

                diff = self.pdist(p_dt, t_dt) 
                dist_matrix = self.pdist(p_dt.unsqueeze(1), p_dt.unsqueeze(2))
                
                diff_terms = self.diag_mask[None, ...] * (diff.unsqueeze(1) + diff.unsqueeze(2))
                loss = self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum(dim=(1,2))
                return loss / (b * f * t * h * w)
            else:
                p_dt = pred_dt.reshape(self.n_members, -1)
                t_dt = tar_dt.unsqueeze(0).reshape(1, -1)
                
                diff = self.pdist(p_dt, t_dt)
                dist_matrix = self.pdist(p_dt.unsqueeze(1), p_dt.unsqueeze(0))

                diff_terms = self.diag_mask * (diff.unsqueeze(0) + diff.unsqueeze(1))
                loss = self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum()
                return loss / (b * f * c * t * h * w)

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedCRPSLoss 
        Computes the CRPS loss for the model prediction and target.

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor shape [Cond*B, F, T, C, H, W] where Cond is the number of ensemble members
        target: torch.Tensor
            The target tensor shape [B, F, T, C, H, W]
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        
        # Unfold ensemble dimension from batch dimension to have shape [Cond, B, F, T, C, H, W]
        b, f, t, c, h, w = target.shape
        prediction = prediction.view(self.n_members, b, f, t, c, h, w)

        # checks for dimensions 
        if not prediction.shape[1:] == target.shape:
            raise ValueError(f"Shape of prediction should match shape of target along non-ensemble dimensions, got {prediction.shape} and {target.shape}")
    
        if not prediction.shape[0] == self.n_members:
            raise ValueError(f"Shape of prediction should have ensemble dimension of size {self.n_members}, got {prediction.shape[0]}")

        n = self.n_members
        
        # Manual Cast
        prediction = prediction.to(th.float32)
        target = target.to(th.float32)
        
        # Apply channel weights across channel dims
        prediction *= self.loss_weights[None, None, None, None, :, None, None]
        target *= self.loss_weights[None, None, None, :, None, None]

        if n == 2:
            # Use faster explicit implementation
            crps = self._2member_crps(prediction, target, self.lsm_tensor)

            if average_channels:
                loss = crps.mean()
            else:
                loss = crps.mean(dim=(0, 1, 2, 4, 5))

            # Average Global Mean Bias Penalty (masked by lsm so land is ignored when weights are 0)
            if self.mean_penalty > 0:
                lsm = self.lsm_tensor  # [1, 1, 1, 1, H, W] or all ones if no lsm_file
                # Masked mean over ocean only: sum(x*lsm) / sum(lsm) over the reduced dims
                n_ens, n_b, n_f, n_t = prediction.shape[0], prediction.shape[1], prediction.shape[2], prediction.shape[3]
                ocean_count = n_ens * n_b * n_f * n_t * lsm.sum()
                ens_global_means = (prediction * lsm).sum(dim=(0, 1, 2, 3, 5, 6)) / (ocean_count + 1e-8)  # [C]
                target_ocean_count = n_b * n_f * n_t * lsm.sum()
                target_global_means = (target * lsm).sum(dim=(0, 1, 2, 4, 5)) / (target_ocean_count + 1e-8)  # [C]
                bias_penalty = self.mean_penalty * th.abs(ens_global_means - target_global_means)
                if average_channels:
                    loss += bias_penalty.mean()
                else:
                    loss += bias_penalty

            # Spatial Pooling Loss
            if self.multiscale > 0.:
                crps_scales = 0
                for scale in self.scales:
                    if self.masked_pool:
                        pred, mask_pooled = self._masked_pool(prediction, self.lsm_tensor, scale)
                        tar, _ = self._masked_pool(target, self.lsm_tensor, scale)
                        crps_scale = self._2member_crps(pred, tar, mask_pooled)
                    else:
                        pred, tar, lsm = self._pool(prediction, scale), self._pool(target, scale), self._pool(self.lsm_tensor, scale)
                        crps_scale = self._2member_crps(pred, tar, lsm)

                    if average_channels:
                        crps_scale = crps_scale.mean()
                    else:
                        crps_scale = crps_scale.mean(dim=(0, 1, 2, 4, 5))
                    crps_scales += crps_scale

                crps_scales = crps_scales / len(self.scales)
                loss += self.multiscale * crps_scales
            
            # Temporal Dt Loss (Xt - Xt-1)
            if self.temporal_dt > 0.:
                dt_loss = self._calculate_dt_loss(prediction, target, average_channels)
                loss += self.temporal_dt * dt_loss
            
                
            return loss
        else:
            # Use pairwise distance method
            if not average_channels:
                # Move channels to first dimension and exclude that dimension from the reductions           
                prediction = prediction.permute(4, 0, 1, 2, 3, 5, 6) # [C, Cond, B, F, T, H, W]
                target = target.permute(3, 0, 1, 2, 4, 5) # [C, B, F, T, H, W]

                prediction = prediction.reshape(c, n, -1) # [C, Cond, ...]
                target = target.unsqueeze(1).reshape(c, 1, -1) # [C, 1, ...] (second dim will broadcast across ensemble)

                diff = self.pdist(prediction, target) # [C, Cond]
                dist_matrix = self.pdist(prediction.unsqueeze(1), prediction.unsqueeze(2))  # [C, Cond, Cond]
                
                diff_terms = self.diag_mask[None, ...] * (diff.unsqueeze(1) + diff.unsqueeze(2)) # [C, Cond, Cond], diagonal elements zeroed out
                crps = self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum(dim=(1,2))/(b*f*t*h*w)
            else:
                prediction = prediction.reshape(n, -1)
                target = target.unsqueeze(0).reshape(1, -1) # [1, ...] (first dim will broadcast across ensemble)
                diff = self.pdist(prediction, target) # [Cond]
                dist_matrix = self.pdist(prediction.unsqueeze(1), prediction.unsqueeze(0))  # [Cond, Cond] 

                diff_terms = self.diag_mask * (diff.unsqueeze(0) + diff.unsqueeze(1)) # [Cond, Cond], diagonal elements zeroed out
                crps = self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum()/(b*f*c*t*h*w)

            return crps


class WeightedCRPSLossSpectral(th.nn.MSELoss):

    """
    Probabilistic loss function that allows for user defined weighting of variables when calculating CRPS.
    """

    def __init__(
        self,
        weights: Sequence = [],
        n_members: int = 2,
        alpha: float = 0.95,
        lambda_spec: float = 0.1,
        nside: int = 64,
        lmax: int = 3*64 - 1,
        mmax: int = 3*64 - 1,
        multiscale: float = 0.0,
        lsm_file: str = None,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "land_sea_mask"},
    ):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        n_members: int
            number of ensemble members in the model output
        alpha: float
            hyperparamter for approximating fair CRPS loss. between 0 and 1, 1 corresponds to a fair CRPS loss.
        lambda_spec: float
            weight for the spectral loss. Default is 0, no spectral loss is applied.
        nside: int
            nside for the HEALPix grid. Default is 64.
        lmax: int
            lmax for the SHT. Default is 3*nside - 1.
        mmax: int
            mmax for the SHT. Default is 3*nside - 1.
        multiscale: float, optional
            weight for the multiscale loss. Default is 0, no multiscale loss is applied.
        lsm_file: str
            path to the lsm file. Default is None, no lsm is applied.
        open_dict: dict
            dictionary of keyword arguments for xarray.open_dataset. Default is {"engine": "zarr"}.
        selection_dict: dict
            dictionary of keyword arguments for xarray.open_dataset. Default is {"channel_c": "land_sea_mask"}.
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        self.n_members = n_members
        self.device = None
        self.lambda_spec = lambda_spec
        self.multiscale = multiscale

        # Parameters for "almost fair CRPS" loss. See https://arxiv.org/html/2412.15832v1
        self.coeff_eps = 1 - ((1-alpha) / (n_members))
        self.averaging_coeff = 1 / (2* n_members * (n_members - 1))

        # SHT utils: transform, grid reordering, output indexing
        self.lmax = lmax
        self.mmax = mmax
        self.nside = nside
        self.sht = SHTCUDA(nside=nside, lmax=lmax, mmax=mmax, quad_weights='ring')
        src_grid = earth2grid.healpix.Grid(level=int(np.log2(nside)), pixel_order=HEALPIX_PAD_XY)
        tar_grid = earth2grid.healpix.Grid(level=int(np.log2(nside)), pixel_order=PixelOrder.RING)
        self.reorder_to_ring = earth2grid.get_regridder(src_grid, tar_grid).to(th.float32)
        if self.multiscale > 0:
            self.scales = [200, 400, 800, 1600] # in units of km
            self.isht = iSHTCUDA(nside=nside, lmax=lmax, mmax=mmax, quad_weights='ring')
            self.reorder_from_ring = earth2grid.get_regridder(tar_grid, src_grid).to(th.float32)

        self.lsm_file = lsm_file
        if lsm_file is not None:
            self.lsm_ds = xr.open_dataset(lsm_file, **open_dict).constants.sel(selection_dict)
            self.lsm_tensor = 1 - th.tensor(np.expand_dims(self.lsm_ds.values, (0, 2, 3)))
        else:
            self.lsm_tensor = th.ones(1, 1, 1, 1, 1, 1) # Spoof the tensor dimensions for broadcasting

        # For n>2, will use pairwise distance to copmute [NxN] distance matrix
        # Diagonal elements of (prediciton - target) matrix are zeroed out to avoid double counting
        self.pdist = th.nn.PairwiseDistance(p=1)
        self.diag_mask = th.ones(self.n_members, self.n_members) - th.eye(self.n_members) # Mask to zero out diagonal elements


    def setup(self, trainer):
        """
        pushes constants to cuda device
        """

        if len(trainer.output_variables) != len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)
        self.averaging_coeff = th.tensor(self.averaging_coeff, device=trainer.device)
        self.coeff_eps = th.tensor(self.coeff_eps, device=trainer.device)
        self.reorder_to_ring = self.reorder_to_ring.to(device=trainer.device)
        self.sht = self.sht.to(device=trainer.device)
        self.pdist = self.pdist.to(device=trainer.device)
        self.diag_mask = self.diag_mask.to(device=trainer.device)

        if self.multiscale > 0:
            self.isht = self.isht.to(device=trainer.device)
            self.reorder_from_ring = self.reorder_from_ring.to(device=trainer.device)
        if self.lsm_file is not None:
            self.lsm_tensor = self.lsm_tensor.to(device=trainer.device)

    def _apply_sht(self, x, face_dim, return_abs=True):
        """Apply SHT to a tensor
        Reshape to [..., F*H*W], reorder to ring, apply SHT
        If return_abs is True, return the absolute value of the SHT (real**2 + imag**2)

        Parameters
        ----------
        x: torch.Tensor
            The tensor to apply SHT to
        face_dim: int
            The dimension of the tensor corrsponding to HEALPix faces
        return_abs: bool, optional
            Whether to return the absolute value of the SHT (real**2 + imag**2)
        """
        x = th.movedim(x, face_dim, -3)
        if x.shape[-3:] != (12, self.nside, self.nside):
            raise ValueError(f"Shape of input tensor should be [..., F, ..., H, W] with F in position {face_dim}, got {x.shape}")
        
        x = x.reshape(*x.shape[:-3], -1)
        x = self.reorder_to_ring(x.contiguous()) # contiguous needed for channels first format in validation loop
        x = self.sht(x)
        if return_abs:
            x = x.real ** 2 + x.imag ** 2
        return x

    def _apply_isht(self, x, face_dim):
        """Apply inverse SHT to a tensor shape [..., l, m]
        Inverse transform, reorder from ring, Reshape to [..., F, H, W], move face dim appropriately
        """

        x = self.isht(x) # [..., l, m] -> [..., F*H*W]
        x = self.reorder_from_ring(x)
        x = x.reshape(*x.shape[:-1], 12, self.nside, self.nside) # [..., F*H*W] -> [..., F, H, W]
        x = th.movedim(x, -3, face_dim) # [..., F, H, W] -> [..., F, ..., H, W]
        return x

    def _l_filter(self, scale, device="cuda"):
        """Return a spherical gaussian filter of scale `scale` (in units of km)
        """
        scale_radians  = scale / 6371.0
        ell = th.arange(self.lmax, device=device, dtype=th.float32)
        return th.exp(-0.5* ell * (ell + 1) * (scale_radians ** 2))

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedCRPSLoss 
        Computes the CRPS loss for the model prediction and target.

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor shape [Cond*B, F, T, C, H, W] where Cond is the number of ensemble members
        target: torch.Tensor
            The target tensor shape [B, F, T, C, H, W]
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        
        # Unfold ensemble dimension from batch dimension to have shape [Cond, B, F, T, C, H, W]
        b, f, t, c, h, w = target.shape
        prediction = prediction.view(self.n_members, b, f, t, c, h, w)

        # checks for dimensions 
        if not prediction.shape[1:] == target.shape:
            raise ValueError(f"Shape of prediction should match shape of target along non-ensemble dimensions, got {prediction.shape} and {target.shape}")
    
        if not prediction.shape[0] == self.n_members:
            raise ValueError(f"Shape of prediction should have ensemble dimension of size {self.n_members}, got {prediction.shape[0]}")

        n = self.n_members

        # Manual cast
        prediction = prediction.to(th.float32)
        target = target.to(th.float32)

        # Apply channel weights across channel dims
        prediction *= self.loss_weights[None, None, None, None, :, None, None]
        target *= self.loss_weights[None, None, None, :, None, None]

        if n == 2:
            # Use faster explicit implementation
            diff_target = th.abs(prediction - target.unsqueeze(0)).sum(dim=0) # [B, F, T, C, H, W]
            diff_ensemble = th.abs(prediction[0] - prediction[1]) # [B, F, T, C, H, W]
            crps = self.averaging_coeff*(diff_target - self.coeff_eps * diff_ensemble) # [B, F, T, C, H, W]

            if average_channels:
                loss = crps.mean()
            else:
                loss = crps.mean(dim=(0, 1, 2, 4, 5))

            if self.lambda_spec > 0:

                with th.cuda.amp.autocast(enabled=False):
                    # # Reorder predictions: [N, B, F, T, C, H, W] -> [N, B, T, C, F*H*W]
                    # pred_ring = self.reorder_to_ring(prediction.permute(0, 1, 3, 4, 2, 5, 6).reshape(n, b, t, c, f*h*w))

                    # # Reorder targets: [B, F, T, C, H, W] -> [B, T, C, F*H*W]
                    # tar_ring = self.reorder_to_ring(target.permute(0, 2, 3, 1, 4, 5).reshape(b, t, c, f*h*w))

                    # # Compute SHT of predictions and targets
                    # sht_pred = self.sht(pred_ring) # [N, B, T, C, l, m]
                    # sht_tar = self.sht(tar_ring) # [B, T, C, l, m]
                    # sht_pred = sht_pred.real ** 2 + sht_pred.imag ** 2
                    # sht_tar = sht_tar.real ** 2 + sht_tar.imag ** 2

                    sht_pred = self._apply_sht(prediction, face_dim=2, return_abs=True)
                    sht_tar = self._apply_sht(target, face_dim=1, return_abs=True)

                    diff_sht_target = th.abs(sht_pred - sht_tar.unsqueeze(0)).sum(dim=(0, 4, 5)) # [B, T, C]
                    diff_sht_ensemble = th.abs(sht_pred[0] - sht_pred[1]).sum(dim=(-1,-2)) # [B, T, C] 
                    crps_sht = self.averaging_coeff * (diff_sht_target - self.coeff_eps * diff_sht_ensemble) # [B, T, C]

                    # Compute spectral afCRPS
                    if average_channels:
                        spec_loss = crps_sht.mean()
                    else:
                        spec_loss = crps_sht.mean(dim=(0, 1))

                loss += self.lambda_spec * spec_loss
            
            if self.multiscale > 0:
                for scale in self.scales:
                    l_filter = self._l_filter(scale, device=prediction.device)
                    with th.cuda.amp.autocast(enabled=False):
                        sht_pred= self._apply_sht(prediction, face_dim=2, return_abs=False)
                        sht_tar = self._apply_sht(target, face_dim=1, return_abs=False)

                        l_filter_pred = l_filter[None, None, None, None, :, None] # [1, 1, 1, 1, lmax, 1]
                        l_filter_tar = l_filter[None, None, None, :, None] # [1, 1, 1, lmax, 1]

                        pred_smooth = self._apply_isht(l_filter_pred * sht_pred, face_dim=2)
                        tar_smooth = self._apply_isht(l_filter_tar * sht_tar, face_dim=1)

                        diff_target = th.abs(pred_smooth - tar_smooth.unsqueeze(0)).sum(dim=0) # [B, F, T, C, H, W]
                        diff_ensemble = th.abs(pred_smooth[0] - pred_smooth[1]) # [B, F, T, C, H, W]
                        crps = self.averaging_coeff*(diff_target - self.coeff_eps * diff_ensemble) # [B, F, T, C, H, W]

                        crps *= self.lsm_tensor

                        if average_channels:
                            loss += self.multiscale * crps.mean()
                        else:
                            loss += self.multiscale * crps.mean(dim=(0, 1, 2, 4, 5))
                        
            return loss
        
        else:
            # Use pairwise distance method
            if not average_channels:
                # Move channels to first dimension and exclude that dimension from the reductions           
                prediction = prediction.permute(4, 0, 1, 2, 3, 5, 6) # [C, Cond, B, F, T, H, W]
                target = target.permute(3, 0, 1, 2, 4, 5) # [C, B, F, T, H, W]

                pred = prediction.reshape(c, n, -1) # [C, Cond, ...]
                tar = target.unsqueeze(1).reshape(c, 1, -1) # [C, 1, ...] (second dim will broadcast across ensemble)

                diff = self.pdist(pred, tar) # [C, Cond]
                dist_matrix = self.pdist(pred.unsqueeze(1), pred.unsqueeze(2))  # [C, Cond, Cond]
                
                diff_terms = self.diag_mask[None, ...] * (diff.unsqueeze(1) + diff.unsqueeze(2)) # [C, Cond, Cond], diagonal elements zeroed out
                loss = self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum(dim=(1,2))/(b*f*t*h*w)

                if self.lambda_spec > 0:
                    with th.cuda.amp.autocast(enabled=False):
                        # # Reorder predictions: [C, Cond, B, F, T, H, W] -> [C, Cond, B, T, F*H*W]
                        # pred_ring = self.reorder_to_ring(prediction.permute(0, 1, 2, 4, 3, 5, 6).reshape(c, n, b, t, f*h*w))

                        # # Reorder targets: [C, B, F, T, H, W] -> [C, B, T, F*H*W]
                        # tar_ring = self.reorder_to_ring(target.permute(0, 1, 3, 2, 4, 5).reshape(c, b, t, f*h*w))

                        # # Compute SHT of predictions and targets
                        # sht_pred = self.sht(pred_ring).reshape(c, n, -1) # [C, Cond, B, T, l, m] -> [C, Cond, ...]
                        # sht_tar = self.sht(tar_ring).unsqueeze(1).reshape(c, 1, -1) # [C, B, T, l, m] -> [C, 1, ...] (second dim will broadcast across ensemble)
                        # sht_pred = sht_pred.real ** 2 + sht_pred.imag ** 2
                        # sht_tar = sht_tar.real ** 2 + sht_tar.imag ** 2

                        sht_pred = self._apply_sht(prediction, face_dim=3, return_abs=True).reshape(c, n, -1)
                        sht_tar = self._apply_sht(target, face_dim=2, return_abs=True).unsqueeze(1).reshape(c, 1, -1)

                        diff = self.pdist(sht_pred, sht_tar) # [C, Cond]
                        dist_matrix = self.pdist(sht_pred.unsqueeze(1), sht_pred.unsqueeze(2))  # [C, Cond, Cond]
                        
                        diff_terms = self.diag_mask[None, ...] * (diff.unsqueeze(1) + diff.unsqueeze(2)) # [C, Cond, Cond], diagonal elements zeroed out
                        loss += self.lambda_spec * self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum(dim=(1,2))/(b*t*self.lmax*self.mmax)

            else:
                pred = prediction.reshape(n, -1)
                tar = target.unsqueeze(0).reshape(1, -1) # [1, ...] (first dim will broadcast across ensemble)
                diff = self.pdist(pred, tar) # [Cond]
                dist_matrix = self.pdist(pred.unsqueeze(1), pred.unsqueeze(0))  # [Cond, Cond] 

                diff_terms = self.diag_mask * (diff.unsqueeze(0) + diff.unsqueeze(1)) # [Cond, Cond], diagonal elements zeroed out
                loss = self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum()/(b*f*c*t*h*w)

                if self.lambda_spec > 0:
                    with th.cuda.amp.autocast(enabled=False):
                        # # Reorder predictions: [Cond, B, F, T, C, H, W] -> [Cond, B, T, C, F*H*W]
                        # pred_ring = self.reorder_to_ring(prediction.permute(0, 1, 3, 4, 2, 5, 6).reshape(n, b, t, c, f*h*w))

                        # # Reorder targets: [B, F, T, C, H, W] -> [B, T, C, F*H*W]
                        # tar_ring = self.reorder_to_ring(target.permute(0, 2, 3, 1, 4, 5).reshape(b, t, c, f*h*w))

                        # # Compute SHT of predictions and targets
                        # sht_pred = self.sht(pred_ring).reshape(n, -1) # [Cond, B, T, C, l, m] -> [Cond, ...]
                        # sht_tar = self.sht(tar_ring).unsqueeze(0).reshape(1, -1) # [B, T, C, l, m] -> [1, ...] (first dim will broadcast across ensemble)
                        # sht_pred = sht_pred.real ** 2 + sht_pred.imag ** 2
                        # sht_tar = sht_tar.real ** 2 + sht_tar.imag ** 2
                        sht_pred = self._apply_sht(prediction, face_dim=2, return_abs=True).reshape(n, -1)
                        sht_tar = self._apply_sht(target, face_dim=1, return_abs=True).unsqueeze(0).reshape(1, -1)

                        diff = self.pdist(sht_pred, sht_tar) # [Cond]
                        dist_matrix = self.pdist(sht_pred.unsqueeze(1), sht_pred.unsqueeze(0))  # [Cond, Cond] 
                        
                        diff_terms = self.diag_mask * (diff.unsqueeze(0) + diff.unsqueeze(1)) # [Cond, Cond], diagonal elements zeroed out
                        loss += self.lambda_spec * self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum()/(b*t*self.lmax*self.mmax)

            return loss
        

class CosineAnnealedOceanCRPSLoss(th.nn.Module):
    """
    Hybrid loss that combines an ensemble-anchored weighted ocean MSE with a
    probabilistic `WeightedCRPSLoss` using a cosine annealing schedule:

        L_total(t) = alpha(t) * lambda_mse * L_MSE + (1 - alpha(t)) * L_CRPS

    where alpha(t) decays smoothly from 1 -> 0 over a configured number of training steps.

    Parameters
    ----------
    mse_weights : Sequence
        Per-channel weights for the deterministic MSE anchor. Length must match the
        number of output variables for the ocean model (i.e. `len(trainer.output_variables)`).
    crps_weights : Sequence
        Per-channel weights for the CRPS term, passed directly to `WeightedCRPSLoss`.
        Length must also match `len(trainer.output_variables)`.
    n_members : int, optional
        Number of ensemble members in the model output. The batch dimension of
        `prediction` is assumed to be `n_members * batch_size`.
    alpha : float, optional
        Fair-CRPS coefficient for `WeightedCRPSLoss` (as in the original class).
        Values in (0, 1]; 1.0 recovers the fair CRPS formulation.
    mean_penalty : float, optional
        Weight for the global-mean bias penalty inside `WeightedCRPSLoss`. If 0.0,
        no mean-bias penalty is applied.
    lsm_file : str, optional
        Path to the land–sea mask dataset. If given, both MSE and CRPS terms ignore
        land points according to this mask.
    open_dict : dict, optional
        Keyword arguments passed to `xarray.open_dataset` when loading `lsm_file`
        (e.g. `{"engine": "zarr"}`).
    selection_dict : dict, optional
        Selection dictionary used to extract the mask variable from `lsm_file`
        (e.g. `{"channel_c": "land_sea_mask"}` or `{"channel_c": "lsm"}`).
    lsm_binary_mask : bool, optional
        If True, use a binary LSM in `WeightedCRPSLoss` (land < lsm_binary_threshold → 1, else 0).
        If False (default), use continuous mask (1 - land_fraction). Passed to
        `WeightedCRPSLoss`.
    lsm_binary_threshold : float, optional
        Float in [0, 1], default 0.5. When lsm_binary_mask is True, this is the land-fraction
        threshold used for the binary mask. Passed to `WeightedCRPSLoss`.
    multiscale : float, optional
        Weight for the multiscale CRPS term in `WeightedCRPSLoss`. Zero disables
        multiscale CRPS.
    masked_pool : bool, optional
        If True, spatial pooling in the multiscale CRPS term uses only ocean pixels
        (land ignored). When `lsm_file` is provided, masked pooling is effectively
        enabled for that path.
    temporal_dt : float, optional
        Weight for the temporal-gradient CRPS term (time-difference CRPS). Zero
        disables this term.
    mse_scale : float, optional
        Static scaling factor for the MSE anchor. Used when `auto_calibrate_scale`
        is False. Effective scale is applied as `lambda_mse * alpha(t) * L_MSE`.
    auto_calibrate_scale : bool, optional
        If True, ignore `mse_scale` initially and, on the first forward pass, set
        `lambda_mse` to approximately match the magnitudes of CRPS and MSE on that
        batch: `lambda_mse ≈ L_CRPS / (L_MSE + eps)`. The calibrated scale is then
        held fixed for the remainder of training.
    decay_steps : int, optional
        Total number of global training steps over which to decay `alpha(t)` from
        1 to 0. If None, this is inferred in `setup()` using
        `decay_fraction * trainer.max_epochs * len(trainer.dataloader_train)`.
    decay_fraction : float, optional
        Fraction (0, 1] of the total nominal training steps to use when inferring
        `decay_steps`. For example, `0.5` means the MSE anchor decays away over
        the first half of training.
    """

    def __init__(
        self,
        mse_weights: Sequence,
        crps_weights: Sequence,
        n_members: int = 2,
        alpha: float = 0.95,
        mean_penalty: float = 0.0,
        lsm_file: Optional[str] = None,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "land_sea_mask"},
        lsm_binary_mask: bool = False,
        lsm_binary_threshold: float = 0.5,
        multiscale: float = 0.0,
        masked_pool: bool = False,
        temporal_dt: float = 0.0,
        mse_scale: float = 1.0,
        auto_calibrate_scale: bool = False,
        decay_steps: Optional[int] = None,
        decay_fraction: float = 0.5,
    ):
        super().__init__()

        # Deterministic anchor configuration
        self.mse_weights = th.tensor(mse_weights)
        self.mse_scale = float(mse_scale)
        self.auto_calibrate_scale = bool(auto_calibrate_scale)
        self._mse_scale_eff: Optional[float] = None

        # Probabilistic CRPS loss (handles LSM loading and most weighting options)
        self.crps_loss = WeightedCRPSLoss(
            weights=crps_weights,
            n_members=n_members,
            alpha=alpha,
            mean_penalty=mean_penalty,
            lsm_file=lsm_file,
            open_dict=open_dict,
            selection_dict=selection_dict,
            lsm_binary_mask=lsm_binary_mask,
            lsm_binary_threshold=lsm_binary_threshold,
            multiscale=multiscale,
            masked_pool=masked_pool,
            temporal_dt=temporal_dt,
        )

        # Scheduling state
        self.training_step: int = 0
        self.decay_steps: Optional[int] = decay_steps
        self.decay_fraction: float = float(decay_fraction)

        # Cached LSM tensor (copied from inner CRPS loss in setup)
        self.lsm_tensor: Optional[th.Tensor] = None

        # Scalars for logging
        self.last_mse: Optional[th.Tensor] = None
        self.last_crps: Optional[th.Tensor] = None
        self.last_total: Optional[th.Tensor] = None
        self.last_alpha: Optional[th.Tensor] = None
        self.last_mse_scale_eff: Optional[th.Tensor] = None

    @property
    def n_members(self) -> int:
        return self.crps_loss.n_members

    def setup(self, trainer):
        """
        Push constants to device and infer decay_steps if not explicitly provided.
        Expects trainer to define `device`, `output_variables`, and optionally
        `max_epochs` and `dataloader_train`.
        """
        # Validate channel alignment with trainer variables
        if len(trainer.output_variables) != len(self.mse_weights):
            raise ValueError("Length of outputs and mse_weights is not the same!")

        # Move weights to device
        self.mse_weights = self.mse_weights.to(device=trainer.device)

        # Delegate to inner CRPS loss (this also moves its lsm_tensor to device)
        self.crps_loss.setup(trainer)

        # Share the land–sea mask from the CRPS loss for the MSE anchor
        self.lsm_tensor = self.crps_loss.lsm_tensor

        # Store total epochs and configure epoch-based decay horizon.
        # If an explicit decay_steps was provided, interpret it as an
        # override on the number of decay epochs.
        self.max_epochs = getattr(trainer, "max_epochs", None)
        if self.max_epochs is not None:
            if self.decay_steps is not None:
                # Treat user-provided decay_steps as decay_epochs for simplicity.
                self.decay_epochs = max(1, int(self.decay_steps))
            else:
                self.decay_epochs = max(1, int(self.decay_fraction * self.max_epochs))
        else:
            self.decay_epochs = None
        # #region agent log
        try:
            import json
            import time
            with open("/pscratch/sd/z/zespinos/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({"hypothesisId": "H2", "location": "healpix_loss.py:setup", "message": "decay_config", "data": {"max_epochs": self.max_epochs, "decay_epochs": self.decay_epochs, "decay_fraction": self.decay_fraction}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        self.current_epoch = 0

    def set_training_epoch(self, epoch: int) -> None:
        """Set the current training epoch for the annealing schedule."""
        # #region agent log
        try:
            import json
            import time
            with open("/pscratch/sd/z/zespinos/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({"hypothesisId": "H1_H4", "location": "healpix_loss.py:set_training_epoch", "message": "epoch_set", "data": {"epoch_arg": epoch}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        self.current_epoch = max(0, int(epoch))

    def set_training_step(self, step: int) -> None:
        """
        Backwards-compatible no-op for APIs that still call set_training_step.
        Epoch-based scheduling is preferred; use set_training_epoch instead.
        """
        pass

    def get_alpha(self) -> float:
        return float(self._alpha())

    def _alpha(self) -> float:
        # If decay horizon is unknown, keep the MSE anchor fully on.
        if self.decay_epochs is None or self.decay_epochs <= 0:
            return 1.0
        t = min(self.current_epoch, self.decay_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * float(t) / float(self.decay_epochs)))

    def _compute_ensemble_mse(
        self,
        prediction: th.Tensor,
        target: th.Tensor,
        average_channels: bool = True,
    ) -> th.Tensor:
        """
        Compute ensemble-anchored, ocean-masked, channel-weighted MSE:

            L_MSE = (1/N) * sum_i (y_i - y_true)^2

        where y_i are ensemble members and y_true is the deterministic target.

        prediction: [Cond*B, F, T, C, H, W]
        target:     [B, F, T, C, H, W]
        """
        b, f, t, c, h, w = target.shape
        n = self.n_members

        if prediction.ndim != 6 or target.ndim != 6:
            raise ValueError(
                f"Expected prediction and target to be 6D, got {prediction.shape} and {target.shape}"
            )

        if prediction.shape[0] != n * b:
            raise ValueError(
                f"Expected prediction first dim {n*b} (= n_members * batch), got {prediction.shape[0]}"
            )

        # [Cond, B, F, T, C, H, W]
        pred_ens = prediction.view(n, b, f, t, c, h, w)
        # Broadcast deterministic target across ensemble dimension
        tar_ens = target.unsqueeze(0).expand(n, -1, -1, -1, -1, -1, -1)

        # Squared error
        se = (pred_ens - tar_ens) ** 2

        # Apply per-channel weights
        se = se * self.mse_weights[None, None, None, None, :, None, None]

        # Apply ocean mask if available (matches CRPS lsm semantics)
        if self.lsm_tensor is not None:
            # WeightedCRPSLoss constructs lsm_tensor with shape [1, F, 1, 1, H, W].
            # This is already broadcastable to [N, B, F, T, C, H, W] along the
            # ensemble (N), batch (B), time (T), and channel (C) dimensions, so we
            # simply rely on standard broadcasting here without reshaping.
            se = se * self.lsm_tensor

        if average_channels:
            # Global average over all dims
            return se.mean()
        else:
            # Keep per-channel, average over ensemble, batch, faces, time, and spatial dims
            return se.mean(dim=(0, 1, 2, 3, 5, 6))

    def forward(self, prediction: th.Tensor, target: th.Tensor, average_channels: bool = True) -> th.Tensor:
        """
        Forward pass computing both the ensemble-anchored ocean-weighted MSE and the
        WeightedCRPSLoss, then blending them using the cosine schedule.

        prediction: [Cond*B, F, T, C, H, W]
        target:     [B, F, T, C, H, W]
        """
        # Compute component losses
        mse_loss = self._compute_ensemble_mse(prediction, target, average_channels=average_channels)
        crps_loss = self.crps_loss(prediction, target, average_channels=average_channels)

        # Ensure CRPS is a tensor broadcastable with MSE (per-channel or scalar)
        if not th.is_tensor(crps_loss):
            crps_loss = th.tensor(crps_loss, device=prediction.device, dtype=mse_loss.dtype)

        # Initialize / update effective MSE scale if auto-calibration is enabled
        if self._mse_scale_eff is None:
            if self.auto_calibrate_scale:
                with th.no_grad():
                    mse_scalar = mse_loss.mean() if mse_loss.ndim > 0 else mse_loss
                    crps_scalar = crps_loss.mean() if isinstance(crps_loss, th.Tensor) and crps_loss.ndim > 0 else crps_loss
                    eps = 1e-8
                    ratio = (crps_scalar / (mse_scalar + eps)).detach()
                    self._mse_scale_eff = float(ratio)
            else:
                self._mse_scale_eff = float(self.mse_scale)

        mse_scale_eff = float(self._mse_scale_eff if self._mse_scale_eff is not None else self.mse_scale)

        # Cosine schedule
        alpha = self._alpha()
        # #region agent log
        try:
            import json
            import time
            with open("/pscratch/sd/z/zespinos/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({"hypothesisId": "H3_H5", "location": "healpix_loss.py:forward", "message": "alpha_computed", "data": {"current_epoch": getattr(self, "current_epoch", None), "decay_epochs": getattr(self, "decay_epochs", None), "alpha": alpha}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion

        # Blend losses
        total = mse_scale_eff * alpha * mse_loss + (1.0 - alpha) * crps_loss

        # Cache logging scalars (detached)
        mse_scalar = mse_loss.mean() if mse_loss.ndim > 0 else mse_loss
        crps_scalar = crps_loss.mean() if isinstance(crps_loss, th.Tensor) and crps_loss.ndim > 0 else crps_loss
        total_scalar = total.mean() if isinstance(total, th.Tensor) and total.ndim > 0 else total

        self.last_mse = mse_scalar.detach()
        self.last_crps = crps_scalar.detach()
        self.last_total = total_scalar.detach()
        self.last_alpha = th.tensor(alpha, device=prediction.device, dtype=total_scalar.dtype).detach()
        self.last_mse_scale_eff = th.tensor(mse_scale_eff, device=prediction.device, dtype=total_scalar.dtype).detach()

        return total
