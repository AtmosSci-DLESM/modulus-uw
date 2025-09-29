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

from typing import Sequence

import numpy as np
import torch as th
import xarray as xr

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
        alpha: float = 0.95
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
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        if n_members < 2:
            raise ValueError("n_members must be at least 2 for CRPS loss to be defined")
        else:    
            self.n_members = n_members
        self.device = None

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

        # Apply channel weights across channel dims
        prediction *= self.loss_weights[None, None, None, None, :, None, None]
        target *= self.loss_weights[None, None, None, :, None, None]

        if n == 2:
            # Use faster explicit implementation
            diff_target = th.abs(prediction - target.unsqueeze(0)).sum(dim=0) # [B, F, T, C, H, W]
            diff_ensemble = th.abs(prediction[0] - prediction[1]) # [B, F, T, C, H, W]
            crps = self.averaging_coeff*(diff_target - self.coeff_eps * diff_ensemble) # [B, F, T, C, H, W]

            if average_channels:
                return crps.mean()
            else:
                return crps.mean(dim=(0, 1, 2, 4, 5))
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
