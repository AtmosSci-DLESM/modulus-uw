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

import gc
import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.masked import masked_tensor, as_masked_tensor
import xarray as xr
from omegaconf import DictConfig, OmegaConf
import zarr

from physicsnemo.datapipes.meta import DatapipeMetaData
from physicsnemo.utils.insolation import insolation

from . import couplers
from .base_timeseries_dataset_zarr import BaseTimeSeriesDatasetZarr
from .timeseries_dataset_zarr import TimeSeriesDatasetZarr

logger = logging.getLogger(__name__)


@dataclass
class MetaData(DatapipeMetaData):
    """Metadata for this datapipe"""

    name: str = "CoupledTimeSeries"
    # Optimization
    auto_device: bool = False
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = False


class CoupledTimeSeriesDatasetZarr(TimeSeriesDatasetZarr):
    """Dataset for coupling time series data with external earth system components.
    
    This class extends the base time series functionality to include coupling with
    external data sources like ocean models, land models, etc. It supports:
    - Integration of coupled data during training/inference
    - Masking of coupled fields
    - Addition of training noise to improve generalization
    - Step-by-step integration for forecasting
    """

    def __init__(
        self,
        ds_path: str,
        scaling: DictConfig,
        input_variables: Sequence,
        output_variables: Sequence = None,
        constant_variables: Sequence = None,
        input_time_dim: int = 1,
        output_time_dim: int = 1,
        data_time_step: Union[int, str] = "3h",
        time_step: Union[int, str] = "6h",
        gap: Union[int, str, None] = None,
        batch_size: int = 32,
        drop_last: bool = False,
        add_insolation: bool = False,
        forecast_init_times: Optional[Sequence] = None,
        start_date: Optional[Union[int, str]] = None,
        end_date: Optional[Union[int, str]] = None,
        couplings: Sequence = [],
        meta: DatapipeMetaData = MetaData(),
        add_train_noise: bool = False,
        train_noise_params: DictConfig = None,
        train_noise_seed: int = 42,
        land_masked_fields: Optional[Sequence] = None,
        sea_masked_fields: Optional[Sequence] = None,
        mask_threshold: float = 0.5,
    ):
        """Initialize coupled time series dataset.
        
        Parameters
        ----------
        couplings : Sequence
            List of coupling configurations, each containing:
            - coupler: Name of coupler class to use
            - params: Parameters for the coupler
        add_train_noise : bool, default=False
            Whether to add noise during training
        train_noise_params : DictConfig, optional
            Configuration for training noise, containing:
            - inputs: Dict mapping variable names to noise std
            - couplings: Dict mapping variable names to noise std
        train_noise_seed : int, default=42
            Random seed for noise generation
            
        Other parameters are same as BaseTimeSeriesDatasetZarr.
        See base class for detailed parameter descriptions.
        """
        self.coupled_variables = []
        for c in couplings:
            self.coupled_variables.append(c["params"]["variables"])

        # We setup couplers first so superclass can properly initialize
        # and set the scaling
        self.ds_path = ds_path
        self.ds = zarr.open(ds_path)
        self.couplings = [
            getattr(couplers, c["coupler"])(
                self.ds,
                **OmegaConf.to_object(DictConfig(c))["params"],
            )
            for c in couplings
        ]

        super().__init__(
            ds_path=ds_path,
            scaling=scaling,
            input_variables=input_variables,
            output_variables=output_variables,
            constant_variables=constant_variables,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
            data_time_step=data_time_step,
            time_step=time_step,
            gap=gap,
            batch_size=batch_size,
            drop_last=drop_last,
            add_insolation=add_insolation,
            forecast_init_times=forecast_init_times,
            start_date=start_date,
            end_date=end_date,
            land_masked_fields=land_masked_fields,
            sea_masked_fields=sea_masked_fields,
            mask_threshold=mask_threshold,
            meta=meta,
        )

        # calculate static indices for coupling
        for c in self.couplings:
            c.compute_coupled_indices(self.interval, self.data_time_step)
        # keep track of integration steps
        self.integration_step = (
            1  # starts at 1 because first step is done by __getitem__
        )
        self.last_batch = None  # keeps track of batch info for coupler
        self.curr_item = None  # keeps track of current initialization
    
    def _get_masking(self) -> None:
        """Setup masking fields for land and sea with coupling-specific additions.
        
        Extends base masking setup to handle coupled fields:
        - Identifies coupled fields that need masking
        - Creates PyTorch tensor versions of masks for GPU usage
        - Sets up separate indices for land/sea masking of coupled fields
        """
        super()._get_masking()

        self.masked_coupled_idx = -1
        if self.land_masked_fields and self.land_masked_fields[0] in self.coupled_variables:
            self.masked_coupled_idx = self.coupled_variables.index(self.land_masked_fields[0])
        # use dicts here so we can check individual instances
        self.coupled_land_masked_fields = {}
        self.coupled_sea_masked_fields = {}
        for index, coupled_variables in enumerate(self.coupled_variables):
            if self.land_masked_fields and len(self.land_masked_fields) > 0:
                land_indices = [
                    coupled_variables.index(field)
                    for field in self.land_masked_fields if field in coupled_variables
                ]
                self.coupled_land_masked_fields[index] = land_indices
            if self.sea_masked_fields and len(self.sea_masked_fields) > 0:
                sea_indices = [
                    coupled_variables.index(field)
                    for field in self.sea_masked_fields if field in coupled_variables
                ]
                self.coupled_sea_masked_fields[index] = sea_indices
                
        # for use with the couplers
        self.land_mask_tensor = torch.tensor(self.land_mask)
        self.sea_mask_tensor = torch.tensor(self.sea_mask)
        
    def _get_scaling_da(self) -> None:
        """ extend scaling to include coupling-specific additions.
        
        Extends base scaling setup to:
        - Create scaling parameters for coupled variables
        - Pass scaling info to coupler objects
        """
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc["zeros"] = {"mean": 0.0, "std": 1.0}
        scaling_da = scaling_df.to_xarray().astype("float32")

        for c in self.couplings:
            c.set_scaling(scaling_da)
        super()._get_scaling_da()

    def __getitem__(self, item: int) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
        """Get a batch of coupled time series data.

        This implementation extends the base time series data loading to include:
        1. Integration of coupled data sources
        2. Application of masking to coupled fields
        3. Addition of training noise if enabled
        4. Tracking of batch info for step-by-step integration

        Parameters
        ----------
        item : int
            Sample index

        Returns
        -------
        Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]
            In forecast mode: List of input arrays
            In training mode: Tuple of (input arrays, target array)
            
            Input arrays are in order:
            - Model inputs [B, F, T, C, H, W]
            - Insolation (if enabled) [B, F, T, 1, H, W]
            - Constants (if provided) [F, C, H, W]
            - Coupled inputs (if provided) [B, T, C, H, W]
            
            Target array has shape [B, F, T, C, H, W]
            where:
            B = batch size
            F = faces
            T = time steps
            C = channels
            H = height
            W = width

        Raises
        ------
        IndexError
            If item is out of range
        """

        if self.forecast_mode:
            inputs_result = super().__getitem__(item)
        else:
            inputs_result, targets = super().__getitem__(item)

        # start range
        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__")
        
        # used by the couplers to determine what time index to load
        # see method "next_integration()" for details
        time_index, this_batch = self._get_time_index(item)
        batch = {"time": slice(*time_index)}
        self.last_batch = batch

        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__:retrieve_coupled")
        # retrieve coupled inputs
        if len(self.couplings) > 0:
            integrated_couplings = np.concatenate(
                [
                    c.construct_integrated_couplings(batch, this_batch)
                    for c in self.couplings
                ],
                axis=2,
            )

            if self.masked_coupled_idx > -1:
                integrated_couplings_masked = integrated_couplings[:,:,self.masked_coupled_idx] * self.land_mask
                integrated_couplings[:,:,self.masked_coupled_idx] = integrated_couplings_masked #.data
        torch.cuda.nvtx.range_pop() # CoupledTimeSeriesDataset:__getitem__:retrieve_coupled

        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__:process_batch")
        # Insolation
        if self.add_insolation:
            # update current item and reset integration_step counter for further integrations which need
            # insolation but bypass this method see method "next_integration()" for details
            self.curr_item = item
            self.integration_step = 1

        if not self.forecast_mode and self.add_train_noise:
            torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__:add_train_noise")
            for c in self.couplings:
                for i, v in enumerate(c.variables):
                    integrated_couplings[i, :, :] += self.rng.normal(
                        loc=0,
                        scale=self.train_noise_params["couplings"][v]["std"],
                        size=integrated_couplings[i, :, :].shape,
                    )
            torch.cuda.nvtx.range_pop()

        # append integrated couplings
        if len(self.couplings) > 0:
            inputs_result.append(integrated_couplings)

        torch.cuda.nvtx.range_pop() # CoupledTimeSeriesDataset:__getitem__
        if self.forecast_mode:
            return inputs_result

        return inputs_result, targets

    def _apply_masks(self, tensor: torch.Tensor, transpose_first: bool = True) -> torch.Tensor:
        """Apply land and sea masks to the input tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to mask
        transpose_first : bool
            Whether to transpose before masking

        Returns
        -------
        torch.Tensor
            Masked tensor
        """
        if not (self.land_mask_indices or self.sea_mask_indices):
            return tensor
        
        if transpose_first:
            tensor = tensor.transpose(1, 2)

            if self.land_mask_indices:
                self.land_mask_tensor = self.land_mask_tensor.to(tensor.device)
                for idx in self.land_mask_indices:
                    tensor[:,:,:,idx] = tensor[:,:,:,idx] * self.land_mask_tensor

            if self.sea_mask_indices:
                self.sea_mask_tensor = self.sea_mask_tensor.to(tensor.device)
                for idx in self.sea_mask_indices:
                    tensor[:,:,:,idx] = tensor[:,:,:,idx] * self.sea_mask_tensor
            
        if transpose_first:
            tensor = tensor.transpose(1, 2)
        
        return tensor

    def _get_next_insolation(self, time_offset: int) -> torch.Tensor:
        """Calculate insolation for next integration step.
        
        Parameters
        ----------
        time_offset : int
            Time offset for insolation calculation
            
        Returns
        -------
        torch.Tensor
            Insolation tensor
        """
        sol = torch.tensor(
            insolation(
                self._get_forecast_sol_times(self.curr_item) + time_offset,
                self.lat,
                self.lon,
            )[:, None]
        )
        decoder_inputs = np.empty(
            (1, self.input_time_dim + self.output_time_dim, 1) + self.spatial_dims,
            dtype="float32",
        )
        decoder_inputs[0] = sol
        return torch.tensor(decoder_inputs.transpose(0, 3, 1, 2, 4, 5))

    def _get_next_couplings(self, offset: int) -> Optional[torch.Tensor]:
        """Get coupled inputs for next integration step.
        
        Parameters
        ----------
        offset : int
            Integration offset
            
        Returns
        -------
        Optional[torch.Tensor]
            Coupled inputs tensor if couplings exist, None otherwise
        """
        if not self.couplings:
            return None
        
        integrated_couplings = np.concatenate(
            [c.construct_integrated_couplings(batch=self.last_batch, integration_offset=offset) 
             for c in self.couplings], 
            axis=2
        )
        
        if self.masked_coupled_idx > -1:
            integrated_couplings_masked = integrated_couplings[:,:,self.masked_coupled_idx] * self.land_mask
            integrated_couplings[:,:,self.masked_coupled_idx] = integrated_couplings_masked
        
        return torch.tensor(integrated_couplings)

    def next_integration(self, model_outputs: torch.Tensor, constants: torch.Tensor) -> List[torch.Tensor]:
        """Get inputs for next integration step with coupling data.
        
        Parameters
        ----------
        model_outputs : torch.Tensor
            Model outputs from previous step [B, F, T, C, H, W]
        constants : torch.Tensor
            Constant fields [F, C, H, W]
            
        Returns
        -------
        List[torch.Tensor]
            List of input tensors for next step
        """
        inputs_result = []
        
        # Get prognostic inputs from model outputs
        init_time_dim = len(self._input_indices[0])
        prognostic_inputs = model_outputs[:, :, -init_time_dim:]
        prognostic_inputs = self._apply_masks(prognostic_inputs)
        inputs_result.append(prognostic_inputs)
        
        # Add insolation if needed
        if self.add_insolation:
            time_offset = self.time_step * self.output_time_dim * self.integration_step
            inputs_result.append(self._get_next_insolation(time_offset))
        
        # Add constants
        inputs_result.append(constants)

        # Add coupled inputs if any
        offset = self.interval * self.output_time_dim * self.integration_step
        coupled_inputs = self._get_next_couplings(offset)
        if coupled_inputs is not None:
            inputs_result.append(coupled_inputs)

        # Increment integration step
        self.integration_step += 1

        return inputs_result
