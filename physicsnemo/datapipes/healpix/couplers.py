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
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import cftime
import numpy as np
import pandas as pd
import torch as th
import xarray as xr
import zarr as zr

logger = logging.getLogger(__name__)


class BaseCoupler(ABC):
    """
    Base class for couplers used to interface two components of earth system.

    This class contains common functionality shared by different coupler implementations.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        batch_size: int,
        variables: Sequence,
        presteps: int = 0,
        input_time_dim: int = 2,
        output_time_dim: int = 2,
        input_times: Sequence = [pd.Timedelta("24h"), pd.Timedelta("48h")],
        prepared_coupled_data: bool = True,
        time_first: bool = True,
    ):
        """
        Parameters
        ----------
        dataset: xr.Dataset
            xarray Dataset that holds coupled data
        batch_size: int
            number of batch size during training.
            forecasting batch size should be 1
        variables: Sequence
            sequence of strings that indicate the coupled variable
            names in the dataset. All names should be in the dataset with 
            an optional time component at the end, eg ttr-48h
        presteps: int, optional
            the number of model steps used to initialize the hidden state.
            If not using a GRU, prestep is 0, default 0
        input_time_dim: int, optional
            number of input times into the model, default 2
        output_time_dim: int, optional
            number of output times for each model step, default 2
        input_times: Sequence, optional
            sequence of pandas Timedelta objects that indicate which times are to be coupled,
            default [pd.Timedelta("24h"), pd.Timedelta("48h")]
        prepared_coupled_data: boolean, optional
            If True assumes data in dataset has been prepared appropriately for training:
            averages have already been calculated so that each time step denotes
            the right side of a averaging_window window.
            This is highly recommended for training, default True
        time_first: boolean, optional
            Whether the coupled data should be permuted to have the time dimension first 
            [T, B, C, F, H, W] rather than [B, F, T, C, H, W]
        """
        # extract important meta data from ds
        self.ds = dataset
        self.batch_size = batch_size
        self.spatial_dims = self.ds["inputs"].shape[2:]
        self.variables = variables
        self.presteps = presteps
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.coupled_integration_dim = self._compute_coupled_integration_dim()
        self.input_times = [pd.Timedelta(t) for t in input_times]
        self.output_channels = len(self.variables) * len(self.input_times)
        self.timevar_dim = self._compute_timevar_dim()
        self.coupled_inputs_shape = None
        self.coupled_scaling = None
        self._coupled_offsets = None
        self.coupled_mode = False
        self.integrated_couplings = None
        self.ds_variable_indices = []
        self.time_first = time_first

        if not prepared_coupled_data:
            raise NotImplementedError("Data preparation not yet implemented")

        if type(self.ds) == xr.Dataset:
            self.use_zarr = False
        elif type(self.ds) == zr.Group:
            self.use_zarr = True
            # Iterate over self.variables in the outer loop so selected indices
            # follow the order of self.variables (matching the xarray path's
            # `.sel(channel_in=self.variables)`), not the dataset's native
            # channel_in order. Downstream code (e.g. coupled_scaling built
            # from self.variables in set_scaling) assumes the coupled channel
            # axis is ordered by self.variables.
            # Use `[:]` so zarr v3 yields hashable scalar strings, not Array views.
            channel_in = list(self.ds["channel_in"][:])
            self.ds_variable_indices = [
                i for v in self.variables for i, ic in enumerate(channel_in) if ic == v
            ]
            # Check if any of the requested variables are not in the dataset
            missing_variables = set(self.variables) - set(channel_in)
            if len(missing_variables) > 0:
                raise ValueError(f"Missing variables in dataset for coupling: {missing_variables}")
        else:
            raise TypeError(
                f"Coupler only supports xarray Datasets or zarr Groups, got {type(self.ds)}"
            )

    def _compute_coupled_integration_dim(self):
        return self.presteps + max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_timevar_dim(self):
        return len(self.input_times) * len(self.variables)

    @abstractmethod
    def compute_coupled_indices(self, interval, data_time_step):
        """
        Called by CoupledDataset to compute static indices for training samples.
        Must be implemented by subclasses as the logic varies between coupler types.

        Parameters
        ----------
        interval: int
            ratio of dataset timestep to model dt
        data_time_step:
            dataset timestep
        """
        pass

    def set_scaling(self, scaling_da):
        """
        Called by CoupledDataset to compute static indices for training samples

        Parameters
        ----------
        scaling_da: xarray.DataArray
            values used to scale input data, uses mean and std
        """
        # verify all the channels are there for scaling, this avoids an opaque
        # "not all values found in index 'index'"" error that looks like its from hydra
        missing_channels = set(self.variables) - set(scaling_da.index.values)
        if len(missing_channels) > 0:
            raise KeyError(
                f"Coupled variable(s) not found in scaling values: {missing_channels}"
            )

        coupled_scaling = scaling_da.sel(index=self.variables).rename(
            {"index": "channel_in"}
        )
        self.coupled_scaling = {
            "mean": np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
            "std": np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4)),
        }

    def setup_coupling(self, coupled_module):
        """
        Sets up the coupling between the coupled variables and the provided module

        Parameters
        ----------
        coupled_module: physicsnemo.datapipes.healpix.TimeSeriesDataset
            The module which this coupler will be coupled against.
        """
        # To expedite the coupling process the coupled_forecast
        # get proper channels from coupled component output
        output_channels = coupled_module.output_variables
        # A bit convoluted. Some variable names are present in the dataset as is,
        # Some prepared coupled variables are given a suffix for training associated
        # with a time increment suach as a trailing average increment e.g. 'z1000-48H'.
        # Some variables may have an additional suffix, e.g. 'z1000-3H-48H'. The final
        # suffix (if it exists) is used to determine the coupling increment.
        channel_indices = [
            i
            for i, oc in enumerate(output_channels)
            for v in self.variables
            # extract everthing before the last "-" if there is one in the name
            if (("-" not in v and oc == v) or (oc == "-".join(v.split("-")[:-1])))
        ]
        # check for missing variables
        if len(self.variables) != len(channel_indices):
            found_channels = [
                oc
                for oc in output_channels
                for v in self.variables
                # extract everthing before the last -
                if (("-" not in v and oc == v) or (oc == "-".join(v.split("-")[:-1])))
            ]
            missing_channels = set(self.variables) - set(found_channels)
            raise ValueError(f"Missing variables in coupled module: {missing_channels}")
        self.coupled_channel_indices = channel_indices

    def reset_coupler(self):
        self.coupled_mode = False
        self.integrated_couplings = None
        self.preset_coupled_fields = None

    @abstractmethod
    def set_coupled_fields(self, coupled_fields: th.tensor):
        """
        Set the data for the coupled field for the next iteration of the dataloader.
        Must be implemented by subclasses as the processing logic varies.

        Parameters
        ----------
        coupled_fields: th.tensor
            The data to use when the dataloader requests coupled fields. Expected
            format is [B, F, T, C, H, W]
        """
        pass

    def _construct_integrated_couplings_from_dataset(self, batch, bsize):
        """
        Common logic for constructing integrated couplings from dataset.
        Used by both ConstantCoupler and TrailingAverageCoupler.
        """
        # reset integrated couplings
        self.integrated_couplings = np.empty(
            (bsize, self.coupled_integration_dim, self.timevar_dim) + self.spatial_dims
        )

        # Offsets may be negative (a refresh period puts the prestep slot in the
        # previous period), so the read starts at the smallest offset and the
        # stored offsets are shifted by it when indexing.
        self._offset_min = int(self._coupled_offsets.min())
        start = batch["time"].start + self._offset_min
        if start < 0:
            raise ValueError(
                f"Coupled data needed from time index {start}, before the start of "
                "the dataset; start the split at least one refresh period later."
            )
        index_range = slice(start, batch["time"].start + int(self._coupled_offsets.max()) + 1)

        # extract coupled variables
        if self.use_zarr:
            # Loading the contiguous time slice into memory and then pulling out the semi-random
            # variable indices is quicker than trying to do this all at once.
            ds_index_range = self.ds["inputs"][index_range]
            ds_index_range = ds_index_range[:, self.ds_variable_indices]
        else:
            ds_index_range = (
                self.ds["inputs"].sel(channel_in=self.variables)
                .isel(time=index_range)
                .compute()
            )

        return ds_index_range

    def construct_integrated_couplings(
        self,
        batch=None,
        bsize=None,
    ):
        """
        Construct array of coupled inputs that includes values required for
        model integration steps.

        Parameters
        ----------
        batch: Sequence
            indices of dataset sample dimension associated with current batch
        bsize: int
            batch size

        Returns
        -------
        numpy.ndarray: The coupled data
        """
        if self.coupled_mode:
            return self.preset_coupled_fields
        else:
            if (batch is None) or (bsize is None):
                raise ValueError(
                    "batch and bsize must be provided when not in coupled_mode"
                )

            ds_index_range = self._construct_integrated_couplings_from_dataset(
                batch, bsize
            )

            # Apply scaling if available
            if self.coupled_scaling is not None:
                ds_index_range -= self.coupled_scaling["mean"]
                ds_index_range /= self.coupled_scaling["std"]

            # use static offsets to create integrated coupling array
            for b in range(bsize):
                for i in range(self.coupled_integration_dim):
                    offsets = self._coupled_offsets[b, i, :] - self._offset_min
                    if self.use_zarr:
                        coupling_temp = ds_index_range[offsets, :]
                    else:
                        coupling_temp = ds_index_range.isel(time=offsets).to_numpy()
                    self.integrated_couplings[b, i, :, :, :] = coupling_temp.reshape(
                        (self.timevar_dim,) + coupling_temp.shape[2:]
                    )
            if self.time_first:
                return self.integrated_couplings.transpose((1, 0, 2, 3, 4, 5)).astype(
                    "float32"
                )  # cast to float for compatibility
            else:
                return self.integrated_couplings.astype("float32")


class ConstantCoupler(BaseCoupler):
    """
    coupler used to interface two component of earth system

    constant coupler will take the the coupled field at integration time and
    force the model with this field consistently
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        batch_size: int,
        variables: Sequence,
        presteps: int = 0,
        input_time_dim: int = 2,
        output_time_dim: int = 2,
        input_times: Sequence = [pd.Timedelta("24h"), pd.Timedelta("48h")],
        prepared_coupled_data=True,
        refresh_period: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        dataset: xr.Dataset
            xarray Dataset that holds coupled data
        batch_size: int
            number of batch size during training.
            forecasting batch size should be 1
        variables: Sequence
            sequence of strings that indicate the coupled variable
            names in the dataset
        presteps: int, optional
            the number of model steps used to initialize the hidden state.
            If not using a GRU, prestep is 0, default 0
        input_time_dim: int, optional
            number of input times into the model, default 2
        output_time_dim: int, optional
            number of output times for each model step, default 2
        input_times: Sequence, optional
            sequence of pandas Timedelta objects that indicate which times are to be coupled,
            default [pd.Timedelta("24h"), pd.Timedelta("48h")]
        prepared_coupled_data: boolean, optional
            If True assumes data in dataset has been prepared appropriately for training:
            averages have already been calculated so that each time step denotes
            the right side of a averaging_window window.
            This is highly recommended for training, default True
        refresh_period: str, optional
            How often the coupled field is renewed, as in a sequentially coupled
            model (e.g. "24h"). The window is cut into periods of this length
            starting at the initial time t0, and a model step uses the field at the
            start of the period that holds its input time: the prestep (input time
            before t0) sees the previous period's field, steps with input times in
            [0, refresh_period) see the field at t0, the next period the field at
            t0 + refresh_period, and so on. Must be a multiple of the model step.
            Default None keeps one field (the window's first input time) for the
            whole window.
        """
        self.refresh_period = (
            pd.Timedelta(refresh_period) if refresh_period is not None else None
        )
        self._slot_periods = None
        self._previous_fields = None
        self._period0_source = None
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            variables=variables,
            presteps=presteps,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
            input_times=input_times,
            prepared_coupled_data=prepared_coupled_data,
        )

    def compute_coupled_indices(self, interval, data_time_step):
        """
        Called by CoupledDataset to compute static indices for training
        samples

        Parameters
        ----------
        interval: int
            ratio of dataset timestep to model dt
        data_time_step:
            dataset timestep
        """
        # create array of static coupled offsets that accompany each batch
        self._coupled_offsets = np.empty(
            [self.batch_size, self.coupled_integration_dim, len(self.input_times)]
        )
        shifts = np.array([ts / data_time_step for ts in self.input_times])
        if self.refresh_period is None:
            slot_base = np.zeros(self.coupled_integration_dim)
        else:
            # Offsets count data steps from each sample's first frame, which is the
            # first prestep input; t0 (the last input time of step 0) lies
            # ``interval * (input_time_dim * (1 + presteps) - 1)`` steps later.
            step = self.input_time_dim * interval
            refresh = self.refresh_period / pd.Timedelta(data_time_step)
            if refresh <= 0 or refresh % step != 0:
                raise ValueError(
                    f"refresh_period {self.refresh_period} must be a positive "
                    f"multiple of the model step ({step} data steps)"
                )
            refresh = int(refresh)
            t0 = interval * (self.input_time_dim * (1 + self.presteps) - 1)
            # Slot i feeds model step i - presteps, whose input time is
            # (i - presteps) * step; floor division puts the presteps in period -1.
            self._slot_periods = np.array(
                [((i - self.presteps) * step) // refresh for i in range(self.coupled_integration_dim)]
            )
            slot_base = t0 + self._slot_periods * refresh
        for b in range(self.batch_size):
            for i in range(self.coupled_integration_dim):
                self._coupled_offsets[b, i, :] = b + slot_base[i] + shifts

        self._coupled_offsets = self._coupled_offsets.astype(int)

    def construct_integrated_couplings(self, batch=None, bsize=None):
        out = super().construct_integrated_couplings(batch, bsize)
        if self.refresh_period is not None and not self.coupled_mode:
            # Keep a reference to this window's first-period field (no copy): when
            # inference then switches to set_coupled_fields it is the previous field.
            self._period0_source = self.integrated_couplings
        return out

    def reset_coupler(self):
        super().reset_coupler()
        self._previous_fields = None
        # _period0_source is kept: it is the dataset's field for the initial condition, which a
        # reset between rollouts from that IC (e.g. ensemble batches) does not change. Reading
        # the next initial condition replaces it.

    def set_coupled_fields(self, coupled_fields: th.tensor):
        """
        Set the data for the coupled field for the next iteration of the dataloader.
        Instead of loading data from the dataset the data from coupled_fields will
        be returned instead.

        Parameters
        ----------
        coupled_fields: th.tensor
            The data to use when the dataloader requests coupled fields. Expected
            format is [B, F, T, C, H, W]
        """
        # create buffer for coupling
        coupled_fields = coupled_fields[
            :, :, :, self.coupled_channel_indices, :, :
        ] 
        self.preset_coupled_fields = th.empty(
            [coupled_fields.shape[0], self.spatial_dims[0], self.coupled_integration_dim, self.timevar_dim]
            + list(self.spatial_dims[1:])
        )
        # we use a constant set of values so we just copy time 0

        # broadcast the first time step to the entire coupled integration dimension
        self.preset_coupled_fields[:, :, :, :, :, :] = coupled_fields[:, :, :1, :, :, :]
        if self.refresh_period is not None:
            # Slots of the previous period (the prestep) keep the field the last
            # window used; later periods cannot be known yet and repeat this one.
            if self._previous_fields is None and self._period0_source is not None:
                src = th.from_numpy(self._period0_source[:, self.presteps])  # [B, C, F, H, W]
                # cast: the dataset buffer may be float64 while the preset buffer is float32
                self._previous_fields = src.permute(0, 2, 1, 3, 4)[:, :, None].to(
                    self.preset_coupled_fields.dtype
                )
            if self._previous_fields is not None:
                prev = np.flatnonzero(self._slot_periods < 0)
                self.preset_coupled_fields[:, :, th.as_tensor(prev)] = self._previous_fields
            self._previous_fields = coupled_fields[:, :, :1].clone()
        if self.time_first:
            self.preset_coupled_fields = self.preset_coupled_fields.permute(2, 0, 3, 1, 4, 5)
        # flag for construct integrated coupling method to use this array
        self.coupled_mode = True


class TrailingAverageCoupler(BaseCoupler):
    """
    coupler used to interface two components of the earth system

    Trailing average coupler uses coupled input times as the right side of
    an average that is taken over an "averaging_window" window size.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        batch_size: int,
        variables: Sequence,
        presteps: int = 0,
        input_time_dim: int = 2,
        output_time_dim: int = 2,
        averaging_window: str = "24h",
        input_times: Sequence = [pd.Timedelta("24h"), pd.Timedelta("48h")],
        prepared_coupled_data=True,
    ):
        """
        Parameters
        ----------
        dataset: xr.Dataset
            xarray Dataset that holds coupled data
        batch_size: int
            number of batch size during training.
            forecasting batch size should be 1
        variables: Sequence
            sequence of strings that indicate the coupled variable
            names in the dataset
        presteps: int, optional
            the number of model steps used to initialize the hidden state.
            If not using a GRU, prestep is 0, default 0
        input_time_dim: int, optional
            number of input times into the model, default 2
        output_time_dim: int, optional
            number of output times for each model step, default 2
        averaging_window: str, optional
            period over which coupled data is averaged before sent back to model, default "24h"
        input_times: Sequence, optional
            sequence of pandas Timedelta objects that indicate which times are to be coupled,
            default [pd.Timedelta("24h"), pd.Timedelta("48h")]
        prepared_coupled_data: boolean, optional
            If True assumes data in dataset has been prepared appropriately for training:
            averages have already been calculated so that each time step denotes
            the right side of a averaging_window window.
            This is highly recommended for training, default True
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            variables=variables,
            presteps=presteps,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
            input_times=input_times,
            prepared_coupled_data=prepared_coupled_data,
        )

        # TrailingAverageCoupler-specific attributes
        self.averaging_window = pd.Timedelta(averaging_window)

        if self.use_zarr:
            cf_dates = cftime.num2pydate(
                self.ds["time"][:],
                units=self.ds["time"].attrs["units"],
                calendar=self.ds["time"].attrs["calendar"],
            )
            dates = [np.datetime64(date.isoformat()) for date in cf_dates]
            self.time_da = np.asarray(dates)
        else:
            self.time_da = self.ds["time"].values
        self._set_time_increments()

    def compute_coupled_indices(self, interval, data_time_step):
        """
        Called by CoupledDataset to compute static indices for training
        samples

        Parameters
        ----------
        interval: int
            ratio of dataset timestep to model dt
        data_time_step:
            dataset timestep
        """
        # create array of static coupled offsets that accompany each batch
        self._coupled_offsets = np.empty(
            [self.batch_size, self.coupled_integration_dim, len(self.input_times)]
        )
        for b in range(self.batch_size):
            for i in range(self.coupled_integration_dim):
                self._coupled_offsets[b, i, :] = (
                    b
                    + (self.input_time_dim * i + 1) * interval
                    + np.array([ts / data_time_step for ts in self.input_times])
                )

        self._coupled_offsets = self._coupled_offsets.astype(int)

    def _set_time_increments(self):
        # get the dt of the dataset
        dt = pd.Timedelta(self.time_da[1] - self.time_da[0]).total_seconds()
        # assert that the time increments are divisible by the dt of the dataset
        if np.any([t.total_seconds() % dt != 0 for t in self.input_times]):
            raise ValueError(
                f"Coupled input times {self.input_times} "
                f"({[t.total_seconds() for t in self.input_times]} in secs) are not divisible by dataset dt: {dt}"
            )
        self.time_increments = [t.total_seconds() / dt for t in self.input_times]

    def setup_coupling(self, coupled_module):
        # Call parent method first to set basic coupling
        super().setup_coupling(coupled_module)

        # TrailingAverageCoupler-specific setup
        # find averaging periods from component output
        averaging_window_max_indices = [
            i // pd.Timedelta(coupled_module.time_step) for i in self.input_times
        ]
        di = averaging_window_max_indices[0]
        # TODO: Now support output_time_dim =/= input_time_dim, but presteps need to be 0, will add support for presteps>0
        averaging_slices = []
        for j in range(self.coupled_integration_dim):
            averaging_slices.append([])
            for i, r in enumerate(averaging_window_max_indices):
                averaging_slices[j].append(
                    slice(
                        self.input_time_dim * j * di + i * di,
                        self.input_time_dim * j * di + r,
                    )
                )
        self.averaging_slices = averaging_slices

    def set_coupled_fields(self, coupled_fields: th.tensor):
        """
        Set the data for the coupled field for the next iteration of the dataloader.
        Instead of loading data from the dataset the data from coupled_fields will
        be returned instead.

        Parameters
        ----------
        coupled_fields: th.tensor
            The data to use when the dataloader requests coupled fields. Expected
            format is [B, F, T, C, H, W]
        """
        coupled_fields = coupled_fields[:, :, :, self.coupled_channel_indices, :, :]
        # TODO: Now support output_time_dim =/= input_time_dim, but presteps need to be 0, will add support for presteps>0
        coupled_averaging_periods = []
        for j in range(self.coupled_integration_dim):
            averaging_periods = [
                coupled_fields[:, :, s, :, :, :].mean(dim=2, keepdim=True)
                for s in self.averaging_slices[j]
            ]
            coupled_averaging_periods.append(th.concat(averaging_periods, dim=3))
        self.preset_coupled_fields = th.concat(
            coupled_averaging_periods, dim=2
        )
        if self.time_first:
            self.preset_coupled_fields = self.preset_coupled_fields.permute(2, 0, 3, 1, 4, 5)
        # flag for construct integrated coupling method to use this array
        self.coupled_mode = True


class WindowMeanCoupler(BaseCoupler):
    """
    Coupler that forces each model step with the time mean of the coupled fields over
    that step's window [t, t + window], t being the step's input time.

    The mean is the trapezoidal average of the stored samples in the window (half
    weight on the two end samples), so over a 24 h window every hour of the day gets
    the same weight whatever the start hour. It is computed on the fly from the
    stored instantaneous fields, read and normalized as the base class reads them, so
    no pre-averaged variables are needed. The window equals the model step: the
    prestep sees the window ending at the first step's input time t0, step k the
    window [t0 + k * step, t0 + (k + 1) * step], and in rollouts the forcing is
    renewed every window.

    Coupled inference: start_rollout seeds the series with the first window's dataset
    couplings and the sender's state at t0; set_coupled_fields then averages each
    window's sender outputs together with the sender's state at the window start, and
    the prestep slots keep the means of the previous windows.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        batch_size: int,
        variables: Sequence,
        presteps: int = 0,
        input_time_dim: int = 1,
        output_time_dim: int = 1,
        input_times: Sequence = ("24h",),
    ):
        """
        Parameters
        ----------
        dataset: xr.Dataset
            xarray Dataset or zarr Group that holds the coupled data
        batch_size: int
            number of samples per dataset item
        variables: Sequence
            coupled variable names in the dataset
        presteps: int, optional
            the number of model steps used to initialize the hidden state, default 0
        input_time_dim: int, optional
            number of input times per model step; only 1 is supported, default 1
        output_time_dim: int, optional
            number of output times per model call, default 1
        input_times: Sequence, optional
            one entry: the averaging window, which must equal the model time step,
            default ("24h",)
        """
        if input_time_dim != 1:
            raise ValueError(f"WindowMeanCoupler needs input_time_dim 1, got {input_time_dim}")
        if len(input_times) != 1:
            raise ValueError(f"WindowMeanCoupler takes one input time (the window), got {list(input_times)}")
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            variables=variables,
            presteps=presteps,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
            input_times=input_times,
        )
        self._weights = None
        self._frame_offsets = None
        self._sender_last = None
        self._previous_means = None

    @staticmethod
    def _trapezoid_weights(n):
        """Weights of the n + 1 samples spanning a window of n sample intervals."""
        w = np.ones(n + 1) / n
        w[[0, -1]] *= 0.5
        return w

    def compute_coupled_indices(self, interval, data_time_step):
        """
        Called by CoupledDataset to compute static indices for training samples.
        Offsets count data steps from each sample's first frame, which is the first
        prestep input; slot i averages the n + 1 frames from its input time on.

        Parameters
        ----------
        interval: int
            ratio of model time step to dataset time step
        data_time_step:
            dataset time step
        """
        step = self.input_time_dim * interval
        n = self.input_times[0] / pd.Timedelta(data_time_step)
        if n != step:
            raise ValueError(
                f"WindowMeanCoupler window {self.input_times[0]} must equal the model "
                f"time step ({step} data steps of {data_time_step})"
            )
        n = int(n)
        t0 = interval * (self.input_time_dim * (1 + self.presteps) - 1)
        slot_start = t0 + (np.arange(self.coupled_integration_dim) - self.presteps) * step
        # [B, slots, n + 1]: the frames each slot averages
        self._frame_offsets = (
            np.arange(self.batch_size)[:, None, None]
            + slot_start[None, :, None]
            + np.arange(n + 1)[None, None, :]
        ).astype(int)
        # The base class reads one offset per (sample, slot): give it one slot per frame.
        self._coupled_offsets = self._frame_offsets.reshape(self.batch_size, -1, 1)
        self._weights = self._trapezoid_weights(n)

    def construct_integrated_couplings(self, batch=None, bsize=None):
        """
        Window means of the coupled fields for every slot of the batch's samples,
        [slots, B, C, F, H, W].
        """
        if self.coupled_mode:
            return self.preset_coupled_fields
        # The base class reads, normalizes and casts the coupled data at one offset per
        # (sample, slot). Read the window frames through it as one slot per frame, then
        # average each slot's frames.
        n_slots, n_frames = self._frame_offsets.shape[1:]
        self.coupled_integration_dim = n_slots * n_frames
        try:
            frames = super().construct_integrated_couplings(batch, bsize)  # [slots * frames, B, C, F, H, W]
        finally:
            self.coupled_integration_dim = n_slots
        frames = frames.reshape((n_slots, n_frames) + frames.shape[1:])
        return np.tensordot(self._weights, frames, axes=(0, 1)).astype(frames.dtype)

    def reset_coupler(self):
        super().reset_coupler()
        self._sender_last = None
        self._previous_means = None

    def start_rollout(self, first_couplings, sender_state):
        """
        Seed coupled inference before the first set_coupled_fields.

        Parameters
        ----------
        first_couplings: th.tensor
            The couplings the receiver got for its first window from the dataset,
            [slots, B, C, F, H, W]; its prestep slots carry over to the next window.
        sender_state: th.tensor
            The sender's fields at the first window's start, [B, F, T, C, H, W] in the
            sender's output channel order (its last time is used).
        """
        self._previous_means = th.as_tensor(first_couplings)[: self.presteps].permute(1, 3, 0, 2, 4, 5).float().cpu()
        self._sender_last = sender_state[:, :, -1:, self.coupled_channel_indices].float().cpu().clone()

    def set_coupled_fields(self, coupled_fields: th.tensor):
        """
        Set the coupled forcing of the next window from the sender's outputs over it.

        Parameters
        ----------
        coupled_fields: th.tensor
            The sender's outputs over the window, [B, F, T, C, H, W], at the data's
            time resolution; the step slots split T evenly, each averaging its outputs
            with the field just before them.
        """
        if self._sender_last is None:
            raise RuntimeError("WindowMeanCoupler.start_rollout must seed coupled inference first")
        x = coupled_fields[:, :, :, self.coupled_channel_indices].float().cpu()
        n_steps = self.coupled_integration_dim - self.presteps
        n = len(self._weights) - 1
        if x.shape[2] != n_steps * n:
            raise ValueError(
                f"expected {n_steps * n} sender outputs ({n_steps} windows of {n} samples, "
                f"as in training), got {x.shape[2]}"
            )
        w = th.as_tensor(self._weights, dtype=x.dtype).view(1, 1, -1, 1, 1, 1)
        means = []
        for s in range(n_steps):
            frames = th.cat([self._sender_last, x[:, :, s * n : (s + 1) * n]], dim=2)
            means.append((frames * w).sum(dim=2, keepdim=True))
            self._sender_last = x[:, :, (s + 1) * n - 1 : (s + 1) * n].clone()
        slots = th.cat([self._previous_means] + means, dim=2)  # [B, F, slots, C, H, W]
        self._previous_means = slots[:, :, slots.shape[2] - self.presteps :]
        self.preset_coupled_fields = slots.permute(2, 0, 3, 1, 4, 5)
        self.coupled_mode = True
