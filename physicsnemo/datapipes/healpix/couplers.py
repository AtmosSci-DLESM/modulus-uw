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
from typing import Sequence

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
        self.incoming_coupled_scaling = None
        self.outgoing_coupled_scaling = None
        self.incoming_variables = None
        self.outgoing_variable_order = None
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
                raise ValueError(
                    f"Missing variables in dataset for coupling: {missing_variables}"
                )
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

    def set_coupled_scaling(self, incoming_coupled_scaling):
        """Set incoming denorm stats; outgoing renorm uses ``coupled_scaling`` tiled.

        Requires :meth:`set_scaling` and :meth:`setup_coupling`. Incoming keys
        follow ``incoming_variables``; outgoing is reordered to
        ``outgoing_variable_order`` and tiled across ``input_times``.
        """
        if self.coupled_scaling is None or self.incoming_variables is None:
            raise RuntimeError(
                "set_scaling and setup_coupling must be called before "
                "set_coupled_scaling."
            )
        self.incoming_coupled_scaling = self._prepare_incoming_coupled_scaling(
            incoming_coupled_scaling
        )
        self.outgoing_coupled_scaling = self._prepare_outgoing_coupled_scaling()

    def _prepare_incoming_coupled_scaling(self, incoming_coupled_scaling):
        """Parse incoming stats and broadcast to [B, F, T, C, H, W]."""
        return self._broadcast_stats(
            *self._coupled_scaling_to_mean_std(
                incoming_coupled_scaling, self.incoming_variables
            )
        )

    def _prepare_outgoing_coupled_scaling(self):
        """Reorder ``coupled_scaling`` to post-index order and tile across input times."""
        var_idx = {v: i for i, v in enumerate(self.variables)}
        order = [var_idx[v] for v in self.outgoing_variable_order]
        cs_mean = np.asarray(self.coupled_scaling["mean"], dtype=np.float32).ravel()
        cs_std = np.asarray(self.coupled_scaling["std"], dtype=np.float32).ravel()
        n = len(self.input_times)
        return self._broadcast_stats(
            np.tile(cs_mean[order], n), np.tile(cs_std[order], n)
        )

    @staticmethod
    def _coupled_scaling_to_mean_std(scaling, keys: Sequence[str]):
        if isinstance(scaling, (xr.Dataset, xr.DataArray)):
            selected = scaling.sel(index=list(keys))
            return (
                np.asarray(selected["mean"].to_numpy(), dtype=np.float32).ravel(),
                np.asarray(selected["std"].to_numpy(), dtype=np.float32).ravel(),
            )
        return (
            np.asarray([scaling[k]["mean"] for k in keys], dtype=np.float32),
            np.asarray([scaling[k]["std"] for k in keys], dtype=np.float32),
        )

    @staticmethod
    def _broadcast_stats(mean, std) -> dict:
        mean = np.asarray(mean, dtype=np.float32).ravel()
        std = np.asarray(std, dtype=np.float32).ravel()
        if np.any(std == 0):
            raise ValueError("scaling std must be non-zero for all channels")
        # Broadcast to channel dimension [B, F, T, C, H, W] to match the shape of the coupled fields
        return {
            "mean": th.as_tensor(mean).view(1, 1, 1, -1, 1, 1),
            "std": th.as_tensor(std).view(1, 1, 1, -1, 1, 1),
        }

    @staticmethod
    def _denormalize(x: th.Tensor, scaling: dict) -> th.Tensor:
        mean = scaling["mean"].to(device=x.device, dtype=x.dtype)
        std = scaling["std"].to(device=x.device, dtype=x.dtype)
        return x * std + mean

    @staticmethod
    def _renormalize(x: th.Tensor, scaling: dict) -> th.Tensor:
        mean = scaling["mean"].to(device=x.device, dtype=x.dtype)
        std = scaling["std"].to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def rescale_coupled_fields_through_physical(
        self, coupled_fields: th.Tensor, physical_op
    ) -> th.Tensor:
        """
        Rescale coupled fields through physical space.
        Fields may overflow when denormalized, so we run in float32 mid-flight.
        float32 → Denorm → ``physical_op`` → renorm → restore input dtype.

        Parameters
        ----------
        coupled_fields: th.Tensor
            The coupled fields to rescale.
        physical_op: callable
            The physical operation to apply to the coupled fields.
        """
        orig_dtype = coupled_fields.dtype
        physical_fields = self._denormalize(
            coupled_fields.float(), self.incoming_coupled_scaling
        )
        transformed_fields = physical_op(physical_fields)
        normalized_fields = self._renormalize(
            transformed_fields, self.outgoing_coupled_scaling
        )
        return normalized_fields.to(dtype=orig_dtype)

    def _store_preset_coupled_fields(self, coupled_fields: th.Tensor, physical_op):
        """Index channels, optionally rescale through physical space, then store."""
        coupled_fields = coupled_fields[:, :, :, self.coupled_channel_indices, :, :]
        if self.incoming_coupled_scaling is not None:
            preset = self.rescale_coupled_fields_through_physical(
                coupled_fields, physical_op
            )
        else:
            preset = physical_op(coupled_fields)
        if self.time_first:
            preset = preset.permute(2, 0, 3, 1, 4, 5)
        self.preset_coupled_fields = preset
        self.coupled_mode = True

    def setup_coupling(self, coupled_module):
        """Bind incoming prognostic names in the receiving variable order."""
        outputs = list(coupled_module.output_variables)
        incoming = []
        for variable in self.variables:
            name = variable if variable in outputs else variable.rsplit("-", 1)[0]
            if name not in outputs:
                raise ValueError(f"Missing variable in coupled module: {variable}")
            incoming.append(name)
        self.coupled_channel_indices = [outputs.index(name) for name in incoming]
        self.incoming_variables = incoming
        self.outgoing_variable_order = list(self.variables)
        inputs = list(getattr(coupled_module, "input_variables", outputs))
        self.incoming_input_indices = [
            inputs.index(name) if name in inputs else None for name in incoming
        ]

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
            (bsize, self.coupled_integration_dim, self.timevar_dim) + self.spatial_dims,
            dtype=np.float32,
        )

        index_range = slice(
            batch["time"].start,
            batch["time"].start + self._coupled_offsets[-1, -1, -1] + 1,
        )

        # extract coupled variables
        if self.use_zarr:
            ds_index_range = self.ds["inputs"].oindex[
                index_range, self.ds_variable_indices, :, :, :
            ]
        else:
            ds_index_range = (
                self.ds["inputs"]
                .sel(channel_in=self.variables)
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
                    if self.use_zarr:
                        coupling_temp = ds_index_range[
                            self._coupled_offsets[b, i, :], :
                        ]
                    else:
                        coupling_temp = ds_index_range.isel(
                            time=self._coupled_offsets[b, i, :]
                        ).to_numpy()
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

    The latest supplied source time is held across the complete receiving
    integration. Earlier source times remain available to the source model;
    they are not replayed within the receiving integration.
    """

    snapshot_policy = "latest"

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
                self._coupled_offsets[b, i, :] = b + np.array(
                    [ts / data_time_step for ts in self.input_times]
                )

        self._coupled_offsets = self._coupled_offsets.astype(int)

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

        def _broadcast_latest_time(fields: th.Tensor) -> th.Tensor:
            # Clone so later mutations of coupled_fields cannot alter presets.
            return (
                fields[:, :, -1:]
                .expand(-1, -1, self.coupled_integration_dim, -1, -1, -1)
                .clone()
            )

        self._store_preset_coupled_fields(coupled_fields, _broadcast_latest_time)


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
        integration_time_dim=None,
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
        integration_time_dim: int, optional
            Predicted time steps per native model call, used for forcing stride
            and number of integration blocks. Defaults to input_time_dim.
        prepared_coupled_data: boolean, optional
            If True assumes data in dataset has been prepared appropriately for training:
            averages have already been calculated so that each time step denotes
            the right side of a averaging_window window.
            This is highly recommended for training, default True
        """
        self.integration_time_dim = (
            input_time_dim if integration_time_dim is None else integration_time_dim
        )
        if (
            isinstance(self.integration_time_dim, bool)
            or not isinstance(self.integration_time_dim, int)
            or self.integration_time_dim < 1
        ):
            raise ValueError("integration_time_dim must be a positive integer")
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
            raw_time = self.ds["time"][:]
            if np.issubdtype(raw_time.dtype, np.datetime64):
                self.time_da = raw_time.astype("datetime64[ns]")
            else:
                attrs = self.ds["time"].attrs
                units, calendar = attrs["units"], attrs["calendar"]
                units = units[0] if isinstance(units, (list, tuple)) else units
                calendar = (
                    calendar[0] if isinstance(calendar, (list, tuple)) else calendar
                )
                cf_dates = cftime.num2pydate(raw_time, units=units, calendar=calendar)
                self.time_da = np.asarray(
                    [np.datetime64(date.isoformat()) for date in cf_dates]
                )
        else:
            self.time_da = self.ds["time"].values
        self._set_time_increments()

    def _compute_coupled_integration_dim(self):
        """Count native calls using their output stride rather than history size."""
        return self.presteps + max(self.output_time_dim // self.integration_time_dim, 1)

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
                    + (self.integration_time_dim * i + 1) * interval
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
                        self.integration_time_dim * j * di + i * di,
                        self.integration_time_dim * j * di + r,
                    )
                )
        self.averaging_slices = averaging_slices

    def set_coupled_fields(self, coupled_fields: th.tensor, initial_state=None):
        """
        Set the data for the coupled field for the next iteration of the dataloader.
        Instead of loading data from the dataset the data from coupled_fields will
        be returned instead.

        When :meth:`set_coupled_scaling` has been called, fields are
        denormalized, averaged in physical space, then renormalized.

        Parameters
        ----------
        coupled_fields: th.tensor
            The data to use when the dataloader requests coupled fields. Expected
            format is [B, F, T, C, H, W]
        initial_state: th.tensor, optional
            Source prognostic history in the same format, using source input
            channel order. Its last state supplies the inclusive left endpoint;
            each averaging slice also includes its right endpoint. Omit to
            retain the existing forecast-only averaging convention.
        """

        if initial_state is not None:
            indices = self.incoming_input_indices
            if any(index is None for index in indices):
                raise ValueError(
                    "inclusive forcing needs each source variable in initial state"
                )
            # Align the source input to the source output channel ABI before the
            # existing index/physical-space path. Only coupled fields are read.
            initial = coupled_fields.new_zeros(
                *initial_state.shape[:2],
                1,
                coupled_fields.shape[3],
                *initial_state.shape[-2:],
            )
            initial[:, :, :, self.coupled_channel_indices] = initial_state[
                :, :, -1:, indices
            ]
            coupled_fields = th.cat((initial, coupled_fields), dim=2)

        def _trailing_average(fields: th.Tensor) -> th.Tensor:
            # TODO: Now support output_time_dim =/= input_time_dim, but presteps
            # need to be 0, will add support for presteps>0
            coupled_averaging_periods = []
            for j in range(self.coupled_integration_dim):
                averaging_periods = [
                    fields[
                        :,
                        :,
                        slice(s.start, s.stop + (initial_state is not None)),
                        :,
                        :,
                        :,
                    ].mean(dim=2, keepdim=True)
                    for s in self.averaging_slices[j]
                ]
                coupled_averaging_periods.append(th.concat(averaging_periods, dim=3))
            return th.concat(coupled_averaging_periods, dim=2)

        self._store_preset_coupled_fields(coupled_fields, _trailing_average)
