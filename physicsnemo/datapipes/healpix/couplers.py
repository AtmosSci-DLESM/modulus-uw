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

"""HEALPix couplers for exchanging fields between Earth-system components.

Overview
--------
A coupler takes prognostic outputs from one component (e.g. atmosphere) and
prepares them as forcing for another (e.g. ocean). Two strategies are provided:

* ``ConstantCoupler`` — broadcast the first available time step across the
  coupled integration window.
* ``TrailingAverageCoupler`` — average over trailing windows whose right edges
  are given by ``input_times`` (e.g. 48 h and 96 h).

Tensor layouts
--------------
``set_coupled_fields`` expects::

    [B, F, T, C, H, W]

With default ``time_first=True``, ``construct_integrated_couplings`` returns::

    [I, B, timevar, F, H, W]

where ``timevar = len(variables) * len(input_times)`` in **period-major** order::

    [p0_v0, p0_v1, ..., p1_v0, p1_v1, ...]

Example for ``variables=["z1000", "ws10m"]``, ``input_times=["48h", "96h"]``::

    [z1000@48h, ws10m@48h, z1000@96h, ws10m@96h]

Call ``set_coupled_scaling(incoming)`` after ``set_scaling`` / ``setup_coupling``
to denorm→operate→renorm (float32 mid-flight); omit it for the legacy path.
"""

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import cftime
import numpy as np
import pandas as pd
import torch as th
import xarray as xr
import zarr as zr

from physicsnemo.datapipes.healpix.coupling_ops import (
    CONSTANT,
    TRAILING_AVERAGE,
    CouplingOp,
    denormalize,
    renormalize,
    rescale_through_physical,
)

logger = logging.getLogger(__name__)


class BaseCoupler(ABC):
    """
    Base class for couplers used to interface two components of earth system.

    This class contains common functionality shared by different coupler implementations.
    """

    #: Coupling strategy this class implements, one of the mode names in
    #: :mod:`physicsnemo.datapipes.healpix.coupling_ops`. Set by subclasses.
    coupling_method: str = None

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
        # Ops built by coupling_op(), keyed by rescale_in_physical_space. Built
        # lazily and dropped whenever the configuration they captured changes.
        self._coupling_ops = {}
        self.incoming_coupled_scaling = None
        self.outgoing_coupled_scaling = None
        # Set by setup_coupling: source-module channel names for denorm lookups,
        # and sink-side names in the same post-index order for renorm lookups.
        self.incoming_variables = None
        self.outgoing_variable_order = None

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

    def _invalidate_coupling_ops(self):
        """Drop cached ops so the next :meth:`coupling_op` rebuilds them.

        Called whenever state that :meth:`CouplingOp.from_coupler` captures
        changes, so a cached op can never outlive the configuration it read.
        """
        self._coupling_ops.clear()

    @property
    def incoming_coupled_scaling(self):
        """Statistics that normalized the source component's fields, or None."""
        return self._incoming_coupled_scaling

    @incoming_coupled_scaling.setter
    def incoming_coupled_scaling(self, scaling):
        self._incoming_coupled_scaling = scaling
        self._invalidate_coupling_ops()

    @property
    def outgoing_coupled_scaling(self):
        """Statistics used to renormalize the coupled result, or None."""
        return self._outgoing_coupled_scaling

    @outgoing_coupled_scaling.setter
    def outgoing_coupled_scaling(self, scaling):
        self._outgoing_coupled_scaling = scaling
        self._invalidate_coupling_ops()

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

    _denormalize = staticmethod(denormalize)
    _renormalize = staticmethod(renormalize)

    def rescale_coupled_fields_through_physical(
        self, coupled_fields: th.Tensor, physical_op
    ) -> th.Tensor:
        """
        Rescale coupled fields through physical space.

        Parameters
        ----------
        coupled_fields: th.Tensor
            The coupled fields to rescale.
        physical_op: callable
            The physical operation to apply to the coupled fields.
        """
        return rescale_through_physical(
            coupled_fields,
            physical_op,
            self.incoming_coupled_scaling,
            self.outgoing_coupled_scaling,
        )

    def coupling_op(self, rescale_in_physical_space: bool = True) -> CouplingOp:
        """Return this coupler's coupling as a standalone differentiable module.

        The returned :class:`~physicsnemo.datapipes.healpix.coupling_ops.CouplingOp`
        carries the coupler's channel indices, time reduction, and scaling
        statistics, so callers that need to couple fields inside an autograd
        graph (distributed inference, coupled training) can run exactly what the
        dataloader runs instead of reimplementing it.

        The op is built once per ``rescale_in_physical_space`` value and cached,
        because ``set_coupled_fields`` runs this on every dataloader step and
        constructing an :class:`~torch.nn.Module` per step is pure overhead.
        Reconfiguring the coupler drops the cache, so the op cannot go stale. As
        a consequence the instance is shared between callers: moving it with
        ``.to()`` sticks, and mutating it is visible to everyone.

        Parameters
        ----------
        rescale_in_physical_space: bool, optional
            Whether the op should act on this coupler's scaling statistics.
            Pass False to reduce in normalized space with the statistics still
            attached, which reproduces the numbers a caller produced before
            physical-space rescaling was available. Default True.

        Returns
        -------
        CouplingOp
            Reflecting the coupler's configuration as of the last
            ``setup_coupling`` or ``set_coupled_scaling`` call.
        """
        op = self._coupling_ops.get(rescale_in_physical_space)
        if op is None:
            op = CouplingOp.from_coupler(
                self, rescale_in_physical_space=rescale_in_physical_space
            )
            self._coupling_ops[rescale_in_physical_space] = op
        return op

    def _store_preset_coupled_fields(self, coupled_fields: th.Tensor):
        """Apply the coupling operation and buffer the result for the dataloader."""
        self.preset_coupled_fields = self.coupling_op()(coupled_fields)
        self.coupled_mode = True

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
        # Order follows source output_channels (may differ from self.variables).
        channel_indices = []
        incoming_variables = []
        outgoing_variable_order = []
        for i, oc in enumerate(output_channels):
            for v in self.variables:
                if ("-" not in v and oc == v) or (
                    oc == "-".join(v.split("-")[:-1])
                ):
                    channel_indices.append(i)
                    incoming_variables.append(oc)
                    outgoing_variable_order.append(v)
                    break
        if len(self.variables) != len(channel_indices):
            missing_channels = set(self.variables) - set(outgoing_variable_order)
            raise ValueError(f"Missing variables in coupled module: {missing_channels}")
        self.coupled_channel_indices = channel_indices
        self.incoming_variables = incoming_variables
        self.outgoing_variable_order = outgoing_variable_order
        self._invalidate_coupling_ops()

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

        index_range = slice(
            batch["time"].start,
            batch["time"].start + self._coupled_offsets[-1, -1, -1] + 1,
        )

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

    constant coupler will take the the coupled field at integration time and
    force the model with this field consistently
    """

    coupling_method = CONSTANT

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
        **kwargs,
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
            **kwargs,
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
        self._store_preset_coupled_fields(coupled_fields)


class TrailingAverageCoupler(BaseCoupler):
    """
    coupler used to interface two components of the earth system

    Trailing average coupler uses coupled input times as the right side of
    an average that is taken over an "averaging_window" window size.
    """

    coupling_method = TRAILING_AVERAGE

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
        **kwargs,
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
            **kwargs,
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
        # super().setup_coupling already invalidated, but averaging_slices is
        # assigned after that call, so the windows need their own invalidation.
        self._invalidate_coupling_ops()

    def set_coupled_fields(self, coupled_fields: th.tensor):
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
        """
        self._store_preset_coupled_fields(coupled_fields)
