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

Coupled-field scaling (denorm → operate → renorm)
-------------------------------------------------
During inference, coupled fields are usually already z-scored with the *source*
component's instantaneous stats (e.g. ``z1000``, ``ws10m``). The *sink*
component, however, often has trailing-window fields with different statistical
distributions due to operations such as averaging (e.g. ``z1000-48H``, ``ws10-48H``).

Averaging in the wrong z-score space is an affine mismatch, not float noise::

    # Wrong: average already-normalized with instant μ/σ, then treat as trailing
    denorm_trailing( mean( norm_instant(x) ) )
        ≠  mean(x)     # bias on the order of tens of meters for z1000

When ``set_coupled_scaling(incoming, outgoing)`` has been called,
``set_coupled_fields`` instead works in physical units::

    source (instant z-score)          sink (trailing z-score)
    ------------------------          -----------------------
    x_z = (x - μ_in) / σ_in
              │
              ▼
        denorm with μ_in, σ_in   ──►  x_phys
              │
              ▼
        ConstantCoupler: broadcast t=0
        TrailingAverageCoupler: mean over windows
              │
              ▼
        renorm with μ_out, σ_out ──►  y_z = (y_phys - μ_out) / σ_out

ASCII flow for ``TrailingAverageCoupler``::

    set_coupled_fields(x_z)          # [B, F, T, C, H, W], C = len(variables)
            │
            ├─► denormalize:  x_phys = x_z * σ_in + μ_in
            │
            ├─► trailing mean over each input_times window
            │         └── concat period-major → [B, F, I, timevar, H, W]
            │
            └─► renormalize:  y_z = (x_avg - μ_out) / σ_out

Omit both scalings to keep the legacy path (operate directly in whatever space
the caller already provided).

Examples
--------
**1. Instant → trailing (typical atmos → ocean forcing)**

Call :meth:`setup_coupling` first (stores the source module's matched
``output_variables`` as incoming keys). Outgoing keys are ``self.variables``
repeated once per ``input_times`` entry (length ``self.output_channels``)::

    from omegaconf import OmegaConf

    scaling = OmegaConf.load("configs/data/scaling/hpx64.yaml")
    # or: scaling = cfg.data.scaling

    coupler = TrailingAverageCoupler(
        dataset=ocean_ds,
        batch_size=1,
        variables=["z1000-48H", "ws10m-48H"],
        input_times=["48h", "96h"],
        averaging_window="48h",
    )
    coupler.setup_coupling(atmos_module)  # saves incoming names, e.g. z1000, ws10m
    coupler.set_coupled_scaling(scaling, scaling)
    # incoming keys: atmos output_variables matched in setup_coupling
    # outgoing keys: [z1000-48H, ws10m-48H, z1000-48H, ws10m-48H]

**2. Same object as ``set_scaling`` (xarray ``scaling_da``)**

The Dataset built inside CoupledDataset also works::

    scaling_df = pd.DataFrame.from_dict(OmegaConf.to_object(scaling)).T
    scaling_da = scaling_df.to_xarray().astype("float32")
    coupler.setup_coupling(atmos_module)
    coupler.set_coupled_scaling(scaling_da, scaling_da)

**3. ConstantCoupler**

Same method; the physical-space op broadcasts time 0 instead of averaging::

    coupler = ConstantCoupler(
        dataset=ds,
        batch_size=1,
        variables=["sst"],
        input_times=["0h"],
    )
    coupler.setup_coupling(ocean_module)
    coupler.set_coupled_scaling(scaling, scaling)

Related tests
-------------
* ``test_healpix_coupler_coupled_field_scaling.py`` — API, recovery of physical
  trailing means, adversarial edge cases.
* ``test_healpix_coupler_instant_vs_trailing_scaling.py`` — quantifies the
  mismatch when averaging in the wrong z-score space.
* ``test_healpix_coupler_znorm_stability.py`` — matched-stats float32 stability.
"""

import logging
from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Union

import cftime
import numpy as np
import pandas as pd
import torch as th
import xarray as xr
import zarr as zr

logger = logging.getLogger(__name__)

# Variable-keyed scaling (hpx64.yaml / cfg.data.scaling), OmegaConf, or
# the xarray Dataset produced by ``pd.DataFrame.from_dict(scaling).T.to_xarray()``.
CoupledFieldScaling = Union[Mapping, xr.Dataset, xr.DataArray, None]


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

        Notes
        -----
        Optional denorm/renorm stats for ``set_coupled_fields`` are configured
        separately via :meth:`set_coupled_scaling`.
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
        self.incoming_coupled_scaling = None
        self.outgoing_coupled_scaling = None
        # Set by setup_coupling: source-module channel names for denorm lookups.
        self.incoming_variables = None

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

    def set_coupled_scaling(
        self,
        incoming_coupled_scaling: CoupledFieldScaling,
        outgoing_coupled_scaling: CoupledFieldScaling,
    ):
        """Configure denorm/renorm stats used inside ``set_coupled_fields``.

        Accepts the same variable-keyed format as ``configs/data/scaling/hpx64.yaml``
        / ``cfg.data.scaling`` (and the xarray ``scaling_da`` built from it by
        CoupledDataset), so callers can pass those objects with little or no
        conversion.

        Variable keys are not passed explicitly:

        * **Incoming** (denorm): ``self.incoming_variables``, populated by
          :meth:`setup_coupling` from the coupled module's matched
          ``output_variables``.
        * **Outgoing** (renorm): ``self.variables`` duplicated once per
          ``input_times`` entry, length ``self.output_channels`` (period-major).

        When set, ``set_coupled_fields`` denormalizes with incoming stats,
        performs its physical-space operation (broadcast or trailing average),
        then renormalizes with outgoing stats.

        Parameters
        ----------
        incoming_coupled_scaling:
            Variable-keyed scaling mapping ``{name: {"mean": ..., "std": ...}}``
            (YAML / OmegaConf / dict) or an xarray Dataset/DataArray with an
            ``index`` coordinate (same object passed to :meth:`set_scaling`).
        outgoing_coupled_scaling:
            Same format as ``incoming_coupled_scaling``.
        """
        if incoming_coupled_scaling is None or outgoing_coupled_scaling is None:
            raise ValueError(
                "Both incoming_coupled_scaling and outgoing_coupled_scaling "
                "must be provided together (or omit set_coupled_scaling)."
            )
        if self.incoming_variables is None:
            raise RuntimeError(
                "setup_coupling must be called before set_coupled_scaling "
                "so incoming_variables are available from the coupled module."
            )

        incoming_variables = list(self.incoming_variables)
        # Period-major: [v0, v1, ..., v0, v1, ...] across input_times.
        outgoing_variables = list(self.variables) * len(self.input_times)
        if len(outgoing_variables) != self.output_channels:
            raise RuntimeError(
                "outgoing_variables length "
                f"{len(outgoing_variables)} != output_channels "
                f"{self.output_channels}"
            )
        if len(incoming_variables) != len(self.variables):
            raise ValueError(
                "incoming_variables from setup_coupling must have length "
                f"{len(self.variables)}; got {len(incoming_variables)}"
            )

        in_mean, in_std = self._coupled_scaling_to_mean_std(
            incoming_coupled_scaling,
            incoming_variables,
            name="incoming_coupled_scaling",
        )
        out_mean, out_std = self._coupled_scaling_to_mean_std(
            outgoing_coupled_scaling,
            outgoing_variables,
            name="outgoing_coupled_scaling",
        )
        self.incoming_coupled_scaling = self._mean_std_to_broadcast_tensors(
            in_mean, in_std, name="incoming_coupled_scaling"
        )
        self.outgoing_coupled_scaling = self._mean_std_to_broadcast_tensors(
            out_mean, out_std, name="outgoing_coupled_scaling"
        )

    # Backwards-compatible alias used briefly during the API rename.
    set_coupled_field_scaling = set_coupled_scaling

    @staticmethod
    def _coupled_scaling_to_mean_std(scaling, keys: Sequence[str], name: str):
        """Extract ordered mean/std arrays from YAML-style or scaling_da input."""
        try:
            from omegaconf import OmegaConf, DictConfig, ListConfig

            if isinstance(scaling, (DictConfig, ListConfig)) or (
                hasattr(OmegaConf, "is_config") and OmegaConf.is_config(scaling)
            ):
                scaling = OmegaConf.to_object(scaling)
        except ImportError:
            pass

        if isinstance(scaling, (xr.Dataset, xr.DataArray)):
            if "index" not in scaling.coords and "index" not in getattr(
                scaling, "dims", ()
            ):
                # DataFrame.to_xarray() uses the frame index as a dimension named
                # after the index; CoupledDataset always renames via .sel(index=...).
                index_name = scaling.dims[0] if scaling.dims else None
                if index_name is None:
                    raise TypeError(
                        f"{name}: xarray scaling object has no index coordinate"
                    )
                scaling = scaling.rename({index_name: "index"})
            available = set(np.asarray(scaling["index"].values).tolist())
            missing = [k for k in keys if k not in available]
            if missing:
                raise KeyError(
                    f"{name} missing variables {missing}; "
                    f"available={sorted(available)}"
                )
            selected = scaling.sel(index=list(keys))
            mean = np.asarray(selected["mean"].to_numpy(), dtype=np.float32).reshape(-1)
            std = np.asarray(selected["std"].to_numpy(), dtype=np.float32).reshape(-1)
            return mean, std

        if not isinstance(scaling, Mapping):
            raise TypeError(
                f"{name} must be a variable-keyed scaling mapping "
                f"(hpx64.yaml / cfg.data.scaling) or an xarray scaling_da; "
                f"got {type(scaling)}"
            )

        # Reject the old flat {"mean": [...], "std": [...]} form with a clear error.
        if (
            "mean" in scaling
            and "std" in scaling
            and not isinstance(scaling.get("mean"), Mapping)
        ):
            sample_vals = [
                v for v in scaling.values() if isinstance(v, Mapping) and "mean" in v
            ]
            if not sample_vals:
                raise TypeError(
                    f"{name} must be variable-keyed like hpx64.yaml "
                    f"(e.g. {{'z1000': {{'mean': ..., 'std': ...}}, ...}}) "
                    f"or an xarray scaling_da; flat mean/std arrays are not supported."
                )

        missing = [k for k in keys if k not in scaling]
        if missing:
            raise KeyError(
                f"{name} missing variables {missing}; "
                f"available={sorted(scaling.keys())}"
            )
        try:
            mean = np.asarray(
                [scaling[k]["mean"] for k in keys], dtype=np.float32
            ).reshape(-1)
            std = np.asarray(
                [scaling[k]["std"] for k in keys], dtype=np.float32
            ).reshape(-1)
        except (KeyError, TypeError) as exc:
            raise TypeError(
                f"{name} entries must be mappings with 'mean' and 'std' "
                f"(hpx64.yaml style); error on keys {list(keys)}: {exc}"
            ) from exc
        return mean, std

    @staticmethod
    def _mean_std_to_broadcast_tensors(mean, std, name: str) -> dict:
        if mean.size != std.size:
            raise ValueError(
                f"{name} mean and std lengths differ: {mean.size} vs {std.size}"
            )
        if np.any(std == 0):
            raise ValueError(f"{name} std must be non-zero for all channels")
        # Broadcast over [B, F, T, C, H, W]
        return {
            "mean": th.as_tensor(mean).view(1, 1, 1, -1, 1, 1),
            "std": th.as_tensor(std).view(1, 1, 1, -1, 1, 1),
        }

    @property
    def use_coupled_field_rescaling(self) -> bool:
        return (
            self.incoming_coupled_scaling is not None
            and self.outgoing_coupled_scaling is not None
        )

    def denormalize_coupled_fields(self, coupled_fields: th.Tensor) -> th.Tensor:
        """Undo incoming z-score: ``x * std_in + mean_in``."""
        if self.incoming_coupled_scaling is None:
            raise RuntimeError("incoming_coupled_scaling has not been set")
        mean = self.incoming_coupled_scaling["mean"].to(
            device=coupled_fields.device, dtype=coupled_fields.dtype
        )
        std = self.incoming_coupled_scaling["std"].to(
            device=coupled_fields.device, dtype=coupled_fields.dtype
        )
        return coupled_fields * std + mean

    def renormalize_coupled_fields(self, coupled_fields: th.Tensor) -> th.Tensor:
        """Apply outgoing z-score: ``(x - mean_out) / std_out``."""
        if self.outgoing_coupled_scaling is None:
            raise RuntimeError("outgoing_coupled_scaling has not been set")
        mean = self.outgoing_coupled_scaling["mean"].to(
            device=coupled_fields.device, dtype=coupled_fields.dtype
        )
        std = self.outgoing_coupled_scaling["std"].to(
            device=coupled_fields.device, dtype=coupled_fields.dtype
        )
        # Outgoing tensors may be [B, F, T, C, H, W] with C == timevar_dim.
        if mean.shape[3] != coupled_fields.shape[3]:
            raise ValueError(
                "outgoing_coupled_scaling channel count "
                f"{mean.shape[3]} does not match field channels "
                f"{coupled_fields.shape[3]}"
            )
        return (coupled_fields - mean) / std

    def rescale_coupled_fields_through_physical(
        self, coupled_fields: th.Tensor, physical_op
    ) -> th.Tensor:
        """Denormalize → apply ``physical_op`` in physical space → renormalize.

        Parameters
        ----------
        coupled_fields:
            Incoming fields already channel-selected, layout ``[B, F, T, C, H, W]``.
        physical_op:
            Callable ``(Tensor) -> Tensor`` operating in physical units. Output
            must keep layout ``[B, F, T', C', H, W]`` where ``C'`` matches
            ``outgoing_coupled_scaling``.
        """
        physical = self.denormalize_coupled_fields(coupled_fields)
        transformed = physical_op(physical)
        return self.renormalize_coupled_fields(transformed)

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
        # Source-module channel names in the same order as coupled_channel_indices;
        # used by set_coupled_scaling as incoming (denorm) keys.
        self.incoming_variables = [output_channels[i] for i in channel_indices]

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
        coupled_fields = coupled_fields[
            :, :, :, self.coupled_channel_indices, :, :
        ]

        def _broadcast_first_time(fields: th.Tensor) -> th.Tensor:
            # Keep layout [B, F, T, C, H, W]; expand T to coupled_integration_dim.
            return fields[:, :, :1, :, :, :].expand(
                -1, -1, self.coupled_integration_dim, -1, -1, -1
            )

        if self.use_coupled_field_rescaling:
            self.preset_coupled_fields = self.rescale_coupled_fields_through_physical(
                coupled_fields, _broadcast_first_time
            )
        else:
            self.preset_coupled_fields = th.empty(
                [
                    coupled_fields.shape[0],
                    self.spatial_dims[0],
                    self.coupled_integration_dim,
                    self.timevar_dim,
                ]
                + list(self.spatial_dims[1:])
            )
            # Broadcast the first time step across the coupled integration dim.
            self.preset_coupled_fields[:, :, :, :, :, :] = coupled_fields[
                :, :, :1, :, :, :
            ]

        if self.time_first:
            self.preset_coupled_fields = self.preset_coupled_fields.permute(
                2, 0, 3, 1, 4, 5
            )
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
        coupled_fields = coupled_fields[:, :, :, self.coupled_channel_indices, :, :]

        def _trailing_average(fields: th.Tensor) -> th.Tensor:
            # TODO: Now support output_time_dim =/= input_time_dim, but presteps
            # need to be 0, will add support for presteps>0
            coupled_averaging_periods = []
            for j in range(self.coupled_integration_dim):
                averaging_periods = [
                    fields[:, :, s, :, :, :].mean(dim=2, keepdim=True)
                    for s in self.averaging_slices[j]
                ]
                coupled_averaging_periods.append(th.concat(averaging_periods, dim=3))
            return th.concat(coupled_averaging_periods, dim=2)

        if self.use_coupled_field_rescaling:
            self.preset_coupled_fields = self.rescale_coupled_fields_through_physical(
                coupled_fields, _trailing_average
            )
        else:
            self.preset_coupled_fields = _trailing_average(coupled_fields)

        if self.time_first:
            self.preset_coupled_fields = self.preset_coupled_fields.permute(
                2, 0, 3, 1, 4, 5
            )
        # flag for construct integrated coupling method to use this array
        self.coupled_mode = True
