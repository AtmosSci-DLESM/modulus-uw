import os
import numpy as np

import pandas as pd
import torch as th
import xarray as xr
from dask.diagnostics import ProgressBar

from physicsnemo.models.module import Module

import logging
logger = logging.getLogger(__name__)


class gt_model(Module):
    """Ground Truth Model — returns normalized targets from a verification zarr for diagnostics.

    Intended for use with ``NV-dlesm/inference/coupled_forecast_hdf5_3component.py``, which
    calls ``set_output`` when ``diagnostic_model`` is True and invokes ``forward`` with the
    same list signature as other DLWP models (prognostic tensor is ``input[0]``).

    ``loads_from_checkpoint`` is False so the coupled forecast script keeps the Hydra-instantiated
    module instead of replacing it via ``Module.from_checkpoint``.
    """

    def __init__(
        self,
        gt_dataset_path: str,
        input_time_dim: int,
        output_time_dim: int,
        input_channels: list,
        output_channels: list,
        n_constants: int,
        decoder_input_channels: int,
        enable_healpixpad: bool,
        presteps: int = None,
    ):
        """
        Parameters
        ----------
        gt_dataset_path: str
            path to the ground truth dataset
        input_time_dim: int
            number of time steps in the input array
        output_time_dim: int
            number of time steps in the output array
        input_channels: list
            list of input channels. This isn't necessary for the behavior of the model, but is included for API compatibility.
        output_channels: list
            list of output channels
        n_constants: int
            number of constants. This isn't necessary for the behavior of the model, but is included for API compatibility.
        decoder_input_channels: int
            number of decoder input channels. This isn't necessary for the behavior of the model, but is included for API compatibility.
        enable_healpixpad: bool
            whether to enable healpix padding. This isn't necessary for the behavior of the model, but is included for API compatibility.
        presteps: int = None
            number of presteps to take. This isn't necessary for the behavior of the model, but is included for API compatibility.
        """
        self.diagnostic_model = True  # coupled_forecast_hdf5_3component: set_output(...)
        # When False, coupled_forecast_hdf5_3component keeps this Hydra instance instead of
        # replacing it with Module.from_checkpoint (no trainable weights in checkpoint).
        self.loads_from_checkpoint = False
        self.presteps = presteps
        
        super().__init__()
        path = os.path.normpath(str(gt_dataset_path).rstrip("/"))
        if path.endswith(".zarr"):
            self.gt_dataset = xr.open_zarr(path)
        else:
            self.gt_dataset = xr.open_dataset(gt_dataset_path, engine="zarr")

        self.forecast_dates = None
        self.integration_time_dim = None
        self.integration_counter = 0
        self.initialization_counter = 0
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.enable_healpixpad = enable_healpixpad

    def set_output(self, forecast_dates, forecast_integrations, data_module):

        # set fields necessary for gt forecasting 
        self.forecast_dates = forecast_dates 
        self.forecast_integrations = forecast_integrations
        self.mean = th.tensor(data_module.test_dataset.target_scaling["mean"].transpose(0, 2, 1, 3, 4))
        self.std = th.tensor(data_module.test_dataset.target_scaling["std"].transpose(0, 2, 1, 3, 4))
        self.output_vars = data_module.test_dataset.output_variables
        self.delta_t = pd.Timedelta(data_module.time_step)

    def forward(self, input):

        if self.forecast_dates is None:
            raise RuntimeError("gt_model: call set_output(...) before forward.")
        # Same calling convention as HEALPixResNet: list [prognostics, ...]
        ref = input[0] if isinstance(input, (list, tuple)) else input
        # check if we're on a new initialization
        if self.integration_counter == self.forecast_integrations:
            self.initialization_counter += 1
            self.integration_counter = 0

        dt = self.delta_t
        n_ch = len(self.output_vars)
        # 12 HEALPix faces, 64x64 per face at nside 64 (coupled stacks use this layout)
        output_array = th.empty(
            (1, 12, self.output_time_dim, n_ch, 64, 64),
            device=ref.device,
            dtype=ref.dtype,
        )

        init_time = self.forecast_dates[self.initialization_counter]
        for i in range(self.output_time_dim):
            valid_time = (self.integration_counter * self.output_time_dim + (i + 1)) * dt + init_time
            da = self.gt_dataset.targets.sel(channel_out=self.output_vars, time=valid_time)
            block = th.as_tensor(da.values.transpose([1, 0, 2, 3]), device=ref.device, dtype=ref.dtype)
            if block.ndim == 4:
                block = block.unsqueeze(0)
            output_array[:, :, i, :, :, :] = block

        self.integration_counter += 1

        mean = self.mean.to(device=ref.device, dtype=ref.dtype)
        std = self.std.to(device=ref.device, dtype=ref.dtype)
        output_array = (output_array - mean) / std

        return output_array


class climo_model(Module):
    """Climatology Model — returns normalized targets from a verification zarr for diagnostics.

    Intended for use with ``NV-dlesm/inference/coupled_forecast_hdf5_3component.py``, which
    calls ``set_output`` when ``diagnostic_model`` is True and invokes ``forward`` with the
    same list signature as other DLWP models (prognostic tensor is ``input[0]``).

    ``loads_from_checkpoint`` is False so the coupled forecast script keeps the Hydra-instantiated
    module instead of replacing it via ``Module.from_checkpoint``.
    """
    def __init__(
        self,
        gt_dataset_path: str,
        date_range: pd.date_range,
        input_time_dim: int,
        output_time_dim: int,
        input_channels: list,
        output_channels: list,
        n_constants: int,
        decoder_input_channels: int,
        enable_healpixpad: bool,
        presteps: int = None,
    ):
        """
        Parameters
        ----------
        gt_dataset_path: str
            path to the ground truth dataset
        date_range: pd.date_range
            date range to construct the climatology from
        input_time_dim: int
            number of time steps in the input array
        output_time_dim: int
            number of time steps in the output array
        input_channels: list
            list of input channels. This isn't necessary for the behavior of the model, but is included for API compatibility.
        output_channels: list
            list of output channels
        n_constants: int
            number of constants. This isn't necessary for the behavior of the model, but is included for API compatibility.
        decoder_input_channels: int
            number of decoder input channels. This isn't necessary for the behavior of the model, but is included for API compatibility.
        enable_healpixpad: bool
            whether to enable healpix padding. This isn't necessary for the behavior of the model, but is included for API compatibility.
        presteps: int = None
            number of presteps to take. This isn't necessary for the behavior of the model, but is included for API compatibility.
        """

        self.diagnostic_model = True  # coupled_forecast_hdf5_3component: set_output(...)
        # When False, coupled_forecast_hdf5_3component keeps this Hydra instance instead of
        # replacing it with Module.from_checkpoint (no trainable weights in checkpoint).
        self.loads_from_checkpoint = False
        self.presteps = presteps

        # initialize base class
        super().__init__()

        # set up year range and leap years in range
        self.year_range = np.arange(date_range.start, date_range.end)
        self.leap_years_in_range = np.array([y for y in self.year_range if y % 4 == 0])
                
        self.climo_dataset_path = gt_dataset_path.replace('.zarr', f'_climo_{date_range.start}-{date_range.end}.zarr')
        self.gt_dataset_path = gt_dataset_path

        # we'll use the representative year 2000 (leap year) to 
        # give our time dimension coordinates 
        self.time_coords = pd.date_range(start='2000-01-01T00:00', end='2000-12-31T22:00', freq='3h')

        # load dataset if it exists other wise construct climatology from gt dataset
        if not os.path.exists(self.climo_dataset_path):
            self.construct_climatology()
        logger.info(f'Loading climatology from {self.climo_dataset_path}')
        self.climatology = xr.open_zarr(self.climo_dataset_path)

        # declare fields necessary for forecasting
        self.forecast_dates = None
        self.integration_time_dim = None
        self.integration_counter = 0
        self.initialization_counter = 0
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.enable_healpixpad = enable_healpixpad

    def construct_climatology(self):
        logger.info(f'Constructing climatology from {self.gt_dataset_path}')
        # open ground truth dataset
        self.gt_dataset = xr.open_zarr(self.gt_dataset_path, chunks={'time': 1})

        # for each time in self.time_coords, construct a list of 
        # pandas-interpretable strings for that date and time in 
        # the date range
        timestep_climos = []
        for i,date in enumerate(self.time_coords):

            time_str = date.strftime('%Y-%m-%dT%H:%M')
            # special case for leap year
            if time_str[:10] == '2000-02-29':
                timstamps = [time_str.replace('2000',str(y)) for y in self.leap_years_in_range]
            else:
                timstamps = [time_str.replace('2000',str(y)) for y in self.year_range]

            # append to list for later concatenation         
            timestep_climos.append(self.gt_dataset.sel(time=timstamps).mean(dim='time'))
            
        # concatenate timestep climos along time dimension
        time_axis = xr.DataArray(
            self.time_coords,  # same length as timestep_climos
            dims=("time",),
            name="time",
        )
        climo = xr.concat(timestep_climos, dim=time_axis)
        # encode channels as strings 
        climo = climo.assign_coords(channel_out=climo.channel_out.astype(str),
                                    channel_in=climo.channel_in.astype(str),
                                    channel_c=climo.channel_c.astype(str))

        # enforce chunking
        climo = climo.chunk({'time': 1})

        # save in chunks to output file 
        logger.info(f'Saving climatology to {self.climo_dataset_path}')
        os.makedirs(os.path.dirname(self.climo_dataset_path), exist_ok=True)
        with ProgressBar():
            climo.to_zarr(self.climo_dataset_path, mode='w')
        logger.info(f'...finished')
    
    def set_output(self, forecast_dates, forecast_integrations, data_module):

        # set fields necessary for gt forecasting 
        self.forecast_dates = forecast_dates 
        self.forecast_integrations = forecast_integrations
        self.mean = th.tensor(data_module.test_dataset.target_scaling["mean"].transpose(0, 2, 1, 3, 4))
        self.std = th.tensor(data_module.test_dataset.target_scaling["std"].transpose(0, 2, 1, 3, 4))
        self.output_vars = data_module.test_dataset.output_variables
        self.delta_t = pd.Timedelta(data_module.time_step)

    def forward(self, input):

        if self.forecast_dates is None:
            raise RuntimeError("climo_model: call set_output(...) before forward.")
        # Same calling convention as HEALPixResNet: list [prognostics, ...]
        ref = input[0] if isinstance(input, (list, tuple)) else input
        # check if we're on a new initialization
        if self.integration_counter == self.forecast_integrations:
            self.initialization_counter += 1
            self.integration_counter = 0

        dt = self.delta_t
        n_ch = len(self.output_vars)
        # 12 HEALPix faces, 64x64 per face at nside 64 (coupled stacks use this layout)
        output_array = th.empty(
            (1, 12, self.output_time_dim, n_ch, 64, 64),
            device=ref.device,
            dtype=ref.dtype,
        )

        init_time = self.forecast_dates[self.initialization_counter]
        for i in range(self.output_time_dim):

            valid_time = (self.integration_counter * self.output_time_dim + (i + 1)) * dt + init_time
            # since our climatology time dimension uses representative year 2000, 
            # we need to convert the valid time to a string that matches the climatology 
            # time dimension so we can properly sample the observed climatology
            selection_time = valid_time.strftime('%Y-%m-%dT%H:%M')
            selection_time = f'2000{selection_time[4:]}'

            da = self.climatology.targets.sel(channel_out=self.output_vars, time=selection_time)
            block = th.as_tensor(da.values.transpose([1, 0, 2, 3]), device=ref.device, dtype=ref.dtype)
            if block.ndim == 4:
                block = block.unsqueeze(0)
            output_array[:, :, i, :, :, :] = block

        self.integration_counter += 1

        mean = self.mean.to(device=ref.device, dtype=ref.dtype)
        std = self.std.to(device=ref.device, dtype=ref.dtype)
        output_array = (output_array - mean) / std

        return output_array
