from abc import ABC, abstractmethod
import gc
import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Tuple, List

import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import xarray as xr
import zarr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData
from modulus.utils.insolation import insolation

logger = logging.getLogger(__name__)


def _is_object_store_path(path: str) -> bool:
    """Check if path is an object store path (contains :// or ::).
    
    Parameters
    ----------
    path : str
        Path to check
        
    Returns
    -------
    bool
        True if path appears to be an object store path
    """
    return "://" in path or "::" in path


def _check_fsspec_availability(path: str) -> None: # pragma: no cover
    """Check if fsspec is available for object store paths.
    
    Parameters
    ----------
    path : str
        Path to check
        
    Raises
    ------
    ImportError
        If path is an object store path but fsspec is not available
    """
    if _is_object_store_path(path):
        try:
            import fsspec
        except ImportError:
            raise ImportError(
                f"fsspec is required to access object store paths like '{path}'. "
                "Please install fsspec with: pip install fsspec"
            )

class BaseTimeSeriesDatasetZarr(Dataset, Datapipe, ABC):
    """Abstract base class for time series datasets using Zarr storage.
    
    This class provides the core functionality for loading and processing time series data
    stored in Zarr format. It handles data loading, scaling, masking, and time management.
    Subclasses must implement the __getitem__ method to define specific data retrieval logic.
    """
    
    def __init__(
        self,
        ds_path: str,
        input_variables: Sequence,
        output_variables: Sequence = None,
        constant_variables: Sequence = None,
        scaling: DictConfig = None,
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
        land_masked_fields: Optional[Sequence] = None,
        sea_masked_fields: Optional[Sequence] = None,
        mask_threshold: float = 0.5,
        add_train_noise: bool = False,
        train_noise_params: DictConfig = None,
        train_noise_seed: int = 42,
        meta: DatapipeMetaData = None,
    ):
        """Initialize base time series dataset.
        
        Parameters
        ----------
        ds_path : str
            Path to the Zarr dataset
        input_variables : Sequence
            Variables to use as model inputs
        output_variables : Sequence, optional
            Variables to predict as outputs. If None, uses input_variables
        constant_variables : Sequence, optional
            Constant fields used as additional inputs
        scaling : DictConfig, optional
            Configuration for data scaling/normalization
        input_time_dim : int, default=1
            Number of time steps in input sequence
        output_time_dim : int, default=1
            Number of time steps to predict
        data_time_step : Union[int, str], default="3h"
            Either integer hours or a str interpretable by pandas
            Time resolution of raw data
        time_step : Union[int, str], default="6h"
            Either integer hours or a str interpretable by pandas
            Time step between predictions
        gap : Union[int, str, None], optional
            Either integer hours or a str interpretable by pandas
            Time gap between input and output sequences
        batch_size : int, default=32
            Number of samples per batch
        drop_last : bool, default=False
            Whether to drop last incomplete batch
        add_insolation : bool, default=False
            Whether to add solar insolation as input
        forecast_init_times : Sequence, optional
            A Sequence of pandas Timestamps
            Specific times to initialize forecasts
        start_date : Union[int, str], optional
            Start date/index from which to load data
        end_date : Union[int, str], optional
            End date/index to which to load data
        land_masked_fields : Sequence, optional
            Fields to mask over land
        sea_masked_fields : Sequence, optional
            Fields to mask over sea
        mask_threshold : float, default=0.5
            Threshold for land-sea mask
        add_train_noise : bool, default=False
            Whether to add train noise
        train_noise_params : DictConfig, optional
            Standard deviation of train noise
        train_noise_seed : int, default=42
            Seed for train noise
        meta : DatapipeMetaData, optional
            Metadata for the datapipe
        """
        Datapipe.__init__(self, meta=meta)
        
        self.ds_path = ds_path
        self.scaling = OmegaConf.to_object(scaling) if scaling else None
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.data_time_step = self._convert_time_step(data_time_step)
        self.time_step = self._convert_time_step(time_step)
        self.gap = self._convert_time_step(gap if gap is not None else time_step)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.add_insolation = add_insolation
        self.forecast_init_times = forecast_init_times
        self.forecast_mode = self.forecast_init_times is not None
        self.input_variables = input_variables
        self.land_masked_fields = land_masked_fields
        self.sea_masked_fields = sea_masked_fields
        self.mask_threshold = mask_threshold
        self.output_variables = input_variables if output_variables is None else output_variables
        self.constant_variables = constant_variables
        self.all_variables = list(set(self.input_variables).union(self.output_variables))
        self.all_scaling = None

        # Check if fsspec is available for object store paths
        _check_fsspec_availability(ds_path)
        
        self.ds = zarr.open(ds_path)

        # Validate channels exist
        channels = set(self.input_variables).union(self.output_variables)
        missing_channels = channels - set(self.ds.channel_in[:])
        if len(missing_channels) > 0:
            raise KeyError(
                f"Input, coupled, or output variables not found in dataset: {missing_channels}"
            )

        self._get_time_da(self.ds_path, start_date, end_date)

        self.all_variable_indices = [
            int(np.where(self.ds.channel_in[:] == ch)[0][0]) for ch in self.all_variables
        ]
        self.constant_variable_indices = [
            int(np.where(self.ds.channel_c[:] == ch)[0][0]) for ch in self.constant_variables
        ]
        self.input_variable_indices = [
            self.all_variables.index(inp_ch) for inp_ch in self.input_variables
        ]
        self.output_variable_indices = [
            self.all_variables.index(out_ch) for out_ch in self.output_variables
        ]

        # Length of the data window needed for one sample
        if self.forecast_mode:
            self._window_length = self.interval * (self.input_time_dim - 1) + 1
        else:
            self._window_length = (
                self.interval * (self.input_time_dim - 1)
                + 1
                + (self.gap // self.data_time_step)
                + self.interval * (self.output_time_dim - 1)
            )
            
        self._batch_window_length = self.batch_size + self._window_length - 1
        self._output_delay = self.interval * (self.input_time_dim - 1) + (
            self.gap // self.data_time_step
        )
        
        # Indices within a batch
        self._input_indices = [
            list(range(n, n + self.interval * self.input_time_dim, self.interval))
            for n in range(self.batch_size)
        ]
        self._output_indices = [
            list(
                range(
                    n + self._output_delay,
                    n + self.interval * self.output_time_dim + self._output_delay,
                    self.interval,
                )
            )
            for n in range(self.batch_size)
        ]

        self.spatial_dims = (
            self.ds.face.shape[0],
            self.ds.height.shape[0],
            self.ds.width.shape[0],
        )

        # Cached values
        self.lat = np.asarray(self.ds.lat)
        self.lon = np.asarray(self.ds.lon)
        self.input_scaling = None
        self.target_scaling = None
        self.constant_scaling = None
        self.constants = None

        self._get_masking()
        if self.scaling:
            self._get_scaling_da()
        if self.constant_variables:
            self.constants = self.get_constants()

        self.add_train_noise = add_train_noise
        self.train_noise_params = train_noise_params
        if self.add_train_noise:
            self.rng = np.random.default_rng(train_noise_seed)

    @staticmethod
    def _convert_time_step(dt: Union[int, str]) -> pd.Timedelta:
        """Convert time step specification to Timedelta.

        Parameters
        ----------
        dt : Union[int, str]
            Either integer hours or string time to convert to Timedelta

        Returns
        -------
        pd.Timedelta
            Converted time delta object
        """
        return pd.Timedelta(hours=dt) if isinstance(dt, int) else pd.Timedelta(dt)

    def _get_time_da(
        self,
        ds_path: str,
        start_date: Optional[Union[int, str]],
        end_date: Optional[Union[int, str]]
    ) -> None:
        """Load and decode time array from dataset.

        Sets up time-related attributes including total samples, start index,
        and forecast initialization indices.

        Parameters
        ----------
        ds_path : str
            Path to Zarr dataset
        start_date : Optional[Union[int, str]]
            Start date/index for data slice
        end_date : Optional[Union[int, str]]
            End date/index for data slice
        """
        # Check if fsspec is available for object store paths
        _check_fsspec_availability(ds_path)
        
        ds = xr.open_zarr(ds_path)

        if "time" not in ds:
            raise KeyError(f"Dataset missing time. Dataset provided {ds_path}")

        self.time_da = ds.time.copy(deep=True)
        self.total_samples = self.time_da.sel(time=slice(start_date, end_date)).shape[0]

        if start_date:
            if isinstance(start_date, int):
                self.start_index = start_date
            else:
                self.start_index = int(np.where(self.time_da == np.datetime64(start_date))[0][0])
        else:
            self.start_index = 0

        # Validate time stepping
        if (self.time_step % self.data_time_step).total_seconds() != 0:
            raise ValueError(
                f"'time_step' must be a multiple of 'data_time_step' "
                f"(got {self.time_step} and {self.data_time_step}"
            )
        if (self.gap % self.data_time_step).total_seconds() != 0:
            raise ValueError(
                f"'gap' must be a multiple of 'data_time_step' "
                f"(got {self.gap} and {self.data_time_step}"
            )
        self.interval = self.time_step // self.data_time_step

        # Verify timestep matches data
        ds_dt = pd.Timedelta((self.time_da[1] - self.time_da[0]).values)
        if not (ds_dt == self.data_time_step):
            warnings.warn(
                f"Dataset dt {ds_dt} doesn't match configuration dt {self.data_time_step}. "
                "This could be a configuration error or a dataset mismatch."
            )

        # Find indices of init times for forecast mode
        if self.forecast_mode:
            if self.batch_size != 1:
                self.batch_size = 1
                warnings.warn(
                    "providing 'forecast_init_times' to TimeSeriesDataset requires `batch_size=1`; "
                    "setting it now"
                )
            self._forecast_init_indices = np.array(
                [
                    int(np.where(self.time == s)[0][0])
                    for s in self.forecast_init_times
                ],
                dtype="int",
            ) - ((self.input_time_dim - 1) * self.interval)
        else:
            self._forecast_init_indices = None

    def _get_masking(self) -> None:
        """Setup masking fields for land and sea regions.
        
        Creates masks and indices for masking specific fields over land/sea regions
        based on land-sea mask in dataset. Sets the following attributes:
        - sea_mask: Binary mask for sea points
        - land_mask: Binary mask for land points
        - land_mask_indices: Indices of fields to mask over land
        - sea_mask_indices: Indices of fields to mask over sea
        """
        if "lsm" in self.ds.channel_c:
            lsm_index = np.where(self.ds.channel_c[:] == "lsm")[0][0]
            land_vals = ma.masked_less_equal(self.ds.constants[lsm_index], self.mask_threshold).mask
            sea_vals = ma.masked_greater(self.ds.constants[lsm_index], self.mask_threshold).mask
            self.sea_mask = (
                sea_vals * np.ones_like(sea_vals) +
                land_vals * np.zeros_like(land_vals)
            )
            self.land_mask = 1 - self.sea_mask
        elif "land_sea_mask" in self.ds.channel_c:
            lsm_index = np.where(self.ds.channel_c[:] == "land_sea_mask")[0][0]
            land_vals = ma.masked_less_equal(self.ds.constants[lsm_index], self.mask_threshold).mask
            sea_vals = ma.masked_greater(self.ds.constants[lsm_index], self.mask_threshold).mask
            self.sea_mask = (
                sea_vals * np.ones_like(sea_vals) +
                land_vals * np.zeros_like(land_vals)
            )
            self.land_mask = 1 - self.sea_mask
        else:
            self.sea_mask = None
            self.land_mask = None

        if self.land_masked_fields and len(self.land_masked_fields) > 0:
            self.land_mask_indices = [
                self.all_variables.index(field)
                for field in self.land_masked_fields if field in self.all_variables
            ]
        else:
            self.land_mask_indices = None

        if self.sea_masked_fields and len(self.sea_masked_fields) > 0:
            self.sea_mask_indices = [
                self.all_variables.index(field)
                for field in self.sea_masked_fields if field in self.all_variables
            ]
        else:
            self.sea_mask_indices = None

    def _get_scaling_da(self) -> None:
        """Setup data scaling parameters.
        
        Processes scaling configuration and sets up scaling parameters for:
        - Input variables
        - Target/output variables
        - All variables combined
        - Constant fields
        
        Raises
        ------
        KeyError
            If scaling parameters are missing for any variables
        """
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc["zeros"] = {"mean": 0.0, "std": 1.0}
        scaling_da = scaling_df.to_xarray().astype("float32")

        try:
            self.input_scaling = scaling_da.sel(index=self.input_variables).rename(
                {"index": "channel_in"}
            )
            self.input_scaling = {
                "mean": np.expand_dims(
                    self.input_scaling["mean"].values.copy(), (0, 2, 3, 4)
                ),
                "std": np.expand_dims(
                    self.input_scaling["std"].values.copy(), (0, 2, 3, 4)
                ),
            }
        except (ValueError, KeyError):
            missing = [
                m for m in self.input_variables if m not in list(self.scaling.keys())
            ]
            raise KeyError(
                f"Input channels {missing} not found in the scaling config dict data.scaling ({list(self.scaling.keys())})"
            )

        try:
            self.target_scaling = scaling_da.sel(index=self.output_variables).rename(
                {"index": "channel_out"}
            )
            self.target_scaling = {
                "mean": np.expand_dims(
                    self.target_scaling["mean"].values.copy(), (0, 2, 3, 4)
                ),
                "std": np.expand_dims(
                    self.target_scaling["std"].values.copy(), (0, 2, 3, 4)
                ),
            }
        except (ValueError, KeyError):
            missing = [
                m for m in self.output_variables if m not in list(self.scaling.keys())
            ]
            raise KeyError(
                f"Target channels {missing} not found in the scaling config dict data.scaling ({list(self.scaling.keys())})"
            )

        self.all_scaling = scaling_da.sel(index=self.all_variables).rename(
            {"index": "channel_in"}
        )
        self.all_scaling = {
            "mean": np.expand_dims(
                self.all_scaling["mean"].values.copy(), (0, 2, 3, 4)
            ),
            "std": np.expand_dims(
                self.all_scaling["std"].values.copy(), (0, 2, 3, 4)
            ),
        }

        if self.constant_variables:
            # Check that all constant variables are present in scaling data
            missing_constants = [
                var for var in self.constant_variables 
                if var not in list(self.scaling.keys())
            ]
            if missing_constants:
                raise KeyError(
                    f"Constant variables {missing_constants} not found in the scaling config dict data.scaling ({list(self.scaling.keys())})"
                )

            try:
                
                self.constant_scaling = scaling_da.sel(
                    index=self.constant_variables
                ).rename({"index": "channel_out"})
                self.constant_scaling = {
                    "mean": np.expand_dims(
                        self.constant_scaling["mean"].values.copy(), (1, 2, 3)
                    ),
                    "std": np.expand_dims(
                        self.constant_scaling["std"].values.copy(), (1, 2, 3)
                    ),
                }
            except (ValueError, KeyError):
                missing = [
                    m for m in self.constant_variables if m not in list(self.scaling.keys())
                ]
                raise KeyError(
                    f"Constant channels {missing} not found in the scaling config dict data.scaling ({list(self.scaling.keys())})"
                )

    def get_constants(self) -> np.ndarray:
        """Get constant fields used in dataset.

        Returns
        -------
        np.ndarray
            Array of constant fields with shape [F, C, H, W]
            where F=faces, C=channels, H=height, W=width
        """
        if self.constants is not None:
            return self.constants
        
        const = np.asarray(self.ds.constants[self.constant_variable_indices])

        if self.constant_scaling:
            const = (const - self.constant_scaling["mean"]) / self.constant_scaling["std"]

        self.constants = np.transpose(const, axes=(1, 0, 2, 3))
        return self.constants

    def _get_time_index(self, item: int) -> Tuple[Tuple[int, int], int]:
        """Get time indices for specified sample.

        Parameters
        ----------
        item : int
            Sample index

        Returns
        -------
        Tuple[Tuple[int, int], int]
            ((start_index, end_index), batch_size)
            Time window indices and actual batch size
        """
        window_start_index = (
            self._forecast_init_indices[item]
            if self.forecast_mode
            else item * self.batch_size + self.start_index
        )
        window_max_index = (
            window_start_index + self._window_length
            if self.forecast_mode
            else (item + 1) * self.batch_size + self._window_length + self.start_index
        )
        if not self.drop_last and window_max_index > self.total_samples:
            batch_size = self.batch_size - (window_max_index - self.total_samples)
        else:
            batch_size = self.batch_size
        return (window_start_index, window_max_index), batch_size

    def _get_forecast_sol_times(self, item: int) -> np.ndarray:
        """Get times for calculating solar insolation.

        Parameters
        ----------
        item : int
            Sample index

        Returns
        -------
        np.ndarray
            Array of timestamps for insolation calculation
        """
        time_index, _ = self._get_time_index(item)
        if self.forecast_mode:
            timedeltas = (
                self._input_indices[0] + self._output_indices[0]
            ) * self.data_time_step
            return self.time_da[self.start_index].values + timedeltas
        return self.time_da[slice(*time_index)].values

    def __len__(self) -> int:
        """Get number of samples available in the dataset based on
        timedeltas, gaps, start and end dates.

        Returns
        -------
        int
            Total number of available samples
        """
        if self.forecast_mode:
            return len(self._forecast_init_indices)
        length = (self.total_samples - self._window_length + 1) / self.batch_size
        if self.drop_last:
            return int(np.floor(length))
        return int(np.ceil(length))

    @abstractmethod
    def __getitem__(self, item: int) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
        """Get requested sample - must be implemented by subclasses.

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
            - Insolation (if enabled)
            - Constants (if provided)
            - Additional data (in subclasses)
            
            Target array has shape [B, F, T, C, H, W]
            where:
            B = batch size
            F = faces
            T = time steps
            C = channels
            H = height
            W = width
        """
        pass 