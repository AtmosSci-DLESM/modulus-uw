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
from typing import Dict, Sequence

import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)


def _average_virtual_temperature_from_geopotential_height(z1, z2, p1, p2, R, g0):
    return g0 / (R * torch.log(p1 / p2)) * (z2 - z1)


def _virtual_temperature_from_geopotential_height(z1, z2, p1, p2, T1, R, g0):
    return (2.0 * g0) / (R * np.log(p1 / p2)) * (z2 - z1) - T1


class HydrostaticBalance(torch.nn.Module):
    def __init__(
        self,
        z_pressure_levels: Dict[int, float],
        anchor_z_channel: int,
        anchor_T_channel: int,
        R: float,
        g0: float,
    ) -> None:
        super().__init__()
        self.R = R
        self.g0 = g0

        assert (
            anchor_z_channel in z_pressure_levels.keys()
        ), f"anchor_z_channel ({anchor_z_channel}) not in z_pressure_levels ({z_pressure_levels.keys()})"

        # Sort channels by pressure levels for ease of use later
        self.z_pressure_levels = dict(
            sorted(z_pressure_levels.items(), key=lambda item: item[1])
        )
        self.anchor_z_channel = anchor_z_channel
        self.anchor_T_channel = anchor_T_channel
        self.anchor_pressure = z_pressure_levels[anchor_z_channel]
        self.anchor_z_index = list(self.z_pressure_levels.keys()).index(
            anchor_z_channel
        )
        self.z_channels = list(self.z_pressure_levels.keys())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor that contains the geopotential heights (Z) at the channels and pressure levels specified in the constructor
            [B, F, C, H, W] is the format
        Returns
        -------
        torch.Tensor
        """
        Tv_size = [
            len(self.z_pressure_levels) if i == 2 else s for i, s in enumerate(x.size())
        ]
        Tv = torch.empty(Tv_size, dtype=x.dtype, layout=x.layout, device=x.device)
        Tv[:, :, self.anchor_z_index, ...] = x[:, :, self.anchor_T_channel, ...]

        if self.anchor_z_index > 0:
            # Go down in index (up in vertical level) from anchor
            for i in range(self.anchor_z_index, 0, -1):
                z_channel = self.z_channels[i]
                z_channel_m1 = self.z_channels[i - 1]
                zi = x[:, :, z_channel, ...]
                zim1 = x[:, :, z_channel_m1, ...]
                Tv[:, :, i - 1, ...] = _virtual_temperature_from_geopotential_height(
                    zi,
                    zim1,
                    self.z_pressure_levels[z_channel],
                    self.z_pressure_levels[z_channel_m1],
                    Tv[:, :, i, ...],
                    self.R,
                    self.g0,
                )

        if self.anchor_z_index < len(self.z_pressure_levels) - 1:
            # Go up in index (down in vertical level) from anchor
            for i in range(self.anchor_z_index, len(self.z_pressure_levels) - 1):
                z_channel = self.z_channels[i]
                z_channel_p1 = self.z_channels[i + 1]
                zi = x[:, :, z_channel, ...]
                zip1 = x[:, :, z_channel_p1, ...]
                Tv[:, :, i + 1, ...] = _virtual_temperature_from_geopotential_height(
                    zi,
                    zip1,
                    self.z_pressure_levels[z_channel],
                    self.z_pressure_levels[z_channel_p1],
                    Tv[:, :, i, ...],
                    self.R,
                    self.g0,
                )

        return Tv


class DifferentialHydrostaticBalanceConstraint(torch.nn.Module):
    def __init__(
        self,
        z_pressure_levels: Dict[int, float],
        Tv_pressure_levels: Dict[int, float],
        anchor_z_channel: int,
        anchor_T_channel: int,
        R: float,
        g0: float,
        extend_to_surface: bool = False,
    ) -> None:
        super().__init__()
        self.R = R
        self.g0 = g0
        self.extend_to_surface = extend_to_surface

        assert (
            anchor_z_channel in z_pressure_levels.keys()
        ), f"anchor_z_channel ({anchor_z_channel}) not in z_pressure_levels ({z_pressure_levels.keys()})"
        assert (
            anchor_T_channel in Tv_pressure_levels.keys()
        ), f"anchor_T_channel ({anchor_T_channel}) not in Tv_pressure_levels ({Tv_pressure_levels.keys()})"

        # Sort channels by pressure levels for ease of use later
        self.z_pressure_levels = dict(
            sorted(
                z_pressure_levels.items(),
                key=lambda item: (
                    not isinstance(item[1], (int, float)), item[1] if isinstance(item[1], (int, float)) else float('inf')
                )
            )
        )
        self.Tv_pressure_levels = dict(
            sorted(
                Tv_pressure_levels.items(),
                key=lambda item: (
                    not isinstance(item[1], (int, float)), item[1] if isinstance(item[1], (int, float)) else float('inf')
                )
            )
        )
        self.anchor_z_channel = anchor_z_channel
        self.anchor_T_channel = anchor_T_channel
        self.anchor_pressure = z_pressure_levels[anchor_z_channel]
        self.anchor_z_index = list(self.z_pressure_levels.keys()).index(
            anchor_z_channel
        )
        self.anchor_T_index = list(self.Tv_pressure_levels.keys()).index(
            anchor_T_channel
        )
        self.z_channels = list(self.z_pressure_levels.keys())
        self.Tv_channels = list(self.Tv_pressure_levels.keys())

        pressure_levels = sorted(
            {k: v for k, v in z_pressure_levels.items() if not isinstance(v, str)}.values()
        )
        self.register_buffer(
            'pressure_levels',
            torch.tensor(pressure_levels).view(1,1,-1,1,1),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor that contains the geopotential heights (Z) at the channels and pressure levels specified in the constructor
            [B, F, C, H, W] is the format
        Returns
        -------
        torch.Tensor
        """
        Tv_avg_size = [
            len(self.z_pressure_levels) - 1 if i == 2 else s
            for i, s in enumerate(x.size())
        ]
        Tv_avg = torch.empty(
            Tv_avg_size, dtype=x.dtype, layout=x.layout, device=x.device
        )

        # Go up in index (down in vertical level) from anchor
        # TODO: remove constraint that first z_channel is 0
        for i in range(len(self.z_pressure_levels) - 1 - self.extend_to_surface):
            z_channel = self.z_channels[i]
            z_channel_p1 = self.z_channels[i + 1]
            zi = x[:, :, z_channel, ...]
            zip1 = x[:, :, z_channel_p1, ...]
            Tv_avg[
                :, :, i, ...
            ] = _average_virtual_temperature_from_geopotential_height(
                zi,
                zip1,
                torch.tensor(self.z_pressure_levels[z_channel]),
                torch.tensor(self.z_pressure_levels[z_channel_p1]),
                self.R,
                self.g0,
            )

        Tv_model_avg_size = [
            len(self.Tv_pressure_levels) - 1 if i == 2 else s
            for i, s in enumerate(x.size())
        ]
        Tv_model_avg = torch.empty(
            Tv_model_avg_size, dtype=x.dtype, layout=x.layout, device=x.device
        )

        # Go up in index (down in vertical level) from anchor
        for i in range(len(self.Tv_pressure_levels) - 1 - self.extend_to_surface):
            Tv_channel = self.Tv_channels[i]
            Tv_channel_p1 = self.Tv_channels[i + 1]
            Tvi = x[:, :, Tv_channel, ...]
            Tvip1 = x[:, :, Tv_channel_p1, ...]
            Tv_model_avg[:, :, i, ...] = 0.5 * (Tvi + Tvip1)

        if self.extend_to_surface:
            # Generate tensor of pressure levels
            p_levs_size = [
                len(self.pressure_levels) if i == 2 else s
                for i, s in enumerate(x.size())
            ]
            p_levs = torch.ones(
                p_levs_size, dtype=x.dtype, layout=x.layout, device=x.device
            )
            p_levs = p_levs * self.pressure_levels

            # Get boolean mask of levels above surface using geopotential heights
            # Assumes first z_channel is the highest altitude level
            above_surface_mask = (
                x[:, :, : len(self.z_pressure_levels)-1, :, :] >
                x[:, :, len(self.z_pressure_levels)-1:len(self.z_pressure_levels), :, :]
            )

            # Get index of lowest level above surface
            lowest_lev_idx = above_surface_mask.int().flip(2).argmax(dim=2)
            lowest_lev_idx = (len(self.z_pressure_levels)-1) - 1 - lowest_lev_idx

            # Compute average virtual temperature between lowest level and surface
            # from geopotential heights
            zi = torch.gather(x, 2, lowest_lev_idx.unsqueeze(2)).squeeze(2)
            zip1 = x[:, :, self.z_channels[-1], ...]
            p1 = torch.gather(p_levs, 2, lowest_lev_idx.unsqueeze(2)).squeeze(2)
            p2 = x[:, :, -1, ...]
            Tv_avg[
                :, :, -1, ...
            ] = _average_virtual_temperature_from_geopotential_height(
                zi,
                zip1,
                p1,
                p2,
                self.R,
                self.g0,
            )

            # Get average model virtual temperature between lowest level and surface
            Tvi = torch.gather(x, 2, lowest_lev_idx.unsqueeze(2)).squeeze(2)
            Tvip1 = x[:, :, self.Tv_channels[-1], ...]
            Tv_model_avg[:, :, -1, ...] = 0.5 * (Tvi + Tvip1)

        return Tv_avg, Tv_model_avg

class WeightedMSEWithHydrostasy(torch.nn.MSELoss):

    """
    Loss object that adds a differential Hydrostatic balance constraint in addition to
    user defined weighting of variables when calculating MSE
    """

    def __init__(
        self,
        hPa_levels: Sequence[int],
        channels: Sequence[str],
        weights: Sequence,
        alpha: Sequence[float],  # K
        scaling: Dict[str, Dict[str, float]],
        src_directory: str,
        dst_directory: str,
        dataset_name: str,
        surface_geopotential_name: str,
        surface_geopotential_mean: float = -597.7115478515625,
        surface_geopotential_std: float = 55658.21484375,
        convert_topography_to_meters: bool = True,
        R: float = 287,  # J K^{-1} kg^{-1}
        g0: float = 9.81,  # m s^{-2}
        topography_masking: bool = True,
    ):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        """
        super().__init__()
        self.loss_weights = torch.tensor(weights)
        self.device = None
        self.g0 = g0
        self.convert_topography_to_meters = convert_topography_to_meters
        self.topography_masking = topography_masking

        if "surface" in hPa_levels:
            self.extend_to_surface = True
            hPa_levels.remove("surface")
            logger.info(
                "Extending hydrostasy constraint to surface level, ensure that surface pressure is constrained to be non-negative using constraints module."
            )
            if "sp" not in channels and "sp-boxcox" not in channels:
                raise ValueError(
                    "Surface pressure channel (sp or sp-boxcox) must be included in model channels when extending hydrostasy constraint to surface."
                )
            self.transformed_sp = True if "sp-boxcox" in channels else False
        else:
            self.extend_to_surface = False

        # Get channel index to pressure level mapping
        self.pressure_levels = sorted(hPa_levels)
        self.z_pressure_levels = {
            channels.index(f"z{int(pl)}"): pl
            for i, pl in enumerate(self.pressure_levels)
        }
        self.T_pressure_levels = {
            channels.index(f"t{int(pl)}"): pl
            for i, pl in enumerate(self.pressure_levels)
        }
        self.q_pressure_levels = {
            channels.index(f"q{int(pl)}"): pl
            for i, pl in enumerate(self.pressure_levels)
            if f"q{int(pl)}" in channels
        }
        # Get offset to map from q channels here to Tv channels
        # Relies on all pressure levels below a threshold to have q channels
        for i, pl in enumerate(self.pressure_levels):
            if f"q{int(pl)}" in channels:
                self.q_index_offset = i
                break

        # Create mapping for new tensor that holds only the constraint variables
        self.z_constraint_pressure_levels = {
            i: pl for i, pl in enumerate(self.pressure_levels)
        }
        self.Tv_constraint_pressure_levels = {
            len(self.z_constraint_pressure_levels) + i: pl
            for i, pl in enumerate(self.pressure_levels)
        }

        # Get scaling weights
        self.z_mean = torch.Tensor(
            [
                scaling[f"z{int(pl)}"]["mean"]
                for i, pl in enumerate(self.pressure_levels)
            ]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.z_std = torch.Tensor(
            [scaling[f"z{int(pl)}"]["std"] for i, pl in enumerate(self.pressure_levels)]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.T_mean = torch.Tensor(
            [
                scaling[f"t{int(pl)}"]["mean"]
                for i, pl in enumerate(self.pressure_levels)
            ]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.T_std = torch.Tensor(
            [scaling[f"t{int(pl)}"]["std"] for i, pl in enumerate(self.pressure_levels)]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.q_mean = torch.Tensor(
            [
                scaling[f"q{int(pl)}"]["mean"]
                for i, pl in enumerate(self.q_pressure_levels.values())
            ]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.q_std = torch.Tensor(
            [
                scaling[f"q{int(pl)}"]["std"]
                for i, pl in enumerate(self.q_pressure_levels.values())
            ]
        ).reshape((1, 1, 1, -1, 1, 1))

        # Set per level alphas
        if len(alpha) != len(hPa_levels) - 1 + self.extend_to_surface:
            raise AssertionError(
                f"Incorrect number of alpha values. Expected len(hPa_levels)-1+self.extend_to_surface [{len(hPa_levels)-1+self.extend_to_surface}], got {len(alpha)}"
            )
        self.alpha = torch.Tensor(alpha).reshape((1, 1, -1, 1, 1))

        # Molecular weight ratio factor of water vapor to air
        self.Mw_ratio = 28.97 / 18.016 - 1.0  # 0.6078

        if self.extend_to_surface:
            self.T_pressure_levels[channels.index("t2m")] = "surface"
            self.q_pressure_levels[channels.index("q2m")] = "surface"
            # Add entries to account for adding surface geopotential height
            self.z_constraint_pressure_levels[len(self.pressure_levels)] = "surface"
            self.Tv_constraint_pressure_levels[
                len(self.z_constraint_pressure_levels) + len(self.pressure_levels)
            ] = "surface"
            self.Tv_constraint_pressure_levels = {
                k+1: v for k, v in self.Tv_constraint_pressure_levels.items()
            }

            self.T2m_mean = torch.tensor(
                scaling["t2m"]["mean"]
            ).reshape((1, 1, 1, -1, 1, 1))
            self.T2m_std = torch.tensor(
                scaling["t2m"]["std"]
            ).reshape((1, 1, 1, -1, 1, 1))
            self.q2m_mean = torch.tensor(
                scaling["q2m"]["mean"]
            ).reshape((1, 1, 1, -1, 1, 1))
            self.q2m_std = torch.tensor(
                scaling["q2m"]["std"]
            ).reshape((1, 1, 1, -1, 1, 1))

            self.T_mean = torch.cat((self.T_mean, self.T2m_mean), dim=3)
            self.T_std = torch.cat((self.T_std, self.T2m_std), dim=3)
            self.q_mean = torch.cat((self.q_mean, self.q2m_mean), dim=3)
            self.q_std = torch.cat((self.q_std, self.q2m_std), dim=3)

            sp_name = 'sp-boxcox' if self.transformed_sp else 'sp'

            self.sp_mean = torch.tensor(
                scaling[sp_name]["mean"]
            ).reshape((1, 1, 1, -1, 1, 1))
            self.sp_std = torch.tensor(
                scaling[sp_name]["std"]
            ).reshape((1, 1, 1, -1, 1, 1))
            self.sp_mapping = channels.index(sp_name)

        # Create the constraint
        # TODO: remove anchor levels since it's not needed for the
        # differential constraint
        self.constraint = DifferentialHydrostaticBalanceConstraint(
            self.z_constraint_pressure_levels,
            self.Tv_constraint_pressure_levels,
            0,
            len(self.z_pressure_levels)+self.extend_to_surface,
            R,
            self.g0,
            self.extend_to_surface,
        )

        self.num_z_levels = len(self.z_constraint_pressure_levels)
        self.num_Tv_levels = len(self.Tv_constraint_pressure_levels)
        self.z_level_mapping = torch.tensor(list(self.z_pressure_levels.keys()))
        self.T_level_mapping = torch.tensor(list(self.T_pressure_levels.keys()))
        self.q_level_mapping = torch.tensor(list(self.q_pressure_levels.keys()))

        # Get topography information
        ds = xr.open_zarr(f"{src_directory}/{dataset_name}.zarr")
        self.topography = (
            surface_geopotential_std * ds.constants.sel(channel_c=surface_geopotential_name).values
            + surface_geopotential_mean
        )
        if self.convert_topography_to_meters:
            self.topography /= self.g0

        self.topography = torch.tensor(
            self.topography[np.newaxis, :, np.newaxis, :, :], dtype=torch.float
        )
        logger.info(
            f"Min/Max topography (m): {self.topography.min()}/{self.topography.max()}"
        )

    def setup(self, trainer):
        """
        pushes weights to cuda device
        """

        if len(trainer.output_variables) + len(self.z_pressure_levels) + self.extend_to_surface - 1 != len(
            self.loss_weights
        ):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)

        # Move means and stds
        self.z_mean = self.z_mean.to(device=trainer.device)
        self.z_std = self.z_std.to(device=trainer.device)
        self.T_mean = self.T_mean.to(device=trainer.device)
        self.T_std = self.T_std.to(device=trainer.device)
        self.q_mean = self.q_mean.to(device=trainer.device)
        self.q_std = self.q_std.to(device=trainer.device)
        if self.extend_to_surface:
            self.sp_mean = self.sp_mean.to(device=trainer.device)
            self.sp_std = self.sp_std.to(device=trainer.device)

        # Move alphas
        self.alpha = self.alpha.to(device=trainer.device)

        # Move indexing arrays for CUDA graphs
        self.z_level_mapping = self.z_level_mapping.to(device=trainer.device)
        self.T_level_mapping = self.T_level_mapping.to(device=trainer.device)
        self.q_level_mapping = self.q_level_mapping.to(device=trainer.device)

        # Move topography
        self.topography = self.topography.to(device=trainer.device)

    def reverse_transform_sp(
        self,
        x,
        lam=19.632015209145543,
        max_val=1068.063515625,
    ):
        x = torch.exp(torch.log(x * lam + 1 + 1e-8)/lam) # reverse Box-Cox
        x = x * max_val # Rescaling back to hPa
        return x

    def scale(self, x):
        """
        Scale inputs to physical values and compute virtual temperature
        Tensors are expected to be in the shape [N, F, B, C, H, W]
        """
        N, F, B, C, H, W = x.shape
        C_scaled = self.num_z_levels + self.num_Tv_levels
        if self.extend_to_surface:
            C_scaled += 1  # Add surface pressure
        x_scaled = torch.zeros(
            (N, F, B, C_scaled, H, W),
            device=x.device,
            dtype=torch.float,
        )
        # Get scaled geopotential heights
        x_scaled[:, :, :, : self.num_z_levels-self.extend_to_surface, :, :] = (
            x[:, :, :, self.z_level_mapping, :, :] * self.z_std + self.z_mean
        ) / self.g0  # divide by g0 for heights
        if self.extend_to_surface:
            # Add surface geopotential
            surface_Z = self.topography.unsqueeze(2)
            surface_Z = torch.tile(surface_Z, (N, 1, B, 1, 1, 1))
            x_scaled[:, :, :, self.num_z_levels:self.num_z_levels+1, :, :] = surface_Z

            # Add surface pressure
            sp = x[:, :, :, self.sp_mapping, :, :] * self.sp_std + self.sp_mean
            if self.transformed_sp:
                sp = self.reverse_transform_sp(sp)
            x_scaled[:, :, :, -1, :, :] = sp

        # Get scaled temperatures
        x_scaled[:, :, :, self.num_z_levels : self.num_z_levels + self.num_Tv_levels, :, :] = (
            x[:, :, :, self.T_level_mapping, :, :] * self.T_std + self.T_mean
        )
        # Add q correction to get virtual temperature for levels with non-zero q
        x_scaled[
            :,
            :,
            :,
            (self.num_z_levels + self.q_index_offset) : self.num_z_levels + self.num_Tv_levels,
            :,
            :,
        ] *= 1.0 + self.Mw_ratio * (
            x[:, :, :, self.q_level_mapping, :, :] * self.q_std + self.q_mean
        )

        # transpose B dim to before F
        x_scaled = x_scaled.transpose(1, 2)

        # Combine N and B dimensions and return
        return x_scaled.reshape((-1, F, C_scaled, H, W))

    def error_histogram(self, prediction, bins, accumulator=None):
        N, F, B, C, H, W = tuple(prediction.shape)

        if not (prediction.ndim == 6):
            raise AssertionError("Expected predictions to have 6 dimensions")

        # Scale to physical units and compute virtual temperature
        x = self.scale(prediction)
        Tv_avg, Tv_model_avg = self.constraint(x)
        Tv_error = Tv_avg - Tv_model_avg
        # Mask out error in regions below the surface
        if self.topography_masking:
            Tv_error[x[:, :, 1 : self.num_z_levels, :, :] < self.topography] = 0.0

        vlevels = Tv_error.shape[2]
        if accumulator is None:
            if isinstance(bins, int):
                accumulator = torch.zeros(
                    (vlevels, bins), dtype=torch.float32, device=prediction.device
                )
            else:
                accumulator = torch.zeros(
                    (vlevels, bins.shape[1] - 1),
                    dtype=torch.float32,
                    device=prediction.device,
                )
        if isinstance(bins, int):
            bin_edges = torch.zeros(
                (vlevels, bins + 1),
                dtype=prediction.dtype,
                device=prediction.device,
            )
        else:
            bin_edges = bins

        for l in range(vlevels):
            hist, be = torch.histogram(
                torch.absolute(Tv_error[:, :, l, :, :]),
                bins=bins if isinstance(bins, int) else bin_edges[l, :],
            )
            accumulator[l, :] += hist
            bin_edges[l, :] = be

        return accumulator, bin_edges

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedMSE pass
        Tensors are expected to be in the shape [N, F, B, C, H, W]
        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """

        # Need to scale back to physical units here so disable autocast
        # and explicitly cast to float32
        with torch.cuda.amp.autocast(enabled=False):
            prediction = prediction.float()
            target = target.float()

            N, F, B, C, H, W = tuple(prediction.shape)

            if not (prediction.ndim == 6 and target.ndim == 6):
                raise AssertionError("Expected predictions to have 6 dimensions")

            # Scale to physical units and compute virtual temperature
            x = self.scale(prediction)
            Tv_avg, Tv_model_avg = self.constraint(x)
            Tv_error = ((Tv_avg - Tv_model_avg) / self.alpha) ** 2

            # Mask out error in regions below the surface
            if self.topography_masking:
                Tv_error[x[:, :, 1 : self.num_z_levels, :, :] < self.topography] = 0.0

            # Compute the error tolerant loss
            Tv_loss = (Tv_error / (1 + torch.exp(1 - Tv_error))).mean(dim=(0, 1, 3, 4))

            data_loss = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5))
            d = torch.concatenate((data_loss, Tv_loss)) * self.loss_weights

            if average_channels:
                return torch.mean(d)
            else:
                return d