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
import json
import logging
import time

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

_DEBUG_ENSEMBLE_LOG_PATH = "/pscratch/sd/z/zespinos/.cursor/debug-6e7764.log"
_DEBUG_SESSION_ID = "6e7764"


def _debug_ensemble_rank0() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _debug_ensemble_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        entry = {
            "sessionId": _DEBUG_SESSION_ID,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "runId": "pre-fix",
        }
        with open(_DEBUG_ENSEMBLE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


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
    Ocean MSE loss with:
    1) Land-sea-mask weighting (optional binary or continuous mask).
    2) Per-channel weights (e.g. sic more than sst).
    3) Optional variogram and spectral terms for single-member (deterministic) predictions.
    """

    def __init__(
        self,
        lsm_file: str,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "lsm"},
        weights: Sequence = [],
        lsm_binary_mask: bool = False,
        lsm_binary_threshold: float = 0.5,
        variogram: float = 0.0,
        variogram_p: float = 0.5,
        variogram_lags: Sequence[int] = (1, 2, 3),
        spectral_weight: float = 0.0,
        spectral_window: int = 32,
        spectral_stride: int = 16,
        spectral_min_ring: int = 32,
        spectral_variables: Optional[Sequence[str]] = None,
    ):
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
        # Binary LSM: same semantics as WeightedCRPSLoss
        self.lsm_binary_mask = lsm_binary_mask
        self.lsm_binary_threshold = lsm_binary_threshold
        # Variogram (single-member: no ensemble mean)
        self.variogram = variogram
        self.variogram_p = variogram_p
        self.variogram_lags = list(variogram_lags)
        # Spectral (ocean power-spectrum)
        self.spectral_weight = float(spectral_weight)
        self.spectral_window = int(spectral_window)
        self.spectral_stride = int(spectral_stride)
        self.spectral_min_ring = int(spectral_min_ring)
        self.spectral_variables = list(spectral_variables) if spectral_variables is not None else None
        self._spectral_channel_idx = None
        self._spectral_lookup = None
        self._spectral_hann = None
        self._spectral_nside = None
        self._spectral_first_forward_done = False
        # Cached scalar components for TensorBoard logging
        self.last_mse = None
        self.last_variogram = None
        self.last_spectral = None

    def setup(self, trainer):
        """
        Load LSM (binary or continuous), move to device, init spectral swaths if needed.
        """
        ### 1. OCEAN LSM ###
        self.lsm_ds = xr.open_dataset(self.lsm_file, **self.open_dict).constants.sel(
            self.selection_dict
        )
        lsm_values = np.expand_dims(self.lsm_ds.values, (0, 2, 3))
        if self.lsm_binary_mask:
            lsm_binary = (lsm_values < self.lsm_binary_threshold).astype(np.float32)
            self.lsm_tensor = th.tensor(lsm_binary)
        else:
            self.lsm_tensor = 1 - th.tensor(lsm_values.astype(np.float32))
        self.lsm_tensor = self.lsm_tensor.to(device=trainer.device)

        ### 2. WEIGHTS ###
        if not len(trainer.output_variables) == len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")
        self.loss_weights = self.loss_weights.to(device=trainer.device)

        ### 3. SPECTRAL SWATHS (if spectral term enabled) ###
        if self.spectral_weight > 0:
            self._init_spectral_swaths(trainer)

    def _init_spectral_swaths(self, trainer):
        """Precompute ocean swath lookup and Hann window for spectral loss (6D single-member)."""
        log = logging.getLogger(__name__)
        out_vars = list(trainer.output_variables)
        n_c = len(out_vars)
        if self.spectral_variables is None:
            self._spectral_channel_idx = list(range(n_c))
        else:
            name_to_idx = {v: i for i, v in enumerate(out_vars)}
            self._spectral_channel_idx = []
            for name in self.spectral_variables:
                if name not in name_to_idx:
                    raise ValueError(
                        f"spectral_variables entry '{name}' not in trainer.output_variables {out_vars}"
                    )
                self._spectral_channel_idx.append(name_to_idx[name])
        mask_spatial = self.lsm_tensor[0, :, 0, 0]
        f, h, w = mask_spatial.shape
        if f != 12 or h != w:
            raise ValueError(
                f"Expected HEALPix grid 12 x nside x nside, got F={f} H={h} W={w}"
            )
        nside = int(h)
        self._spectral_nside = nside
        npix = 12 * nside * nside
        device = trainer.device
        mask_np = mask_spatial.cpu().numpy()
        window_len = self.spectral_window
        stride = self.spectral_stride
        min_row_len = self.spectral_min_ring
        all_windows = []
        n_rows_with_swaths = 0
        spectral_ocean_threshold = 0.5
        for face in range(12):
            for row in range(nside):
                row_mask = mask_np[face, row, :]
                if row_mask.size < window_len:
                    continue
                if self.lsm_binary_mask:
                    ocean = row_mask > spectral_ocean_threshold
                else:
                    ocean = row_mask >= spectral_ocean_threshold
                run_starts = np.where(np.diff(np.concatenate([[False], ocean, [False]]).astype(np.int8)) == 1)[0]
                run_ends = np.where(np.diff(np.concatenate([[False], ocean, [False]]).astype(np.int8)) == -1)[0]
                row_contributed = False
                for rs, re in zip(run_starts, run_ends):
                    run_len = re - rs
                    if run_len < window_len:
                        continue
                    for j in range(0, run_len - window_len + 1, stride):
                        win_slice = row_mask[rs + j : rs + j + window_len]
                        if self.lsm_binary_mask:
                            if not np.all(win_slice > spectral_ocean_threshold):
                                continue
                        else:
                            if not np.all(win_slice >= spectral_ocean_threshold):
                                continue
                        base = face * (nside * nside) + row * nside + rs + j
                        global_indices = base + np.arange(window_len, dtype=np.int64)
                        all_windows.append(global_indices)
                        row_contributed = True
                if row_contributed:
                    n_rows_with_swaths += 1
        n_swaths = len(all_windows)
        if n_swaths == 0:
            log.warning(
                "spectral_loss: no valid ocean swaths found (n_swaths=0). "
                "Spectral term disabled. Check land-sea mask and spectral_window."
            )
            self._spectral_lookup = None
            self._spectral_hann = None
            return
        idx_windows = np.stack(all_windows, axis=0).astype(np.int64)
        self._spectral_lookup = th.from_numpy(idx_windows).to(device)
        self._spectral_hann = th.hann_window(
            window_len, periodic=True, device=device, dtype=th.float32
        )
        log.info(
            "spectral_loss: n_swaths=%d, n_rows_with_swaths=%d, lsm_binary_mask=%s (PAD_XY, no regrid)",
            n_swaths,
            n_rows_with_swaths,
            self.lsm_binary_mask,
        )

    def _to_flat_pad_xy(self, x: th.Tensor, face_dim: int) -> th.Tensor:
        """Flatten HEALPix PAD_XY (face) layout to 1D [..., F*H*W].
        Moves face_dim to the end; accepts trailing (12, nside, nside) or (nside, nside, 12)
        and normalizes to (12, nside, nside) so flatten order is face-major (PAD_XY).
        """
        x = th.movedim(x, face_dim, -1)
        trailing = x.shape[-3:]
        nside = self._spectral_nside
        # #region agent log
        try:
            import json
            with open("/pscratch/sd/z/zespinos/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({"hypothesisId": "B", "location": "healpix_loss.py:_to_flat_pad_xy", "message": "after movedim", "data": {"face_dim": face_dim, "trailing": list(trailing), "nside": nside, "x_shape_before": list(x.shape)}, "timestamp": __import__("time").time()}) + "\n")
        except Exception:
            pass
        # #endregion
        if trailing == (12, nside, nside):
            pass
        elif trailing == (nside, nside, 12):
            x = x.permute(*range(x.ndim - 3), -1, -3, -2)
        else:
            raise ValueError(
                f"Expected trailing (12, nside, nside) or (nside, nside, 12), got {trailing}"
            )
        return x.contiguous().reshape(*x.shape[:-3], -1)

    def _calculate_variogram_loss_single(self, prediction, target, average_channels=True):
        """
        Single-member variogram loss: compare prediction vs target p-th order absolute
        differences at lags. prediction/target 6D [N, B, F, C, H, W]; no ensemble mean.
        """
        p = self.variogram_p
        has_real_mask = self.lsm_tensor.shape[-1] > 1

        total_vs_sum = 0
        total_mask_sum = 0

        # 6D [N, 12, T, C, H, W]: valid_h.sum() already includes the 12 faces; do not multiply by face dim again
        n_c = target.shape[3]
        n_batch_no_face = target.shape[0] * target.shape[2]  # N * T (denominator count excluding faces and C)
        n_batch_no_face_times_c = n_batch_no_face * n_c    # N * T * C for average_channels

        for lag in self.variogram_lags:
            # Horizontal lags (shift along W)
            pred_h1, pred_h2 = prediction[..., :, :-lag], prediction[..., :, lag:]
            tar_h1, tar_h2 = target[..., :, :-lag], target[..., :, lag:]

            if has_real_mask:
                valid_h = self.lsm_tensor[..., :, :-lag] * self.lsm_tensor[..., :, lag:]
            else:
                valid_h = self.lsm_tensor

            exp_pred_h = th.abs(pred_h1 - pred_h2).clamp(min=1e-6).pow(p)
            tar_diff_h = th.abs(tar_h1 - tar_h2).clamp(min=1e-6).pow(p)
            vs_h = ((exp_pred_h - tar_diff_h) ** 2) * valid_h

            # Vertical lags (shift along H)
            pred_v1, pred_v2 = prediction[..., :-lag, :], prediction[..., lag:, :]
            tar_v1, tar_v2 = target[..., :-lag, :], target[..., lag:, :]

            if has_real_mask:
                valid_v = self.lsm_tensor[..., :-lag, :] * self.lsm_tensor[..., lag:, :]
            else:
                valid_v = self.lsm_tensor

            exp_pred_v = th.abs(pred_v1 - pred_v2).clamp(min=1e-6).pow(p)
            tar_diff_v = th.abs(tar_v1 - tar_v2).clamp(min=1e-6).pow(p)
            vs_v = ((exp_pred_v - tar_diff_v) ** 2) * valid_v

            if average_channels:
                total_vs_sum += vs_h.sum() + vs_v.sum()
                if has_real_mask:
                    total_mask_sum += (valid_h.sum() + valid_v.sum()) * n_batch_no_face_times_c
                else:
                    total_mask_sum += vs_h.numel() + vs_v.numel()
            else:
                total_vs_sum += vs_h.sum(dim=(0, 1, 2, 4, 5)) + vs_v.sum(dim=(0, 1, 2, 4, 5))
                if has_real_mask:
                    total_mask_sum += (valid_h.sum() + valid_v.sum()) * n_batch_no_face
                else:
                    total_mask_sum += (vs_h.numel() + vs_v.numel()) // n_c

        return total_vs_sum / (total_mask_sum + 1e-8)

    def _compute_power_spectrum_loss_single(self, prediction, target, average_channels=True):
        """
        Log-spectral MSE between prediction and target over ocean swaths.
        prediction/target 6D: [N, B, F, T, C, H, W] with F=12 (faces), T=time, or
        [N, B, F, C, H, W] with F=12. Single-member, no ensemble.

        The ocean swath lookup and flat 1D layout are in HEALPix PAD_XY order (12 faces).
        Model output is always [..., 12, T, C, H, W] (face at dim 1) regardless of
        enable_healpixpad; we detect 12 faces at dim 1 or 2 and flatten accordingly.
        """
        if self._spectral_lookup is None:
            return th.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        idx = self._spectral_channel_idx
        # 6D [N, 12, T, C, H, W]: face at index 1, channels at 3
        # 7D [N, B, 12, T, C, H, W]: face at index 2, channels at 4
        if prediction.ndim == 6:
            if prediction.shape[1] != 12:
                return th.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
            if (prediction.shape[4] != self._spectral_nside or prediction.shape[5] != self._spectral_nside):
                return th.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
            pred_spec = prediction[:, :, :, idx, :, :]
            tar_spec = target[:, :, :, idx, :, :]
            face_dim_flat = 1
        elif prediction.ndim == 7:
            if prediction.shape[2] != 12:
                return th.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
            if (prediction.shape[5] != self._spectral_nside or prediction.shape[6] != self._spectral_nside):
                return th.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
            pred_spec = prediction[:, :, :, :, idx, :, :]
            tar_spec = target[:, :, :, :, idx, :, :]
            face_dim_flat = 2
        else:
            return th.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        # pred_spec: 6D (n, 12, t, c_spec, h, w) or 7D (n, b, 12, t, c_spec, h, w)
        # After _to_flat_pad_xy(face_dim) the face dim is folded into npix
        if pred_spec.ndim == 6:
            n, b, t, c_spec, h, w = pred_spec.shape
            n_btc = n * t * c_spec
        else:
            n, b, f, t, c_spec, h, w = pred_spec.shape
            n_btc = n * b * t * c_spec
        npix = 12 * self._spectral_nside ** 2
        with th.cuda.amp.autocast(enabled=False):
            pred_f = pred_spec.float()
            tar_f = tar_spec.float()
            pred_flat = self._to_flat_pad_xy(pred_f, face_dim_flat)
            tar_flat = self._to_flat_pad_xy(tar_f, face_dim_flat)
            pred_ring = pred_flat.reshape(n_btc, npix)
            tar_ring = tar_flat.reshape(n_btc, npix)
            lookup = self._spectral_lookup
            n_swaths, window_len = lookup.shape
            pred_swaths = pred_ring[:, lookup]
            tar_swaths = tar_ring[:, lookup]
            pred_swaths = pred_swaths.reshape(n_btc, n_swaths, window_len)
            tar_swaths = tar_swaths.reshape(n_btc, n_swaths, window_len)
            hann = self._spectral_hann
            pred_tapered = pred_swaths * hann[None, None, :]
            tar_tapered = tar_swaths * hann[None, None, :]
            pred_fft = th.fft.rfft(pred_tapered, dim=-1)
            tar_fft = th.fft.rfft(tar_tapered, dim=-1)
            pred_power = (pred_fft.real ** 2 + pred_fft.imag ** 2)[..., 1:]
            tar_power = (tar_fft.real ** 2 + tar_fft.imag ** 2)[..., 1:]
            n_freq = pred_power.shape[-1]
            pred_power_mean_swath = pred_power.mean(dim=1)
            tar_power_mean_swath = tar_power.mean(dim=1)
            if pred_spec.ndim == 6:
                pred_power_mean_swath = pred_power_mean_swath.reshape(n, t, c_spec, n_freq)
                tar_power_mean_swath = tar_power_mean_swath.reshape(n, t, c_spec, n_freq)
            else:
                pred_power_mean_swath = pred_power_mean_swath.reshape(n, b, t, c_spec, n_freq)
                tar_power_mean_swath = tar_power_mean_swath.reshape(n, b, t, c_spec, n_freq)
            eps = 1e-8
            log_pred = th.log(pred_power_mean_swath + eps)
            log_tar = th.log(tar_power_mean_swath + eps)
            mse_per_sample = ((log_pred - log_tar) ** 2).mean(dim=-1)
            spec_weights = self.loss_weights[idx]
            if average_channels:
                if pred_spec.ndim == 6:
                    weighted = mse_per_sample * spec_weights.view(1, 1, -1)
                    c_total = prediction.shape[3]
                    spec_loss = weighted.sum() / (n * t * c_total)
                else:
                    weighted = mse_per_sample * spec_weights.view(1, 1, 1, -1)
                    c_total = prediction.shape[4]
                    spec_loss = weighted.sum() / (n * b * t * c_total)
            else:
                if pred_spec.ndim == 6:
                    weighted = mse_per_sample * spec_weights.view(1, 1, -1)
                    spec_loss = weighted.mean(dim=(0, 1))
                else:
                    weighted = mse_per_sample * spec_weights.view(1, 1, 1, -1)
                    spec_loss = weighted.mean(dim=(0, 1, 2))
        return spec_loss

    def forward(self, prediction, target, average_channels=True):
        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum(
                dim=(0, 1, 2, 4, 5)
            )
            self.lsm_sum_calculated = True

        ocean_err = ((target - prediction) ** 2) * self.lsm_tensor
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        ocean_mean_err = ocean_mean_err * self.loss_weights

        if average_channels:
            loss = th.sum(ocean_mean_err) / self.lsm_sum
        else:
            loss = ocean_mean_err / self.lsm_var_sum

        # Cache the base MSE term (before adding variogram / spectral components)
        try:
            base_mse = loss.mean() if isinstance(loss, th.Tensor) and loss.ndim > 0 else loss
            if isinstance(base_mse, th.Tensor):
                self.last_mse = base_mse.detach()
            else:
                self.last_mse = None
        except Exception:
            # Never let logging-side bookkeeping affect the training loss
            self.last_mse = None

        self.last_variogram = None
        self.last_spectral = None

        if self.variogram > 0.0:
            vs_loss = self._calculate_variogram_loss_single(prediction, target, average_channels)
            loss = loss + self.variogram * vs_loss
            self.last_variogram = (self.variogram * vs_loss).mean().detach() if th.is_tensor(vs_loss) else None

        if self.spectral_weight > 0.0 and self._spectral_lookup is not None:
            spec_loss = self._compute_power_spectrum_loss_single(prediction, target, average_channels)
            if spec_loss is not None and th.is_tensor(spec_loss):
                if average_channels:
                    loss = loss + self.spectral_weight * spec_loss
                else:
                    spectral_contrib = th.zeros_like(loss)
                    spectral_contrib[self._spectral_channel_idx] = self.spectral_weight * spec_loss
                    loss = loss + spectral_contrib
                self.last_spectral = (self.spectral_weight * spec_loss).mean().detach() if spec_loss.ndim > 0 else (self.spectral_weight * spec_loss).detach()

        return loss

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
        scales: Sequence[int] = [4, 8, 16, 32],
        temporal_dt: float = 0.0,
        variogram: float = 0.0,
        variogram_p: float = 0.5,
        variogram_lags: Sequence[int] = [1, 2, 3],
        spectral_weight: float = 0.0,
        spectral_window: int = 32,
        spectral_stride: int = 16,
        spectral_min_ring: int = 32,
        spectral_variables: Optional[Sequence[str]] = None,
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
        scales: Sequence[int], optional
            scales for the multiscale CRPS loss. Default is [4, 8, 16, 32].
        variogram: float, optional
            weight for the variogram score loss. Default is 0, no variogram loss is applied.
            Penalizes mismatched spatial structure by comparing p-th order absolute differences
            at short pixel lags (interior-only, computed per HEALPix face).
        variogram_p: float, optional
            order of the absolute differences in the variogram score. Default is 0.5.
        variogram_lags: Sequence[int], optional
            lags for the variogram score. Default is [1, 2, 3].
        spectral_weight: float, optional
            weight for the ocean power-spectrum penalty. When 0.0, the spectral term is disabled.
        spectral_window: int, optional
            window length (HEALPix pixels) for 1D FFT along each zonal ocean swath. Default 32.
        spectral_stride: int, optional
            stride for sliding window along each latitude ring. Default 16.
        spectral_min_ring: int, optional
            minimum number of pixels in a ring to be considered. Default 32.
        spectral_variables: Sequence[str], optional
            names of output variables to which the spectral term applies. If None, all channels.
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        if n_members < 2:
            raise ValueError("n_members must be at least 2 for CRPS loss to be defined")
        else:    
            self.n_members = n_members
        self.device = None
        # Mean penalty term
        self.mean_penalty = mean_penalty
        # Temporal dt term
        self.temporal_dt = temporal_dt
        # Multiscale terms
        self.multiscale = multiscale
        self.masked_pool = masked_pool
        self.scales = scales
        # Variogram terms
        self.variogram = variogram
        self.variogram_p = variogram_p
        self.variogram_lags = variogram_lags
        # Spectral (ocean power-spectrum) term; defaults keep it disabled
        self.spectral_weight = float(spectral_weight)
        self.spectral_window = int(spectral_window)
        self.spectral_stride = int(spectral_stride)
        self.spectral_min_ring = int(spectral_min_ring)
        self.spectral_variables = list(spectral_variables) if spectral_variables is not None else None
        self._spectral_channel_idx = None
        self._spectral_lookup = None
        self._spectral_hann = None
        self._spectral_nside = None
        self._spectral_first_forward_done = False
        # LSM terms
        self.lsm_binary_mask = lsm_binary_mask

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

        # Cached per-component loss scalars for TensorBoard logging
        self.last_crps_base = None
        self.last_mean_penalty = None
        self.last_multiscale = None
        self.last_temporal_dt = None
        self.last_variogram = None
        self.last_spectral = None
        # Cached ensemble-diagnostics (spread/skill/SSR) for TensorBoard
        self.last_spread = None
        self.last_skill = None
        self.last_spread_skill_score = None
        self._debug_ens_fwd_count = 0
        self._dbg_training_epoch = -1
        self._dbg_model_ref = None

    def set_training_epoch(self, epoch: int) -> None:
        """Trainer hook so diagnostics can correlate with the scheduler epoch index."""
        self._dbg_training_epoch = int(epoch)

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

        if self.spectral_weight > 0:
            self._init_spectral_swaths(trainer)

        self._dbg_model_ref = getattr(trainer, "model", None)

    def _compute_spread_skill_ssr(
        self,
        prediction: th.Tensor,
        target: th.Tensor,
    ):
        """
        Compute domain-averaged spread, skill, and spread-skill ratio (SSR)
        from the unweighted ensemble prediction and target.

        prediction: [Cond, B, F, T, C, H, W]
        target:     [B, F, T, C, H, W]
        """
        if prediction.ndim != 7 or target.ndim != 6:
            raise ValueError(
                f"_compute_spread_skill_ssr expects prediction [Cond, B, F, T, C, H, W] and target [B, F, T, C, H, W], "
                f"got {prediction.shape} and {target.shape}"
            )

        n = self.n_members
        if n < 2:
            raise ValueError("Spread/skill diagnostics require at least 2 ensemble members")

        # Ensemble mean at each grid point
        mu = prediction.mean(dim=0)  # [B, F, T, C, H, W]

        # Local ensemble variance with Bessel's correction (N-1 in denominator)
        diff = prediction - mu.unsqueeze(0)  # [Cond, B, F, T, C, H, W]
        sigma2 = (diff * diff).sum(dim=0) / float(max(n - 1, 1))  # [B, F, T, C, H, W]

        # Local squared error of ensemble mean
        se = (mu - target) ** 2  # [B, F, T, C, H, W]

        # Apply land–sea mask as weight; mask is [1, F, 1, 1, H, W] or all ones
        # Broadcasting in the products below yields [B, F, T, C, H, W].
        weighted_sigma2 = sigma2 * self.lsm_tensor  # [B, F, T, C, H, W]
        weighted_se = se * self.lsm_tensor          # [B, F, T, C, H, W]

        eps = 1e-8
        # Effective count of masked pixels (same for every channel since mask has no C dim)
        m_eff = self.lsm_tensor.sum()

        # Reduce over batch, faces, time, H, W – keep channels separate
        # -> per-channel variance and MSE
        spread_var_c = weighted_sigma2.sum(dim=(0, 1, 2, 4, 5)) / (m_eff + eps)  # [C]
        skill_mse_c = weighted_se.sum(dim=(0, 1, 2, 4, 5)) / (m_eff + eps)       # [C]

        spread_c = th.sqrt(spread_var_c + eps)  # [C]
        skill_c = th.sqrt(skill_mse_c + eps)    # [C]
        ssr_c = spread_c / (skill_c + eps)      # [C]

        # Return per-channel diagnostics; caller can aggregate as needed
        return spread_c.detach(), skill_c.detach(), ssr_c.detach()

    def _init_spectral_swaths(self, trainer):
        """Precompute ocean swath lookup table and Hann window for spectral loss (run once in setup)."""
        log = logging.getLogger(__name__)
        # 1. Resolve channel indices
        out_vars = list(trainer.output_variables)
        n_c = len(out_vars)
        if self.spectral_variables is None:
            self._spectral_channel_idx = list(range(n_c))
        else:
            name_to_idx = {v: i for i, v in enumerate(out_vars)}
            self._spectral_channel_idx = []
            for name in self.spectral_variables:
                if name not in name_to_idx:
                    raise ValueError(
                        f"spectral_variables entry '{name}' not in trainer.output_variables {out_vars}"
                    )
                self._spectral_channel_idx.append(name_to_idx[name])
        # 2. Infer geometry from lsm_tensor [1, F, 1, 1, H, W]
        mask_spatial = self.lsm_tensor[0, :, 0, 0]
        f, h, w = mask_spatial.shape
        if f != 12 or h != w:
            raise ValueError(
                f"Expected HEALPix grid 12 x nside x nside, got F={f} H={h} W={w}"
            )
        nside = int(h)
        self._spectral_nside = nside
        npix = 12 * nside * nside
        device = trainer.device
        # 3. Build ocean swaths in native PAD_XY (face) layout: no regridding. Each row in each face is a contiguous 1D segment.
        mask_np = mask_spatial.cpu().numpy()
        window_len = self.spectral_window
        stride = self.spectral_stride
        min_row_len = self.spectral_min_ring  # min length to consider a row
        all_windows = []
        n_rows_with_swaths = 0
        spectral_ocean_threshold = 0.5
        for face in range(12):
            for row in range(nside):
                row_mask = mask_np[face, row, :]
                if row_mask.size < window_len:
                    continue
                if self.lsm_binary_mask:
                    ocean = row_mask > spectral_ocean_threshold
                else:
                    ocean = row_mask >= spectral_ocean_threshold
                run_starts = np.where(np.diff(np.concatenate([[False], ocean, [False]]).astype(np.int8)) == 1)[0]
                run_ends = np.where(np.diff(np.concatenate([[False], ocean, [False]]).astype(np.int8)) == -1)[0]
                row_contributed = False
                for rs, re in zip(run_starts, run_ends):
                    run_len = re - rs
                    if run_len < window_len:
                        continue
                    for j in range(0, run_len - window_len + 1, stride):
                        win_slice = row_mask[rs + j : rs + j + window_len]
                        if self.lsm_binary_mask:
                            if not np.all(win_slice > spectral_ocean_threshold):
                                continue
                        else:
                            if not np.all(win_slice >= spectral_ocean_threshold):
                                continue
                        base = face * (nside * nside) + row * nside + rs + j
                        global_indices = base + np.arange(window_len, dtype=np.int64)
                        all_windows.append(global_indices)
                        row_contributed = True
                if row_contributed:
                    n_rows_with_swaths += 1
        n_swaths = len(all_windows)
        if n_swaths == 0:
            log.warning(
                "spectral_loss: no valid ocean swaths found (n_swaths=0). "
                "Spectral term disabled. Check land-sea mask and spectral_window."
            )
            self._spectral_lookup = None
            self._spectral_hann = None
            return
        idx_windows = np.stack(all_windows, axis=0).astype(np.int64)
        self._spectral_lookup = th.from_numpy(idx_windows).to(device)
        self._spectral_hann = th.hann_window(
            window_len, periodic=True, device=device, dtype=th.float32
        )
        # Verification logging
        log.info(
            "spectral_loss: n_swaths=%d, n_rows_with_swaths=%d, lsm_binary_mask=%s (PAD_XY, no regrid)",
            n_swaths,
            n_rows_with_swaths,
            self.lsm_binary_mask,
        )
        mask_flat = mask_np.reshape(-1)
        mask_at_swath = mask_flat[idx_windows.ravel()]
        mask_min, mask_max = float(mask_at_swath.min()), float(mask_at_swath.max())
        if self.lsm_binary_mask and (mask_min < 0.999 or mask_max > 1.001):
            log.warning(
                "spectral_loss: binary mask check failed (mask at swath indices min=%s max=%s). "
                "Expected all 1.0 for pure-ocean swaths.",
                mask_min,
                mask_max,
            )
        else:
            log.info(
                "spectral_loss: mask at swath indices min=%s max=%s (pure ocean check ok)",
                mask_min,
                mask_max,
            )

    def _to_flat_pad_xy(self, x: th.Tensor, face_dim: int) -> th.Tensor:
        """Flatten HEALPix PAD_XY (face) layout to 1D [..., F*H*W]. No regridding."""
        x = th.movedim(x, face_dim, -3)
        if x.shape[-3:] != (12, self._spectral_nside, self._spectral_nside):
            raise ValueError(
                f"Expected trailing (12, nside, nside), got {x.shape[-3:]}"
            )
        # movedim returns a view with altered strides. Call contiguous() so the following
        # reshape is a view rather than an implicit copy; keeps intent and cost explicit.
        return x.contiguous().reshape(*x.shape[:-3], -1)

    def _compute_power_spectrum_loss(
        self, prediction: th.Tensor, target: th.Tensor, average_channels: bool
    ) -> th.Tensor:
        """
        Compute log-spectral MSE between prediction and target over ocean swaths.
        prediction: [Cond, B, F, T, C, H, W], target: [B, F, T, C, H, W] (unweighted).
        """
        if self._spectral_lookup is None:
            return th.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        idx = self._spectral_channel_idx
        pred_spec = prediction[:, :, :, :, idx, :, :]
        tar_spec = target[:, :, :, idx, :, :]
        n_cond, b, f, t, c_spec, h, w = pred_spec.shape
        npix = 12 * self._spectral_nside ** 2
        with th.cuda.amp.autocast(enabled=False):
            pred_f = pred_spec.float()
            tar_f = tar_spec.float()
            pred_flat = self._to_flat_pad_xy(pred_f, face_dim=2)
            tar_flat = self._to_flat_pad_xy(tar_f, face_dim=1)
            pred_ring = pred_flat.reshape(n_cond * b * t * c_spec, npix)
            tar_ring = tar_flat.reshape(b * t * c_spec, npix)
            lookup = self._spectral_lookup
            n_swaths, window_len = lookup.shape
            pred_swaths = pred_ring[:, lookup]
            tar_swaths = tar_ring[:, lookup]
            pred_swaths = pred_swaths.reshape(n_cond, b * t * c_spec, n_swaths, window_len)
            tar_swaths = tar_swaths.reshape(1, b * t * c_spec, n_swaths, window_len)
            hann = self._spectral_hann
            pred_tapered = pred_swaths * hann[None, None, None, :]
            tar_tapered = tar_swaths * hann[None, None, None, :]
            pred_fft = th.fft.rfft(pred_tapered, dim=-1)
            tar_fft = th.fft.rfft(tar_tapered, dim=-1)
            pred_power = (pred_fft.real ** 2 + pred_fft.imag ** 2)[..., 1:]
            tar_power = (tar_fft.real ** 2 + tar_fft.imag ** 2)[..., 1:]
            n_freq = pred_power.shape[-1]
            # Average over swaths first, but keep batch/time/ensemble separate
            pred_power_mean_swath = pred_power.mean(dim=2)
            tar_power_mean_swath = tar_power.mean(dim=2)
            pred_power_mean_swath = pred_power_mean_swath.reshape(
                n_cond, b, t, c_spec, n_freq
            )
            tar_power_mean_swath = tar_power_mean_swath.reshape(
                b, t, c_spec, n_freq
            )
            eps = 1e-8
            # Compute log-spectra per member, batch, time, channel
            log_pred = th.log(pred_power_mean_swath + eps)                # [n_cond, B, T, C_spec, F]
            log_tar = th.log(tar_power_mean_swath + eps).unsqueeze(0)     # [1,      B, T, C_spec, F]
            # MSE over frequency per (member, batch, time, channel)
            mse_per_sample = ((log_pred - log_tar) ** 2).mean(dim=-1)     # [n_cond, B, T, C_spec]
            spec_weights = self.loss_weights[idx]                          # [C_spec]
            if average_channels:
                # Weight channels, then average over (ensemble, batch, time) but divide by
                # total channel count C so spectral_weight is comparable to base CRPS (which
                # uses .mean() over all dims including C). Otherwise C_spec=2 with C=20 would
                # amplify the spectral term by 10x.
                weighted = mse_per_sample * spec_weights.view(1, 1, 1, -1)
                c_total = prediction.shape[4]
                spec_loss = weighted.sum() / (n_cond * b * t * c_total)
            else:
                # Keep per-channel loss: average over ensemble, batch, time only
                weighted = mse_per_sample * spec_weights.view(1, 1, 1, -1)
                spec_loss = weighted.mean(dim=(0, 1, 2))                   # [C_spec]
        return spec_loss

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

    def _calculate_variogram_loss(self, prediction, target, average_channels=True):
        """
        Calculates the interior-only Masked Variogram Score using spatial tensor slicing.
        For each lag h in self.variogram_lags, computes horizontal and vertical p-th order
        absolute differences across ensemble members, then penalizes squared error against
        the target differences. Only valid ocean pixel pairs (both endpoints ocean) contribute.

        Expects prediction shape: [Cond, B, F, T, C, H, W]
        Expects target shape:     [B, F, T, C, H, W]
        """
        p = self.variogram_p
        has_real_mask = self.lsm_tensor.shape[-1] > 1

        total_vs_sum = 0
        total_mask_sum = 0

        n_b, n_t, n_c = target.shape[0], target.shape[2], target.shape[3]

        for lag in self.variogram_lags:
            # --- Horizontal lags (shift along W) ---
            pred_h1, pred_h2 = prediction[..., :, :-lag], prediction[..., :, lag:]
            tar_h1, tar_h2 = target[..., :, :-lag], target[..., :, lag:]

            if has_real_mask:
                valid_h = self.lsm_tensor[..., :, :-lag] * self.lsm_tensor[..., :, lag:]
            else:
                valid_h = self.lsm_tensor

            # Clamp to avoid infinite gradient from x^(p-1) at x=0 when p < 1
            exp_pred_h = th.abs(pred_h1 - pred_h2).clamp(min=1e-6).pow(p).mean(dim=0)
            tar_diff_h = th.abs(tar_h1 - tar_h2).clamp(min=1e-6).pow(p)
            vs_h = ((exp_pred_h - tar_diff_h) ** 2) * valid_h

            # --- Vertical lags (shift along H) ---
            pred_v1, pred_v2 = prediction[..., :-lag, :], prediction[..., lag:, :]
            tar_v1, tar_v2 = target[..., :-lag, :], target[..., lag:, :]

            if has_real_mask:
                valid_v = self.lsm_tensor[..., :-lag, :] * self.lsm_tensor[..., lag:, :]
            else:
                valid_v = self.lsm_tensor

            exp_pred_v = th.abs(pred_v1 - pred_v2).clamp(min=1e-6).pow(p).mean(dim=0)
            tar_diff_v = th.abs(tar_v1 - tar_v2).clamp(min=1e-6).pow(p)
            vs_v = ((exp_pred_v - tar_diff_v) ** 2) * valid_v

            # --- Accumulate ---
            if average_channels:
                total_vs_sum += vs_h.sum() + vs_v.sum()
                if has_real_mask:
                    total_mask_sum += (valid_h.sum() + valid_v.sum()) * n_b * n_t * n_c
                else:
                    total_mask_sum += vs_h.numel() + vs_v.numel()
            else:
                total_vs_sum += vs_h.sum(dim=(0, 1, 2, 4, 5)) + vs_v.sum(dim=(0, 1, 2, 4, 5))
                if has_real_mask:
                    total_mask_sum += (valid_h.sum() + valid_v.sum()) * n_b * n_t
                else:
                    total_mask_sum += (vs_h.numel() + vs_v.numel()) // n_c

        return total_vs_sum / (total_mask_sum + 1e-8)

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

        # Ensemble spread/skill diagnostics on unweighted fields (before channel weights)
        try:
            spread, skill, ssr = self._compute_spread_skill_ssr(prediction, target)
            # Per-channel diagnostics returned from helper; log channel-averaged scalars
            self.last_spread = spread.mean()
            self.last_skill = skill.mean()
            self.last_spread_skill_score = ssr.mean()
        except Exception:
            # Do not let diagnostic failures affect training loss
            self.last_spread = None
            self.last_skill = None
            self.last_spread_skill_score = None

        # #region agent log
        self._debug_ens_fwd_count += 1
        _dc = self._debug_ens_fwd_count
        if _debug_ensemble_rank0() and n == 2 and (_dc <= 5 or _dc % 50 == 0):
            with th.no_grad():
                pd = prediction
                d01 = (pd[0] - pd[1]).abs()
                h1_max = float(d01.max())
                h1_mean = float(d01.mean())
                s2 = 0.5 * (pd[0] - pd[1]) ** 2
                lsm_d = self.lsm_tensor.to(device=pd.device, dtype=pd.dtype)
                weighted_s2 = s2 * lsm_d
                m_eff = float(lsm_d.sum())
                sum_w_s2_ch0 = float(weighted_s2[:, :, :, 0].sum())
                h2_same_member = (
                    float((pd[0, 0] - pd[0, 1]).abs().max())
                    if b > 1
                    else -1.0
                )
                ls = self.last_spread
                ls_f = float(ls.detach()) if ls is not None and th.is_tensor(ls) else None
                cln_lambda_sample = []
                try:
                    root = self._dbg_model_ref
                    if root is not None:
                        root = root.module if hasattr(root, "module") else root
                        for m in root.modules():
                            if m.__class__.__name__ == "ConditionalLayerNorm" and hasattr(
                                m, "_lambda"
                            ):
                                cln_lambda_sample.append(float(getattr(m, "_lambda", -1.0)))
                                if len(cln_lambda_sample) >= 8:
                                    break
                except Exception:
                    pass
                intl_m = None
                try:
                    pf = pd.reshape(n * b, f, t, c, h, w)
                    intl_m = float(
                        th.stack(
                            [
                                (pf[2 * i] - pf[2 * i + 1]).abs().max()
                                for i in range(min(b, pf.shape[0] // 2))
                            ]
                        ).max()
                    )
                except Exception:
                    pass
                _debug_ensemble_log(
                    "H1-H2-H3-H5",
                    "healpix_loss.py:WeightedCRPSLoss.forward",
                    "ensemble_pred_vs_spread_diag",
                    {
                        "call_idx": _dc,
                        "trainer_epoch": int(getattr(self, "_dbg_training_epoch", -1)),
                        "n_members": int(n),
                        "b": int(b),
                        "pred_shape": list(pd.shape),
                        "m0_minus_m1_abs_max": h1_max,
                        "m0_minus_m1_abs_mean": h1_mean,
                        "same_member_diff_max_b0_b1": h2_same_member,
                        "flat_interleaved_adjacent_max": intl_m,
                        "cln_lambda_first_modules": cln_lambda_sample,
                        "masked_sigma2_sum_channel0": sum_w_s2_ch0,
                        "lsm_m_eff": m_eff,
                        "last_spread_scalar": ls_f,
                    },
                )
        # #endregion

        # Spectral term on unweighted fields (before channel weights)
        spec_loss = None
        if self.spectral_weight > 0.0 and self._spectral_lookup is not None:
            spec_loss = self._compute_power_spectrum_loss(
                prediction, target, average_channels
            )
        
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

            self.last_crps_base = loss.mean().detach()

            # Average Global Mean Bias Penalty (masked by lsm so land is ignored when weights are 0)
            if self.mean_penalty > 0:
                lsm = self.lsm_tensor  # [1, F, 1, 1, H, W] or all ones if no lsm_file
                # Masked mean over ocean only: sum(x*lsm) / sum(lsm) over the reduced dims
                n_ens, n_b, n_f, n_t = prediction.shape[0], prediction.shape[1], prediction.shape[2], prediction.shape[3]
                # lsm.sum() already includes the face dimension F, so we should not multiply by n_f again here.
                ocean_count = n_ens * n_b * n_t * lsm.sum()
                ens_global_means = (prediction * lsm).sum(dim=(0, 1, 2, 3, 5, 6)) / (ocean_count + 1e-8)  # [C]
                target_ocean_count = n_b * n_t * lsm.sum()
                target_global_means = (target * lsm).sum(dim=(0, 1, 2, 4, 5)) / (target_ocean_count + 1e-8)  # [C]
                bias_penalty = self.mean_penalty * th.abs(ens_global_means - target_global_means)
                if average_channels:
                    loss += bias_penalty.mean()
                else:
                    loss += bias_penalty
                self.last_mean_penalty = bias_penalty.mean().detach()

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
                self.last_multiscale = (self.multiscale * crps_scales).mean().detach()
            
            # Temporal Dt Loss (Xt - Xt-1)
            if self.temporal_dt > 0.:
                dt_loss = self._calculate_dt_loss(prediction, target, average_channels)
                loss += self.temporal_dt * dt_loss
                self.last_temporal_dt = (self.temporal_dt * dt_loss).mean().detach()

            # Masked Variogram Score (Spatial Structure Penalty)
            if self.variogram > 0.:
                vs_loss = self._calculate_variogram_loss(prediction, target, average_channels)
                loss += self.variogram * vs_loss
                self.last_variogram = (self.variogram * vs_loss).mean().detach()

            # Ocean power-spectrum term
            if self.spectral_weight > 0.0 and spec_loss is not None:
                if average_channels:
                    # spec_loss is scalar (channels already averaged)
                    spectral_contrib = self.spectral_weight * spec_loss
                    loss = loss + spectral_contrib
                    self.last_spectral = spectral_contrib.detach()
                else:
                    # spec_loss has shape [C_spec]; apply only to selected channels
                    spectral_contrib = th.zeros_like(loss)
                    spectral_contrib[self._spectral_channel_idx] = (
                        self.spectral_weight * spec_loss
                    )
                    loss = loss + spectral_contrib
                    self.last_spectral = (self.spectral_weight * spec_loss).mean().detach()
                if not self._spectral_first_forward_done:
                    self._spectral_first_forward_done = True
                    log = logging.getLogger(__name__)
                    n_swaths = self._spectral_lookup.shape[0]
                    log.info(
                        "spectral_loss: first forward applied, n_swaths=%d, last_spectral=%.6f",
                        n_swaths,
                        self.last_spectral.item(),
                    )
            else:
                self.last_spectral = None

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

        # Cached per-component loss scalars for TensorBoard logging
        self.last_crps_base = None
        self.last_spectral = None

        # Cached ensemble diagnostics (spread/skill/SSR) for TensorBoard logging
        self.last_spread = None
        self.last_skill = None
        self.last_spread_skill_score = None


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
        # Move LSM tensor to the trainer device even when no lsm_file is provided.
        # Spread/skill/SSR diagnostics use `self.lsm_tensor` unconditionally, so a
        # CPU/GPU mismatch would otherwise cause diagnostics to silently fail.
        self.lsm_tensor = self.lsm_tensor.to(device=trainer.device)

    def _compute_spread_skill_ssr(self, prediction: th.Tensor, target: th.Tensor):
        """
        Compute domain-averaged spread, skill, and spread-skill ratio (SSR)
        from the unweighted ensemble prediction and target.

        prediction: [Cond, B, F, T, C, H, W]
        target:     [B, F, T, C, H, W]
        """
        if prediction.ndim != 7 or target.ndim != 6:
            raise ValueError(
                f"_compute_spread_skill_ssr expects prediction [Cond, B, F, T, C, H, W] and target [B, F, T, C, H, W], "
                f"got {prediction.shape} and {target.shape}"
            )

        n = self.n_members
        if n < 2:
            raise ValueError("Spread/skill diagnostics require at least 2 ensemble members")

        # Ensemble mean at each grid point
        mu = prediction.mean(dim=0)  # [B, F, T, C, H, W]

        # Local ensemble variance with Bessel's correction (N-1 in denominator)
        diff = prediction - mu.unsqueeze(0)  # [Cond, B, F, T, C, H, W]
        sigma2 = (diff * diff).sum(dim=0) / float(max(n - 1, 1))  # [B, F, T, C, H, W]

        # Local squared error of ensemble mean
        se = (mu - target) ** 2  # [B, F, T, C, H, W]

        # Apply land–sea mask as weight; mask is [1, F, 1, 1, H, W] or all ones
        weighted_sigma2 = sigma2 * self.lsm_tensor  # [B, F, T, C, H, W]
        weighted_se = se * self.lsm_tensor  # [B, F, T, C, H, W]

        eps = 1e-8
        m_eff = self.lsm_tensor.sum()

        # Reduce over batch, faces, time, H, W – keep channels separate
        spread_var_c = weighted_sigma2.sum(dim=(0, 1, 2, 4, 5)) / (m_eff + eps)  # [C]
        skill_mse_c = weighted_se.sum(dim=(0, 1, 2, 4, 5)) / (m_eff + eps)  # [C]

        spread_c = th.sqrt(spread_var_c + eps)  # [C]
        skill_c = th.sqrt(skill_mse_c + eps)  # [C]
        ssr_c = spread_c / (skill_c + eps)  # [C]

        return spread_c.detach(), skill_c.detach(), ssr_c.detach()

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

        # Reset cached diagnostics (avoid stale values if something goes wrong).
        self.last_crps_base = None
        self.last_spectral = None
        self.last_spread = None
        self.last_skill = None
        self.last_spread_skill_score = None

        # Ensemble spread/skill diagnostics on unweighted fields (before channel weights)
        try:
            spread, skill, ssr = self._compute_spread_skill_ssr(prediction, target)
            self.last_spread = spread.mean().detach()
            self.last_skill = skill.mean().detach()
            self.last_spread_skill_score = ssr.mean().detach()
        except Exception:
            self.last_spread = None
            self.last_skill = None
            self.last_spread_skill_score = None

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

            # Cache base CRPS contribution before adding auxiliary spectral/multiscale terms
            self.last_crps_base = loss.mean().detach()

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

                spectral_contrib = self.lambda_spec * spec_loss
                loss = loss + spectral_contrib
                # Log spectral contribution as a scalar (channel-averaged if needed)
                self.last_spectral = (
                    spectral_contrib.detach()
                    if average_channels
                    else spectral_contrib.mean().detach()
                )
            else:
                self.last_spectral = None
            
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

                # Cache base CRPS contribution before adding auxiliary spectral term
                self.last_crps_base = loss.mean().detach()

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
                        spectral_contrib = (
                            self.lambda_spec
                            * self.averaging_coeff
                            * (diff_terms - self.coeff_eps * dist_matrix).sum(dim=(1,2))
                            / (b * t * self.lmax * self.mmax)
                        )

                        loss = loss + spectral_contrib
                        self.last_spectral = spectral_contrib.mean().detach()

            else:
                pred = prediction.reshape(n, -1)
                tar = target.unsqueeze(0).reshape(1, -1) # [1, ...] (first dim will broadcast across ensemble)
                diff = self.pdist(pred, tar) # [Cond]
                dist_matrix = self.pdist(pred.unsqueeze(1), pred.unsqueeze(0))  # [Cond, Cond] 

                diff_terms = self.diag_mask * (diff.unsqueeze(0) + diff.unsqueeze(1)) # [Cond, Cond], diagonal elements zeroed out
                loss = self.averaging_coeff * (diff_terms - self.coeff_eps * dist_matrix).sum()/(b*f*c*t*h*w)

                # Cache base CRPS contribution before adding auxiliary spectral term
                self.last_crps_base = loss.mean().detach()

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
                        spectral_contrib = (
                            self.lambda_spec
                            * self.averaging_coeff
                            * (diff_terms - self.coeff_eps * dist_matrix).sum()
                            / (b * t * self.lmax * self.mmax)
                        )
                        loss = loss + spectral_contrib
                        self.last_spectral = spectral_contrib.detach()

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
    variogram : float, optional
        Weight for the variogram score loss inside `WeightedCRPSLoss`. Zero
        disables the variogram penalty.
    variogram_p : float, optional
        Order of the absolute differences in the variogram score (default 0.5).
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
        variogram: float = 0.0,
        variogram_p: float = 0.5,
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
            variogram=variogram,
            variogram_p=variogram_p,
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
        _inner = getattr(self, "crps_loss", None)
        if _inner is not None and hasattr(_inner, "set_training_epoch"):
            try:
                _inner.set_training_epoch(epoch)
            except Exception:
                pass

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

        # Forward CRPS sub-component scalars so the trainer can find them on the outer criterion
        for attr in (
            "last_crps_base",
            "last_mean_penalty",
            "last_multiscale",
            "last_temporal_dt",
            "last_variogram",
            "last_spectral",
            "last_spread",
            "last_skill",
            "last_spread_skill_score",
        ):
            setattr(self, attr, getattr(self.crps_loss, attr, None))

        return total
