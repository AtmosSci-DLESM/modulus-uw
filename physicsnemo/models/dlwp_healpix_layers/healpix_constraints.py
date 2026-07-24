import numpy as np
import torch
import xarray as xr

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

logger = PythonLogger(name="physicsnemo.models.dlwp_healpix_layers.healpix_constraints")
if DistributedManager.is_initialized():
    logger0 = RankZeroLoggingWrapper(logger, DistributedManager())
else:
    logger0 = logger

'''
Constraints for the DLWP HEALPix model. All constraints should take two arguments:
- prediction: the predicted tensor from the model
- input: the input tensor to the model
The constraint should return the prediction tensor with the constraints applied.
Both the prediction and input tensors are expected to be in the shape [B, F, T, C, H, W].
The input tensor may not be used for some constraints but is always expected as
an input argument for consistency.
'''


def replace_value_keep_gradient(
    x: torch.Tensor, new_value: torch.Tensor
) -> torch.Tensor:
    """
    Use ``new_value`` in the forward pass but keep ``x``'s gradient in the backward pass.

    Straight-through estimator for hard corrections (e.g. nonnegative clamps): the
    forward value is the projected ``new_value``, while backward treats the op as
    identity so saturated cells still receive a learning signal.
    """
    return x + (new_value - x).detach()


class NonnegativeConstraint(torch.nn.Module):
    def __init__(
        self,
        variables: list[str],
        in_channels: list[str],
        out_channels: list[str],
        scaling: dict[str, dict[str, float]],
        keep_grad_through_clamp: bool = False,
    ):
        """
        Parameters
        ----------
        variables: list[str]
            List of variable names to apply the constraint to.
        in_channels: list[str]
            List of all input channel names in the model.
        out_channels: list[str]
            List of all output channel names in the model.
        scaling: dict[str, dict[str, float]]
            Dictionary containing the mean and std for each variable.
        keep_grad_through_clamp: bool, optional
            If True, apply the nonnegative clamp with a straight-through
            estimator: forward values are still clamped to physical zero, but
            gradients flow as if the clamp were identity so below-threshold
            predictions still get a learning signal. Default False.
        """
        super().__init__()
        self.variables = variables
        if out_channels is not None:
            self.channels = out_channels
        else:
            self.channels = in_channels
        self.scaling = scaling
        self.keep_grad_through_clamp = keep_grad_through_clamp

        # Only apply constraint to variables that are used by model
        missing = [var for var in self.variables if var not in self.channels]
        self.variables = [var for var in self.variables if var in self.channels]

        if missing:
            logger0.warning(
                f"Requested non-negative constrained variables "
                f"{missing} not found in model channels and will be ignored."
            )

        constrained_set = set(self.variables)
        per_channel = [
            (0.0 - scaling[name]["mean"]) / scaling[name]["std"]
            if name in constrained_set
            else float("-inf")
            for name in self.channels
        ]
        thresholds = torch.tensor(per_channel, dtype=torch.float32).view(
            1, 1, 1, -1, 1, 1
        )
        self.register_buffer("thresholds", thresholds, persistent=False)

    def forward(self, prediction, input):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        thresholds = self.thresholds.to(dtype=prediction.dtype)
        clamped = torch.maximum(prediction, thresholds)
        if self.keep_grad_through_clamp:
            return replace_value_keep_gradient(prediction, clamped)
        return clamped

class DryAirMassConstraint(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[str],
        out_channels: list[str],
        scaling: dict[str, dict[str, float]],
    ):
        """
        Parameters
        ----------
        in_channels: list[str]
            List of all input channel names in the model.
        out_channels: list[str]
            List of all output channel names in the model.
        scaling: dict[str, dict[str, float]]
            Dictionary containing the mean and std for each variable.
        """
        super().__init__()
        if out_channels is not None:
            self.channels = out_channels
        else:
            self.channels = in_channels
        self.scaling = scaling

        self.sp_channel_index = self.channels.index("sp")
        self.tcwv_channel_index = self.channels.index("tcwv")

        ps_mean = torch.tensor(scaling['sp']['mean'])
        ps_std = torch.tensor(scaling['sp']['std'])
        tcwv_mean = torch.tensor(scaling['tcwv']['mean'])
        tcwv_std = torch.tensor(scaling['tcwv']['std'])
        self.register_buffer('ps_mean', ps_mean, persistent=False) 
        self.register_buffer('ps_std', ps_std, persistent=False)
        self.register_buffer('tcwv_mean', tcwv_mean, persistent=False)
        self.register_buffer('tcwv_std', tcwv_std, persistent=False)

        sp_channel_mask = torch.zeros(len(self.channels), dtype=torch.float32)
        sp_channel_mask[self.sp_channel_index] = 1.0
        self.register_buffer(
            "sp_channel_mask",
            sp_channel_mask.view(1, 1, 1, -1, 1, 1),
            persistent=False,
        )

        self.g0 = 9.81

    def forward(self, prediction, input):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        
        # Need to scale to physical units and compute small differences of large
        # surface pressures (in Pa), so disable autocast and force float32 precision
        with torch.amp.autocast('cuda', enabled=False):
            prediction = prediction.float()
            input = input.float()

            # Slice on dim 3 with constant bounds (compile-friendly; avoids
            # index_select + buffer index in backward).
            sp = prediction[:, :, :, self.sp_channel_index : self.sp_channel_index + 1, :, :]
            sp = sp * self.ps_std + self.ps_mean
            tcwv = prediction[:, :, :, self.tcwv_channel_index : self.tcwv_channel_index + 1, :, :]
            tcwv = tcwv * self.tcwv_std + self.tcwv_mean

            # Get sp and tcwv from last time step of input tensor. Used to 
            # compute initial dry air mass which is to be conserved.
            sp_0 = input[:, :, -1:, self.sp_channel_index : self.sp_channel_index + 1, :, :]
            sp_0 = sp_0 * self.ps_std + self.ps_mean
            tcwv_0 = input[:, :, -1:, self.tcwv_channel_index : self.tcwv_channel_index + 1, :, :]
            tcwv_0 = tcwv_0 * self.tcwv_std + self.tcwv_mean

            # Get predicted and initial dry sp
            sp_dry = sp - self.g0 * tcwv
            sp_0_dry = sp_0 - self.g0 * tcwv_0
            # Correction is spatial average of dry air mass difference
            correction = (sp_dry - sp_0_dry).mean(dim=[1,4,5], keepdim=True)
            sp_corrected = sp - correction

            # Ensure sp is non-negative and rescale back to normalized space
            sp_corrected = torch.clamp(sp_corrected, min=0.)
            sp_corrected = (sp_corrected - self.ps_mean) / self.ps_std

            mask = self.sp_channel_mask.to(
                device=prediction.device, dtype=prediction.dtype
            )
            out = prediction * (1.0 - mask) + sp_corrected * mask

            return out