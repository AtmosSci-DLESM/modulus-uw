import logging
import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)

class NonnegativeConstraint(torch.nn.Module):
    def __init__(
        self,
        variables: list[str],
        in_channels: list[str],
        out_channels: list[str],
        scaling: dict[str, dict[str, float]],
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
        """
        super().__init__()
        self.variables = variables
        if out_channels is not None:
            self.channels = out_channels
        else:
            self.channels = in_channels
        self.scaling = scaling

        # Only apply constraint to variables that are used by model
        self.variables = [var for var in self.variables if var in self.channels]

        logger.warning(
            f"Requested non-negative constrained variables "
            f"{[v for v in variables if v not in self.variables]} not found in "
            f"model channels and will be ignored."
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
        return torch.maximum(
            prediction, self.thresholds.to(dtype=prediction.dtype)
        )

class DryAirMassConstraint(torch.nn.Module):
    def __init__(
        self,
        channels: list[str],
        scaling: dict[str, dict[str, float]],
    ):
        """
        Parameters
        ----------
        channels: list[str]
            List of all input channel names in the model.
        scaling: dict[str, dict[str, float]]
            Dictionary containing the mean and std for each variable.
        """
        super().__init__()
        self.channels = channels
        self.scaling = scaling

        sp_idx = torch.tensor(
            channels.index('sp'),
            dtype=torch.long
        )
        self.register_buffer('sp_idx', sp_idx, persistent=False)

        tcwv_idx = torch.tensor(
            channels.index("tcwv"),
            dtype=torch.long
        )
        self.register_buffer('tcwv_idx', tcwv_idx, persistent=False)

        ps_mean = torch.tensor(scaling['sp']['mean'])
        ps_std = torch.tensor(scaling['sp']['std'])
        tcwv_mean = torch.tensor(scaling['tcwv']['mean'])
        tcwv_std = torch.tensor(scaling['tcwv']['std'])
        self.register_buffer('ps_mean', ps_mean, persistent=False) 
        self.register_buffer('ps_std', ps_std, persistent=False)
        self.register_buffer('tcwv_mean', tcwv_mean, persistent=False)
        self.register_buffer('tcwv_std', tcwv_std, persistent=False)

        sp_channel_mask = torch.zeros(len(channels), dtype=torch.float32)
        sp_channel_mask[channels.index("sp")] = 1.0
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
        with torch.amp.autocast('cuda',enabled=False):
            prediction = prediction.float()
            input = input.float()

            # Get predicted sp and tcwv
            sp = torch.index_select(prediction, dim=3, index=self.sp_idx)
            sp = sp * self.ps_std + self.ps_mean
            tcwv = torch.index_select(prediction, dim=3, index=self.tcwv_idx)
            tcwv = tcwv * self.tcwv_std + self.tcwv_mean

            # Get sp and tcwv from last time step of input tensor. Used to 
            # compute initial dry air mass which is to be conserved.
            sp_0 = torch.index_select(input, dim=3, index=self.sp_idx)[:, :, -1:]
            sp_0 = sp_0 * self.ps_std + self.ps_mean
            tcwv_0 = torch.index_select(input, dim=3, index=self.tcwv_idx)[:, :, -1:]
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