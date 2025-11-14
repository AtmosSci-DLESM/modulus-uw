import numpy as np
import torch
import xarray as xr

class NonnegativeConstraint(torch.nn.Module):
    def __init__(
        self,
        variables: list[str],
        channels: list[str],
        scaling: dict[str, dict[str, float]],
    ):
        """
        Parameters
        ----------
        variables: list[str]
            List of variable names to apply the constraint to.
        channels: list[str]
            List of all input channel names in the model.
        scaling: dict[str, dict[str, float]]
            Dictionary containing the mean and std for each variable.
        """
        super().__init__()
        self.variables = variables
        self.channels = channels
        self.scaling = scaling

        # Only apply constraint to variables that are used by model
        self.variables = [var for var in self.variables if var in channels]

        var_indices = torch.tensor(
            [channels.index(var) for var in self.variables],
            dtype=torch.long
        )
        self.register_buffer('var_indices', var_indices, persistent=False)

        self.var_means = torch.tensor([scaling[var]['mean'] for var in self.variables])
        self.var_stds = torch.tensor([scaling[var]['std'] for var in self.variables])

        thresholds = (0. - self.var_means) / self.var_stds
        thresholds = thresholds.view(1, 1, 1, -1, 1, 1)
        self.register_buffer('thresholds', thresholds, persistent=False)

    def forward(self, prediction, input):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        x = prediction
        selected_vars = torch.index_select(x, dim=3, index=self.var_indices)
        clamped = torch.maximum(selected_vars, self.thresholds).to(x.dtype)
        x.index_copy_(3, self.var_indices, clamped)

        return x

class DryAirMassConstraint(torch.nn.Module):
    def __init__(
        self,
        channels: list[str],
        scaling: dict[str, dict[str, float]],
        transformed_sp: bool = False,
        sp_boxcox_lambda: float = 19.632015209145543,
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
        self.transformed_sp = transformed_sp
        self.sp_boxcox_lambda = sp_boxcox_lambda

        sp_name = "sp-boxcox" if transformed_sp else "sp"
        sp_idx = torch.tensor(
            channels.index(sp_name),
            dtype=torch.long
        )
        self.register_buffer('sp_idx', sp_idx, persistent=False)

        tcwv_idx = torch.tensor(
            channels.index("tcwv"),
            dtype=torch.long
        )
        self.register_buffer('tcwv_idx', tcwv_idx, persistent=False)

        ps_mean = torch.tensor(scaling[sp_name]['mean'])
        ps_std = torch.tensor(scaling[sp_name]['std'])
        tcwv_mean = torch.tensor(scaling["tcwv"]['mean'])
        tcwv_std = torch.tensor(scaling["tcwv"]['std'])
        self.register_buffer('ps_mean', ps_mean, persistent=False) 
        self.register_buffer('ps_std', ps_std, persistent=False)
        self.register_buffer('tcwv_mean', tcwv_mean, persistent=False)
        self.register_buffer('tcwv_std', tcwv_std, persistent=False)

        if transformed_sp:
            # Find threshold in Box-Cox space that corresponds to 0 Pa in physical space
            sp_threshold = (0.**self.sp_boxcox_lambda-1)/self.sp_boxcox_lambda
            sp_threshold = (sp_threshold - ps_mean) / ps_std
        else:
            sp_threshold = (0. - ps_mean) / ps_std
        sp_threshold = sp_threshold.view(1, 1, 1, -1, 1, 1)
        self.register_buffer('sp_threshold', sp_threshold, persistent=False)

        self.g0 = 9.81

    def transform_sp(
        self,
        x,
        lam=19.632015209145543,
        max_val=1068.063515625,
    ):
        x = x / max_val
        x = (x**lam - 1) / lam  # Box-Cox
        return x

    def reverse_transform_sp(
        self,
        x,
        lam=19.632015209145543,
        max_val=1068.063515625,
    ):
        x = torch.exp(torch.log(x * lam + 1 + 1e-8)/lam) # reverse Box-Cox
        x = x * max_val # Rescaling back to hPa
        return x

    def forward(self, prediction, input):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        with torch.cuda.amp.autocast(enabled=False):
            x = prediction.float()

            # Get predicted sp and tcwv
            sp = torch.index_select(x, dim=3, index=self.sp_idx)
            sp = sp * self.ps_std + self.ps_mean
            if self.transformed_sp:
                sp = self.reverse_transform_sp(sp)
            tcwv = torch.index_select(x, dim=3, index=self.tcwv_idx)
            tcwv = tcwv * self.tcwv_std + self.tcwv_mean

            # Get last time step sp and tcwv from input
            sp_0 = torch.index_select(input, dim=3, index=self.sp_idx)[:, :, -1:]
            sp_0 = sp_0 * self.ps_std + self.ps_mean
            if self.transformed_sp:
                sp_0 = self.reverse_transform_sp(sp_0)
            tcwv_0 = torch.index_select(input, dim=3, index=self.tcwv_idx)[:, :, -1:]
            tcwv_0 = tcwv_0 * self.tcwv_std + self.tcwv_mean

            # Get predicted and initial dry sp
            sp_dry = sp - self.g0 * tcwv/100.
            sp_0_dry = sp_0 - self.g0 * tcwv_0/100.
            # Correction is spatial average of dry air mass difference
            correction = (sp_dry - sp_0_dry).mean(dim=[1,4,5], keepdim=True)
            sp_corrected = sp - correction

            # Ensure sp is non-negative and rescale back to normalized space
            sp_corrected = torch.clamp(sp_corrected, min=0.)
            if self.transformed_sp:
                sp_corrected = self.transform_sp(sp_corrected)
            sp_corrected = (sp_corrected - self.ps_mean) / self.ps_std

            x.index_copy_(3, self.sp_idx, sp_corrected)

            return x
