import numpy as np
import torch
import xarray as xr

class NonnegativeConstraint(torch.nn.Module):
    def __init__(
        self,
        variables: list[str],
        in_channels: list[str],
        out_channels: list[str],
        scaling: dict[str, dict[str, float]],
        sp_boxcox_lambda: float = 19.632015209145543,
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
        sp_boxcox_lambda: float
            The lambda parameter for the Box-Cox transformed surface pressure.
            Only used if sp-boxcox is an input/output variable.
        """
        super().__init__()
        self.variables = variables
        if out_channels is not None:
            self.channels = out_channels
        else:
            self.channels = in_channels
        self.scaling = scaling
        self.sp_boxcox_lambda = sp_boxcox_lambda

        # Only apply constraint to variables that are used by model
        self.variables = [var for var in self.variables if var in self.channels]

        var_indices = torch.tensor(
            [self.channels.index(var) for var in self.variables],
            dtype=torch.long
        )
        self.register_buffer('var_indices', var_indices, persistent=False)

        self.var_means = torch.tensor([scaling[var]['mean'] for var in self.variables])
        self.var_stds = torch.tensor([scaling[var]['std'] for var in self.variables])

        thresholds = (0. - self.var_means) / self.var_stds
        if 'sp-boxcox' in self.variables:
            sp_idx = self.variables.index('sp-boxcox')
            # Find threshold in Box-Cox space that corresponds to 0 Pa in physical space
            thresholds[sp_idx] = (0.**self.sp_boxcox_lambda-1)/self.sp_boxcox_lambda
            thresholds[sp_idx] = (thresholds[sp_idx] - self.var_means[sp_idx]) / self.var_stds[sp_idx]
        thresholds = thresholds.view(1, 1, 1, -1, 1, 1)
        self.register_buffer('thresholds', thresholds, persistent=False)

    def forward(self, x):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        selected_vars = torch.index_select(x, dim=3, index=self.var_indices)
        clamped = torch.maximum(selected_vars, self.thresholds).to(x.dtype)
        x.index_copy_(3, self.var_indices, clamped)

        return x
