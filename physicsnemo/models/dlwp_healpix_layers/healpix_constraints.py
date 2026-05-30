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

    def forward(self, x):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        selected_vars = torch.index_select(x, dim=3, index=self.var_indices)
        clamped = torch.maximum(selected_vars, self.thresholds).to(x.dtype)
        x.index_copy_(3, self.var_indices, clamped)

        return x


class ImputationConstraint(torch.nn.Module):
    def __init__(
        self,
        variables: list[str],
        channels: list[str],
        impute_values: list[float],
        mask_file: str,
        scaling: dict[str, dict[str, float]],
    ):
        """
        Parameters
        ----------
        variables: list[str]
            List of variable names to apply the constraint to.
        channels: list[str]
            List of all input channel names in the model.
        impute_values: list[float]
            List of values to impute for each variable.
        mask_file: str
            Path to the mask file.
        scaling: dict[str, dict[str, float]]
            Dictionary containing the mean and std for each variable.
        """
        super().__init__()
        self.variables = variables
        self.channels = channels
        self.scaling = scaling

        # flag for moving to device, handled in forward pass
        self._on_device = False

        # Only apply constraint to variables that are used by model
        self.variables = [var for var in self.variables if var in channels]

        self.var_means = torch.tensor([scaling[var]['mean'] for var in self.variables])
        self.var_stds = torch.tensor([scaling[var]['std'] for var in self.variables])

        # calculate imputs values
        self.impute_values = self.calculate_impute_values(impute_values)
        # self.mask = self.calculate_mask()
        self.mask = self.prepare_mask(mask_file)

    def calculate_impute_values(self, impute_values):
        '''
        Calculate the impute values in normalized space.
        '''
        impute_values_norm  = []
        for i, var in enumerate(self.variables):
            impute_values_norm.append((impute_values[i] - self.scaling[var]['mean']) / self.scaling[var]['std'])
        impute_values = torch.tensor(impute_values_norm)
        # add singleton dimension for batch, time, face, height, and width
        return impute_values.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5)

    def prepare_mask(self, mask_file):
        '''
        Prepare the mask for the imputation constraint.
        '''
        # invert mask
        mask = torch.tensor(xr.open_dataset(mask_file).mask.values, dtype=torch.bool)
        mask = ~mask

        # add singleton dimension for batch, time, and channel
        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)   
        return torch.tensor(mask)

    def forward(self, x):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''

        if not self._on_device:
            self.mask = self.mask.to(x.device)
            self.impute_values = self.impute_values.to(x.device)
            self._on_device = True

        constrained =  torch.where(self.mask, self.impute_values, x)

        # printing statement useful for checking constraint behavior 
        # if True:            
        #     print_exit= '/pscratch/sd/n/nacc/veggie-dltm/dlesym_pipeline/Notebooks/test_constraint'
        #     print(f'saving inputs to {print_exit}_x.npy, {print_exit}_constrained.npy')
        #     # convert to numpy as save as pickle
        #     np.save(f'{print_exit}_x.npy', x.cpu().numpy())
        #     np.save(f'{print_exit}_constrained.npy', constrained.cpu().numpy())

        return constrained