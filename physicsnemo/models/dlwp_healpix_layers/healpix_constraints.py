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

class BoundConstraint(torch.nn.Module):
    def __init__(
        self,
        bounds: dict[str, list[float]],
        channels: list[str],
        scaling: dict[str, dict[str, float]],
    ):
        """
        Parameters
        ----------
        bounds: dict[str, list[float]]
            Dictionary mapping variable names to [min, max] limits.
            Use None for unbounded limits.
            Example: {'sic': [0.0, 1.0], 'sit': [0.0, None]}
        channels: list[str]
            List of all input channel names in the model.
        scaling: dict[str, dict[str, float]]
            Dictionary containing the mean and std for each variable.
        """
        super().__init__()
        self.bounds = bounds
        self.channels = channels
        self.scaling = scaling

        # Only apply constraint to variables that are used by model
        self.active_vars = [var for var in self.bounds.keys() if var in channels]

        var_indices = torch.tensor(
            [channels.index(var) for var in self.active_vars],
            dtype=torch.long
        )
        self.register_buffer('var_indices', var_indices, persistent=False)

        # Pre-compute normalized thresholds
        min_thresholds = []
        max_thresholds = []

        for var in self.active_vars:
            mu = scaling[var]['mean']
            std = scaling[var]['std']
            phys_min, phys_max = self.bounds[var]

            # Normalize Lower Bound: (Min - Mean) / Std
            if phys_min is not None:
                norm_min = (phys_min - mu) / std
            else:
                norm_min = float('-inf')

            # Normalize Upper Bound: (Max - Mean) / Std
            if phys_max is not None:
                norm_max = (phys_max - mu) / std
            else:
                norm_max = float('inf')

            min_thresholds.append(norm_min)
            max_thresholds.append(norm_max)

        # Reshape for broadcasting against [B, F, T, C, H, W]
        shape_view = (1, 1, 1, -1, 1, 1)

        self.register_buffer(
            'min_thresholds',
            torch.tensor(min_thresholds).view(shape_view),
            persistent=False,
        )
        self.register_buffer(
            'max_thresholds',
            torch.tensor(max_thresholds).view(shape_view),
            persistent=False,
        )

    def forward(self, x):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        selected_vars = torch.index_select(x, dim=3, index=self.var_indices)

        # Apply clamping (handles both min and max simultaneously)
        clamped = torch.clamp(
            selected_vars,
            min=self.min_thresholds,
            max=self.max_thresholds,
        ).to(x.dtype)

        x.index_copy_(3, self.var_indices, clamped)

        return x
