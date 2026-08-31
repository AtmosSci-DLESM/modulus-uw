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


class ClampInteriorSTE(torch.autograd.Function):
    """Hard clamp in forward. Backward is identity inside the interval.

    Outside, pass ``grad_output`` only when it points into the feasible set:
    increase when below the lower bound, decrease when above the upper bound.
    Outward gradients are dropped so the pre-clamp latent does not drift
    further out of bounds while the forward value stays at the bound.
    """

    @staticmethod
    def forward(ctx, x, lower, upper):
        ctx.save_for_backward(x, lower, upper)
        return torch.minimum(torch.maximum(x, lower), upper)

    @staticmethod
    def backward(ctx, grad_output):
        x, lower, upper = ctx.saved_tensors
        below = x < lower
        above = x > upper
        inside = ~(below | above)
        zeros = torch.zeros_like(grad_output)
        grad = torch.where(inside, grad_output, zeros)
        grad = torch.where(below & (grad_output < 0), grad_output, grad)
        grad = torch.where(above & (grad_output > 0), grad_output, grad)
        return grad, None, None


def clamp_keep_interior_grad(
    x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Hard clamp ``x`` to ``[lower, upper]`` with interior-only STE backward."""
    return ClampInteriorSTE.apply(x, lower, upper)


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
            If True, forward is still the hard clamp to physical zero, but
            backward passes ``∂L/∂y`` on saturated cells only when it points
            into the feasible set (increase when below the lower bound).
            Default False uses the true clamp Jacobian (zeros on
            saturated cells).
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
        from physicsnemo.datapipes.healpix.scaling_utils import stack_normalized_bounds

        thresholds_np = stack_normalized_bounds(
            self.channels,
            scaling,
            {name: 0.0 for name in constrained_set},
            unconstrained=float("-inf"),
        )
        thresholds = torch.as_tensor(thresholds_np, dtype=torch.float32)
        self.register_buffer("thresholds", thresholds, persistent=False)
        self.register_buffer(
            "unconstrained_upper",
            torch.tensor(float("inf")),
            persistent=False,
        )

    def forward(self, prediction, input):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        thresholds = self.thresholds.to(dtype=prediction.dtype)
        if self.keep_grad_through_clamp:
            return clamp_keep_interior_grad(
                prediction,
                thresholds,
                self.unconstrained_upper.to(dtype=prediction.dtype),
            )
        return torch.maximum(prediction, thresholds)


class BoundedConstraint(torch.nn.Module):
    def __init__(
        self,
        lower_bounds: dict[str, float] | None = None,
        upper_bounds: dict[str, float] | None = None,
        in_channels: list[str] | None = None,
        out_channels: list[str] | None = None,
        scaling: dict[str, dict[str, float]] | None = None,
        keep_grad_through_clamp: bool = False,
    ):
        """
        Clamp model outputs to optional physical lower and/or upper bounds.

        Bounds are specified in physical units and converted to normalized
        thresholds using ``scaling``. Channels without a bound on a given side
        are left unconstrained on that side.

        Parameters
        ----------
        lower_bounds: dict[str, float], optional
            Per-variable physical lower bounds (inclusive).
        upper_bounds: dict[str, float], optional
            Per-variable physical upper bounds (inclusive).
        in_channels: list[str]
            List of all input channel names in the model.
        out_channels: list[str]
            List of all output channel names in the model.
        scaling: dict[str, dict[str, float]]
            Dictionary containing the mean and std for each variable.
        keep_grad_through_clamp: bool, optional
            If True, forward is still the hard clamp, but backward passes
            ``∂L/∂y`` on saturated cells only when it points into the feasible
            set (increase below a lower bound, decrease above an upper bound).
            Default False uses the true clamp Jacobian (zeros on saturated
            cells).
        """
        super().__init__()
        lower_bounds = lower_bounds or {}
        upper_bounds = upper_bounds or {}
        if not lower_bounds and not upper_bounds:
            raise ValueError(
                "BoundedConstraint requires at least one of lower_bounds or "
                "upper_bounds."
            )

        if out_channels is not None:
            self.channels = out_channels
        else:
            self.channels = in_channels
        self.scaling = scaling
        self.keep_grad_through_clamp = keep_grad_through_clamp

        requested = set(lower_bounds) | set(upper_bounds)
        for name in requested:
            if name in lower_bounds and name in upper_bounds:
                if lower_bounds[name] > upper_bounds[name]:
                    raise ValueError(
                        f"BoundedConstraint for {name!r}: lower bound "
                        f"{lower_bounds[name]} exceeds upper bound "
                        f"{upper_bounds[name]}."
                    )

        missing = [var for var in requested if var not in self.channels]
        if missing:
            logger0.warning(
                f"Requested bounded variables {missing} not found in model "
                "channels and will be ignored."
            )
        self.constrained_variables = {
            name for name in requested if name in self.channels
        }

        from physicsnemo.datapipes.healpix.scaling_utils import stack_normalized_bounds

        lower_thresholds_np = stack_normalized_bounds(
            self.channels,
            scaling,
            {
                name: lower_bounds[name]
                for name in self.constrained_variables
                if name in lower_bounds
            },
            unconstrained=float("-inf"),
        )
        upper_thresholds_np = stack_normalized_bounds(
            self.channels,
            scaling,
            {
                name: upper_bounds[name]
                for name in self.constrained_variables
                if name in upper_bounds
            },
            unconstrained=float("inf"),
        )
        lower_thresholds = torch.as_tensor(lower_thresholds_np, dtype=torch.float32)
        upper_thresholds = torch.as_tensor(upper_thresholds_np, dtype=torch.float32)
        self.register_buffer("lower_thresholds", lower_thresholds, persistent=False)
        self.register_buffer("upper_thresholds", upper_thresholds, persistent=False)

    def forward(self, prediction, input):
        '''
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        '''
        lower = self.lower_thresholds.to(dtype=prediction.dtype)
        upper = self.upper_thresholds.to(dtype=prediction.dtype)
        if self.keep_grad_through_clamp:
            return clamp_keep_interior_grad(prediction, lower, upper)
        clamped = torch.maximum(prediction, lower)
        return torch.minimum(clamped, upper)

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