"""Input-skip truncate: replace the carried state with a downsample/upsample round trip.

``y := DownUp(x) + (y − x)`` on selected prognostics. High-frequency content of
the skip is whatever the configured resample removes; the residual for this
step is left unchanged. Diagnostics have no skip and are not modified.

Both blocks are Hydra configs. They must not contain trainable parameters, so
the upsample is the resample itself and not the 3×3 that follows it in the UNet.
"""

from __future__ import annotations

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _cfg_int(cfg, key: str, default: int) -> int:
    if OmegaConf.is_config(cfg):
        value = OmegaConf.select(cfg, key, default=default)
    elif isinstance(cfg, dict):
        value = cfg.get(key, default)
    else:
        value = default
    return int(value)


class InputSkipTruncateConstraint(torch.nn.Module):
    """Downsample then upsample the residual-add skip.

    RecUNet has already run on the unfiltered state. This rewrites selected
    channels as ``y := DownUp(x) + (y − x)``.
    """

    def __init__(
        self,
        down_sampling_block,
        up_sampling_block,
        in_channels: list[str],
        out_channels: list[str],
        nside: int = 64,
        hpx_padding_mode: str | None = "isolatitude",
        compile_padding: bool = False,
        enable_nhwc: bool = False,
        variables: list[str] | None = None,
    ):
        super().__init__()
        self.in_names = list(in_channels)
        self.out_names = list(out_channels)
        self.nside = int(nside)
        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(f"nside must be a positive power of 2, got {nside}")

        selected = list(self.in_names if variables is None else variables)
        if not selected:
            raise ValueError("truncate constraint needs at least one prognostic variable")
        pred_idx: list[int] = []
        orig_idx: list[int] = []
        for name in selected:
            if name not in self.out_names:
                raise ValueError(f"truncate variable {name!r} is not in out_channels")
            if name not in self.in_names:
                raise ValueError(
                    f"truncate variable {name!r} is not in in_channels; "
                    "the skip round trip needs a prognostic"
                )
            pred_idx.append(self.out_names.index(name))
            orig_idx.append(self.in_names.index(name))
        self._pred_idx = tuple(pred_idx)
        self._orig_idx = tuple(orig_idx)
        # Buffers, not forward allocations: CUDA-graph capture cannot allocate
        # the index tensor while the train graph is being recorded.
        self.register_buffer(
            "_pred_index", torch.tensor(pred_idx, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_orig_index", torch.tensor(orig_idx, dtype=torch.long), persistent=False
        )

        if isinstance(down_sampling_block, torch.nn.Module) or isinstance(
            up_sampling_block, torch.nn.Module
        ):
            raise TypeError(
                "down_sampling_block and up_sampling_block must be configs. "
                "Set _recursive_: false on the truncate constraint so nside and "
                "channel count are injected before the blocks are built."
            )
        scale = _cfg_int(up_sampling_block, "scale_factor", 2)
        stride = _cfg_int(down_sampling_block, "stride", 2)
        if scale != stride:
            raise ValueError(
                f"upsample scale_factor {scale} must match downsample stride {stride}"
            )
        if scale < 2 or self.nside % scale != 0:
            raise ValueError(f"cannot resample nside {self.nside} by {scale}")
        n_sel = len(self._pred_idx)
        # A single depthwise channel (groups=1) makes Inductor's scheduler raise
        # KeyError while compiling the train backward. Duplicate the channel for
        # the resample only; both blocks are channel-wise, so the kept channel
        # matches a true one-channel round trip.
        self._pad_channel = n_sel == 1
        n_mod = 2 if self._pad_channel else n_sel
        shared = dict(
            in_channels=n_mod,
            enable_nhwc=bool(enable_nhwc),
            hpx_padding_mode=hpx_padding_mode,
            compile_padding=bool(compile_padding),
        )
        self.down = instantiate(down_sampling_block, nside=self.nside, **shared)
        self.up = instantiate(
            up_sampling_block,
            nside=self.nside // scale,
            out_channels=n_mod,
            **shared,
        )
        learned = [name for name, param in self.named_parameters() if param.requires_grad]
        if learned:
            raise ValueError(
                "input-skip truncate blocks must have no trainable parameters; "
                f"found {learned}"
            )

    def _select(self, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(3, index)

    def _replace(
        self, tensor: torch.Tensor, index: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        # index_copy on dim 3. An empty leading slice (PRESsfc is channel 0)
        # is what a cat-based replace would build, and Inductor drops that op.
        return torch.index_copy(tensor, 3, index, values)

    def _roundtrip(self, state: torch.Tensor) -> torch.Tensor:
        """``[B, F, T, C, H, W]`` through the configured down and up blocks."""
        if self._pad_channel:
            state = state.repeat(1, 1, 1, 2, 1, 1)
        batch, faces, time, channels, height, width = state.shape
        folded = state.permute(0, 2, 1, 3, 4, 5).reshape(batch * time * faces, channels, height, width)
        restored = self.up(self.down(folded))
        if self._pad_channel:
            restored = restored[:, :1]
            channels = 1
        if restored.shape[-2:] != (height, width) or restored.shape[1] != channels:
            raise RuntimeError(
                f"round trip returned {tuple(restored.shape)}, "
                f"expected {(batch * time * faces, channels, height, width)}"
            )
        return (
            restored.view(batch, time, faces, channels, height, width)
            .permute(0, 2, 1, 3, 4, 5)
            .contiguous()
        )

    def forward(self, prediction: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        prediction, input:
            ``[B, F, T, C, H, W]``. ``input`` is the prognostic skip. If time
            lengths differ, the last input time is used for every output step.
        """
        orig_dtype = prediction.dtype
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            orig = input.float()
            if orig.shape[2] != prediction.shape[2]:
                orig = orig[:, :, -1:]
            orig_sel = self._select(orig, self._orig_index)
            pred_sel = self._select(prediction, self._pred_index)
            filtered = self._roundtrip(orig_sel)
            updated = filtered + (pred_sel - orig_sel)
            out = self._replace(prediction, self._pred_index, updated)
        return out.to(dtype=orig_dtype)
