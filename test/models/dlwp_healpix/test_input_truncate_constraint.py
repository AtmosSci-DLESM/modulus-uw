"""Input-skip truncate keeps constants, drops the learned upsample, and skips diagnostics."""

import pytest
import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf

from physicsnemo.models.dlwp_healpix_layers.healpix_input_truncate_constraint import (
    InputSkipTruncateConstraint,
)

NSIDE = 8


def _constraint(**kwargs):
    down = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.DealiasedDownsample",
        "resample_filter": [1.0, 2.0, 1.0],
        "stride": 2,
        "reflection_equivariant": True,
    }
    up = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_ring_mean_pchip.RingMeanPCHIPUpsampleFaces",
        "scale_factor": 2,
    }
    defaults = dict(
        down_sampling_block=down,
        up_sampling_block=up,
        in_channels=["a", "b"],
        out_channels=["a", "b", "diag"],
        nside=NSIDE,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        enable_nhwc=False,
    )
    defaults.update(kwargs)
    return InputSkipTruncateConstraint(**defaults)


def test_skip_roundtrip_on_prognostics_only():
    mod = _constraint()
    assert not any(param.requires_grad for param in mod.parameters())

    constant = torch.full((1, 12, 1, 2, NSIDE, NSIDE), 2.5)
    pred = constant.repeat(1, 1, 2, 1, 1, 1)
    pred = torch.cat([pred, torch.full((1, 12, 2, 1, NSIDE, NSIDE), 7.0)], dim=3)
    out = mod(pred, constant)
    assert torch.allclose(out[:, :, :, :2], pred[:, :, :, :2])
    assert torch.equal(out[:, :, :, 2], pred[:, :, :, 2])

    torch.manual_seed(0)
    state = torch.randn(1, 12, 1, 2, NSIDE, NSIDE)
    pred = state.repeat(1, 1, 2, 1, 1, 1)
    diag = torch.randn(1, 12, 2, 1, NSIDE, NSIDE)
    pred = torch.cat([pred, diag], dim=3)
    out = mod(pred, state)
    roundtrip = mod._roundtrip(state)
    assert torch.allclose(out[:, :, :, :2], roundtrip.expand_as(out[:, :, :, :2]), atol=1e-5)
    assert torch.equal(out[:, :, :, 2:3], diag)

    residual = torch.randn_like(state)
    pred_res = torch.cat([state + residual, diag[:, :, :1]], dim=3)
    out_res = mod(pred_res, state)
    assert torch.allclose(out_res[:, :, :, :2], roundtrip + residual, atol=1e-5)


def test_hydra_leaves_resample_blocks_unbuilt_until_nside_is_known():
    cfg = OmegaConf.create(
        {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_input_truncate_constraint.InputSkipTruncateConstraint",
            "_recursive_": False,
            "down_sampling_block": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.DealiasedDownsample",
                "resample_filter": [1.0, 2.0, 1.0],
                "stride": 2,
                "reflection_equivariant": True,
            },
            "up_sampling_block": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_ring_mean_pchip.RingMeanPCHIPUpsampleFaces",
                "scale_factor": 2,
            },
            "in_channels": ["a"],
            "out_channels": ["a"],
            "nside": NSIDE,
            "hpx_padding_mode": "isolatitude",
            "compile_padding": False,
        }
    )
    mod = instantiate(cfg)
    assert mod.nside == NSIDE
    built = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    built._recursive_ = True
    with pytest.raises(InstantiationException, match="_recursive_"):
        instantiate(built)
