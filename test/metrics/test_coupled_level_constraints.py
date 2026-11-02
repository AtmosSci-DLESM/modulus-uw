"""A pressure level fed by the coupler gives the same constraint values as the same level predicted by the model."""
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest
import torch
import xarray as xr

from physicsnemo.metrics.climate.healpix_loss import WeightedMSE
from physicsnemo.metrics.climate.healpix_soft_constraints import (
    AxialAngularMomentumSoftConstraint,
    CapAxialAngularMomentumSoftConstraint,
    HydrostasySoftConstraint,
    LossWithSoftConstraints,
)

NSIDE, B, T, PRESTEPS = 4, 2, 3, 1
LEVELS = [300.0, 500.0, 700.0, 850.0]
TOP = 300  # the level moved from the model channels to the coupler


@dataclass
class _Trainer:
    device: torch.device
    output_variables: Sequence[str]


def _topography(tmp_path):
    path = tmp_path / "topo.zarr"
    if not path.exists():
        zeros = np.zeros((12, 1, NSIDE, NSIDE), np.float32)
        xr.Dataset(
            {"constants": (("face", "channel_c", "height", "width"), zeros)},
            coords={"channel_c": ["z"]},
        ).to_zarr(path)
    return str(path)


def _split(full, coupled, x, dim=3):
    """Model-only channels of ``x`` and the coupled ones, each in list order."""
    keep = [i for i, n in enumerate(full) if n not in coupled]
    cpl = [full.index(n) for n in coupled]
    return x.index_select(dim, torch.tensor(keep)), x.index_select(dim, torch.tensor(cpl))


def _coupled_slots(pred_c):
    """``[T_slots, B, C, F, H, W]``: a junk prestep slot, then the field each output step reads."""
    steps = pred_c.permute(2, 0, 3, 1, 4, 5)  # [B,F,T,C,H,W] -> [T,B,C,F,H,W]
    junk = torch.full_like(steps[:PRESTEPS], 1.0e3)
    return torch.cat([junk, steps], dim=0)


def _hydro(tmp_path, channels, coupled=()):
    names = [f"{v}{int(p)}" for v in "ztq" for p in LEVELS]
    scaling = {n: {"mean": 0.0, "std": 1.0} for n in names}
    scaling.update({f"z{int(p)}": {"mean": 3.0e4 - 30.0 * p, "std": 500.0} for p in LEVELS})
    scaling.update({f"t{int(p)}": {"mean": 250.0, "std": 10.0} for p in LEVELS})
    scaling.update({f"q{int(p)}": {"mean": 1.0e-3, "std": 5.0e-4} for p in LEVELS})
    mod = HydrostasySoftConstraint(
        hPa_levels=LEVELS, channels=channels, weights=[1.0] * 3, alpha=[0.2, 0.3, 0.4], scaling=scaling,
        dataset_path=_topography(tmp_path), surface_geopotential_name="z", surface_geopotential_mean=0.0,
        surface_geopotential_std=1.0, topography_masking=False, coupled_channels=coupled,
    )
    mod.setup(_Trainer(torch.device("cpu"), channels))
    return mod


def test_hydrostasy_coupled_level_matches_predicted_level(tmp_path):
    full = [f"{v}{int(p)}" for v in "ztq" for p in LEVELS] + ["sp"]
    coupled = [f"z{TOP}", f"t{TOP}", f"q{TOP}"]
    model = [n for n in full if n not in coupled]
    torch.manual_seed(0)
    pred = torch.randn(B, 12, T, len(full), NSIDE, NSIDE)
    pred_m, pred_c = _split(full, coupled, pred)
    slots = _coupled_slots(pred_c)

    ref = _hydro(tmp_path, full)
    mod = _hydro(tmp_path, model, coupled)
    assert not ref.needs_coupled_input and mod.needs_coupled_input
    # Only the z/t/q levels are copied (not sp), with the coupled level last.
    assert mod.with_coupled(pred_m, slots).shape[3] == 3 * len(LEVELS)
    torch.testing.assert_close(
        mod.constraint_loss(pred_m, None, average_channels=False, coupled_input=slots),
        ref.constraint_loss(pred, None, average_channels=False),
    )
    torch.testing.assert_close(mod.physical_rmse(pred_m, None, coupled_input=slots), ref.physical_rmse(pred, None))

    # The trainer-facing wrapper forwards coupled_input only to constraints that read it.
    wrapper = LossWithSoftConstraints(data_loss=WeightedMSE(weights=[1.0] * len(model)), constraints=[mod])
    wrapper.setup(_Trainer(torch.device("cpu"), model))
    assert wrapper.needs_coupled_input
    out = wrapper(pred_m, pred_m, average_channels=False, coupled_input=slots)
    torch.testing.assert_close(out[-3:], ref.constraint_loss(pred, None, average_channels=False))
    with pytest.raises(ValueError, match="coupled_input"):
        wrapper(pred_m, pred_m)
    with pytest.raises(ValueError, match="channels"):
        mod.physical_rmse(pred_m[:, :, :, 1:], None, coupled_input=slots)


def _aam_scaling(names):
    sc = {n: {"mean": 0.0, "std": 10.0} for n in names}
    sc["sp"] = {"mean": 1.0e5, "std": 1.0e3}
    sc["avg_iews-6h"] = {"mean": 0.0, "std": 0.1}
    sc["avg_iegwss-6h"] = {"mean": 0.0, "std": 0.05}
    return sc


@pytest.mark.parametrize("cls", [AxialAngularMomentumSoftConstraint, CapAxialAngularMomentumSoftConstraint])
def test_aam_coupled_level_matches_predicted_level(tmp_path, cls):
    winds = [f"{v}{int(p)}" for v in "uv" for p in LEVELS]
    full_out = winds + ["sp", "avg_iews-6h", "avg_iegwss-6h"]
    full_in = winds + ["sp"]
    coupled = [f"u{TOP}", f"v{TOP}"]
    model_out = [n for n in full_out if n not in coupled]
    model_in = [n for n in full_in if n not in coupled]

    def make(out, inp, cpl=()):
        mod = cls(
            hPa_levels=LEVELS, channels=out, input_channels=inp, scaling=_aam_scaling(full_out),
            dataset_path=_topography(tmp_path), surface_geopotential_name="z", alpha=1.0e18, coupled_channels=cpl,
        )
        mod.setup(_Trainer(torch.device("cpu"), out))
        return mod

    torch.manual_seed(1)
    pred = torch.randn(B, 12, T, len(full_out), NSIDE, NSIDE)
    inp = torch.randn(B, 12, 1, len(full_in), NSIDE, NSIDE)
    pred_m, pred_c = _split(full_out, coupled, pred)
    # The coupled initial condition is the first output step's slot.
    for j, name in enumerate(coupled):
        inp[:, :, :, full_in.index(name)] = pred_c[:, :, :1, j]
    inp_m, _ = _split(full_in, coupled, inp)
    slots = _coupled_slots(pred_c)

    ref, mod = make(full_out, full_in), make(model_out, model_in, coupled)
    assert not ref.needs_coupled_input and mod.needs_coupled_input
    torch.testing.assert_close(
        mod.constraint_loss(pred_m, None, input=inp_m, coupled_input=slots),
        ref.constraint_loss(pred, None, input=inp),
    )
    torch.testing.assert_close(
        mod.physical_rmse(pred_m, None, input=inp_m, coupled_input=slots),
        ref.physical_rmse(pred, None, input=inp),
    )
