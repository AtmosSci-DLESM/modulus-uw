"""Dry-air and AAM soft constraints with the surface pressure derived from z/t/q (no sp/tcwv channels)."""
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import xarray as xr

from physicsnemo.metrics.climate.healpix_soft_constraints import (
    AxialAngularMomentumSoftConstraint,
    DryAirMassSoftConstraint,
)

R = 287.04
LEVELS = [500.0, 700.0, 850.0, 1000.0]
NSIDE = 2


@dataclass
class _Trainer:
    device: torch.device
    output_variables: Sequence[str]


def _channels():
    prog = [f"{v}{int(p)}" for v in "ztqu" for p in LEVELS]
    return prog, prog + ["msl", "avg_iews-6h", "avg_iegwss-6h"]


def _scaling(names):
    sc = {n: {"mean": 0.0, "std": 1.0} for n in names}
    for p in LEVELS:
        sc[f"u{int(p)}"] = {"mean": 0.0, "std": 10.0}
    sc["avg_iews-6h"] = {"mean": 0.0, "std": 0.1}
    sc["avg_iegwss-6h"] = {"mean": 0.0, "std": 0.05}
    return sc


def _topo(tmp_path):
    ds = xr.Dataset({"constants": (("face", "channel_c", "height", "width"), np.zeros((12, 1, NSIDE, NSIDE), np.float32))},
                    coords={"channel_c": ["z"]})
    path = tmp_path / "topo.zarr"
    if not path.exists():
        ds.to_zarr(path)
    return str(path)


def _state(inp_names, out_names, ps=101325.0, dz=0.0, T=1):
    """Isothermal column with surface pressure ps (flat surface), z shifted by dz; [1,12,T,C,H,W]."""
    C = len(out_names)
    x = torch.zeros(1, 12, T, C, NSIDE, NSIDE)
    for p in LEVELS:
        x[:, :, :, out_names.index(f"z{int(p)}")] = R * 260.0 * np.log(ps / (p * 100.0)) + dz
        x[:, :, :, out_names.index(f"t{int(p)}")] = 260.0
        x[:, :, :, out_names.index(f"q{int(p)}")] = 0.002
    return x


def _derived_cfg(tmp_path):
    return dict(hPa_levels=LEVELS, dataset_path=_topo(tmp_path), surface_geopotential_name="z")


def test_dry_air_derived_zero_when_conserved_and_penalizes_z_shift(tmp_path):
    prog, out = _channels()
    mod = DryAirMassSoftConstraint(channels=out, scaling=_scaling(out), input_channels=prog, alpha=1.0, weight=1.0,
                                   derived_surface_pressure=_derived_cfg(tmp_path))
    mod.setup(_Trainer(torch.device("cpu"), out))
    assert not mod.needs_input_diagnostics
    inp = _state(prog, prog)
    pred = _state(prog, out, T=2)
    assert float(mod.constraint_loss(pred, pred, input=inp)) < 1e-6
    shifted = _state(prog, out, T=2, dz=300.0).requires_grad_(True)   # ~ +360 Pa of surface pressure
    loss = mod.constraint_loss(shifted, shifted, input=inp)
    assert float(loss.detach()) > 1.0
    loss.backward()
    assert torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum() > 0
    assert float(mod.physical_rmse(shifted, shifted, input=inp)) > 100.0


def test_aam_derived_runs_without_sp_channel(tmp_path):
    prog, out = _channels()
    mod = AxialAngularMomentumSoftConstraint(
        hPa_levels=LEVELS, channels=out, scaling=_scaling(out), dataset_path=_topo(tmp_path), surface_geopotential_name="z",
        surface_geopotential_mean=0.0, surface_geopotential_std=1.0, input_channels=prog, alpha=1.0e19,
        derived_surface_pressure=_derived_cfg(tmp_path),
    )
    mod.setup(_Trainer(torch.device("cpu"), out))
    assert not mod.needs_input_diagnostics
    inp = _state(prog, prog)
    pred = _state(prog, out, T=2).requires_grad_(True)
    loss = mod.constraint_loss(pred, pred, input=inp)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.isfinite(mod.physical_rmse(pred, pred, input=inp))


def test_mountain_torque_anomaly_form_removes_the_static_hydrostatic_part(tmp_path):
    prog, out = _channels()
    rng = np.random.default_rng(0)
    topo = (rng.random((12, 1, NSIDE, NSIDE)) * 2000.0).astype(np.float32)          # metres * g0 via convert_topography_to_meters
    ds = xr.Dataset({"constants": (("face", "channel_c", "height", "width"), topo * 9.81)}, coords={"channel_c": ["z"]})
    path = tmp_path / "mount.zarr"
    ds.to_zarr(path)

    def build(**kw):
        m = AxialAngularMomentumSoftConstraint(
            hPa_levels=LEVELS, channels=out, scaling=_scaling(out), dataset_path=str(path), surface_geopotential_name="z",
            surface_geopotential_mean=0.0, surface_geopotential_std=1.0, input_channels=prog, alpha=1.0e19,
            derived_surface_pressure=_derived_cfg(tmp_path), **kw)
        m.setup(_Trainer(torch.device("cpu"), out))
        m.dh_dlambda = torch.randn_like(m.dh_dlambda)                                # the SHT derivative only exists on CUDA
        return m

    plain, anomaly = build(), build(mountain_torque_anomaly=True)
    sp_hydro = anomaly.ps_ref.clone()                                                  # surface pressure that depends on h only
    # patch the derived pressure to return the h-only reference
    for m in (plain, anomaly):
        m._sp_from_prediction = lambda tensor, _s=sp_hydro: _s.expand(tensor.shape[0], 12, tensor.shape[2], 1, NSIDE, NSIDE).clone()
    pred = torch.zeros(1, 12, 2, len(out), NSIDE, NSIDE)
    assert float(anomaly._torques_from_prediction(pred).abs().max()) < 1.0e-3 * float(plain._torques_from_prediction(pred).abs().max())
    # the sign option flips the mountain part
    neg = build(mountain_torque_sign=-1.0)
    neg.dh_dlambda = plain.dh_dlambda.clone(); neg._sp_from_prediction = plain._sp_from_prediction
    assert torch.allclose(neg._torques_from_prediction(pred), -plain._torques_from_prediction(pred))
