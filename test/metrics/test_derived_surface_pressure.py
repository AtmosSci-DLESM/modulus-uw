import numpy as np
import pytest
import torch

from physicsnemo.metrics.climate.derived_surface_pressure import (
    DerivedSurfacePressure,
    surface_pressure_from_levels,
)

R, G = 287.04, 9.80665
LEVELS = torch.tensor([1000.0, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]) * 100.0


def _isothermal(ps, phi_s, T=260.0):
    # Phi(p) = Phi_s + R T ln(ps / p)
    z = phi_s + R * T * torch.log(ps / LEVELS.view(-1, 1, 1))
    return z, torch.full_like(z, T), torch.zeros_like(z)


def _const_lapse(ps, phi_s, T0=288.15, gamma=0.0065):
    # T(z) = T0 - gamma h with h = z / g measured from sea level; p/ps = (T/T_s)^(g/(R gamma))
    ts = T0 - gamma * phi_s / G
    T = ts * (LEVELS.view(-1, 1, 1) / ps) ** (R * gamma / G)
    z = phi_s + (ts - T) / gamma * G
    return z, T, torch.zeros_like(z)


def test_isothermal_recovers_ps_ocean_and_land():
    ps = torch.tensor([[101325.0, 100200.0, 95000.0, 70000.0]]).view(1, 2, 2)
    phi_s = torch.tensor([[0.0, 0.0, 600.0, 30000.0]]).view(1, 2, 2)
    z, t, q = _isothermal(ps, phi_s)
    out = surface_pressure_from_levels(z, t, q, phi_s, LEVELS)
    assert torch.allclose(out, ps, rtol=0, atol=2.0)  # open ocean: lapse-rate extrapolation below 1000 hPa


def test_constant_lapse_rate_atmosphere():
    ps = torch.tensor([[101325.0, 99000.0, 92000.0, 75000.0]]).view(1, 2, 2)
    phi_s = torch.tensor([[0.0, 0.0, 800.0, 20000.0]]).view(1, 2, 2)
    z, t, q = _const_lapse(ps, phi_s)
    out = surface_pressure_from_levels(z, t, q, phi_s, LEVELS)
    # ocean branch is exact for a constant lapse rate; the between-levels branch is a linear-in-ln p approximation
    assert torch.allclose(out, ps, rtol=0, atol=1.0)


def test_column_water_constant_q():
    ps = torch.full((1, 1, 1), 101325.0)
    phi_s = torch.zeros(1, 1, 1)
    z, t, q = _isothermal(ps, phi_s)
    q = torch.full_like(q, 0.004)
    _, w = surface_pressure_from_levels(z, t, q, phi_s, LEVELS, return_column_water=True)
    # q constant over [50 hPa, ps]: W = q (ps - 50 hPa) / g
    assert torch.allclose(w, torch.tensor([[[0.004 * (101325.0 - 5000.0) / G]]]), rtol=1e-4)


def test_module_layout_gradients_and_dry_pressure():
    names = [f"{v}{int(p / 100)}" for v in "ztq" for p in LEVELS.tolist()]
    scaling = {n: {"mean": 0.0, "std": 1.0} for n in names}
    mod = DerivedSurfacePressure([p / 100 for p in LEVELS.tolist()], names, scaling)
    ps = torch.full((1, 2, 2), 101325.0)
    z, t, q = _isothermal(ps, torch.zeros(1, 2, 2))
    q = q + 0.003
    x = torch.cat([z, t, q], dim=0).permute(0, 1, 2).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,3L,2,2]
    x = x.reshape(1, 1, 1, -1, 2, 2).requires_grad_(True)
    sp, w = mod.sp_tcwv(x)
    assert sp.shape == (1, 1, 1, 1, 2, 2)
    assert torch.allclose(sp, torch.full_like(sp, 101325.0), atol=5.0)
    dry = mod.sp_dry(x)
    assert torch.allclose(dry, sp - G * w)
    sp.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
