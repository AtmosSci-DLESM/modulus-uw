"""Tests for the latitude-cap axial angular-momentum soft constraint (budget terms checked against independent per-pixel loops)."""
from dataclasses import dataclass
from typing import Sequence

import earth2grid
import numpy as np
import pytest
import torch
import xarray as xr
from earth2grid.healpix import HEALPIX_PAD_XY

from physicsnemo.metrics.climate.healpix_soft_constraints import (
    AxialAngularMomentumSoftConstraint,
    CapAxialAngularMomentumSoftConstraint,
    _EARTH_OMEGA,
    _EARTH_RADIUS_M,
)

NSIDE = 4
LEVELS = [300.0, 500.0, 700.0, 850.0]
A, G, OM, DT = _EARTH_RADIUS_M, 9.81, _EARTH_OMEGA, 6 * 3600.0
D_OMEGA = 4 * np.pi / (12 * NSIDE ** 2)
SP0 = 100000.0                                   # 1000 hPa: every layer is above ground
COLUMN = (850.0 - 300.0) * 100.0                 # sum of layer thicknesses [Pa]
PROG = [f"{v}{int(p)}" for v in "uv" for p in LEVELS] + ["sp"]
OUT = [f"{v}{int(p)}" for v in "uv" for p in LEVELS] + ["sp", "avg_iews-6h", "avg_iegwss-6h"]


@dataclass
class _Trainer:
    device: torch.device
    output_variables: Sequence[str]


def _scaling():
    sc = {n: {"mean": 0.0, "std": 10.0} for n in OUT}
    sc["sp"] = {"mean": 1.0e5, "std": 1.0e3}
    sc["avg_iews-6h"] = {"mean": 0.0, "std": 0.1}
    sc["avg_iegwss-6h"] = {"mean": 0.0, "std": 0.05}
    return sc


def _geometry():
    lat = np.deg2rad(np.asarray(earth2grid.healpix.Grid(level=int(np.log2(NSIDE)), pixel_order=HEALPIX_PAD_XY).lat, dtype=float))
    ring_lat = np.unique(np.round(lat, 8))
    ring = np.searchsorted(ring_lat, np.round(lat, 8))
    return lat, ring, ring_lat, 0.5 * (ring_lat[1:] + ring_lat[:-1])


LAT, RING, RING_LAT, BND_LAT = _geometry()
NR, NB = len(RING_LAT), len(BND_LAT)


def _make(tmp_path, alpha=1.0, prog=None, **kw):
    ds = xr.Dataset({"constants": (("face", "channel_c", "height", "width"), np.zeros((12, 1, NSIDE, NSIDE), np.float32))}, coords={"channel_c": ["z"]})
    path = tmp_path / "topo.zarr"
    if not path.exists():
        ds.to_zarr(path)
    prog = PROG if prog is None else prog
    mod = CapAxialAngularMomentumSoftConstraint(hPa_levels=LEVELS, channels=OUT, scaling=_scaling(), dataset_path=str(path),
                                                surface_geopotential_name="z", input_channels=prog, alpha=alpha, **kw)
    mod.setup(_Trainer(torch.device("cpu"), OUT))
    return mod


def _fields(u=0.0, v=0.0, sp=SP0, tau=0.0):
    """Physical per-pixel fields -> normalized [1,12,1,C,H,W] tensors. u, v: [L] or scalar x pixel; sp, tau scalar or [12,H,W]."""
    sc = _scaling()
    x = torch.zeros(1, 12, 1, len(OUT), NSIDE, NSIDE)
    for i, p in enumerate(LEVELS):
        for name, val in ((f"u{int(p)}", u), (f"v{int(p)}", v)):
            val = torch.as_tensor(val, dtype=torch.float32)
            x[0, :, 0, OUT.index(name)] = ((val if val.ndim == 0 else val[i] if val.ndim == 4 else val) - sc[name]["mean"]) / sc[name]["std"]
    x[0, :, 0, OUT.index("sp")] = (torch.as_tensor(sp, dtype=torch.float32) - 1.0e5) / 1.0e3
    x[0, :, 0, OUT.index("avg_iews-6h")] = torch.as_tensor(tau, dtype=torch.float32) / 0.1
    return x


def _ring_field(values):
    """Value per ring [NR] -> [12,H,W] pixel field."""
    return torch.as_tensor(np.asarray(values, dtype=np.float32)[RING].reshape(12, NSIDE, NSIDE))


def _ic_from(state):
    return state[:, :, :, [OUT.index(n) for n in PROG]]


def _resid(mod, ic, preds, diag=None):
    pred = torch.cat(preds, dim=2)
    return mod._boundary_residuals(pred, _ic_from(ic) if diag is None else ic, diag)


def _cap_members(b):
    """Pixel mask of the cap of boundary b (northern: rings > b for boundaries north of the equator, else rings <= b)."""
    return (RING > b) if BND_LAT[b] > 0 else (RING <= b)


def test_steady_zonal_flow_has_zero_residual(tmp_path):
    mod = _make(tmp_path)
    jet = _ring_field(30.0 * np.cos(RING_LAT) ** 2)
    state = _fields(u=jet, v=0.0)
    r = _resid(mod, state, [state, state])
    assert r.shape == (1, 2, NB)
    assert torch.count_nonzero(r) == 0


def test_poleward_mass_transfer_matches_independent_loop(tmp_path):
    mod = _make(tmp_path)
    rng = np.random.default_rng(0)
    dsp = torch.as_tensor(rng.normal(0.0, 50.0, (12, NSIDE, NSIDE)).astype(np.float32))
    ic, pred = _fields(), _fields(sp=SP0 + dsp)
    r = _resid(mod, ic, [pred])[0, 0].numpy()
    ds = (pred[0, :, 0, OUT.index("sp")] * 1.0e3 + 1.0e5 - SP0).numpy().astype(np.float64).reshape(-1)
    cos2_pix = np.cos(LAT.reshape(-1)) ** 2
    for b in range(NB):
        m = _cap_members(b).reshape(-1)
        ref = OM * A ** 4 / G * D_OMEGA / DT * np.sum((cos2_pix[m] - np.cos(BND_LAT[b]) ** 2) * ds[m])
        assert r[b] == pytest.approx(ref, rel=1e-6, abs=1e-3)


def test_uniform_wind_change_in_one_ring_hits_only_enclosing_caps(tmp_path):
    mod = _make(tmp_path)
    r0, du = 6, 5.0
    pred = _fields(u=_ring_field(np.where(np.arange(NR) == r0, du, 0.0)))
    r = _resid(mod, _fields(), [pred])[0, 0].numpy()
    n_pix = int((RING == r0).sum())
    expected = A ** 3 / G * D_OMEGA * np.cos(RING_LAT[r0]) * n_pix * du * COLUMN / DT
    for b in range(NB):
        inside = bool(_cap_members(b)[RING == r0].all())
        assert r[b] == pytest.approx(expected if inside else 0.0, rel=1e-9, abs=1e-3)


def test_boundary_flux_only_changes_adjacent_boundaries(tmp_path):
    mod = _make(tmp_path)
    b0, u0, v0 = 5, 8.0, 3.0
    on = np.isin(np.arange(NR), [b0, b0 + 1])
    state = _fields(u=_ring_field(np.where(on, u0, 0.0)), v=_ring_field(np.where(on, v0, 0.0)))
    r = _resid(mod, state, [state])[0, 0].numpy()
    q = np.where(on, u0 * v0 * COLUMN, 0.0)                      # zonal-mean column u v of every ring
    for b in range(NB):
        q_hat = 0.5 * (q[b] + q[b + 1])
        s = 1.0 if BND_LAT[b] > 0 else -1.0
        assert r[b] == pytest.approx(-s * 2 * np.pi * A ** 2 * np.cos(BND_LAT[b]) ** 2 / G * q_hat, rel=1e-9, abs=1e-3)


def test_surface_stress_torque_on_northern_caps(tmp_path):
    mod = _make(tmp_path)
    tau0 = 0.02
    state = _fields()
    pred = _fields(tau=tau0)
    r = _resid(mod, state, [pred])[0, 0].numpy()
    cos_pix = np.cos(LAT.reshape(-1))
    for b in range(NB):
        m = _cap_members(b).reshape(-1)
        assert r[b] == pytest.approx(A ** 3 * D_OMEGA * tau0 * cos_pix[m].sum(), rel=1e-6)


def test_hemispheric_mirror_symmetry(tmp_path):
    mod = _make(tmp_path)
    rng = np.random.default_rng(3)
    sym = lambda: (lambda x: 0.5 * (x + x[::-1]))(rng.normal(size=NR))
    anti = lambda: (lambda x: 0.5 * (x - x[::-1]))(rng.normal(size=NR))
    ic = _fields(u=_ring_field(10 * sym()), v=_ring_field(5 * anti()), sp=SP0 + _ring_field(40 * sym()))
    pred = _fields(u=_ring_field(10 * sym()), v=_ring_field(5 * anti()), sp=SP0 + _ring_field(40 * sym()))
    r = _resid(mod, ic, [pred])[0, 0].numpy()
    np.testing.assert_allclose(r, r[::-1], rtol=1e-9, atol=1e-6 * np.abs(r).max())


def test_sum_over_rings_is_the_global_budget_of_the_parent_class(tmp_path):
    mod = _make(tmp_path)
    rng = np.random.default_rng(5)
    ic = _fields(u=_ring_field(rng.normal(0, 10, NR)), v=_ring_field(rng.normal(0, 5, NR)), sp=SP0 + _ring_field(rng.normal(0, 30, NR)))
    pred = _fields(u=_ring_field(rng.normal(0, 10, NR)), v=_ring_field(rng.normal(0, 5, NR)), sp=SP0 + _ring_field(rng.normal(0, 30, NR)), tau=0.01)
    ic_in = _ic_from(ic)
    per_ring_a, _, _ = mod._ring_budget(pred, ic_in, None)
    ref = AxialAngularMomentumSoftConstraint._budget_residuals(mod, pred.float(), ic_in.float(), None)
    # The parent evaluates M(t+dt) - M(t) from ~1e28 totals in float32; this class differences per ring in float64.
    np.testing.assert_allclose(per_ring_a.sum(-1).numpy(), ref.numpy(), rtol=2e-3)


def test_loss_rmse_gradient_and_spec(tmp_path):
    mod = _make(tmp_path, alpha=np.linspace(1e17, 1e19, NB))
    rng = np.random.default_rng(7)
    ic = _fields(u=_ring_field(rng.normal(0, 10, NR)))
    pred = _fields(u=_ring_field(rng.normal(0, 10, NR)), sp=SP0 + _ring_field(rng.normal(0, 30, NR))).requires_grad_(True)
    loss = mod.constraint_loss(pred, None, input=_ic_from(ic))
    assert loss.ndim == 0 and torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().max() > 0
    rmse = mod.physical_rmse(pred.detach(), None, input=_ic_from(ic))
    per = mod.physical_rmse(pred.detach(), None, input=_ic_from(ic), average_channels=False)
    assert rmse.ndim == 0 and torch.isfinite(rmse) and rmse > 1e15
    assert per.shape == (len(mod.report_names),) and mod.report_names == ["S65", "S50", "S35", "S20", "N20", "N35", "N50", "N65"]
    assert mod.constraint_spec()[0] == "cap_aam"
    with pytest.raises(ValueError):
        _make(tmp_path, alpha=[1.0, 2.0])


def test_normalize_by_boundary_matches_geometric_per_boundary_alphas(tmp_path):
    alpha = 2.0e4  # N/m
    a = 6.371e6
    norm = _make(tmp_path, alpha=alpha, normalize_by_boundary=True)
    ref = _make(tmp_path, alpha=alpha * 2.0 * np.pi * a ** 2 * np.cos(BND_LAT) ** 2)
    rng = np.random.default_rng(11)
    ic = _fields(u=_ring_field(rng.normal(0, 10, NR)), v=_ring_field(rng.normal(0, 5, NR)))
    pred = _fields(u=_ring_field(rng.normal(0, 10, NR)), v=_ring_field(rng.normal(0, 5, NR)), sp=SP0 + _ring_field(rng.normal(0, 30, NR)), tau=0.01)
    ic_in = _ic_from(ic)
    torch.testing.assert_close(norm.constraint_loss(pred, None, input=ic_in), ref.constraint_loss(pred, None, input=ic_in))
    torch.testing.assert_close(norm.physical_rmse(pred, None, input=ic_in), ref.physical_rmse(pred, None, input=ic_in))
    with pytest.raises(ValueError):
        _make(tmp_path, alpha=[1.0] * NB, normalize_by_boundary=True)


def test_initial_condition_from_input_diagnostics_matches_prognostic_input(tmp_path):
    prog_small = [n for n in PROG if not n.startswith("v") and n != "sp"]
    mod_diag = _make(tmp_path, prog=prog_small)
    mod_full = _make(tmp_path)
    assert mod_diag.needs_input_diagnostics and not mod_full.needs_input_diagnostics
    rng = np.random.default_rng(9)
    ic = _fields(u=_ring_field(rng.normal(0, 10, NR)), v=_ring_field(rng.normal(0, 5, NR)), sp=SP0 + _ring_field(rng.normal(0, 30, NR)))
    pred = _fields(u=_ring_field(rng.normal(0, 10, NR)), v=_ring_field(rng.normal(0, 5, NR)))
    diag_names = mod_diag.diagnostic_channels
    ref = mod_full._boundary_residuals(pred, _ic_from(ic), None)
    got = mod_diag._boundary_residuals(pred, ic[:, :, :, [OUT.index(n) for n in prog_small]], ic[:, :, :, [OUT.index(n) for n in diag_names]])
    torch.testing.assert_close(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (SHT, graph capture)")
def test_cuda_graph_capture_matches_eager(tmp_path):
    dev = torch.device("cuda")
    ds = xr.Dataset({"constants": (("face", "channel_c", "height", "width"), (np.random.default_rng(1).random((12, 1, NSIDE, NSIDE)) * 3e3).astype(np.float32))}, coords={"channel_c": ["z"]})
    ds.to_zarr(tmp_path / "topo_cuda.zarr")
    mod = CapAxialAngularMomentumSoftConstraint(hPa_levels=LEVELS, channels=OUT, scaling=_scaling(), dataset_path=str(tmp_path / "topo_cuda.zarr"),
                                                surface_geopotential_name="z", input_channels=PROG, alpha=1e18)
    mod.setup(_Trainer(dev, OUT))
    assert float(mod.dh_dlambda.abs().max()) > 0
    rng = np.random.default_rng(2)
    ic = _fields(u=_ring_field(rng.normal(0, 10, NR)), sp=SP0 + _ring_field(rng.normal(0, 30, NR))).to(dev)
    static_pred = torch.cat([_fields(u=_ring_field(rng.normal(0, 10, NR)), sp=SP0 + _ring_field(rng.normal(0, 30, NR))) for _ in range(2)], dim=2).to(dev).requires_grad_(True)
    inp = _ic_from(ic)
    eager_pred = static_pred.detach().clone().requires_grad_(True)  # fresh leaf: autograd binds a leaf's AccumulateGrad to the stream it was first used on
    eager = mod.constraint_loss(eager_pred, None, input=inp)
    eager.backward()
    g_eager = eager_pred.grad.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            mod.constraint_loss(static_pred, None, input=inp).backward()
    torch.cuda.current_stream().wait_stream(s)
    static_pred.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mod.constraint_loss(static_pred, None, input=inp)
        out.backward()
    graph.replay()
    torch.testing.assert_close(out, eager)
    torch.testing.assert_close(static_pred.grad, g_eager, rtol=1e-4, atol=1e-8)
