# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Polar-cap corrections of the HEALPix down- and up-sampling layers."""

import math

import numpy as np
import pytest
import torch

earth2grid = pytest.importorskip("earth2grid")
from earth2grid.healpix import HEALPIX_PAD_XY

from physicsnemo.models.dlwp_healpix_layers.healpix_polar_cap import (
    POLE_CORRECTION_MODE,
    PolarCapDownCorrection,
    PolarCapUpCorrection,
    _colatitude,
    _ring,
    build_down_correction,
    build_up_correction,
    cap_depth,
)

_REFL_FACE_ORDER = [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
_ROT_FACE_ORDER = [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10]


def _grid(nside):
    g = earth2grid.healpix.Grid(level=int(math.log2(nside)), pixel_order=HEALPIX_PAD_XY)
    return (
        np.asarray(g.lat, dtype=np.float64).reshape(12, nside, nside),
        np.asarray(g.lon, dtype=np.float64).reshape(12, nside, nside),
    )


def _reflect(x):
    y = torch.rot90(torch.flip(x, dims=[-1]), dims=(-1, -2))
    return y.reshape(-1, 12, *y.shape[1:])[:, _REFL_FACE_ORDER].reshape(-1, *y.shape[1:])


def _rotate(x):
    return x.reshape(-1, 12, *x.shape[1:])[:, _ROT_FACE_ORDER].reshape(-1, *x.shape[1:])


def _fields(nside):
    """[12, 10, n, n] float64: scalars x, y, z; (u, v) of a uniform flow, rotation about x, rotation about z, grad(z)."""
    lat, lon = _grid(nside)
    ph, la = torch.tensor(np.deg2rad(lat)), torch.tensor(np.deg2rad(lon))
    X, Y, Z = torch.cos(ph) * torch.cos(la), torch.cos(ph) * torch.sin(la), torch.sin(ph)
    el = torch.stack([-torch.sin(la), torch.cos(la), torch.zeros_like(la)])
    ep = torch.stack([-torch.sin(ph) * torch.cos(la), -torch.sin(ph) * torch.sin(la), torch.cos(ph)])
    r = torch.stack([X, Y, Z])

    def uv(V):
        return (V * el).sum(0), (V * ep).sum(0)

    def cross(w):
        return torch.cross(torch.tensor(w, dtype=torch.float64).view(3, 1, 1, 1).expand_as(r), r, dim=0)

    ux, vx = uv(torch.stack([torch.ones_like(X), 0 * X, 0 * X]))
    ur, vr = uv(cross([1.0, 0, 0]))
    uz, vz = uv(cross([0, 0, 1.0]))
    return torch.stack([X, Y, Z, ux, vx, ur, vr, uz, vz, torch.cos(ph)], 1)


_SCALARS, _WINDS = [0, 1, 2], [3, 4, 5, 6, 7, 8]  # (u, v) pairs; channel 9 (cos lat) is a smooth scalar too


def _ring_rms(err, truth, nside, rings):
    """rms(err)/std(truth) per channel over the pixels of polar rings `rings` (both poles). err, truth: [12, C, n, n]."""
    idx = np.concatenate([_ring(nside, north, k)[0] for north in (True, False) for k in rings])
    e = err.permute(1, 0, 2, 3).reshape(err.shape[1], -1)[:, idx]
    std = truth.permute(1, 0, 2, 3).reshape(truth.shape[1], -1).std(-1).clamp_min(1e-9)
    return e.pow(2).mean(-1).sqrt() / std


@pytest.mark.parametrize("nside", [8, 16])
def test_ring_geometry_matches_earth2grid(nside):
    lat, lon = _grid(nside)
    for north in (True, False):
        for k in range(1, nside):
            idx, lam = _ring(nside, north, k)
            assert idx.size == 4 * k
            f, rem = idx // (nside * nside), idx % (nside * nside)
            h, w = rem // nside, rem % nside
            la = lat[f, h, w]
            expected = math.degrees(math.asin(1.0 - k * k / (3.0 * nside * nside))) * (1 if north else -1)
            np.testing.assert_allclose(la, expected, atol=1e-9)
            np.testing.assert_allclose(np.cos(lam), np.cos(np.deg2rad(lon[f, h, w])), atol=1e-9)
            np.testing.assert_allclose(np.sin(lam), np.sin(np.deg2rad(lon[f, h, w])), atol=1e-9)
            assert np.isclose(math.degrees(_colatitude(nside, k)), 90.0 - abs(expected))
            # the ring is every pixel of that latitude
            assert np.isclose(np.abs(lat), abs(expected), atol=1e-7).sum() >= idx.size


@pytest.mark.parametrize("nside", [8, 32])
def test_corrections_preserve_constants(nside):
    _, _, Mx, My = build_down_correction(nside)
    np.testing.assert_allclose(Mx.sum(1) + My.sum(1), 1.0, atol=1e-12)
    _, _, Mc, Mf = build_up_correction(nside // 2)
    np.testing.assert_allclose(Mc.sum(1) + Mf.sum(1), 1.0, atol=1e-12)


def test_cap_depth():
    assert cap_depth(32) == 4 and cap_depth(16) == 4 and cap_depth(4) == 3 and cap_depth(2) == 1


def _plain_down(nside, channels, reflect=True):
    from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import DealiasedDownsample

    return DealiasedDownsample(
        in_channels=channels,
        resample_filter=[1.0, 2.0, 1.0],
        stride=2,
        hpx_padding_mode="isolatitude",
        nside=nside,
        reflection_equivariant=reflect,
    ).double()


_CROSS_POLAR = [3, 4, 5, 6]  # uniform flow and rotation about x: the winds that cross the pole


def test_down_correction_removes_polar_error_for_scalars_and_winds():
    nf = 64
    fine, coarse = _fields(nf), _fields(nf // 2)
    down = _plain_down(nf, fine.shape[1])
    cap = PolarCapDownCorrection(nf)
    with torch.no_grad():
        y = down(fine)
        y_fixed = cap(fine, y.clone())
    old1, new1 = (_ring_rms(t - coarse, coarse, nf // 2, [1]) for t in (y, y_fixed))
    old24, new24 = (_ring_rms(t - coarse, coarse, nf // 2, range(2, 5)) for t in (y, y_fixed))
    # the innermost ring is rebuilt exactly (waves 0 and 1 are all there is on four pixels)
    assert new1.max() < 1e-3, new1
    assert old1[_CROSS_POLAR].min() > 0.2, old1
    # rings 2-4 keep their wave-2 content: better than the blur alone, not exact
    assert (new24[_CROSS_POLAR] < 0.9 * old24[_CROSS_POLAR]).all(), (old24, new24)
    assert (new24 <= old24 + 1e-6).all(), (old24, new24)
    # nothing outside the corrected rings moves
    moved = (y_fixed - y).abs() > 0
    assert not bool(moved.permute(1, 0, 2, 3).reshape(y.shape[1], -1)[:, np.concatenate([_ring(nf // 2, n_, k)[0] for n_ in (True, False) for k in range(5, 16)])].any())


def test_up_correction_removes_polar_error_for_scalars_and_winds():
    from physicsnemo.models.dlwp_healpix_layers.healpix_ring_mean_pchip import RingMeanPCHIPUpsample

    nc = 32
    coarse, fine = _fields(nc), _fields(2 * nc)
    flat = lambda t: t.permute(1, 0, 2, 3).reshape(1, t.shape[1], -1).float()
    unflat = lambda v, n: v.reshape(v.shape[1], 12, n, n).permute(1, 0, 2, 3).double()
    plain, fixed = RingMeanPCHIPUpsample(nc), RingMeanPCHIPUpsample(nc, pole_correction=True)
    with torch.no_grad():
        f0, f1 = plain(flat(coarse)), fixed(flat(coarse))
        y0, y1 = unflat(f0, 2 * nc), unflat(f1, 2 * nc)
    old1, new1 = (_ring_rms(t - fine, fine, 2 * nc, [1]) for t in (y0, y1))
    old24, new24 = (_ring_rms(t - fine, fine, 2 * nc, range(2, 5)) for t in (y0, y1))
    assert new1.max() < 1e-3, new1
    assert old1[_CROSS_POLAR].min() > 0.4, old1
    assert (new24[_CROSS_POLAR] < 0.5 * old24[_CROSS_POLAR]).all(), (old24, new24)
    # fine rings beyond 2K are the plain PCHIP output, bit for bit
    idx = np.concatenate([_ring(2 * nc, n_, k)[0] for n_ in (True, False) for k in range(1, 2 * cap_depth(nc) + 1)])
    keep = np.ones(12 * (2 * nc) ** 2, dtype=bool)
    keep[idx] = False
    assert torch.equal(f0[..., keep], f1[..., keep])


@pytest.mark.parametrize("nside", [16, 32])
def test_corrections_commute_with_reflection_and_rotation(nside):
    torch.manual_seed(0)
    cap_d, cap_u = PolarCapDownCorrection(nside), PolarCapUpCorrection(nside // 2)
    x, y = torch.randn(24, 3, nside, nside), torch.randn(24, 3, nside // 2, nside // 2)
    for op in (_reflect, _rotate):
        torch.testing.assert_close(cap_d(op(x), op(y).clone()), op(cap_d(x, y.clone())), rtol=0, atol=2e-6)
    nc = nside // 2
    fine, coarse = torch.randn(2, 3, 12 * nside * nside), torch.randn(2, 3, 12 * nc * nc)

    def op_flat(f, n):  # reflection / rotation on flat face-major pixels
        return f

    for op in (_reflect, _rotate):
        t = lambda v, n: op(v.reshape(2, 3, 12, n, n).permute(0, 2, 1, 3, 4).reshape(24, 3, n, n)).reshape(2, 12, 3, n, n).permute(0, 2, 1, 3, 4).reshape(2, 3, -1)
        torch.testing.assert_close(cap_u(t(fine, nside), t(coarse, nc)), t(cap_u(fine.clone(), coarse), nside), rtol=0, atol=2e-6)


def test_zonal_fields_stay_zonal():
    nc = 16
    lat_f, _ = _grid(2 * nc)
    lat_c, _ = _grid(nc)
    zon = lambda lat: torch.tensor(np.cos(2 * np.deg2rad(lat)) + 0.3 * np.sin(np.deg2rad(lat)), dtype=torch.float32)
    coarse = zon(lat_c).reshape(1, 1, -1)
    fine = torch.zeros(1, 1, 12 * (2 * nc) ** 2)
    fine = PolarCapUpCorrection(nc)(fine.clone(), coarse)  # a zonal coarse field gives a zonal correction
    for north in (True, False):
        for r in range(1, 2 * cap_depth(nc) + 1):
            idx = _ring(2 * nc, north, r)[0]
            assert float(fine[0, 0, idx].max() - fine[0, 0, idx].min()) < 1e-5
    x = zon(lat_f).reshape(12, 1, 2 * nc, 2 * nc)
    y = zon(lat_c).reshape(12, 1, nc, nc).clone()
    y = PolarCapDownCorrection(2 * nc)(x, y)
    for north in (True, False):
        for k in range(1, cap_depth(nc) + 1):
            idx = _ring(nc, north, k)[0]
            vals = y.reshape(12, nc * nc)[idx // (nc * nc), idx % (nc * nc)]
            assert float(vals.max() - vals.min()) < 1e-5


def test_corrections_are_linear_and_differentiable():
    torch.manual_seed(1)
    cap = PolarCapDownCorrection(16)
    x1, x2, y1, y2 = (torch.randn(12, 2, s, s) for s in (16, 16, 8, 8))
    out = lambda x, y: cap(x, y.clone())
    torch.testing.assert_close(out(2 * x1 + 3 * x2, 2 * y1 + 3 * y2), 2 * out(x1, y1) + 3 * out(x2, y2), rtol=1e-4, atol=1e-4)
    x1.requires_grad_(True)
    y1.requires_grad_(True)
    out(x1, y1).square().sum().backward()
    assert torch.isfinite(x1.grad).all() and torch.isfinite(y1.grad).all() and float(x1.grad.abs().sum()) > 0
    up = PolarCapUpCorrection(8)
    f, c = torch.randn(1, 2, 12 * 256, requires_grad=True), torch.randn(1, 2, 12 * 64, requires_grad=True)
    up(f.clone(), c).square().sum().backward()
    assert torch.isfinite(f.grad).all() and torch.isfinite(c.grad).all() and float(c.grad.abs().sum()) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_down_correction_cuda_channels_last_and_graph_capture(dtype):
    cap = PolarCapDownCorrection(64).cuda()
    x = torch.randn(24, 8, 64, 64, device="cuda", dtype=dtype).to(memory_format=torch.channels_last)
    y = torch.randn(24, 8, 32, 32, device="cuda", dtype=dtype).to(memory_format=torch.channels_last)
    ref = cap(x, y.clone())
    cpu = PolarCapDownCorrection(64)(x.float().cpu(), y.float().cpu().clone())
    torch.testing.assert_close(ref.float().cpu(), cpu, rtol=2e-2 if dtype == torch.bfloat16 else 1e-5, atol=2e-2 if dtype == torch.bfloat16 else 1e-5)
    static_y = y.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cap(x, static_y.clone())
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = cap(x, static_y)
    static_y.copy_(y)
    graph.replay()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def _padding_has_pole_mode():
    from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import make_hpx_padding_layer

    try:
        make_hpx_padding_layer(1, POLE_CORRECTION_MODE, False, nside=8)
        return True
    except ValueError:
        return False


def test_flag_selects_the_corrections_in_the_pchip_layers():
    from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import DealiasedDownsample
    from physicsnemo.models.dlwp_healpix_layers.healpix_ring_mean_pchip import (
        ReflectionSteerableRingMeanPCHIPConv,
        RingMeanPCHIPUpsampleFaces,
    )

    assert RingMeanPCHIPUpsampleFaces(in_channels=2, nside=8, hpx_padding_mode=POLE_CORRECTION_MODE).resample.cap is not None
    assert RingMeanPCHIPUpsampleFaces(in_channels=2, nside=8, hpx_padding_mode="isolatitude").resample.cap is None
    assert RingMeanPCHIPUpsampleFaces(in_channels=2, nside=8).resample.cap is None
    down = DealiasedDownsample(in_channels=2, nside=16, hpx_padding_mode="isolatitude")
    assert not down.pole_correction and not any(isinstance(m, PolarCapDownCorrection) for m in down.modules())
    with pytest.raises(ValueError, match="requires nside"):
        DealiasedDownsample(in_channels=2, hpx_padding_mode=POLE_CORRECTION_MODE)
    if not _padding_has_pole_mode():
        pytest.skip("the isolatitude_pole_correction padding is on another branch")
    down = DealiasedDownsample(in_channels=2, nside=16, hpx_padding_mode=POLE_CORRECTION_MODE, reflection_equivariant=True)
    assert down.pole_correction and sum(isinstance(m, PolarCapDownCorrection) for m in down.modules()) == 1
    conv = ReflectionSteerableRingMeanPCHIPConv(in_channels=4, out_channels=4, nside=8, hpx_padding_mode=POLE_CORRECTION_MODE)
    assert conv.resample.cap is not None
    assert ReflectionSteerableRingMeanPCHIPConv(in_channels=4, out_channels=4, nside=8).resample.cap is None
