# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: E402
import os
import sys

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import pytest
import torch
from pytest_utils import import_or_fail


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixPaddingIsolatitude_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers import HEALPixPaddingIsolatitude

    pad = HEALPixPaddingIsolatitude(padding=2, nside=16)
    assert isinstance(pad, HEALPixPaddingIsolatitude)

    with pytest.raises(ValueError, match="invalid value for 'padding'"):
        HEALPixPaddingIsolatitude(padding=0, nside=16)
    with pytest.raises(ValueError, match="nside must be a positive int"):
        HEALPixPaddingIsolatitude(padding=1, nside=0)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("padding", [1, 2, 3, 4, 5])
def test_HEALPixPaddingIsolatitude_forward_shape(device, padding, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers import HEALPixPaddingIsolatitude

    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    num_faces = 12
    # Keep CUDA allocations small so shared / low-memory GPUs do not OOM.
    batch_size = 1 if device == "cuda:0" else 2
    hw = 16
    c = 2 if device == "cuda:0" else 4
    if device == "cuda:0":
        torch.cuda.empty_cache()

    pad_mod = HEALPixPaddingIsolatitude(padding=padding, nside=hw)
    tensor_size = (batch_size * num_faces, c, hw, hw)
    try:
        invar = torch.rand(tensor_size, device=device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            pytest.skip("CUDA OOM allocating HEALPixPaddingIsolatitude test input")
        raise

    outvar = pad_mod(invar)
    hw_p = hw + 2 * padding
    assert outvar.shape == (batch_size * num_faces, c, hw_p, hw_p)


@import_or_fail("hydra")
@pytest.mark.parametrize("padding", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("hw", [16, 32, 64])
@pytest.mark.parametrize("enable_nhwc", [False, True])
def test_healpix_padding_isolatitude_matches_folded_reference(
    padding, hw, enable_nhwc, pytestconfig
):
    """HEALPixPaddingIsolatitude must match isolatitude_pad_folded (gather = reference)."""
    from physicsnemo.models.dlwp_healpix_layers import HEALPixPaddingIsolatitude
    from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
        isolatitude_pad_folded,
    )

    if 2 * padding > hw:
        pytest.skip("face size too small for padding (isolatitude corner synthesis)")

    torch.manual_seed(0)
    batch_size = 2
    num_faces = 12
    c = 3
    x = torch.randn(batch_size * num_faces, c, hw, hw)

    ref = isolatitude_pad_folded(x, padding, enable_nhwc)
    y = HEALPixPaddingIsolatitude(
        padding=padding, nside=hw, enable_nhwc=enable_nhwc
    )(x)

    # Gather path uses 0.5 * (g0 + g1) in a form that can differ by ~1 ULP from the
    # reference on some output cells.
    torch.testing.assert_close(y, ref, rtol=1.0e-5, atol=1.0e-6)


@import_or_fail("hydra")
@pytest.mark.parametrize("enable_nhwc", [False, True])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_isolatitude_pad_triton_matches_gather_fwd_bwd(enable_nhwc, dtype, pytestconfig):
    """CUDA fused pad must match the ATen gather path in value and input grad."""
    from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
        HEALPixPaddingIsolatitude,
    )
    from physicsnemo.models.dlwp_healpix_layers.isolatitude_pad_triton import (
        isolatitude_pad_cuda_available,
    )

    if not isolatitude_pad_cuda_available():
        pytest.skip("Triton CUDA pad kernel unavailable")

    torch.manual_seed(0)
    padding, hw, batch_size, c = 1, 16, 2, 8
    dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    x_cpu = torch.randn(batch_size * 12, c, hw, hw, dtype=torch.float32)
    if enable_nhwc:
        x_cpu = x_cpu.to(memory_format=torch.channels_last)
    x_cpu = x_cpu.detach().requires_grad_(True)

    pad_cpu = HEALPixPaddingIsolatitude(
        padding=padding, nside=hw, enable_nhwc=enable_nhwc
    )
    y_cpu = pad_cpu(x_cpu)
    go = torch.randn_like(y_cpu)
    (g_cpu,) = torch.autograd.grad(y_cpu, x_cpu, go)

    x_gpu = x_cpu.detach().to(device="cuda", dtype=dt).requires_grad_(True)
    if enable_nhwc:
        x_gpu = x_gpu.to(memory_format=torch.channels_last)
    pad_gpu = HEALPixPaddingIsolatitude(
        padding=padding, nside=hw, enable_nhwc=enable_nhwc
    ).to("cuda")
    y_gpu = pad_gpu(x_gpu)
    go_gpu = go.to(device="cuda", dtype=dt)
    if enable_nhwc:
        go_gpu = go_gpu.to(memory_format=torch.channels_last)
        assert y_gpu.is_contiguous(memory_format=torch.channels_last)
    (g_gpu,) = torch.autograd.grad(y_gpu, x_gpu, go_gpu)

    rtol = 2.0e-2 if dtype == "bf16" else 1.0e-5
    atol = 2.0e-2 if dtype == "bf16" else 1.0e-6
    torch.testing.assert_close(y_gpu.float().cpu(), y_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(g_gpu.float().cpu(), g_cpu, rtol=rtol, atol=atol)


@import_or_fail("hydra")
@pytest.mark.parametrize("padding", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("hw", [16, 32, 64])
@pytest.mark.parametrize("enable_nhwc", [False, True])
def test_healpix_padding_karlbauer_matches_earth2grid_v2(padding, hw, enable_nhwc, pytestconfig):
    """Karlbauer HEALPixPadding and earth2grid HEALPixPaddingv2 agree for several pad widths."""
    from physicsnemo.models.dlwp_healpix_layers import (
        HEALPixPadding,
        HEALPixPaddingv2,
        have_earth2grid,
    )

    if not have_earth2grid:
        pytest.skip("earth2grid.healpix.pad not available")

    # earth2grid pad matches Karlbauer on CPU; run there to avoid GPU OOM on shared nodes.
    device = "cpu"
    torch.manual_seed(1)
    batch_size = 2
    num_faces = 12
    c = 5
    x = torch.randn(batch_size * num_faces, c, hw, hw, device=device)

    y1 = HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc)(x)
    y2 = HEALPixPaddingv2(padding=padding, enable_nhwc=enable_nhwc)(x)

    torch.testing.assert_close(y1, y2, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# isolatitude_pole_correction
# ---------------------------------------------------------------------------
_REFL_FACE_ORDER = [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]


def _reflect(x):
    """Equatorial reflection on folded [B*12, C, H, W]: in-face R and the face permutation."""
    y = torch.rot90(torch.flip(x, dims=[-1]), dims=(-1, -2))
    y = y.reshape(-1, 12, *y.shape[1:])[:, _REFL_FACE_ORDER]
    return y.reshape(-1, *y.shape[2:])


def _rotate90(x):
    """90 degree longitude rotation: cycle the faces inside each of the three face rows."""
    perm = [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10]
    y = x.reshape(-1, 12, *x.shape[1:])[:, perm]
    return y.reshape(-1, *x.shape[1:])


def _pole_pad(nside, nhwc=False, device="cpu"):
    from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import make_hpx_padding_layer

    return make_hpx_padding_layer(1, "isolatitude_pole_correction", nhwc, nside=nside).to(device)


def _plain_pad(nside, nhwc=False, device="cpu"):
    from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import make_hpx_padding_layer

    return make_hpx_padding_layer(1, "isolatitude", nhwc, nside=nside).to(device)


def _grid(nside):
    import math

    import numpy as np

    earth2grid = pytest.importorskip("earth2grid")
    from earth2grid.healpix import HEALPIX_PAD_XY

    g = earth2grid.healpix.Grid(level=int(math.log2(nside)), pixel_order=HEALPIX_PAD_XY)
    lat = np.asarray(g.lat, dtype=np.float64).reshape(12, nside, nside)
    lon = np.asarray(g.lon, dtype=np.float64).reshape(12, nside, nside)
    return lat, lon


def _slot_positions(nside):
    """(face, row, col) of the (ring-0 a, ring-0 b, corner) slots of every polar face, north then south."""
    hp = nside + 2
    out = []
    for north in (True, False):
        for q in range(4):
            face = q if north else 8 + q
            if north:
                out.append((face, [(0, 1), (1, 0), (0, 0)]))
            else:
                out.append((face, [(hp - 2, hp - 1), (hp - 1, hp - 2), (hp - 1, hp - 1)]))
    return out


def _reference_slots(x, nside):
    """P and C from earth2grid pixel longitudes; rings are picked by latitude, not by index arithmetic.

    x: float64 numpy [12, n, n] (one channel). Returns {(face): (P, C)}.
    """
    import numpy as np

    lat, lon = _grid(nside)
    out = {}
    for north in (True, False):
        polar = lat > 0 if north else lat < 0
        levels = np.unique(np.round(np.abs(lat[polar]), 9))[::-1]  # highest |lat| first
        rings = [np.argwhere(np.isclose(np.abs(lat), v) & polar) for v in levels[:2]]
        assert len(rings[0]) == 4 and len(rings[1]) == 8
        for f in (range(4) if north else range(8, 12)):
            own = next(p for p in rings[0] if p[0] == f)
            lam0 = np.deg2rad(lon[tuple(own)])
            m0, m1 = [], []
            for ring in rings:
                v = np.array([x[tuple(p)] for p in ring])
                lam = np.deg2rad(np.array([lon[tuple(p)] for p in ring]))
                m0.append(v.mean())
                m1.append(2.0 * np.mean(v * np.cos(lam - lam0)))
            P = (2 * m0[0] - m0[1]) + (2 * m1[0] - m1[1])
            out[f] = (P, 2 * P - x[tuple(own)])
    return out


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize("nside", [8, 16])
def test_pole_correction_slots_match_reference_and_rest_is_isolatitude(device, nhwc, nside, pytestconfig):
    import numpy as np

    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    x = torch.randn(2 * 12, 3, nside, nside, dtype=torch.float32, device=device)
    y = _pole_pad(nside, nhwc, device)(x)
    y0 = _plain_pad(nside, nhwc, device)(x)
    hp = nside + 2
    full = torch.zeros(12, hp, hp, dtype=torch.bool)
    for face, slots in _slot_positions(nside):
        for r, c in slots:
            full[face, r, c] = True
    yv, y0v = y.reshape(2, 12, 3, hp, hp).cpu(), y0.reshape(2, 12, 3, hp, hp).cpu()
    m = full.view(1, 12, 1, hp, hp).expand_as(yv)
    assert torch.equal(yv[~m], y0v[~m])  # everything outside the 24 slots is plain isolatitude
    assert not torch.equal(yv[m], y0v[m])
    xs = x.reshape(2, 12, 3, nside, nside).cpu().double().numpy()
    for b in range(2):
        for ch in range(3):
            ref = _reference_slots(xs[b, :, ch], nside)
            for face, slots in _slot_positions(nside):
                P, C = ref[face]
                got = [float(yv[b, face, ch, r, c]) for r, c in slots]
                np.testing.assert_allclose(got, [P, P, C], rtol=2e-5, atol=2e-5)


def _analytic_fields(nside):
    """Scalar coordinates and the (u, v) components of smooth vector fields, float64 [12, k, n, n]."""
    import numpy as np

    lat, lon = _grid(nside)
    ph, la = torch.tensor(np.deg2rad(lat)), torch.tensor(np.deg2rad(lon))
    X, Y = torch.cos(ph) * torch.cos(la), torch.cos(ph) * torch.sin(la)
    el = torch.stack([-torch.sin(la), torch.cos(la), torch.zeros_like(la)])
    ep = torch.stack([-torch.sin(ph) * torch.cos(la), -torch.sin(ph) * torch.sin(la), torch.cos(ph)])
    r = torch.stack([X, Y, torch.sin(ph)])

    def uv(V):
        return (V * el).sum(0), (V * ep).sum(0)

    ux, vx = uv(torch.stack([torch.ones_like(X), 0 * X, 0 * X]))
    omega = torch.tensor([1.0, 0, 0], dtype=torch.float64).view(3, 1, 1, 1).expand_as(r)
    ur, vr = uv(torch.cross(omega, r, dim=0))
    return torch.stack([X, Y], 1), torch.stack([ux, vx], 1), torch.stack([ur, vr], 1), (el, ep)


@import_or_fail("hydra")
def test_pole_correction_is_first_order_exact_for_scalars_and_winds(pytestconfig):
    """Scalar fields with a gradient through the pole and cross-polar winds: slot error well below one pixel step.

    Targets: scalar corner = stored across pixel, scalar ring-0 slot = ring-1 mean; wind corner = minus the stored
    across vector, wind ring-0 slot = the pole vector (3-D mean of the 4 ring-1 vectors) in the face's own frame.
    The old fill is shown to be far worse for winds so the test cannot pass vacuously.
    """
    import numpy as np

    nside = 64
    lat, lon = _grid(nside)
    scal, wind_unif, wind_rot, (el, ep) = _analytic_fields(nside)
    new, old = _pole_pad(nside), _plain_pad(nside)
    n = nside

    def pad(f, layer):
        return layer(f.float()).double()

    def ring1(north):
        return [(q, 0, 0) for q in range(4)] if north else [(8 + q, n - 1, n - 1) for q in range(4)]

    results = {}
    for name, f, is_wind in (("scalar", scal, False), ("uniform", wind_unif, True), ("rotation", wind_rot, True)):
        yn, yo = pad(f, new), pad(f, old)
        err_new, err_old, step = [], [], []
        for north in (True, False):
            r1 = ring1(north)
            for q in range(4):
                face = q if north else 8 + q
                own = r1[q]
                across = r1[(q + 2) % 4]
                slots = [s for fc, s in _slot_positions(n) if fc == face][0]
                if is_wind:
                    vecs = []
                    for (fc, i, j) in r1:
                        u, v = f[fc, 0, i, j], f[fc, 1, i, j]
                        vecs.append(u * el[:, fc, i, j] + v * ep[:, fc, i, j])
                    pv = torch.stack(vecs).mean(0)
                    pv[2] = 0.0
                    ownframe = (el[:, face, own[1], own[2]], ep[:, face, own[1], own[2]])
                    tgt_pole = torch.stack([pv @ ownframe[0], pv @ ownframe[1]])
                    tgt_corner = -f[across[0], :, across[1], across[2]]
                else:
                    tgt_pole = f[[fc for fc, _, _ in r1], :, [i for _, i, _ in r1], [j for _, _, j in r1]].mean(0)
                    tgt_corner = f[across[0], :, across[1], across[2]]
                r, c = slots[0]
                rc, cc = slots[2]
                nb = (0, 1) if north else (n - 1, n - 2)
                step.append(float((f[face, :, nb[0], nb[1]] - f[face, :, own[1], own[2]]).norm()))
                err_new.append([float((yn[face, :, r, c] - tgt_pole).norm()), float((yn[face, :, rc, cc] - tgt_corner).norm())])
                err_old.append([float((yo[face, :, r, c] - tgt_pole).norm()), float((yo[face, :, rc, cc] - tgt_corner).norm())])
        results[name] = (np.mean(err_new, 0) / np.mean(step), np.mean(err_old, 0) / np.mean(step))
    for name in ("scalar", "uniform", "rotation"):
        new_err, old_err = results[name]
        assert new_err.max() < 0.1, (name, new_err)
    for name in ("uniform", "rotation"):
        new_err, old_err = results[name]
        assert old_err.max() > 10 * new_err.max(), (name, new_err, old_err)


@import_or_fail("hydra")
def test_pole_correction_constants_and_zonal_input(pytestconfig):
    nside = 16
    layer = _pole_pad(nside)
    x = torch.full((12, 2, nside, nside), 3.25)
    torch.testing.assert_close(layer(x), _plain_pad(nside)(x), rtol=0, atol=1e-6)
    import numpy as np

    lat, _ = _grid(nside)
    zon = torch.tensor(2.0 * np.sin(np.deg2rad(lat)) + np.cos(3.0 * np.deg2rad(lat)), dtype=torch.float32)
    y = layer(zon[:, None].expand(12, 2, nside, nside).contiguous())
    for faces in (range(0, 4), range(8, 12)):
        vals = [[float(y[f, 0, r, c]) for r, c in slots] for f, slots in _slot_positions(nside) if f in faces]
        for t_ in range(3):
            col = [v[t_] for v in vals]
            assert max(col) - min(col) < 1e-5


@import_or_fail("hydra")
@pytest.mark.parametrize("nside", [8, 16])
def test_pole_correction_commutes_with_reflection_and_rotation(nside, pytestconfig):
    torch.manual_seed(1)
    layer = _pole_pad(nside).double()
    x = torch.randn(2 * 12, 4, nside, nside, dtype=torch.float64)
    torch.testing.assert_close(layer(_reflect(x)), _reflect(layer(x)), rtol=0, atol=1e-12)
    torch.testing.assert_close(layer(_rotate90(x)), _rotate90(layer(x)), rtol=0, atol=1e-12)


@import_or_fail("hydra")
@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("pole", [True, False])
def test_pole_correction_cuda_matches_cpu_fwd_bwd(nhwc, dtype, pole, pytestconfig):
    """Fused CUDA path (pole kernels, masked main kernels) against the ATen path, value and input grad.

    ``pole=False`` also checks that plain isolatitude is unchanged by the masking added to the main kernels.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from physicsnemo.models.dlwp_healpix_layers.isolatitude_pad_triton import isolatitude_pad_cuda_available

    if not isolatitude_pad_cuda_available():
        pytest.skip("Triton CUDA pad kernel unavailable")
    torch.manual_seed(0)
    nside, b, c = 16, 2, 8
    dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    mk = _pole_pad if pole else _plain_pad
    x_cpu = torch.randn(b * 12, c, nside, nside)
    x_cpu = (x_cpu.to(memory_format=torch.channels_last) if nhwc else x_cpu).detach().requires_grad_(True)
    y_cpu = mk(nside, nhwc)(x_cpu)
    go = torch.randn_like(y_cpu)
    (g_cpu,) = torch.autograd.grad(y_cpu, x_cpu, go)
    x_gpu = x_cpu.detach().to(device="cuda", dtype=dt)
    x_gpu = (x_gpu.to(memory_format=torch.channels_last) if nhwc else x_gpu).requires_grad_(True)
    y_gpu = mk(nside, nhwc, "cuda")(x_gpu)
    if nhwc:
        assert y_gpu.is_contiguous(memory_format=torch.channels_last)
    go_gpu = go.to(device="cuda", dtype=dt)
    go_gpu = go_gpu.to(memory_format=torch.channels_last) if nhwc else go_gpu
    (g_gpu,) = torch.autograd.grad(y_gpu, x_gpu, go_gpu)
    rtol = atol = 3e-2 if dtype == "bf16" else 1e-5
    torch.testing.assert_close(y_gpu.float().cpu(), y_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(g_gpu.float().cpu(), g_cpu, rtol=rtol, atol=atol)


@import_or_fail("hydra")
def test_pole_correction_gradcheck_cpu(pytestconfig):
    layer = _pole_pad(4).double()
    x = torch.randn(12, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-6)


@import_or_fail("hydra")
def test_pole_correction_compiled_matches_eager(pytestconfig):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    layer = _pole_pad(16, True, "cuda")
    x = torch.randn(2 * 12, 8, 16, 16, device="cuda").to(memory_format=torch.channels_last).requires_grad_(True)
    go = torch.randn(2 * 12, 8, 18, 18, device="cuda").to(memory_format=torch.channels_last)
    y_e = layer(x)
    (g_e,) = torch.autograd.grad(y_e, x, go)
    y_c = torch.compile(layer)(x)
    (g_c,) = torch.autograd.grad(y_c, x, go)
    torch.testing.assert_close(y_c, y_e, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(g_c, g_e, rtol=1e-5, atol=1e-5)


@import_or_fail("hydra")
def test_pole_correction_factory_and_validation(pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
        HEALPixPaddingIsolatitude,
        make_hpx_padding_layer,
    )

    assert make_hpx_padding_layer(1, "isolatitude_pole_correction", False, nside=16).pole_correction
    assert not make_hpx_padding_layer(1, "isolatitude", False, nside=16).pole_correction
    with pytest.raises(ValueError, match="requires nside"):
        make_hpx_padding_layer(1, "isolatitude_pole_correction", False)
    with pytest.raises(ValueError, match="padding == 1"):
        make_hpx_padding_layer(2, "isolatitude_pole_correction", False, nside=16)
    with pytest.raises(ValueError, match="padding == 1"):
        HEALPixPaddingIsolatitude(padding=1, nside=1, pole_correction=True)
    with pytest.raises(ValueError, match="Unsupported hpx_padding_mode"):
        make_hpx_padding_layer(1, "isolatitude_pole", False, nside=16)
