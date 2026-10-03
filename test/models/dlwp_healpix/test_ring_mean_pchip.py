"""Ring-mean PCHIP keeps a zonal map zonal and does not overshoot a step."""

import math

import numpy as np
import torch

import earth2grid
from earth2grid.healpix import HEALPIX_PAD_XY
from physicsnemo.models.dlwp_healpix_layers.healpix_ring_mean_pchip import (
    RingMeanPCHIPUpsample,
    _bilinear_regrid,
    _geographic_latitude,
)
from physicsnemo.models.layers.activations import Tanh


def _ring_spread(values: torch.Tensor, lat: np.ndarray) -> float:
    """Largest within-ring standard deviation of a flat pixel vector."""
    key = np.round(lat, decimals=8)
    spread = 0.0
    flat = values.detach().cpu().numpy().reshape(-1)
    for ring in np.unique(key):
        spread = max(spread, float(np.std(flat[key == ring])))
    return spread


def test_zonal_cosine_stays_constant_on_each_fine_ring():
    nside = 8
    src = earth2grid.healpix.Grid(level=int(math.log2(nside)), pixel_order=HEALPIX_PAD_XY)
    dst = earth2grid.healpix.Grid(level=int(math.log2(nside * 2)), pixel_order=HEALPIX_PAD_XY)
    src_lat = _geographic_latitude(src)
    field = np.cos(src_lat)
    layer = RingMeanPCHIPUpsample(nside=nside)
    out = layer(torch.from_numpy(field.astype(np.float32)).view(1, 1, -1))
    assert _ring_spread(out, _geographic_latitude(dst)) < 1e-6


def test_zonal_step_does_not_overshoot():
    nside = 8
    src = earth2grid.healpix.Grid(level=int(math.log2(nside)), pixel_order=HEALPIX_PAD_XY)
    lat = _geographic_latitude(src)
    field = np.where(lat >= 0.0, 1.0, -1.0).astype(np.float32)
    layer = RingMeanPCHIPUpsample(nside=nside)
    out = layer(torch.from_numpy(field).view(1, 1, -1))
    assert float(out.min()) >= -1.0 - 1e-5
    assert float(out.max()) <= 1.0 + 1e-5


def test_bilinear_gather_matches_regridder_forward_and_backward():
    nside = 32
    layer = RingMeanPCHIPUpsample(nside=nside)
    n_in = 12 * nside * nside
    source = torch.randn(2, 8, n_in)
    left = source.detach().clone().requires_grad_(True)
    right = source.detach().clone().requires_grad_(True)
    gathered = _bilinear_regrid(left, layer.regrid.index, layer.regrid.weight)
    reference = layer.regrid(right)
    assert torch.allclose(gathered, reference, rtol=1e-5, atol=1e-5)
    grad_out = torch.randn_like(gathered)
    gathered.backward(grad_out)
    reference.backward(grad_out)
    assert torch.allclose(left.grad, right.grad, rtol=1e-5, atol=1e-5)


def test_steerable_wrapper_matches_conv_output_shape():
    from physicsnemo.models.dlwp_healpix_layers.healpix_ring_mean_pchip import (
        ReflectionSteerableRingMeanPCHIPConv,
    )

    nside = 4
    block = ReflectionSteerableRingMeanPCHIPConv(
        in_channels=4,
        out_channels=4,
        nside=nside,
        activation=Tanh(),
        hpx_padding_mode="isolatitude",
    )
    x = torch.randn(12, 4, nside, nside)
    y = block(x)
    assert y.shape == (12, 4, nside * 2, nside * 2)
    assert torch.isfinite(y).all()


def test_activation_dtype_resample_matches_fp32_and_backward():
    nside = 8
    n_in = 12 * nside * nside
    src = earth2grid.healpix.Grid(level=int(math.log2(nside)), pixel_order=HEALPIX_PAD_XY)
    field = np.cos(_geographic_latitude(src)).astype(np.float32)
    source = torch.from_numpy(field).view(1, 1, -1)
    fp32 = RingMeanPCHIPUpsample(nside=nside, resample_in_fp32=True)
    bf16 = RingMeanPCHIPUpsample(nside=nside, resample_in_fp32=False)
    reference = fp32(source)
    out = bf16(source.to(dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), reference, rtol=2e-2, atol=2e-2)
    # A fp32 caller must not keep the full-resolution resample in fp32.
    from_fp32 = bf16(source)
    assert from_fp32.dtype == torch.bfloat16
    assert torch.allclose(from_fp32.float(), reference, rtol=2e-2, atol=2e-2)

    grad_in = source.detach().to(dtype=torch.bfloat16).requires_grad_(True)
    bf16(grad_in).sum().backward()
    assert grad_in.grad is not None
    assert torch.isfinite(grad_in.grad).all()
    assert grad_in.grad.shape[-1] == n_in
