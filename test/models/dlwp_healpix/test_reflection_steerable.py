# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch as th

from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import (
    HEALPixLayer,
    SmoothedInterpolate,
)
from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
    HEALPixPaddingIsolatitude,
)
from physicsnemo.models.dlwp_healpix_layers.reflection_ops import (
    apply_R_kernel,
    bank_sizes,
    compute_sin_lat_faces,
    expand_sin_lat_folded,
    hpx_reflect_typed,
    hpx_spatial_reflect,
)
from physicsnemo.models.dlwp_healpix_layers.reflection_steerable_blocks import (
    ReflectionSteerableSmoothedInterpolateConv,
    ReflectionSteerableSymmetricConvNeXtBlock,
)
from physicsnemo.models.dlwp_healpix_layers.reflection_steerable_conv import (
    ParitySplitActivation,
    ReflectionSteerableConv1x1,
    ReflectionSteerableConv2d,
    require_reflection_steerable_tanh_activation,
)
from physicsnemo.models.layers.activations import CappedGELU, Tanh


def _random_typed(batch: int, channels: int, nside: int, odd_fraction: float, device="cpu"):
    n_even, n_odd = bank_sizes(channels, odd_fraction)
    x = th.randn(batch * 12, channels, nside, nside, device=device)
    return x, n_even, n_odd


def test_require_reflection_steerable_tanh_activation():
    act = require_reflection_steerable_tanh_activation(Tanh(), where="test")
    assert isinstance(act, Tanh)
    assert isinstance(act, th.nn.Tanh)
    # torch.nn.Tanh still accepted (subclass check)
    assert isinstance(
        require_reflection_steerable_tanh_activation(th.nn.Tanh(), where="test"), th.nn.Tanh
    )
    assert require_reflection_steerable_tanh_activation(None, where="test") is None
    try:
        require_reflection_steerable_tanh_activation(CappedGELU(cap_value=10), where="test")
        raise AssertionError("expected TypeError for CappedGELU")
    except TypeError:
        pass


def test_parity_block_rejects_non_tanh_config_activation():
    try:
        ReflectionSteerableSymmetricConvNeXtBlock(
            in_channels=8,
            latent_channels=8,
            out_channels=8,
            activation=CappedGELU(cap_value=10),
            odd_fraction=0.25,
            nside=8,
        )
        raise AssertionError("expected TypeError for CappedGELU")
    except TypeError:
        pass
    ReflectionSteerableSymmetricConvNeXtBlock(
        in_channels=8,
        latent_channels=8,
        out_channels=8,
        activation=Tanh(),
        odd_fraction=0.25,
        nside=8,
    )


def test_parity_split_activation_requires_tanh_by_default():
    try:
        ParitySplitActivation(8, 0.25, CappedGELU(cap_value=10), Tanh(), unified=False)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass
    # Ablation escape hatch
    ParitySplitActivation(
        8, 0.25, CappedGELU(cap_value=10), Tanh(), unified=False, allow_non_tanh=True
    )


def test_sin_lat_is_odd_under_rho():
    nside = 8
    sin_faces = compute_sin_lat_faces(nside)
    x = sin_faces
    xr = hpx_spatial_reflect(x)
    th.testing.assert_close(xr, -x, atol=1e-5, rtol=1e-5)


def test_isolatitude_pad_intertwines_with_rho():
    nside = 8
    padding = 1
    pad = HEALPixPaddingIsolatitude(padding=padding, nside=nside)
    x = th.randn(2 * 12, 3, nside, nside)
    px = pad(x)
    rx = hpx_spatial_reflect(x)
    prx = pad(rx)
    rpx = hpx_spatial_reflect(px)
    th.testing.assert_close(prx, rpx, atol=1e-5, rtol=1e-4)


def test_hpx_smoother_kernel_is_r_symmetric():
    k = th.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / 4.0
    k = k.view(1, 1, 3, 3)
    th.testing.assert_close(apply_R_kernel(k), k, atol=1e-6, rtol=1e-6)


def test_smoothed_interpolate_equivariant():
    th.manual_seed(3)
    nside = 8
    odd_fraction = 0.25
    in_ch = 8
    n_even, _ = bank_sizes(in_ch, odd_fraction)
    layer = HEALPixLayer(
        layer=SmoothedInterpolate,
        in_channels=in_ch,
        scale_factor=2,
        mode="nearest",
        trim_size=1,
        hpx_padding_mode="isolatitude",
        nside=nside,
    )
    x, _, _ = _random_typed(2, in_ch, nside, odd_fraction)
    y = layer(x)
    y_ref = hpx_reflect_typed(layer(hpx_reflect_typed(x, n_even)), n_even)
    th.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-4)


def test_parity_smoothed_interpolate_conv_equivariant():
    th.manual_seed(4)
    nside = 8
    odd_fraction = 0.25
    in_ch = 8
    n_even, _ = bank_sizes(in_ch, odd_fraction)
    block = ReflectionSteerableSmoothedInterpolateConv(
        in_channels=in_ch,
        out_channels=in_ch,
        kernel_size=3,
        scale_factor=2,
        mode="nearest",
        odd_fraction=odd_fraction,
        hpx_padding_mode="isolatitude",
        nside=nside,
    )
    x, _, _ = _random_typed(2, in_ch, nside, odd_fraction)
    y = block(x)
    y_ref = hpx_reflect_typed(block(hpx_reflect_typed(x, n_even)), n_even)
    th.testing.assert_close(y, y_ref, atol=1e-4, rtol=1e-3)


def test_reflection_steerable_conv2d_equivariant():
    th.manual_seed(0)
    nside = 8
    odd_fraction = 0.25
    in_ch, out_ch = 8, 8
    conv = ReflectionSteerableConv2d(in_ch, out_ch, kernel_size=3, odd_fraction=odd_fraction)
    pad = HEALPixPaddingIsolatitude(padding=1, nside=nside)
    x, n_even, _ = _random_typed(2, in_ch, nside, odd_fraction)

    def f(inp):
        return conv(pad(inp))

    y = f(x)
    y_ref = hpx_reflect_typed(f(hpx_reflect_typed(x, n_even)), n_even)
    th.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-4)


def test_reflection_steerable_conv1x1_same_parity_equivariant():
    th.manual_seed(2)
    nside = 8
    odd_fraction = 0.25
    in_ch, out_ch = 8, 8
    conv = ReflectionSteerableConv1x1(in_ch, out_ch, odd_fraction=odd_fraction, use_sin_lat_gate=False)
    x, n_even, _ = _random_typed(2, in_ch, nside, odd_fraction)
    y = conv(x)
    y_ref = hpx_reflect_typed(conv(hpx_reflect_typed(x, n_even)), bank_sizes(out_ch, odd_fraction)[0])
    th.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-4)


def test_reflection_steerable_conv1x1_equivariant():
    th.manual_seed(1)
    nside = 8
    odd_fraction = 0.25
    in_ch, out_ch = 8, 8
    conv = ReflectionSteerableConv1x1(in_ch, out_ch, odd_fraction=odd_fraction, use_sin_lat_gate=True)
    x, n_even, _ = _random_typed(2, in_ch, nside, odd_fraction)
    sin_faces = compute_sin_lat_faces(nside)
    s = expand_sin_lat_folded(sin_faces, x.shape[0])

    y = conv(x, sin_lat=s)
    x_r = hpx_reflect_typed(x, n_even)
    s_for_rx = expand_sin_lat_folded(sin_faces, x_r.shape[0])
    y_r = conv(x_r, sin_lat=s_for_rx)
    y_ref = hpx_reflect_typed(y_r, bank_sizes(out_ch, odd_fraction)[0])
    th.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-4)
