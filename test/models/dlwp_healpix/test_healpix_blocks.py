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

import common
import pytest
import torch
from pytest_utils import import_or_fail


@pytest.fixture
def test_data():
    # create dummy data
    def generate_test_data(faces=12, channels=2, img_size=16, device="cpu"):
        test_data = torch.eye(img_size).to(device)
        test_data = test_data[(None,) * 2]
        test_data = test_data.expand([faces, channels, -1, -1])

        return test_data

    return generate_test_data


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ConvGRUBlock_initialization(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        ConvGRUBlock,
    )

    in_channels = 2
    conv_gru_func = ConvGRUBlock(in_channels=in_channels).to(device)
    assert isinstance(conv_gru_func, ConvGRUBlock)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ConvGRUBlock_forward(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        ConvGRUBlock,
    )

    in_channels = 2
    tensor_size = 16
    conv_gru_func = ConvGRUBlock(in_channels=in_channels).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, in_channels, tensor_size, tensor_size])

    outvar = conv_gru_func(invar)
    assert outvar.shape == out_shape

    # check if tracking history
    outvar_hist = conv_gru_func(invar)
    assert not common.compare_output(outvar_hist, outvar)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ConvNeXtBlock_initialization(device, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        ConvNeXtBlock,
    )

    in_channels = 2
    convnext_block = ConvNeXtBlock(in_channels=in_channels).to(device)
    assert isinstance(convnext_block, ConvNeXtBlock)

    in_channels = 2
    out_channels = 2
    convnext_block = ConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(convnext_block, ConvNeXtBlock)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ConvNeXtBlock_forward(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        ConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    tensor_size = 16
    convnext_block = ConvNeXtBlock(in_channels=in_channels).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = convnext_block(invar)
    assert outvar.shape == out_shape

    out_channels = 2
    convnext_block = ConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert outvar.shape == out_shape


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_DoubleConvNeXtBlock_initialization(device, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        DoubleConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    latent_channels = 1
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)
    assert isinstance(doubleconvnextblock, DoubleConvNeXtBlock)

    latent_channels = 2
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(doubleconvnextblock, DoubleConvNeXtBlock)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_DoubleConvNeXtBlock_forward(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        DoubleConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    latent_channels = 1
    tensor_size = 16
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = doubleconvnextblock(invar)
    assert outvar.shape == out_shape

    latent_channels = 2
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)

    outvar = doubleconvnextblock(invar)
    assert outvar.shape == out_shape


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_SymmetricConvNeXtBlock_initialization(device, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
    ).to(device)
    assert isinstance(symmetric_convnextblock, SymmetricConvNeXtBlock)

    latent_channels = 2
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(symmetric_convnextblock, SymmetricConvNeXtBlock)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_SymmetricConvNeXtBlock_forward(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    tensor_size = 16
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
    ).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = symmetric_convnextblock(invar)
    assert outvar.shape == out_shape


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_Multi_SymmetricConvNeXtBlock_initialization(device, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        Multi_SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    multi_symmetric_convnextblock = Multi_SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(multi_symmetric_convnextblock, Multi_SymmetricConvNeXtBlock)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_Multi_SymmetricConvNeXtBlock_forward(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        Multi_SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    tensor_size = 16
    multi_symmetric_convnextblock = Multi_SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(multi_symmetric_convnextblock, Multi_SymmetricConvNeXtBlock)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = multi_symmetric_convnextblock(invar)
    assert outvar.shape == out_shape


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_BasicConvBlock_initialization(device, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        BasicConvBlock,
    )

    in_channels = 3
    out_channels = 1
    latent_channels = 2
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)
    assert isinstance(conv_block, BasicConvBlock)

    # test w/ activation and latent channels
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(conv_block, BasicConvBlock)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_BasicConvBlock_forward(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        BasicConvBlock,
    )

    in_channels = 3
    out_channels = 1
    tensor_size = 16
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)

    invar = test_data(
        channels=in_channels, faces=24, img_size=tensor_size, device=device
    )

    outvar = conv_block(invar)
    out_shape = torch.Size([24, out_channels, tensor_size, tensor_size])

    assert outvar.shape == out_shape


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_MaxPool_initialization(device, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        MaxPool,
    )

    pooling = 2
    maxpool_block = MaxPool(pooling=pooling).to(device)
    assert isinstance(maxpool_block, MaxPool)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_MaxPool_forward(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers import (
        MaxPool,
    )

    pooling = 2
    size = 16
    channels = 4
    maxpool_block = MaxPool(pooling=pooling).to(device)

    invar = test_data(
        faces=1, channels=channels, img_size=(size * pooling), device=device
    )
    outvar = test_data(faces=1, channels=channels, img_size=size, device=device)

    assert common.compare_output(outvar, maxpool_block(invar))


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_AvgPool_initialization(device, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        AvgPool,
    )

    pooling = 2
    avgpool_block = AvgPool(pooling=pooling).to(device)
    assert isinstance(avgpool_block, AvgPool)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_AvgPool_forward(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        AvgPool,
    )

    pooling = 2
    size = 32
    channels = 4
    avgpool_block = AvgPool(pooling=pooling).to(device)

    invar = test_data(
        faces=1, channels=channels, img_size=(size * pooling), device=device
    )
    outvar = test_data(faces=1, channels=channels, img_size=size, device=device)

    # averaging across 1,0
    outvar = outvar * 0.5

    assert common.compare_output(outvar, avgpool_block(invar))


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_DealiasedDownsample_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers import DealiasedDownsample

    channels = 4
    block = DealiasedDownsample(in_channels=channels).to(device)
    assert isinstance(block, DealiasedDownsample)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_DealiasedDownsample_inner_conv_matches_f_conv2d(device, test_data, pytestconfig):
    """Depthwise blur conv (no HEALPix wrapper) matches F.conv2d with symmetric padding."""
    import torch.nn.functional as F

    from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import DealiasBlurConv2d

    c, m, s = 2, 3, 2
    filt = (1.0, 2.0, 1.0)
    blur_conv = DealiasBlurConv2d(in_channels=c, stride=s, resample_filter=filt).to(device)
    w = blur_conv.weight.to(device)

    x = test_data(channels=c, img_size=32, device=device)
    assert torch.allclose(blur_conv.weight, w, atol=1e-5, rtol=1e-5)
    p = (m - 1) // 2
    ref = F.conv2d(x, w, padding=p, stride=s, groups=c)
    out = F.conv2d(x, blur_conv.weight, padding=p, stride=s, groups=c)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    "resample_filter,stride",
    [
        ((1.0,), 1),
        ((1.0,), 2),
        ((1.0, 2.0, 1.0), 2),
        ((1.0, 2.0, 1.0), 4),
        ((1.0, 4.0, 6.0, 4.0, 1.0), 2),
        ((1.0, 4.0, 6.0, 4.0, 1.0), 4),
    ],
)
def test_DealiasedDownsample_multiple_filters_and_strides(
    device, test_data, pytestconfig, resample_filter, stride
):
    from physicsnemo.models.dlwp_healpix_layers import DealiasedDownsample

    channels = 3
    h = 64
    block = DealiasedDownsample(
        in_channels=channels,
        resample_filter=resample_filter,
        stride=stride,
        enable_healpixpad=False,
    ).to(device)

    invar = test_data(channels=channels, img_size=h, device=device)
    out = block(invar)
    assert out.shape == (
        invar.shape[0],
        invar.shape[1],
        h // stride,
        h // stride,
    )
    assert torch.isfinite(out).all()


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("stride", [1, 2, 4, 8])
def test_DealiasedDownsample_filter_len1_matches_strided_slice_all_power2_strides(
    device, test_data, pytestconfig, stride
):
    from physicsnemo.models.dlwp_healpix_layers import DealiasedDownsample

    channels = 2
    h = 64
    block = DealiasedDownsample(
        in_channels=channels,
        resample_filter=(1.0,),
        stride=stride,
        enable_healpixpad=False,
    ).to(device)

    invar = test_data(channels=channels, img_size=h, device=device)
    out = block(invar)
    expected = invar[..., ::stride, ::stride]
    assert out.shape == expected.shape
    assert common.compare_output(expected, out)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("stride", [2, 4])
def test_DealiasedDownsample_reflection_equivariant_intertwines(
    device, test_data, pytestconfig, stride
):
    """Local Z₂ Reynolds wrap: down(ρx) ≈ ρ down(x) for triangle-3 stride-2 stacks."""
    from physicsnemo.models.dlwp_healpix_layers import DealiasedDownsample
    from physicsnemo.models.dlwp_healpix_layers.reflection_ops import hpx_spatial_reflect

    channels = 2
    nside = 16
    block = DealiasedDownsample(
        in_channels=channels,
        resample_filter=(1.0, 2.0, 1.0),
        stride=stride,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=nside,
        reflection_equivariant=True,
    ).to(device)
    x = test_data(channels=channels, img_size=nside, device=device)
    with torch.no_grad():
        y = block(x)
        y_from_rx = block(hpx_spatial_reflect(x))
        ry = hpx_spatial_reflect(y)
    assert torch.allclose(y_from_rx, ry, atol=1e-5, rtol=1e-4)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_DealiasedDownsample_strided_does_not_intertwine_on_even_nside(
    device, test_data, pytestconfig
):
    """Control: bare stride-2 BlurPool breaks R-intertwining on even faces."""
    from physicsnemo.models.dlwp_healpix_layers import DealiasedDownsample
    from physicsnemo.models.dlwp_healpix_layers.reflection_ops import hpx_spatial_reflect

    channels = 2
    nside = 16
    block = DealiasedDownsample(
        in_channels=channels,
        resample_filter=(1.0, 2.0, 1.0),
        stride=2,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=nside,
        reflection_equivariant=False,
    ).to(device)
    # Non-symmetric input so the sublattice mismatch is visible.
    x = torch.randn(12, channels, nside, nside, device=device)
    with torch.no_grad():
        err = (block(hpx_spatial_reflect(x)) - hpx_spatial_reflect(block(x))).abs().max()
    assert err.item() > 1e-3


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_DealiasedDownsample_reflection_equivariant_zonally_uniform_stays_even(
    device, pytestconfig
):
    """Soft zonal check: zonally uniform even field stays ρ-symmetric after Π(down).

    HEALPix face CNNs are not zonally equivariant in general, so we do not require
    zero ring spread after strided blur — only that a zonally uniform cos(lat)
    input (even under ρ) remains even under ρ at the coarser grid.
    """
    from earth2grid.healpix import coordinates as hpx_coordinates
    from physicsnemo.models.dlwp_healpix_layers import DealiasedDownsample
    from physicsnemo.models.dlwp_healpix_layers.reflection_ops import hpx_spatial_reflect

    channels = 1
    nside = 16
    x_coords = (torch.arange(nside, dtype=torch.float32) + 0.5) / nside
    xx, yy = torch.meshgrid(x_coords, x_coords, indexing="ij")
    faces = []
    for f in range(12):
        ff = torch.full_like(xx, f, dtype=torch.long)
        xs, ys = hpx_coordinates.face_to_global(xx, yy, ff)
        _lon, lat_deg = hpx_coordinates.global_to_angular(xs, ys)
        faces.append(torch.cos(torch.deg2rad(lat_deg)))
    field = torch.stack(faces, dim=0).unsqueeze(1).to(device)
    # Input is zonally uniform and even under ρ.
    assert torch.allclose(field, hpx_spatial_reflect(field), atol=1e-5, rtol=1e-4)

    block = DealiasedDownsample(
        in_channels=channels,
        resample_filter=(1.0, 2.0, 1.0),
        stride=2,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=nside,
        reflection_equivariant=True,
    ).to(device)
    with torch.no_grad():
        y = block(field)
    assert torch.allclose(y, hpx_spatial_reflect(y), atol=1e-5, rtol=1e-4)


@import_or_fail("hydra")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_DealiasedDownsample_reflection_equivariant_cuda_graph_capture(pytestconfig):
    """Reflection-equivariant downsample must not allocate during CUDA graph capture."""
    from physicsnemo.models.dlwp_healpix_layers import DealiasedDownsample

    device = torch.device("cuda:0")
    channels = 2
    nside = 16
    block = DealiasedDownsample(
        in_channels=channels,
        resample_filter=(1.0, 2.0, 1.0),
        stride=2,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=nside,
        reflection_equivariant=True,
    ).to(device)
    x = torch.randn(12, channels, nside, nside, device=device)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            y = block(x)
    assert y.shape == (12, channels, nside // 2, nside // 2)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_TransposedConvUpsample_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers import (
        TransposedConvUpsample,  #
    )

    transposed_conv_upsample_block = TransposedConvUpsample().to(device)
    assert isinstance(transposed_conv_upsample_block, TransposedConvUpsample)

    transposed_conv_upsample_block = TransposedConvUpsample(
        activation=torch.nn.ReLU()
    ).to(device)
    assert isinstance(transposed_conv_upsample_block, TransposedConvUpsample)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_TransposedConvUpsample_forward(device, test_data, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        TransposedConvUpsample,
    )

    in_channels = 2
    out_channels = 1
    size = 16

    transposed_conv_upsample_block = TransposedConvUpsample(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)

    invar = test_data(faces=1, channels=in_channels, img_size=size, device=device)
    outsize = torch.Size([1, out_channels, size * 2, size * 2])

    outvar = transposed_conv_upsample_block(invar)
    assert outvar.shape == outsize

    transposed_conv_upsample_block = TransposedConvUpsample(
        activation=torch.nn.ReLU()
    ).to(device)

    invar = test_data(faces=1, channels=(in_channels + 1), img_size=size, device=device)
    outvar = transposed_conv_upsample_block(invar)
    assert outvar.shape == outsize


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_Interpolate_initialization(device, pytestconfig):

    from physicsnemo.models.dlwp_healpix_layers import (
        Interpolate,
    )

    scale = 2
    mode = "linear"
    interpolation_block = Interpolate(scale_factor=scale, mode=mode).to(device)
    assert isinstance(interpolation_block, Interpolate)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_Interpolate_forward(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers import (
        Interpolate,
    )

    scale = 2
    mode = "linear"
    interpolation_block = Interpolate(scale_factor=scale, mode=mode).to(device)

    tensor_size = torch.randint(low=2, high=4, size=(3,)).tolist()
    invar = torch.rand(tensor_size).to(device)

    outvar = torch.nn.functional.interpolate(
        invar,
        scale_factor=scale,
        mode=mode,
    ).to(device)

    assert common.compare_output(outvar, interpolation_block(invar))
