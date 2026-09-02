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
