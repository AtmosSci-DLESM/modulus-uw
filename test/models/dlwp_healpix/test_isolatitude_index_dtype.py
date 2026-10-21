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

"""Preserve the int64 gather contract required by cluster Torch releases."""
import pytest
import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
    HEALPixPaddingIsolatitude,
    isolatitude_pad_folded,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_isolatitude_int64_gather_values_and_gradients(monkeypatch, device, dtype):
    """Check older gather API semantics and reference forward/backward parity."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    original = torch.Tensor.gather
    calls = []

    def strict_gather(tensor, dim, index, **kwargs):
        assert index.dtype == torch.int64
        calls.append(index.numel())
        return original(tensor, dim, index, **kwargs)

    monkeypatch.setattr(torch.Tensor, "gather", strict_gather)
    torch.manual_seed(7)
    x = torch.randn(12, 2, 16, 16, device=device, dtype=dtype, requires_grad=True)
    reference_input = x.detach().clone().requires_grad_(True)
    pad = HEALPixPaddingIsolatitude(padding=2, nside=16).to(device)
    # Exercise the actual int32 index buffers that triggered Torch2.8 failure.
    assert pad._index0.dtype == torch.int32
    actual = pad(x)
    expected = isolatitude_pad_folded(reference_input, 2, False)
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        x.grad, reference_input.grad, rtol=tolerance, atol=tolerance
    )
    assert len(calls) == 2
