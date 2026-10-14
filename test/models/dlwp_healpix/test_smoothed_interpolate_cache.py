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

"""CPU regressions for native resampling across inference and training."""

import copy

import pytest
import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import (
    SmoothedInterpolate,
    SmoothedInterpolateConv,
)


@pytest.mark.parametrize("inference_constructor", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_smoothing_inference_cache_then_backward(inference_constructor, dtype):
    """A cast or inference-created kernel must not poison subsequent gradients."""
    with torch.inference_mode(inference_constructor):
        layer = SmoothedInterpolate(in_channels=2)
    reference = SmoothedInterpolate(in_channels=2)
    original_values = reference.smoother_kernel.clone()
    original_keys = set(layer.state_dict())
    torch.manual_seed(11)
    values = torch.randn(1, 2, 8, 8, dtype=dtype)
    with torch.inference_mode():
        evaluated = layer(values)
    assert not torch.is_inference(layer.smoother_kernel)
    assert not layer.smoother_kernel.requires_grad
    torch.testing.assert_close(
        layer.smoother_kernel.float(), original_values, rtol=0, atol=0
    )
    inputs = values.clone().requires_grad_()
    reference_inputs = values.clone().requires_grad_()
    actual, expected = layer(inputs), reference(reference_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, evaluated, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    assert torch.isfinite(inputs.grad).all()
    assert set(layer.state_dict()) == original_keys


def test_native_healpix_smoothing_inference_then_training_cast():
    """Actual shared face-exchange upsampling retains parameter/input gradients."""
    layer = SmoothedInterpolateConv(
        in_channels=2, out_channels=3, hpx_padding_mode="karlbauer", nside=8
    ).double()
    reference = copy.deepcopy(layer)
    original_state = copy.deepcopy(layer.state_dict())
    smoother = next(
        module for module in layer.modules() if isinstance(module, SmoothedInterpolate)
    )
    original_kernel = smoother.smoother_kernel.clone()
    values = torch.randn(12, 2, 8, 8, dtype=torch.float64)
    with torch.inference_mode():
        layer(values)
    inputs = values.clone().requires_grad_()
    reference_inputs = values.clone().requires_grad_()
    actual, expected = layer(inputs), reference(reference_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    for parameter, other in zip(layer.parameters(), reference.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    assert not torch.is_inference(smoother.smoother_kernel)
    torch.testing.assert_close(
        smoother.smoother_kernel.float(), original_kernel, rtol=0, atol=0
    )
    assert set(layer.state_dict()) == set(original_state)
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)
    # Casting the cache back after inference also leaves ordinary immutable weights.
    layer.float()
    with torch.inference_mode():
        layer(values.float())
    layer(torch.randn_like(values.float(), requires_grad=True)).sum().backward()
    assert not torch.is_inference(smoother.smoother_kernel)
