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

"""Regressions for native resampling and recurrence across inference and training."""

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
        smoother.smoother_kernel.float(), original_kernel.float(), rtol=0, atol=0
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_smoothing_compiled_graph_replays(dtype):
    """Casted smoothing weights remain local to each graph replay."""
    reference = SmoothedInterpolate(in_channels=2).cuda()
    candidate = copy.deepcopy(reference)
    compiled = torch.compile(candidate, mode="reduce-overhead", dynamic=False)
    state_keys = set(candidate.state_dict())
    master_kernel = candidate.smoother_kernel
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        inputs = torch.randn(
            12, 2, 8, 8, device="cuda", dtype=dtype, requires_grad=True
        )
        expected, actual = reference(inputs), compiled(inputs)
        torch.testing.assert_close(actual, expected)
        expected_gradient = torch.autograd.grad(
            expected.float().square().mean(), inputs
        )[0]
        actual_gradient = torch.autograd.grad(actual.float().square().mean(), inputs)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert candidate.smoother_kernel is master_kernel
        assert set(candidate.state_dict()) == state_keys


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("enable_nhwc", [False, True])
def test_convgru_reset_releases_hidden_state(dtype, enable_nhwc):
    """Repeated forecasts restart from current input without retaining old graphs."""
    from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import ConvGRUBlock

    torch.manual_seed(7)
    block = ConvGRUBlock(in_channels=3, enable_nhwc=enable_nhwc).to(dtype=dtype)
    inputs = torch.randn(12, 3, 8, 8, dtype=dtype)
    if enable_nhwc:
        inputs = inputs.to(memory_format=torch.channels_last)
    expected = block(inputs).detach().clone()
    state_keys = set(block.state_dict())
    for _ in range(3):
        block.h = torch.full_like(block.h, float("nan"))
        block.reset()
        state = inputs.detach().clone().requires_grad_()
        output = block(state)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        gradients = torch.autograd.grad(
            output.float().square().mean(), (state, *block.parameters())
        )
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert set(block.state_dict()) == state_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_convgru_compiled_graphs_restart_forecasts():
    """Graph replay must discard prior forecast state before reading its tensors."""
    import copy

    from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import ConvGRUBlock

    class RestartedGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = ConvGRUBlock(in_channels=3)

        def forward(self, inputs):
            self.block.reset()
            first = self.block(inputs)
            return first + self.block(inputs)

    torch.manual_seed(7)
    reference = RestartedGRU().cuda()
    candidate = copy.deepcopy(reference)
    compiled = torch.compile(candidate, mode="reduce-overhead", dynamic=False)
    state_keys = set(candidate.state_dict())
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        inputs = torch.randn(12, 3, 8, 8, device="cuda", requires_grad=True)
        expected = reference(inputs)
        actual = compiled(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        expected_gradients = torch.autograd.grad(
            expected.square().mean(), (inputs, *reference.parameters())
        )
        actual_gradients = torch.autograd.grad(
            actual.square().mean(), (inputs, *candidate.parameters())
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=1e-4, atol=1e-6
            )
        assert set(candidate.state_dict()) == state_keys
