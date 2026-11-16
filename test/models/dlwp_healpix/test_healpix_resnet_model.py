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
from utils import fix_random_seeds

from physicsnemo.models.dlwp_healpix import HEALPixResNet

omegaconf = pytest.importorskip("omegaconf")

SIZE = 16  # face height/width


@pytest.fixture
def conv_block_dict():
    return omegaconf.DictConfig(
        {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.ConvNeXtBlock",
            "kernel_size": 3,
            "upscale_factor": 2,
            "n_layers": 1,
            "activation": {
                "_target_": "physicsnemo.models.layers.activations.CappedGELU",
                "cap_value": 10,
            },
        }
    )


@pytest.fixture
def output_layer_dict():
    return omegaconf.DictConfig(
        {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.BasicConvBlock",
            "kernel_size": 1,
            "n_layers": 1,
        }
    )


@pytest.fixture
def make_model(conv_block_dict, output_layer_dict):
    """Factory for small ResNets; keyword arguments override the defaults."""

    def _make(**overrides):
        kwargs = dict(
            n_channels=[8, 8],
            dilations=[1, 2],
            conv_block=conv_block_dict,
            output_layer=output_layer_dict,
            input_channels=3,
            output_channels=3,
            n_constants=2,
            decoder_input_channels=1,
            input_time_dim=1,
            output_time_dim=1,
        )
        kwargs.update(overrides)
        return HEALPixResNet(**kwargs)

    return _make


def make_inputs(
    in_channels=3, n_constants=2, input_time_dim=1, output_time_dim=1, batch=2
):
    """[prognostics, decoder inputs (TOA insolation), constants]"""
    fix_random_seeds(seed=42)
    return [
        torch.randn(batch, 12, input_time_dim, in_channels, SIZE, SIZE),
        torch.randn(batch, 12, output_time_dim, 1, SIZE, SIZE),
        torch.randn(12, n_constants, SIZE, SIZE),
    ]


def test_HEALPixResNet_builds_one_block_per_n_channels_plus_output(make_model):
    model = make_model(n_channels=[8, 8, 8], dilations=[1, 2, 3])
    assert len(model.resnet) == 4  # 3 conv blocks + output layer


def test_HEALPixResNet_integration_steps(make_model):
    model = make_model(input_time_dim=2, output_time_dim=6)
    assert model.integration_steps == 3


def test_HEALPixResNet_invalid_time_dims(make_model):
    with pytest.raises(
        ValueError, match="'output_time_dim' must be a multiple of 'input_time_dim'"
    ):
        make_model(input_time_dim=2, output_time_dim=3)


def test_HEALPixResNet_requires_constants_or_decoder_inputs(make_model):
    with pytest.raises(NotImplementedError, match="no constant fields and no decoder"):
        make_model(n_constants=0, decoder_input_channels=0)


def test_HEALPixResNet_coupling_requirements(make_model):
    couplings = [{"params": {"variables": ["t2m"], "input_times": ["0h"]}}]
    with pytest.raises(NotImplementedError, match="coupled models with no constant"):
        make_model(n_constants=0, couplings=couplings)
    with pytest.raises(NotImplementedError, match="coupled models with no decoder"):
        make_model(decoder_input_channels=0, couplings=couplings)


def test_HEALPixResNet_counts_coupled_channels(make_model):
    couplings = [
        {"params": {"variables": ["a", "b"], "input_times": ["0h", "6h", "12h"]}}
    ]
    model = make_model(couplings=couplings)
    assert model.coupled_channels == 6
    # first conv block consumes prognostics + decoder inputs + constants + coupled
    assert model._compute_input_channels() == 3 + 1 + 2 + 6


def test_HEALPixResNet_forward_multi_step_shapes(make_model):
    model = make_model(input_time_dim=2, output_time_dim=6)
    inputs = make_inputs(input_time_dim=2, output_time_dim=6)
    out = model(inputs)
    assert out.shape == (2, 12, 6, 3, SIZE, SIZE)
    last = model(inputs, output_only_last=True)
    assert last.shape == (2, 12, 2, 3, SIZE, SIZE)
    torch.testing.assert_close(last, out[:, :, -2:])


def test_HEALPixResNet_no_decoder_inputs_and_no_constants_variants(make_model):
    # no decoder inputs
    model = make_model(decoder_input_channels=0)
    x, _, constants = make_inputs()
    assert model([x, constants]).shape == (2, 12, 1, 3, SIZE, SIZE)
    # no constants
    model = make_model(n_constants=0)
    x, decoder, _ = make_inputs()
    assert model([x, decoder]).shape == (2, 12, 1, 3, SIZE, SIZE)


def test_HEALPixResNet_coupled_forward(make_model):
    couplings = [{"params": {"variables": ["a", "b"], "input_times": ["0h"]}}]
    model = make_model(couplings=couplings, output_time_dim=2)
    x, decoder, constants = make_inputs(output_time_dim=2)
    # coupled inputs: [T, B, C, F, H, W], one entry per integration step
    coupled = torch.randn(2, 2, 2, 12, SIZE, SIZE)
    out = model([x, decoder, constants, coupled])
    assert out.shape == (2, 12, 2, 3, SIZE, SIZE)
    assert torch.isfinite(out).all()


def test_HEALPixResNet_constraints_receive_prediction_and_input(make_model):
    scaling = {"v0": {"mean": 0.0, "std": 1.0}}
    constraints = omegaconf.DictConfig(
        {
            "nonneg": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_constraints.NonnegativeConstraint",
                "variables": ["v0"],
                "in_channels": ["v0", "v1", "v2"],
                "out_channels": ["v0", "v1", "v2"],
                "scaling": {
                    k: scaling.get(k, {"mean": 0.0, "std": 1.0})
                    for k in ["v0", "v1", "v2"]
                },
            }
        }
    )
    model = make_model(constraints=constraints)
    out = model(make_inputs())
    assert (out[:, :, :, 0] >= 0).all()  # constrained channel is clamped at zero


def test_HEALPixResNet_backward(make_model):
    model = make_model()
    out = model(make_inputs())
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
