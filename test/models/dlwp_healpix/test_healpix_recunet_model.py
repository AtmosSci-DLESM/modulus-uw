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
from utils import fix_random_seeds
from pytest_utils import import_or_fail

from physicsnemo.models.dlwp_healpix import HEALPixRecUNet

omegaconf = pytest.importorskip("omegaconf")


@pytest.fixture
def conv_next_block_dict(in_channels=3, out_channels=1):
    activation_block = {
        "_target_": "physicsnemo.models.layers.activations.CappedGELU",
        "cap_value": 10,
    }
    conv_block = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.ConvNeXtBlock",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "activation": activation_block,
        "kernel_size": 3,
        "dilation": 1,
        "upscale_factor": 4,
        "_recursive_": True,
    }
    return conv_block


@pytest.fixture
def down_sampling_block_dict():
    down_sampling_block = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.AvgPool",
        "pooling": 2,
    }
    return down_sampling_block


@pytest.fixture
def encoder_dict(conv_next_block_dict, down_sampling_block_dict, recurrent_block_dict):
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetEncoder",
        "conv_block": conv_next_block_dict,
        "down_sampling_block": down_sampling_block_dict,
        "recurrent_block": recurrent_block_dict,
        "_recursive_": False,
        "n_channels": [136, 68, 34],
        "dilations": [1, 2, 4],
    }
    return encoder


@pytest.fixture
def up_sampling_block_dict(in_channels=3, out_channels=1):
    """Block dict fixture."""
    activation_block = {
        "_target_": "physicsnemo.models.layers.activations.CappedGELU",
        "cap_value": 10,
    }
    up_sampling_block = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.TransposedConvUpsample",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "activation": activation_block,
        "upsampling": 2,
    }
    return omegaconf.DictConfig(up_sampling_block)


@pytest.fixture
def output_layer_dict(in_channels=3, out_channels=2):
    output_layer = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.BasicConvBlock",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }
    return omegaconf.DictConfig(output_layer)


@pytest.fixture
def recurrent_block_dict(in_channels=3):
    recurrent_block = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.ConvGRUBlock",
        "in_channels": in_channels,
        "kernel_size": 1,
        "_recursive_": False,
    }
    return omegaconf.DictConfig(recurrent_block)


@pytest.fixture
def decoder_dict(
    conv_next_block_dict,
    up_sampling_block_dict,
    output_layer_dict,
    recurrent_block_dict,
):
    decoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetDecoder",
        "conv_block": conv_next_block_dict,
        "up_sampling_block": up_sampling_block_dict,
        "recurrent_block": recurrent_block_dict,
        "output_layer": output_layer_dict,
        "_recursive_": False,
        "n_channels": [34, 68, 136],
        "dilations": [4, 2, 1],
    }
    return omegaconf.DictConfig(decoder)


@pytest.fixture
def test_data():
    # create dummy data
    def generate_test_data(
        batch_size=8, time_dim=1, channels=7, img_size=16, device="cpu"
    ):
        test_data = torch.randn(batch_size, 12, time_dim, channels, img_size, img_size)

        return test_data.to(device)

    return generate_test_data


@pytest.fixture
def constant_data():
    # create dummy data
    def generate_constant_data(channels=2, img_size=16, device="cpu"):
        constants = torch.randn(12, channels, img_size, img_size)

        return constants.to(device)

    return generate_constant_data


@pytest.fixture
def insolation_data():
    # create dummy data
    def generate_insolation_data(batch_size=8, time_dim=1, img_size=16, device="cpu"):
        insolation = torch.randn(batch_size, 12, time_dim, 1, img_size, img_size)

        return insolation.to(device)

    return generate_insolation_data


@import_or_fail("omegaconf")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixRecUNet_initialize(device, encoder_dict, decoder_dict, pytestconfig):
    in_channels = 3
    out_channels = 3
    n_constants = 1
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4

    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    ).to(device)
    assert isinstance(model, HEALPixRecUNet)

    # test fail case for bad input and output time dims
    with pytest.raises(
        ValueError, match=("'output_time_dim' must be a multiple of 'input_time_dim'")
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            n_constants=n_constants,
            decoder_input_channels=decoder_input_channels,
            input_time_dim=2,
            output_time_dim=3,
        ).to(device)

    # test fail case for couplings with no constants or decoder input channels
    with pytest.raises(
        NotImplementedError,
        match=("support for coupled models with no constant field"),
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=2,
            n_constants=0,
            couplings=["t2m", "v10m"],
        ).to(device)

    # test fail case for couplings with no decoder input channels
    with pytest.raises(
        NotImplementedError,
        match=("support for coupled models with no decoder inputs"),
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=0,
            n_constants=2,
            couplings=["t2m", "v10m"],
        ).to(device)

    with pytest.raises(
        NotImplementedError, match=("support for coupled models with no decoder")
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=0,
            n_constants=2,
            couplings=["t2m", "v10m"],
        ).to(device)

    with pytest.raises(
        NotImplementedError,
        match=("support for models with no constant fields and no decoder"),
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=0,
            n_constants=0,
        ).to(device)

    del model
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixRecUNet_integration_steps(
    device, encoder_dict, decoder_dict, pytestconfig
):
    in_channels = 2
    out_channels = 2
    n_constants = 1
    decoder_input_channels = 0
    input_time_dim = 2
    output_time_dim = 4

    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    ).to(device)

    assert model.integration_steps == output_time_dim // input_time_dim
    del model
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixRecUNet_reset(
    device,
    encoder_dict,
    decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    # create a smaller version of the dlwp healpix model
    in_channels = 2
    out_channels = 2
    n_constants = 2
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        time_dim=2 * input_time_dim, channels=in_channels, img_size=size, device=device
    )
    decoder_inputs = insolation_data(
        time_dim=2 * output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
        delta_time="6h",
    ).to(device)

    out_var = model(inputs)
    model.reset()

    assert common.compare_output(out_var, model(inputs))

    del model, inputs, out_var
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixRecUNet_forward(
    device,
    encoder_dict,
    decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    # create a smaller version of the dlwp healpix model
    in_channels = 2
    out_channels = 2
    n_constants = 2
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4
    batch_size = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=batch_size,
        time_dim=2 * input_time_dim,
        channels=in_channels,
        img_size=size,
        device=device,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size,
        time_dim=2 * output_time_dim,
        img_size=size,
        device=device,
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
        delta_time="6h",
        reset_cycle="6h",
    ).to(device)

    # one forward step to initialize recurrent states
    output = model(inputs)

    expected_shape = [batch_size, 12, output_time_dim, out_channels, size, size]
    assert list(output.shape) == expected_shape

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="dlwp_healpix.pth",
        rtol=1e-2,
    )

    output = model(inputs, output_only_last=True)
    expected_shape = [batch_size, 12, input_time_dim, out_channels, size, size]
    assert list(output.shape) == expected_shape

    # no decoder inputs
    inputs = [x, constants]
    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=0,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
        delta_time="6h",
    ).to(device)

    # one forward step to initialize recurrent states
    model(inputs)

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="dlwp_healpix_const.pth",
        rtol=1e-2,
    )

    # no constants
    inputs = [x, decoder_inputs]
    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=0,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
        delta_time="6h",
    ).to(device)

    # one forward step to initialize recurrent states
    model(inputs)

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="dlwp_healpix_decoder.pth",
        rtol=1e-2,
    )

    del model, inputs
    torch.cuda.empty_cache()


RESIDUAL_MODES = ("same-index-add", "last-reference", "chained")
PRESTEPS = 1


@pytest.fixture
def small_recunet_dicts(
    conv_next_block_dict,
    down_sampling_block_dict,
    up_sampling_block_dict,
    output_layer_dict,
    recurrent_block_dict,
):
    """Two-level UNet dicts, small enough for the residual tests to run on CPU."""
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetEncoder",
        "conv_block": conv_next_block_dict,
        "down_sampling_block": down_sampling_block_dict,
        "recurrent_block": recurrent_block_dict,
        "_recursive_": False,
        "n_channels": [16, 8],
        "dilations": [1, 2],
    }
    decoder = omegaconf.DictConfig(
        {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetDecoder",
            "conv_block": conv_next_block_dict,
            "up_sampling_block": up_sampling_block_dict,
            "recurrent_block": recurrent_block_dict,
            "output_layer": output_layer_dict,
            "_recursive_": False,
            "n_channels": [8, 16],
            "dilations": [2, 1],
        }
    )
    return encoder, decoder


def _build_residual_recunet(
    small_recunet_dicts,
    residual_prediction,
    in_channels=3,
    out_channels=5,
    input_time_dim=2,
    output_time_dim=4,
    size=16,
    device="cpu",
):
    """Build the two-level model with karlbauer padding so it runs without CUDA."""
    encoder, decoder = small_recunet_dicts
    model = HEALPixRecUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=2,
        decoder_input_channels=1,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        presteps=PRESTEPS,
        residual_prediction=residual_prediction,
        hpx_padding_mode="karlbauer",
        nside=(size, size // 2),
        delta_time="6h",
    ).to(device)
    model.eval()
    return model


def _expected_prognostics(mode, deltas, state):
    """Documented closed form for each mode, given raw deltas and the input state."""
    if mode == "same-index-add":
        return deltas + state
    if mode == "last-reference":
        return deltas + state[:, :, -1:]
    if mode == "chained":
        return state[:, :, -1:] + torch.cumsum(deltas, dim=2)
    raise ValueError(f"unhandled residual prediction mode: {mode}")


@import_or_fail("omegaconf")
@pytest.mark.parametrize("out_channels", [3, 5])
def test_HEALPixRecUNet_residual_prediction_closed_form(
    out_channels,
    small_recunet_dicts,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """Each mode must combine the raw deltas with the input state as documented.

    Run with out_channels == input_channels (prognostics only) and with extra
    diagnostic channels, which must stay absolute in every mode.
    """
    in_channels = 3
    input_time_dim = 2
    output_time_dim = 4
    batch_size = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=batch_size,
        time_dim=2 * input_time_dim,
        channels=in_channels,
        img_size=size,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size, time_dim=2 * output_time_dim, img_size=size
    )
    constants = constant_data(channels=2, img_size=size)
    inputs = [x, decoder_inputs, constants]

    model = _build_residual_recunet(
        small_recunet_dicts,
        "none",
        in_channels=in_channels,
        out_channels=out_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        size=size,
    )

    # The presteps prime the recurrent states, so step 0 reads its state from
    # the presteps-th block of input times rather than the first one.
    state = x[:, :, PRESTEPS * input_time_dim : (PRESTEPS + 1) * input_time_dim]

    # "none" returns the decoder output untouched, which is the deltas the other
    # modes build on. Only the first integration step can be compared across
    # modes, since later steps recycle mode-dependent state.
    with torch.no_grad():
        raw = model(inputs)
    assert raw.shape == (batch_size, 12, output_time_dim, out_channels, size, size)

    first_step = slice(0, input_time_dim)
    deltas = raw[:, :, first_step, :in_channels]
    diagnostics = raw[:, :, first_step, in_channels:]

    for mode in RESIDUAL_MODES:
        model.residual_prediction = mode
        with torch.no_grad():
            out = model(inputs)

        assert out.shape == raw.shape, mode
        assert torch.isfinite(out).all(), mode
        assert torch.allclose(
            out[:, :, first_step, :in_channels],
            _expected_prognostics(mode, deltas, state),
            atol=1e-6,
        ), f"prognostics do not match the closed form for {mode}"
        assert torch.allclose(
            out[:, :, first_step, in_channels:], diagnostics, atol=1e-6
        ), f"diagnostics are not absolute for {mode}"

    del inputs, model
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
def test_HEALPixRecUNet_residual_prediction_modes_differ(
    small_recunet_dicts,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """Guard against a mode silently falling through to no residual at all."""
    in_channels = 3
    input_time_dim = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=1, time_dim=2 * input_time_dim, channels=in_channels, img_size=size
    )
    decoder_inputs = insolation_data(
        batch_size=1, time_dim=2 * input_time_dim, img_size=size
    )
    constants = constant_data(channels=2, img_size=size)
    inputs = [x, decoder_inputs, constants]

    model = _build_residual_recunet(
        small_recunet_dicts,
        "none",
        in_channels=in_channels,
        out_channels=in_channels,
        input_time_dim=input_time_dim,
        output_time_dim=input_time_dim,
        size=size,
    )

    outputs = {}
    for mode in ("none",) + RESIDUAL_MODES:
        model.residual_prediction = mode
        with torch.no_grad():
            outputs[mode] = model(inputs)

    for mode in RESIDUAL_MODES:
        assert not torch.allclose(
            outputs[mode], outputs["none"]
        ), f"{mode} left the prediction unchanged"

    # same-index-add and last-reference only agree when input_time_dim == 1
    assert not torch.allclose(outputs["same-index-add"], outputs["last-reference"])
    assert not torch.allclose(outputs["chained"], outputs["last-reference"])

    del inputs, model
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
@pytest.mark.parametrize("residual_prediction", ("none",) + RESIDUAL_MODES)
def test_HEALPixRecUNet_residual_prediction_backward(
    residual_prediction,
    small_recunet_dicts,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """Every mode must stay differentiable with respect to the input state."""
    in_channels = 2
    input_time_dim = 2
    size = 8

    fix_random_seeds(seed=0)
    x = test_data(
        batch_size=1, time_dim=2 * input_time_dim, channels=in_channels, img_size=size
    ).requires_grad_(True)
    decoder_inputs = insolation_data(
        batch_size=1, time_dim=2 * input_time_dim, img_size=size
    )
    constants = constant_data(channels=2, img_size=size)

    model = _build_residual_recunet(
        small_recunet_dicts,
        residual_prediction,
        in_channels=in_channels,
        out_channels=in_channels + 1,
        input_time_dim=input_time_dim,
        output_time_dim=input_time_dim,
        size=size,
    )

    out = model([x, decoder_inputs, constants])
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    del model
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
def test_HEALPixRecUNet_residual_prediction_validation(
    small_recunet_dicts, pytestconfig
):
    """Unknown modes are rejected and legacy booleans map onto the named modes."""
    with pytest.raises(ValueError, match="Invalid residual prediction type"):
        _build_residual_recunet(small_recunet_dicts, "not-a-mode")

    # The UNet spelling for the last-reference mode was never valid here
    with pytest.raises(ValueError, match="Invalid residual prediction type"):
        _build_residual_recunet(small_recunet_dicts, "same-reference")

    # True must stay same-index-add: that is what the old boolean flag did here,
    # unlike HEALPixUNet where it mapped onto last-reference.
    assert (
        _build_residual_recunet(small_recunet_dicts, True).residual_prediction
        == "same-index-add"
    )
    assert (
        _build_residual_recunet(small_recunet_dicts, False).residual_prediction
        == "none"
    )

    torch.cuda.empty_cache()
