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


"""CPU correctness checks for shared deterministic ocean behavior."""

import copy

import pytest
import torch

from physicsnemo.models.dlwp_healpix.glorys_ocean import (
    FixedInverseValidationWeights,
    GlorysOceanModel,
    OceanStatePolicy,
    masked_channel_mse,
    masked_channel_sums,
)


def recunet_config(channels=3, forcing=3, nside=8):
    """Use actual atmosphere blocks at all three levels with tiny CPU widths."""
    blocks = "physicsnemo.models.dlwp_healpix_layers.healpix_blocks."
    activation = {
        "_target_": "physicsnemo.models.layers.activations.CappedGELU",
        "cap_value": 10,
    }
    conv = {
        "_target_": blocks + "Multi_SymmetricConvNeXtBlock",
        "kernel_size": 3,
        "upscale_factor": 2,
        "activation": activation,
    }
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_encoder.UNetEncoder",
        "_recursive_": False,
        "conv_block": copy.deepcopy(conv),
        "down_sampling_block": {
            "_target_": blocks + "DealiasedDownsample",
            "resample_filter": [1, 2, 1],
            "stride": 2,
        },
        "recurrent_block": None,
        "n_channels": [4, 4, 4],
        "n_layers": [1, 1, 1],
        "dilations": [1, 1, 1],
    }
    decoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_decoder.UNetDecoder",
        "_recursive_": False,
        "conv_block": copy.deepcopy(conv),
        "up_sampling_block": {
            "_target_": blocks + "SmoothedInterpolateConv",
            "kernel_size": 3,
            "scale_factor": 2,
            "mode": "nearest",
            "activation": copy.deepcopy(activation),
        },
        "recurrent_block": {"_target_": blocks + "ConvGRUBlock", "kernel_size": 1},
        "output_layer": {
            "_target_": blocks + "BasicConvBlock",
            "kernel_size": 1,
            "n_layers": 1,
            "activation": None,
        },
        "n_channels": [4, 4, 4],
        "n_layers": [1, 1, 1],
        "dilations": [1, 1, 1],
    }
    return {
        "_target_": "physicsnemo.models.dlwp_healpix.HEALPixRecUNet.HEALPixRecUNet",
        "_recursive_": False,
        "encoder": encoder,
        "decoder": decoder,
        "input_channels": channels,
        "output_channels": channels,
        "n_constants": 1,
        "decoder_input_channels": forcing,
        "input_time_dim": 2,
        "output_time_dim": 2,
        "delta_time": "96h",
        "reset_cycle": "192h",
        "presteps": 0,
        "enable_nhwc": False,
        "compile_padding": False,
        "hpx_padding_mode": "karlbauer",
        "nside": [nside, nside // 2, nside // 4],
        "couplings": [],
        "constraints": None,
        "residual_prediction": True,
    }


def build_model(channels, forcing, mean, std, mask, **kwargs):
    """Inject the complete native node into the scientific wrapper."""
    return GlorysOceanModel(
        channels,
        forcing,
        mean,
        std,
        mask,
        backbone_config=recunet_config(len(channels), len(forcing), mask.shape[-1]),
        **kwargs,
    )


def output_convolution(model):
    """Locate the existing decoder's actual output convolution for probes."""
    return next(
        layer
        for layer in model.backbone.decoder.output_layer.modules()
        if isinstance(layer, torch.nn.Conv2d)
    )


def setup_model():
    """Construct a small CPU block with channel-specific land domains."""
    torch.manual_seed(12)
    channels = ["thetao_0", "sic", "mlotst"]
    mask = torch.ones(3, 12, 8, 8, dtype=torch.bool)
    mask[:, :, 0, 0] = False
    mask[0, :, 0, 1] = False
    model = build_model(
        channels,
        ["z", "wind", "radiation"],
        [10, 0.5, 5],
        [2, 0.25, 2],
        mask,
    )
    return model, mask


def test_native_corresponding_history_residual_and_constraints():
    """Native zero increments return matching histories, each advancing 192h."""
    from physicsnemo.models.dlwp_healpix.HEALPixRecUNet import HEALPixRecUNet

    model, mask = setup_model()
    assert type(model.backbone) is HEALPixRecUNet
    assert model.backbone.presteps == 0
    assert model.backbone.delta_t == 96
    assert model.backbone.residual_prediction is True
    assert model.backbone.integration_steps == 1
    history = torch.zeros(1, 2, 3, 12, 8, 8)
    history[:, 0, 0] = -7
    history[:, 1, 0] = 3
    convolution = output_convolution(model)
    with torch.no_grad():
        convolution.weight.zero_()
        convolution.bias.zero_()
    output = model(history, torch.zeros_like(history))
    assert torch.all(output[:, 0, 0][..., mask[0]] == -7)
    assert torch.all(output[:, 1, 0][..., mask[0]] == 3)
    with torch.no_grad():
        convolution.bias.copy_(torch.tensor([1, 100, -100, 2, -100, -100.0]))
    output = model(history, torch.zeros_like(history))
    assert torch.all(output[:, 0, 0][..., mask[0]] == -6)
    assert torch.all(output[:, 1, 0][..., mask[0]] == 5)
    physical = model.state_policy.to_physical(output)
    assert torch.all(physical[:, 0, 1][..., mask[1]] == 1)
    assert torch.all(physical[:, 1, 1][..., mask[1]] == 0)
    assert torch.all(physical[:, :, 2][..., mask[2]] == 0)
    assert torch.all(output[..., ~mask] == 0)


def test_history_feedback_land_invariance_and_checkpoint():
    """Invalid histories cannot affect coastal outputs over repeated feedback."""
    model, mask = setup_model()
    with torch.no_grad():
        output_convolution(model).weight.normal_(0, 0.1)
    history = torch.randn(2, 2, 3, 12, 8, 8)
    forcing = torch.randn_like(history)
    contaminated = history.clone()
    contaminated[..., ~mask] = float("nan")
    for _ in range(3):
        clean_output = model(history, forcing)
        output = model(contaminated, forcing)
        torch.testing.assert_close(output, clean_output, rtol=0, atol=0)
        assert torch.isfinite(output).all()
        assert torch.all(output[..., ~mask] == 0)
        physical = model.state_policy.to_physical(output)
        assert (
            (physical[:, :, 1][..., mask[1]] >= 0)
            & (physical[:, :, 1][..., mask[1]] <= 1)
        ).all()
        assert (physical[:, :, 2][..., mask[2]] >= 0).all()
        history = clean_output
        contaminated = output.detach().clone()
        contaminated[..., ~mask] = 1e10
    restored, _ = setup_model()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored(history, forcing), model(history, forcing))


def test_face_exchange_affects_neighbor_output():
    """A single-face impulse reaches another face through the existing HPX stem."""
    model, _ = setup_model()
    from physicsnemo.models.dlwp_healpix_layers.healpix_layers import HEALPixLayer

    layer = next(
        layer
        for layer in model.backbone.encoder.modules()
        if isinstance(layer, HEALPixLayer)
        and isinstance(layer.layers[-1], torch.nn.Conv2d)
        and layer.layers[-1].kernel_size == (3, 3)
    )
    with torch.no_grad():
        convolution = layer.layers[-1]
        convolution.weight.fill_(1)
        convolution.bias.zero_()
    features = torch.zeros(12, layer.layers[-1].in_channels, 8, 8)
    features[0, 0] = 1
    output = layer(features)
    assert output[1:].abs().sum() > 0


def test_mask_sums_dynamic_targets_gradient_and_aggregation():
    """Mask invalid targets before arithmetic and aggregate sums before ratios."""
    target = torch.zeros(2, 2, 2, 12, 2, 2)
    prediction = torch.full_like(target, 2, requires_grad=True)
    validity = torch.ones_like(target, dtype=torch.bool)
    validity[0, :, 1] = False
    target[0, :, 1] = float("nan")
    target[1, 0, 0, 0, 0, 0] = float("inf")
    with torch.no_grad():
        prediction[0, :, 1] = float("nan")
    numerator, denominator = masked_channel_sums(prediction, target, validity)
    torch.testing.assert_close(denominator, torch.tensor([191.0, 96.0]))
    torch.testing.assert_close(numerator, denominator * 4)
    numerator.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[0, :, 1] == 0).all()
    sums = [
        masked_channel_sums(
            prediction[i : i + 1], target[i : i + 1], validity[i : i + 1]
        )
        for i in range(2)
    ]
    torch.testing.assert_close(sum(item[0] for item in sums), numerator)
    torch.testing.assert_close(sum(item[1] for item in sums), denominator)
    torch.testing.assert_close(
        masked_channel_mse(prediction, target, validity), torch.tensor([4.0, 4.0])
    )
    torch.testing.assert_close(
        masked_channel_mse(prediction, target, torch.zeros_like(validity)),
        torch.zeros(2),
    )


def test_fractional_denominator_and_half_accumulation():
    """Fractional masks divide by actual weight and half reductions stay finite."""
    prediction = torch.tensor([[[[[[2.0]]]]]])
    target = torch.zeros_like(prediction)
    torch.testing.assert_close(
        masked_channel_mse(prediction, target, 0.25), torch.tensor([4.0])
    )
    prediction = torch.ones(2, 2, 1, 12, 64, 64, dtype=torch.float16)
    numerator, denominator = masked_channel_sums(
        prediction, torch.zeros_like(prediction), 1
    )
    assert numerator.dtype == torch.float32
    assert denominator.item() == 196608
    assert torch.isfinite(numerator).all()


def test_balancing_boundary_manual_and_resume():
    """Weights freeze once at completed epoch ten and restore exactly."""
    schedule = FixedInverseValidationWeights(
        ["sst", "sic"], manual_multipliers={"sic": 3}
    )
    assert not schedule.freeze(9, [1, 4])
    torch.testing.assert_close(schedule.weights, torch.ones(2))
    assert schedule.phase == "equal"
    assert schedule.freeze(10, [1, 4])
    torch.testing.assert_close(schedule.weights, torch.tensor([1.6, 1.2]))
    restored = FixedInverseValidationWeights(
        ["sst", "sic"], manual_multipliers={"sic": 3}
    )
    restored.load_state_dict(copy.deepcopy(schedule.state_dict()))
    assert restored.phase == "fixed"
    assert restored.calibration_epoch.item() == 10
    assert not restored.freeze(20, [4, 1])
    torch.testing.assert_close(restored.weights, schedule.weights)
    torch.testing.assert_close(restored.reference_mse, torch.tensor([1.0, 4.0]))
    assert restored(torch.ones(2)).item() == pytest.approx(1.4)


def test_balancing_safeguards_and_config_validation():
    """Reject invalid references/settings and mismatched resume configurations."""
    schedule = FixedInverseValidationWeights(
        ["sst", "sic"], equal_epochs=1, max_weight=1.5
    )
    for invalid in ([float("nan"), 1], [-1, 1], [1]):
        with pytest.raises(ValueError):
            schedule.freeze(1, invalid)
    assert schedule.freeze(1, [0, 1])
    assert schedule.weights.max() <= 1.5
    assert torch.isfinite(schedule.weights).all()
    incompatible = FixedInverseValidationWeights(
        ["sic", "sst"], equal_epochs=1, max_weight=1.5
    )
    with pytest.raises(ValueError):
        incompatible.load_state_dict(schedule.state_dict())
    different_manual = FixedInverseValidationWeights(
        ["sst", "sic"], equal_epochs=1, max_weight=1.5, manual_multipliers={"sic": 2}
    )
    with pytest.raises(ValueError):
        different_manual.load_state_dict(schedule.state_dict())
    with pytest.raises(ValueError):
        FixedInverseValidationWeights(["sst"], manual_multipliers={"bad": 2})


def test_full_channel_contract_backward_and_policy_validation():
    """All 29 targets and 3 forcing channels support CPU backpropagation."""
    channels = (
        [f"thetao_{i}" for i in range(12)]
        + [f"so_{i}" for i in range(12)]
        + ["sic", "sithick", "zos", "sst", "mlotst"]
    )
    mask = torch.ones(29, 12, 8, 8, dtype=torch.bool)
    model = build_model(
        channels,
        ["z", "wind", "rad"],
        torch.zeros(29),
        torch.ones(29),
        mask,
    )
    history = torch.zeros(1, 2, 29, 12, 8, 8)
    forcing = torch.zeros(1, 2, 3, 12, 8, 8)
    output = model(history, forcing)
    assert output.shape == history.shape
    masked_channel_mse(output, torch.ones_like(output), mask).mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    with pytest.raises(ValueError):
        OceanStatePolicy(["sst"], [0], [0], mask[:1])
    with pytest.raises(ValueError):
        model(history[:, :1], forcing)
    with pytest.raises(ValueError):
        model(history, forcing[:, :1])


def test_lazy_exports_preserve_layer_api_without_config_dependencies():
    """Retain every original export while core HPX layers avoid Hydra imports."""
    import physicsnemo.models.dlwp_healpix as models
    import physicsnemo.models.dlwp_healpix_layers as layers

    expected = {
        "AvgPool",
        "BasicConvBlock",
        "ConvGRUBlock",
        "ConvNeXtBlock",
        "DealiasedDownsample",
        "DoubleConvNeXtBlock",
        "Interpolate",
        "MaxPool",
        "Multi_SymmetricConvNeXtBlock",
        "SymmetricConvNeXtBlock",
        "TransposedConvUpsample",
        "SmoothedInterpolateConv",
        "UNetDecoder",
        "UNetEncoder",
        "HEALPixLayer",
        "HEALPixFoldFaces",
        "HEALPixPadding",
        "HEALPixPaddingIsolatitude",
        "HEALPixPaddingv2",
        "HEALPixUnfoldFaces",
        "have_earth2grid",
        "make_hpx_padding_layer",
        "warn_deprecated_enable_healpixpad",
    }
    assert set(layers.__all__) == expected
    assert {"HEALPixRecUNet", "HEALPixUNet"} <= set(models.__all__)
    assert layers.HEALPixLayer.__module__.endswith("healpix_layers")
    assert models.GlorysOceanModel is GlorysOceanModel
    # User-installed Hydra may be loaded by pytest's own plugin; the model
    # package must still avoid loading the optional legacy implementation.
    with pytest.raises(AttributeError):
        getattr(layers, "missing_export")
    with pytest.raises(AttributeError):
        getattr(models, "missing_export")


def test_checkpoint_weights_only_and_hpx64_contract():
    """State_dict checkpoints preserve mask/scaling and all HPX64 state outputs."""
    import io

    model, _ = setup_model()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored, _ = setup_model()
    restored.load_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
    torch.testing.assert_close(restored.state_policy.mask, model.state_policy.mask)
    torch.testing.assert_close(restored.state_policy.mean, model.state_policy.mean)
    torch.testing.assert_close(restored.state_policy.std, model.state_policy.std)
    channels = (
        [f"thetao_{i}" for i in range(12)]
        + [f"so_{i}" for i in range(12)]
        + ["sic", "sithick", "zos", "sst", "mlotst"]
    )
    model = build_model(
        channels,
        ["z", "wind", "rad"],
        torch.zeros(29),
        torch.ones(29),
        torch.ones(29, 12, 64, 64, dtype=torch.bool),
    )
    with torch.no_grad():
        output = model(
            torch.zeros(1, 2, 29, 12, 64, 64), torch.zeros(1, 2, 3, 12, 64, 64)
        )
    assert output.shape == (1, 2, 29, 12, 64, 64)
    assert torch.isfinite(output).all()


def test_predicted_nonfinite_ocean_is_rejected_before_sanitization():
    """Native predictions raise on ocean while invalid land may be erased."""
    model, mask = setup_model()
    history = torch.zeros(1, 2, 3, 12, 8, 8)
    history[0, 0, 0, 0, 1, 1] = float("nan")
    assert torch.isfinite(model.state_policy(history)).all()

    def contaminate_land(module, inputs, output):
        output = output.clone()
        output.permute(0, 2, 3, 1, 4, 5)[..., ~mask] = float("nan")
        return output

    hook = model.backbone.register_forward_hook(contaminate_land)
    assert torch.isfinite(model(history, torch.zeros_like(history))).all()
    hook.remove()
    with torch.no_grad():
        output_convolution(model).bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        model(history, torch.zeros_like(history))


def test_sst_absolute_infill_gradients_feedback_and_checkpoint():
    """Shared SST infill preserves ocean, rejects raw land influence and differentiates."""
    mask = torch.ones(1, 12, 8, 8, dtype=torch.bool)
    mask[0, 0, :2, :2] = False
    policy = OceanStatePolicy(["sst"], [280], [4], mask, sst_nlat=33, sst_nlon=64)
    history = torch.randn(1, 2, 1, 12, 8, 8, requires_grad=True)
    output = policy(history)
    torch.testing.assert_close(output[..., mask], history[..., mask], rtol=0, atol=0)
    assert torch.isfinite(output).all()
    contaminated = history.detach().clone()
    contaminated[..., ~mask] = float("nan")
    torch.testing.assert_close(policy(contaminated), output, rtol=0, atol=0)
    output[..., ~mask].sum().backward()
    assert torch.isfinite(history.grad).all()
    assert history.grad[..., mask].abs().sum() > 0
    assert history.grad[..., ~mask].abs().sum() == 0
    model = build_model(
        ["sst"],
        ["z"],
        [280],
        [4],
        mask,
        sst_nlat=33,
        sst_nlon=64,
    )
    with torch.no_grad():
        output_convolution(model).weight.normal_(0, 0.1)
    restored = build_model(
        ["sst"],
        ["z"],
        [280],
        [4],
        mask,
        sst_nlat=33,
        sst_nlon=64,
    )
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    state = history.detach()
    forcing = torch.ones_like(state)
    for _ in range(3):
        state = model(state, forcing)
        torch.testing.assert_close(state, restored(history.detach(), forcing))
        history = state
        assert torch.isfinite(state).all()
    keys = model.state_dict()
    assert any("sst_interpolator.regrid" in key for key in keys)
    assert any("sst_interpolator.passes.0.lower" in key for key in keys)


def test_sst_linear_passes_match_original_xarray_sequence():
    """Torch stencils match original coordinate-free xarray interpolation passes."""
    import xarray as xr

    from physicsnemo.models.dlwp_healpix.glorys_ocean import TripleInterpSST

    mask = torch.ones(12, 8, 8, dtype=torch.bool)
    mask[0, :2, :2] = False
    filler = TripleInterpSST(mask, nlat=33, nlon=64)
    ocean = torch.arange(mask.numel(), dtype=torch.float64).reshape_as(mask)
    original = torch.where(mask, ocean, float("nan"))
    raw = filler.regrid(original.reshape(-1)).detach().numpy()
    reference = xr.DataArray(raw, dims=["latitude", "longitude"])
    reference = reference.interpolate_na("longitude", use_coordinate=False)
    reference = (
        reference.roll(longitude=32, roll_coords=False)
        .interpolate_na("longitude", use_coordinate=False)
        .roll(longitude=32, roll_coords=False)
    )
    reference = (
        reference.roll(latitude=16, roll_coords=False)
        .interpolate_na("latitude", use_coordinate=False)
        .roll(latitude=17, roll_coords=False)
    )
    expected = filler.regrid_back(torch.as_tensor(reference.values)).reshape_as(ocean)
    expected = torch.where(mask, ocean, expected)
    torch.testing.assert_close(filler(ocean), expected, rtol=1e-12, atol=1e-12)


def test_actual_stats_strict_bounds_and_low_precision_policy():
    """Adversarial real scaling satisfies float32 and float64 strict decoders."""
    # Actual stats-train-1993-2018.npz persisted float32 scaling for SIC and MLD.
    mean = torch.tensor([0.05289500951766968, 43.130428314208984])
    std = torch.tensor([0.20912887156009674, 51.219581604003906])
    mask = torch.ones(2, 12, 8, 8, dtype=torch.bool)
    policy = OceanStatePolicy(["sic", "mlotst"], mean, std, mask)
    torch.manual_seed(47)
    states = torch.randn(8, 2, 2, 12, 8, 8) * 1e6
    states[0, 0] = -1e30
    states[0, 1] = 1e30
    for dtype in (torch.float32, torch.bfloat16):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = policy(states.to(dtype), strict=True)
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()
        for decode_dtype in (torch.float32, torch.float64):
            physical = output.to(decode_dtype) * policy.std.to(
                decode_dtype
            ) + policy.mean.to(decode_dtype)
            assert (physical[:, :, 0] >= 0).all()
            assert (physical[:, :, 0] <= 1).all()
            assert (physical[:, :, 1] >= 0).all()
    output = policy(states, strict=True)
    torch.testing.assert_close(policy(output, strict=True), output, rtol=0, atol=0)
    previous = copy.deepcopy(policy.state_dict())
    previous["_extra_state"]["policy"] = "sst-triple-interp-other-neutral-v2"
    with pytest.raises(ValueError):
        policy.load_state_dict(previous)


def test_strict_bounds_actual_scaling_native_model_and_gradients():
    """Native residual feedback obeys strict bounds with real scaling."""
    mean = [0.05289500951766968, 43.130428314208984]
    std = [0.20912887156009674, 51.219581604003906]
    mask = torch.ones(2, 12, 8, 8, dtype=torch.bool)
    model = build_model(["sic", "mlotst"], ["z"], mean, std, mask)
    history = torch.zeros(4, 2, 2, 12, 8, 8)
    torch.manual_seed(18)
    for _ in range(3):
        with torch.no_grad():
            output_convolution(model).bias.normal_(0, 100)
        output = model(history, torch.zeros(4, 2, 1, 12, 8, 8))
        physical = (
            output.double() * model.state_policy.std.double()
            + model.state_policy.mean.double()
        )
        assert (physical[:, :, 0] >= 0).all() and (physical[:, :, 0] <= 1).all()
        assert (physical[:, :, 1] >= 0).all()
        output.sum().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        model.zero_grad()
        history = output.detach()


def test_module_archive_json_extra_state_and_feedback_parity(tmp_path):
    """Standard Module archives preserve all scientific keys and feedback outputs."""
    import json

    from physicsnemo.models.module import Module

    mask = torch.ones(3, 12, 8, 8, dtype=torch.bool)
    mask[:, 0, :2, :2] = False
    model = build_model(
        ["sic", "sst", "mlotst"],
        ["z"],
        torch.tensor([0.05, 280, 43]),
        torch.tensor([0.21, 4, 51]),
        mask,
        sst_nlat=33,
        sst_nlon=64,
    )
    assert isinstance(model, Module)
    assert model.device == torch.device("cpu")
    assert "device_buffer" not in model.state_dict()
    assert not any(key.startswith("inner_model.") for key in model.state_dict())
    original_keys = set(model.state_dict())
    with torch.no_grad():
        output_convolution(model).weight.normal_(0, 0.03)
        model.state_policy.mean[:, :, 0].add_(0.001)
    args = json.loads(json.dumps(model.checkpoint_args()))
    assert args["__args__"]["mean"] == model.state_policy.mean.flatten().tolist()
    archive = tmp_path / "ocean.mdlus"
    model.save(str(archive))
    restored = Module.from_checkpoint(str(archive))
    assert set(restored.state_dict()) == original_keys
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(
                restored.state_dict()[key], value, rtol=0, atol=0
            )
        else:
            assert restored.state_dict()[key] == value
    assert restored.checkpoint_args() == args
    missing_stencil = copy.deepcopy(restored.state_dict())
    missing_stencil.pop("state_policy.sst_interpolator.passes.0.lower")
    with pytest.raises(RuntimeError, match="state keys differ"):
        restored.load_state_dict(missing_stencil, strict=False)
    history = torch.randn(1, 2, 3, 12, 8, 8)
    forcing = torch.randn(1, 2, 1, 12, 8, 8)
    for _ in range(3):
        expected = model(history, forcing)
        actual = restored(history, forcing)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        history = expected.detach()


def test_module_loader_rejects_missing_keys_even_with_nonstrict_request(tmp_path):
    """Generic strict=False archive loading cannot omit scientific state or policy."""
    import io
    import json
    import tarfile

    from physicsnemo.models.module import Module

    model, _ = setup_model()
    state = model.state_dict()
    for missing in (
        "_extra_state",
        "state_policy._extra_state",
        "state_policy.mask",
        "state_policy.mean",
        next(
            key
            for key in state
            if key.endswith("weight") and key.startswith("backbone.")
        ),
    ):
        invalid = copy.deepcopy(state)
        invalid.pop(missing)
        with pytest.raises(RuntimeError, match="state keys differ"):
            model.load_state_dict(invalid, strict=False)
    invalid = copy.deepcopy(state)
    invalid["state_policy._extra_state"]["policy"] = "unsupported-policy"
    with pytest.raises(ValueError, match="policy/channel ordering differs"):
        model.load_state_dict(invalid, strict=False)
    invalid = copy.deepcopy(state)
    invalid.pop("state_policy.mask")
    buffer = io.BytesIO()
    torch.save(invalid, buffer)
    payloads = {
        "model.pt": buffer.getvalue(),
        "args.json": json.dumps(model.checkpoint_args()).encode(),
        "metadata.json": json.dumps(
            {"mdlus_file_version": Module.__model_checkpoint_version__}
        ).encode(),
    }
    archive = tmp_path / "missing-mask.mdlus"
    with tarfile.open(archive, "w") as tar:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="state keys differ"):
        Module.from_checkpoint(str(archive))


def test_native_recurrence_resets_and_inputs_preserve_forcing_alignment():
    """All actual decoder GRUs execute once, reset per call, and receive both windows."""
    from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import ConvGRUBlock

    model, _ = setup_model()
    history = torch.randn(1, 2, 3, 12, 8, 8)
    forcing = torch.randn_like(history)
    seen = []
    hook = model.backbone.register_forward_pre_hook(
        lambda module, args: seen.append(args[0])
    )
    first = model(history, forcing)
    hook.remove()
    torch.testing.assert_close(
        seen[0][0], model.state_policy(history).permute(0, 3, 1, 2, 4, 5)
    )
    torch.testing.assert_close(seen[0][1], forcing.permute(0, 3, 1, 2, 4, 5))
    torch.testing.assert_close(seen[0][2], model.land_constant[0])
    recurrent = [
        layer for layer in model.backbone.modules() if isinstance(layer, ConvGRUBlock)
    ]
    assert len(recurrent) == 3
    assert all(layer.h.abs().sum() > 0 for layer in recurrent)
    for layer in recurrent:
        layer.h = torch.full_like(layer.h, 1e9)
    second = model(history, forcing)
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_ocean_interface_rejects_changed_timing_channels_and_mask_grid():
    """Reject incompatible sampling interfaces without validating native architecture."""
    model, mask = setup_model()
    for key, value in (
        ("presteps", 1),
        ("residual_prediction", False),
        ("delta_time", "6h"),
        ("n_constants", 0),
        ("input_channels", 4),
        ("output_channels", 4),
        ("decoder_input_channels", 2),
        ("nside", [16, 8, 4]),
    ):
        config = copy.deepcopy(model.backbone_config)
        config[key] = value
        with pytest.raises(ValueError):
            GlorysOceanModel(
                model.channels,
                model.forcing_channels,
                [10, 0.5, 5],
                [2, 0.25, 2],
                mask,
                config,
            )
    bad = copy.deepcopy(model.state_dict())
    bad["_extra_state"]["reference"] = "latest-history"
    with pytest.raises(ValueError):
        model.load_state_dict(bad)


def test_layer_public_exports_and_torch_only_activation_import():
    """Preserve every prior public layer name and avoid optional timm on activation import."""
    import sys

    import physicsnemo.models.layers as layers

    expected = {
        "WeightNormLinear",
        "DecoderLayer",
        "Conv1dFCLayer",
        "ConvNdKernel1Layer",
        "UpSample2D",
        "ConvBlock",
        "SpectralConv1d",
        "SquarePlus",
        "GaborFilter",
        "Conv2dFCLayer",
        "FCLayer",
        "Stan",
        "SpectralConv4d",
        "SwinTransformer",
        "EncoderLayer",
        "get_activation",
        "FuserLayer",
        "DownSample2D",
        "CappedGELU",
        "SirenLayerType",
        "UpSample3D",
        "ConvNdFCLayer",
        "DownSample3D",
        "WeightFactLinear",
        "SirenLayer",
        "Identity",
        "DGMLayer",
        "CubeEmbedding",
        "Mlp",
        "FourierFilter",
        "CappedLeakyReLU",
        "Conv3dFCLayer",
        "SpectralConv2d",
        "FourierLayer",
        "SpectralConv3d",
    }
    assert set(layers.__all__) == expected
    assert expected <= set(dir(layers))
    from physicsnemo.models.layers import CappedGELU
    from physicsnemo.models.layers.activations import CappedGELU as direct

    assert CappedGELU is direct
    assert "physicsnemo.models.layers.transformer_layers" not in sys.modules
    with pytest.raises(AttributeError):
        getattr(layers, "missing_export")


def test_native_architecture_and_nhwc_options_are_consumed():
    """Let native modules execute configured families, layout and reset options."""
    config = recunet_config()
    config["enable_nhwc"] = True
    config["reset_cycle"] = "384h"
    for side in ("encoder", "decoder"):
        config[side]["conv_block"] = {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.BasicConvBlock",
            "kernel_size": 3,
            "activation": {
                "_target_": "physicsnemo.models.layers.activations.CappedGELU",
                "cap_value": 10,
            },
        }
    model = GlorysOceanModel(
        ["thetao_0", "sic", "mlotst"],
        ["z", "wind", "radiation"],
        [10, 0.5, 5],
        [2, 0.25, 2],
        torch.ones(3, 12, 8, 8),
        config,
    )
    assert model.backbone.enable_nhwc is True
    assert model.backbone.reset_cycle == 384
    assert model.backbone.encoder.encoder[0][-1].__class__.__name__ == "BasicConvBlock"
    assert model.backbone_config == config
    output = model(torch.zeros(1, 2, 3, 12, 8, 8), torch.zeros(1, 2, 3, 12, 8, 8))
    assert output.shape == (1, 2, 3, 12, 8, 8)
    assert torch.isfinite(output).all()


def test_native_padding_and_level_errors_are_delegated():
    """Unsupported native settings fail in the shared constructor, not ocean guards."""
    from hydra.errors import InstantiationException

    mask = torch.ones(3, 12, 8, 8)
    for key, value in (("hpx_padding_mode", "unsupported"), ("nside", [8, 4])):
        config = recunet_config()
        config[key] = value
        with pytest.raises(InstantiationException):
            GlorysOceanModel(
                ["thetao_0", "sic", "mlotst"],
                ["z", "wind", "radiation"],
                [10, 0.5, 5],
                [2, 0.25, 2],
                mask,
                config,
            )
