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


def setup_model():
    """Construct a small CPU block with channel-specific land domains."""
    torch.manual_seed(12)
    channels = ["thetao_0", "sic", "mlotst"]
    mask = torch.ones(3, 12, 4, 4, dtype=torch.bool)
    mask[:, :, 0, 0] = False
    mask[0, :, 0, 1] = False
    model = GlorysOceanModel(channels, ["z", "wind", "radiation"],
                            [10, 0.5, 5], [2, 0.25, 2], mask,
                            hidden_channels=4, depth=1)
    return model, mask


def test_latest_history_residual_and_constraints():
    """Both leads reference t and enforce physical bounds after reconstruction."""
    model, mask = setup_model()
    history = torch.zeros(1, 2, 3, 12, 4, 4)
    history[:, 0, 0] = -7
    history[:, 1, 0] = 3
    residual = torch.zeros_like(history)
    residual[:, 0, 0] = 1
    residual[:, 1, 0] = 2
    residual[:, 0, 1] = 100
    residual[:, 1, 1] = -100
    residual[:, :, 2] = -100
    output = model.reconstruct(history, residual)
    assert torch.all(output[:, 0, 0][..., mask[0]] == 4)
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
        model.head.weight.normal_(0, 0.1)
    history = torch.randn(2, 2, 3, 12, 4, 4)
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
        assert ((physical[:, :, 1][..., mask[1]] >= 0) &
                (physical[:, :, 1][..., mask[1]] <= 1)).all()
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
    layer = model.stem
    with torch.no_grad():
        convolution = layer.layers[-1]
        convolution.weight.fill_(1)
        convolution.bias.zero_()
    features = torch.zeros(12, 13, 4, 4)
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
    torch.testing.assert_close(denominator, torch.tensor([191., 96.]))
    torch.testing.assert_close(numerator, denominator * 4)
    numerator.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[0, :, 1] == 0).all()
    sums = [masked_channel_sums(prediction[i:i+1], target[i:i+1], validity[i:i+1])
            for i in range(2)]
    torch.testing.assert_close(sum(item[0] for item in sums), numerator)
    torch.testing.assert_close(sum(item[1] for item in sums), denominator)
    torch.testing.assert_close(masked_channel_mse(prediction, target, validity), torch.tensor([4., 4.]))
    torch.testing.assert_close(masked_channel_mse(prediction, target, torch.zeros_like(validity)), torch.zeros(2))


def test_fractional_denominator_and_half_accumulation():
    """Fractional masks divide by actual weight and half reductions stay finite."""
    prediction = torch.tensor([[[[[[2.]]]]]])
    target = torch.zeros_like(prediction)
    torch.testing.assert_close(masked_channel_mse(prediction, target, 0.25), torch.tensor([4.]))
    prediction = torch.ones(2, 2, 1, 12, 64, 64, dtype=torch.float16)
    numerator, denominator = masked_channel_sums(prediction, torch.zeros_like(prediction), 1)
    assert numerator.dtype == torch.float32
    assert denominator.item() == 196608
    assert torch.isfinite(numerator).all()


def test_balancing_boundary_manual_and_resume():
    """Weights freeze once at completed epoch ten and restore exactly."""
    schedule = FixedInverseValidationWeights(["sst", "sic"], manual_multipliers={"sic": 3})
    assert not schedule.freeze(9, [1, 4])
    torch.testing.assert_close(schedule.weights, torch.ones(2))
    assert schedule.phase == "equal"
    assert schedule.freeze(10, [1, 4])
    torch.testing.assert_close(schedule.weights, torch.tensor([1.6, 1.2]))
    restored = FixedInverseValidationWeights(["sst", "sic"], manual_multipliers={"sic": 3})
    restored.load_state_dict(copy.deepcopy(schedule.state_dict()))
    assert restored.phase == "fixed"
    assert restored.calibration_epoch.item() == 10
    assert not restored.freeze(20, [4, 1])
    torch.testing.assert_close(restored.weights, schedule.weights)
    torch.testing.assert_close(restored.reference_mse, torch.tensor([1., 4.]))
    assert restored(torch.ones(2)).item() == pytest.approx(1.4)


def test_balancing_safeguards_and_config_validation():
    """Reject invalid references/settings and mismatched resume configurations."""
    schedule = FixedInverseValidationWeights(["sst", "sic"], equal_epochs=1, max_weight=1.5)
    for invalid in ([float("nan"), 1], [-1, 1], [1]):
        with pytest.raises(ValueError):
            schedule.freeze(1, invalid)
    assert schedule.freeze(1, [0, 1])
    assert schedule.weights.max() <= 1.5
    assert torch.isfinite(schedule.weights).all()
    incompatible = FixedInverseValidationWeights(["sic", "sst"], equal_epochs=1, max_weight=1.5)
    with pytest.raises(ValueError):
        incompatible.load_state_dict(schedule.state_dict())
    different_manual = FixedInverseValidationWeights(["sst", "sic"], equal_epochs=1, max_weight=1.5,
                                                      manual_multipliers={"sic": 2})
    with pytest.raises(ValueError):
        different_manual.load_state_dict(schedule.state_dict())
    with pytest.raises(ValueError):
        FixedInverseValidationWeights(["sst"], manual_multipliers={"bad": 2})


def test_full_channel_contract_backward_and_policy_validation():
    """All 29 targets and 3 forcing channels support CPU backpropagation."""
    channels = [f"thetao_{i}" for i in range(12)] + [f"so_{i}" for i in range(12)] + ["sic", "sithick", "zos", "sst", "mlotst"]
    mask = torch.ones(29, 12, 4, 4, dtype=torch.bool)
    model = GlorysOceanModel(channels, ["z", "wind", "rad"], torch.zeros(29), torch.ones(29), mask,
                            hidden_channels=4, depth=1)
    history = torch.zeros(1, 2, 29, 12, 4, 4)
    forcing = torch.zeros(1, 2, 3, 12, 4, 4)
    output = model(history, forcing)
    assert output.shape == history.shape
    masked_channel_mse(output, torch.ones_like(output), mask).mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    with pytest.raises(ValueError):
        OceanStatePolicy(["sst"], [0], [0], mask[:1])
    with pytest.raises(ValueError):
        model(history[:, :1], forcing)
    with pytest.raises(ValueError):
        model(history, forcing[:, :1])


def test_lazy_exports_preserve_layer_api_without_config_dependencies():
    """Retain every original export while core HPX layers avoid Hydra imports."""
    import sys

    import physicsnemo.models.dlwp_healpix as models
    import physicsnemo.models.dlwp_healpix_layers as layers

    expected = {
        "AvgPool", "BasicConvBlock", "ConvGRUBlock", "ConvNeXtBlock",
        "DealiasedDownsample", "DoubleConvNeXtBlock", "Interpolate", "MaxPool",
        "Multi_SymmetricConvNeXtBlock", "SymmetricConvNeXtBlock",
        "TransposedConvUpsample", "SmoothedInterpolateConv", "UNetDecoder",
        "UNetEncoder", "HEALPixLayer", "HEALPixFoldFaces", "HEALPixPadding",
        "HEALPixPaddingIsolatitude", "HEALPixPaddingv2", "HEALPixUnfoldFaces",
        "have_earth2grid", "make_hpx_padding_layer", "warn_deprecated_enable_healpixpad",
    }
    assert set(layers.__all__) == expected
    assert {"HEALPixRecUNet", "HEALPixUNet"} <= set(models.__all__)
    assert layers.HEALPixLayer.__module__.endswith("healpix_layers")
    assert models.GlorysOceanModel is GlorysOceanModel
    assert "hydra" not in sys.modules
    assert "omegaconf" not in sys.modules
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
    channels = [f"thetao_{i}" for i in range(12)] + [f"so_{i}" for i in range(12)] + ["sic", "sithick", "zos", "sst", "mlotst"]
    model = GlorysOceanModel(channels, ["z", "wind", "rad"], torch.zeros(29), torch.ones(29),
                            torch.ones(29, 12, 64, 64, dtype=torch.bool), hidden_channels=4, depth=1)
    with torch.no_grad():
        output = model(torch.zeros(1, 2, 29, 12, 64, 64), torch.zeros(1, 2, 3, 12, 64, 64))
    assert output.shape == (1, 2, 29, 12, 64, 64)
    assert torch.isfinite(output).all()


def test_predicted_nonfinite_ocean_is_rejected_before_sanitization():
    """Predicted instability raises while invalid land and missing observations sanitize."""
    model, mask = setup_model()
    history = torch.zeros(1, 2, 3, 12, 4, 4)
    history[0, 0, 0, 0, 1, 1] = float("nan")
    assert torch.isfinite(model.state_policy(history)).all()
    residual = torch.zeros_like(history)
    residual[..., ~mask] = float("nan")
    assert torch.isfinite(model.reconstruct(history, residual)).all()
    residual[0, 0, 0, 0, 1, 1] = float("nan")
    with pytest.raises(FloatingPointError):
        model.reconstruct(history, residual)
    with torch.no_grad():
        model.head.bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        model(history, torch.zeros_like(history))


def test_sst_absolute_infill_gradients_feedback_and_checkpoint():
    """Shared SST infill preserves ocean, rejects raw land influence and differentiates."""
    mask = torch.ones(1, 12, 4, 4, dtype=torch.bool)
    mask[0, 0, :2, :2] = False
    policy = OceanStatePolicy(["sst"], [280], [4], mask, sst_nlat=33, sst_nlon=64)
    history = torch.randn(1, 2, 1, 12, 4, 4, requires_grad=True)
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
    model = GlorysOceanModel(["sst"], ["z"], [280], [4], mask,
                            hidden_channels=4, depth=1, sst_nlat=33, sst_nlon=64)
    with torch.no_grad():
        model.head.weight.normal_(0, 0.1)
    restored = GlorysOceanModel(["sst"], ["z"], [280], [4], mask,
                               hidden_channels=4, depth=1, sst_nlat=33, sst_nlon=64)
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

    mask = torch.ones(12, 4, 4, dtype=torch.bool)
    mask[0, :2, :2] = False
    filler = TripleInterpSST(mask, nlat=33, nlon=64)
    ocean = torch.arange(mask.numel(), dtype=torch.float64).reshape_as(mask)
    original = torch.where(mask, ocean, float("nan"))
    raw = filler.regrid(original.reshape(-1)).detach().numpy()
    reference = xr.DataArray(raw, dims=["latitude", "longitude"])
    reference = reference.interpolate_na("longitude", use_coordinate=False)
    reference = reference.roll(longitude=32, roll_coords=False).interpolate_na("longitude", use_coordinate=False).roll(longitude=32, roll_coords=False)
    reference = reference.roll(latitude=16, roll_coords=False).interpolate_na("latitude", use_coordinate=False).roll(latitude=17, roll_coords=False)
    expected = filler.regrid_back(torch.as_tensor(reference.values)).reshape_as(ocean)
    expected = torch.where(mask, ocean, expected)
    torch.testing.assert_close(filler(ocean), expected, rtol=1e-12, atol=1e-12)
