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

from physicsnemo.models.dlwp_healpix_layers.healpix_constraints import (
    DryAirMassConstraint,
    NonnegativeConstraint,
)


def _reference_forward(prediction, channels, constrained_names, scaling):
    constrained_set = set(constrained_names)
    per = [
        (0.0 - scaling[c]["mean"]) / scaling[c]["std"] if c in constrained_set else float("-inf")
        for c in channels
    ]
    t = torch.tensor(per, dtype=torch.float32, device=prediction.device).view(
        1, 1, 1, -1, 1, 1
    )
    return torch.maximum(prediction, t.to(dtype=prediction.dtype))


def _reference_dry_air_mass_forward(prediction, input_tensor, channels, scaling, g0=9.81):
    """Legacy in-place write path; used only to validate the functional implementation."""
    sp_idx = channels.index("sp")
    tcwv_idx = channels.index("tcwv")
    sp_idx_t = torch.tensor([sp_idx], dtype=torch.long)
    tcwv_idx_t = torch.tensor([tcwv_idx], dtype=torch.long)

    ps_mean = torch.tensor(scaling["sp"]["mean"], dtype=torch.float32)
    ps_std = torch.tensor(scaling["sp"]["std"], dtype=torch.float32)
    tcwv_mean = torch.tensor(scaling["tcwv"]["mean"], dtype=torch.float32)
    tcwv_std = torch.tensor(scaling["tcwv"]["std"], dtype=torch.float32)

    pred = prediction.float().clone()
    inp = input_tensor.float().clone()

    sp = torch.index_select(pred, dim=3, index=sp_idx_t)
    sp_phys = sp * ps_std + ps_mean
    tcwv = torch.index_select(pred, dim=3, index=tcwv_idx_t)
    tcwv_phys = tcwv * tcwv_std + tcwv_mean

    sp_0 = torch.index_select(inp, dim=3, index=sp_idx_t)[:, :, -1:]
    sp_0 = sp_0 * ps_std + ps_mean
    tcwv_0 = torch.index_select(inp, dim=3, index=tcwv_idx_t)[:, :, -1:]
    tcwv_0 = tcwv_0 * tcwv_std + tcwv_mean

    sp_dry = sp_phys - g0 * tcwv_phys
    sp_0_dry = sp_0 - g0 * tcwv_0
    correction = (sp_dry - sp_0_dry).mean(dim=[1, 4, 5], keepdim=True)
    sp_corrected_phys = sp_phys - correction
    sp_corrected_phys = torch.clamp(sp_corrected_phys, min=0.0)
    sp_corrected = (sp_corrected_phys - ps_mean) / ps_std

    pred.index_copy_(3, sp_idx_t, sp_corrected)
    return pred


def test_dry_air_mass_matches_reference_index_copy():
    channels = ["t2m", "tcwv", "sp"]
    scaling = {
        "sp": {"mean": 100000.0, "std": 5000.0},
        "tcwv": {"mean": 25.0, "std": 15.0},
    }
    mod = DryAirMassConstraint(in_channels=channels, out_channels=channels, scaling=scaling)
    torch.manual_seed(42)
    b, f, t, h, w = 2, 1, 3, 4, 4
    c = len(channels)
    prediction = torch.randn(b, f, t, c, h, w)
    inp = torch.randn(b, f, t, c, h, w)
    ref = _reference_dry_air_mass_forward(prediction, inp, channels, scaling)
    out = mod(prediction, inp)
    assert torch.allclose(out, ref)
    assert not out.data_ptr() == prediction.data_ptr()


def test_dry_air_mass_non_sp_channels_unchanged():
    channels = ["tcwv", "sp", "x"]
    scaling = {
        "sp": {"mean": 1e5, "std": 1e3},
        "tcwv": {"mean": 20.0, "std": 10.0},
    }
    mod = DryAirMassConstraint(in_channels=channels, out_channels=channels, scaling=scaling)
    torch.manual_seed(7)
    prediction = torch.randn(1, 1, 2, len(channels), 3, 3)
    inp = torch.randn_like(prediction)
    out = mod(prediction, inp)
    for i, name in enumerate(channels):
        if name != "sp":
            assert torch.allclose(out[..., i, :, :], prediction[..., i, :, :])


def test_dry_air_mass_sp_channel_mask_buffer():
    channels = ["a", "sp", "tcwv", "b"]
    scaling = {
        "sp": {"mean": 0.0, "std": 1.0},
        "tcwv": {"mean": 0.0, "std": 1.0},
    }
    mod = DryAirMassConstraint(in_channels=channels, out_channels=channels, scaling=scaling)
    m = mod.sp_channel_mask.view(-1)
    assert m.sum().item() == 1.0
    assert m[channels.index("sp")].item() == 1.0


def test_dry_air_mass_torch_compile_forward():
    channels = ["tcwv", "sp"]
    scaling = {
        "sp": {"mean": 100000.0, "std": 5000.0},
        "tcwv": {"mean": 25.0, "std": 15.0},
    }
    mod = DryAirMassConstraint(in_channels=channels, out_channels=channels, scaling=scaling)
    torch.manual_seed(0)
    prediction = torch.randn(1, 1, 2, 2, 3, 3)
    inp = torch.randn(1, 1, 2, 2, 3, 3)
    ref = mod(prediction, inp)
    try:
        compiled = torch.compile(mod)
    except Exception:
        pytest.skip("torch.compile not available or failed to compile")
    out = compiled(prediction, inp)
    assert torch.allclose(out, ref)


def test_dry_air_mass_torch_compile_backward():
    """Inductor failed on backward with index_select+buffer index; guard with aot backward."""
    channels = ["tcwv", "sp"]
    scaling = {
        "sp": {"mean": 100000.0, "std": 5000.0},
        "tcwv": {"mean": 25.0, "std": 15.0},
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = DryAirMassConstraint(in_channels=channels, out_channels=channels, scaling=scaling).to(device)
    try:
        compiled = torch.compile(mod)
    except Exception:
        pytest.skip("torch.compile not available or failed to compile")
    torch.manual_seed(1)
    prediction = torch.randn(
        1, 1, 2, 2, 4, 4, device=device, requires_grad=True
    )
    inp = torch.randn(1, 1, 2, 2, 4, 4, device=device)
    out = compiled(prediction, inp)
    out.sum().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_nonnegative_threshold_buffer_matches_formula():
    channels = ["q", "t", "sp"]
    scaling = {
        "q": {"mean": 1.0, "std": 2.0},
        "t": {"mean": 3.0, "std": 4.0},
        "sp": {"mean": 5.0, "std": 6.0},
    }
    mod = NonnegativeConstraint(
        variables=["q", "sp"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
    )
    flat = mod.thresholds.view(-1)
    assert flat[0].item() == pytest.approx((0.0 - 1.0) / 2.0)
    assert torch.isneginf(flat[1])
    assert flat[2].item() == pytest.approx((0.0 - 5.0) / 6.0)


def test_nonnegative_forward_matches_reference():
    channels = ["a", "b", "c"]
    scaling = {
        "a": {"mean": 0.5, "std": 0.25},
        "b": {"mean": 0.0, "std": 1.0},
        "c": {"mean": 10.0, "std": 2.0},
    }
    mod = NonnegativeConstraint(
        variables=["a", "c"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
    )
    torch.manual_seed(0)
    prediction = torch.randn(2, 1, 1, 3, 4, 4)
    out = mod(prediction, prediction)
    expected = _reference_forward(
        prediction, channels, {"a", "c"}, scaling
    )
    assert torch.equal(out, expected)
    assert not out.data_ptr() == prediction.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_nonnegative_forward_dtype_cast(dtype):
    channels = ["x"]
    scaling = {"x": {"mean": 1.0, "std": 2.0}}
    mod = NonnegativeConstraint(
        variables=["x"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
    )
    prediction = torch.tensor([[[[[[-3.0, 0.5]]]]]], dtype=dtype)
    out = mod(prediction, prediction)
    thr = (0.0 - 1.0) / 2.0
    assert out.dtype == dtype
    assert out[0, 0, 0, 0, 0, 0].item() == pytest.approx(thr)
    assert out[0, 0, 0, 0, 0, 1].item() == pytest.approx(0.5)


def test_nonnegative_variables_not_in_channels_ignored():
    channels = ["only"]
    scaling = {"only": {"mean": 2.0, "std": 1.0}}
    mod = NonnegativeConstraint(
        variables=["only", "missing"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
    )
    flat = mod.thresholds.view(-1)
    assert flat.numel() == 1
    assert flat[0].item() == pytest.approx((0.0 - 2.0) / 1.0)


def test_nonnegative_no_constrained_variables_all_neg_inf():
    channels = ["a", "b"]
    scaling = {
        "a": {"mean": 0.0, "std": 1.0},
        "b": {"mean": 0.0, "std": 1.0},
    }
    mod = NonnegativeConstraint(
        variables=["ghost"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
    )
    assert torch.all(torch.isneginf(mod.thresholds.view(-1)))
    prediction = torch.randn(1, 1, 1, 2, 2, 2)
    out = mod(prediction, prediction)
    assert torch.equal(out, prediction)


def test_nonnegative_torch_compile_forward():
    channels = ["x", "y"]
    scaling = {
        "x": {"mean": 0.0, "std": 1.0},
        "y": {"mean": 1.0, "std": 2.0},
    }
    mod = NonnegativeConstraint(
        variables=["x"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
    )
    prediction = torch.randn(1, 1, 1, 2, 2, 2)
    ref = mod(prediction, prediction)
    try:
        compiled = torch.compile(mod)
    except Exception:
        pytest.skip("torch.compile not available or failed to compile")
    out = compiled(prediction, prediction)
    assert torch.allclose(out, ref)


def test_nonnegative_keep_grad_through_clamp_forward_unchanged():
    """STE must not change forward values; only the backward path."""
    channels = ["x"]
    # Physical zero is at normalized (0 - 0) / 1 = 0
    scaling = {"x": {"mean": 0.0, "std": 1.0}}
    plain = NonnegativeConstraint(
        variables=["x"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
        keep_grad_through_clamp=False,
    )
    ste = NonnegativeConstraint(
        variables=["x"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
        keep_grad_through_clamp=True,
    )
    prediction = torch.tensor([[[[[[-2.0, 0.5, 3.0]]]]]])
    assert torch.equal(plain(prediction, prediction), ste(prediction, prediction))
    assert torch.equal(
        ste(prediction, prediction),
        torch.tensor([[[[[[0.0, 0.5, 3.0]]]]]]),
    )


def test_nonnegative_keep_grad_through_clamp_passes_gradient():
    """Below-threshold cells get zero grad with plain clamp, identity with STE."""
    channels = ["x"]
    scaling = {"x": {"mean": 0.0, "std": 1.0}}
    # One saturated-negative cell and one interior cell
    raw = torch.tensor([[[[[[-2.0, 1.0]]]]]])

    plain = NonnegativeConstraint(
        variables=["x"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
        keep_grad_through_clamp=False,
    )
    x_plain = raw.clone().requires_grad_(True)
    plain(x_plain, x_plain).sum().backward()
    torch.testing.assert_close(
        x_plain.grad, torch.tensor([[[[[[0.0, 1.0]]]]]])
    )

    ste = NonnegativeConstraint(
        variables=["x"],
        in_channels=channels,
        out_channels=channels,
        scaling=scaling,
        keep_grad_through_clamp=True,
    )
    x_ste = raw.clone().requires_grad_(True)
    out = ste(x_ste, x_ste)
    torch.testing.assert_close(out, torch.tensor([[[[[[0.0, 1.0]]]]]]))
    out.sum().backward()
    torch.testing.assert_close(x_ste.grad, torch.ones_like(raw))


def test_replace_value_keep_gradient_identity_backward():
    from physicsnemo.models.dlwp_healpix_layers.healpix_constraints import (
        replace_value_keep_gradient,
    )

    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    new_value = torch.clamp(x, min=0.0)
    out = replace_value_keep_gradient(x, new_value)
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 2.0]))
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
