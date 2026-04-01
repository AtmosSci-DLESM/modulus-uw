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
