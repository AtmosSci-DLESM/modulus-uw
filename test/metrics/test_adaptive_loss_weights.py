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

from dataclasses import dataclass
from typing import Sequence

import pytest
import torch

from physicsnemo.metrics.climate.adaptive_loss_weights import (
    AdaptiveLossWeights,
    reduce_per_term_loss,
)
from physicsnemo.metrics.climate.healpix_loss import WeightedMSE


@dataclass
class _DummyTrainer:
    device: torch.device
    output_variables: Sequence[str]
    dataloader_train: Sequence


class _FixedTerms(torch.nn.Module):
    """Inner criterion that returns a constant per-term vector."""

    def __init__(self, terms: torch.Tensor):
        super().__init__()
        self.register_buffer("terms", terms.clone())

    def forward(self, prediction, target, average_channels=True):
        del prediction, target
        if average_channels:
            return self.terms.mean()
        return self.terms.clone()


def test_reduce_per_term_loss_matches_mean_for_data_only():
    terms = torch.tensor([1.0, 3.0, 5.0])
    assert torch.allclose(
        reduce_per_term_loss(terms, n_data_variables=3), terms.mean()
    )


def test_adaptive_scalar_equals_sum_weighted_over_c():
    C = 3
    inner = _FixedTerms(torch.tensor([2.0, 4.0, 6.0, 10.0]))
    wrap = AdaptiveLossWeights(
        inner=inner,
        n_data_variables=C,
        constraint_groups=[{"name": "dry_air", "scales": [0.5]}],
        warmup_epochs=0,
        ema_window_epochs=5,
        steps_per_epoch=10,
        variable_names=["a", "b", "c"],
    )
    trainer = _DummyTrainer(
        device=torch.device("cpu"),
        output_variables=["a", "b", "c"],
        dataloader_train=[0] * 10,
    )
    wrap.setup(trainer)
    # Force known weights (bypass EMA).
    wrap.weights.copy_(torch.tensor([1.0, 2.0, 0.5, 0.25]))

    pred = torch.zeros(1, 1, 1, 1, 1, 1)
    target = torch.zeros_like(pred)
    scalar = wrap(pred, target)
    per = wrap(pred, target, average_channels=False)
    expected = reduce_per_term_loss(per, n_data_variables=C)
    assert torch.allclose(scalar, expected)
    assert torch.allclose(
        scalar,
        torch.tensor((2.0 * 1.0 + 4.0 * 2.0 + 6.0 * 0.5 + 10.0 * 0.25) / C),
    )


def test_uniform_variable_weights_reproduce_mean_L():
    C = 4
    L = torch.tensor([1.0, 2.0, 3.0, 4.0])
    inner = _FixedTerms(L)
    wrap = AdaptiveLossWeights(
        inner=inner,
        n_data_variables=C,
        constraint_groups=[],
        warmup_epochs=100,
        ema_window_epochs=5,
        steps_per_epoch=8,
    )
    trainer = _DummyTrainer(
        device=torch.device("cpu"),
        output_variables=[f"v{i}" for i in range(C)],
        dataloader_train=[0] * 8,
    )
    wrap.setup(trainer)
    assert torch.allclose(wrap.weights, torch.ones(C))

    pred = torch.zeros(1, 1, 1, 1, 1, 1)
    scalar = wrap(pred, pred)
    assert torch.allclose(scalar, L.mean())


def test_constraint_scale_s_contributes_s_over_c_of_T():
    """After warmup, a constraint with scale s adds s/C * T when m_k = T."""
    C = 2
    # Data EMA equal → T = 4; constraint EMA also 4 → w_k = s * T / m_k = s.
    terms = torch.tensor([4.0, 4.0, 4.0])
    s = 0.001
    inner = _FixedTerms(terms)
    wrap = AdaptiveLossWeights(
        inner=inner,
        n_data_variables=C,
        constraint_groups=[{"name": "aam", "scales": [s]}],
        warmup_epochs=0,
        ema_window_epochs=5,
        steps_per_epoch=4,
        variable_names=["u", "v"],
    )
    trainer = _DummyTrainer(
        device=torch.device("cpu"),
        output_variables=["u", "v"],
        dataloader_train=[0] * 4,
    )
    wrap.setup(trainer)

    pred = torch.zeros(1, 1, 1, 1, 1, 1)
    # Seed EMA from one step then apply adaptive weights.
    _ = wrap(pred, pred)
    wrap.post_backward_update(epoch=0)
    # m = terms, T = 4, w = [1, 1, s]
    assert torch.allclose(
        wrap.weights, torch.tensor([1.0, 1.0, s]), rtol=1e-5, atol=1e-8
    )

    scalar = wrap(pred, pred)
    T = 4.0
    # data part = T; constraint adds s*T; scalar = (2*T + s*T) / C? 
    # weighted terms = [4, 4, 4*s]; sum/C = (8 + 4s)/2 = 4 + 2s
    # Plan: constraint term equals s*T and adds s*T/C to the scalar.
    # s*T/C = 0.001*4/2 = 0.002; data part sum(weighted vars)/C = 8/2 = 4.
    expected_data = T
    expected_constraint = s * T / C
    assert torch.allclose(
        scalar, torch.tensor(expected_data + expected_constraint), rtol=1e-5
    )
    assert torch.allclose(
        wrap.log_buffers["loss_constraint/aam"],
        torch.tensor(expected_constraint),
        rtol=1e-5,
    )
    assert torch.allclose(
        wrap.log_buffers["loss_data"], torch.tensor(expected_data), rtol=1e-5
    )


def test_post_backward_updates_ema_only_when_called():
    C = 2
    inner = _FixedTerms(torch.tensor([1.0, 3.0]))
    wrap = AdaptiveLossWeights(
        inner=inner,
        n_data_variables=C,
        warmup_epochs=1,
        ema_window_epochs=2,
        steps_per_epoch=2,
    )
    trainer = _DummyTrainer(
        device=torch.device("cpu"),
        output_variables=["a", "b"],
        dataloader_train=[0, 0],
    )
    wrap.setup(trainer)
    pred = torch.zeros(1, 1, 1, 1, 1, 1)
    _ = wrap(pred, pred)
    # Eval path: forward ran but no post_backward → EMA unchanged.
    assert int(wrap.ema_initialized.item()) == 0
    wrap.post_backward_update(epoch=0)
    assert int(wrap.ema_initialized.item()) == 1
    assert torch.allclose(wrap.ema, torch.tensor([1.0, 3.0]))
    # Still in warmup: weights remain relative scales (ones).
    assert torch.allclose(wrap.weights, torch.ones(C))

    # Second step with different inner terms via mutating buffer.
    inner.terms.copy_(torch.tensor([5.0, 7.0]))
    _ = wrap(pred, pred)
    wrap.post_backward_update(epoch=1)
    # β = 1 - 1/(2*2) = 0.75; m = 0.75*[1,3] + 0.25*[5,7]
    expected = 0.75 * torch.tensor([1.0, 3.0]) + 0.25 * torch.tensor([5.0, 7.0])
    assert torch.allclose(wrap.ema, expected)
    # Adaptive weights now on (epoch >= 1).
    T = expected.mean()
    assert torch.allclose(wrap.weights, T / expected.clamp_min(wrap.eps))


def test_state_dict_contains_ema_and_weights():
    wrap = AdaptiveLossWeights(
        inner=_FixedTerms(torch.tensor([1.0, 2.0])),
        n_data_variables=2,
        steps_per_epoch=1,
    )
    keys = wrap.state_dict().keys()
    assert "ema" in keys
    assert "weights" in keys
    assert "ema_initialized" in keys
    assert "relative_scales" in keys


def test_wraps_weighted_mse_like_base_config():
    weights = [1.0, 1.0, 1.0]
    data_loss = WeightedMSE(weights=weights)
    wrap = AdaptiveLossWeights(
        inner=data_loss,
        n_data_variables=3,
        constraint_groups=[],
        warmup_epochs=10,
        steps_per_epoch=5,
        variable_names=["x", "y", "z"],
    )
    trainer = _DummyTrainer(
        device=torch.device("cpu"),
        output_variables=["x", "y", "z"],
        dataloader_train=[0] * 5,
    )
    wrap.setup(trainer)
    pred = torch.randn(2, 1, 1, 3, 4, 4)
    target = torch.randn_like(pred)
    scalar = wrap(pred, target)
    per_mse = data_loss(pred, target, average_channels=False)
    assert torch.allclose(scalar, per_mse.mean())
    assert "loss" in wrap.log_buffers
    assert "loss_data" in wrap.log_buffers


def test_n_data_variables_inferred_from_output_variables():
    inner = _FixedTerms(torch.tensor([2.0, 4.0, 6.0, 1.0]))
    wrap = AdaptiveLossWeights(
        inner=inner,
        constraint_groups=[{"name": "dry_air", "scales": [0.001]}],
        warmup_epochs=1,
        steps_per_epoch=4,
    )
    assert wrap._built is False
    trainer = _DummyTrainer(
        device=torch.device("cpu"),
        output_variables=["a", "b", "c"],
        dataloader_train=[0] * 4,
    )
    wrap.setup(trainer)
    assert wrap.n_data_variables == 3
    assert wrap.n_terms == 4
    assert wrap.variable_names == ["a", "b", "c"]
    assert wrap.term_names[-1] == "dry_air"
    pred = torch.zeros(1, 1, 1, 1, 1, 1)
    scalar = wrap(pred, pred)
    assert scalar.ndim == 0


def test_explicit_n_data_variables_must_match_outputs():
    wrap = AdaptiveLossWeights(
        inner=_FixedTerms(torch.tensor([1.0, 2.0])),
        n_data_variables=2,
        steps_per_epoch=1,
    )
    trainer = _DummyTrainer(
        device=torch.device("cpu"),
        output_variables=["only_one"],
        dataloader_train=[0],
    )
    with pytest.raises(ValueError, match="does not match"):
        wrap.setup(trainer)
