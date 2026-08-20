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
"""Tests for deterministic → probabilistic CLN weight transfer."""

import os
import sys
from functools import partial
from unittest.mock import patch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import pytest
import torch
import torch.nn as nn
from pytest_utils import import_or_fail


def _make_cln_partial(*, init_cln_to_zero: bool = True, scale_center: float = 1.0):
    from physicsnemo.models.dlwp_healpix_layers.normalization import ConditionalLayerNorm

    return partial(
        ConditionalLayerNorm,
        condition_shape=8,
        mlp_hidden_dims=[16, 16],
        init_cln_to_zero=init_cln_to_zero,
        scale_center=scale_center,
    )


def _make_block(*, with_cln: bool, in_channels: int = 2, latent_channels: int = 2):
    from physicsnemo.models.dlwp_healpix_layers import Multi_SymmetricConvNeXtBlock

    kwargs = dict(
        in_channels=in_channels,
        latent_channels=latent_channels,
        out_channels=in_channels,
        activation=nn.ReLU(),
        n_layers=1,
    )
    if with_cln:
        kwargs["conditional_layer_norm"] = _make_cln_partial(
            init_cln_to_zero=True, scale_center=1.0
        )
    return Multi_SymmetricConvNeXtBlock(**kwargs)


def _non_cln_params_ordered(module: nn.Module):
    """Parameter tensors excluding ConditionalLayerNorm, in named_parameters order."""
    from physicsnemo.models.dlwp_healpix_layers.normalization import ConditionalLayerNorm

    skip = set()
    for name, child in module.named_modules():
        if isinstance(child, ConditionalLayerNorm):
            for pname, _ in child.named_parameters():
                skip.add(f"{name}.{pname}" if name else pname)
    return [(n, p) for n, p in module.named_parameters() if n not in skip]


def _perturb_non_cln_(module: nn.Module, delta: float = 1.0) -> None:
    with torch.no_grad():
        for _, p in _non_cln_params_ordered(module):
            p.add_(delta)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cpu"])
def test_transfer_copies_backbone_leaves_cln_at_init(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers.checkpoint_transfer import (
        assert_no_cln_params_copied,
        transfer_weights_skipping_cln,
    )
    from physicsnemo.models.dlwp_healpix_layers.normalization import ConditionalLayerNorm

    torch.manual_seed(0)
    src = _make_block(with_cln=False).to(device)
    tgt = _make_block(with_cln=True).to(device)
    _perturb_non_cln_(tgt, 1.0)

    report = transfer_weights_skipping_cln(src, tgt)
    assert report.n_copied > 0
    assert report.n_cln_left_at_init > 0
    assert_no_cln_params_copied(report)

    src_params = _non_cln_params_ordered(src)
    tgt_params = _non_cln_params_ordered(tgt)
    assert len(src_params) == len(tgt_params)
    for (sn, sp), (tn, tp) in zip(src_params, tgt_params):
        assert sp.shape == tp.shape, f"{sn} vs {tn}"
        assert torch.equal(sp, tp), f"{sn} vs {tn}"

    for module in tgt.modules():
        if isinstance(module, ConditionalLayerNorm):
            last = module.gamma_beta_mlp[-1]
            assert torch.count_nonzero(last.weight) == 0
            assert torch.count_nonzero(last.bias) == 0


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cpu"])
def test_transfer_forward_finite_with_conditions(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers.checkpoint_transfer import (
        transfer_weights_skipping_cln,
    )

    torch.manual_seed(1)
    src = _make_block(with_cln=False).to(device)
    tgt = _make_block(with_cln=True).to(device)
    transfer_weights_skipping_cln(src, tgt)

    # HEALPixLayer padding expects a multiple of 12 faces along the batch dim.
    x = torch.randn(12, 2, 8, 8, device=device)
    # CLN expands conditions across n_faces=12, so condition batch is 1 here.
    cond = torch.randn(1, 8, device=device)
    y = tgt(x, conditions_cln=cond)
    assert torch.isfinite(y).all()
    assert y.shape == x.shape


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cpu"])
def test_transfer_raises_on_channel_mismatch(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix_layers.checkpoint_transfer import (
        CheckpointTransferError,
        transfer_weights_skipping_cln,
    )

    src = _make_block(with_cln=False, in_channels=2, latent_channels=2).to(device)
    tgt = _make_block(with_cln=True, in_channels=4, latent_channels=2).to(device)
    with pytest.raises(CheckpointTransferError, match="mismatch"):
        transfer_weights_skipping_cln(src, tgt)


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cpu"])
def test_blind_state_dict_load_skips_shifted_conv_keys(device, pytestconfig):
    """Index shifts change state_dict key paths, so blind load leaves later convs unloaded."""
    from physicsnemo.models.dlwp_healpix_layers.checkpoint_transfer import (
        transfer_weights_skipping_cln,
    )

    torch.manual_seed(2)
    src = _make_block(with_cln=False).to(device)
    blind_tgt = _make_block(with_cln=True).to(device)
    remapped_tgt = _make_block(with_cln=True).to(device)
    _perturb_non_cln_(blind_tgt, 1.0)
    _perturb_non_cln_(remapped_tgt, 1.0)

    blind_tgt.load_state_dict(src.state_dict(), strict=False)
    transfer_weights_skipping_cln(src, remapped_tgt)

    src_params = _non_cln_params_ordered(src)
    blind_params = _non_cln_params_ordered(blind_tgt)
    remapped_params = _non_cln_params_ordered(remapped_tgt)

    remapped_all_match = all(
        torch.equal(sp, tp) for (_, sp), (_, tp) in zip(src_params, remapped_params)
    )
    assert remapped_all_match

    blind_all_match = all(
        torch.equal(sp, tp) for (_, sp), (_, tp) in zip(src_params, blind_params)
    )
    assert not blind_all_match, (
        "Expected blind load_state_dict to miss shifted ModuleList keys; "
        "if layout changed, update this regression check."
    )


@import_or_fail("hydra")
@pytest.mark.parametrize("device", ["cpu"])
def test_load_deterministic_mdlus_wrapper(device, pytestconfig):
    """``load_deterministic_weights_into_probabilistic_model`` delegates to tree transfer."""
    from physicsnemo.models.dlwp_healpix_layers.checkpoint_transfer import (
        assert_no_cln_params_copied,
        load_deterministic_weights_into_probabilistic_model,
    )

    torch.manual_seed(3)
    src = _make_block(with_cln=False).to(device)
    tgt = _make_block(with_cln=True).to(device)
    _perturb_non_cln_(tgt, 0.5)

    with patch(
        "physicsnemo.models.dlwp_healpix_layers.checkpoint_transfer.Module.from_checkpoint",
        return_value=src,
    ):
        report = load_deterministic_weights_into_probabilistic_model(
            tgt, "/unused/path.mdlus"
        )

    assert report.n_copied > 0
    assert report.n_cln_left_at_init > 0
    assert_no_cln_params_copied(report)

    for (sn, sp), (tn, tp) in zip(
        _non_cln_params_ordered(src), _non_cln_params_ordered(tgt)
    ):
        assert torch.equal(sp, tp), f"{sn} vs {tn}"
