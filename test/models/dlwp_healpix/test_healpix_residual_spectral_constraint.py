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

from physicsnemo.models.dlwp_healpix_layers.healpix_residual_spectral_constraint import (
    InputSkipSpectralLowPassConstraint,
    ResidualSpectralLowPassConstraint,
)

NSIDE = 8
LMAX = 3 * NSIDE - 1
CUTOFF = 8
CUDA_REASON = "CUDA + cuhpx SHT required"


def _cuda_sht_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from cuhpx import SHTCUDA  # noqa: F401
    except ImportError:
        return False
    return True


def _make_mod(cutoffs=None, **kwargs):
    cutoffs = cutoffs if cutoffs is not None else {"PRESsfc": CUTOFF}
    defaults = dict(
        cutoffs=cutoffs,
        in_channels=["PRESsfc", "TMP2m"],
        out_channels=["PRESsfc", "TMP2m", "PRATEsfc"],
        nside=NSIDE,
        lmax=LMAX,
    )
    defaults.update(kwargs)
    return ResidualSpectralLowPassConstraint(**defaults)


def test_rejects_empty_cutoffs():
    with pytest.raises(ValueError, match="at least one"):
        _make_mod(cutoffs={})


def test_rejects_unknown_variable():
    with pytest.raises(ValueError, match="not in out_channels"):
        _make_mod(cutoffs={"not_a_var": 4})


def test_rejects_diagnostic_variable():
    with pytest.raises(ValueError, match="diagnostic"):
        _make_mod(cutoffs={"PRATEsfc": 4})


def test_rejects_cutoff_below_one():
    with pytest.raises(ValueError, match="must be >= 1"):
        _make_mod(cutoffs={"PRESsfc": 0})


def test_rejects_cutoff_above_lmax():
    with pytest.raises(ValueError, match="lmax"):
        _make_mod(cutoffs={"PRESsfc": LMAX + 1})


@pytest.mark.skipif(not _cuda_sht_available(), reason=CUDA_REASON)
def test_unlisted_channels_unchanged():
    device = "cuda"
    mod = _make_mod().to(device)
    torch.manual_seed(0)
    b, f, t, c, h = 1, 12, 1, 3, NSIDE
    orig = torch.randn(b, f, t, 2, h, h, device=device)
    pred = torch.randn(b, f, t, c, h, h, device=device)
    pred[:, :, :, :2] = orig + torch.randn_like(orig)
    out = mod(pred, orig)
    assert torch.allclose(out[:, :, :, 1], pred[:, :, :, 1])
    assert torch.allclose(out[:, :, :, 2], pred[:, :, :, 2])
    assert not torch.allclose(out[:, :, :, 0], pred[:, :, :, 0])


def _ell_power(mod, faces):
    alm = mod._to_alm(faces)
    return (alm.real ** 2 + alm.imag ** 2).sum(dim=-1)[0, 0, 0]


@pytest.mark.skipif(not _cuda_sht_available(), reason=CUDA_REASON)
def test_high_ell_residual_dropped_low_ell_kept():
    device = "cuda"
    mod = _make_mod().to(device)
    torch.manual_seed(1)
    b, t, h = 1, 1, NSIDE
    orig = torch.zeros(b, 12, t, 2, h, h, device=device)
    noise = torch.randn(b, 12, t, 1, h, h, device=device)
    alm = mod._to_alm(noise)
    ell = torch.arange(LMAX, device=device).view(1, 1, 1, LMAX, 1)
    high = mod._from_alm(alm * (ell >= CUTOFF).to(alm.real.dtype))
    low = mod._from_alm(alm * (ell < CUTOFF).to(alm.real.dtype))
    residual = high + low
    pred = torch.zeros(b, 12, t, 3, h, h, device=device)
    pred[:, :, :, :1] = orig[:, :, :, :1] + residual
    out = mod(pred, orig)
    out_res = out[:, :, :, :1] - orig[:, :, :, :1]
    p_in = _ell_power(mod, residual)
    p_out = _ell_power(mod, out_res)
    assert p_out[CUTOFF:].sum() < 1e-4 * p_in[CUTOFF:].sum().clamp(min=1e-12)
    assert torch.allclose(p_out[:CUTOFF], p_in[:CUTOFF], rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not _cuda_sht_available(), reason=CUDA_REASON)
def test_backward_through_filter():
    device = "cuda"
    mod = _make_mod().to(device)
    torch.manual_seed(2)
    orig = torch.zeros(1, 12, 1, 2, NSIDE, NSIDE, device=device)
    pred = torch.randn(1, 12, 1, 3, NSIDE, NSIDE, device=device, requires_grad=True)
    out = mod(pred, orig)
    out.square().mean().backward()
    assert pred.grad is not None
    assert pred.grad[:, :, :, 0].abs().sum() > 0
    # Unlisted channels are identity; grad must pass through.
    assert pred.grad[:, :, :, 1].abs().sum() > 0


def _split_ell(mod, faces, cutoff):
    alm = mod._to_alm(faces)
    ell = torch.arange(mod.lmax, device=faces.device).view(1, 1, 1, mod.lmax, 1)
    dtype = alm.real.dtype
    high = mod._from_alm(alm * (ell >= cutoff).to(dtype))
    low = mod._from_alm(alm * (ell < cutoff).to(dtype))
    return high, low


@pytest.mark.skipif(not _cuda_sht_available(), reason=CUDA_REASON)
def test_input_skip_drops_high_ell_of_state_keeps_residual():
    device = "cuda"
    mod = InputSkipSpectralLowPassConstraint(
        cutoffs={"PRESsfc": CUTOFF},
        in_channels=["PRESsfc", "TMP2m"],
        out_channels=["PRESsfc", "TMP2m", "PRATEsfc"],
        nside=NSIDE,
        lmax=LMAX,
    ).to(device)
    torch.manual_seed(3)
    b, t, h = 1, 1, NSIDE
    state = torch.randn(b, 12, t, 1, h, h, device=device)
    state_high, state_low = _split_ell(mod, state, CUTOFF)
    # Zero residual: output high-ℓ of PRESsfc is removed; low-ℓ of x is kept.
    orig = torch.zeros(b, 12, t, 2, h, h, device=device)
    orig[:, :, :, :1] = state_high + state_low
    pred = orig.new_zeros(b, 12, t, 3, h, h)
    pred[:, :, :, :2] = orig
    pred[:, :, :, 1] = torch.randn_like(orig[:, :, :, 1])
    pred[:, :, :, 2] = torch.randn(b, 12, t, h, h, device=device)
    out = mod(pred, orig)
    p_out = _ell_power(mod, out[:, :, :, :1])
    p_state = _ell_power(mod, state_high + state_low)
    assert p_out[CUTOFF:].sum() < 1e-4 * p_state[CUTOFF:].sum().clamp(min=1e-12)
    assert torch.allclose(p_out[:CUTOFF], p_state[:CUTOFF], rtol=1e-3, atol=1e-5)
    assert torch.allclose(out[:, :, :, 1], pred[:, :, :, 1])
    assert torch.allclose(out[:, :, :, 2], pred[:, :, :, 2])

    # Zero input: high-ℓ of the residual is kept.
    residual = torch.randn(b, 12, t, 1, h, h, device=device)
    res_high, res_low = _split_ell(mod, residual, CUTOFF)
    orig.zero_()
    pred.zero_()
    pred[:, :, :, :1] = res_high + res_low
    out = mod(pred, orig)
    p_res = _ell_power(mod, res_high + res_low)
    p_kept = _ell_power(mod, out[:, :, :, :1])
    assert torch.allclose(p_kept[CUTOFF:], p_res[CUTOFF:], rtol=1e-3, atol=1e-5)
    assert torch.allclose(p_kept[:CUTOFF], p_res[:CUTOFF], rtol=1e-3, atol=1e-5)
