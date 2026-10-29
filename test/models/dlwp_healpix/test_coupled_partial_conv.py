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

"""Tests for coupled-input partial convolution stem."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest
import torch as th
import xarray as xr

import torch.nn as nn

from physicsnemo.models.dlwp_healpix_layers.coupled_partial_conv import (
    DEFAULT_COUPLED_PARTIAL_CONV_STEM_TARGET,
    CoupledPartialConvStem,
    PartialHEALPixConv2d,
    build_coupled_partial_conv_stem,
    coupled_variable_channel_names,
    load_spatial_mask,
)
from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
    HEALPixFoldFaces,
    HEALPixUnfoldFaces,
)
from physicsnemo.models.dlwp_healpix_layers.reflection_ops import (
    apply_R_kernel,
    hpx_spatial_reflect,
    project_even_kernel,
)


def _reflect_bfchw(x: th.Tensor) -> th.Tensor:
    """Equatorial HEALPix reflection on ``[B, F, C, H, W]`` (spatial only)."""
    fold = HEALPixFoldFaces()
    unfold = HEALPixUnfoldFaces(num_faces=x.shape[1])
    return unfold(hpx_spatial_reflect(fold(x)))


def _reflect_bfchw_typed(x: th.Tensor, odd_channels: Sequence[int]) -> th.Tensor:
    """Spatial HEALPix reflection plus sign flip on odd channel indices."""
    y = _reflect_bfchw(x).clone()
    for i in odd_channels:
        y[:, :, i] *= -1
    return y


class _IdentityCoupledStem(nn.Module):
    """Minimal custom stem used to verify Hydra ``_target_`` wiring."""

    def __init__(
        self,
        channel_masks,
        hpx_padding_mode=None,
        nside=None,
        compile_padding=False,
        channel_is_odd=None,
        **kwargs,
    ):
        super().__init__()
        self.register_buffer(
            "channel_masks",
            channel_masks.permute(1, 0, 2, 3).contiguous(),
            persistent=True,
        )

    def forward(self, coupled):
        return coupled


N_FACES = 12
NSIDE = 8


def _toy_couplings(variables=("sst", "sic"), input_times=("0h",)):
    return [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "variables": list(variables),
                "input_times": list(input_times),
            },
        }
    ]


def _write_lsm_zarr(path, ocean_frac: np.ndarray):
    """Write constants/lsm as land fraction ``1 - ocean_frac``."""
    land = 1.0 - ocean_frac.astype(np.float32)
    ds = xr.Dataset(
        {
            "constants": (
                ("channel_c", "face", "height", "width"),
                land[np.newaxis, ...],
            )
        },
        coords={"channel_c": ["lsm"]},
    )
    ds.to_zarr(path, mode="w")


@pytest.fixture
def ocean_frac():
    # Mostly ocean on even faces, mostly land on odd faces; fractional coast band.
    m = np.zeros((N_FACES, NSIDE, NSIDE), dtype=np.float32)
    m[0::2] = 1.0
    m[1::2] = 0.0
    m[:, :2, :] = 0.3  # soft coastal strip
    return m


def test_coupled_variable_channel_names_expands_times():
    couplings = _toy_couplings(variables=("sst", "sic"), input_times=("0h", "6h"))
    assert coupled_variable_channel_names(couplings) == [
        "sst",
        "sst",
        "sic",
        "sic",
    ]


def test_load_spatial_mask_soft_and_hard(tmp_path, ocean_frac):
    zpath = tmp_path / "mask.zarr"
    _write_lsm_zarr(zpath, ocean_frac)

    soft = load_spatial_mask(
        str(zpath),
        data_var="constants",
        selection_dict={"channel_c": "lsm"},
        invert=True,
        threshold=None,
    )
    hard = load_spatial_mask(
        str(zpath),
        data_var="constants",
        selection_dict={"channel_c": "lsm"},
        invert=True,
        threshold=0.5,
    )
    assert soft.shape == (N_FACES, NSIDE, NSIDE)
    assert th.allclose(soft, th.from_numpy(ocean_frac))
    assert set(hard.unique().tolist()) <= {0.0, 1.0}
    assert th.all(hard[ocean_frac > 0.5] == 1.0)
    assert th.all(hard[ocean_frac <= 0.5] == 0.0)


def test_build_stem_none_config_returns_none():
    assert build_coupled_partial_conv_stem(None, _toy_couplings()) is None


def test_build_stem_requires_all_variables(tmp_path, ocean_frac):
    zpath = tmp_path / "mask.zarr"
    _write_lsm_zarr(zpath, ocean_frac)
    cfg = {
        "kernel_size": 3,
        "masks": {
            "sst": {
                "dataset_path": str(zpath),
                "data_var": "constants",
                "selection_dict": {"channel_c": "lsm"},
                "invert": True,
                "threshold": None,
            }
        },
    }
    with pytest.raises(ValueError, match="missing entries"):
        build_coupled_partial_conv_stem(cfg, _toy_couplings())


@pytest.mark.parametrize("hpx_padding_mode", ["karlbauer", "earth2grid"])
def test_stem_forward_finite_and_shape(tmp_path, ocean_frac, hpx_padding_mode):
    if hpx_padding_mode == "earth2grid":
        from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
            have_earth2grid,
        )

        if not have_earth2grid:
            pytest.skip("earth2grid not available")

    zpath = tmp_path / "mask.zarr"
    _write_lsm_zarr(zpath, ocean_frac)
    mask_cfg = {
        "dataset_path": str(zpath),
        "data_var": "constants",
        "selection_dict": {"channel_c": "lsm"},
        "invert": True,
        "threshold": None,
    }
    stem = build_coupled_partial_conv_stem(
        {
            "kernel_size": 3,
            "eps": 1.0e-8,
            "masks": {"sst": mask_cfg, "sic": {**mask_cfg, "threshold": 0.5}},
        },
        _toy_couplings(),
        hpx_padding_mode=hpx_padding_mode,
        nside=NSIDE,
    )
    assert isinstance(stem, CoupledPartialConvStem)
    x = th.randn(2, N_FACES, 2, NSIDE, NSIDE)
    y = stem(x)
    assert y.shape == x.shape
    assert th.isfinite(y).all()


def test_hard_mask_fill_independence(tmp_path, ocean_frac):
    """Changing values only on hard-invalid cells must not change stem output."""
    zpath = tmp_path / "mask.zarr"
    _write_lsm_zarr(zpath, ocean_frac)
    hard_cfg = {
        "dataset_path": str(zpath),
        "data_var": "constants",
        "selection_dict": {"channel_c": "lsm"},
        "invert": True,
        "threshold": 0.5,
    }
    stem = build_coupled_partial_conv_stem(
        {"kernel_size": 3, "masks": {"sst": hard_cfg, "sic": hard_cfg}},
        _toy_couplings(),
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
    )
    # Freeze feature weights for a deterministic comparison
    with th.no_grad():
        w = stem.pconv._conv_weight(stem.pconv.feature_conv)
        w.fill_(1.0 / 9.0)
        if stem.pconv.bias is not None:
            stem.pconv.bias.zero_()

    # channel_masks buffer is [1, F, C, H, W]
    invalid = stem.channel_masks == 0.0

    x_a = th.randn(1, N_FACES, 2, NSIDE, NSIDE)
    x_b = th.where(invalid.expand_as(x_a), th.full_like(x_a, 123.0), x_a)
    y_a = stem(x_a)
    y_b = stem(x_b)
    assert th.allclose(y_a, y_b, atol=1e-5, rtol=1e-5)


def test_per_var_soft_vs_hard_masks_differ(tmp_path, ocean_frac):
    zpath = tmp_path / "mask.zarr"
    _write_lsm_zarr(zpath, ocean_frac)
    base = {
        "dataset_path": str(zpath),
        "data_var": "constants",
        "selection_dict": {"channel_c": "lsm"},
        "invert": True,
    }
    stem = build_coupled_partial_conv_stem(
        {
            "kernel_size": 3,
            "masks": {
                "sst": {**base, "threshold": None},
                "sic": {**base, "threshold": 0.5},
            },
        },
        _toy_couplings(),
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
    )
    # Soft channel (0) keeps fractional mass; hard channel (1) is binary.
    # channel_masks buffer layout: [F, C, H, W]
    soft_m = stem.channel_masks[:, 0]
    hard_m = stem.channel_masks[:, 1]
    assert th.any((soft_m > 0) & (soft_m < 1))
    assert set(hard_m.unique().tolist()) <= {0.0, 1.0}


def test_partial_conv_matches_standard_when_mask_ones():
    """With M=1 everywhere, PConv (bias=0) matches depthwise conv."""
    channels = 2
    pconv = PartialHEALPixConv2d(
        channels=channels,
        kernel_size=3,
        eps=1.0e-8,
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
        bias=False,
    )
    with th.no_grad():
        w = pconv._conv_weight(pconv.feature_conv)
        w.normal_(0, 0.1)

    x = th.randn(N_FACES, channels, NSIDE, NSIDE)
    mask = th.ones_like(x)
    y_p = pconv(x, mask)
    y_ref = pconv.feature_conv(x)
    assert th.allclose(y_p, y_ref, atol=1e-5, rtol=1e-5)


def test_disabled_stem_leaves_coupled_tensor_untouched_in_reshape_logic():
    """build(... None) is the disabled path used by RecUNet/UNet."""
    assert build_coupled_partial_conv_stem(None, _toy_couplings()) is None


def test_stem_hydra_target_default(tmp_path, ocean_frac):
    zpath = tmp_path / "mask.zarr"
    _write_lsm_zarr(zpath, ocean_frac)
    mask_cfg = {
        "dataset_path": str(zpath),
        "data_var": "constants",
        "selection_dict": {"channel_c": "lsm"},
        "invert": True,
        "threshold": 0.5,
    }
    stem = build_coupled_partial_conv_stem(
        {
            "masks": {"sst": mask_cfg, "sic": mask_cfg},
            "stem": {
                "_target_": DEFAULT_COUPLED_PARTIAL_CONV_STEM_TARGET,
                "kernel_size": 3,
                "dilation": 1,
                "eps": 1.0e-8,
                "bias": True,
            },
        },
        _toy_couplings(),
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
    )
    assert isinstance(stem, CoupledPartialConvStem)
    assert stem.pconv.kernel_size == 3
    assert stem.pconv.dilation == 1


def test_custom_stem_target(tmp_path, ocean_frac):
    zpath = tmp_path / "mask.zarr"
    _write_lsm_zarr(zpath, ocean_frac)
    mask_cfg = {
        "dataset_path": str(zpath),
        "data_var": "constants",
        "selection_dict": {"channel_c": "lsm"},
        "invert": True,
        "threshold": 0.5,
    }
    # Pass the class object so Hydra need not import via a pytest module path.
    stem = build_coupled_partial_conv_stem(
        {
            "masks": {"sst": mask_cfg, "sic": mask_cfg},
            "stem": {"_target_": _IdentityCoupledStem},
        },
        _toy_couplings(),
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
    )
    assert isinstance(stem, _IdentityCoupledStem)
    x = th.randn(1, N_FACES, 2, NSIDE, NSIDE)
    assert th.equal(stem(x), x)


def test_partial_conv_feature_kernels_are_even():
    """Depthwise feature kernels satisfy K ≈ R(K); bias is per-channel (even)."""
    pconv = PartialHEALPixConv2d(
        channels=3,
        kernel_size=3,
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
        bias=True,
    )
    pconv.eval()
    w = pconv._conv_weight(pconv.feature_conv)
    th.testing.assert_close(w, apply_R_kernel(w), atol=1e-6, rtol=1e-6)
    th.testing.assert_close(w, project_even_kernel(w), atol=1e-6, rtol=1e-6)
    assert pconv.bias is not None
    assert pconv.bias.shape == (3,)


def test_stem_intertwines_with_ones_mask():
    """With ρ-symmetric ones mask and even kernels, stem(ρx) ≈ ρ stem(x)."""
    channels = 2
    masks = th.ones(channels, N_FACES, NSIDE, NSIDE)
    stem = CoupledPartialConvStem(
        channel_masks=masks,
        kernel_size=3,
        bias=True,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=NSIDE,
    )
    stem.eval()
    th.manual_seed(0)
    x = th.randn(2, N_FACES, channels, NSIDE, NSIDE)
    with th.no_grad():
        y = stem(x)
        y_from_rx = stem(_reflect_bfchw(x))
        ry = _reflect_bfchw(y)
    th.testing.assert_close(y_from_rx, ry, atol=1e-5, rtol=1e-4)


def test_partial_conv_even_kernel_survives_adam_step():
    """One Adam update + eval() keeps feature kernels on the even manifold."""
    pconv = PartialHEALPixConv2d(
        channels=2,
        kernel_size=3,
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
        bias=False,
    )
    opt = th.optim.Adam(pconv.parameters(), lr=1e-2)
    x = th.randn(N_FACES, 2, NSIDE, NSIDE)
    mask = th.ones_like(x)
    pconv.train()
    loss = pconv(x, mask).pow(2).mean()
    loss.backward()
    opt.step()
    pconv.eval()
    w = pconv._conv_weight(pconv.feature_conv)
    th.testing.assert_close(w, apply_R_kernel(w), atol=1e-5, rtol=1e-5)


def test_odd_channel_bias_stays_zero_through_adam():
    """Odd channels keep even kernels but force bias=0 even after an Adam step."""
    pconv = PartialHEALPixConv2d(
        channels=2,
        kernel_size=3,
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
        bias=True,
        channel_is_odd=[False, True],
    )
    assert bool(pconv.channel_is_odd.tolist() == [False, True])
    pconv.eval()
    assert float(pconv.bias[1].item()) == 0.0

    opt = th.optim.Adam(pconv.parameters(), lr=1e-1)
    x = th.randn(N_FACES, 2, NSIDE, NSIDE)
    mask = th.ones_like(x)
    pconv.train()
    # Seed a nonzero odd bias; the grad hook + eval project must clear it.
    with th.no_grad():
        pconv.bias[:] = 1.0
    loss = pconv(x, mask).pow(2).mean()
    loss.backward()
    assert float(pconv.bias.grad[1].item()) == 0.0
    opt.step()
    pconv.eval()
    assert float(pconv.bias[1].abs().item()) < 1e-8
    w = pconv._conv_weight(pconv.feature_conv)
    th.testing.assert_close(w, apply_R_kernel(w), atol=1e-5, rtol=1e-5)


def test_stem_intertwines_mixed_even_odd_channels():
    """Ones mask + mixed parity: stem(ρx) ≈ ρ stem(x) with odd-channel sign flip."""
    channels = 2
    masks = th.ones(channels, N_FACES, NSIDE, NSIDE)
    stem = CoupledPartialConvStem(
        channel_masks=masks,
        kernel_size=3,
        bias=True,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=NSIDE,
        channel_is_odd=[False, True],
    )
    stem.eval()
    th.manual_seed(1)
    x = th.randn(2, N_FACES, channels, NSIDE, NSIDE)
    with th.no_grad():
        y = stem(x)
        y_from_rx = stem(_reflect_bfchw_typed(x, odd_channels=[1]))
        ry = _reflect_bfchw_typed(y, odd_channels=[1])
    th.testing.assert_close(y_from_rx, ry, atol=1e-5, rtol=1e-4)


def test_builder_wires_odd_coupled_variables(tmp_path, ocean_frac):
    """odd_coupled_variables expands to channel_is_odd on the built stem."""
    zpath = tmp_path / "mask.zarr"
    _write_lsm_zarr(zpath, ocean_frac)
    mask_cfg = {
        "dataset_path": str(zpath),
        "data_var": "constants",
        "selection_dict": {"channel_c": "lsm"},
        "invert": True,
        "threshold": 0.5,
    }
    stem = build_coupled_partial_conv_stem(
        {"masks": {"sst": mask_cfg, "v_coup": mask_cfg}},
        _toy_couplings(variables=("sst", "v_coup")),
        hpx_padding_mode="karlbauer",
        nside=NSIDE,
        odd_coupled_variables=["v_coup", "not_in_couplings"],
    )
    assert isinstance(stem, CoupledPartialConvStem)
    assert stem.pconv.channel_is_odd.tolist() == [False, True]
