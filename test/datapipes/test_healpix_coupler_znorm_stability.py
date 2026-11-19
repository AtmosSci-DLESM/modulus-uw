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

"""Numerical-stability tests for coupler ``set_coupled_fields`` in z-norm space.

Inference feeds already z-normalized tensors into ``set_coupled_fields``.
``TrailingAverageCoupler`` then averages over time in that normalized space.
For an affine transform ``(x - mean) / std`` this is algebraically equivalent to
averaging in physical space and then normalizing, but float32 arithmetic with
HPX64-scale means/stds (e.g. ``ttr`` ~ 1e6 vs ``q850`` ~ 1e-3) can diverge.

These tests draw random physical fields from the distributions in
``NV-dlesm/configs/data/scaling/hpx64.yaml`` and compare:

1. Coupler average in float32 z-norm space (production path)
2. float64 physical average, then normalize (high-accuracy reference)
3. float32 physical average, then normalize (physical-space float32 path)

so the magnitude of any z-norm vs physical discrepancy is explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

xr = pytest.importorskip("xarray")

from physicsnemo.datapipes.healpix.couplers import (  # noqa: E402
    ConstantCoupler,
    TrailingAverageCoupler,
)

# Subset of NV-dlesm/configs/data/scaling/hpx64.yaml used by coupled configs.
# Values are (mean, std) in physical units.
HPX64_SCALING = {
    "sst": (291.37762451171875, 10.202408790588379),
    "z1000": (937.166748046875, 901.9874877929688),
    "z1000-48H": (934.4945, 842.1188),
    "ws10m": (6.144942283630371, 3.6602368354797363),
    "ws10-48H": (6.081215, 3.1224248),
    "ttr": (-872297.06, 166696.2),
    "ttr-3h": (-2617476.9631782947, 494939.84),
    "t2m": (287.40704345703125, 15.39192008972168),
    "q850": (0.005995592568069696, 0.004197916481643915),
    "sic": (0.0, 1.0),
}

_FACE, _HEIGHT, _WIDTH = 4, 8, 8
_BATCH = 2
_SEED = 0


def _means_stds(variables):
    means = torch.tensor([HPX64_SCALING[v][0] for v in variables], dtype=torch.float64)
    stds = torch.tensor([HPX64_SCALING[v][1] for v in variables], dtype=torch.float64)
    return means, stds


def _timevar_means_stds(variables, n_input_times):
    """Tile per-variable stats across TrailingAverage's period-major timevar axis.

    Channel order after ``set_coupled_fields`` is
    ``[period0_var0, period0_var1, ..., period1_var0, ...]``.
    """
    means, stds = _means_stds(variables)
    return means.repeat(n_input_times), stds.repeat(n_input_times)


def _broadcast_stats(means, stds, ndim=6, channel_dim=3):
    """Reshape (C,) stats to broadcast over a 6-D tensor."""
    shape = [1] * ndim
    shape[channel_dim] = -1
    return means.view(*shape), stds.view(*shape)


def _sample_physical(variables, timedim, generator):
    """Draw ~N(mean, std) physical fields in float64, shape [B, F, T, C, H, W]."""
    means, stds = _means_stds(variables)
    means_b, stds_b = _broadcast_stats(means, stds)
    noise = torch.randn(
        _BATCH,
        _FACE,
        timedim,
        len(variables),
        _HEIGHT,
        _WIDTH,
        dtype=torch.float64,
        generator=generator,
    )
    return means_b + stds_b * noise


def _normalize(physical, variables, dtype=torch.float32):
    means, stds = _means_stds(variables)
    means_b, stds_b = _broadcast_stats(means, stds, channel_dim=3)
    return ((physical - means_b) / stds_b).to(dtype)


def _denormalize_timevar(znorm, variables, n_input_times, dtype=torch.float64):
    """Denormalize a time_first coupler output ``[I, B, timevar, F, H, W]``."""
    means, stds = _timevar_means_stds(variables, n_input_times)
    means_b, stds_b = _broadcast_stats(means, stds, channel_dim=2)
    return znorm.to(dtype) * stds_b + means_b


def _normalize_timevar(physical_avg, variables, n_input_times, dtype=torch.float32):
    """Normalize a time_first averaged tensor ``[I, B, timevar, F, H, W]``."""
    means, stds = _timevar_means_stds(variables, n_input_times)
    means_b, stds_b = _broadcast_stats(means, stds, channel_dim=2)
    return ((physical_avg - means_b) / stds_b).to(dtype)


def _make_dataset(variables, n_time=32):
    return xr.Dataset(
        data_vars={
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                np.zeros(
                    (n_time, len(variables), _FACE, _HEIGHT, _WIDTH),
                    dtype="float32",
                ),
            )
        },
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq="3h"),
            "channel_in": list(variables),
            "face": np.arange(_FACE),
            "height": np.arange(_HEIGHT),
            "width": np.arange(_WIDTH),
        },
    )


def _configure_trailing_average(coupler, input_times, data_time_step="3h"):
    averaging_window_max_indices = [
        pd.Timedelta(t) // pd.Timedelta(data_time_step) for t in input_times
    ]
    di = averaging_window_max_indices[0]
    averaging_slices = []
    for j in range(coupler.coupled_integration_dim):
        averaging_slices.append([])
        for i, r in enumerate(averaging_window_max_indices):
            averaging_slices[j].append(
                slice(
                    coupler.input_time_dim * j * di + i * di,
                    coupler.input_time_dim * j * di + r,
                )
            )
    coupler.averaging_slices = averaging_slices
    coupler.coupled_channel_indices = list(range(len(coupler.variables)))
    return averaging_slices


def _reference_trailing_average(physical, averaging_slices, dtype):
    """Mirror TrailingAverageCoupler.set_coupled_fields averaging in ``dtype``."""
    fields = physical.to(dtype)
    periods = []
    for slices in averaging_slices:
        averaged = [
            fields[:, :, s, :, :, :].mean(dim=2, keepdim=True) for s in slices
        ]
        periods.append(torch.concat(averaged, dim=3))
    # [B, F, integration, timevar, H, W] -> [integration, B, timevar, F, H, W]
    return torch.concat(periods, dim=2).permute(2, 0, 3, 1, 4, 5)


def _max_abs_err(a, b):
    return (a - b).abs().max().item()


def _per_channel_max_abs(a, b, variables):
    """Max abs error per channel; ``a``/``b`` are [I, B, C, F, H, W]."""
    return {
        var: (a[:, :, i] - b[:, :, i]).abs().max().item()
        for i, var in enumerate(variables)
    }


@pytest.mark.parametrize(
    "variables",
    [
        # Typical ocean->atmos ConstantCoupler field
        ["sst"],
        # Trailing-average ocean forcings from sst-z1000-ws.yaml
        ["z1000-48H", "ws10-48H"],
        # Wide dynamic range: OLM-scale OLR vs humidity vs SST
        ["ttr", "q850", "sst"],
        # Accumulated TTR is even larger in magnitude
        ["ttr-3h", "z1000", "ws10m"],
    ],
)
def test_trailing_average_znorm_matches_physical_float64_reference(variables):
    """Production z-norm float32 average should match float64 physical reference."""
    generator = torch.Generator().manual_seed(_SEED)
    input_times = ["6h", "12h"]
    # Need enough timesteps for the longest averaging window end (12h / 3h = 4)
    timedim = 8
    physical = _sample_physical(variables, timedim, generator)

    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(variables),
        batch_size=_BATCH,
        variables=variables,
        presteps=0,
        averaging_window="6h",
        input_times=input_times,
        input_time_dim=2,
        output_time_dim=2,
    )
    averaging_slices = _configure_trailing_average(coupler, input_times)

    znorm_f32 = _normalize(physical, variables, dtype=torch.float32)
    coupler.set_coupled_fields(znorm_f32)
    got = coupler.construct_integrated_couplings()

    # Reference: average in physical float64, then normalize.
    phys_avg_f64 = _reference_trailing_average(
        physical, averaging_slices, dtype=torch.float64
    )
    ref = _normalize_timevar(
        phys_avg_f64, variables, n_input_times=len(input_times), dtype=torch.float32
    )

    per_channel = _per_channel_max_abs(got, ref, variables * len(input_times))
    max_err = _max_abs_err(got, ref)

    # Affine equivalence should keep z-norm float32 within a few ULPs of the
    # float64 physical reference for O(1) standardized values. Extreme
    # physical magnitudes (ttr) still stay well under 1e-5 in z-space.
    assert max_err < 1e-5, (
        f"z-norm TrailingAverage diverged from physical float64 reference: "
        f"max_err={max_err:.3e}, per_channel={per_channel}"
    )


@pytest.mark.parametrize(
    "variables",
    [
        ["sst"],
        ["z1000-48H", "ws10-48H"],
        ["ttr", "q850", "sst"],
        ["ttr-3h", "z1000", "ws10m"],
    ],
)
def test_trailing_average_znorm_vs_physical_float32_error(variables):
    """Quantify whether z-norm float32 averaging is stabler than physical float32.

    Both are compared against a float64 physical reference. For large-magnitude
    variables (``ttr``, ``ttr-3h``), averaging in physical float32 before
    normalizing typically accumulates more error than averaging already
    standardized values.
    """
    generator = torch.Generator().manual_seed(_SEED + 1)
    input_times = ["6h", "12h"]
    timedim = 8
    physical = _sample_physical(variables, timedim, generator)

    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(variables),
        batch_size=_BATCH,
        variables=variables,
        presteps=0,
        averaging_window="6h",
        input_times=input_times,
        input_time_dim=2,
        output_time_dim=2,
    )
    averaging_slices = _configure_trailing_average(coupler, input_times)
    n_times = len(input_times)
    timevar_names = variables * n_times

    # Production path
    coupler.set_coupled_fields(_normalize(physical, variables, dtype=torch.float32))
    znorm_path = coupler.construct_integrated_couplings().to(torch.float64)

    # Physical float32 path: cast -> average -> normalize
    phys_avg_f32 = _reference_trailing_average(
        physical, averaging_slices, dtype=torch.float32
    ).to(torch.float64)
    physical_f32_path = _normalize_timevar(
        phys_avg_f32, variables, n_input_times=n_times, dtype=torch.float64
    )

    # Reference
    phys_avg_f64 = _reference_trailing_average(
        physical, averaging_slices, dtype=torch.float64
    )
    reference = _normalize_timevar(
        phys_avg_f64, variables, n_input_times=n_times, dtype=torch.float64
    )

    err_znorm = _max_abs_err(znorm_path, reference)
    err_physical_f32 = _max_abs_err(physical_f32_path, reference)
    per_channel_znorm = _per_channel_max_abs(znorm_path, reference, timevar_names)
    per_channel_phys = _per_channel_max_abs(physical_f32_path, reference, timevar_names)

    # z-norm path should never be dramatically worse than physical float32.
    assert err_znorm <= err_physical_f32 * 2.0 + 1e-7, (
        f"z-norm averaging unexpectedly less stable than physical float32: "
        f"err_znorm={err_znorm:.3e}, err_physical_f32={err_physical_f32:.3e}, "
        f"per_channel_znorm={per_channel_znorm}, per_channel_phys={per_channel_phys}"
    )


def test_trailing_average_ocean_forcing_config_znorm_roundtrip():
    """sst-z1000-ws style TrailingAverageCoupler: z-norm avg denorms to physical avg."""
    variables = ["z1000-48H", "ws10-48H"]
    # Match configs/data/module/sst-z1000-ws.yaml cadence with a compact window
    # that still exercises multi-step averaging on 3h data.
    input_times = ["48h", "96h"]
    data_time_step = "3h"
    # longest window end is 96h / 3h = 32 steps
    timedim = 40
    generator = torch.Generator().manual_seed(42)
    physical = _sample_physical(variables, timedim, generator)

    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(variables, n_time=64),
        batch_size=_BATCH,
        variables=variables,
        presteps=0,
        averaging_window="48h",
        input_times=input_times,
        input_time_dim=1,
        output_time_dim=1,
    )
    averaging_slices = _configure_trailing_average(
        coupler, input_times, data_time_step=data_time_step
    )

    coupler.set_coupled_fields(_normalize(physical, variables, dtype=torch.float32))
    znorm_avg = coupler.construct_integrated_couplings()
    denormed = _denormalize_timevar(
        znorm_avg, variables, n_input_times=len(input_times), dtype=torch.float64
    )

    phys_avg = _reference_trailing_average(
        physical, averaging_slices, dtype=torch.float64
    )
    # Coupler returns time_first [I, B, timevar, F, H, W]; reference matches that layout.
    max_err = _max_abs_err(denormed, phys_avg)
    per_channel = _per_channel_max_abs(
        denormed, phys_avg, variables * len(input_times)
    )

    # Physical units: z1000-48H std ~ 840, ws10-48H std ~ 3. Allow a small
    # absolute error after the float32 z-norm roundtrip.
    assert max_err < 1e-2, (
        f"denorm(z-norm average) drifted from physical average: "
        f"max_err={max_err:.3e}, per_channel={per_channel}"
    )


@pytest.mark.parametrize(
    "variables",
    [
        ["sst"],
        ["sst", "sic"],
        ["ttr", "q850", "sst"],
    ],
)
def test_constant_coupler_preserves_znorm_values(variables):
    """ConstantCoupler must not corrupt z-norm fields when broadcasting time 0."""
    generator = torch.Generator().manual_seed(_SEED + 2)
    timedim = 4
    physical = _sample_physical(variables, timedim, generator)
    znorm = _normalize(physical, variables, dtype=torch.float32)

    coupler = ConstantCoupler(
        dataset=_make_dataset(variables),
        batch_size=_BATCH,
        variables=variables,
        input_times=["0h"],
        input_time_dim=1,
        output_time_dim=3,
        presteps=0,
    )
    coupler.coupled_channel_indices = list(range(len(variables)))
    coupler.set_coupled_fields(znorm)
    out = coupler.construct_integrated_couplings()

    # time_first layout: [integration, B, C, F, H, W]
    expected = znorm[:, :, :1, :, :, :].permute(2, 0, 3, 1, 4, 5)
    expected = expected.expand(coupler.coupled_integration_dim, -1, -1, -1, -1, -1)

    assert out.shape[0] == coupler.coupled_integration_dim
    assert torch.equal(out, expected), (
        f"ConstantCoupler altered z-norm values: "
        f"max_err={_max_abs_err(out, expected):.3e}, "
        f"per_channel={_per_channel_max_abs(out, expected, variables)}"
    )


def test_mixed_scale_channels_remain_independent_in_znorm_average():
    """Channels with wildly different physical scales must not leak into each other."""
    variables = ["ttr-3h", "q850", "sst"]
    generator = torch.Generator().manual_seed(7)
    input_times = ["6h", "12h"]
    timedim = 8

    # Construct physical fields with *independent* constant-per-channel values
    # so any cross-channel bleed is obvious after averaging.
    means, stds = _means_stds(variables)
    # Use mean + k*std so each channel is exactly k in z-space.
    k = torch.tensor([-2.0, 0.5, 1.25], dtype=torch.float64)
    physical = (
        means.view(1, 1, 1, -1, 1, 1) + stds.view(1, 1, 1, -1, 1, 1) * k.view(1, 1, 1, -1, 1, 1)
    ).expand(_BATCH, _FACE, timedim, -1, _HEIGHT, _WIDTH).contiguous()
    # Add small independent noise so the average is not trivially constant in time
    noise = torch.randn(
        _BATCH,
        _FACE,
        timedim,
        len(variables),
        _HEIGHT,
        _WIDTH,
        dtype=torch.float64,
        generator=generator,
    )
    physical = physical + stds.view(1, 1, 1, -1, 1, 1) * 0.01 * noise

    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(variables),
        batch_size=_BATCH,
        variables=variables,
        presteps=0,
        averaging_window="6h",
        input_times=input_times,
        input_time_dim=2,
        output_time_dim=2,
    )
    averaging_slices = _configure_trailing_average(coupler, input_times)

    coupler.set_coupled_fields(_normalize(physical, variables, dtype=torch.float32))
    got = coupler.construct_integrated_couplings().to(torch.float64)

    phys_avg = _reference_trailing_average(
        physical, averaging_slices, dtype=torch.float64
    )
    ref = _normalize_timevar(
        phys_avg, variables, n_input_times=len(input_times), dtype=torch.float64
    )

    for period in range(len(input_times)):
        for i, var in enumerate(variables):
            timevar_idx = period * len(variables) + i
            err = (got[:, :, timevar_idx] - ref[:, :, timevar_idx]).abs().max().item()
            assert err < 1e-5, (
                f"channel {var!r} period {period} leaked or drifted: err={err:.3e}"
            )
            # Each channel's mean z-value should stay near its planted k, not near
            # another channel's k (guards against last-channel / order bugs).
            channel_mean = got[:, :, timevar_idx].mean().item()
            assert abs(channel_mean - k[i].item()) < 0.05, (
                f"channel {var!r} period {period} mean z={channel_mean:.4f} drifted "
                f"from planted k={k[i].item():.4f} toward another channel"
            )
