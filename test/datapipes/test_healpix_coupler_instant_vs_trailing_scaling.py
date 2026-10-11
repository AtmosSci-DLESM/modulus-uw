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

"""Instant vs trailing-mean scaling mismatch for TrailingAverageCoupler.

Coupled ocean forcings (``sst-z1000-ws.yaml``) average atmospheric fields over
48H / 96H windows. Instantaneous model outputs are typically z-scored with
``z1000`` / ``ws10m`` stats, while the prepared trailing fields use
``z1000-48H`` / ``ws10-48H``.

These tests compare:

1. **Physical trailing mean** (no normalization) — reference
2. **Mismatched path**: z-norm with *instant* stats → average in z-space →
   denorm with *trailing* stats (``-48H`` / synthetic ``-96H`` per period)
3. **Matched path**: z-norm and denorm with the same trailing stats

HPX64 only publishes ``*-48H`` trailing stats (no ``*-96H``). A synthetic
``*-96H`` pair is included solely so the 96H input-time slot can use a
distinct denorm, matching the user's requested instant→48H/96H setup.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

xr = pytest.importorskip("xarray")

from physicsnemo.datapipes.healpix.couplers import TrailingAverageCoupler  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "znorm_stability",
    Path(__file__).with_name("test_healpix_coupler_znorm_stability.py"),
)
_znorm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_znorm)

HPX64_SCALING = _znorm.HPX64_SCALING
_BATCH = _znorm._BATCH
_FACE = _znorm._FACE
_HEIGHT = _znorm._HEIGHT
_WIDTH = _znorm._WIDTH
_SEED = _znorm._SEED
_broadcast_stats = _znorm._broadcast_stats
_configure_trailing_average = _znorm._configure_trailing_average
_make_dataset = _znorm._make_dataset
_max_abs_err = _znorm._max_abs_err
_reference_trailing_average = _znorm._reference_trailing_average

# Instantaneous stats from hpx64.yaml
INSTANT_VARS = ["z1000", "ws10m"]

# Trailing-window stats from hpx64.yaml (48H only in the file)
TRAILING_48H_VARS = ["z1000-48H", "ws10-48H"]

# Synthetic 96H stats: longer window → mean between instant and 48H, slightly
# smaller std. NOT in hpx64.yaml — used only for period-specific denorm demos.
_SYNTHETIC_96H = {
    "z1000-96H": (
        0.5 * (HPX64_SCALING["z1000"][0] + HPX64_SCALING["z1000-48H"][0]),
        HPX64_SCALING["z1000-48H"][1] * 0.92,
    ),
    "ws10-96H": (
        0.5 * (HPX64_SCALING["ws10m"][0] + HPX64_SCALING["ws10-48H"][0]),
        HPX64_SCALING["ws10-48H"][1] * 0.92,
    ),
}

SCALING = {**HPX64_SCALING, **_SYNTHETIC_96H}

# Period-major denorm keys for input_times ["48h", "96h"]
# channel order after coupler: [p0_z1000, p0_ws10, p1_z1000, p1_ws10]
TRAILING_DENORM_BY_PERIOD = [
    ["z1000-48H", "ws10-48H"],  # 48h window
    ["z1000-96H", "ws10-96H"],  # 96h window (synthetic)
]


def _stats(keys):
    means = torch.tensor([SCALING[k][0] for k in keys], dtype=torch.float64)
    stds = torch.tensor([SCALING[k][1] for k in keys], dtype=torch.float64)
    return means, stds


def _normalize_with(physical, keys, dtype=torch.float32):
    means, stds = _stats(keys)
    means_b, stds_b = _broadcast_stats(means, stds, channel_dim=3)
    return ((physical - means_b) / stds_b).to(dtype)


def _denorm_timevar_with(znorm, keys, dtype=torch.float64):
    """Denorm time_first [I, B, timevar, ...] with one mean/std per timevar slot."""
    means, stds = _stats(keys)
    means_b, stds_b = _broadcast_stats(means, stds, channel_dim=2)
    return znorm.to(dtype) * stds_b + means_b


def _flatten_period_keys(period_keys):
    """[[v0,v1], [v0,v1]] → [v0,v1,v0,v1] matching coupler timevar order."""
    return [k for period in period_keys for k in period]


def _sample_instant_physical(timedim, generator):
    """Draw fields from instantaneous N(μ,σ) — like raw atmos output."""
    means, stds = _stats(INSTANT_VARS)
    means_b, stds_b = _broadcast_stats(means, stds, channel_dim=3)
    noise = torch.randn(
        _BATCH,
        _FACE,
        timedim,
        len(INSTANT_VARS),
        _HEIGHT,
        _WIDTH,
        dtype=torch.float64,
        generator=generator,
    )
    return means_b + stds_b * noise


def _run_trailing_average(physical_or_znorm, input_times=("48h", "96h")):
    """Apply TrailingAverageCoupler.set_coupled_fields; return time_first tensor."""
    # Coupler only needs spatial dims from the dataset; channel names are local.
    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(INSTANT_VARS, n_time=64),
        batch_size=_BATCH,
        variables=INSTANT_VARS,
        presteps=0,
        averaging_window="48h",
        input_times=list(input_times),
        input_time_dim=1,
        output_time_dim=1,
    )
    slices = _configure_trailing_average(
        coupler, list(input_times), data_time_step="3h"
    )
    coupler.set_coupled_fields(physical_or_znorm.to(torch.float32))
    return coupler.construct_integrated_couplings(), slices


def test_mismatched_instant_norm_trailing_denorm_vs_physical_mean():
    """Norm with z1000/ws10m, denorm with *-48H/*-96H ≠ physical trailing mean."""
    generator = torch.Generator().manual_seed(_SEED + 10)
    input_times = ("48h", "96h")
    timedim = 40  # covers 96h / 3h = 32 steps
    physical = _sample_instant_physical(timedim, generator)

    # Reference: trailing mean in physical space (no normalization).
    _, slices = _run_trailing_average(physical, input_times)
    physical_mean = _reference_trailing_average(
        physical, slices, dtype=torch.float64
    )

    # Mismatched path: instant z-norm → average → trailing denorm (48H/96H).
    znorm_instant = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)
    znorm_avg, _ = _run_trailing_average(znorm_instant, input_times)
    denorm_keys = _flatten_period_keys(TRAILING_DENORM_BY_PERIOD)
    mismatched = _denorm_timevar_with(znorm_avg, denorm_keys)

    # Matched control: trailing-48H z-norm → average → same trailing-48H denorm.
    # (Both periods use -48H here — the only trailing stats in hpx64.yaml.)
    matched_keys_flat = TRAILING_48H_VARS * len(input_times)
    znorm_trail = _normalize_with(physical, TRAILING_48H_VARS, dtype=torch.float32)
    znorm_avg_matched, _ = _run_trailing_average(znorm_trail, input_times)
    matched = _denorm_timevar_with(znorm_avg_matched, matched_keys_flat)

    err_mismatch = _max_abs_err(mismatched, physical_mean)
    err_matched = _max_abs_err(matched, physical_mean)

    # Systematic bias from μ/σ swap should dwarf float32 noise.
    assert err_mismatch > 1.0, (
        f"expected large systematic error from instant↔trailing stat mismatch, "
        f"got {err_mismatch:.3e}"
    )
    assert err_matched < 1e-2, (
        f"matched trailing stats should recover physical mean, got {err_matched:.3e}"
    )
    assert err_mismatch > 100 * err_matched


def test_mismatched_error_is_dominated_by_mean_shift():
    """Closed-form check: denorm_t(mean(norm_i(x))) bias follows the affine map."""
    generator = torch.Generator().manual_seed(_SEED + 11)
    input_times = ("48h", "96h")
    timedim = 40
    physical = _sample_instant_physical(timedim, generator)

    _, slices = _run_trailing_average(physical, input_times)
    physical_mean = _reference_trailing_average(
        physical, slices, dtype=torch.float64
    )

    znorm_instant = _normalize_with(physical, INSTANT_VARS, dtype=torch.float64)
    # Average in float64 to isolate scaling mismatch from float32 noise.
    znorm_avg = _reference_trailing_average(
        znorm_instant, slices, dtype=torch.float64
    )
    denorm_keys = _flatten_period_keys(TRAILING_DENORM_BY_PERIOD)
    mismatched = _denorm_timevar_with(znorm_avg, denorm_keys)

    means_i, stds_i = _stats(INSTANT_VARS)
    # Per timevar slot: period-major [z1000, ws10, z1000, ws10]
    for p, period_keys in enumerate(TRAILING_DENORM_BY_PERIOD):
        for v, trail_key in enumerate(period_keys):
            tv = p * len(INSTANT_VARS) + v
            mu_i, sig_i = means_i[v], stds_i[v]
            mu_t, sig_t = SCALING[trail_key]
            # predicted = (mean_x - μ_i) / σ_i * σ_t + μ_t
            predicted = (physical_mean[:, :, tv] - mu_i) / sig_i * sig_t + mu_t
            got = mismatched[:, :, tv]
            assert torch.allclose(got, predicted, rtol=0, atol=1e-9), (
                f"slot {trail_key}@{['48h','96h'][p]} affine mismatch"
            )


def test_per_channel_mismatch_errors_reported():
    """Collect per-slot errors for charting / presentation."""
    generator = torch.Generator().manual_seed(42)
    input_times = ("48h", "96h")
    physical = _sample_instant_physical(40, generator)

    _, slices = _run_trailing_average(physical, input_times)
    physical_mean = _reference_trailing_average(
        physical, slices, dtype=torch.float64
    )

    znorm_avg, _ = _run_trailing_average(
        _normalize_with(physical, INSTANT_VARS, dtype=torch.float32),
        input_times,
    )
    mismatched = _denorm_timevar_with(
        znorm_avg, _flatten_period_keys(TRAILING_DENORM_BY_PERIOD)
    )

    matched_avg, _ = _run_trailing_average(
        _normalize_with(physical, TRAILING_48H_VARS, dtype=torch.float32),
        input_times,
    )
    matched = _denorm_timevar_with(matched_avg, TRAILING_48H_VARS * 2)

    periods = ["48h", "96h"]
    rows = []
    for p, period in enumerate(periods):
        for v, instant_key in enumerate(INSTANT_VARS):
            tv = p * len(INSTANT_VARS) + v
            trail_key = TRAILING_DENORM_BY_PERIOD[p][v]
            err_m = (mismatched[:, :, tv] - physical_mean[:, :, tv]).abs()
            err_ok = (matched[:, :, tv] - physical_mean[:, :, tv]).abs()
            rows.append(
                {
                    "period": period,
                    "instant": instant_key,
                    "trailing": trail_key,
                    "mismatch_max": err_m.max().item(),
                    "mismatch_mean": err_m.mean().item(),
                    "matched_max": err_ok.max().item(),
                    "physical_std": physical_mean[:, :, tv].std().item(),
                }
            )

    # z1000 mismatch should be O(10–100+) in geopotential meters given μ/σ swap.
    z_rows = [r for r in rows if r["instant"] == "z1000"]
    assert all(r["mismatch_max"] > 1.0 for r in z_rows)
    assert all(r["matched_max"] < 1e-2 for r in rows)
