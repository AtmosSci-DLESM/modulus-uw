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

"""Adversarial tests for coupler incoming/outgoing coupled-field scaling.

When instantaneous model outputs are z-scored with ``z1000`` / ``ws10m`` but the
coupled training target uses trailing-window stats (``z1000-48H``, …), averaging
in z-space and denorming with trailing stats is systematically wrong.

``set_coupled_scaling(incoming)`` (outgoing from ``coupled_scaling``) fixes this by denormalizing →
operating in physical space → renormalizing inside ``set_coupled_fields``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

xr = pytest.importorskip("xarray")

from physicsnemo.datapipes.healpix.couplers import (  # noqa: E402
    ConstantCoupler,
    TrailingAverageCoupler,
)

_znorm_spec = importlib.util.spec_from_file_location(
    "znorm_stability",
    Path(__file__).with_name("test_healpix_coupler_znorm_stability.py"),
)
_znorm = importlib.util.module_from_spec(_znorm_spec)
_znorm_spec.loader.exec_module(_znorm)

_ivt_spec = importlib.util.spec_from_file_location(
    "instant_vs_trailing",
    Path(__file__).with_name("test_healpix_coupler_instant_vs_trailing_scaling.py"),
)
_ivt = importlib.util.module_from_spec(_ivt_spec)
_ivt_spec.loader.exec_module(_ivt)

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

INSTANT_VARS = _ivt.INSTANT_VARS
TRAILING_48H_VARS = _ivt.TRAILING_48H_VARS
TRAILING_DENORM_BY_PERIOD = _ivt.TRAILING_DENORM_BY_PERIOD
SCALING = _ivt.SCALING
_stats = _ivt._stats
_normalize_with = _ivt._normalize_with
_denorm_timevar_with = _ivt._denorm_timevar_with
_flatten_period_keys = _ivt._flatten_period_keys
_sample_instant_physical = _ivt._sample_instant_physical

INPUT_TIMES = ("48h", "96h")
TIMEDIM = 40  # covers 96h / 3h = 32 steps


def _yaml_scaling(keys=None):
    """Build an hpx64.yaml-style ``{name: {mean, std}}`` mapping."""
    items = SCALING.items() if keys is None else ((k, SCALING[k]) for k in keys)
    return {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in items}


def _scaling_da(keys=None):
    """xarray form produced by CoupledDataset from the YAML mapping."""
    import pandas as pd

    return (
        pd.DataFrame.from_dict(_yaml_scaling(keys)).T.to_xarray().astype("float32")
    )


def _scaling_da_from_mapping(scaling_map):
    """xarray scaling_da from an already-built ``{name: {mean, std}}`` dict."""
    import pandas as pd

    return pd.DataFrame.from_dict(scaling_map).T.to_xarray().astype("float32")


def _make_trailing_coupler(
    *,
    incoming=None,
    variables=None,
    incoming_variables=None,
    input_times=None,
    install_coupled_scaling=True,
):
    """Build a TrailingAverageCoupler.

    ``variables`` are the coupler/outgoing keys (default trailing-48H).
    ``incoming_variables`` are source-module names from setup_coupling
    (default instant keys). Tensor channels are still length
    ``len(incoming_variables)`` with indices ``range(n)``.

    When ``incoming`` is set, ``set_scaling`` is called first (outgoing base),
    then ``set_coupled_scaling(incoming)``.
    """
    variables = list(variables or TRAILING_48H_VARS)
    incoming_variables = list(incoming_variables or INSTANT_VARS)
    input_times = list(input_times or INPUT_TIMES)
    assert len(variables) == len(incoming_variables)
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
    # Simulate setup_coupling without a full coupled module.
    coupler.coupled_channel_indices = list(range(len(incoming_variables)))
    coupler.incoming_variables = incoming_variables
    coupler.outgoing_variable_order = list(variables)
    if install_coupled_scaling:
        coupler.set_scaling(_scaling_da(variables))
    if incoming is not None:
        coupler.set_coupled_scaling(incoming)
    _configure_trailing_average(coupler, input_times, data_time_step="3h")
    coupler.coupled_channel_indices = list(range(len(incoming_variables)))
    return coupler


def _make_constant_coupler(
    *,
    incoming=None,
    variables=None,
    incoming_variables=None,
    install_coupled_scaling=True,
):
    variables = list(variables or TRAILING_48H_VARS)
    incoming_variables = list(incoming_variables or INSTANT_VARS)
    assert len(variables) == len(incoming_variables)
    coupler = ConstantCoupler(
        dataset=_make_dataset(variables),
        batch_size=_BATCH,
        variables=variables,
        input_times=["0h"],
        input_time_dim=1,
        output_time_dim=3,
        presteps=0,
    )
    coupler.coupled_channel_indices = list(range(len(incoming_variables)))
    coupler.incoming_variables = incoming_variables
    coupler.outgoing_variable_order = list(variables)
    if install_coupled_scaling:
        coupler.set_scaling(_scaling_da(variables))
    if incoming is not None:
        coupler.set_coupled_scaling(incoming)
    return coupler


def _renorm_timevar_physical(physical_avg, keys, dtype=torch.float32):
    """Normalize time_first [I, B, timevar, ...] with per-slot outgoing stats."""
    means, stds = _stats(keys)
    means_b, stds_b = _broadcast_stats(means, stds, channel_dim=2)
    return ((physical_avg - means_b) / stds_b).to(dtype)


# ---------------------------------------------------------------------------
# 1. API validation
# ---------------------------------------------------------------------------


def test_set_coupled_scaling_none_raises():
    coupler = _make_trailing_coupler()
    with pytest.raises(ValueError, match="incoming_coupled_scaling must be provided"):
        coupler.set_coupled_scaling(None)


def test_set_coupled_scaling_requires_setup_coupling():
    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(TRAILING_48H_VARS),
        batch_size=_BATCH,
        variables=TRAILING_48H_VARS,
        averaging_window="48h",
        input_times=list(INPUT_TIMES),
        input_time_dim=1,
        output_time_dim=1,
    )
    coupler.set_scaling(_scaling_da(TRAILING_48H_VARS))
    with pytest.raises(RuntimeError, match="setup_coupling must be called"):
        coupler.set_coupled_scaling(_yaml_scaling())


def test_set_coupled_scaling_requires_set_scaling():
    coupler = _make_trailing_coupler(install_coupled_scaling=False)
    with pytest.raises(RuntimeError, match="set_scaling must be called"):
        coupler.set_coupled_scaling(_yaml_scaling())


def test_flat_mean_std_arrays_rejected():
    with pytest.raises(TypeError, match="variable-keyed"):
        _make_trailing_coupler(
            incoming={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        )


def test_missing_variable_keys_raises():
    with pytest.raises(KeyError, match="missing variables"):
        _make_trailing_coupler(
            incoming={"z1000": {"mean": 0.0, "std": 1.0}},  # missing ws10m
        )


def test_entry_missing_mean_or_std_raises():
    with pytest.raises(TypeError, match="mean.*std"):
        _make_trailing_coupler(
            incoming={
                "z1000": {"mean": 0.0},  # missing std
                "ws10m": {"mean": 0.0, "std": 1.0},
            },
        )


def test_zero_std_raises():
    bad = _yaml_scaling(INSTANT_VARS)
    bad["ws10m"]["std"] = 0.0
    with pytest.raises(ValueError, match="std must be non-zero"):
        _make_trailing_coupler(incoming=bad)


def test_use_coupled_field_rescaling_property():
    bare = _make_trailing_coupler()
    assert bare.use_coupled_field_rescaling is False
    scaled = _make_trailing_coupler(incoming=_yaml_scaling())
    assert scaled.use_coupled_field_rescaling is True
    assert scaled.outgoing_coupled_scaling["mean"].shape[3] == scaled.output_channels


def test_scaling_da_xarray_form_accepted():
    coupler = _make_trailing_coupler(incoming=_scaling_da())
    assert coupler.use_coupled_field_rescaling is True
    assert coupler.outgoing_coupled_scaling["mean"].shape[3] == coupler.output_channels


def test_outgoing_comes_from_coupled_scaling():
    """Outgoing renorm stats must match set_scaling / coupled_scaling, tiled."""
    coupler = _make_trailing_coupler(incoming=_yaml_scaling())
    cs_mean = coupler.coupled_scaling["mean"].reshape(-1)
    out_mean = coupler.outgoing_coupled_scaling["mean"].flatten().numpy()
    expected = np.tile(cs_mean, len(INPUT_TIMES))
    assert np.allclose(out_mean, expected)


def test_setup_coupling_sets_incoming_variables():
    """Matched source output_variables become incoming_variables."""
    scaling = _yaml_scaling()
    scaling["ws10m-48H"] = dict(scaling["ws10-48H"])

    class _FakeModule:
        output_variables = ["sst", "z1000", "ws10m", "t2m"]
        time_step = "3h"

    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(["z1000-48H", "ws10m-48H"], n_time=8),
        batch_size=1,
        variables=["z1000-48H", "ws10m-48H"],
        averaging_window="48h",
        input_times=["48h", "96h"],
        input_time_dim=1,
        output_time_dim=1,
    )
    coupler.setup_coupling(_FakeModule())
    assert coupler.incoming_variables == ["z1000", "ws10m"]
    assert coupler.outgoing_variable_order == ["z1000-48H", "ws10m-48H"]
    assert coupler.coupled_channel_indices == [1, 2]
    coupler.set_scaling(_scaling_da_from_mapping(scaling))
    coupler.set_coupled_scaling(scaling)
    assert abs(
        coupler.incoming_coupled_scaling["mean"].flatten()[0].item()
        - SCALING["z1000"][0]
    ) < 1e-3
    assert abs(
        coupler.outgoing_coupled_scaling["mean"].flatten()[0].item()
        - SCALING["z1000-48H"][0]
    ) < 1e-3
    # Duplicated across 2 input_times.
    assert coupler.outgoing_coupled_scaling["mean"].shape[3] == 4


# ---------------------------------------------------------------------------
# 2. Backward compatibility
# ---------------------------------------------------------------------------


def test_scaling_follows_post_index_channel_order():
    """If source channel order differs from self.variables, scaling must follow indices."""
    scaling = _yaml_scaling()
    scaling["ws10m-48H"] = dict(scaling["ws10-48H"])

    class _FakeModule:
        # ws10m appears *before* z1000 in the source output
        output_variables = ["sst", "ws10m", "z1000", "t2m"]
        time_step = "3h"

    # Coupler variables listed z1000-first (opposite of source order).
    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(["z1000-48H", "ws10m-48H"], n_time=64),
        batch_size=_BATCH,
        variables=["z1000-48H", "ws10m-48H"],
        averaging_window="48h",
        input_times=["48h"],
        input_time_dim=1,
        output_time_dim=1,
    )
    coupler.setup_coupling(_FakeModule())
    assert coupler.coupled_channel_indices == [1, 2]  # ws10m, z1000 in source order
    assert coupler.incoming_variables == ["ws10m", "z1000"]
    assert coupler.outgoing_variable_order == ["ws10m-48H", "z1000-48H"]
    coupler.set_scaling(_scaling_da_from_mapping(scaling))
    coupler.set_coupled_scaling(scaling)

    # Build a full source-shaped tensor and fill only the matched channels.
    # Channel layout matches FakeModule.output_variables.
    timedim = 20
    full = torch.zeros(_BATCH, _FACE, timedim, 4, _HEIGHT, _WIDTH, dtype=torch.float32)
    # After index [1,2] → [ws10m, z1000]. Plant distinct z-scores.
    full[:, :, :, 1, :, :] = 1.0   # ws10m = +1σ
    full[:, :, :, 2, :, :] = 2.0   # z1000 = +2σ
    # setup_coupling already set averaging_slices; do not call
    # _configure_trailing_average (it would reset coupled_channel_indices).
    coupler.set_coupled_fields(full)
    out = coupler.construct_integrated_couplings()  # [I,B,timevar,F,H,W]

    # Denorm with outgoing order [ws10m-48H, z1000-48H] should recover
    # physical = μ + σ * z for each post-index channel.
    means_o = torch.tensor(
        [scaling["ws10m-48H"]["mean"], scaling["z1000-48H"]["mean"]],
        dtype=torch.float64,
    )
    stds_o = torch.tensor(
        [scaling["ws10m-48H"]["std"], scaling["z1000-48H"]["std"]],
        dtype=torch.float64,
    )
    means_i = torch.tensor(
        [scaling["ws10m"]["mean"], scaling["z1000"]["mean"]],
        dtype=torch.float64,
    )
    stds_i = torch.tensor(
        [scaling["ws10m"]["std"], scaling["z1000"]["std"]],
        dtype=torch.float64,
    )
    phys = means_i + stds_i * torch.tensor([1.0, 2.0], dtype=torch.float64)
    expected_z = ((phys - means_o) / stds_o).to(torch.float32)
    got = out[0, 0, :, 0, 0, 0]  # [timevar]
    assert torch.allclose(got, expected_z, rtol=0, atol=1e-4), (
        f"order mismatch: got {got.tolist()}, expected {expected_z.tolist()}"
    )


def test_trailing_average_without_scaling_unchanged():
    generator = torch.Generator().manual_seed(_SEED + 20)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)

    bare = _make_trailing_coupler(
        variables=INSTANT_VARS, incoming_variables=INSTANT_VARS
    )
    bare.set_coupled_fields(znorm.clone())
    out_bare = bare.construct_integrated_couplings()

    # Sanity: matches reference average in z-space (old path).
    slices = bare.averaging_slices
    ref = _reference_trailing_average(znorm.to(torch.float64), slices, dtype=torch.float32)
    assert _max_abs_err(out_bare, ref) < 1e-5
    assert bare.use_coupled_field_rescaling is False


def test_constant_without_scaling_unchanged():
    generator = torch.Generator().manual_seed(_SEED + 21)
    means, stds = _stats(INSTANT_VARS)
    noise = torch.randn(
        _BATCH, _FACE, 4, len(INSTANT_VARS), _HEIGHT, _WIDTH,
        dtype=torch.float64, generator=generator,
    )
    physical = means.view(1, 1, 1, -1, 1, 1) + stds.view(1, 1, 1, -1, 1, 1) * noise
    znorm = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)

    bare = _make_constant_coupler(
        variables=INSTANT_VARS, incoming_variables=INSTANT_VARS
    )
    bare.set_coupled_fields(znorm.clone())
    out = bare.construct_integrated_couplings()

    expected = znorm[:, :, :1].permute(2, 0, 3, 1, 4, 5).expand(
        bare.coupled_integration_dim, -1, -1, -1, -1, -1
    )
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# 3. Core fix: denorm → avg → renorm recovers physical trailing mean
# ---------------------------------------------------------------------------


def test_rescaling_recovers_physical_trailing_mean():
    """Instant-norm input + duplicated trailing outgoing stats → physical mean."""
    generator = torch.Generator().manual_seed(_SEED + 30)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm_instant = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)

    outgoing_keys = TRAILING_48H_VARS * len(INPUT_TIMES)
    coupler = _make_trailing_coupler(
        incoming=_yaml_scaling(),
    )
    slices = coupler.averaging_slices
    coupler.set_coupled_fields(znorm_instant)
    znorm_out = coupler.construct_integrated_couplings()

    # Coupler output is in outgoing z-space; denorm to compare with physical mean.
    recovered = _denorm_timevar_with(znorm_out, outgoing_keys)
    physical_mean = _reference_trailing_average(
        physical, slices, dtype=torch.float64
    )
    err_with_api = _max_abs_err(recovered, physical_mean)

    # Contrast: old mismatched path (instant norm → avg → trailing denorm).
    bare = _make_trailing_coupler(
        variables=INSTANT_VARS, incoming_variables=INSTANT_VARS
    )
    bare.set_coupled_fields(znorm_instant.clone())
    mismatched = _denorm_timevar_with(
        bare.construct_integrated_couplings(), outgoing_keys
    )
    err_without_api = _max_abs_err(mismatched, physical_mean)

    assert err_with_api < 1e-2, (
        f"API path should recover physical trailing mean, got {err_with_api:.3e}"
    )
    assert err_without_api > 1.0, (
        f"expected large mismatch without API, got {err_without_api:.3e}"
    )
    assert err_without_api > 100 * err_with_api

    # Per-slot report for z1000 / ws10.
    for p, period in enumerate(INPUT_TIMES):
        for v, instant_key in enumerate(INSTANT_VARS):
            tv = p * len(INSTANT_VARS) + v
            e_ok = (recovered[:, :, tv] - physical_mean[:, :, tv]).abs().max().item()
            e_bad = (mismatched[:, :, tv] - physical_mean[:, :, tv]).abs().max().item()
            print(
                f"{instant_key}@{period}: with_api={e_ok:.4e} without_api={e_bad:.4e}"
            )
            assert e_ok < 1e-2
            assert e_bad > e_ok * 100


def test_rescaling_with_tiled_48h_outgoing():
    """Tiled -48H outgoing (len=n_var) also recovers physical mean."""
    generator = torch.Generator().manual_seed(_SEED + 31)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm_instant = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)

    coupler = _make_trailing_coupler(
        incoming=_yaml_scaling(),  # length 2 → tiled
    )
    slices = coupler.averaging_slices
    coupler.set_coupled_fields(znorm_instant)
    recovered = _denorm_timevar_with(
        coupler.construct_integrated_couplings(),
        TRAILING_48H_VARS * len(INPUT_TIMES),
    )
    physical_mean = _reference_trailing_average(
        physical, slices, dtype=torch.float64
    )
    assert _max_abs_err(recovered, physical_mean) < 1e-2


# ---------------------------------------------------------------------------
# 4. Tiled outgoing vs explicit timevar outgoing
# ---------------------------------------------------------------------------


def test_tiled_vs_explicit_timevar_outgoing_identical():
    generator = torch.Generator().manual_seed(_SEED + 40)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)
    incoming = _yaml_scaling()

    tiled = _make_trailing_coupler(
        incoming=incoming,  # len 2
    )
    explicit = _make_trailing_coupler(
        incoming=incoming,  # len 4 explicit repeat
    )
    tiled.set_coupled_fields(znorm.clone())
    explicit.set_coupled_fields(znorm.clone())
    out_tiled = tiled.construct_integrated_couplings()
    out_explicit = explicit.construct_integrated_couplings()

    assert out_tiled.shape == out_explicit.shape
    assert torch.allclose(out_tiled, out_explicit, rtol=0, atol=0), (
        f"tiled vs explicit diverge: max_err={_max_abs_err(out_tiled, out_explicit):.3e}"
    )


# ---------------------------------------------------------------------------
# 5. ConstantCoupler with scaling
# ---------------------------------------------------------------------------


def test_constant_coupler_affine_transform_across_integration():
    """Denorm(incoming) → broadcast t0 → renorm(outgoing) applied correctly."""
    variables = INSTANT_VARS
    incoming = _yaml_scaling()
    # Deliberately different outgoing stats so the affine map is non-identity.
    outgoing = incoming

    generator = torch.Generator().manual_seed(_SEED + 50)
    means_i, stds_i = _stats(variables)
    noise = torch.randn(
        _BATCH, _FACE, 4, len(variables), _HEIGHT, _WIDTH,
        dtype=torch.float64, generator=generator,
    )
    physical = means_i.view(1, 1, 1, -1, 1, 1) + stds_i.view(1, 1, 1, -1, 1, 1) * noise
    znorm = ((physical - means_i.view(1, 1, 1, -1, 1, 1)) / stds_i.view(1, 1, 1, -1, 1, 1)).to(
        torch.float32
    )

    coupler = _make_constant_coupler(
        incoming=incoming,
    )
    assert coupler.use_coupled_field_rescaling
    coupler.set_coupled_fields(znorm)
    out = coupler.construct_integrated_couplings()  # [I, B, C, F, H, W]

    means_o, stds_o = _stats(TRAILING_48H_VARS)
    # Expected: renorm_o(phys_t0) = (phys_t0 - μ_o) / σ_o
    phys_t0 = physical[:, :, :1].to(torch.float64)
    expected_bf = (phys_t0 - means_o.view(1, 1, 1, -1, 1, 1)) / stds_o.view(
        1, 1, 1, -1, 1, 1
    )
    expected = expected_bf.permute(2, 0, 3, 1, 4, 5).expand(
        coupler.coupled_integration_dim, -1, -1, -1, -1, -1
    ).to(torch.float32)

    assert out.shape[0] == coupler.coupled_integration_dim
    assert _max_abs_err(out, expected) < 1e-4
    # All integration steps identical (broadcast of time 0).
    assert torch.allclose(out[0], out[1], rtol=0, atol=0)
    assert torch.allclose(out[0], out[-1], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 6. Stability analysis vs float64 physical reference
# ---------------------------------------------------------------------------


def test_stability_denorm_mean_renorm_vs_float64_reference():
    generator = torch.Generator().manual_seed(_SEED + 60)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)
    outgoing_keys = TRAILING_48H_VARS * len(INPUT_TIMES)

    coupler = _make_trailing_coupler(
        incoming=_yaml_scaling(),
    )
    slices = coupler.averaging_slices
    coupler.set_coupled_fields(znorm)
    got = coupler.construct_integrated_couplings().to(torch.float64)

    phys_avg_f64 = _reference_trailing_average(
        physical, slices, dtype=torch.float64
    )
    ref = _renorm_timevar_physical(phys_avg_f64, outgoing_keys, dtype=torch.float64)

    err = _max_abs_err(got, ref)
    assert err < 1e-4, f"float32 denorm→mean→renorm drifted from f64 ref: {err:.3e}"

    # Per-slot quantitative errors (z-space).
    slot_errs = {}
    for p, period in enumerate(INPUT_TIMES):
        for v, instant_key in enumerate(INSTANT_VARS):
            tv = p * len(INSTANT_VARS) + v
            e = (got[:, :, tv] - ref[:, :, tv]).abs().max().item()
            slot_errs[f"{instant_key}@{period}"] = e
            print(f"stability {instant_key}@{period}: max_abs_z={e:.4e}")
            assert e < 1e-4


def test_optional_figure_mismatch_vs_api(tmp_path=None):
    """Write a small comparison chart if matplotlib is available; never required."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generator = torch.Generator().manual_seed(42)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)
    outgoing_keys = TRAILING_48H_VARS * len(INPUT_TIMES)

    bare = _make_trailing_coupler(
        variables=INSTANT_VARS, incoming_variables=INSTANT_VARS
    )
    slices = bare.averaging_slices
    physical_mean = _reference_trailing_average(
        physical, slices, dtype=torch.float64
    )

    bare.set_coupled_fields(znorm.clone())
    mismatched = _denorm_timevar_with(
        bare.construct_integrated_couplings(), outgoing_keys
    )

    scaled = _make_trailing_coupler(
        incoming=_yaml_scaling(),
    )
    scaled.set_coupled_fields(znorm.clone())
    recovered = _denorm_timevar_with(
        scaled.construct_integrated_couplings(), outgoing_keys
    )

    labels, err_bad, err_ok = [], [], []
    for p, period in enumerate(INPUT_TIMES):
        for v, instant_key in enumerate(INSTANT_VARS):
            tv = p * len(INSTANT_VARS) + v
            labels.append(f"{instant_key}\n@{period}")
            err_bad.append(
                (mismatched[:, :, tv] - physical_mean[:, :, tv]).abs().max().item()
            )
            err_ok.append(
                (recovered[:, :, tv] - physical_mean[:, :, tv]).abs().max().item()
            )

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, err_bad, width, label="without API (mismatch)", color="#c44e52")
    ax.bar(x + width / 2, err_ok, width, label="with API (recover)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("max |error| vs physical trailing mean")
    ax.set_yscale("log")
    ax.set_title("Coupled-field scaling: instant→trailing mean recovery")
    ax.legend()
    fig.tight_layout()

    out_dir = Path(__file__).with_name("figures")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "coupler_coupled_field_scaling_api.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    assert out_path.is_file()
    # Soft numerical check so this still contributes coverage if run.
    # z1000 mean-shift is O(10–100+); ws10 can be smaller but still ≫ API path.
    z_bad = [err_bad[i] for i, lab in enumerate(labels) if "z1000" in lab]
    assert all(b > 1.0 for b in z_bad)
    assert all(o < 1e-2 for o in err_ok)
    assert all(b > 100 * o for b, o in zip(err_bad, err_ok))


# ---------------------------------------------------------------------------
# 7. Adversarial edge cases
# ---------------------------------------------------------------------------


def test_channel_order_period_major_multi_variable():
    """Output channels stay period-major: [p0_v0, p0_v1, p1_v0, p1_v1]."""
    variables = INSTANT_VARS
    # Plant constant-per-channel physical fields so order is unambiguous.
    means, stds = _stats(variables)
    k = torch.tensor([2.0, -1.5], dtype=torch.float64)
    physical = (
        means.view(1, 1, 1, -1, 1, 1)
        + stds.view(1, 1, 1, -1, 1, 1) * k.view(1, 1, 1, -1, 1, 1)
    ).expand(_BATCH, _FACE, TIMEDIM, -1, _HEIGHT, _WIDTH).contiguous()
    znorm = _normalize_with(physical, variables, dtype=torch.float32)

    outgoing_keys = TRAILING_48H_VARS * len(INPUT_TIMES)
    coupler = _make_trailing_coupler(
        incoming=_yaml_scaling(),
    )
    coupler.set_coupled_fields(znorm)
    out = coupler.construct_integrated_couplings().to(torch.float64)

    means_o, stds_o = _stats(outgoing_keys)
    # Physical average of constant field is the constant; renorm per slot.
    for p in range(len(INPUT_TIMES)):
        for v in range(len(variables)):
            tv = p * len(variables) + v
            phys_val = means[v] + stds[v] * k[v]
            expected_z = (phys_val - means_o[tv]) / stds_o[tv]
            got_mean = out[:, :, tv].mean().item()
            assert abs(got_mean - expected_z.item()) < 1e-3, (
                f"slot {tv} ({outgoing_keys[tv]}) got {got_mean:.4f}, "
                f"expected {expected_z.item():.4f} — channel order bug?"
            )


def test_set_coupled_scaling():
    generator = torch.Generator().manual_seed(_SEED + 70)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)
    outgoing_keys = TRAILING_48H_VARS * len(INPUT_TIMES)

    coupler = _make_trailing_coupler(
        incoming=_yaml_scaling(),
    )
    assert coupler.use_coupled_field_rescaling
    slices = coupler.averaging_slices
    coupler.set_coupled_fields(znorm)
    recovered = _denorm_timevar_with(
        coupler.construct_integrated_couplings(), outgoing_keys
    )
    physical_mean = _reference_trailing_average(
        physical, slices, dtype=torch.float64
    )
    assert _max_abs_err(recovered, physical_mean) < 1e-2


def test_float32_cpu_dtype_and_device():
    generator = torch.Generator().manual_seed(_SEED + 71)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)
    assert znorm.dtype == torch.float32
    assert znorm.device.type == "cpu"

    outgoing_keys = TRAILING_48H_VARS * len(INPUT_TIMES)
    coupler = _make_trailing_coupler(
        incoming=_yaml_scaling(),
    )
    # Exercise denormalize / renormalize helpers directly.
    denormed = coupler.denormalize_coupled_fields(znorm)
    assert denormed.dtype == torch.float32
    assert denormed.device.type == "cpu"

    coupler.set_coupled_fields(znorm)
    out = coupler.construct_integrated_couplings()
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    assert out.shape[2] == len(INSTANT_VARS) * len(INPUT_TIMES)


def test_rescale_promotes_float16_to_float32_then_restores():
    """Physical denorm of z1000 exceeds float16 range; path must use float32 mid-flight."""
    generator = torch.Generator().manual_seed(_SEED + 72)
    physical = _sample_instant_physical(TIMEDIM, generator)
    znorm_f32 = _normalize_with(physical, INSTANT_VARS, dtype=torch.float32)
    znorm_f16 = znorm_f32.to(torch.float16)

    coupler_f16 = _make_trailing_coupler(incoming=_yaml_scaling())
    coupler_f16.set_coupled_fields(znorm_f16)
    out_f16 = coupler_f16.construct_integrated_couplings()
    assert out_f16.dtype == torch.float16
    assert torch.isfinite(out_f16.float()).all()

    coupler_f32 = _make_trailing_coupler(incoming=_yaml_scaling())
    coupler_f32.set_coupled_fields(znorm_f32)
    out_f32 = coupler_f32.construct_integrated_couplings()

    # Restored float16 should track the float32 path within float16 eps.
    assert _max_abs_err(out_f16.float(), out_f32) < 5e-2


def test_rescale_coupled_fields_through_physical_callable():
    coupler = _make_trailing_coupler(
        incoming=_yaml_scaling(),
    )
    # Minimal [B,F,T,C,H,W] with known z values.
    fields = torch.zeros(
        _BATCH, _FACE, 2, len(INSTANT_VARS), _HEIGHT, _WIDTH, dtype=torch.float32
    )
    fields[..., 0, :, :] = 1.0  # +1σ on z1000
    fields[..., 1, :, :] = -2.0  # -2σ on ws10m

    def identity_op(x):
        return x  # keep C=n_var; outgoing was tiled to timevar, so need C match

    # Outgoing was tiled to timevar_dim=4, so identity (C=2) should raise.
    with pytest.raises(ValueError, match="channel count"):
        coupler.rescale_coupled_fields_through_physical(fields, identity_op)

    # Physical op that expands channels to timevar layout by repeating.
    def expand_to_timevar(x):
        return torch.cat([x, x], dim=3)

    out = coupler.rescale_coupled_fields_through_physical(fields, expand_to_timevar)
    assert out.shape[3] == coupler.timevar_dim
    assert out.dtype == torch.float32


def test_denormalize_without_scaling_raises():
    coupler = _make_trailing_coupler(
        variables=INSTANT_VARS, incoming_variables=INSTANT_VARS
    )
    fields = torch.zeros(
        _BATCH, _FACE, 2, len(INSTANT_VARS), _HEIGHT, _WIDTH, dtype=torch.float32
    )
    with pytest.raises(RuntimeError, match="incoming_coupled_scaling"):
        coupler.denormalize_coupled_fields(fields)
    with pytest.raises(RuntimeError, match="outgoing_coupled_scaling"):
        coupler.renormalize_coupled_fields(fields)
