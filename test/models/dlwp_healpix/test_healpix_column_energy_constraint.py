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
import time

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import numpy as np
import pytest
import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_column_energy_constraint import (
    CV,
    GRAVITY,
    LATENT_HEAT_OF_FREEZING,
    LATENT_HEAT_OF_VAPORIZATION,
    RDGAS,
    RVGAS,
    SOLAR_CONSTANT,
    ColumnEnergyConstraint,
    accepts_forcing,
    forcing_from_folded_input,
)

N = 8
TEMP = [f"air_temperature_{k}" for k in range(N)]
WATER = [f"specific_total_water_{k}" for k in range(N)]
FLUX = [
    "LHTFLsfc",
    "SHTFLsfc",
    "DLWRFsfc",
    "ULWRFsfc",
    "DSWRFsfc",
    "USWRFsfc",
    "ULWRFtoa",
    "USWRFtoa",
    "total_frozen_precipitation_rate",
]
IN_CH = ["PRESsfc", *TEMP, *WATER]
OUT_CH = [*IN_CH, *FLUX]
CONST = ["land_fraction", "HGTsfc", "sin_lat"]
SCALING = {n: {"mean": 0.0, "std": 1.0} for n in [*OUT_CH, *CONST]}
DT = 21600.0


def _make(**kwargs):
    defaults = dict(
        in_channels=IN_CH,
        out_channels=OUT_CH,
        scaling=SCALING,
        constant_channels=CONST,
        timestep_seconds=DT,
    )
    defaults.update(kwargs)
    return ColumnEnergyConstraint(**defaults)


class _TwoArg(torch.nn.Module):
    def forward(self, prediction, input):
        return prediction


def test_forcing_slice_and_signature():
    folded = torch.arange(2 * 12 * 8 * 2 * 2, dtype=torch.float32).reshape(24, 8, 2, 2)
    aux = forcing_from_folded_input(
        folded,
        input_channels=3,
        input_time_dim=1,
        decoder_input_channels=1,
        n_constants=2,
        num_faces=12,
    )
    assert aux["insolation"].shape == (2, 12, 1, 1, 2, 2)
    assert aux["constants"].shape == (2, 12, 1, 2, 2, 2)
    assert torch.equal(aux["insolation"][:, :, 0, 0], folded[:, 3].reshape(2, 12, 2, 2))
    assert torch.equal(aux["constants"][:, :, 0, 1], folded[:, 5].reshape(2, 12, 2, 2))
    assert accepts_forcing(_make())
    assert not accepts_forcing(_TwoArg())


def _energy_and_factor(ps, temperature, q, height):
    """ACE formulas with vertical on the last axis. ``ps`` is [..., 1]."""
    ak = np.asarray(
        [
            1.0001825094223022,
            5119.89501953125,
            13881.3310546875,
            19343.51171875,
            20087.0859375,
            15596.6953125,
            8880.453125,
            3057.265625,
            0.0,
        ]
    )
    bk = np.asarray(
        [
            0.0,
            0.0,
            0.005377814639359713,
            0.059728413820266724,
            0.2034912109375,
            0.43839120864868164,
            0.6806430220603943,
            0.8739292621612549,
            1.0,
        ]
    )
    # Move vertical to the end: temperature is [B, F, T, L, H, W].
    t = np.moveaxis(temperature, 3, -1)
    qq = np.moveaxis(q, 3, -1)
    ps_s = np.squeeze(ps, axis=3)
    p_if = ak.reshape((1,) * ps_s.ndim + (-1,)) + bk.reshape((1,) * ps_s.ndim + (-1,)) * ps_s[..., None]
    tv = t * (1.0 + (RVGAS / RDGAS - 1.0) * qq)
    dlogp = np.diff(np.log(np.clip(p_if, 1.0, None)), axis=-1)
    dz = dlogp * RDGAS * tv / GRAVITY
    hs = np.squeeze(np.where(height < 0, 0.0, height), axis=3)
    cumulative = np.cumsum(dz[..., ::-1], axis=-1)[..., ::-1]
    height_if = np.concatenate([cumulative + hs[..., None], hs[..., None]], axis=-1)
    z = 0.5 * (height_if[..., :-1] + height_if[..., 1:])
    energy = CV * t + LATENT_HEAT_OF_VAPORIZATION * qq + GRAVITY * z
    dp = np.diff(p_if, axis=-1)
    column = (energy * dp).sum(axis=-1) / GRAVITY
    q_times_dlogp = dz * GRAVITY / t
    cum = np.cumsum(q_times_dlogp[..., ::-1], axis=-1)[..., ::-1]
    factor = ((CV - 0.5 * q_times_dlogp + cum) * dp).sum(axis=-1) / GRAVITY
    return column, factor


def test_uniform_offset_closes_energy_budget():
    rng = np.random.default_rng(0)
    spatial = (2, 12, 1, 1, 4, 4)
    levels = (2, 12, 1, N, 4, 4)
    ps0 = 98000.0 + 200.0 * rng.standard_normal(spatial)
    ps = ps0 + 50.0 * rng.standard_normal(spatial)
    t0 = 260.0 + np.linspace(0, 30, N).reshape(1, 1, 1, N, 1, 1)
    t0 = t0 + rng.standard_normal(levels)
    t = t0 + 0.4 * rng.standard_normal(levels)
    q0 = np.clip(0.003 + 0.0005 * rng.standard_normal(levels), 1e-4, 0.02)
    q = np.clip(q0 + 0.0001 * rng.standard_normal(levels), 1e-4, 0.02)
    height = 200.0 + 50.0 * rng.standard_normal(spatial)
    insol = 0.4 + 0.2 * rng.random(spatial)
    fluxes = {
        "LHTFLsfc": 80 + 5 * rng.standard_normal(spatial),
        "SHTFLsfc": 20 + 2 * rng.standard_normal(spatial),
        "DLWRFsfc": 300 + 5 * rng.standard_normal(spatial),
        "ULWRFsfc": 350 + 5 * rng.standard_normal(spatial),
        "DSWRFsfc": 200 + 5 * rng.standard_normal(spatial),
        "USWRFsfc": 40 + 2 * rng.standard_normal(spatial),
        "ULWRFtoa": 220 + 5 * rng.standard_normal(spatial),
        "USWRFtoa": 80 + 2 * rng.standard_normal(spatial),
        "total_frozen_precipitation_rate": np.clip(1e-5 * rng.random(spatial), 0, None),
    }
    pred = torch.zeros(2, 12, 1, len(OUT_CH), 4, 4)
    orig = torch.zeros(2, 12, 1, len(IN_CH), 4, 4)
    pred[:, :, :, 0:1] = torch.from_numpy(ps).float()
    orig[:, :, :, 0:1] = torch.from_numpy(ps0).float()
    pred[:, :, :, 1 : 1 + N] = torch.from_numpy(t).float()
    orig[:, :, :, 1 : 1 + N] = torch.from_numpy(t0).float()
    pred[:, :, :, 1 + N : 1 + 2 * N] = torch.from_numpy(q).float()
    orig[:, :, :, 1 + N : 1 + 2 * N] = torch.from_numpy(q0).float()
    for i, name in enumerate(FLUX):
        pred[:, :, :, 1 + 2 * N + i] = torch.from_numpy(fluxes[name]).float().squeeze(3)
    constants = torch.zeros(2, 12, 1, 3, 4, 4)
    constants[:, :, :, 1:2] = torch.from_numpy(height).float()
    aux = {
        "insolation": torch.from_numpy(insol).float(),
        "constants": constants,
    }
    mod = _make()
    with torch.no_grad():
        out = mod(pred, orig, aux=aux)
    t_new = out[:, :, :, 1 : 1 + N].numpy()
    delta = t_new - t
    assert np.max(np.abs(delta - delta[:, :1])) < 1e-4
    assert np.max(delta.std(axis=(1, 4, 5))) < 1e-4
    column0, _ = _energy_and_factor(ps0, t0, q0, height)
    column, factor = _energy_and_factor(ps, t, q, height)
    column_new, _ = _energy_and_factor(ps, t_new, q, height)
    lhf, shf, dlw, ulw, dsw, usw, ulw_toa, usw_toa, frozen = (
        np.squeeze(fluxes[n], axis=3) for n in FLUX
    )
    into = (np.squeeze(insol, axis=3) * SOLAR_CONSTANT - usw_toa - ulw_toa) - (
        (dsw - usw + dlw - ulw) - lhf - shf - frozen * LATENT_HEAT_OF_FREEZING
    )
    gm = lambda x: x.mean(axis=(1, 3, 4))
    desired = gm(column0) + gm(into) * DT
    expected_offset = (desired - gm(column)) / gm(factor)
    assert np.max(np.abs(delta - expected_offset[:, None, :, None, None, None])) < 1e-3
    residual = np.abs(gm(column_new) - desired)
    assert np.max(residual) < 1.0e-6 * np.max(np.abs(gm(column_new)))


def test_requires_aux():
    pred = torch.zeros(1, 12, 1, len(OUT_CH), 2, 2)
    orig = torch.zeros(1, 12, 1, len(IN_CH), 2, 2)
    pred[:, :, :, 1 : 1 + N] = 280.0
    orig[:, :, :, 1 : 1 + N] = 280.0
    pred[:, :, :, 0] = 1.0e5
    orig[:, :, :, 0] = 1.0e5
    with pytest.raises(ValueError, match="aux"):
        _make()(pred, orig)


def test_nside64_is_fast():
    h = w = 64
    pred = torch.zeros(1, 12, 1, len(OUT_CH), h, w)
    orig = torch.zeros(1, 12, 1, len(IN_CH), h, w)
    pred[:, :, :, 0] = 98500.0
    orig[:, :, :, 0] = 98500.0
    pred[:, :, :, 1 : 1 + N] = 270.0
    orig[:, :, :, 1 : 1 + N] = 270.0
    pred[:, :, :, 1 + N : 1 + 2 * N] = 0.002
    orig[:, :, :, 1 + N : 1 + 2 * N] = 0.002
    pred[:, :, :, 1 + 2 * N :] = 10.0
    aux = {
        "insolation": torch.full((1, 12, 1, 1, h, w), 0.5),
        "constants": torch.zeros(1, 12, 1, 3, h, w),
    }
    aux["constants"][:, :, :, 1] = 100.0
    mod = _make()
    with torch.no_grad():
        started = time.perf_counter()
        mod(pred, orig, aux=aux)
        elapsed = time.perf_counter() - started
    assert elapsed < 1.0, elapsed


def test_frozen_precipitation_optional():
    """Without a frozen channel the budget matches a zero frozen channel."""
    rng = torch.Generator().manual_seed(0)
    pred = torch.zeros(1, 12, 1, len(OUT_CH), 4, 4)
    orig = torch.zeros(1, 12, 1, len(IN_CH), 4, 4)
    pred[:, :, :, 0] = orig[:, :, :, 0] = 98500.0
    pred[:, :, :, 1 : 1 + N] = 270.0 + torch.rand(1, 12, 1, N, 4, 4, generator=rng)
    orig[:, :, :, 1 : 1 + N] = 270.0
    pred[:, :, :, 1 + N : 1 + 2 * N] = orig[:, :, :, 1 + N : 1 + 2 * N] = 0.002
    pred[:, :, :, 1 + 2 * N :] = 100.0 * torch.rand(1, 12, 1, len(FLUX), 4, 4, generator=rng)
    pred[:, :, :, -1] = 0.0
    aux = {
        "insolation": torch.full((1, 12, 1, 1, 4, 4), 0.5),
        "constants": torch.zeros(1, 12, 1, 3, 4, 4),
    }
    no_frozen = _make(
        out_channels=OUT_CH[:-1],
        frozen_precipitation=None,
        scaling={k: v for k, v in SCALING.items() if k != FLUX[-1]},
    )
    with torch.no_grad():
        out = _make()(pred, orig, aux=aux)
        out_nf = no_frozen(pred[:, :, :, :-1], orig, aux=aux)
    assert torch.allclose(out[:, :, :, :-1], out_nf)
    assert not torch.allclose(out_nf[:, :, :, 1 : 1 + N], pred[:, :, :, 1 : 1 + N])



def _coupled_top_case():
    """Layer 0 prescribed: (module, pred, orig, aux, physical arrays)."""
    rng = np.random.default_rng(5)
    spatial = (2, 12, 1, 1, 4, 4)
    levels = (2, 12, 1, N, 4, 4)
    ps0 = 98000.0 + 200.0 * rng.standard_normal(spatial)
    ps = ps0 + 50.0 * rng.standard_normal(spatial)
    t0 = 260.0 + np.linspace(0, 30, N).reshape(1, 1, 1, N, 1, 1) + rng.standard_normal(levels)
    t = t0 + 0.4 * rng.standard_normal(levels)
    q0 = np.clip(0.003 + 0.0005 * rng.standard_normal(levels), 1e-4, 0.02)
    q = np.clip(q0 + 0.0001 * rng.standard_normal(levels), 1e-4, 0.02)
    t[:, :, :, 0], q[:, :, :, 0] = t0[:, :, :, 0], q0[:, :, :, 0]
    height = 200.0 + 50.0 * rng.standard_normal(spatial)
    insol = 0.4 + 0.2 * rng.random(spatial)
    flux = np.concatenate(
        [m + s * rng.standard_normal(spatial) for m, s in
         ((80, 5), (20, 2), (300, 5), (350, 5), (200, 5), (40, 2), (220, 5), (80, 2), (0, 0))],
        axis=3,
    )
    top = [TEMP[0], WATER[0]]
    model_t = list(range(1, N))
    pred = torch.from_numpy(np.concatenate([ps, t[:, :, :, model_t], q[:, :, :, model_t], flux], axis=3)).float()
    orig = torch.from_numpy(np.concatenate([ps0, t0[:, :, :, model_t], q0[:, :, :, model_t]], axis=3)).float()
    constants = torch.zeros(2, 12, 1, 3, 4, 4)
    constants[:, :, :, 1:2] = torch.from_numpy(height).float()
    coupled = np.concatenate([np.zeros(spatial), t0[:, :, :, :1], q0[:, :, :, :1]], axis=3)
    aux = {
        "insolation": torch.from_numpy(insol).float(),
        "constants": constants,
        "coupled": torch.from_numpy(coupled).float(),
    }
    mod = _make(
        in_channels=[n for n in IN_CH if n not in top],
        out_channels=[n for n in OUT_CH if n not in top],
        coupled_channels=["sea_surface_temperature", *top],
    )
    arrays = dict(ps0=ps0, ps=ps, t0=t0, t=t, q0=q0, q=q, height=height, insol=insol, flux=flux, model_t=model_t)
    return mod, pred, orig, aux, arrays


def test_coupled_top_level_closes_full_column_budget():
    # Layer 0 comes from the coupler: it is not written, and the predicted
    # layers' offset closes the budget of the full column, layer 0 included.
    mod, pred, orig, aux, a = _coupled_top_case()
    with pytest.raises(ValueError, match="coupled"):
        mod(pred, orig, aux={k: v for k, v in aux.items() if k != "coupled"})
    with torch.no_grad():
        out = mod(pred, orig, aux=aux).numpy()
    t, model_t = a["t"], a["model_t"]
    t_new = t.copy()
    t_new[:, :, :, model_t] = out[:, :, :, 1:N]
    delta = t_new[:, :, :, model_t] - t[:, :, :, model_t]
    assert np.max(np.abs(delta - delta[:, :1, :, :1])) < 1e-4
    np.testing.assert_array_equal(out[:, :, :, N:], pred[:, :, :, N:].numpy())
    column0, _ = _energy_and_factor(a["ps0"], a["t0"], a["q0"], a["height"])
    column_new, _ = _energy_and_factor(a["ps"], t_new, a["q"], a["height"])
    lhf, shf, dlw, ulw, dsw, usw, ulw_toa, usw_toa, frozen = np.moveaxis(a["flux"], 3, 0)
    into = (np.squeeze(a["insol"], axis=3) * SOLAR_CONSTANT - usw_toa - ulw_toa) - (
        (dsw - usw + dlw - ulw) - lhf - shf - frozen * LATENT_HEAT_OF_FREEZING
    )
    gm = lambda x: x.mean(axis=(1, 3, 4))
    desired = gm(column0) + gm(into) * DT
    assert np.max(np.abs(gm(column_new) - desired)) < 1.0e-6 * np.max(np.abs(gm(column_new)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_coupled_top_level_captures_in_cuda_graph():
    # Training and rollouts capture the model, constraints included, in CUDA graphs.
    mod, pred, orig, aux, _ = _coupled_top_case()
    dev = torch.device("cuda")
    mod = mod.to(dev)
    pred, orig = pred.to(dev), orig.to(dev)
    aux = {k: v.to(dev) for k, v in aux.items()}
    with torch.no_grad():
        eager = mod(pred, orig, aux=aux)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mod(pred, orig, aux=aux)  # warm-up on a side stream, as before a capture
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mod(pred, orig, aux=aux)
        graph.replay()
    torch.testing.assert_close(captured, eager)
