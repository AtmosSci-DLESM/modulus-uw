# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Three-way equivalence harness for HEALPix coupling math.

The same coupling operation is implemented three times in the DLESyM stack:

1. ``ConstantCoupler`` / ``TrailingAverageCoupler`` in this library, driven by
   ``set_coupled_fields``.
2. A differentiable transcription in the distributed inferencer
   (``diff_atmos_coupling`` / ``diff_ocean_coupling``).
3. A second transcription in the coupled training model
   (``prepare_couplings`` / ``update_coupling``).

The inference and training versions are transcribed here verbatim so a single
synthetic ``[B, F, T, C, H, W]`` tensor can be driven through all three and
compared without importing the downstream repositories. Tests are parameterized
over the number of coupled channels and over window uniformity because several
divergences are latent: they cancel for a single coupled variable or for
uniformly spaced averaging windows and only appear otherwise.

Divergences that are known and not yet reconciled are marked ``xfail(strict)``,
so they flip to failures the moment the implementations converge.
"""

import numpy as np
import pytest
import torch
from conftest import requires_module

from physicsnemo.datapipes.healpix.couplers import (
    ConstantCoupler,
    TrailingAverageCoupler,
)
from physicsnemo.datapipes.healpix.coupling_ops import (
    CouplingOp,
    concat_integrated_couplings,
)

_BATCH, _FACE, _HEIGHT, _WIDTH = 2, 4, 3, 3
_SEED = 11


# ---------------------------------------------------------------------------
# Transcriptions of the downstream implementations
# ---------------------------------------------------------------------------


def inference_constant_coupling(state, channel_indices, integration_dim):
    """Transcribed from ``Inferencer.diff_atmos_coupling``.

    Note the channel slice is applied *after* the permute, so ``-1:`` selects
    the last coupled channel rather than all of them.
    """
    state = state[:, :, :, channel_indices, :, :].permute(2, 0, 3, 1, 4, 5)
    return torch.cat(
        [state[:1, :, -1:, :, :, :] for _ in range(integration_dim)], dim=0
    )


def inference_trailing_average_coupling(
    output, channel_indices, integration_dim, averaging_slices
):
    """Transcribed from ``Inferencer.diff_ocean_coupling``."""
    output = output[:, :, :, channel_indices, :, :]
    coupled_averaging_periods = []
    for j in range(integration_dim):
        averaging_periods = [
            output[:, :, s, :, :, :].mean(dim=2, keepdim=True)
            for s in averaging_slices[j]
        ]
        coupled_averaging_periods.append(torch.concat(averaging_periods, dim=3))
    return torch.concat(coupled_averaging_periods, dim=2).permute(2, 0, 3, 1, 4, 5)


def training_constant_coupling(atmos_step_data, var_idx, time_idx):
    """Transcribed from ``CoupledModel.prepare_couplings``, constant branch.

    ``var_idx`` is applied to dimension 0 (batch) rather than the channel
    dimension, which is how the source reads today.
    """
    coupling_data = atmos_step_data[:, :, time_idx][var_idx, :, :]
    return coupling_data.permute(2, 0, 3, 1, 4, 5)


def training_trailing_average_coupling(
    atmos_step_data, var_idx, n_output_times, n_integrations, n_time_idx
):
    """Transcribed from ``CoupledModel.prepare_couplings``, trailing branch.

    Partitions the time axis into ``n_output_times`` equal contiguous chunks
    rather than using the library's ``averaging_slices``.
    """
    coupling_data = atmos_step_data[:, :, :, var_idx, :, :]
    chunks = coupling_data.chunk(n_output_times, dim=2)
    if len(chunks) != n_integrations * n_time_idx:
        raise ValueError("Coupling does not evenly align with expected time dimensions")
    coupling_data_avg = [
        torch.concat(
            [
                x.mean(dim=2, keepdim=True)
                for x in chunks[j * n_time_idx : (j + 1) * n_time_idx]
            ],
            dim=3,
        )
        for j in range(n_integrations)
    ]
    return torch.cat(coupling_data_avg, dim=2).permute(2, 0, 3, 1, 4, 5)


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


class _FakeSource:
    """Stands in for the source component's TimeSeriesDataset."""

    def __init__(self, output_variables, time_step):
        self.output_variables = list(output_variables)
        self.time_step = time_step


def _make_dataset(variables, n_time, time_step="3h"):
    xr = pytest.importorskip("xarray")
    pd = pytest.importorskip("pandas")

    return xr.Dataset(
        data_vars={
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                np.zeros(
                    (n_time, len(variables), _FACE, _HEIGHT, _WIDTH), dtype="float32"
                ),
            )
        },
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq=time_step),
            "channel_in": list(variables),
            "face": np.arange(_FACE),
            "height": np.arange(_HEIGHT),
            "width": np.arange(_WIDTH),
        },
    )


def _source_fields(n_time, n_source_channels, seed=_SEED):
    """Synthetic source-component output in ``[B, F, T, C, H, W]`` layout."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        _BATCH,
        _FACE,
        n_time,
        n_source_channels,
        _HEIGHT,
        _WIDTH,
        dtype=torch.float32,
        generator=generator,
    )


def _coupled_and_source_names(n_coupled, suffix="24H", n_extra=1):
    """Coupled variable names plus a source channel list that brackets them.

    The source list is padded so the coupled channels never start at index 0,
    which keeps an off-by-zero in index derivation from passing silently.
    """
    base = [f"v{i}" for i in range(n_coupled)]
    coupled = [f"{b}-{suffix}" for b in base]
    source = [f"pad{j}" for j in range(n_extra)] + base + ["tail"]
    return coupled, source


def _make_trailing_coupler(
    n_coupled, input_times, input_time_dim, output_time_dim, time_step="3h"
):
    coupled, source = _coupled_and_source_names(n_coupled)
    coupler = TrailingAverageCoupler(
        dataset=_make_dataset(coupled, n_time=64, time_step=time_step),
        batch_size=_BATCH,
        variables=coupled,
        presteps=0,
        averaging_window="24h",
        input_times=list(input_times),
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    )
    coupler.setup_coupling(_FakeSource(source, time_step))
    return coupler, source


def _make_constant_coupler(n_coupled, input_time_dim, output_time_dim, time_step="3h"):
    coupled, source = _coupled_and_source_names(n_coupled, suffix="0H")
    coupler = ConstantCoupler(
        dataset=_make_dataset(coupled, n_time=16, time_step=time_step),
        batch_size=_BATCH,
        variables=coupled,
        input_times=["0h"],
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        presteps=0,
    )
    coupler.setup_coupling(_FakeSource(source, time_step))
    return coupler, source


# The aligned trailing-average configuration: uniformly spaced windows whose
# right edges tile the time axis exactly, which is the only regime in which the
# library's averaging_slices and the training chunk() partition agree.
#
#   time_step 3h, input_times 24h/48h -> di = 8, window indices [8, 16]
#   input_time_dim 2, output_time_dim 4 -> coupled_integration_dim 2
#   slices: j=0 -> (0,8), (8,16);  j=1 -> (16,24), (24,32)
UNIFORM = dict(
    input_times=["24h", "48h"],
    input_time_dim=2,
    output_time_dim=4,
    n_time=32,
    n_output_times=4,
)

# Non-uniform windows: right edges at 24h and 72h give windows of unequal
# width, so equal chunking cannot reproduce them.
#   window indices [8, 24]
#   slices: j=0 -> (0,8), (8,24);  j=1 -> (16,24), (24,40)
NON_UNIFORM = dict(
    input_times=["24h", "72h"],
    input_time_dim=2,
    output_time_dim=4,
    n_time=40,
    n_output_times=4,
)


# ---------------------------------------------------------------------------
# Trailing average: library vs inference
# ---------------------------------------------------------------------------


@requires_module(["xarray", "pandas"])
@pytest.mark.parametrize("n_coupled", [1, 3])
@pytest.mark.parametrize(
    "config", [UNIFORM, NON_UNIFORM], ids=["uniform", "nonuniform"]
)
def test_trailing_library_matches_inference(n_coupled, config):
    """The inference transcription reproduces the library exactly.

    Both consume the coupler's own ``averaging_slices``, so they agree for any
    window layout as long as no scaling is configured.
    """
    coupler, source = _make_trailing_coupler(
        n_coupled,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    fields = _source_fields(config["n_time"], len(source))

    coupler.set_coupled_fields(fields)
    library = coupler.construct_integrated_couplings()

    inference = inference_trailing_average_coupling(
        fields,
        coupler.coupled_channel_indices,
        coupler.coupled_integration_dim,
        coupler.averaging_slices,
    )

    assert library.shape == inference.shape
    assert torch.equal(library, inference)


# ---------------------------------------------------------------------------
# Trailing average: library vs training
# ---------------------------------------------------------------------------


@requires_module(["xarray", "pandas"])
@pytest.mark.parametrize("n_coupled", [1, 3])
def test_trailing_library_matches_training_uniform_windows(n_coupled):
    """With uniformly tiling windows the training chunk partition agrees."""
    config = UNIFORM
    coupler, source = _make_trailing_coupler(
        n_coupled,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    fields = _source_fields(config["n_time"], len(source))

    coupler.set_coupled_fields(fields)
    library = coupler.construct_integrated_couplings()

    training = training_trailing_average_coupling(
        fields,
        coupler.coupled_channel_indices,
        config["n_output_times"],
        coupler.coupled_integration_dim,
        len(coupler.input_times),
    )

    assert library.shape == training.shape
    assert torch.equal(library, training)


@requires_module(["xarray", "pandas"])
@pytest.mark.parametrize("n_coupled", [1, 3])
@pytest.mark.xfail(
    strict=True,
    reason="training partitions the time axis into equal chunks, the library "
    "uses averaging_slices derived from input_times; these disagree whenever "
    "the averaging windows are not uniformly spaced",
)
def test_trailing_library_matches_training_nonuniform_windows(n_coupled):
    config = NON_UNIFORM
    coupler, source = _make_trailing_coupler(
        n_coupled,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    fields = _source_fields(config["n_time"], len(source))

    coupler.set_coupled_fields(fields)
    library = coupler.construct_integrated_couplings()

    training = training_trailing_average_coupling(
        fields,
        coupler.coupled_channel_indices,
        config["n_output_times"],
        coupler.coupled_integration_dim,
        len(coupler.input_times),
    )

    assert torch.equal(library, training)


# ---------------------------------------------------------------------------
# Constant coupling: library vs inference
# ---------------------------------------------------------------------------


@requires_module(["xarray", "pandas"])
def test_constant_library_matches_inference_single_channel():
    """With one coupled channel the inference ``-1:`` slice is a no-op."""
    coupler, source = _make_constant_coupler(1, input_time_dim=1, output_time_dim=3)
    fields = _source_fields(4, len(source))

    coupler.set_coupled_fields(fields)
    library = coupler.construct_integrated_couplings()
    inference = inference_constant_coupling(
        fields, coupler.coupled_channel_indices, coupler.coupled_integration_dim
    )

    assert library.shape == inference.shape
    assert torch.equal(library, inference)


@requires_module(["xarray", "pandas"])
@pytest.mark.xfail(
    strict=True,
    reason="inference slices [-1:] on the channel axis after permuting, so it "
    "keeps only the last coupled channel; the library keeps all of them. The "
    "two agree only while exactly one variable is coupled",
)
def test_constant_library_matches_inference_multi_channel():
    coupler, source = _make_constant_coupler(3, input_time_dim=1, output_time_dim=3)
    fields = _source_fields(4, len(source))

    coupler.set_coupled_fields(fields)
    library = coupler.construct_integrated_couplings()
    inference = inference_constant_coupling(
        fields, coupler.coupled_channel_indices, coupler.coupled_integration_dim
    )

    assert torch.equal(library, inference)


# ---------------------------------------------------------------------------
# Constant coupling: library vs training
# ---------------------------------------------------------------------------


@requires_module(["xarray", "pandas"])
@pytest.mark.xfail(
    strict=True,
    reason="the training constant branch applies var_idx to dimension 0 "
    "(batch) instead of the channel dimension",
)
def test_constant_library_matches_training():
    coupler, source = _make_constant_coupler(1, input_time_dim=1, output_time_dim=3)
    fields = _source_fields(4, len(source))

    coupler.set_coupled_fields(fields)
    library = coupler.construct_integrated_couplings()

    # Constant coupling repeats one source timestep integration_dim times.
    time_idx = [0] * coupler.coupled_integration_dim
    training = training_constant_coupling(
        fields, coupler.coupled_channel_indices, time_idx
    )

    assert torch.equal(library, training)


@requires_module(["xarray", "pandas"])
def test_constant_training_indexing_lands_on_batch_axis():
    """Pin the signature of the training constant-branch indexing bug.

    ``var_idx`` is applied to dimension 0, so it selects batch members. The
    coupled channel axis is left untouched and the source channel count leaks
    through into the output.
    """
    coupler, source = _make_constant_coupler(1, input_time_dim=1, output_time_dim=3)
    fields = _source_fields(4, len(source))
    time_idx = [0] * coupler.coupled_integration_dim

    training = training_constant_coupling(
        fields, coupler.coupled_channel_indices, time_idx
    )

    # [len(time_idx), len(var_idx), C_source, F, H, W] instead of
    # [integration, B, C_coupled, F, H, W]
    assert training.shape == (
        len(time_idx),
        len(coupler.coupled_channel_indices),
        len(source),
        _FACE,
        _HEIGHT,
        _WIDTH,
    )


@requires_module(["xarray", "pandas"])
def test_constant_training_indexing_raises_when_channels_exceed_batch():
    """The same bug is fatal, not merely wrong, in the common case.

    Coupled channel indices routinely exceed the batch size, and indexing the
    batch axis with them raises.
    """
    coupler, source = _make_constant_coupler(3, input_time_dim=1, output_time_dim=3)
    fields = _source_fields(4, len(source))
    time_idx = [0] * coupler.coupled_integration_dim
    assert max(coupler.coupled_channel_indices) >= _BATCH

    with pytest.raises(IndexError, match="out of bounds for dimension 0"):
        training_constant_coupling(fields, coupler.coupled_channel_indices, time_idx)


# ---------------------------------------------------------------------------
# Physical-space rescaling is absent from both downstream transcriptions
# ---------------------------------------------------------------------------


@requires_module(["xarray", "pandas"])
def test_downstream_transcriptions_ignore_physical_rescaling():
    """Neither transcription applies denorm/renorm, so both drift once
    ``set_coupled_scaling`` is configured."""
    pd = pytest.importorskip("pandas")

    config = UNIFORM
    coupler, source = _make_trailing_coupler(
        3,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    # Incoming stats for the matched source channels, outgoing for the coupled
    # variables. Distinct means/stds make the affine map non-identity.
    incoming = {
        v: {"mean": 100.0 * (i + 1), "std": 5.0 * (i + 1)}
        for i, v in enumerate(coupler.incoming_variables)
    }
    outgoing = {
        v: {"mean": 90.0 * (i + 1), "std": 4.0 * (i + 1)}
        for i, v in enumerate(coupler.variables)
    }
    coupler.set_scaling(
        pd.DataFrame.from_dict(outgoing).T.to_xarray().astype("float32")
    )
    coupler.set_coupled_scaling(incoming)

    fields = _source_fields(config["n_time"], len(source))
    coupler.set_coupled_fields(fields)
    library = coupler.construct_integrated_couplings()

    inference = inference_trailing_average_coupling(
        fields,
        coupler.coupled_channel_indices,
        coupler.coupled_integration_dim,
        coupler.averaging_slices,
    )
    training = training_trailing_average_coupling(
        fields,
        coupler.coupled_channel_indices,
        config["n_output_times"],
        coupler.coupled_integration_dim,
        len(coupler.input_times),
    )

    assert not torch.allclose(library, inference, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(library, training, rtol=1e-3, atol=1e-3)
    # Without scaling all three agree, so the gap is the rescaling and nothing
    # else about the averaging.
    coupler.incoming_coupled_scaling = None
    coupler.outgoing_coupled_scaling = None
    coupler.set_coupled_fields(fields)
    assert torch.equal(coupler.construct_integrated_couplings(), inference)


@requires_module(["xarray", "pandas"])
def test_coupling_op_is_reused_across_calls():
    """set_coupled_fields runs per step, so the op must not be rebuilt each time."""
    config = UNIFORM
    coupler, _ = _make_trailing_coupler(
        3,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    first = coupler.coupling_op()
    assert coupler.coupling_op() is first
    # A different rescaling decision is a different op, not a mutation of this one.
    vetoed = coupler.coupling_op(rescale_in_physical_space=False)
    assert vetoed is not first
    assert coupler.coupling_op(rescale_in_physical_space=False) is vetoed
    assert first.rescale_in_physical_space is True


@requires_module(["xarray", "pandas"])
def test_coupling_op_cache_is_dropped_when_scaling_changes():
    """A cached op must never outlive the configuration it captured."""
    pd = pytest.importorskip("pandas")

    config = UNIFORM
    coupler, source = _make_trailing_coupler(
        3,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    fields = _source_fields(config["n_time"], len(source))
    before = coupler.coupling_op()
    unscaled = before(fields)
    assert before.rescales_through_physical is False

    coupler.set_scaling(
        pd.DataFrame.from_dict(
            {
                v: {"mean": 90.0 * (i + 1), "std": 4.0 * (i + 1)}
                for i, v in enumerate(coupler.variables)
            }
        )
        .T.to_xarray()
        .astype("float32")
    )
    coupler.set_coupled_scaling(
        {
            v: {"mean": 100.0 * (i + 1), "std": 5.0 * (i + 1)}
            for i, v in enumerate(coupler.incoming_variables)
        }
    )

    after = coupler.coupling_op()
    assert after is not before
    assert after.rescales_through_physical is True
    assert not torch.allclose(after(fields), unscaled, rtol=1e-3, atol=1e-3)

    # Clearing the statistics by direct assignment must invalidate too, since
    # that is how a caller reverts to normalized-space coupling.
    coupler.incoming_coupled_scaling = None
    coupler.outgoing_coupled_scaling = None
    reverted = coupler.coupling_op()
    assert reverted is not after
    assert torch.equal(reverted(fields), unscaled)


@requires_module(["xarray", "pandas"])
def test_coupling_op_cache_is_dropped_when_coupling_is_set_up_again():
    """setup_coupling can change channel indices and averaging windows."""
    config = UNIFORM
    coupler, source = _make_trailing_coupler(
        3,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    before = coupler.coupling_op()
    coupler.setup_coupling(_FakeSource(source, "3h"))
    assert coupler.coupling_op() is not before


@requires_module(["xarray", "pandas"])
def test_vetoing_rescaling_reproduces_pre_rescaling_inference_numbers():
    """A configured coupler can still emit the old, un-rescaled numbers.

    This is what makes the scaling change auditable: the same coupler, with its
    statistics attached, reproduces either side of the change depending only on
    ``rescale_in_physical_space``.
    """
    pd = pytest.importorskip("pandas")

    config = UNIFORM
    coupler, source = _make_trailing_coupler(
        3,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    incoming = {
        v: {"mean": 100.0 * (i + 1), "std": 5.0 * (i + 1)}
        for i, v in enumerate(coupler.incoming_variables)
    }
    outgoing = {
        v: {"mean": 90.0 * (i + 1), "std": 4.0 * (i + 1)}
        for i, v in enumerate(coupler.variables)
    }
    coupler.set_scaling(
        pd.DataFrame.from_dict(outgoing).T.to_xarray().astype("float32")
    )
    coupler.set_coupled_scaling(incoming)

    fields = _source_fields(config["n_time"], len(source))
    reference = inference_trailing_average_coupling(
        fields,
        coupler.coupled_channel_indices,
        coupler.coupled_integration_dim,
        coupler.averaging_slices,
    )

    vetoed = coupler.coupling_op(rescale_in_physical_space=False)
    honored = coupler.coupling_op()

    assert vetoed.rescales_through_physical is False
    assert honored.rescales_through_physical is True
    assert torch.equal(vetoed(fields), reference)
    assert not torch.allclose(honored(fields), reference, rtol=1e-3, atol=1e-3)
    # Vetoing must not drop the statistics, only the decision to act on them.
    assert vetoed.incoming_scaling is not None


# ---------------------------------------------------------------------------
# The unified op reproduces the transcriptions it replaces
# ---------------------------------------------------------------------------


@requires_module(["xarray", "pandas"])
@pytest.mark.parametrize("n_coupled", [1, 3])
@pytest.mark.parametrize(
    "config", [UNIFORM, NON_UNIFORM], ids=["uniform", "nonuniform"]
)
def test_coupling_op_reproduces_inference_transcription(n_coupled, config):
    """What the inferencer now calls matches the math it used to inline."""
    coupler, source = _make_trailing_coupler(
        n_coupled,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    fields = _source_fields(config["n_time"], len(source))

    got = coupler.coupling_op()(fields)
    expected = inference_trailing_average_coupling(
        fields,
        coupler.coupled_channel_indices,
        coupler.coupled_integration_dim,
        coupler.averaging_slices,
    )
    assert torch.equal(got, expected)


@requires_module(["xarray", "pandas"])
@pytest.mark.parametrize("n_coupled", [1, 3])
def test_coupling_op_from_spec_reproduces_training_transcription(n_coupled):
    """What the coupled model now calls matches the math it used to inline.

    The spec-derived path is checked against the training transcription with the
    batch-axis indexing bug corrected, which is the behavior change the shared
    op deliberately makes.
    """
    config = UNIFORM
    coupler, source = _make_trailing_coupler(
        n_coupled,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    fields = _source_fields(config["n_time"], len(source))
    n_windows = len(coupler.input_times)
    n_integrations = coupler.coupled_integration_dim

    # Equal contiguous chunks, the partition the training code derives from its
    # output timesteps.
    n_chunks = config["n_output_times"]
    size = -(-config["n_time"] // n_chunks)
    flat = [
        slice(start, min(start + size, config["n_time"]))
        for start in range(0, config["n_time"], size)
    ]
    windows = [flat[j * n_windows : (j + 1) * n_windows] for j in range(n_integrations)]

    op = CouplingOp.from_spec(
        {
            "method": "trailing_average",
            "variable_indices": coupler.coupled_channel_indices,
            "timestep_indices": list(range(n_windows)),
        },
        averaging_slices=windows,
    )

    expected = training_trailing_average_coupling(
        fields,
        coupler.coupled_channel_indices,
        n_chunks,
        n_integrations,
        n_windows,
    )
    assert torch.equal(op(fields), expected)


@requires_module(["xarray", "pandas"])
def test_coupling_op_from_spec_constant_matches_library():
    """The spec-derived constant op agrees with the library coupler.

    This is the divergence from ``test_constant_library_matches_training``
    closing: both now select on the channel axis and keep every coupled channel.
    """
    coupler, source = _make_constant_coupler(3, input_time_dim=1, output_time_dim=3)
    fields = _source_fields(4, len(source))

    coupler.set_coupled_fields(fields)
    library = coupler.construct_integrated_couplings()

    op = CouplingOp.from_spec(
        {
            "method": "constant",
            "variable_indices": coupler.coupled_channel_indices,
            "timestep_indices": [0] * coupler.coupled_integration_dim,
        }
    )
    assert torch.equal(op(fields), library)


# ---------------------------------------------------------------------------
# End-to-end differentiability of the inference coupling exchange
# ---------------------------------------------------------------------------


@requires_module(["xarray", "pandas"])
@pytest.mark.parametrize("with_scaling", [False, True])
def test_grad_reaches_source_state_through_coupled_fields(with_scaling):
    """A loss on the gathered coupled fields must reach the source state.

    This is the composition ``Inferencer.fetch_next_couplings`` performs:
    preset the couplers from a live model state, then gather their coupled
    fields. It used to run through ``numpy.concatenate``, which silently
    detached the state.
    """
    pd = pytest.importorskip("pandas")

    config = UNIFORM
    coupler, source = _make_trailing_coupler(
        2,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    if with_scaling:
        outgoing = {
            v: {"mean": 90.0 * (i + 1), "std": 4.0 * (i + 1)}
            for i, v in enumerate(coupler.variables)
        }
        coupler.set_scaling(
            pd.DataFrame.from_dict(outgoing).T.to_xarray().astype("float32")
        )
        coupler.set_coupled_scaling(
            {
                v: {"mean": 100.0 * (i + 1), "std": 5.0 * (i + 1)}
                for i, v in enumerate(coupler.incoming_variables)
            }
        )

    state = _source_fields(config["n_time"], len(source)).requires_grad_(True)
    coupler.set_coupled_fields(state)
    coupled = concat_integrated_couplings([coupler], dtype=torch.float32)

    assert coupled.requires_grad
    coupled.square().mean().backward()

    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    # Only the coupled channels are forced, so the others must stay untouched.
    coupled_channels = set(coupler.coupled_channel_indices)
    for channel in range(len(source)):
        grad = state.grad[:, :, :, channel].abs().sum().item()
        assert (grad > 0) == (channel in coupled_channels)


@requires_module(["xarray", "pandas"])
def test_concat_integrated_couplings_handles_dataset_and_preset_couplers():
    """Numpy from the dataset and tensors from a preset coupler both concatenate."""
    config = UNIFORM
    preset, source = _make_trailing_coupler(
        2,
        config["input_times"],
        config["input_time_dim"],
        config["output_time_dim"],
    )
    fields = _source_fields(config["n_time"], len(source))
    preset.set_coupled_fields(fields)

    class _NumpyCoupler:
        """Stands in for a coupler reading from the dataset."""

        def __init__(self, like):
            self.array = np.zeros(like.shape, dtype="float32")

        def construct_integrated_couplings(self):
            return self.array

    coupled = concat_integrated_couplings(
        [preset, _NumpyCoupler(preset.preset_coupled_fields)], dtype=torch.float32
    )
    assert coupled.shape[2] == 2 * preset.preset_coupled_fields.shape[2]
    assert torch.is_tensor(coupled)
