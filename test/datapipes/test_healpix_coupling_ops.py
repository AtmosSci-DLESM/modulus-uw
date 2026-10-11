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

"""Tests for the stateless HEALPix coupling operations."""

from functools import partial

import pytest
import torch

from physicsnemo.datapipes.healpix.coupling_ops import (
    CouplingOp,
    apply_coupling,
    constant_coupling,
    denormalize,
    renormalize,
    rescale_through_physical,
    trailing_average_coupling,
)

_BATCH, _FACE, _HEIGHT, _WIDTH = 2, 4, 3, 3


def _fields(n_time=8, n_channels=3, dtype=torch.float32, requires_grad=False):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(
        _BATCH,
        _FACE,
        n_time,
        n_channels,
        _HEIGHT,
        _WIDTH,
        dtype=dtype,
        generator=generator,
    )
    return x.requires_grad_(requires_grad)


def _scaling(n_channels, mean, std, dtype=torch.float32):
    return {
        "mean": torch.full((1, 1, 1, n_channels, 1, 1), mean, dtype=dtype),
        "std": torch.full((1, 1, 1, n_channels, 1, 1), std, dtype=dtype),
    }


# ---------------------------------------------------------------------------
# constant_coupling
# ---------------------------------------------------------------------------


def test_constant_coupling_broadcasts_first_time_step():
    fields = _fields(n_time=5)
    out = constant_coupling(fields, integration_dim=3)

    assert out.shape == (_BATCH, _FACE, 3, 3, _HEIGHT, _WIDTH)
    for i in range(3):
        assert torch.equal(out[:, :, i], fields[:, :, 0])


@pytest.mark.parametrize("time_index", [0, 2, -1, -3])
def test_constant_coupling_honors_time_index(time_index):
    fields = _fields(n_time=5)
    out = constant_coupling(fields, integration_dim=2, time_index=time_index)
    expected = fields[:, :, time_index]
    assert torch.equal(out[:, :, 0], expected)
    assert torch.equal(out[:, :, 1], expected)


def test_constant_coupling_does_not_alias_input():
    """The result must survive later mutation of the source tensor."""
    fields = _fields(n_time=4)
    out = constant_coupling(fields, integration_dim=2)
    snapshot = out.clone()
    fields.mul_(-5.0)
    assert torch.equal(out, snapshot)


# ---------------------------------------------------------------------------
# trailing_average_coupling
# ---------------------------------------------------------------------------


def test_trailing_average_coupling_averages_each_window():
    fields = _fields(n_time=16, n_channels=2)
    slices = [[slice(0, 4), slice(4, 8)], [slice(8, 12), slice(12, 16)]]
    out = trailing_average_coupling(fields, slices)

    # [B, F, integration, timevar, H, W] with timevar period-major
    assert out.shape == (_BATCH, _FACE, 2, 4, _HEIGHT, _WIDTH)
    for j, windows in enumerate(slices):
        for i, window in enumerate(windows):
            expected = fields[:, :, window].mean(dim=2)
            got = out[:, :, j, i * 2 : (i + 1) * 2]
            assert torch.allclose(got, expected, rtol=0, atol=1e-6)


def test_trailing_average_coupling_supports_uneven_windows():
    fields = _fields(n_time=12, n_channels=1)
    slices = [[slice(0, 2), slice(2, 8)]]
    out = trailing_average_coupling(fields, slices)
    assert torch.allclose(out[:, :, 0, 0], fields[:, :, 0:2, 0].mean(dim=2))
    assert torch.allclose(out[:, :, 0, 1], fields[:, :, 2:8, 0].mean(dim=2))


# ---------------------------------------------------------------------------
# scaling round trip
# ---------------------------------------------------------------------------


def test_denormalize_renormalize_round_trip():
    fields = _fields()
    scaling = _scaling(3, mean=100.0, std=7.0)
    assert torch.allclose(
        renormalize(denormalize(fields, scaling), scaling), fields, atol=1e-5
    )


def test_rescale_through_physical_matches_manual_pipeline():
    fields = _fields(n_time=8, n_channels=2)
    incoming = _scaling(2, mean=900.0, std=90.0)
    # Two windows over two variables gives a timevar axis of four.
    outgoing = _scaling(4, mean=800.0, std=80.0)
    slices = [[slice(0, 4), slice(4, 8)]]
    op = partial(trailing_average_coupling, averaging_slices=slices)

    got = rescale_through_physical(fields, op, incoming, outgoing)
    expected = renormalize(op(denormalize(fields, incoming)), outgoing)
    assert torch.allclose(got, expected, atol=1e-5)


def test_rescale_through_physical_uses_float32_midflight_and_restores_dtype():
    """A float16 denorm of large-magnitude fields would overflow."""
    incoming = _scaling(1, mean=60000.0, std=5000.0, dtype=torch.float32)
    outgoing = _scaling(1, mean=60000.0, std=5000.0, dtype=torch.float32)
    fields = _fields(n_time=4, n_channels=1).to(torch.float16)
    slices = [[slice(0, 4)]]
    op = partial(trailing_average_coupling, averaging_slices=slices)

    out = rescale_through_physical(fields, op, incoming, outgoing)
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# apply_coupling
# ---------------------------------------------------------------------------


def test_apply_coupling_selects_channels_and_permutes():
    fields = _fields(n_time=4, n_channels=5)
    out = apply_coupling(fields, [3, 1], partial(constant_coupling, integration_dim=2))
    # [T, B, C, F, H, W]
    assert out.shape == (2, _BATCH, 2, _FACE, _HEIGHT, _WIDTH)
    assert torch.equal(out[0, :, 0], fields[:, :, 0, 3])
    assert torch.equal(out[0, :, 1], fields[:, :, 0, 1])


def test_apply_coupling_time_first_false_keeps_source_layout():
    fields = _fields(n_time=4, n_channels=3)
    out = apply_coupling(
        fields,
        [0, 2],
        partial(constant_coupling, integration_dim=2),
        time_first=False,
    )
    assert out.shape == (_BATCH, _FACE, 2, 2, _HEIGHT, _WIDTH)


def test_apply_coupling_allows_repeated_channel_indices():
    """Several coupled variables may share one source channel."""
    fields = _fields(n_time=4, n_channels=3)
    out = apply_coupling(fields, [1, 1], partial(constant_coupling, integration_dim=1))
    assert torch.equal(out[0, :, 0], out[0, :, 1])


def test_apply_coupling_rejects_half_configured_scaling():
    fields = _fields()
    op = partial(constant_coupling, integration_dim=1)
    with pytest.raises(ValueError, match="must be provided together"):
        apply_coupling(fields, [0], op, incoming_scaling=_scaling(1, 0.0, 1.0))
    with pytest.raises(ValueError, match="must be provided together"):
        apply_coupling(fields, [0], op, outgoing_scaling=_scaling(1, 0.0, 1.0))


def test_apply_coupling_is_pure():
    """Calling twice gives the same answer and leaves the input untouched."""
    fields = _fields(n_time=8, n_channels=2)
    before = fields.clone()
    op = partial(
        trailing_average_coupling, averaging_slices=[[slice(0, 4), slice(4, 8)]]
    )
    first = apply_coupling(fields, [0, 1], op)
    second = apply_coupling(fields, [0, 1], op)
    assert torch.equal(first, second)
    assert torch.equal(fields, before)


# ---------------------------------------------------------------------------
# differentiability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_scaling", [False, True])
@pytest.mark.parametrize("mode", ["constant", "trailing"])
def test_apply_coupling_propagates_gradients(mode, with_scaling):
    fields = _fields(n_time=8, n_channels=2, requires_grad=True)
    if mode == "constant":
        op = partial(constant_coupling, integration_dim=2)
        n_outgoing = 2
    else:
        op = partial(
            trailing_average_coupling, averaging_slices=[[slice(0, 4), slice(4, 8)]]
        )
        # Two windows over two variables gives a timevar axis of four.
        n_outgoing = 4
    scaling = (
        dict(
            incoming_scaling=_scaling(2, 900.0, 90.0),
            outgoing_scaling=_scaling(n_outgoing, 800.0, 80.0),
        )
        if with_scaling
        else {}
    )

    out = apply_coupling(fields, [0, 1], op, **scaling)
    out.square().sum().backward()

    assert fields.grad is not None
    assert torch.isfinite(fields.grad).all()
    assert fields.grad.abs().sum() > 0


def test_rescale_through_physical_keeps_float64_precision():
    """Promotion must not silently downcast inputs wider than float32."""
    fields = _fields(n_time=4, n_channels=1, dtype=torch.float64)
    incoming = _scaling(1, 900.0, 90.0, dtype=torch.float64)
    outgoing = _scaling(1, 800.0, 80.0, dtype=torch.float64)
    op = partial(trailing_average_coupling, averaging_slices=[[slice(0, 4)]])

    out = rescale_through_physical(fields, op, incoming, outgoing)
    expected = renormalize(op(denormalize(fields, incoming)), outgoing)
    assert out.dtype == torch.float64
    # float32 mid-flight would leave an error many orders of magnitude larger.
    assert (out - expected).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
# CouplingOp
# ---------------------------------------------------------------------------


def _constant_op(**kwargs):
    return CouplingOp(
        mode="constant", channel_indices=[0, 2], integration_dim=2, **kwargs
    )


def _trailing_op(**kwargs):
    return CouplingOp(
        mode="trailing_average",
        channel_indices=[0, 2],
        averaging_slices=[[slice(0, 2), slice(2, 4)]],
        **kwargs,
    )


def test_coupling_op_matches_apply_coupling():
    fields = _fields(n_time=4, n_channels=3)
    op = _trailing_op()
    expected = apply_coupling(
        fields,
        [0, 2],
        partial(
            trailing_average_coupling,
            averaging_slices=[[slice(0, 2), slice(2, 4)]],
        ),
    )
    assert torch.equal(op(fields), expected)


def test_coupling_op_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown coupling mode"):
        CouplingOp(mode="median", channel_indices=[0], integration_dim=1)


def test_coupling_op_requires_mode_specific_config():
    with pytest.raises(ValueError, match="requires integration_dim"):
        CouplingOp(mode="constant", channel_indices=[0])
    with pytest.raises(ValueError, match="requires averaging_slices"):
        CouplingOp(mode="trailing_average", channel_indices=[0])


def test_coupling_op_state_dict_is_empty():
    """Indices and statistics are derived, so they must not enter checkpoints."""
    op = _trailing_op()
    op.set_scaling(_scaling(2, 900.0, 90.0), _scaling(4, 800.0, 80.0))
    assert op.state_dict() == {}


def test_coupling_op_buffers_follow_to():
    op = _trailing_op()
    op.set_scaling(_scaling(2, 900.0, 90.0), _scaling(4, 800.0, 80.0))
    op = op.to(torch.float64)
    assert op.incoming_mean.dtype == torch.float64
    # Index buffers are integral and must not be swept up by a dtype cast.
    assert op.channel_indices.dtype == torch.long


def test_coupling_op_set_scaling_round_trips_and_clears():
    op = _trailing_op()
    assert op.incoming_scaling is None
    assert op.outgoing_scaling is None

    incoming, outgoing = _scaling(2, 900.0, 90.0), _scaling(4, 800.0, 80.0)
    op.set_scaling(incoming, outgoing)
    assert torch.equal(op.incoming_scaling["mean"], incoming["mean"])
    assert torch.equal(op.outgoing_scaling["std"], outgoing["std"])

    op.set_scaling(None, None)
    assert op.incoming_scaling is None
    assert op.outgoing_scaling is None


def test_coupling_op_set_scaling_rejects_half_configured():
    op = _trailing_op()
    with pytest.raises(ValueError, match="must be set together"):
        op.set_scaling(_scaling(2, 0.0, 1.0), None)


# ---------------------------------------------------------------------------
# Gating physical-space rescaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["constant", "trailing_average"])
def test_apply_coupling_veto_reduces_in_normalized_space(mode):
    """rescale_in_physical_space=False must ignore statistics that are present."""
    fields = _fields(n_time=4, n_channels=3)
    if mode == "constant":
        physical_op = partial(constant_coupling, integration_dim=2)
        n_out = 2
    else:
        physical_op = partial(
            trailing_average_coupling, averaging_slices=[[slice(0, 2), slice(2, 4)]]
        )
        n_out = 4

    incoming, outgoing = _scaling(2, 900.0, 90.0), _scaling(n_out, 800.0, 80.0)
    unscaled = apply_coupling(fields, [0, 2], physical_op)
    vetoed = apply_coupling(
        fields,
        [0, 2],
        physical_op,
        incoming_scaling=incoming,
        outgoing_scaling=outgoing,
        rescale_in_physical_space=False,
    )
    honored = apply_coupling(
        fields,
        [0, 2],
        physical_op,
        incoming_scaling=incoming,
        outgoing_scaling=outgoing,
    )

    assert torch.equal(vetoed, unscaled)
    assert not torch.allclose(honored, unscaled)


def test_apply_coupling_veto_still_rejects_half_configured_scaling():
    """A half-configured pair is a config error even when rescaling is vetoed."""
    fields = _fields(n_time=4, n_channels=3)
    with pytest.raises(ValueError, match="must be provided together"):
        apply_coupling(
            fields,
            [0, 2],
            partial(constant_coupling, integration_dim=2),
            incoming_scaling=_scaling(2, 0.0, 1.0),
            rescale_in_physical_space=False,
        )


def test_coupling_op_reports_effective_rescaling_decision():
    op = _trailing_op()
    assert op.rescale_in_physical_space is True
    # Allowed but no statistics yet, so nothing is rescaled.
    assert op.rescales_through_physical is False

    op.set_scaling(_scaling(2, 900.0, 90.0), _scaling(4, 800.0, 80.0))
    assert op.rescales_through_physical is True

    vetoed = _trailing_op(rescale_in_physical_space=False)
    vetoed.set_scaling(_scaling(2, 900.0, 90.0), _scaling(4, 800.0, 80.0))
    assert vetoed.rescales_through_physical is False


@pytest.mark.parametrize("mode", ["constant", "trailing_average"])
def test_coupling_op_veto_matches_unscaled_forward(mode):
    fields = _fields(n_time=4, n_channels=3)
    build = _constant_op if mode == "constant" else _trailing_op
    n_out = 2 if mode == "constant" else 4

    baseline = build()
    vetoed = build(rescale_in_physical_space=False)
    vetoed.set_scaling(_scaling(2, 900.0, 90.0), _scaling(n_out, 800.0, 80.0))

    assert torch.equal(vetoed(fields), baseline(fields))


def test_coupling_op_from_spec_reads_rescaling_from_config():
    spec = {
        "method": "constant",
        "variable_indices": [0, 2],
        "timestep_indices": [0, 0],
        "rescale_in_physical_space": False,
    }
    assert CouplingOp.from_spec(spec).rescale_in_physical_space is False
    # An explicit argument overrides the spec.
    assert (
        CouplingOp.from_spec(
            spec, rescale_in_physical_space=True
        ).rescale_in_physical_space
        is True
    )
    # A silent spec defaults to allowing it.
    del spec["rescale_in_physical_space"]
    assert CouplingOp.from_spec(spec).rescale_in_physical_space is True


def test_coupling_op_forward_does_not_mutate_state():
    """Repeated calls must be identical; forward is used under CUDA graphs."""
    fields = _fields(n_time=4, n_channels=3)
    op = _trailing_op()
    before = {k: v.clone() for k, v in op.named_buffers()}
    first, second = op(fields), op(fields)
    assert torch.equal(first, second)
    for name, buffer in op.named_buffers():
        assert torch.equal(buffer, before[name])


def test_coupling_op_from_spec_constant_derives_integration_and_time_index():
    op = CouplingOp.from_spec(
        {
            "method": "constant",
            "variable_indices": [1, 3],
            "timestep_indices": [-1, -1, -1],
        }
    )
    assert op.mode == "constant"
    assert op.integration_dim == 3
    assert op.time_index == -1
    assert op.channel_indices.tolist() == [1, 3]

    fields = _fields(n_time=5, n_channels=4)
    assert torch.equal(op(fields)[0, :, 0], fields[:, :, -1, 1])


def test_coupling_op_from_spec_rejects_varying_constant_timesteps():
    with pytest.raises(ValueError, match="single repeated source"):
        CouplingOp.from_spec(
            {
                "method": "constant",
                "variable_indices": [0],
                "timestep_indices": [0, 1],
            }
        )


def test_coupling_op_from_spec_accepts_channel_index_override():
    """A caller may hand over indices it already holds as a tensor."""
    indices = torch.tensor([2, 0])
    op = CouplingOp.from_spec(
        {
            "method": "constant",
            "variable_indices": [0, 1],
            "timestep_indices": [0],
        },
        channel_indices=indices,
    )
    assert op.channel_indices.tolist() == [2, 0]


# ---------------------------------------------------------------------------
# gradcheck
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_scaling", [False, True])
@pytest.mark.parametrize("mode", ["constant", "trailing_average"])
def test_coupling_op_gradcheck(mode, with_scaling):
    generator = torch.Generator().manual_seed(3)
    fields = torch.randn(
        2, 2, 4, 3, 2, 2, dtype=torch.float64, generator=generator
    ).requires_grad_(True)

    if mode == "constant":
        op = _constant_op()
        n_outgoing = 2
    else:
        op = _trailing_op()
        n_outgoing = 4
    if with_scaling:
        op.set_scaling(
            _scaling(2, 900.0, 90.0, dtype=torch.float64),
            _scaling(n_outgoing, 800.0, 80.0, dtype=torch.float64),
        )

    assert torch.autograd.gradcheck(op, (fields,), eps=1e-6, atol=1e-8)
