"""HEALPixRecUNet GRU state carried across forward() calls, as coupled inference does when a model takes one
step per coupler window: priming once and carrying the state (reset_cycle=inf) must give the same steps as one
multi-step forward that primes at its first step."""

import os
import sys

import pytest
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_reflection_recunet import _legacy_encoder_decoder, _structural_encoder_decoder  # noqa: E402

from physicsnemo.models.dlwp_healpix import HEALPixRecUNet  # noqa: E402

_STEPS, _NSIDE = 4, 8


def _model(kind):
    encoder, decoder = _structural_encoder_decoder() if kind == "structural" else _legacy_encoder_decoder()
    channels, constants = ["t", "u", "v"], ["sin_lat", "lsm"]
    structural = kind == "structural"
    th.manual_seed(0)
    model = HEALPixRecUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=3,
        output_channels=3,
        n_constants=2,
        decoder_input_channels=1,
        input_time_dim=1,
        output_time_dim=_STEPS,
        delta_time="6h",
        reset_cycle="24h",
        presteps=0,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=[8, 4],
        reflection_equivariance_mode="structural" if structural else "off",
        odd_prognostic_variables=["v"] if structural else None,
        odd_constants=["sin_lat"] if structural else None,
        channels=channels,
        constants=constants,
        scaling={k: {"mean": 0.0, "std": 1.0} for k in channels + constants},
    )
    return model.eval()


def _inputs(device="cpu"):
    g = th.Generator().manual_seed(1)
    prog = th.randn(1, 12, 1, 3, _NSIDE, _NSIDE, generator=g)
    insol = th.randn(1, 12, _STEPS, 1, _NSIDE, _NSIDE, generator=g)
    constants = th.randn(12, 2, _NSIDE, _NSIDE, generator=g)
    return [x.to(device) for x in (prog, insol, constants)]


def _multi_step(model, prog, insol, constants):
    """One forward over all steps; it primes the GRU at its first step (24 h reset cycle)."""
    model.output_time_dim, model.reset_cycle = _STEPS, 24
    model.reset()
    return model([prog, insol, constants])


def _carried(model, prog, insol, constants):
    """One step per forward, GRU state carried between calls (reset_cycle=inf, primed once)."""
    model.output_time_dim, model.reset_cycle = 1, float("inf")
    model.reset()
    outs, state = [], prog
    for k in range(_STEPS):
        out = model([state, insol[:, :, k : k + 1], constants])
        outs.append(out)
        state = out[:, :, -1:]
    return th.cat(outs, dim=2)


@pytest.mark.parametrize("kind", ["structural", "legacy"])
def test_carried_single_steps_equal_one_multi_step_forward(kind):
    model = _model(kind)
    with th.no_grad():
        multi = _multi_step(model, *_inputs())
        carried = _carried(model, *_inputs())
    assert multi.shape == carried.shape == (1, 12, _STEPS, 3, _NSIDE, _NSIDE)
    assert th.equal(multi, carried)
