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
from physicsnemo.models.dlwp_healpix_layers.healpix_blocks import set_static_hidden_state  # noqa: E402

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


@pytest.mark.parametrize("kind", ["structural", "legacy"])
def test_static_hidden_state_keeps_eager_results(kind):
    model = _model(kind)
    with th.no_grad():
        multi = _multi_step(model, *_inputs())
        set_static_hidden_state(model)
        carried = _carried(model, *_inputs())
    assert th.equal(multi, carried)


@pytest.mark.skipif(not th.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("kind", ["structural", "legacy"])
def test_prime_and_carry_cuda_graphs_replay_the_carried_steps(kind):
    # Coupled inference with a carried state: one graph primes the GRU (first window),
    # another continues from the state the previous replay left (later windows).
    model = _model(kind).cuda()
    prog, insol, constants = _inputs("cuda")
    with th.no_grad():
        eager = _carried(model, prog, insol, constants)
        set_static_hidden_state(model)
        static = [prog.clone(), insol[:, :, :1].clone(), constants.clone()]
        stream = th.cuda.Stream()
        stream.wait_stream(th.cuda.current_stream())
        with th.cuda.stream(stream):  # warm-up allocates the hidden-state buffers
            model.reset()
            model(static)
            model(static)
        th.cuda.current_stream().wait_stream(stream)
        model.reset()
        hidden = [m.h.data_ptr() for m in model.modules() if hasattr(m, "static_hidden_state")]
        prime, carry = th.cuda.CUDAGraph(), th.cuda.CUDAGraph()
        with th.cuda.graph(prime):
            out_prime = model(static)
        with th.cuda.graph(carry, pool=prime.pool()):
            out_carry = model(static)
        # Both graphs read and write the hidden-state buffers allocated in the warm-up.
        assert [m.h.data_ptr() for m in model.modules() if hasattr(m, "static_hidden_state")] == hidden
        outs, state = [], prog
        for k in range(_STEPS):
            static[0].copy_(state)
            static[1].copy_(insol[:, :, k : k + 1])
            (prime if k == 0 else carry).replay()
            out = (out_prime if k == 0 else out_carry).clone()
            outs.append(out)
            state = out[:, :, -1:]
        replayed = th.cat(outs, dim=2)
    th.testing.assert_close(replayed, eager, rtol=1e-5, atol=1e-6)


def test_static_hidden_state_guards():
    model = _model("structural")
    set_static_hidden_state(model)
    with th.no_grad():
        model([*_inputs()[:1], _inputs()[1][:, :, :_STEPS], _inputs()[2]])
    gru = next(m for m in model.modules() if hasattr(m, "static_hidden_state"))
    with pytest.raises(RuntimeError, match="cannot take inputs"):
        gru(th.zeros(2, *gru.h.shape[1:]))
    with pytest.raises(RuntimeError, match="disable grad"):
        gru(th.zeros_like(gru.h))
    averaged = _model("structural")
    averaged.enforce_reflectional_equivariance = True
    with pytest.raises(ValueError, match="averaged"):
        set_static_hidden_state(averaged)
