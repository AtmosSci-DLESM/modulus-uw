import importlib.util
import math

import pytest
import torch

from physicsnemo.models.dlwp_healpix import HEALPixRecUNet
from physicsnemo.models.dlwp_healpix_layers.healpix_input_noise import (
    SpectralInputNoise,
    degree_variance,
    load_noise_spectrum,
    save_noise_spectrum,
    spectral_ramp,
)

pytest.importorskip("omegaconf")
pytest.importorskip("xarray")

has_sht = (
    torch.cuda.is_available()
    and importlib.util.find_spec("cuhpx") is not None
    and importlib.util.find_spec("earth2grid") is not None
)
requires_sht = pytest.mark.skipif(not has_sht, reason="needs CUDA, cuhpx and earth2grid")

NSIDE = 64
LMAX = 3 * NSIDE - 1
CHANNELS = ["a", "b", "c", "q"]
SCALING = {
    "a": {"mean": 0.0, "std": 1.0},
    "b": {"mean": 0.0, "std": 2.0},
    "c": {"mean": 0.0, "std": 2.0},
    "q": {"mean": 0.5, "std": 1.0},
}


def _spectrum() -> torch.Tensor:
    """Rows: a has noise in 32..120, b none, c is 4x a, q equals a."""
    ell = torch.arange(LMAX).double()
    band = (ell >= 32) & (ell <= 120)
    a = torch.zeros(LMAX, dtype=torch.float64)
    a[band] = 1e-3 * 64.0 / ell[band]
    return torch.stack([a, torch.zeros_like(a), 4.0 * a, a])


def _expected_degree_variance(row: int, std: float, ell_start=32, ell_full=64):
    window = spectral_ramp(torch.arange(LMAX), ell_start, ell_full)
    window[0] = 0.0
    return _spectrum()[row] * window.square() / std**2


@pytest.fixture
def spectrum_path(tmp_path):
    path = str(tmp_path / "spectrum.nc")
    save_noise_spectrum(path, CHANNELS, _spectrum(), {"note": "synthetic test spectrum"})
    return path


def _make(spectrum_path, **kwargs):
    kwargs.setdefault("variance_scale_range", (1.0, 1.0))
    return SpectralInputNoise(spectrum_path, CHANNELS, SCALING, **kwargs)


def _noise(module, batch, seed=0):
    """Draw noise on a zero state; returns ``[B, C, 12, H, W]``."""
    state = torch.zeros(batch, 12, 1, len(CHANNELS), NSIDE, NSIDE, device="cuda")
    torch.manual_seed(seed)
    return module(state)[:, :, 0].permute(0, 2, 1, 3, 4)


def _analyse(field):
    """Degree variance of ``[N, 12, H, W]`` face fields, averaged over N."""
    import earth2grid
    from cuhpx import SHTCUDA
    from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder

    sht = SHTCUDA(nside=NSIDE, lmax=LMAX, mmax=LMAX, quad_weights="ring")
    to_ring = (
        earth2grid.get_regridder(
            earth2grid.healpix.Grid(level=6, pixel_order=HEALPIX_PAD_XY),
            earth2grid.healpix.Grid(level=6, pixel_order=PixelOrder.RING),
        )
        .to(torch.float32)
        .to(field.device)
    )
    flat = field.reshape(field.shape[0], -1).contiguous()
    return degree_variance(sht(to_ring(flat))).mean(0).double().cpu()


def test_spectral_ramp_shape():
    ell = torch.arange(100)
    w = spectral_ramp(ell, 32, 64)
    assert torch.all(w[:33] == 0)
    assert torch.all(w[64:] == 1)
    assert torch.all(w[1:] >= w[:-1])
    assert w[48].item() == pytest.approx(0.5, abs=1e-6)
    with pytest.raises(ValueError):
        spectral_ramp(ell, 64, 64)


def test_spectrum_file_round_trip(tmp_path):
    path = str(tmp_path / "s.nc")
    save_noise_spectrum(path, CHANNELS, _spectrum())
    channels, values = load_noise_spectrum(path)
    assert channels == CHANNELS
    assert torch.equal(values, _spectrum())
    with pytest.raises(ValueError):
        save_noise_spectrum(path, CHANNELS[:2], _spectrum())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variance_scale_range": (2.0, 1.0)},
        {"variance_scale_range": (-1.0, 1.0)},
        {"variance_scale_range": (1.0,)},
        {"amplitude": -1.0},
        {"ell_start": 64, "ell_full": 64},
        {"ell_full": LMAX + 1},
        {"nside": 48},
        {"lower_bounds": {"nope": 0.0}},
    ],
)
def test_invalid_configuration_raises(spectrum_path, kwargs):
    with pytest.raises(ValueError):
        SpectralInputNoise(spectrum_path, CHANNELS, SCALING, **kwargs)


def test_missing_spectrum_file_is_unavailable_and_training_use_raises(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        module = SpectralInputNoise(str(tmp_path / "absent.nc"), CHANNELS, SCALING)
    assert not module.available
    assert "unavailable" in caplog.text
    # No state-dict entries: checkpoints load with or without the layer.
    assert len(module.state_dict()) == 0
    with pytest.raises(RuntimeError, match="spectrum file not found"):
        module(torch.zeros(1, 12, 1, len(CHANNELS), NSIDE, NSIDE))


def test_channel_missing_from_spectrum_file_raises(spectrum_path):
    with pytest.raises(ValueError, match="no entry"):
        SpectralInputNoise(spectrum_path, CHANNELS + ["extra"], SCALING)


@requires_sht
def test_state_dict_is_empty(spectrum_path):
    assert len(_make(spectrum_path).state_dict()) == 0


@requires_sht
def test_pixel_variance_and_degree_spectrum_match_target(spectrum_path):
    module = _make(spectrum_path)
    noise = _noise(module, batch=64)
    for row, name in ((0, "a"), (2, "c")):
        target = _expected_degree_variance(row, SCALING[name]["std"])
        field = noise[:, row]
        assert field.var().item() == pytest.approx(target.sum().item(), rel=0.05)
        measured = _analyse(field)
        # Per-degree match averaged over bands of ten degrees.
        for lo in range(30, 121, 10):
            band = slice(lo, lo + 10)
            if target[band].sum() > 0:
                assert measured[band].sum().item() == pytest.approx(
                    target[band].sum().item(), rel=0.10
                )
        # Nothing at or below ell_start: the draw has no large-scale content.
        assert measured[:32].sum().item() < 1e-3 * measured.sum().item()
    # The module reports the expected RMS it will produce at scale 1.
    assert module.expected_rms["a"] == pytest.approx(
        math.sqrt(_expected_degree_variance(0, 1.0).sum().item()), rel=1e-6
    )


@requires_sht
def test_zero_spectrum_channel_gets_no_noise_and_means_are_unchanged(spectrum_path):
    noise = _noise(_make(spectrum_path), batch=16)
    assert torch.all(noise[:, 1] == 0)
    # No degree-0 term, so the spatial mean of every sample stays at zero.
    means = noise.reshape(16, len(CHANNELS), -1).mean(-1)
    assert means.abs().max().item() < 1e-3 * noise.std().item()


@requires_sht
def test_channels_and_samples_are_independent_and_draws_are_seeded(spectrum_path):
    module = _make(spectrum_path)
    noise = _noise(module, batch=32, seed=3)
    flat = noise.reshape(32, len(CHANNELS), -1)
    # a and c share a spectrum shape; their noise must still be uncorrelated.
    a, c = flat[:, 0], flat[:, 2]
    corr = (a * c).sum() / (a.norm() * c.norm())
    assert abs(corr.item()) < 0.02
    corr_samples = (flat[0, 0] * flat[1, 0]).sum() / (flat[0, 0].norm() * flat[1, 0].norm())
    assert abs(corr_samples.item()) < 0.02
    assert torch.equal(noise, _noise(module, batch=32, seed=3))
    assert not torch.equal(noise, _noise(module, batch=32, seed=4))


@requires_sht
@pytest.mark.parametrize("scale_range", [(1.0, 1.0), (0.0, 2.0), (0.0, 5.0), (3.0, 3.0)])
def test_variance_scale_range(spectrum_path, scale_range):
    lo, hi = scale_range
    module = _make(spectrum_path, variance_scale_range=scale_range)
    batch = 128
    noise = _noise(module, batch=batch)
    target = _expected_degree_variance(0, 1.0).sum().item()
    per_sample = noise[:, 0].reshape(batch, -1).var(dim=1) / target
    # Every sample sits inside [lo, hi] up to sampling noise of the pixel variance.
    assert per_sample.min().item() >= lo * 0.9 - 1e-6
    assert per_sample.max().item() <= hi * 1.1 + 1e-6
    if lo == hi:
        assert per_sample.mean().item() == pytest.approx(lo, rel=0.05)
    else:
        mean = 0.5 * (lo + hi)
        spread = (hi - lo) / math.sqrt(12 * batch)
        assert per_sample.mean().item() == pytest.approx(mean, abs=4 * spread + 0.03)
        assert per_sample.max().item() - per_sample.min().item() > 0.6 * (hi - lo)


@requires_sht
def test_amplitude_scales_standard_deviation(spectrum_path):
    base = _noise(_make(spectrum_path), batch=16, seed=1)
    doubled = _noise(_make(spectrum_path, amplitude=2.0), batch=16, seed=1)
    assert torch.allclose(doubled, 2.0 * base, rtol=1e-5, atol=1e-7)


@requires_sht
def test_lower_bounds_clamp_after_noise(spectrum_path):
    kwargs = {"variance_scale_range": (5.0, 5.0)}
    clamped = _make(spectrum_path, lower_bounds={"q": 0.0}, **kwargs)
    unclamped = _make(spectrum_path, **kwargs)
    floor = (0.0 - SCALING["q"]["mean"]) / SCALING["q"]["std"]
    state = torch.zeros(8, 12, 1, len(CHANNELS), NSIDE, NSIDE, device="cuda")
    state[:, :, :, 3] = floor  # q sits at its physical zero, so half the noise crosses it
    torch.manual_seed(0)
    out = clamped(state)
    torch.manual_seed(0)
    ref = unclamped(state)
    q = out[:, :, :, 3]
    assert q.min().item() >= floor - 1e-6
    assert (q == floor).float().mean().item() > 0.3
    assert ref[:, :, :, 3].min().item() < floor - 0.5
    # Unbounded channels, and the pixels of q the noise left above the bound, are untouched.
    assert torch.equal(out[:, :, :, :3], ref[:, :, :, :3])
    above = ref[:, :, :, 3] > floor
    assert torch.equal(q[above], ref[:, :, :, 3][above])


@requires_sht
def test_state_dtype_and_broadcast_over_time(spectrum_path):
    module = _make(spectrum_path)
    state = torch.zeros(2, 12, 3, len(CHANNELS), NSIDE, NSIDE, device="cuda", dtype=torch.bfloat16)
    out = module(state)
    assert out.dtype == torch.bfloat16
    assert out.shape == state.shape
    # One field per member and channel, shared across the time axis.
    assert torch.equal(out[:, :, 0], out[:, :, 1])
    assert torch.equal(out[:, :, 0], out[:, :, 2])
    with pytest.raises(ValueError, match="expected state"):
        module(state[:, :6])
    with pytest.raises(ValueError, match="expected state"):
        module(state[:, :, :, :2])


@requires_sht
@pytest.mark.parametrize("compiled", [False, True])
def test_works_under_bf16_autocast_and_matches_fp32_statistics(spectrum_path, compiled):
    # Training runs under autocast (and torch.compile); the transform must stay in fp32.
    module = _make(spectrum_path)
    apply = torch.compile(module, backend="eager") if compiled else module
    state = torch.zeros(16, 12, 1, len(CHANNELS), NSIDE, NSIDE, device="cuda")
    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = apply(state)
    assert out.dtype == torch.float32
    target = _expected_degree_variance(0, SCALING["a"]["std"]).sum().item()
    assert out[:, :, 0, 0].var().item() == pytest.approx(target, rel=0.1)


@requires_sht
def test_noise_gradient_passes_through_state(spectrum_path):
    module = _make(spectrum_path)
    state = torch.zeros(1, 12, 1, len(CHANNELS), NSIDE, NSIDE, device="cuda", requires_grad=True)
    module(state).sum().backward()
    assert torch.equal(state.grad, torch.ones_like(state))


# ---------------------------------------------------------------------------
# RecUNet wiring
# ---------------------------------------------------------------------------

_ACT = {"_target_": "physicsnemo.models.layers.activations.CappedGELU", "cap_value": 10}


def _recunet(**kwargs):
    from omegaconf import DictConfig

    conv = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.ConvNeXtBlock",
        "in_channels": 3,
        "out_channels": 1,
        "activation": _ACT,
        "kernel_size": 3,
        "dilation": 1,
        "upscale_factor": 4,
        "_recursive_": True,
    }
    rec = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.ConvGRUBlock",
        "in_channels": 3,
        "kernel_size": 1,
        "_recursive_": False,
    }
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetEncoder",
        "conv_block": conv,
        "down_sampling_block": {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.AvgPool",
            "pooling": 2,
        },
        "recurrent_block": rec,
        "_recursive_": False,
        "n_channels": [136, 68, 34],
        "dilations": [1, 2, 4],
    }
    decoder = DictConfig(
        {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetDecoder",
            "conv_block": conv,
            "up_sampling_block": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.TransposedConvUpsample",
                "in_channels": 3,
                "out_channels": 1,
                "activation": _ACT,
                "upsampling": 2,
            },
            "recurrent_block": rec,
            "output_layer": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.BasicConvBlock",
                "in_channels": 3,
                "out_channels": 2,
                "kernel_size": 1,
                "dilation": 1,
                "n_layers": 1,
            },
            "_recursive_": False,
            "n_channels": [34, 68, 136],
            "dilations": [4, 2, 1],
        }
    )
    return HEALPixRecUNet(
        encoder=DictConfig(encoder),
        decoder=decoder,
        input_channels=2,
        output_channels=2,
        n_constants=2,
        decoder_input_channels=1,
        input_time_dim=1,
        output_time_dim=4,
        presteps=1,
        enable_healpixpad=True,
        delta_time="6h",
        **kwargs,
    )


class _AddConstant(torch.nn.Module):
    """Deterministic stand-in for the noise layer that records every call."""

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, state):
        self.calls.append(state.clone())
        return state + self.value


def _inputs(batch=2, size=16):
    torch.manual_seed(0)
    x = torch.randn(batch, 12, 2, 2, size, size)  # presteps + 1 time slices
    insolation = torch.randn(batch, 12, 8, 1, size, size)
    constants = torch.randn(12, 2, size, size)
    return [x, insolation, constants]


def _zero_decoder(model):
    # Zero decoder weights make every step the identity on the prognostics, so each
    # output is exactly the perturbed state that step received.
    with torch.no_grad():
        for p in model.decoder.parameters():
            p.zero_()
    return model


def test_recunet_default_has_no_noise_and_is_unchanged_in_train_mode():
    model = _zero_decoder(_recunet())
    assert model.input_noise is None
    inputs = _inputs()
    model.train()
    out = model(inputs)
    assert torch.equal(out[:, :, 0], inputs[0][:, :, 1])


def test_recunet_perturbs_initial_state_once_and_every_fed_back_state():
    c = 0.25
    model = _zero_decoder(_recunet())
    fake = _AddConstant(c)
    model.input_noise = fake
    inputs = _inputs()
    x = inputs[0]
    model.train()
    out = model(inputs)
    steps = out.shape[2]
    assert steps == 4
    # initial state perturbed once (warm-up and current share it), then three fed-back states
    assert len(fake.calls) == 4
    assert torch.equal(fake.calls[0], x)  # the whole initial-state tensor, once
    for k in range(steps):
        expected = x[:, :, 1] + c * (k + 1)
        assert torch.allclose(out[:, :, k, :2], expected, atol=1e-6)
    # fed-back calls received the previous (already perturbed) prediction
    for k in range(1, 4):
        assert torch.allclose(fake.calls[k][:, :, 0], x[:, :, 1] + c * k, atol=1e-6)
    # The caller's tensor is not modified in place.
    assert torch.equal(x, _inputs()[0])


def test_recunet_eval_mode_is_noise_free_and_matches_model_without_noise():
    model = _zero_decoder(_recunet())
    fake = _AddConstant(1.0)
    model.input_noise = fake
    model.eval()
    inputs = _inputs()
    with torch.no_grad():
        noisy = model(inputs)
        reference = _zero_decoder(_recunet()).eval()(inputs)
    assert fake.calls == []
    assert torch.equal(noisy, reference)


def test_recunet_residual_and_constraints_see_the_perturbed_input():
    seen = []

    class Recorder(torch.nn.Module):
        def forward(self, prediction, input):
            seen.append(input.clone())
            return prediction

    c = 0.5
    model = _zero_decoder(_recunet())
    model.input_noise = _AddConstant(c)
    model.constraints = [Recorder()]
    inputs = _inputs()
    model.train()
    model(inputs)
    assert len(seen) == 4
    assert torch.allclose(seen[0][:, :, 0], inputs[0][:, :, 1] + c, atol=1e-6)


@requires_sht
def test_recunet_builds_noise_from_config_and_adds_no_state(spectrum_path):
    names = ["x0", "x1"]
    path = spectrum_path.replace("spectrum.nc", "two.nc")
    save_noise_spectrum(path, names, _spectrum()[:2, :47])
    cfg = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_input_noise.SpectralInputNoise",
        "spectrum_path": path,
        "in_channels": names,
        "scaling": {n: {"mean": 0.0, "std": 1.0} for n in names},
        "nside": 16,
        "lmax": 47,
        "ell_start": 4,
        "ell_full": 8,
        "variance_scale_range": [0.0, 2.0],
    }
    model = _recunet(input_noise=cfg)
    assert isinstance(model.input_noise, SpectralInputNoise)
    assert model.input_noise.scale_hi == 2.0
    assert len(model.state_dict()) == len(_recunet().state_dict())
