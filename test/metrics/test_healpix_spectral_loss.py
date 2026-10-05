"""Spectral power loss: log-degree weights, and a CUDA SHT identity check."""

from types import SimpleNamespace

import pytest
import torch

from physicsnemo.metrics.climate.healpix_spectral_loss import (
    POWER_FLOOR_FRACTION,
    SpectralPowerSoftConstraint,
    degree_power,
    log_ell_relative_power_loss,
)

NL = 191


def _inv_ell(n_l=NL):
    ell = torch.arange(1, n_l, dtype=torch.float64)
    inv = 1.0 / ell
    return inv, inv.sum()


def _loss_with_fractional_error(degrees, n_l=NL):
    """Unit fractional error on ``degrees`` and zero elsewhere, including l=0."""
    target = torch.ones(n_l, dtype=torch.float64)
    pred = target.clone()
    for ell in degrees:
        pred[ell] = 2.0 * target[ell]
    inv, inv_sum = _inv_ell(n_l)
    return log_ell_relative_power_loss(pred, target, inv, inv_sum)


def _harmonic_weight(degrees, n_l=NL):
    inv, inv_sum = _inv_ell(n_l)
    chosen = sum(inv[ell - 1].item() for ell in degrees)
    return chosen / inv_sum.item()


def test_octave_weight_is_sum_of_inv_ell():
    """Equal fractional errors contribute in proportion to sum(1/l)."""
    low = range(2, 5)  # l = 2–4
    high = range(64, 129)  # l = 64–128
    low_loss = _loss_with_fractional_error(low).item()
    high_loss = _loss_with_fractional_error(high).item()
    assert low_loss == pytest.approx(_harmonic_weight(low))
    assert high_loss == pytest.approx(_harmonic_weight(high))
    # A true octave sums to about log(2). l = 64–127 is that octave; l = 128
    # is the first degree of the next one, so the named window sits just above.
    octave = sum(1.0 / ell for ell in range(64, 128))
    assert octave == pytest.approx(0.693, rel=0.02)
    named = sum(1.0 / ell for ell in high)
    assert named == pytest.approx(octave + 1.0 / 128)


def test_large_absolute_error_does_not_swamp_a_small_fraction():
    """A 1000-unit miss on a 1e6 low-l coefficient is a 0.1% relative error."""
    target = torch.ones(NL, dtype=torch.float64)
    pred = target.clone()
    target[1] = 1.0e6
    pred[1] = 1.0e6 + 1.0e3
    for ell in range(64, 129):
        pred[ell] = 2.0 * target[ell]
    inv, inv_sum = _inv_ell()
    loss = log_ell_relative_power_loss(pred, target, inv, inv_sum).item()
    high_only = _harmonic_weight(range(64, 129))
    low_only = (1.0e-3) ** 2 * _harmonic_weight([1])
    assert loss == pytest.approx(high_only + low_only, rel=1e-6)
    assert low_only < 1e-3 * high_only


def test_l0_does_not_enter_the_loss():
    target = torch.ones(NL, dtype=torch.float64)
    pred = target.clone()
    pred[0] = 1.0e6
    inv, inv_sum = _inv_ell()
    loss = log_ell_relative_power_loss(pred, target, inv, inv_sum)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_floor_caps_a_near_zero_degree():
    target = torch.ones(NL, dtype=torch.float64)
    pred = target.clone()
    target[10] = 0.0
    pred[10] = 1.0
    inv, inv_sum = _inv_ell()
    loss = log_ell_relative_power_loss(pred, target, inv, inv_sum).item()
    mean_target = target[1:].mean().item()
    floor = POWER_FLOOR_FRACTION * mean_target
    relative_sq = (1.0 / floor) ** 2
    expected = relative_sq * (1.0 / 10) / inv_sum.item()
    assert loss == pytest.approx(expected)


def test_scale_broadcasts_to_each_variable():
    mod = SpectralPowerSoftConstraint(
        name="spectral", relative_loss_scale=0.1, nside=8
    )
    mod.setup(
        SimpleNamespace(
            device=torch.device("cpu"),
            output_variables=["TMP2m", "PRATEsfc"],
        )
    )
    name, terms, scale = mod.constraint_spec()
    assert name == "spectral"
    assert terms == ["TMP2m", "PRATEsfc"]
    assert scale == {"TMP2m": 0.1, "PRATEsfc": 0.1}


def _cuda_sht_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from cuhpx import SHTCUDA  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _cuda_sht_available(), reason="CUDA + cuhpx SHT required")
def test_identical_fields_and_constant_offset():
    nside = 8
    mod = SpectralPowerSoftConstraint(
        name="spectral", relative_loss_scale=0.1, nside=nside
    )
    device = torch.device("cuda")
    mod.setup(
        SimpleNamespace(
            device=device,
            output_variables=["a", "b"],
        )
    )
    torch.manual_seed(0)
    field = torch.randn(1, 12, 2, 2, nside, nside, device=device)
    same = mod.constraint_loss(field, field, average_channels=False)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-6)

    # A constant offset is l=0. The spectral term ignores that degree.
    shifted = field + 3.0
    offset = mod.constraint_loss(shifted, field, average_channels=False)
    assert torch.allclose(offset, torch.zeros_like(offset), atol=1e-4)

    constant = torch.ones(1, 12, 1, 1, nside, nside, device=device)
    power = degree_power(mod._to_alm(constant))
    tail = power[..., 1:].max() / power[..., 0]
    assert tail.item() < 1e-3


@pytest.mark.skipif(not _cuda_sht_available(), reason="CUDA + cuhpx SHT required")
def test_grad_matches_stock_sht():
    """Recomputed backward matches differentiating through the SHT."""
    nside = 8
    mod = SpectralPowerSoftConstraint(
        name="spectral", relative_loss_scale=0.1, nside=nside
    )
    device = torch.device("cuda")
    mod.setup(
        SimpleNamespace(device=device, output_variables=["a", "b", "c"])
    )
    torch.manual_seed(1)
    pred = torch.randn(1, 12, 1, 3, nside, nside, device=device, requires_grad=True)
    target = torch.randn_like(pred)
    mod.channel_chunk = 2
    loss = mod.constraint_loss(pred, target, average_channels=True)
    loss.backward()

    pred_ref = pred.detach().clone().requires_grad_(True)
    ring = mod._faces_to_ring(pred_ref)
    alm = mod.sht(ring)
    pred_power = mod._degree_power(alm)
    with torch.no_grad():
        target_power = mod._degree_power(mod._to_alm(target))
    ref = log_ell_relative_power_loss(
        pred_power,
        target_power,
        mod.inv_ell,
        mod.inv_ell_sum,
        floor_fraction=mod.floor_fraction,
    ).mean()
    ref.backward()
    assert torch.allclose(pred.grad, pred_ref.grad, rtol=1e-4, atol=1e-5)
