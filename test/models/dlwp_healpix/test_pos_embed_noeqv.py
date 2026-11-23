"""Positional embedding and the non-equivariant layers used with it."""

import torch
from omegaconf import OmegaConf

from physicsnemo.models.dlwp_healpix_layers import UNetEncoder
from physicsnemo.models.dlwp_healpix_layers.coupled_partial_conv import PartialHEALPixConv2d
from physicsnemo.models.dlwp_healpix_layers.healpix_ring_mean_pchip import RingMeanPCHIPConv
from physicsnemo.models.dlwp_healpix_layers.reflection_ops import project_even_kernel
from physicsnemo.models.layers.activations import Tanh

NSIDE = 8
LAYERS = "physicsnemo.models.dlwp_healpix_layers"


def _encoder(pos_embed):
    torch.manual_seed(0)
    return UNetEncoder(
        conv_block=OmegaConf.create({"_target_": f"{LAYERS}.ConvNeXtBlock", "in_channels": 2}),
        down_sampling_block=OmegaConf.create({"_target_": f"{LAYERS}.DealiasedDownsample", "resample_filter": [1.0, 2.0, 1.0], "stride": 2}),
        n_channels=(4, 8),
        input_channels=2,
        nside=(NSIDE, NSIDE // 2),
        pos_embed=pos_embed,
    )


def test_encoder_pos_embed():
    plain, embedded = _encoder(False), _encoder(True)
    assert set(embedded.state_dict()) - set(plain.state_dict()) == {"pos_embed"}
    # 1-D, so MuonWithAux hands it to the AdamW aux optimizer.
    assert embedded.pos_embed.shape == (12 * 4 * NSIDE * NSIDE,)

    x = torch.randn(2 * 12, 2, NSIDE, NSIDE).to(memory_format=torch.channels_last)
    with_embed = embedded(x)
    assert with_embed[0].is_contiguous(memory_format=torch.channels_last)
    with_embed[-1].sum().backward()
    assert embedded.pos_embed.grad is not None and embedded.pos_embed.grad.abs().sum() > 0
    embed = embedded.pos_embed.detach().clone().view(12, 4, NSIDE, NSIDE)
    with torch.no_grad():
        embedded.pos_embed.zero_()
        without = embedded(x)
    # Level 0 differs by exactly the embedding, the same field for every batch member.
    diff = (with_embed[0] - without[0]).detach().unflatten(0, (2, 12))
    torch.testing.assert_close(diff, embed.expand(2, -1, -1, -1, -1))


def test_ring_mean_pchip_conv_shape_and_grad():
    block = RingMeanPCHIPConv(in_channels=4, out_channels=3, nside=NSIDE, activation=Tanh())
    assert not any("Steerable" in type(m).__name__ for m in block.modules() if m is not block)
    x = torch.randn(12, 4, NSIDE, NSIDE, requires_grad=True)
    y = block(x)
    assert y.shape == (12, 3, 2 * NSIDE, 2 * NSIDE)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def _stem_kernel_after_step(reflection_equivariant):
    torch.manual_seed(0)
    pconv = PartialHEALPixConv2d(
        channels=2, nside=NSIDE, hpx_padding_mode="karlbauer", reflection_equivariant=reflection_equivariant
    )
    x = torch.randn(12, 2, NSIDE, NSIDE)
    opt = torch.optim.SGD(pconv.parameters(), lr=1.0)
    (pconv(x, torch.ones_like(x)) * torch.randn_like(x)).sum().backward()
    opt.step()
    w = pconv._conv_weight(pconv.feature_conv).detach()
    return w, project_even_kernel(w)


def test_stem_projection_switch():
    w, even = _stem_kernel_after_step(True)
    torch.testing.assert_close(w, even)
    w, even = _stem_kernel_after_step(False)
    assert (w - even).abs().max() > 1e-3
