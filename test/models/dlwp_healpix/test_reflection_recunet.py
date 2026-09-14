# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivariance tests for ReflectionSteerable / structural HEALPixRecUNet."""

import torch as th
from omegaconf import OmegaConf

from physicsnemo.models.dlwp_healpix import HEALPixRecUNet
from physicsnemo.models.dlwp_healpix_layers.reflection_ops import (
    hpx_spatial_reflect,
    resolve_reflection_equivariance_mode,
)


def _act_structural():
    return {"_target_": "physicsnemo.models.layers.activations.Tanh"}


def _act_baseline():
    return {"_target_": "physicsnemo.models.layers.activations.CappedGELU", "cap_value": 10}


def _structural_encoder_decoder(n_channels=(16, 16), nside=(8, 4), odd_fraction=0.25):
    conv = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.reflection_steerable_blocks.ReflectionSteerableMulti_SymmetricConvNeXtBlock",
        "activation": _act_structural(),
        "kernel_size": 3,
        "dilation": 1,
        "upscale_factor": 2,
        "n_layers": 1,
        "_recursive_": True,
    }
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetEncoder",
        "conv_block": conv,
        "down_sampling_block": {"_target_": "physicsnemo.models.dlwp_healpix_layers.AvgPool", "pooling": 2},
        "_recursive_": False,
        "n_channels": list(n_channels),
        "dilations": [1] * len(n_channels),
    }
    decoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetDecoder",
        "conv_block": conv,
        "up_sampling_block": {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.Interpolate",
            "scale_factor": 2,
            "mode": "nearest",
        },
        "recurrent_block": {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.reflection_steerable_blocks.ReflectionSteerableConvGRUBlock",
            "kernel_size": 1,
            "_recursive_": False,
        },
        "output_layer": {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.reflection_steerable_blocks.ReflectionSteerableBasicConvBlock",
            "kernel_size": 1,
            "n_layers": 1,
            "_recursive_": False,
        },
        "_recursive_": False,
        "n_channels": list(reversed(n_channels)),
        "dilations": [1] * len(n_channels),
    }
    return OmegaConf.create(encoder), OmegaConf.create(decoder)


def _build_structural_model():
    encoder, decoder = _structural_encoder_decoder()
    channels = ["t", "u", "v"]
    constants = ["sin_lat", "lsm"]
    scaling = {k: {"mean": 0.0, "std": 1.0} for k in channels + constants}
    return HEALPixRecUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=3,
        output_channels=3,
        n_constants=2,
        decoder_input_channels=1,
        input_time_dim=1,
        output_time_dim=1,
        delta_time="6h",
        reset_cycle="24h",
        presteps=0,
        residual_prediction=True,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=[8, 4],
        reflection_equivariance_mode="structural",
        enforce_reflectional_equivariance=False,
        odd_prognostic_variables=["v"],
        odd_constants=["sin_lat"],
        odd_fraction=0.25,
        channels=channels,
        constants=constants,
        scaling=scaling,
    )


def _make_batch(nside=8, batch=1):
    B, F, T = batch, 12, 1
    prog = th.randn(B, F, T, 3, nside, nside)
    di = th.randn(B, F, T + 2, 1, nside, nside)
    constants = th.randn(F, 2, nside, nside)
    return [prog, di, constants]


def _reflect_bftchw(x: th.Tensor, odd_channels):
    """Reflect [B,F,C,H,W] with odd channel sign flips."""
    B, F, C, H, W = x.shape
    folded = x.reshape(B * F, C, H, W)
    folded = hpx_spatial_reflect(folded).clone()
    for i in odd_channels:
        folded[:, i] *= -1
    return folded.reshape(B, F, C, H, W)


def _reflect_batch(batch):
    prog, di, constants = batch
    prog_r = _reflect_bftchw(prog[:, :, 0], [2]).unsqueeze(2)
    di_r = di.clone()
    di_r[:, :, 0] = _reflect_bftchw(di[:, :, 0], [])
    H = constants.shape[-1]
    cf = constants.reshape(12, constants.shape[1], H, H)
    cf = hpx_spatial_reflect(cf).clone()
    cf[:, 0] *= -1
    return [prog_r, di_r, cf]


def test_resolve_mode_backcompat():
    assert resolve_reflection_equivariance_mode(None, False) == "off"
    assert resolve_reflection_equivariance_mode("off", True) == "averaged"
    assert resolve_reflection_equivariance_mode("structural", False) == "structural"
    assert resolve_reflection_equivariance_mode("averaged", False) == "averaged"
    # Legacy aliases
    assert resolve_reflection_equivariance_mode("steerable", False) == "structural"
    assert resolve_reflection_equivariance_mode("reynolds", False) == "averaged"
    try:
        resolve_reflection_equivariance_mode("structural", True)
        raise AssertionError("expected conflict")
    except ValueError:
        pass


def _legacy_encoder_decoder():
    encoder = OmegaConf.create(
        {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetEncoder",
            "conv_block": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.Multi_SymmetricConvNeXtBlock",
                "activation": _act_baseline(),
                "kernel_size": 3,
                "upscale_factor": 2,
                "n_layers": 1,
                "_recursive_": True,
            },
            "down_sampling_block": {"_target_": "physicsnemo.models.dlwp_healpix_layers.AvgPool", "pooling": 2},
            "_recursive_": False,
            "n_channels": [8, 8],
            "dilations": [1, 1],
        }
    )
    decoder = OmegaConf.create(
        {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetDecoder",
            "conv_block": encoder.conv_block,
            "up_sampling_block": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.Interpolate",
                "scale_factor": 2,
                "mode": "nearest",
            },
            "recurrent_block": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.ConvGRUBlock",
                "kernel_size": 1,
                "_recursive_": False,
            },
            "output_layer": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.BasicConvBlock",
                "kernel_size": 1,
                "n_layers": 1,
            },
            "_recursive_": False,
            "n_channels": [8, 8],
            "dilations": [1, 1],
        }
    )
    return encoder, decoder


def test_mode_off_default_path():
    encoder, decoder = _legacy_encoder_decoder()
    model = HEALPixRecUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=3,
        output_channels=3,
        n_constants=2,
        decoder_input_channels=1,
        input_time_dim=1,
        output_time_dim=1,
        presteps=0,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=[8, 4],
        reflection_equivariance_mode="off",
        channels=["t", "u", "v"],
        constants=["sin_lat", "lsm"],
        scaling={k: {"mean": 0.0, "std": 1.0} for k in ["t", "u", "v", "sin_lat", "lsm"]},
    )
    assert model.reflection_equivariance_mode == "off"
    model.eval()
    with th.no_grad():
        out = model(_make_batch())
    assert out.shape[0] == 1


def test_legacy_averaged_still_works():
    """``averaged``, legacy ``reynolds``, and ``enforce_=True`` all run a forward pass."""
    encoder, decoder = _legacy_encoder_decoder()
    common = dict(
        encoder=encoder,
        decoder=decoder,
        input_channels=3,
        output_channels=3,
        n_constants=2,
        decoder_input_channels=1,
        input_time_dim=1,
        output_time_dim=1,
        presteps=0,
        residual_prediction=True,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=[8, 4],
        channels=["t", "u", "v"],
        constants=["sin_lat", "lsm"],
        scaling={k: {"mean": 0.0, "std": 1.0} for k in ["t", "u", "v", "sin_lat", "lsm"]},
        odd_prognostic_variables=["v"],
        odd_constants=["sin_lat"],
    )
    for kwargs in (
        dict(reflection_equivariance_mode="averaged", enforce_reflectional_equivariance=False),
        dict(reflection_equivariance_mode="reynolds", enforce_reflectional_equivariance=False),
        dict(reflection_equivariance_mode=None, enforce_reflectional_equivariance=True),
    ):
        model = HEALPixRecUNet(**common, **kwargs)
        assert model.reflection_equivariance_mode == "averaged"
        model.eval()
        with th.no_grad():
            out = model(_make_batch())
        assert out.shape[0] == 1


def test_structural_recunet_reflected_input_gives_reflected_output():
    th.manual_seed(0)
    model = _build_structural_model()
    model.eval()
    batch = _make_batch()
    with th.no_grad():
        y = model(batch)
        y_r = model(_reflect_batch(batch))
    y_rho = _reflect_bftchw(y[:, :, 0], [2]).unsqueeze(2)
    th.testing.assert_close(y_r, y_rho, rtol=1e-3, atol=1e-4)


def test_structural_symmetric_input_symmetric_output():
    th.manual_seed(1)
    model = _build_structural_model()
    model.eval()
    batch = _make_batch()
    prog, di, constants = batch

    def symmetrize_bftchw(x, odd_idx):
        return 0.5 * (x + _reflect_bftchw(x, odd_idx))

    prog_s = symmetrize_bftchw(prog[:, :, 0], [2]).unsqueeze(2)
    di_s = di.clone()
    di_s[:, :, 0] = symmetrize_bftchw(di[:, :, 0], [])
    H = constants.shape[-1]
    cf = constants.reshape(12, constants.shape[1], H, H)
    rf = hpx_spatial_reflect(cf).clone()
    rf[:, 0] *= -1
    const_s = 0.5 * (cf + rf)

    with th.no_grad():
        y = model([prog_s, di_s, const_s])
    y0 = y[:, :, 0]
    th.testing.assert_close(y0, _reflect_bftchw(y0, [2]), rtol=1e-3, atol=1e-4)


def _build_structural_model_with_odd_diagnostic():
    """Prognostics t,u,v plus absolute odd diagnostic v_diag."""
    encoder, decoder = _structural_encoder_decoder()
    channels = ["t", "u", "v"]
    output_channel_names = ["t", "u", "v", "v_diag"]
    constants = ["sin_lat", "lsm"]
    scaling = {k: {"mean": 0.0, "std": 1.0} for k in channels + ["v_diag"] + constants}
    return HEALPixRecUNet(
        encoder=encoder,
        decoder=decoder,
        input_channels=3,
        output_channels=4,
        n_constants=2,
        decoder_input_channels=1,
        input_time_dim=1,
        output_time_dim=1,
        delta_time="6h",
        reset_cycle="24h",
        presteps=0,
        residual_prediction=True,
        hpx_padding_mode="isolatitude",
        compile_padding=False,
        nside=[8, 4],
        reflection_equivariance_mode="structural",
        enforce_reflectional_equivariance=False,
        odd_prognostic_variables=["v"],
        odd_diagnostic_variables=["v_diag"],
        odd_constants=["sin_lat"],
        odd_fraction=0.25,
        channels=channels,
        output_channel_names=output_channel_names,
        constants=constants,
        scaling=scaling,
    )


def test_structural_odd_diagnostic_bank_and_equivariance():
    th.manual_seed(2)
    model = _build_structural_model_with_odd_diagnostic()
    # Output layout [t, u, v, v_diag]: even = t,u (2); odd = v, v_diag (2)
    assert model.structural_output_n_even == 2
    odd_out = set(model.odd_out_var_idx.tolist())
    assert odd_out == {2, 3}

    model.eval()
    batch = _make_batch()
    with th.no_grad():
        y = model(batch)
        y_r = model(_reflect_batch(batch))
    assert y.shape[3] == 4
    # ρ flips sign on both odd prognostic (v) and odd diagnostic (v_diag)
    y_rho = _reflect_bftchw(y[:, :, 0], [2, 3]).unsqueeze(2)
    th.testing.assert_close(y_r, y_rho, rtol=1e-3, atol=1e-4)


def test_odd_diagnostic_requires_output_channel_names():
    encoder, decoder = _structural_encoder_decoder()
    try:
        HEALPixRecUNet(
            encoder=encoder,
            decoder=decoder,
            input_channels=3,
            output_channels=4,
            n_constants=2,
            decoder_input_channels=1,
            input_time_dim=1,
            output_time_dim=1,
            presteps=0,
            hpx_padding_mode="isolatitude",
            compile_padding=False,
            nside=[8, 4],
            reflection_equivariance_mode="structural",
            odd_prognostic_variables=["v"],
            odd_diagnostic_variables=["v_diag"],
            odd_constants=["sin_lat"],
            channels=["t", "u", "v"],
            constants=["sin_lat", "lsm"],
            scaling={k: {"mean": 0.0, "std": 1.0} for k in ["t", "u", "v", "v_diag", "sin_lat", "lsm"]},
        )
        raise AssertionError("expected ValueError for missing output_channel_names")
    except ValueError as e:
        assert "output_channel_names" in str(e)
