#!/usr/bin/env python
"""Compare forward+backward wall time: off vs averaged vs structural RecUNet.

Production-like defaults match ``dlesym-aq_v2-5_structural`` widths
(nside=64, channels [256,128,128], upscale_factor=4, n_layers=2).

Modes: ``structural`` = ReflectionSteerable layers; ``averaged`` = twin-forward
Reynolds projector. Success criterion: structural cheaper than averaged.

Optional ``--compile`` mirrors ``trainer.compile_model=True`` (``torch.compile``).
"""

from __future__ import annotations

import argparse
import time

import torch as th
from omegaconf import OmegaConf

from physicsnemo.models.dlwp_healpix import HEALPixRecUNet


def _act(structural: bool):
    # Structural path requires tanh; averaged/off keep the baseline CappedGELU.
    if structural:
        return {"_target_": "physicsnemo.models.layers.activations.Tanh"}
    return {"_target_": "physicsnemo.models.layers.activations.CappedGELU", "cap_value": 10}


def _base_blocks(structural: bool, production: bool = True):
    if structural:
        conv_t = (
            "physicsnemo.models.dlwp_healpix_layers.reflection_steerable_blocks."
            "ReflectionSteerableMulti_SymmetricConvNeXtBlock"
        )
        gru_t = (
            "physicsnemo.models.dlwp_healpix_layers.reflection_steerable_blocks.ReflectionSteerableConvGRUBlock"
        )
        out_t = (
            "physicsnemo.models.dlwp_healpix_layers.reflection_steerable_blocks.ReflectionSteerableBasicConvBlock"
        )
        extra = {}
    else:
        conv_t = "physicsnemo.models.dlwp_healpix_layers.Multi_SymmetricConvNeXtBlock"
        gru_t = "physicsnemo.models.dlwp_healpix_layers.ConvGRUBlock"
        out_t = "physicsnemo.models.dlwp_healpix_layers.BasicConvBlock"
        extra = {}
    if production:
        n_channels_enc = [256, 128, 128]
        n_channels_dec = [128, 128, 256]
        upscale_factor = 4
        n_layers = 2
    else:
        n_channels_enc = [128, 64]
        n_channels_dec = [64, 128]
        upscale_factor = 2
        n_layers = 1
    conv = {
        "_target_": conv_t,
        "activation": _act(structural),
        "kernel_size": 3,
        "upscale_factor": upscale_factor,
        "n_layers": n_layers,
        "_recursive_": True,
        **extra,
    }
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetEncoder",
        "conv_block": conv,
        "down_sampling_block": {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.AvgPool",
            "pooling": 2,
        },
        "_recursive_": False,
        "n_channels": n_channels_enc,
        "dilations": [1] * len(n_channels_enc),
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
            "_target_": gru_t,
            "kernel_size": 1,
            "_recursive_": False,
            **extra,
        },
        "output_layer": {
            "_target_": out_t,
            "kernel_size": 1,
            "n_layers": 1,
            **extra,
        },
        "_recursive_": False,
        "n_channels": n_channels_dec,
        "dilations": [1] * len(n_channels_dec),
    }
    return OmegaConf.create(encoder), OmegaConf.create(decoder)


def build_model(mode: str, nside, production: bool = True, odd_fraction: float = 0.25):
    encoder, decoder = _base_blocks(structural=(mode == "structural"), production=production)
    channels = ["t", "u", "v"]
    constants = ["sin_lat", "lsm"]
    scaling = {k: {"mean": 0.0, "std": 1.0} for k in channels + constants}
    kwargs = dict(
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
        nside=list(nside),
        channels=channels,
        constants=constants,
        scaling=scaling,
        odd_prognostic_variables=["v"],
        odd_constants=["sin_lat"],
        odd_fraction=0.25,
    )
    if mode == "structural":
        kwargs["reflection_equivariance_mode"] = "structural"
        kwargs["enforce_reflectional_equivariance"] = False
    elif mode == "averaged":
        kwargs["reflection_equivariance_mode"] = "averaged"
        kwargs["enforce_reflectional_equivariance"] = False
    else:
        kwargs["reflection_equivariance_mode"] = "off"
        kwargs["enforce_reflectional_equivariance"] = False
    return HEALPixRecUNet(**kwargs)


def make_batch(nside=64, device="cuda"):
    B, F, T = 1, 12, 1
    prog = th.randn(B, F, T, 3, nside, nside, device=device)
    di = th.randn(B, F, T + 2, 1, nside, nside, device=device)
    constants = th.randn(F, 2, nside, nside, device=device)
    return [prog, di, constants]


def bench_train(model, batch, warmup=5, iters=20):
    model.train()
    for _ in range(warmup):
        out = model(batch)
        loss = out.float().pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
    if batch[0].is_cuda:
        th.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = model(batch)
        loss = out.float().pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
    if batch[0].is_cuda:
        th.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_fwd(model, batch, warmup=5, iters=20):
    model.eval()
    with th.no_grad():
        for _ in range(warmup):
            _ = model(batch)
        if batch[0].is_cuda:
            th.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(batch)
        if batch[0].is_cuda:
            th.cuda.synchronize()
        return (time.perf_counter() - t0) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--small", action="store_true", help="Use smaller toy widths")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["off", "structural", "averaged"],
        choices=["off", "structural", "averaged"],
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap model in torch.compile (trainer.compile_model=True)",
    )
    parser.add_argument(
        "--compile-both",
        action="store_true",
        help="Run each mode with compile=False and compile=True",
    )
    args = parser.parse_args()

    device = "cuda:0" if th.cuda.is_available() else "cpu"
    production = not args.small
    nside = args.nside
    if production:
        nsides = (nside, nside // 2, nside // 4)
    else:
        nsides = (nside, nside // 2)
    compile_settings = (False, True) if args.compile_both else (bool(args.compile),)
    print(
        f"device={device} production={production} nside={nsides} "
        f"iters={args.iters} compile_settings={compile_settings}"
    )
    batch = make_batch(nside=nside, device=device)

    results = {}
    for do_compile in compile_settings:
        for mode in args.modes:
            model = build_model(mode, nside=nsides, production=production).to(device)
            if do_compile:
                model = th.compile(model)
            warm = max(args.warmup, 15 if do_compile else args.warmup)
            nparams = sum(p.numel() for p in model.parameters())
            fwd = bench_fwd(model, batch, warmup=warm, iters=args.iters)
            train = bench_train(model, batch, warmup=warm, iters=args.iters)
            results[(do_compile, mode)] = (train, nparams, fwd)
            print(
                f"compile={do_compile} {mode}: "
                f"train {train*1e3:.2f} ms/iter, fwd {fwd*1e3:.2f} ms/iter, params={nparams}"
            )
            del model
            if batch[0].is_cuda:
                th.cuda.empty_cache()

    for do_compile in compile_settings:
        if (do_compile, "off") in results and (do_compile, "structural") in results:
            off_t, _, off_f = results[(do_compile, "off")]
            st_t, _, st_f = results[(do_compile, "structural")]
            print(
                f"compile={do_compile} structural/off "
                f"train={st_t/off_t:.3f} fwd={st_f/off_f:.3f}"
            )
        if (do_compile, "averaged") in results and (do_compile, "structural") in results:
            re_t, _, re_f = results[(do_compile, "averaged")]
            st_t, _, st_f = results[(do_compile, "structural")]
            print(
                f"compile={do_compile} structural/averaged "
                f"train={st_t/re_t:.3f} fwd={st_f/re_f:.3f}"
            )


if __name__ == "__main__":
    main()
