#!/usr/bin/env python
"""Ablation wall-time for structural (ReflectionSteerable) optimizations vs theory variants."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from typing import Callable, Iterator

import torch as th

import scripts.bench_structural_vs_averaged as base
from physicsnemo.models.layers.activations import CappedGELU
from physicsnemo.models.dlwp_healpix_layers import reflection_steerable_blocks as psb
from physicsnemo.models.dlwp_healpix_layers.reflection_steerable_conv import ParitySplitActivation


@dataclass(frozen=True)
class Ablation:
    name: str
    description: str
    patch: Callable[[], contextlib.AbstractContextManager]


@contextlib.contextmanager
def _monkeypatch(obj, attr, value) -> Iterator[None]:
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


@contextlib.contextmanager
def _patch_steerc1x1_default(use_sin_lat_gate: bool) -> Iterator[None]:
    orig = psb._ReflectionSteerableConv1x1Wrap.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_sin_lat_gate", use_sin_lat_gate)
        return orig(self, *args, **kwargs)

    with _monkeypatch(psb._ReflectionSteerableConv1x1Wrap, "__init__", __init__):
        yield


@contextlib.contextmanager
def _patch_split_activation() -> Iterator[None]:
    def _even():
        return CappedGELU(cap_value=10)

    def _odd():
        return th.nn.Tanh()

    orig_init = psb.ReflectionSteerableSymmetricConvNeXtBlock.__init__

    def patched_init(self, *args, activation=None, **kwargs):
        orig_init(self, *args, activation=activation, **kwargs)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ParitySplitActivation):
                n = layer.n_even + layer.n_odd
                odd_fraction = layer.n_odd / n if n else 0.25
                self.layers[i] = ParitySplitActivation(
                    n,
                    odd_fraction,
                    _even(),
                    _odd(),
                    unified=False,
                    allow_non_tanh=True,  # ablation only; production forbids this
                )

    with _monkeypatch(psb.ReflectionSteerableSymmetricConvNeXtBlock, "__init__", patched_init):
        yield


@contextlib.contextmanager
def _patch_gru_theory() -> Iterator[None]:
    orig_init = psb.ReflectionSteerableConvGRUBlock.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        n_even = self.n_even
        ch = self.channels
        of = self.odd_fraction
        self.conv_gates = psb.ReflectionSteerableConv1x1(
            in_channels=2 * n_even,
            out_channels=2 * n_even,
            odd_fraction=0.0,
            in_even=2 * n_even,
            out_even=2 * n_even,
            use_sin_lat_gate=False,
        )
        self.conv_can = psb.ReflectionSteerableConv1x1(
            in_channels=2 * ch,
            out_channels=ch,
            odd_fraction=of,
            in_even=2 * n_even,
            out_even=n_even,
            use_sin_lat_gate=True,
        )
        self.needs_sin_lat = True

    def patched_forward(self, inputs, sin_lat_gate=None):
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        xe, xo = psb.split_banks(inputs, self.n_even)
        he, ho = psb.split_banks(self.h, self.n_even)
        gates = self.conv_gates(th.cat([xe, he], dim=1))
        reset_e, update_e = th.sigmoid(gates[:, : self.n_even]), th.sigmoid(gates[:, self.n_even :])
        if self.n_odd > 0:
            reset_o = reset_e[:, : self.n_odd]
            update_o = update_e[:, : self.n_odd]
        else:
            reset_o = reset_e[:, :0]
            update_o = update_e[:, :0]
        combined_can = psb.merge_banks(
            th.cat([xe, reset_e * he], dim=1), th.cat([xo, reset_o * ho], dim=1)
        )
        cnm = th.tanh(self.conv_can(combined_can, sin_lat=sin_lat_gate))
        update_gate = psb.merge_banks(update_e, update_o)
        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next
        return inputs + h_next

    with _monkeypatch(psb.ReflectionSteerableConvGRUBlock, "__init__", patched_init):
        with _monkeypatch(psb.ReflectionSteerableConvGRUBlock, "forward", patched_forward):
            yield


@contextlib.contextmanager
def _patch_gru_full_steerable_gates() -> Iterator[None]:
    orig_init = psb.ReflectionSteerableConvGRUBlock.__init__
    orig_forward = psb.ReflectionSteerableConvGRUBlock.forward

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        ch = self.channels
        n_even = self.n_even
        of = self.odd_fraction
        self.conv_gates = psb.ReflectionSteerableConv1x1(
            in_channels=2 * ch,
            out_channels=2 * ch,
            odd_fraction=of,
            in_even=2 * n_even,
            out_even=2 * n_even,
            use_sin_lat_gate=False,
        )

    def patched_forward(self, inputs, sin_lat_gate=None):
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        gates = self.conv_gates(th.cat([inputs, self.h], dim=1))
        reset_gate, update_gate = th.split(th.sigmoid(gates), self.channels, dim=1)
        combined = th.cat([inputs, reset_gate * self.h], dim=1)
        cnm = th.tanh(self.conv_can(combined))
        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next
        return inputs + h_next

    with _monkeypatch(psb.ReflectionSteerableConvGRUBlock, "__init__", patched_init):
        with _monkeypatch(psb.ReflectionSteerableConvGRUBlock, "forward", patched_forward):
            yield


@contextlib.contextmanager
def _patch_theory_combo() -> Iterator[None]:
    with _patch_steerc1x1_default(True):
        with _patch_gru_theory():
            with _patch_split_activation():
                yield


ABLATIONS = [
    Ablation("current", "same-parity 1×1, even plain gates, unified tanh", contextlib.nullcontext),
    Ablation("sin_lat_1x1", "sin_lat cross-parity at all 1×1", lambda: _patch_steerc1x1_default(True)),
    Ablation("gru_theory", "steerable even gates + sin_lat candidate", _patch_gru_theory),
    Ablation("gru_full_gates", "steerable gates on full [x|h]", _patch_gru_full_steerable_gates),
    Ablation("split_act", "CappedGELU even + tanh odd", _patch_split_activation),
    Ablation("theory_combo", "sin_lat 1×1 + theory GRU + split act", _patch_theory_combo),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--nside", type=int, default=64)
    args = parser.parse_args()

    device = "cuda:0" if th.cuda.is_available() else "cpu"
    nsides = (args.nside, args.nside // 2, args.nside // 4)
    batch = base.make_batch(nside=args.nside, device=device)

    off = base.build_model("off", nside=nsides, production=True).to(device)
    off_train = base.bench_train(off, batch, warmup=args.warmup, iters=args.iters)
    del off
    if batch[0].is_cuda:
        th.cuda.empty_cache()

    print(f"device={device} nside={nsides} iters={args.iters}")
    print(f"off train baseline: {off_train*1e3:.1f} ms\n")
    print(f"{'variant':<16} {'fwd_ms':>8} {'train_ms':>9} {'train/off':>10} {'Δtrain%':>9}")
    print("-" * 56)

    current_train = None
    for ab in ABLATIONS:
        with ab.patch():
            model = base.build_model("structural", nside=nsides, production=True).to(device)
            fwd = base.bench_fwd(model, batch, warmup=args.warmup, iters=args.iters)
            train = base.bench_train(model, batch, warmup=args.warmup, iters=args.iters)
            del model
            if batch[0].is_cuda:
                th.cuda.empty_cache()

        if ab.name == "current":
            current_train = train
        delta = 0.0 if ab.name == "current" else 100 * (train / current_train - 1)
        print(
            f"{ab.name:<16} {fwd*1e3:8.1f} {train*1e3:9.1f} "
            f"{train/off_train:10.3f} {delta:+8.1f}%"
        )


if __name__ == "__main__":
    main()
