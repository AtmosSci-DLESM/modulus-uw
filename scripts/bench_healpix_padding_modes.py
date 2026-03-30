#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark HEALPix padding modes (uncompiled ``torch.nn.Module`` forwards).

Runs **karlbauer** (``HEALPixPadding``), **isolatitude_reference**,
**isolatitude** (gather), and **earth2grid** (``HEALPixPaddingv2`` on CUDA when
available). Does not use ``torch.compile``.

Also checks that **isolatitude** matches **isolatitude_reference** numerically
(unless ``--skip-numerics-check``).

Run from the ``modulus-uw`` repo root with PYTHONPATH set, e.g.::

    mamba activate UWv2

    cd /path/to/modulus-uw
    PYTHONPATH=. python scripts/bench_healpix_padding_modes.py --nside 64
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
    HEALPixPadding,
    HEALPixPaddingIsolatitude,
    HEALPixPaddingIsolatitudeReference,
    HEALPixPaddingv2,
    have_earth2grid,
)


def bench_module(
    mod: torch.nn.Module,
    x: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> float:
    """Mean forward time in seconds (synchronizes CUDA when applicable)."""
    dev = x.device
    mod = mod.to(dev)
    with torch.no_grad():
        _ = mod(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(warmup):
            _ = mod(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = mod(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nside",
        type=int,
        default=64,
        help="HEALPix face height/width (square faces).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=1,
        help="Symmetric pad width per edge (typical 1 for 3x3 conv).",
    )
    parser.add_argument("--batch", type=int, default=2, help="Batch size (faces = batch * 12).")
    parser.add_argument("--channels", type=int, default=8, help="Channel depth C.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--skip-numerics-check",
        action="store_true",
        help="Do not verify isolatitude vs isolatitude_reference.",
    )
    parser.add_argument(
        "--skip-earth2grid",
        action="store_true",
        help="Skip earth2grid (HEALPixPaddingv2) even on CUDA.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Only run CPU benchmarks (no CUDA section).",
    )
    args = parser.parse_args()

    H = args.nside
    p = args.padding
    B, C = args.batch, args.channels

    x_cpu = torch.randn(B * 12, C, H, H, dtype=torch.float32)

    ref = HEALPixPaddingIsolatitudeReference(padding=p)
    opt = HEALPixPaddingIsolatitude(padding=p, nside=H)

    if not args.skip_numerics_check:
        with torch.no_grad():
            y_ref = ref(x_cpu)
            y_opt = opt(x_cpu)
        max_err = (y_ref - y_opt).abs().max().item()
        print(f"isolatitude vs isolatitude_reference: max_abs_diff={max_err}")
        if not torch.allclose(y_ref, y_opt, rtol=0, atol=1e-5):
            print("ERROR: numerical mismatch between isolatitude and isolatitude_reference.", file=sys.stderr)
            sys.exit(1)

    print(
        f"Timing: face_H={H}, padding={p}, batch={B}, channels={C}, "
        f"warmup={args.warmup}, repeats={args.repeats} (torch.compile not used)\n"
    )

    cpu_specs: list[tuple[str, torch.nn.Module]] = [
        ("karlbauer (HEALPixPadding)", HEALPixPadding(padding=p, enable_nhwc=False)),
        ("isolatitude_reference", HEALPixPaddingIsolatitudeReference(padding=p)),
        (
            "isolatitude (gather)",
            HEALPixPaddingIsolatitude(padding=p, nside=H),
        ),
    ]

    print("--- CPU ---")
    for label, mod in cpu_specs:
        t = bench_module(mod, x_cpu, warmup=args.warmup, repeats=args.repeats)
        print(f"  {label:32s}  mean_forward_s={t:.6f}")

    if args.cpu_only:
        return

    print("\n--- earth2grid (HEALPixPaddingv2, CUDA) ---")
    if args.skip_earth2grid:
        print("  skipped: --skip-earth2grid")
    elif not torch.cuda.is_available():
        print("  skipped: no CUDA")
    elif not have_earth2grid:
        print("  skipped: earth2grid healpix pad not importable")
    else:
        x_cuda = x_cpu.to("cuda")
        try:
            t = bench_module(
                HEALPixPaddingv2(padding=p),
                x_cuda,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            print(f"  {'earth2grid':32s}  mean_forward_s={t:.6f}")
        except Exception as e:
            print(f"  failed: {e}")

    if torch.cuda.is_available():
        print("\n--- Same three modes on CUDA (uncompiled) ---")
        x_cuda = x_cpu.to("cuda")
        for label, mod in cpu_specs:
            t = bench_module(
                mod,
                x_cuda,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            print(f"  {label:32s}  mean_forward_s={t:.6f}")


if __name__ == "__main__":
    main()
