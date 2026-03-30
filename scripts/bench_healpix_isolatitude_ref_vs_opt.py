#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Compare HEALPixPaddingIsolatitudeReference vs HEALPixPaddingIsolatitude (numerics + timing).

Run from the ``modulus-uw`` repo root with PYTHONPATH set, e.g. after::

    mamba activate UWv2

    cd /path/to/modulus-uw
    PYTHONPATH=. python scripts/bench_healpix_isolatitude_ref_vs_opt.py
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
        HEALPixPaddingIsolatitude,
        HEALPixPaddingIsolatitudeReference,
    )

    p = args.padding
    nside = args.nside
    device = torch.device(args.device)
    b = args.batch
    c = args.channels
    x = torch.randn(b * 12, c, nside, nside, device=device, dtype=torch.float32)

    ref = HEALPixPaddingIsolatitudeReference(padding=p).to(device)
    opt = HEALPixPaddingIsolatitude(padding=p, nside=nside).to(device)

    y_ref = ref(x)
    y_opt = opt(x)
    max_err = (y_ref - y_opt).abs().max().item()
    print(f"max_abs_diff={max_err}")
    if not torch.allclose(y_ref, y_opt, rtol=0, atol=1e-5):
        raise SystemExit("numerical mismatch between reference and optimized")

    def bench(name: str, fn, warmup: int, repeats: int) -> float:
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats

    t_ref = bench(
        "ref",
        lambda: ref(x),
        args.warmup,
        args.repeats,
    )
    t_opt = bench(
        "opt",
        lambda: opt(x),
        args.warmup,
        args.repeats,
    )
    print(f"mean_time_ref_s={t_ref:.6f}")
    print(f"mean_time_opt_s={t_opt:.6f}")
    if t_ref > 0:
        print(f"speedup_ref_over_opt={t_ref / t_opt:.2f}x")


if __name__ == "__main__":
    main()
