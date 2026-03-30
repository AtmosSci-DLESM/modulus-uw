#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Compare **earth2grid** Karlbauer padding via the **indexing** backend (pure Python
path: ``pure_python.pad`` / ``pad_with_dim``) against **modulus-uw**
``HEALPixPaddingIsolatitude`` (gather-based isolatitude).

earth2grid's public ``pad()`` normally uses a CUDA extension for Karlbauer on GPU.
This script forces ``PaddingBackends.indexing`` so Karlbauer uses the same logical
strategy as in ``earth2grid/healpix/_padding/pure_python.py`` (precomputed indices +
two ``index_select``-style gathers with blending).

modulus isolatitude expects **folded** tensors ``[N * 12, C, H, W]``; earth2grid
``pad`` expects ``[N, 12, C, H, W]``. The script reshapes between them.

Run from ``modulus-uw`` with both packages on ``PYTHONPATH`` / environment::

    mamba activate UWv2
    cd /path/to/modulus-uw
    PYTHONPATH=.:../earth2grid python scripts/bench_healpix_e2g_python_karlbauer_vs_modulus_isolatitude.py

Adjust ``../earth2grid`` if your clone lives elsewhere.
"""

from __future__ import annotations

import argparse
import time

import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import HEALPixPaddingIsolatitude


def bench_callable(
    fn,
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> float:
    with torch.no_grad():
        _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(warmup):
            _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=64, help="Face height/width H (square faces).")
    parser.add_argument("--padding", type=int, default=1, help="Symmetric pad width.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size N (full grid batch).")
    parser.add_argument("--channels", type=int, default=8, help="Channel count C.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--device",
        default=None,
        help="cpu | cuda | cuda:0 (default: cuda if available else cpu).",
    )
    args = parser.parse_args()

    try:
        from earth2grid.healpix import PaddingBackends, pad, pad_backend
    except ImportError as e:
        raise SystemExit(
            "Could not import earth2grid.healpix. Install earth2grid or set PYTHONPATH "
            "to the earth2grid repo root.\n"
            f"Original error: {e}"
        ) from e

    H = args.nside
    p = args.padding
    N, C = args.batch, args.channels
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # [N, 12, C, H, W] — earth2grid pad layout
    x_5d = torch.randn(N, 12, C, H, H, dtype=torch.float32, device=device)
    x_folded = x_5d.reshape(N * 12, C, H, H)

    opt = HEALPixPaddingIsolatitude(
        padding=p,
        nside=H,
        enable_nhwc=False,
    ).to(device)

    def run_e2g_karlbauer_indexing() -> torch.Tensor:
        with pad_backend(PaddingBackends.indexing):
            return pad(x_5d, p, padding_mode="karlbauer")

    def run_modulus_isolatitude() -> torch.Tensor:
        return opt(x_folded)

    t_e2g = bench_callable(run_e2g_karlbauer_indexing, warmup=args.warmup, repeats=args.repeats, device=device)
    t_mod = bench_callable(run_modulus_isolatitude, warmup=args.warmup, repeats=args.repeats, device=device)

    print(f"device={device}  H={H}  padding={p}  N={N}  C={C}  warmup={args.warmup}  repeats={args.repeats}")
    print()
    print("earth2grid Karlbauer (PaddingBackends.indexing → pure_python.pad / pad_with_dim):")
    print(f"  mean_forward_s={t_e2g:.6f}")
    print()
    print("modulus-uw HEALPixPaddingIsolatitude (gather):")
    print(f"  mean_forward_s={t_mod:.6f}")
    print()
    if t_mod > 0:
        print(f"ratio e2g_indexing / modulus_isolat = {t_e2g / t_mod:.3f}x  (values < 1 => e2g faster)")


if __name__ == "__main__":
    main()
