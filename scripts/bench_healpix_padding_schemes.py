#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark HEALPix padding implementations:

1. **earth2grid Karlbauer CUDA** — ``earth2grid.healpix.pad`` with
   ``PaddingBackends.cuda`` (fused ``healpixpad_fprop``). Skipped on CPU (earth2grid
   falls back to indexing on non-CUDA tensors).
2. **earth2grid Karlbauer PyTorch** — same API with ``PaddingBackends.indexing``
   (``pure_python.pad`` / ``pad_with_dim``).
3. **modulus Karlbauer** — ``HEALPixPadding`` (explicit face stitching, Karlbauer et al.).
4. **modulus isolatitude_reference** — ``HEALPixPaddingIsolatitudeReference``
   (``isolatitude_pad_folded`` each forward).
5. **modulus isolatitude** — ``HEALPixPaddingIsolatitude`` (precomputed gather indices).

earth2grid ``pad`` expects ``[N, 12, C, H, W]``; modulus modules expect folded
``[N * 12, C, H, W]``.

Run::

    cd /path/to/modulus-uw
    PYTHONPATH=.:../earth2grid python scripts/bench_healpix_padding_schemes.py

Use ``--device cuda`` for the fused earth2grid CUDA path.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
    HEALPixPadding,
    HEALPixPaddingIsolatitude,
    HEALPixPaddingIsolatitudeReference,
)


def bench_callable(
    fn: Callable[[], torch.Tensor],
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
    parser.add_argument("--nside", type=int, default=64, help="Face height/width H.")
    parser.add_argument("--padding", type=int, default=1, help="Symmetric pad width.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size N.")
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
            "Could not import earth2grid.healpix. Set PYTHONPATH to the earth2grid repo.\n"
            f"Original error: {e}"
        ) from e

    H = args.nside
    p = args.padding
    N, C = args.batch, args.channels
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    x_5d = torch.randn(N, 12, C, H, H, dtype=torch.float32, device=device)
    x_folded = x_5d.reshape(N * 12, C, H, H)

    mod_karlbauer = HEALPixPadding(padding=p, enable_nhwc=False).to(device)
    mod_iso_ref = HEALPixPaddingIsolatitudeReference(padding=p, enable_nhwc=False).to(
        device
    )
    mod_iso = HEALPixPaddingIsolatitude(
        padding=p,
        healpix_face_size=H,
        enable_nhwc=False,
    ).to(device)

    def e2g_karlbauer_cuda() -> torch.Tensor:
        with pad_backend(PaddingBackends.cuda):
            return pad(x_5d, p, padding_mode="karlbauer")

    def e2g_karlbauer_pytorch() -> torch.Tensor:
        with pad_backend(PaddingBackends.indexing):
            return pad(x_5d, p, padding_mode="karlbauer")

    rows: list[tuple[str, float | None, str]] = []

    # 1) earth2grid CUDA — only meaningful on CUDA device
    if device.type == "cuda":
        try:
            t = bench_callable(
                e2g_karlbauer_cuda,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            rows.append(("earth2grid_karlbauer_cuda", t, ""))
        except Exception as ex:  # noqa: BLE001 — surface extension load errors
            rows.append(("earth2grid_karlbauer_cuda", None, f"error: {ex}"))
    else:
        rows.append(
            (
                "earth2grid_karlbauer_cuda",
                None,
                "skipped (needs CUDA tensor + fused op)",
            )
        )

    # 2) earth2grid PyTorch / indexing backend
    t_e2g_pt = bench_callable(
        e2g_karlbauer_pytorch,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    rows.append(("earth2grid_karlbauer_pytorch", t_e2g_pt, ""))

    # 3–5) modulus
    t_mod_kb = bench_callable(
        lambda: mod_karlbauer(x_folded),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    rows.append(("modulus_karlbauer", t_mod_kb, ""))

    t_mod_iref = bench_callable(
        lambda: mod_iso_ref(x_folded),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    rows.append(("modulus_isolatitude_reference", t_mod_iref, ""))

    t_mod_iso = bench_callable(
        lambda: mod_iso(x_folded),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    rows.append(("modulus_isolatitude", t_mod_iso, ""))

    # Print
    print(
        f"device={device}  H={H}  padding={p}  N={N}  C={C}  "
        f"warmup={args.warmup}  repeats={args.repeats}"
    )
    print()
    w = max(len(r[0]) for r in rows)
    baseline_time: float | None = None
    baseline_name: str | None = None
    for name, t, note in rows:
        label = name.ljust(w)
        if t is None:
            print(f"{label}  —  {note}")
            continue
        if baseline_time is None:
            baseline_time = t
            baseline_name = name
            rel = "  rel_wall=1.000× (baseline)"
        else:
            assert baseline_time is not None
            rel = f"  rel_wall={t / baseline_time:.3f}× (vs {baseline_name})"
        print(f"{label}  mean_forward_s={t:.6f}{rel}")
    print()
    print(
        "rel_wall is wall-time ratio vs the first successful scheme in the list "
        "(higher = slower). On CPU, earth2grid_karlbauer_cuda is skipped; "
        "use --device cuda for the fused kernel."
    )


if __name__ == "__main__":
    main()
