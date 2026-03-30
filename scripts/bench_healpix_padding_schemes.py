#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark HEALPix padding implementations:

1. **earth2grid cuda backend** — ``earth2grid.healpix.pad`` with
   ``PaddingBackends.cuda`` (fused ``healpixpad_fprop``). Skipped on CPU.
2. **earth2grid indexing backend** — same API with ``PaddingBackends.indexing``
   (``pure_python.pad`` / ``pad_with_dim``).
3. **earth2grid zephyr backend** — same API with ``PaddingBackends.zephyr``
   (legacy Python implementation).
4. **modulus Karlbauer** — ``HEALPixPadding`` (explicit face stitching, Karlbauer et al.).
5. **modulus isolatitude_reference** — ``HEALPixPaddingIsolatitudeReference``
   (``isolatitude_pad_folded`` each forward).
6. **modulus isolatitude** — ``HEALPixPaddingIsolatitude`` (precomputed gather indices).

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
        "--compile",
        action="store_true",
        help="Enable torch.compile for modulus schemes (and optionally earth2grid).",
    )
    parser.add_argument(
        "--compile-earth2grid",
        action="store_true",
        help="Also compile earth2grid callables (default: false for cleaner apples-to-apples).",
    )
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
        nside=H,
        enable_nhwc=False,
    ).to(device)

    def _e2g_pad_karlbauer(x: torch.Tensor) -> torch.Tensor:
        # earth2grid API differs across versions:
        # - older: pad(x, padding, padding_mode="karlbauer")
        # - newer: pad(x, padding)
        try:
            return pad(x, p, padding_mode="karlbauer")
        except TypeError:
            return pad(x, p)

    def e2g_karlbauer_cuda() -> torch.Tensor:
        with pad_backend(PaddingBackends.cuda):
            return _e2g_pad_karlbauer(x_5d)

    def e2g_karlbauer_indexing() -> torch.Tensor:
        with pad_backend(PaddingBackends.indexing):
            return _e2g_pad_karlbauer(x_5d)

    def e2g_karlbauer_zephyr() -> torch.Tensor:
        with pad_backend(PaddingBackends.zephyr):
            return _e2g_pad_karlbauer(x_5d)

    rows: list[tuple[str, float | None, str]] = []

    def maybe_compile(
        fn: Callable[[], torch.Tensor],
        *,
        enable: bool,
        label: str,
    ) -> tuple[Callable[[], torch.Tensor], str]:
        if not enable:
            return fn, "eager"
        try:
            return torch.compile(fn, fullgraph=False), "compiled"
        except Exception as ex:  # noqa: BLE001
            return fn, f"compile_failed_fallback_eager: {ex}"

    # 1) earth2grid CUDA — only meaningful on CUDA device
    if device.type == "cuda":
        try:
            fn, note = maybe_compile(
                e2g_karlbauer_cuda,
                enable=args.compile and args.compile_earth2grid,
                label="earth2grid_cuda_backend",
            )
            t = bench_callable(
                fn,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            rows.append(("earth2grid_cuda_backend", t, note))
        except Exception as ex:  # noqa: BLE001 — surface extension load errors
            rows.append(("earth2grid_cuda_backend", None, f"error: {ex}"))
    else:
        rows.append(
            (
                "earth2grid_cuda_backend",
                None,
                "skipped (needs CUDA tensor + fused op)",
            )
        )

    # 2) earth2grid indexing backend
    fn_e2g_idx, note_e2g_idx = maybe_compile(
        e2g_karlbauer_indexing,
        enable=args.compile and args.compile_earth2grid,
        label="earth2grid_indexing_backend",
    )
    t_e2g_pt = bench_callable(
        fn_e2g_idx,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    rows.append(("earth2grid_indexing_backend", t_e2g_pt, note_e2g_idx))

    # 3) earth2grid zephyr backend
    try:
        fn_e2g_zephyr, note_e2g_zephyr = maybe_compile(
            e2g_karlbauer_zephyr,
            enable=args.compile and args.compile_earth2grid,
            label="earth2grid_zephyr_backend",
        )
        t_e2g_zephyr = bench_callable(
            fn_e2g_zephyr,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        rows.append(("earth2grid_zephyr_backend", t_e2g_zephyr, note_e2g_zephyr))
    except Exception as ex:  # noqa: BLE001
        rows.append(("earth2grid_zephyr_backend", None, f"error: {ex}"))

    # 4–6) modulus
    fn_mod_kb, note_mod_kb = maybe_compile(
        lambda: mod_karlbauer(x_folded),
        enable=args.compile,
        label="modulus_karlbauer",
    )
    t_mod_kb = bench_callable(
        fn_mod_kb,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    rows.append(("modulus_karlbauer", t_mod_kb, note_mod_kb))

    fn_mod_iref, note_mod_iref = maybe_compile(
        lambda: mod_iso_ref(x_folded),
        enable=args.compile,
        label="modulus_isolatitude_reference",
    )
    t_mod_iref = bench_callable(
        fn_mod_iref,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    rows.append(("modulus_isolatitude_reference", t_mod_iref, note_mod_iref))

    fn_mod_iso, note_mod_iso = maybe_compile(
        lambda: mod_iso(x_folded),
        enable=args.compile,
        label="modulus_isolatitude",
    )
    t_mod_iso = bench_callable(
        fn_mod_iso,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    rows.append(("modulus_isolatitude", t_mod_iso, note_mod_iso))

    # Print
    print(
        f"device={device}  H={H}  padding={p}  N={N}  C={C}  "
        f"warmup={args.warmup}  repeats={args.repeats}  "
        f"compile={args.compile}  compile_earth2grid={args.compile_earth2grid}"
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
        note_suffix = f"  [{note}]" if note else ""
        print(f"{label}  mean_forward_s={t:.6f}{rel}{note_suffix}")
    print()
    print(
        "rel_wall is wall-time ratio vs the first successful scheme in the list "
        "(higher = slower). On CPU, earth2grid_cuda_backend is skipped; "
        "use --device cuda for the fused kernel."
    )


if __name__ == "__main__":
    main()
