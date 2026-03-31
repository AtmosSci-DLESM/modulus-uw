#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch as th

from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
    HEALPixPaddingIsolatitude,
)


@dataclass
class BenchConfig:
    batch: int
    channels: int
    nside: int
    padding: int
    warmup: int
    iters: int
    use_channels_last: bool
    amp_dtype: str
    out_dir: str


class HEALPixPaddingIsolatitudeForceFP32(HEALPixPaddingIsolatitude):
    """Force internal compute in fp32, cast output back to input dtype."""

    def forward(self, data: th.Tensor) -> th.Tensor:
        orig_dtype = data.dtype
        if data.dtype != th.float32:
            data = data.to(dtype=th.float32)
        out = super().forward(data)
        if out.dtype != orig_dtype:
            out = out.to(dtype=orig_dtype)
        return out


def _autocast_ctx(device: str, amp_dtype: Optional[th.dtype]):
    if amp_dtype is None:
        return contextlib.nullcontext()
    if device == "cuda":
        return th.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def _parse_amp_dtype(name: str) -> Optional[th.dtype]:
    mapping = {
        "none": None,
        "fp16": th.float16,
        "bf16": th.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported --amp-dtype {name!r}; use one of {list(mapping)}")
    return mapping[name]


def _make_input(cfg: BenchConfig, device: str, amp_dtype: Optional[th.dtype]) -> th.Tensor:
    bf = cfg.batch * 12
    dtype = th.float32 if amp_dtype is None else amp_dtype
    x = th.randn(bf, cfg.channels, cfg.nside, cfg.nside, device=device, dtype=dtype)
    if cfg.use_channels_last:
        x = x.to(memory_format=th.channels_last)
    return x


def _synchronize(device: str):
    if device == "cuda":
        th.cuda.synchronize()


def _time_module(
    module: th.nn.Module,
    x: th.Tensor,
    device: str,
    amp_dtype: Optional[th.dtype],
    warmup: int,
    iters: int,
) -> dict:
    for _ in range(warmup):
        with _autocast_ctx(device, amp_dtype):
            _ = module(x)
    _synchronize(device)

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with _autocast_ctx(device, amp_dtype):
            _ = module(x)
        _synchronize(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    t = th.tensor(times_ms, dtype=th.float64)
    return {
        "mean_ms": float(t.mean().item()),
        "median_ms": float(t.median().item()),
        "std_ms": float(t.std(unbiased=False).item()),
        "min_ms": float(t.min().item()),
        "max_ms": float(t.max().item()),
        "iters": iters,
    }


def _profile_module(
    tag: str,
    module: th.nn.Module,
    x: th.Tensor,
    device: str,
    amp_dtype: Optional[th.dtype],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    activities = [th.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(th.profiler.ProfilerActivity.CUDA)

    with th.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(8):
            with _autocast_ctx(device, amp_dtype):
                _ = module(x)
        _synchronize(device)

    trace_path = out_dir / f"{tag}.json"
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    return trace_path, table


def main():
    parser = argparse.ArgumentParser(description="Benchmark HEALPixPaddingIsolatitude compile+AMP behavior.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/healpix_padding_bench")
    args = parser.parse_args()

    if not th.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = "cuda"
    amp_dtype = _parse_amp_dtype(args.amp_dtype)
    cfg = BenchConfig(
        batch=args.batch,
        channels=args.channels,
        nside=args.nside,
        padding=args.padding,
        warmup=args.warmup,
        iters=args.iters,
        use_channels_last=args.channels_last,
        amp_dtype=args.amp_dtype,
        out_dir=args.out_dir,
    )
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = _make_input(cfg, device, amp_dtype)

    variants = {
        "eager_native": HEALPixPaddingIsolatitude(cfg.padding, cfg.nside).to(device).eval(),
        "eager_force_fp32": HEALPixPaddingIsolatitudeForceFP32(cfg.padding, cfg.nside).to(device).eval(),
        "compiled_native": th.compile(
            HEALPixPaddingIsolatitude(cfg.padding, cfg.nside).to(device).eval(),
            fullgraph=True,
            dynamic=False,
        ),
        "compiled_force_fp32": th.compile(
            HEALPixPaddingIsolatitudeForceFP32(cfg.padding, cfg.nside).to(device).eval(),
            fullgraph=True,
            dynamic=False,
        ),
    }

    results = {"config": asdict(cfg), "device": th.cuda.get_device_name(0), "variants": {}}

    for name, mod in variants.items():
        stats = _time_module(
            module=mod,
            x=x,
            device=device,
            amp_dtype=amp_dtype,
            warmup=cfg.warmup,
            iters=cfg.iters,
        )
        trace_path, table = _profile_module(
            tag=name,
            module=mod,
            x=x,
            device=device,
            amp_dtype=amp_dtype,
            out_dir=out_dir,
        )
        results["variants"][name] = {
            **stats,
            "trace_path": str(trace_path),
            "profiler_table_top20": table,
        }
        print(
            f"{name:20s} mean={stats['mean_ms']:.3f} ms  "
            f"median={stats['median_ms']:.3f} ms  std={stats['std_ms']:.3f} ms"
        )

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to: {results_path}")
    print(f"Saved Chrome traces to: {out_dir}")
    print("Use chrome://tracing to inspect the JSON traces.")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    main()
