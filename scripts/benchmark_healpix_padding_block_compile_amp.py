#!/usr/bin/env python3
import argparse
import contextlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch as th

from physicsnemo.models.dlwp_healpix_layers.healpix_paddings import (
    HEALPixPaddingIsolatitude,
)


class HEALPixPaddingIsolatitudeForceFP32(HEALPixPaddingIsolatitude):
    """Force internal padding compute in fp32, cast output back."""

    def forward(self, data: th.Tensor) -> th.Tensor:
        orig_dtype = data.dtype
        if data.dtype != th.float32:
            data = data.to(dtype=th.float32)
        out = super().forward(data)
        if out.dtype != orig_dtype:
            out = out.to(dtype=orig_dtype)
        return out


@dataclass
class BlockBenchConfig:
    batch: int
    in_channels: int
    hidden_channels: int
    out_channels: int
    nside: int
    padding: int
    warmup: int
    iters: int
    amp_dtype: str
    channels_last: bool
    lr: float
    out_dir: str


def _parse_amp_dtype(name: str) -> Optional[th.dtype]:
    mapping = {
        "none": None,
        "fp16": th.float16,
        "bf16": th.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported --amp-dtype {name!r}; use one of {list(mapping)}")
    return mapping[name]


def _autocast_ctx(amp_dtype: Optional[th.dtype]):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return th.autocast(device_type="cuda", dtype=amp_dtype)


class TinyBlock(th.nn.Module):
    """
    A minimal block to model real graph context:
    conv_in -> GELU -> HEALPix pad -> conv_out -> reduction loss.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        nside: int,
        pad_module: th.nn.Module,
    ):
        super().__init__()
        self.conv_in = th.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = th.nn.GELU()
        self.pad = th.compile(pad_module)
        # After external padding, keep size with internal padding=1.
        self.conv_out = th.nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.nside = nside

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv_in(x)
        x = self.act(x)
        x = self.pad(x)
        x = self.conv_out(x)
        return x


def _sync():
    th.cuda.synchronize()


def _make_batch(cfg: BlockBenchConfig, amp_dtype: Optional[th.dtype]) -> tuple[th.Tensor, th.Tensor]:
    bf = cfg.batch * 12
    input_dtype = th.float32 if amp_dtype is None else amp_dtype
    x = th.randn(
        bf,
        cfg.in_channels,
        cfg.nside,
        cfg.nside,
        device="cuda",
        dtype=input_dtype,
    )
    target = th.randn(
        bf,
        cfg.out_channels,
        cfg.nside + 2 * cfg.padding,
        cfg.nside + 2 * cfg.padding,
        device="cuda",
        dtype=input_dtype,
    )
    if cfg.channels_last:
        x = x.to(memory_format=th.channels_last)
        target = target.to(memory_format=th.channels_last)
    return x, target


def _benchmark_train_step(
    model: th.nn.Module,
    x: th.Tensor,
    target: th.Tensor,
    amp_dtype: Optional[th.dtype],
    warmup: int,
    iters: int,
    lr: float,
) -> dict:
    opt = th.optim.AdamW(model.parameters(), lr=lr)
    scaler = th.amp.GradScaler('cuda', enabled=(amp_dtype == th.float16))

    def one_step():
        opt.zero_grad(set_to_none=True)
        with _autocast_ctx(amp_dtype):
            y = model(x)
            loss = th.nn.functional.mse_loss(y, target)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    for _ in range(warmup):
        one_step()
    _sync()

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        one_step()
        _sync()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    t = th.tensor(times_ms, dtype=th.float64)
    return {
        "mean_ms": float(t.mean().item()),
        "median_ms": float(t.median().item()),
        "std_ms": float(t.std(unbiased=False).item()),
        "min_ms": float(t.min().item()),
        "max_ms": float(t.max().item()),
        "iters": iters,
    }


def _profile_train_step(
    tag: str,
    model: th.nn.Module,
    x: th.Tensor,
    target: th.Tensor,
    amp_dtype: Optional[th.dtype],
    out_dir: Path,
    lr: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = th.optim.AdamW(model.parameters(), lr=lr)
    scaler = th.amp.GradScaler('cuda', enabled=(amp_dtype == th.float16))

    def one_step():
        opt.zero_grad(set_to_none=True)
        with _autocast_ctx(amp_dtype):
            y = model(x)
            loss = th.nn.functional.mse_loss(y, target)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    with th.profiler.profile(
        activities=[th.profiler.ProfilerActivity.CPU, th.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(6):
            one_step()
        _sync()

    trace_path = out_dir / f"{tag}.json"
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    return trace_path, table


def _make_model(cfg: BlockBenchConfig, force_fp32_padding: bool, compiled: bool) -> th.nn.Module:
    pad_cls = HEALPixPaddingIsolatitudeForceFP32 if force_fp32_padding else HEALPixPaddingIsolatitude
    pad = pad_cls(cfg.padding, cfg.nside)
    model = TinyBlock(
        in_channels=cfg.in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=cfg.out_channels,
        nside=cfg.nside,
        pad_module=pad,
    ).cuda()
    if cfg.channels_last:
        model = model.to(memory_format=th.channels_last)
    model.train()
    if compiled:
        model = th.compile(model)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark conv->HEALPixPad->conv training step under compile+AMP."
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=192)
    parser.add_argument("--out-channels", type=int, default=128)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=str, default="/global/homes/y/yikwill/uw-research/healpix_padding_block_bench")
    args = parser.parse_args()

    if not th.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    cfg = BlockBenchConfig(
        batch=args.batch,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        nside=args.nside,
        padding=args.padding,
        warmup=args.warmup,
        iters=args.iters,
        amp_dtype=args.amp_dtype,
        channels_last=args.channels_last,
        lr=args.lr,
        out_dir=args.out_dir,
    )
    amp_dtype = _parse_amp_dtype(cfg.amp_dtype)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, target = _make_batch(cfg, amp_dtype)

    variants = {
        "eager_native": _make_model(cfg, force_fp32_padding=False, compiled=False),
        "eager_force_fp32": _make_model(cfg, force_fp32_padding=True, compiled=False),
        "compiled_native": _make_model(cfg, force_fp32_padding=False, compiled=True),
        "compiled_force_fp32": _make_model(cfg, force_fp32_padding=True, compiled=True),
    }

    results = {"config": asdict(cfg), "device": th.cuda.get_device_name(0), "variants": {}}
    for name, model in variants.items():
        stats = _benchmark_train_step(
            model=model,
            x=x,
            target=target,
            amp_dtype=amp_dtype,
            warmup=cfg.warmup,
            iters=cfg.iters,
            lr=cfg.lr,
        )
        trace_path, table = _profile_train_step(
            tag=name,
            model=model,
            x=x,
            target=target,
            amp_dtype=amp_dtype,
            out_dir=out_dir,
            lr=cfg.lr,
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


if __name__ == "__main__":
    main()
