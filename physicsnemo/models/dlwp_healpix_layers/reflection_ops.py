# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Z₂ equatorial-reflection helpers for HEALPix ReflectionSteerable layers.

Terminology
-----------
* **Parity (even/odd)** — how a channel (or kernel) transforms under reflection
  ρ: even → +R(·), odd → −R(·). Kept in names like ``odd_fraction``, bank split.
* **ReflectionSteerable** — layers whose weights live in the even/odd subspace so
  each layer intertwines with ρ (hard equivariance in one forward).
* **structural** / **averaged** — values of ``reflection_equivariance_mode``:
  structural = ReflectionSteerable layers; averaged = twin-forward Reynolds
  projector Π(F)(x)=½(F(x)+ρ F(ρx)).

The in-face operator ``R`` matches ``HEALPixRecUNet.hpx_reflect``:
``rot90(flip(·, W), dims=(W, H))``, followed by the face reorder
``[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]`` when acting on full HPX tensors.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch as th
from earth2grid.healpix import coordinates as hpx_coordinates
from omegaconf import DictConfig, OmegaConf

# Face permutation used by HEALPixRecUNet.hpx_reflect (N/S swap with equatorial belt fixed).
REFL_FACE_ORDER = (8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3)


def bank_sizes(n_channels: int, odd_fraction: float) -> Tuple[int, int]:
    """Split ``n_channels`` into (n_even, n_odd) with ``n_odd = round(n * odd_fraction)``."""
    if n_channels < 0:
        raise ValueError(f"n_channels must be >= 0, got {n_channels}")
    if not 0.0 <= odd_fraction <= 1.0:
        raise ValueError(f"odd_fraction must be in [0, 1], got {odd_fraction}")
    n_odd = int(round(n_channels * odd_fraction))
    n_odd = min(max(n_odd, 0), n_channels)
    return n_channels - n_odd, n_odd


def strip_nested_odd_fraction(config: DictConfig, odd_fraction: float, *, label: str) -> DictConfig:
    """Return a config copy without per-block ``odd_fraction``.

    ``odd_fraction`` is owned by ``HEALPixRecUNet`` and passed into encoder/decoder.
    If a nested block still sets ``odd_fraction`` and it disagrees with the UNet
    value, raise so misconfiguration is caught early.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    if OmegaConf.select(cfg, "odd_fraction", default=None) is not None:
        nested = float(cfg.odd_fraction)
        if abs(nested - odd_fraction) > 1e-9:
            raise ValueError(
                f"{label} sets odd_fraction={nested}, but the UNet uses odd_fraction={odd_fraction}. "
                "Set odd_fraction only on HEALPixRecUNet (not on individual blocks)."
            )
        cfg = cfg.copy()
        del cfg["odd_fraction"]
    return cfg


def split_banks(x: th.Tensor, n_even: int) -> Tuple[th.Tensor, th.Tensor]:
    """Split channel dim into even then odd banks. ``x`` is ``[..., C, H, W]`` or ``[N, C, H, W]``."""
    return x[:, :n_even], x[:, n_even:]


def merge_banks(x_even: th.Tensor, x_odd: th.Tensor) -> th.Tensor:
    """Concatenate even and odd banks along channels."""
    if x_even.numel() == 0:
        return x_odd
    if x_odd.numel() == 0:
        return x_even
    return th.cat([x_even, x_odd], dim=1)


def apply_R_spatial(x: th.Tensor) -> th.Tensor:
    """In-face reflection R on a tensor with spatial dims ``(..., H, W)``."""
    return th.rot90(th.flip(x, dims=[-1]), dims=(-1, -2))


def apply_R_kernel(weight: th.Tensor) -> th.Tensor:
    """Apply the same spatial R to a conv kernel ``[O, I, kH, kW]``."""
    return apply_R_spatial(weight)


def project_even_kernel(weight: th.Tensor) -> th.Tensor:
    """Symmetrize kernel so ``K = R(K)``."""
    return 0.5 * (weight + apply_R_kernel(weight))


def project_odd_kernel(weight: th.Tensor) -> th.Tensor:
    """Antisymmetrize kernel so ``K = -R(K)``."""
    return 0.5 * (weight - apply_R_kernel(weight))


def hpx_spatial_reflect(x: th.Tensor, face_order: Optional[th.Tensor] = None) -> th.Tensor:
    """Spatial HEALPix reflection on folded ``[B*12, C, H, W]`` (no channel sign flips).

    ``face_order`` should be a registered buffer on the calling module when training
    with CUDA graphs; allocating indices inside ``forward`` breaks graph capture.
    """
    if face_order is None:
        face_order = th.tensor(REFL_FACE_ORDER, device=x.device, dtype=th.long)
    elif face_order.device != x.device or face_order.dtype != th.long:
        face_order = face_order.to(device=x.device, dtype=th.long)
    y = apply_R_spatial(x)
    y = y.reshape(-1, 12, *y.shape[1:])
    y = th.index_select(y, dim=1, index=face_order)
    return y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])


def hpx_reflect_typed(
    x: th.Tensor,
    n_even: int,
    face_order: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Full ρ on a typed even|odd folded tensor: spatial reflect + sign flip on odd bank."""
    y = hpx_spatial_reflect(x, face_order=face_order)
    if n_even < y.shape[1]:
        y = y.clone()
        y[:, n_even:] = -y[:, n_even:]
    return y


def compute_sin_lat_faces(nside: int, device=None, dtype=th.float32) -> th.Tensor:
    """Return ``sin(lat)`` on HEALPix faces as ``[12, 1, nside, nside]`` (PAD_XY layout)."""
    x = (th.arange(nside, dtype=th.float32) + 0.5) / nside
    y = (th.arange(nside, dtype=th.float32) + 0.5) / nside
    xx, yy = th.meshgrid(x, y, indexing="ij")
    faces = []
    for f in range(12):
        ff = th.full_like(xx, f, dtype=th.long)
        xs, ys = hpx_coordinates.face_to_global(xx, yy, ff)
        _lon, lat_deg = hpx_coordinates.global_to_angular(xs, ys)
        faces.append(th.sin(th.deg2rad(lat_deg)))
    sin_lat = th.stack(faces, dim=0).unsqueeze(1)  # [12, 1, H, W]
    return sin_lat.to(device=device, dtype=dtype)


def expand_sin_lat_folded(sin_lat_faces: th.Tensor, batch_faces: int) -> th.Tensor:
    """Tile ``[12, 1, H, W]`` to folded ``[B*12, 1, H, W]`` matching ``batch_faces = B*12``."""
    if batch_faces % 12 != 0:
        raise ValueError(f"batch_faces must be divisible by 12, got {batch_faces}")
    b = batch_faces // 12
    return sin_lat_faces.repeat(b, 1, 1, 1)


def resolve_reflection_equivariance_mode(
    reflection_equivariance_mode: Optional[str],
    enforce_reflectional_equivariance: bool,
) -> str:
    """Map config knobs to ``off`` | ``averaged`` | ``structural``.

    Modes
    -----
    * ``off`` — no hard reflection equivariance.
    * ``averaged`` — Reynolds / twin-forward projector on outputs and GRU state
      (two forwards per step). Legacy alias: ``reynolds``.
    * ``structural`` — ReflectionSteerable layers (one forward). Legacy alias:
      ``steerable``.

    Rules
    -----
    - Explicit mode wins; conflict with legacy ``enforce_*=True`` raises.
    - Omitted / ``off`` + ``enforce_*=True`` → ``averaged``.
    - Default → ``off``.
    """
    # Legacy aliases kept so older configs/checkpoints keep working.
    _ALIASES = {"reynolds": "averaged", "steerable": "structural"}
    mode = reflection_equivariance_mode
    if mode is not None:
        mode = _ALIASES.get(mode, mode)
    if mode is None or mode == "off":
        if enforce_reflectional_equivariance:
            return "averaged"
        return "off"
    if mode not in ("off", "averaged", "structural"):
        raise ValueError(
            f"reflection_equivariance_mode must be off|averaged|structural "
            f"(legacy aliases: reynolds|steerable), got {mode!r}"
        )
    if enforce_reflectional_equivariance and mode != "averaged":
        raise ValueError(
            f"Conflicting settings: reflection_equivariance_mode={mode!r} but "
            f"enforce_reflectional_equivariance=True (legacy averaged/reynolds). "
            f"Set enforce_reflectional_equivariance=False when using mode={mode!r}."
        )
    return mode


def assert_close_equivariant(
    y: th.Tensor,
    y_from_reflected: th.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Assert ``y`` matches the ρ-transformed prediction from reflected inputs."""
    th.testing.assert_close(y, y_from_reflected, rtol=rtol, atol=atol)


def indices_for_names(all_names: Sequence[str], odd_names: Optional[Sequence[str]]) -> th.Tensor:
    """Long tensor of indices into ``all_names`` that appear in ``odd_names``."""
    if not odd_names:
        return th.tensor([], dtype=th.long)
    missing = [n for n in odd_names if n not in all_names]
    if missing:
        raise ValueError(f"Odd variable(s) {missing} not found in {list(all_names)}")
    return th.tensor([all_names.index(n) for n in odd_names], dtype=th.long)


def reorder_channels_even_odd(x: th.Tensor, odd_idx: th.Tensor) -> Tuple[th.Tensor, th.Tensor, int]:
    """Reorder ``[N, C, H, W]`` so even channels come first, then odd.

    Returns
    -------
    reordered, inverse_index, n_even
        ``inverse_index`` scatters reordered channels back to the original order.
    """
    c = x.shape[1]
    device = x.device
    odd_idx = odd_idx.to(device=device, dtype=th.long)
    odd_set = set(odd_idx.tolist())
    even_idx = th.tensor([i for i in range(c) if i not in odd_set], device=device, dtype=th.long)
    order = th.cat([even_idx, odd_idx]) if odd_idx.numel() else even_idx
    inverse = th.empty(c, device=device, dtype=th.long)
    inverse[order] = th.arange(c, device=device, dtype=th.long)
    return x.index_select(1, order), inverse, int(even_idx.numel())


def apply_channel_order(x: th.Tensor, order: th.Tensor) -> th.Tensor:
    return x.index_select(1, order.to(x.device))
