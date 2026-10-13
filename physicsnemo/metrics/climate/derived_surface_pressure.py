"""Surface pressure and column water derived from pressure-level geopotential, temperature and humidity.

A model that predicts z, t, q on pressure levels but no surface pressure still has a well defined surface
pressure: it is the pressure at which the geopotential equals the surface geopotential (orography).
Finding it only needs the hydrostatic equation, so it is differentiable and carries no learned parameters.

Method (the standard vertical-interpolation / extrapolation assumptions of numerical weather prediction):

* Hydrostatic equation d(Phi) = -R_d T_v d(ln p), with the virtual temperature ``T_v = T (1 + 0.608 q)``.
* The surface lies between two pressure levels (over land, or in a deep low where the 1000 hPa level is below
  ground): take the lowest level above ground ``k`` and the level below it ``k-1`` (below ground, with the
  extrapolated values the data carry). ``T_v`` is linear in ``ln p`` between them, which gives the closed form
  ``s = ln(ps/p_k) = 2 D / (R (Tv_k + sqrt(Tv_k^2 + 2 beta D / R)))`` with ``D = Phi_k - Phi_s`` and
  ``beta = (Tv_{k-1} - Tv_k) / ln(p_{k-1}/p_k)``. It reduces to the isothermal solution as beta -> 0.
* The surface lies below the lowest level (open ocean, ``Phi_s < Phi_1000``): extrapolate down from the lowest
  level with the standard lapse rate, ``ps = p_1 (1 + Gamma D / (g Tv_1))^(g / (R Gamma))`` with
  ``Gamma = 6.5 K/km`` (the convention ECMWF and Trenberth et al. 1993 use below the lowest level).
* Column water is the pressure integral of q from the surface to the top level (trapezoid in pressure between
  levels, constant q between the surface and the lowest above-ground level), ``W = (1/g) int q dp``.

The functions work on physical (denormalized) values with the level axis at dim -3 (``[..., L, H, W]``), so the same
code serves the constraints (layout ``[B, F, T, L, H, W]``) and offline diagnostics.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.utils.checkpoint

G0 = 9.80665
R_DRY = 287.04
EPS_VIRTUAL = 0.608  # Rv / Rd - 1
LAPSE = 0.0065  # K / m


def surface_pressure_from_levels(
    z: torch.Tensor,
    t: torch.Tensor,
    q: torch.Tensor,
    phi_s: torch.Tensor,
    p_levels_pa: torch.Tensor,
    *,
    r_dry: float = R_DRY,
    g0: float = G0,
    lapse: float = LAPSE,
    eps_virtual: float = EPS_VIRTUAL,
    return_column_water: bool = False,
):
    """Surface pressure [Pa] (and optionally column water [kg m^-2]) from pressure-level fields.

    Parameters
    ----------
    z, t, q : torch.Tensor
        Geopotential [m^2 s^-2], temperature [K] and specific humidity [kg/kg], ``[..., L, H, W]``,
        with the level axis ordered from the highest pressure (lowest altitude) to the lowest pressure.
    phi_s : torch.Tensor
        Surface geopotential [m^2 s^-2], broadcastable to ``[..., 1, H, W]``.
    p_levels_pa : torch.Tensor
        Level pressures [Pa], shape ``[L]``, descending.
    """
    L = z.shape[-3]
    if L < 2:
        raise ValueError("need at least two pressure levels")
    z, t, q = z.float(), t.float(), q.float()
    phi_s = phi_s.float()
    p = p_levels_pa.to(z)
    ln_p = torch.log(p)
    tv = t * (1.0 + eps_virtual * q)

    # Number of levels below ground = index of the lowest above-ground level (levels are ordered bottom up).
    above = z > phi_s
    k = (~above).sum(dim=-3, keepdim=True).clamp(max=L - 1)            # [..., 1, H, W]
    km1 = (k - 1).clamp(min=0)
    take = lambda x, idx: torch.gather(x, -3, idx)
    z_k, tv_k = take(z, k), take(tv, k)
    tv_km1 = take(tv, km1)
    ln_pk, ln_pkm1 = ln_p[k], ln_p[km1]
    d = (z_k - phi_s).clamp(min=0.0)                                    # Phi_k - Phi_s

    # Surface between level k-1 (below ground) and level k: T_v linear in ln p.
    dlnp = (ln_pkm1 - ln_pk).clamp(min=1e-6)
    beta = (tv_km1 - tv_k) / dlnp
    root = torch.sqrt((tv_k * tv_k + 2.0 * beta * d / r_dry).clamp(min=1.0))
    s_between = 2.0 * d / (r_dry * (tv_k + root))

    # Surface below the lowest level (k == 0): standard-lapse-rate extrapolation.
    base = (1.0 + lapse * d / (g0 * tv_k)).clamp(min=1e-3)
    s_lapse = (g0 / (r_dry * lapse)) * torch.log(base)

    s = torch.where(k == 0, s_lapse, s_between)
    ps = torch.exp(ln_pk + s)                                           # [..., 1, H, W]
    ps_out = ps.squeeze(-3)
    if not return_column_water:
        return ps_out

    # Column water: q_k (ps - p_k) between the surface and level k, trapezoid between the levels above.
    p_k = p[k]
    w = q.new_zeros(ps.shape) + take(q, k) * (ps - p_k)
    lvl = torch.arange(L - 1, device=z.device).view(*([1] * (z.ndim - 3)), L - 1, 1, 1)
    dp = (p[:-1] - p[1:]).view(*([1] * (z.ndim - 3)), L - 1, 1, 1)
    layer = 0.5 * (q[..., :-1, :, :] + q[..., 1:, :, :]) * dp
    w = w + (layer * (lvl >= k).to(layer.dtype)).sum(dim=-3, keepdim=True)
    return ps_out, (w / g0).squeeze(-3)


class DerivedSurfacePressure(torch.nn.Module):
    """Derive surface pressure and column water from normalized model tensors (layout ``[B, F, T, C, H, W]``).

    Parameters
    ----------
    hPa_levels : sequence of float
        Pressure levels (hPa) of the ``z{p}``, ``t{p}``, ``q{p}`` channels.
    channels : sequence of str
        Channel names of the tensors that will be passed to :meth:`forward` / :meth:`sp_tcwv`.
    scaling : mapping
        ``{name: {"mean": ..., "std": ...}}`` for every ``z/t/q`` channel used.
    dataset_path, surface_geopotential_name, surface_geopotential_mean, surface_geopotential_std :
        Where the orography comes from, as in the hydrostasy and AAM constraints. ``dataset_path=None``
        means a flat surface (Phi_s = 0).
    """

    def __init__(
        self,
        hPa_levels: Sequence[float],
        channels: Sequence[str],
        scaling: Dict[str, Dict[str, float]],
        dataset_path: Optional[str] = None,
        surface_geopotential_name: str = "z",
        surface_geopotential_mean: float = 0.0,
        surface_geopotential_std: float = 1.0,
        g0: float = G0,
        r_dry: float = R_DRY,
        lapse: float = LAPSE,
    ):
        super().__init__()
        levels = sorted((float(p) for p in hPa_levels), reverse=True)
        names = list(channels)

        def tok(p: float) -> str:
            return str(int(p)) if float(p).is_integer() else str(p)

        # LongTensor buffers so that advanced indexing is CUDA-graph safe (a Python list index would copy host -> device).
        for prefix in "ztq":
            idx = torch.tensor([names.index(f"{prefix}{tok(p)}") for p in levels], dtype=torch.long)
            self.register_buffer(f"{prefix}_idx", idx, persistent=False)
        self.g0, self.r_dry, self.lapse = float(g0), float(r_dry), float(lapse)

        def stat(prefix: str, key: str) -> torch.Tensor:
            return torch.tensor(
                [scaling[f"{prefix}{tok(p)}"][key] for p in levels], dtype=torch.float32
            ).view(1, 1, 1, -1, 1, 1)

        for prefix in "ztq":
            self.register_buffer(f"{prefix}_mean", stat(prefix, "mean"), persistent=False)
            self.register_buffer(f"{prefix}_std", stat(prefix, "std"), persistent=False)
        self.register_buffer(
            "p_levels_pa", torch.tensor([p * 100.0 for p in levels], dtype=torch.float32), persistent=False
        )
        if dataset_path is None:
            phi_s = torch.zeros(1, 1, 1, 1, 1, 1)
        else:
            from physicsnemo.metrics.climate.hydrostasy import _load_topography

            # Metres from the loader, converted back so the geopotential is exact in m^2 s^-2.
            topo_m = _load_topography(
                dataset_path,
                surface_geopotential_name,
                surface_geopotential_mean,
                surface_geopotential_std,
                self.g0,
                True,
            )                                                           # [1, F, 1, H, W]
            phi_s = (topo_m * self.g0).unsqueeze(3)                     # [1, F, 1, 1, H, W]
        self.register_buffer("phi_s", phi_s, persistent=False)

    def _levels(self, tensor: torch.Tensor):
        z = tensor[:, :, :, self.z_idx].float() * self.z_std + self.z_mean
        t = tensor[:, :, :, self.t_idx].float() * self.t_std + self.t_mean
        q = tensor[:, :, :, self.q_idx].float() * self.q_std + self.q_mean
        return z, t, q

    def _compute(self, tensor: torch.Tensor, column_water: bool):
        z, t, q = self._levels(tensor)
        return surface_pressure_from_levels(
            z, t, q, self.phi_s, self.p_levels_pa, r_dry=self.r_dry, g0=self.g0,
            lapse=self.lapse, return_column_water=column_water,
        )

    def _run(self, tensor: torch.Tensor, column_water: bool):
        # The derivation touches several [B, F, T, L, H, W] intermediates. It runs one time step at a time, and when a
        # gradient is needed each step is recomputed in the backward pass instead of being kept, so the training graph
        # holds roughly 1/T of the intermediates.
        outs = []
        for i in range(tensor.shape[2]):
            step = tensor[:, :, i : i + 1]
            if torch.is_grad_enabled() and tensor.requires_grad:
                outs.append(torch.utils.checkpoint.checkpoint(self._compute, step, column_water, use_reentrant=False))
            else:
                outs.append(self._compute(step, column_water))
        if column_water:
            return torch.cat([o[0] for o in outs], dim=2), torch.cat([o[1] for o in outs], dim=2)
        return torch.cat(outs, dim=2)

    def sp_tcwv(self, tensor: torch.Tensor):
        """``(sp [Pa], tcwv [kg m^-2])`` each ``[B, F, T, 1, H, W]``."""
        sp, w = self._run(tensor, True)
        return sp.unsqueeze(3), w.unsqueeze(3)

    def sp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Surface pressure [Pa], ``[B, F, T, 1, H, W]``."""
        return self._run(tensor, False).unsqueeze(3)

    def sp_dry(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dry surface pressure ``sp - g W`` [Pa], ``[B, F, T, 1, H, W]``."""
        sp, w = self.sp_tcwv(tensor)
        return sp - self.g0 * w
