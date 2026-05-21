# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mini verification script: afPES with patch_size=1 reduces to WeightedCRPSLoss.

For each alpha in {0.0, 0.5, 0.95, 1.0} and both average_channels modes (True/False),
this script builds a small fake HEALPix-shaped tensor, runs both
`WeightedCRPSLoss` and `WeightedAlmostFairPatchEnergyScoreLoss(patch_size=1)`
with M=2 and no LSM, and asserts the two losses are numerically equal.

Run directly:

    python modulus-uw/test/metrics/verify_afpes_crps_equivalence.py
"""

from dataclasses import dataclass
from typing import Sequence

import torch as th

from physicsnemo.metrics.climate.healpix_loss import (
    WeightedAlmostFairPatchEnergyScoreLoss,
    WeightedCRPSLoss,
)


@dataclass
class _TrainerStub:
    output_variables: Sequence[str]
    device: str


def _build_inputs(
    n_members: int = 2,
    b: int = 2,
    f: int = 12,
    t: int = 1,
    c: int = 2,
    h: int = 8,
    w: int = 8,
    device: str = "cpu",
    seed: int = 0,
):
    th.manual_seed(seed)
    target = th.randn(b, f, t, c, h, w, device=device, dtype=th.float32)
    prediction = th.randn(n_members * b, f, t, c, h, w, device=device, dtype=th.float32)
    return prediction, target


def _build_losses(alpha: float, channels: int, device: str):
    weights = [1.0] * channels
    crps = WeightedCRPSLoss(weights=weights, n_members=2, alpha=alpha)
    afpes = WeightedAlmostFairPatchEnergyScoreLoss(
        weights=weights,
        n_members=2,
        alpha=alpha,
        patch_size=1,
        patch_stride=1,
    )
    trainer = _TrainerStub(output_variables=[f"v{i}" for i in range(channels)], device=device)
    crps.setup(trainer)
    afpes.setup(trainer)
    return crps, afpes


def _check_close(name: str, a: th.Tensor, b: th.Tensor, atol: float = 1e-6, rtol: float = 1e-6):
    a, b = a.detach(), b.detach()
    if not th.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        raise AssertionError(
            f"[FAIL] {name}\n"
            f"  crps  = {a}\n"
            f"  afpes = {b}\n"
            f"  max|diff| = {diff.max().item():.3e}\n"
            f"  mean|diff| = {diff.mean().item():.3e}"
        )
    diff_max = (a - b).abs().max().item() if a.ndim > 0 else abs(a.item() - b.item())
    print(f"  [OK]   {name:60s} max|diff| = {diff_max:.3e}")


def main():
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Running afPES <-> CRPS equivalence checks on device='{device}'.")

    alphas = [0.0, 0.5, 0.95, 1.0]
    channels = 2
    prediction, target = _build_inputs(c=channels, device=device)

    for alpha in alphas:
        print(f"\nalpha = {alpha}")
        crps, afpes = _build_losses(alpha=alpha, channels=channels, device=device)

        for avg in (True, False):
            crps_loss = crps(prediction.clone(), target.clone(), average_channels=avg)
            afpes_loss = afpes(prediction.clone(), target.clone(), average_channels=avg)
            _check_close(
                f"average_channels={avg}",
                crps_loss,
                afpes_loss,
                atol=1e-6,
                rtol=1e-6,
            )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
