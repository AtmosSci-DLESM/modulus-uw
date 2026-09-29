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

"""Adaptive per-term loss weights with an EMA target equalizer."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.distributed as dist


def reduce_per_term_loss(
    terms: torch.Tensor, n_data_variables: int
) -> torch.Tensor:
    """Training scalar: ``sum(per-term losses) / n_data_variables``.

    Copied here so this module stays independent of ``wy/soft_constraints``.
    Dividing by the data-channel count ``C`` keeps the data contribution equal
    to a uniform-weight mean; each constraint term contributes ``1/C`` of its
    weighted value.
    """
    if n_data_variables < 1:
        raise ValueError(
            f"n_data_variables must be >= 1, got {n_data_variables}"
        )
    return terms.sum() / float(n_data_variables)


def _as_float_list(scales: Sequence[float]) -> List[float]:
    return [float(s) for s in scales]


class AdaptiveLossWeights(torch.nn.Module):
    """
    Wrap a training criterion with detached adaptive per-term weights.

    The inner module is always called with ``average_channels=False`` so this
    wrapper sees one unweighted vector
    ``[data variables | constraint terms in group order]``. Weights are applied
    here and reduced with :func:`reduce_per_term_loss`.

    Parameters
    ----------
    inner:
        Criterion returning a per-term vector when ``average_channels=False``
        (e.g. ``WeightedMSE`` or ``LossWithSoftConstraints``).
    n_data_variables:
        Number of leading terms that are data variables (``C``). Constraint
        terms follow in ``constraint_groups`` order. Omit this and ``setup``
        sets it from ``len(trainer.output_variables)`` so the count stays
        aligned with the data layout. A value that disagrees with
        ``output_variables`` raises in ``setup``.
    constraint_groups:
        Ordered groups so Hydra can list relative scales. Each entry is a
        mapping ``{"name": str, "scales": Sequence[float]}``. Example::

            [
              {"name": "hydro", "scales": [0.001]*5 + [0.01]*4},
              {"name": "dry_air", "scales": [0.001]},
              {"name": "aam", "scales": [0.001]},
            ]

        Variable terms have implicit relative scale 1. During warmup, applied
        weights are 1 on variables and ``s_k`` on constraints. After warmup,
        ``w_i = T / m_i`` and ``w_k = s_k * T / m_k`` with ``T = mean(m)`` over
        the data-variable EMA.
    variable_names:
        Optional names for data terms (logging). Defaults to
        ``var_0 … var_{C-1}``, or ``trainer.output_variables`` in ``setup``.
    warmup_epochs:
        Epochs with fixed warmup weights while the EMA still tracks every train
        step. Adaptive weights apply once ``epoch >= warmup_epochs``. Default 1.
    ema_window_epochs:
        E-folding window in epochs for the per-step EMA
        ``m ← β m + (1-β) L`` with
        ``β = 1 - 1/(ema_window_epochs * steps_per_epoch)``. Default 5.
    steps_per_epoch:
        Optional override; otherwise taken from ``len(trainer.dataloader_train)``
        in ``setup``.
    eps:
        Floor for EMA magnitudes when forming ``T / m``.

    Trainer contract
    ----------------
    * Forward stores detached unweighted terms and logging tensors; it does
      **not** update the EMA (safe under CUDA-graph capture of the forward).
    * After backward on a **training** step, call
      :meth:`post_backward_update` with the current epoch index. That all-reduces
      the unweighted vector, updates the EMA, and ``copy_``s new weights into
      the buffer the graph already captured.
    * Do **not** call :meth:`post_backward_update` during eval/validation.
    """

    def __init__(
        self,
        inner: torch.nn.Module,
        n_data_variables: Optional[int] = None,
        constraint_groups: Optional[
            Sequence[Union[Mapping[str, Any], Dict[str, Any]]]
        ] = None,
        variable_names: Optional[Sequence[str]] = None,
        warmup_epochs: int = 1,
        ema_window_epochs: float = 5.0,
        steps_per_epoch: Optional[int] = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        if n_data_variables is not None and int(n_data_variables) < 1:
            raise ValueError(
                f"n_data_variables must be >= 1, got {n_data_variables}"
            )
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if ema_window_epochs <= 0:
            raise ValueError(
                f"ema_window_epochs must be > 0, got {ema_window_epochs}"
            )

        self.inner = inner
        self._n_data_arg = (
            None if n_data_variables is None else int(n_data_variables)
        )
        self._constraint_groups_arg = (
            list(constraint_groups) if constraint_groups is not None else []
        )
        self._variable_names_arg = (
            None if variable_names is None else list(variable_names)
        )
        self.warmup_epochs = int(warmup_epochs)
        self.ema_window_epochs = float(ema_window_epochs)
        self.eps = float(eps)
        self.steps_per_epoch = (
            int(steps_per_epoch) if steps_per_epoch is not None else None
        )
        self._built = False
        # Placeholders until ``_build`` so attribute access before setup fails
        # clearly rather than with a missing-buffer error mid-forward.
        self.n_data_variables = 0
        self.n_terms = 0
        self.variable_names = []
        self.term_names = []
        self.constraint_group_names = []
        self._group_slices = []

        # Propagate soft-constraint input flags for trainer gating.
        self.needs_input = bool(getattr(inner, "needs_input", False))
        self.needs_input_diagnostics = bool(
            getattr(inner, "needs_input_diagnostics", False)
        )

        # Views into ``_log_values``, bound in ``_build`` / ``setup``. Forward
        # only ``copy_``s into that storage. CUDA-graph replay does not re-run
        # Python assignments, so a fresh dict or tensor each forward would be
        # orphaned by the later eval-graph capture and the trainer would log
        # the capture-time values for the rest of the run.
        self.log_buffers: Dict[str, torch.Tensor] = {}
        self._log_tags: List[str] = []

        if self._n_data_arg is not None:
            self._build(self._n_data_arg)

    def _build(self, n_data_variables: int) -> None:
        """Allocate EMA/weight buffers for ``C = n_data_variables`` data terms."""
        if n_data_variables < 1:
            raise ValueError(
                f"n_data_variables must be >= 1, got {n_data_variables}"
            )
        self.n_data_variables = int(n_data_variables)
        self.constraint_group_names = []
        scale_list: List[float] = [1.0] * self.n_data_variables
        self._group_slices = []
        cursor = self.n_data_variables
        for group in self._constraint_groups_arg:
            name = str(group["name"])
            scales = _as_float_list(group["scales"])
            if not scales:
                raise ValueError(f"constraint group {name!r} has empty scales")
            self.constraint_group_names.append(name)
            self._group_slices.append((name, slice(cursor, cursor + len(scales))))
            scale_list.extend(scales)
            cursor += len(scales)

        self.n_terms = len(scale_list)
        names = self._variable_names_arg
        if names is None:
            self.variable_names = [f"var_{i}" for i in range(self.n_data_variables)]
        else:
            if len(names) != self.n_data_variables:
                raise ValueError(
                    f"variable_names length {len(names)} != "
                    f"n_data_variables {self.n_data_variables}"
                )
            self.variable_names = list(names)

        self._rebuild_term_names()
        self.register_buffer(
            "relative_scales",
            torch.tensor(scale_list, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "ema",
            torch.ones(self.n_terms, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "weights",
            self.relative_scales.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "ema_initialized",
            torch.tensor(0, dtype=torch.int32),
            persistent=True,
        )
        # Not checkpointed: replay writes these every forward.
        self.register_buffer(
            "_pending_unweighted",
            torch.zeros(self.n_terms, dtype=torch.float32),
            persistent=False,
        )
        n_logs = len(self._log_tag_list())
        self.register_buffer(
            "_log_values",
            torch.zeros(n_logs, dtype=torch.float32),
            persistent=False,
        )
        self._bind_log_buffers()
        self._built = True

    def _log_tag_list(self) -> List[str]:
        """TB tag order. Must match ``_fill_log_buffers`` exactly."""
        tags = ["loss", "loss_data"]
        for name, sl in self._group_slices:
            tags.append(f"loss_constraint/{name}")
            width = sl.stop - sl.start
            if width > 1:
                tags.extend(f"loss_constraint/{name}/{i}" for i in range(width))
        for term_name in self.term_names:
            tags.append(f"loss_weight/{term_name}")
            tags.append(f"loss_weighted/{term_name}")
            tags.append(f"loss_unweighted/{term_name}")
        return tags

    def _bind_log_buffers(self) -> None:
        tags = self._log_tag_list()
        if self._log_values.numel() != len(tags):
            raise RuntimeError(
                f"AdaptiveLossWeights log buffer has {self._log_values.numel()} "
                f"slots but {len(tags)} tags."
            )
        self._log_tags = tags
        # Views of the registered buffer, so a later ``copy_`` is what replay
        # updates and what the trainer reads.
        self.log_buffers = {
            tag: self._log_values[i] for i, tag in enumerate(tags)
        }

    def _rebuild_term_names(self) -> None:
        self.term_names = list(self.variable_names)
        for name, sl in self._group_slices:
            n = sl.stop - sl.start
            if n == 1:
                self.term_names.append(name)
            else:
                self.term_names.extend(f"{name}/{i}" for i in range(n))

    def setup(self, trainer) -> None:
        if hasattr(self.inner, "setup"):
            self.inner.setup(trainer)
        if self.steps_per_epoch is None:
            loader = getattr(trainer, "dataloader_train", None)
            if loader is None:
                raise ValueError(
                    "AdaptiveLossWeights needs steps_per_epoch or "
                    "trainer.dataloader_train in setup()."
                )
            self.steps_per_epoch = max(int(len(loader)), 1)
        out_vars = list(getattr(trainer, "output_variables", None) or [])
        if not self._built:
            if out_vars:
                n_data = len(out_vars)
            elif self._variable_names_arg:
                n_data = len(self._variable_names_arg)
            else:
                raise ValueError(
                    "AdaptiveLossWeights could not infer n_data_variables. "
                    "Set trainer.output_variables before setup(), or pass "
                    "n_data_variables."
                )
            self._build(n_data)
        elif out_vars and len(out_vars) != self.n_data_variables:
            raise ValueError(
                f"n_data_variables={self.n_data_variables} does not match "
                f"len(output_variables)={len(out_vars)}. Omit n_data_variables "
                "so it is inferred from the trainer outputs."
            )
        if out_vars and len(out_vars) == self.n_data_variables:
            # Prefer trainer channel names for logging when the user did not
            # pass variable_names explicitly (still named var_*).
            if all(n.startswith("var_") for n in self.variable_names):
                self.variable_names = list(out_vars)
                self._rebuild_term_names()
        device = trainer.device
        self.relative_scales = self.relative_scales.to(device=device)
        self.ema = self.ema.to(device=device)
        self.weights = self.weights.to(device=device)
        self.ema_initialized = self.ema_initialized.to(device=device)
        self._pending_unweighted = self._pending_unweighted.to(device=device)
        self._log_values = self._log_values.to(device=device)
        # Rebind after the device move and any variable-name rewrite so the
        # dict views the storage forward will ``copy_`` into.
        self._bind_log_buffers()
        # Warmup weights: 1 on variables, s_k on constraints.
        self.weights.copy_(self.relative_scales)

    def _ema_beta(self) -> float:
        if self.steps_per_epoch is None or self.steps_per_epoch < 1:
            raise RuntimeError(
                "steps_per_epoch is unset; call setup(trainer) first."
            )
        window_steps = self.ema_window_epochs * float(self.steps_per_epoch)
        return 1.0 - 1.0 / window_steps

    def _call_inner(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        input: Optional[torch.Tensor],
        input_diagnostics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        kwargs = {"average_channels": False}
        if self.needs_input:
            kwargs["input"] = input
        if self.needs_input_diagnostics:
            kwargs["input_diagnostics"] = input_diagnostics
        terms = self.inner(prediction, target, **kwargs)
        if terms.dim() == 0:
            terms = terms.unsqueeze(0)
        if terms.numel() != self.n_terms:
            raise ValueError(
                f"Inner criterion returned {terms.numel()} terms but "
                f"AdaptiveLossWeights expects {self.n_terms} "
                f"(n_data_variables={self.n_data_variables} + constraints)."
            )
        return terms.reshape(self.n_terms)

    def _fill_log_buffers(
        self,
        unweighted: torch.Tensor,
        weighted: torch.Tensor,
        scalar: torch.Tensor,
    ) -> None:
        # ``copy_`` into the buffers bound at setup. Assigning a new dict here
        # is not replayed, so the eval-graph capture would leave the trainer
        # reading capture-time tensors for every later training step.
        C = self.n_data_variables
        c_float = float(C)
        values: List[torch.Tensor] = [
            scalar.detach(),
            (weighted[:C].sum() / c_float).detach(),
        ]
        for _name, sl in self._group_slices:
            values.append((weighted[sl].sum() / c_float).detach())
            width = sl.stop - sl.start
            if width > 1:
                for i in range(width):
                    values.append((weighted[sl.start + i] / c_float).detach())
        for i in range(self.n_terms):
            values.append(self.weights[i].detach())
            values.append(weighted[i].detach())
            values.append(unweighted[i].detach())
        if len(values) != self._log_values.numel():
            raise RuntimeError(
                f"AdaptiveLossWeights produced {len(values)} log values "
                f"for {self._log_values.numel()} slots."
            )
        for i, val in enumerate(values):
            self._log_values[i].copy_(val)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        average_channels: bool = True,
        input: Optional[torch.Tensor] = None,
        input_diagnostics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._built:
            raise RuntimeError(
                "AdaptiveLossWeights.setup(trainer) must run before forward "
                "when n_data_variables is inferred."
            )
        if self.needs_input and input is None:
            raise ValueError(
                "AdaptiveLossWeights requires prognostic input because the "
                "inner criterion has needs_input=True."
            )
        if self.needs_input_diagnostics and input_diagnostics is None:
            raise ValueError(
                "AdaptiveLossWeights requires input_diagnostics because the "
                "inner criterion has needs_input_diagnostics=True."
            )

        unweighted = self._call_inner(
            prediction, target, input, input_diagnostics
        )
        # Detach for the post-backward EMA path; keep a graph on weighted terms.
        # ``copy_`` so train and eval graph replays write the same buffer the
        # trainer reads. Do not replace this tensor and do not clear it after
        # the update: replay would not restore the Python reference.
        self._pending_unweighted.copy_(unweighted.detach())
        weighted = unweighted * self.weights
        if not average_channels:
            self._fill_log_buffers(
                unweighted.detach(),
                weighted.detach(),
                reduce_per_term_loss(weighted.detach(), self.n_data_variables),
            )
            return weighted
        scalar = reduce_per_term_loss(weighted, self.n_data_variables)
        self._fill_log_buffers(
            unweighted.detach(), weighted.detach(), scalar.detach()
        )
        return scalar

    def _compute_adaptive_weights(self) -> torch.Tensor:
        m = self.ema.clamp_min(self.eps)
        T = m[: self.n_data_variables].mean()
        return self.relative_scales * (T / m)

    def post_backward_update(self, epoch: int) -> None:
        """All-reduce unweighted terms, update EMA, ``copy_`` weights.

        Call once per training step **after** ``loss.backward()``. Skip on
        eval/validation so the EMA is training-only.
        """
        if self._pending_unweighted is None:
            return
        terms = self._pending_unweighted
        if dist.is_available() and dist.is_initialized():
            # AVG across ranks so every GPU trains with the same weights.
            dist.all_reduce(terms, op=dist.ReduceOp.SUM)
            terms = terms / dist.get_world_size()

        if int(self.ema_initialized.item()) == 0:
            self.ema.copy_(terms)
            self.ema_initialized.fill_(1)
        else:
            beta = self._ema_beta()
            self.ema.mul_(beta).add_(terms, alpha=1.0 - beta)

        if int(epoch) >= self.warmup_epochs:
            self.weights.copy_(self._compute_adaptive_weights())
        else:
            self.weights.copy_(self.relative_scales)
