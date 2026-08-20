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

"""Transfer deterministic backbone weights into a probabilistic (CLN) model.

Conditional LayerNorm modules are inserted into ``nn.ModuleList`` / ``nn.Sequential``
slots, which shifts state-dict indices relative to a deterministic checkpoint.
Blind ``load_state_dict(..., strict=False)`` can therefore map conv weights onto
CLN parameters. This module walks paired module trees, skips
``ConditionalLayerNorm`` targets, and fails hard if any non-CLN parameter is
unmatched or shape-mismatched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn

from physicsnemo.models.dlwp_healpix_layers.normalization import ConditionalLayerNorm
from physicsnemo.models.module import Module


@dataclass
class TransferReport:
    """Summary of a deterministic → probabilistic weight transfer."""

    n_copied: int = 0
    n_cln_left_at_init: int = 0
    copied_keys: List[str] = field(default_factory=list)
    cln_keys: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"TransferReport(n_copied={self.n_copied}, "
            f"n_cln_left_at_init={self.n_cln_left_at_init})"
        )


class CheckpointTransferError(RuntimeError):
    """Raised when non-CLN architecture or shapes do not line up."""


def _is_cln(module: nn.Module) -> bool:
    return isinstance(module, ConditionalLayerNorm)


def _direct_named_tensors(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """Parameters and buffers registered directly on ``module`` (not descendants)."""
    items: List[Tuple[str, torch.Tensor]] = []
    for name, param in module.named_parameters(recurse=False):
        items.append((name, param))
    for name, buf in module.named_buffers(recurse=False):
        items.append((name, buf))
    return items


def _copy_tensor(src: torch.Tensor, tgt: torch.Tensor, path: str) -> None:
    if src.shape != tgt.shape:
        raise CheckpointTransferError(
            f"Shape mismatch at '{path}': source {tuple(src.shape)} vs target {tuple(tgt.shape)}"
        )
    with torch.no_grad():
        tgt.copy_(src)


def transfer_weights_skipping_cln(
    src: nn.Module,
    tgt: nn.Module,
    prefix: str = "",
) -> TransferReport:
    """Copy weights from ``src`` into ``tgt``, skipping ConditionalLayerNorm modules.

    Parameters
    ----------
    src :
        Deterministic (or otherwise CLN-free at the mismatched slots) module.
    tgt :
        Probabilistic target that may contain ``ConditionalLayerNorm`` modules.
    prefix :
        Dotted path prefix for error messages.

    Returns
    -------
    TransferReport
    """
    report = TransferReport()
    _transfer_module(src, tgt, prefix, report)
    return report


def _transfer_module(
    src: nn.Module,
    tgt: nn.Module,
    prefix: str,
    report: TransferReport,
) -> None:
    if _is_cln(tgt):
        # Caller should have skipped CLN slots; treat as leave-at-init if reached.
        for name, _ in _direct_named_tensors(tgt):
            key = f"{prefix}.{name}" if prefix else name
            report.cln_keys.append(key)
            report.n_cln_left_at_init += 1
        for name, child in tgt.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            _record_cln_subtree(child, child_prefix, report)
        return

    if _is_cln(src):
        raise CheckpointTransferError(
            f"Source module at '{prefix or '<root>'}' is ConditionalLayerNorm, "
            "but the deterministic checkpoint should not contain CLN modules."
        )

    # Copy tensors registered directly on this module.
    src_tensors = dict(_direct_named_tensors(src))
    tgt_tensors = dict(_direct_named_tensors(tgt))
    if set(src_tensors) != set(tgt_tensors):
        only_src = sorted(set(src_tensors) - set(tgt_tensors))
        only_tgt = sorted(set(tgt_tensors) - set(src_tensors))
        raise CheckpointTransferError(
            f"Direct parameter/buffer name mismatch at '{prefix or '<root>'}': "
            f"only in source={only_src}, only in target={only_tgt}"
        )
    for name in sorted(src_tensors):
        key = f"{prefix}.{name}" if prefix else name
        _copy_tensor(src_tensors[name], tgt_tensors[name], key)
        report.copied_keys.append(key)
        report.n_copied += 1

    # Align children. ModuleList / Sequential are positional after stripping CLN.
    if isinstance(src, (nn.ModuleList, nn.Sequential)) and isinstance(
        tgt, (nn.ModuleList, nn.Sequential)
    ):
        src_children = list(src)
        tgt_indexed = list(enumerate(tgt))
        tgt_non_cln = [(i, c) for i, c in tgt_indexed if not _is_cln(c)]
        for idx, cln in ((i, c) for i, c in tgt_indexed if _is_cln(c)):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            _record_cln_subtree(cln, child_prefix, report)

        if len(src_children) != len(tgt_non_cln):
            raise CheckpointTransferError(
                f"ModuleList/Sequential length mismatch at '{prefix or '<root>'}': "
                f"source has {len(src_children)} non-CLN children, "
                f"target has {len(tgt_non_cln)} non-CLN children "
                f"(plus {len(tgt) - len(tgt_non_cln)} ConditionalLayerNorm modules)."
            )
        for src_child, (tgt_idx, tgt_child) in zip(src_children, tgt_non_cln):
            child_prefix = f"{prefix}.{tgt_idx}" if prefix else str(tgt_idx)
            _transfer_module(src_child, tgt_child, child_prefix, report)
        return

    if isinstance(src, (nn.ModuleList, nn.Sequential)) != isinstance(
        tgt, (nn.ModuleList, nn.Sequential)
    ):
        raise CheckpointTransferError(
            f"Container type mismatch at '{prefix or '<root>'}': "
            f"source={type(src).__name__}, target={type(tgt).__name__}"
        )

    src_children = dict(src.named_children())
    tgt_children = dict(tgt.named_children())

    for name, tgt_child in tgt_children.items():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if _is_cln(tgt_child):
            if name in src_children:
                raise CheckpointTransferError(
                    f"Target ConditionalLayerNorm at '{child_prefix}' has a "
                    f"source child with the same name; expected CLN-only on target."
                )
            _record_cln_subtree(tgt_child, child_prefix, report)
            continue
        if name not in src_children:
            raise CheckpointTransferError(
                f"Target child '{child_prefix}' has no matching source child "
                f"(non-CLN modules must align)."
            )
        _transfer_module(src_children[name], tgt_child, child_prefix, report)

    for name in src_children:
        if name not in tgt_children:
            child_prefix = f"{prefix}.{name}" if prefix else name
            raise CheckpointTransferError(
                f"Source child '{child_prefix}' has no matching target child."
            )


def _record_cln_subtree(module: nn.Module, prefix: str, report: TransferReport) -> None:
    for name, _ in module.named_parameters():
        key = f"{prefix}.{name}" if prefix else name
        report.cln_keys.append(key)
        report.n_cln_left_at_init += 1
    for name, _ in module.named_buffers():
        key = f"{prefix}.{name}" if prefix else name
        if key not in report.cln_keys:
            report.cln_keys.append(key)
            report.n_cln_left_at_init += 1


def load_deterministic_weights_into_probabilistic_model(
    model: nn.Module,
    checkpoint_path: str,
    *,
    map_location=None,
) -> TransferReport:
    """Load a deterministic ``.mdlus`` checkpoint into a probabilistic ``model``.

    Instantiates the source architecture from the checkpoint, then transfers
    weights into ``model`` while leaving ConditionalLayerNorm parameters at
    their current initialization.

    Parameters
    ----------
    model :
        Already-constructed probabilistic model (caller instantiates from config).
    checkpoint_path :
        Path to a PhysicsNeMo ``.mdlus`` deterministic checkpoint.
    map_location :
        Unused; kept for API symmetry. Source is built via ``Module.from_checkpoint``.

    Returns
    -------
    TransferReport
    """
    del map_location  # from_checkpoint places tensors on the model device
    src = Module.from_checkpoint(checkpoint_path)
    return transfer_weights_skipping_cln(src, model)


def assert_no_cln_params_copied(
    report: TransferReport, cln_substr: str = "gamma_beta_mlp"
) -> None:
    """Helper for tests: ensure no copied key looks like a CLN MLP weight."""
    bad = [k for k in report.copied_keys if cln_substr in k]
    if bad:
        raise AssertionError(f"CLN keys were copied unexpectedly: {bad}")
