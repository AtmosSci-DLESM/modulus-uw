# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Differentiable coupling operations for HEALPix Earth-system components.

These are the tensor operations behind
:mod:`physicsnemo.datapipes.healpix.couplers`, factored out as pure functions so
that inference and training can share one implementation with the dataloader
instead of transcribing the math. Every function here is free of side effects
and preserves the autograd graph, which makes them safe under CUDA graph
capture and activation checkpointing.

The coupling pipeline is always the same three steps, wrapped by
:func:`apply_coupling`:

1. select the coupled channels out of the source component's output,
2. reduce the time axis with a *physical operation*
   (:func:`constant_coupling` or :func:`trailing_average_coupling`),
3. optionally carry that reduction through physical space by denormalizing
   before it and renormalizing after.

Scaling statistics are passed as ``{"mean": Tensor, "std": Tensor}`` mappings
whose entries broadcast against a ``[B, F, T, C, H, W]`` field.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Mapping, Optional, Sequence

import torch as th

__all__ = [
    "CONSTANT",
    "TRAILING_AVERAGE",
    "CouplingOp",
    "apply_coupling",
    "concat_integrated_couplings",
    "constant_coupling",
    "denormalize",
    "renormalize",
    "rescale_through_physical",
    "trailing_average_coupling",
]

CONSTANT = "constant"
TRAILING_AVERAGE = "trailing_average"

PhysicalOp = Callable[[th.Tensor], th.Tensor]
Scaling = Mapping[str, th.Tensor]


def concat_integrated_couplings(
    couplings: Sequence, device=None, dtype=None
) -> th.Tensor:
    """Gather each coupler's current coupled fields into one tensor.

    ``construct_integrated_couplings`` returns a tensor when fields were preset
    with ``set_coupled_fields`` and a numpy array when read from the dataset.
    Both are normalized to a tensor here rather than routed through
    ``numpy.concatenate``, which would silently drop the autograd graph of a
    preset field and reject fields that live on an accelerator.

    Parameters
    ----------
    couplings: Sequence
        Coupler objects, in the order their channels appear in the coupled
        input.
    device: torch.device, optional
        Device to place the result on. Defaults to leaving each field where it
        is.
    dtype: torch.dtype, optional
        Dtype to cast the result to. Defaults to leaving fields as they are.

    Returns
    -------
    th.Tensor
        Coupled fields concatenated along the channel axis (dim 2).
    """
    fields = []
    for coupler in couplings:
        field = coupler.construct_integrated_couplings()
        if not th.is_tensor(field):
            field = th.as_tensor(field)
        if device is not None or dtype is not None:
            field = field.to(device=device, dtype=dtype)
        fields.append(field)
    return th.cat(fields, dim=2)


def denormalize(x: th.Tensor, scaling: Scaling) -> th.Tensor:
    """Map normalized values back to physical units.

    Parameters
    ----------
    x: th.Tensor
        Normalized field.
    scaling: Mapping[str, th.Tensor]
        ``mean`` and ``std`` broadcastable against ``x``.

    Returns
    -------
    th.Tensor
        ``x * std + mean``.
    """
    mean = scaling["mean"].to(device=x.device, dtype=x.dtype)
    std = scaling["std"].to(device=x.device, dtype=x.dtype)
    return x * std + mean


def renormalize(x: th.Tensor, scaling: Scaling) -> th.Tensor:
    """Map physical values into normalized space.

    Parameters
    ----------
    x: th.Tensor
        Field in physical units.
    scaling: Mapping[str, th.Tensor]
        ``mean`` and ``std`` broadcastable against ``x``.

    Returns
    -------
    th.Tensor
        ``(x - mean) / std``.
    """
    mean = scaling["mean"].to(device=x.device, dtype=x.dtype)
    std = scaling["std"].to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def constant_coupling(
    fields: th.Tensor, integration_dim: int, time_index: int = 0
) -> th.Tensor:
    """Hold one time step constant across the coupled integration window.

    Parameters
    ----------
    fields: th.Tensor
        Source fields, ``[B, F, T, C, H, W]``.
    integration_dim: int
        Number of coupled integration steps to produce.
    time_index: int, optional
        Index along the time axis to hold constant. Negative values count from
        the end. Default 0, the first available time step.

    Returns
    -------
    th.Tensor
        ``[B, F, integration_dim, C, H, W]``.
    """
    if time_index < 0:
        time_index += fields.shape[2]
    selected = fields[:, :, time_index : time_index + 1]
    # expand returns a view; clone so later mutation of fields cannot write
    # through into the coupled result.
    return selected.expand(-1, -1, integration_dim, -1, -1, -1).clone()


def trailing_average_coupling(
    fields: th.Tensor, averaging_slices: Sequence[Sequence[slice]]
) -> th.Tensor:
    """Average over trailing windows and stack them period-major.

    Parameters
    ----------
    fields: th.Tensor
        Source fields, ``[B, F, T, C, H, W]``.
    averaging_slices: Sequence[Sequence[slice]]
        One list of time slices per coupled integration step; each slice is one
        averaging window, ordered by ``input_times``.

    Returns
    -------
    th.Tensor
        ``[B, F, integration, timevar, H, W]`` where ``timevar`` runs
        period-major, ``[p0_v0, p0_v1, ..., p1_v0, ...]``.
    """
    coupled_averaging_periods = []
    for slices in averaging_slices:
        averaging_periods = [
            fields[:, :, s, :, :, :].mean(dim=2, keepdim=True) for s in slices
        ]
        coupled_averaging_periods.append(th.concat(averaging_periods, dim=3))
    return th.concat(coupled_averaging_periods, dim=2)


def rescale_through_physical(
    fields: th.Tensor,
    physical_op: PhysicalOp,
    incoming_scaling: Scaling,
    outgoing_scaling: Scaling,
) -> th.Tensor:
    """Run ``physical_op`` in physical units rather than normalized space.

    Averaging in normalized space is only equivalent to averaging in physical
    space when both sides share one affine transform. Coupled fields usually do
    not: instantaneous source outputs are normalized with instantaneous
    statistics while the coupled target uses trailing-window statistics.

    Denormalized values can overflow reduced precision, so the round trip runs
    in at least float32 and the input dtype is restored at the end. Inputs that
    are already wider than float32 keep their precision.

    This rescales unconditionally. Deciding *whether* a coupling should run in
    physical space belongs to the caller; :func:`apply_coupling` owns that
    branch.

    Parameters
    ----------
    fields: th.Tensor
        Normalized source fields.
    physical_op: Callable[[th.Tensor], th.Tensor]
        Time-axis reduction to apply in physical units.
    incoming_scaling: Mapping[str, th.Tensor]
        Statistics that normalized ``fields``.
    outgoing_scaling: Mapping[str, th.Tensor]
        Statistics to renormalize the reduced field with. Must broadcast
        against the output of ``physical_op``.

    Returns
    -------
    th.Tensor
        Reduced field in outgoing normalized space, in the input dtype.
    """
    orig_dtype = fields.dtype
    compute_dtype = th.promote_types(orig_dtype, th.float32)
    physical_fields = denormalize(fields.to(compute_dtype), incoming_scaling)
    transformed_fields = physical_op(physical_fields)
    normalized_fields = renormalize(transformed_fields, outgoing_scaling)
    return normalized_fields.to(dtype=orig_dtype)


def apply_coupling(
    fields: th.Tensor,
    channel_indices: Sequence[int],
    physical_op: PhysicalOp,
    incoming_scaling: Optional[Scaling] = None,
    outgoing_scaling: Optional[Scaling] = None,
    time_first: bool = True,
    rescale_in_physical_space: bool = True,
) -> th.Tensor:
    """Select coupled channels, reduce the time axis, and lay out the result.

    This is the whole coupling operation. It is a pure function of its
    arguments: nothing is cached and ``fields`` is not modified.

    Parameters
    ----------
    fields: th.Tensor
        Source component output, ``[B, F, T, C, H, W]``.
    channel_indices: Sequence[int]
        Channels of ``fields`` that back the coupled variables, in coupled
        order. Indices may repeat when several coupled variables share one
        source channel.
    physical_op: Callable[[th.Tensor], th.Tensor]
        Time-axis reduction, e.g. :func:`constant_coupling` or
        :func:`trailing_average_coupling` bound to their configuration.
    incoming_scaling: Mapping[str, th.Tensor], optional
        Statistics that normalized ``fields``. When given (together with
        ``outgoing_scaling``) the reduction runs in physical space.
    outgoing_scaling: Mapping[str, th.Tensor], optional
        Statistics to renormalize with. Required when ``incoming_scaling`` is
        given.
    time_first: bool, optional
        If True, return ``[T, B, C, F, H, W]``; otherwise keep
        ``[B, F, T, C, H, W]``. Default True.
    rescale_in_physical_space: bool, optional
        Set False to reduce in normalized space even though scaling statistics
        were supplied. This is a veto, not a request: with no statistics the
        reduction runs in normalized space either way. It exists so that
        attaching statistics and acting on them are separate decisions, which
        keeps a caller from silently changing its numbers just because the
        dataset happened to carry statistics. Default True.

    Returns
    -------
    th.Tensor
        The coupled field, laid out per ``time_first``.

    Raises
    ------
    ValueError
        If exactly one of ``incoming_scaling`` and ``outgoing_scaling`` is given.
    """
    # Validated even when rescaling is vetoed: half-configured statistics are a
    # configuration error whether or not this call would have used them.
    if (incoming_scaling is None) != (outgoing_scaling is None):
        raise ValueError(
            "incoming_scaling and outgoing_scaling must be provided together; "
            "got incoming="
            f"{'set' if incoming_scaling is not None else 'None'}, outgoing="
            f"{'set' if outgoing_scaling is not None else 'None'}"
        )

    coupled = fields[:, :, :, channel_indices, :, :]
    if rescale_in_physical_space and incoming_scaling is not None:
        coupled = rescale_through_physical(
            coupled, physical_op, incoming_scaling, outgoing_scaling
        )
    else:
        coupled = physical_op(coupled)
    if time_first:
        coupled = coupled.permute(2, 0, 3, 1, 4, 5)
    return coupled


class CouplingOp(th.nn.Module):
    """One component's coupling, as a module that carries its own indices and stats.

    Wraps :func:`apply_coupling` so that the configuration a coupling needs
    (which source channels, which time reduction, which scaling statistics)
    travels with the object and moves across devices with ``.to()``. ``forward``
    mutates nothing, so the same instance is safe inside CUDA graph capture and
    activation checkpointing as well as in the dataloader.

    Build one with :meth:`from_coupler` when the coupling was derived from a
    dataset, or :meth:`from_spec` when it was derived from a coupled-system
    config.

    Parameters
    ----------
    mode: str
        ``"constant"`` or ``"trailing_average"``.
    channel_indices: Sequence[int]
        Channels of the source component's output that back the coupled
        variables, in coupled order. Indices may repeat.
    integration_dim: int, optional
        Number of coupled integration steps. Required for ``"constant"``.
    time_index: int, optional
        Source time step to hold constant, used by ``"constant"``. Negative
        values count from the end. Default 0.
    averaging_slices: Sequence[Sequence[slice]], optional
        Averaging windows per integration step. Required for
        ``"trailing_average"``.
    time_first: bool, optional
        Whether to emit ``[T, B, C, F, H, W]``. Default True.
    rescale_in_physical_space: bool, optional
        Whether to act on scaling statistics when they are set. Default True.
        See :attr:`rescales_through_physical` for the effective decision.
    """

    def __init__(
        self,
        mode: str,
        channel_indices: Sequence[int],
        integration_dim: Optional[int] = None,
        time_index: int = 0,
        averaging_slices: Optional[Sequence[Sequence[slice]]] = None,
        time_first: bool = True,
        rescale_in_physical_space: bool = True,
    ):
        super().__init__()
        if mode == CONSTANT:
            if integration_dim is None:
                raise ValueError("constant coupling requires integration_dim")
        elif mode == TRAILING_AVERAGE:
            if averaging_slices is None:
                raise ValueError("trailing_average coupling requires averaging_slices")
        else:
            raise ValueError(
                f"unknown coupling mode {mode!r}, expected {CONSTANT!r} or "
                f"{TRAILING_AVERAGE!r}"
            )

        self.mode = mode
        self.time_first = bool(time_first)
        self.rescale_in_physical_space = bool(rescale_in_physical_space)
        self.time_index = int(time_index)
        self.integration_dim = (
            len(averaging_slices) if integration_dim is None else int(integration_dim)
        )
        self.averaging_slices = (
            None
            if averaging_slices is None
            else tuple(tuple(windows) for windows in averaging_slices)
        )

        # Indices and statistics are non-persistent: they are derived from the
        # dataset or the coupled-system config, not learned, so they should not
        # travel in a checkpoint and go stale.
        self.register_buffer(
            "channel_indices",
            th.as_tensor(channel_indices, dtype=th.long),
            persistent=False,
        )
        for name in (
            "incoming_mean",
            "incoming_std",
            "outgoing_mean",
            "outgoing_std",
        ):
            self.register_buffer(name, None, persistent=False)

    @property
    def incoming_scaling(self) -> Optional[dict]:
        """Statistics that normalized the source fields, or None."""
        if self.incoming_mean is None:
            return None
        return {"mean": self.incoming_mean, "std": self.incoming_std}

    @property
    def outgoing_scaling(self) -> Optional[dict]:
        """Statistics used to renormalize the coupled result, or None."""
        if self.outgoing_mean is None:
            return None
        return {"mean": self.outgoing_mean, "std": self.outgoing_std}

    @property
    def rescales_through_physical(self) -> bool:
        """Whether :meth:`forward` will reduce in physical rather than
        normalized space.

        Both conditions have to hold: statistics must be set *and*
        ``rescale_in_physical_space`` must allow using them. Callers that care
        whether their numbers changed should read this rather than inferring it
        from the presence of statistics.
        """
        return self.rescale_in_physical_space and self.incoming_mean is not None

    def set_scaling(
        self, incoming: Optional[Scaling], outgoing: Optional[Scaling]
    ) -> None:
        """Install (or clear) the statistics that move the reduction into
        physical space.

        Parameters
        ----------
        incoming: Mapping[str, th.Tensor], optional
            Statistics that normalized the source fields.
        outgoing: Mapping[str, th.Tensor], optional
            Statistics to renormalize the coupled result with.

        Raises
        ------
        ValueError
            If exactly one of ``incoming`` and ``outgoing`` is given.
        """
        if (incoming is None) != (outgoing is None):
            raise ValueError("incoming and outgoing scaling must be set together")
        if incoming is None:
            self.incoming_mean = None
            self.incoming_std = None
            self.outgoing_mean = None
            self.outgoing_std = None
            return
        self.incoming_mean = th.as_tensor(incoming["mean"])
        self.incoming_std = th.as_tensor(incoming["std"])
        self.outgoing_mean = th.as_tensor(outgoing["mean"])
        self.outgoing_std = th.as_tensor(outgoing["std"])

    def physical_op(self) -> PhysicalOp:
        """The time-axis reduction this coupling applies, bound to its config."""
        if self.mode == CONSTANT:
            return partial(
                constant_coupling,
                integration_dim=self.integration_dim,
                time_index=self.time_index,
            )
        return partial(
            trailing_average_coupling, averaging_slices=self.averaging_slices
        )

    def forward(self, fields: th.Tensor) -> th.Tensor:
        """Couple a source component's output.

        Parameters
        ----------
        fields: th.Tensor
            Source component output, ``[B, F, T, C, H, W]``.

        Returns
        -------
        th.Tensor
            The coupled field, laid out per ``time_first``.
        """
        return apply_coupling(
            fields,
            self.channel_indices,
            self.physical_op(),
            incoming_scaling=self.incoming_scaling,
            outgoing_scaling=self.outgoing_scaling,
            time_first=self.time_first,
            rescale_in_physical_space=self.rescale_in_physical_space,
        )

    @classmethod
    def from_coupler(
        cls, coupler, rescale_in_physical_space: bool = True
    ) -> "CouplingOp":
        """Build the op for a coupler configured against a dataset.

        Reads what ``setup_coupling`` and ``set_coupled_scaling`` derived, so
        the module and the dataloader cannot drift apart.

        Parameters
        ----------
        coupler: physicsnemo.datapipes.healpix.couplers.BaseCoupler
            A coupler whose ``setup_coupling`` has already run.
        rescale_in_physical_space: bool, optional
            Whether to act on the coupler's scaling statistics. Pass False to
            reduce in normalized space while leaving the statistics attached,
            which is how a caller reproduces pre-rescaling numbers. Default
            True.

        Returns
        -------
        CouplingOp
        """
        op = cls(
            mode=coupler.coupling_method,
            channel_indices=coupler.coupled_channel_indices,
            integration_dim=coupler.coupled_integration_dim,
            averaging_slices=getattr(coupler, "averaging_slices", None),
            time_first=coupler.time_first,
            rescale_in_physical_space=rescale_in_physical_space,
        )
        op.set_scaling(
            coupler.incoming_coupled_scaling, coupler.outgoing_coupled_scaling
        )
        return op

    @classmethod
    def from_spec(
        cls,
        spec: Mapping,
        averaging_slices: Optional[Sequence[Sequence[slice]]] = None,
        channel_indices: Optional[Sequence[int]] = None,
        time_first: bool = True,
        rescale_in_physical_space: Optional[bool] = None,
    ) -> "CouplingOp":
        """Build the op from a coupled-system coupling specification.

        Note the naming mismatch between the two derivations: a spec's
        ``variable_indices`` are the source channels this op selects, while its
        ``channel_indices`` are the destination slots in the concatenated
        coupling tensor and are not used here.

        Parameters
        ----------
        spec: Mapping
            Entry with ``method``, ``variable_indices``, and
            ``timestep_indices``. For ``"constant"``, ``timestep_indices`` is
            one source index repeated once per integration step.
        averaging_slices: Sequence[Sequence[slice]], optional
            Averaging windows, required for ``"trailing_average"``. The caller
            owns this because the window layout is a property of the coupled
            system's time scheme rather than of the spec.
        channel_indices: Sequence[int], optional
            Source channels to select, overriding ``spec["variable_indices"]``.
            Lets a caller that has already materialized these indices as a
            tensor hand that tensor over instead of creating a second copy.
        time_first: bool, optional
            Whether to emit ``[T, B, C, F, H, W]``. Default True.
        rescale_in_physical_space: bool, optional
            Whether to act on scaling statistics once they are set. Defaults to
            the spec's ``rescale_in_physical_space`` entry, so a coupled-system
            config can turn physical-space rescaling on per coupling, and to
            True when the spec is silent.

        Returns
        -------
        CouplingOp

        Raises
        ------
        ValueError
            If a constant coupling's ``timestep_indices`` are not a single
            repeated index.
        """
        method = spec["method"]
        timestep_indices = list(spec["timestep_indices"])
        if channel_indices is None:
            channel_indices = spec["variable_indices"]
        if rescale_in_physical_space is None:
            rescale_in_physical_space = spec.get("rescale_in_physical_space", True)
        if method == CONSTANT:
            if len(set(timestep_indices)) != 1:
                raise ValueError(
                    "constant coupling expects a single repeated source "
                    f"timestep, got {timestep_indices}"
                )
            return cls(
                mode=CONSTANT,
                channel_indices=channel_indices,
                integration_dim=len(timestep_indices),
                time_index=int(timestep_indices[0]),
                time_first=time_first,
                rescale_in_physical_space=rescale_in_physical_space,
            )
        return cls(
            mode=method,
            channel_indices=channel_indices,
            averaging_slices=averaging_slices,
            time_first=time_first,
            rescale_in_physical_space=rescale_in_physical_space,
        )
