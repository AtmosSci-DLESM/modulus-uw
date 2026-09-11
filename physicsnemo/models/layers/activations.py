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

import torch
import torch.nn as nn

import physicsnemo  # noqa: F401 for docs

Tensor = torch.Tensor


class Identity(nn.Module):
    """Identity activation function

    Dummy function for removing activations from a model

    Example
    -------
    >>> idnt_func = physicsnemo.models.layers.Identity()
    >>> input = torch.randn(2, 2)
    >>> output = idnt_func(input)
    >>> torch.allclose(input, output)
    True
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class Stan(nn.Module):
    """Self-scalable Tanh (Stan) for 1D Tensors

    Parameters
    ----------
    out_features : int, optional
        Number of features, by default 1

    Note
    ----
    References: Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and others.
    Self-scalable Tanh (Stan): Faster Convergence and Better Generalization
    in Physics-informed Neural Networks. arXiv preprint arXiv:2204.12589, 2022.

    Example
    -------
    >>> stan_func = physicsnemo.models.layers.Stan(out_features=1)
    >>> input = torch.Tensor([[0],[1],[2]])
    >>> stan_func(input)
    tensor([[0.0000],
            [1.5232],
            [2.8921]], grad_fn=<MulBackward0>)
    """

    def __init__(self, out_features: int = 1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.beta.shape[-1]:
            raise ValueError(
                f"The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}"
            )
        return torch.tanh(x) * (1.0 + self.beta * x)


class SquarePlus(nn.Module):
    """Squareplus activation

    Note
    ----
    Reference: arXiv preprint arXiv:2112.11687

    Example
    -------
    >>> sqr_func = physicsnemo.models.layers.SquarePlus()
    >>> input = torch.Tensor([[1,2],[3,4]])
    >>> sqr_func(input)
    tensor([[1.6180, 2.4142],
            [3.3028, 4.2361]])
    """

    def __init__(self):
        super().__init__()
        self.b = 4

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (x + torch.sqrt(x * x + self.b))


class CappedLeakyReLU(torch.nn.Module):
    """
    Implements a ReLU with capped maximum value.

    Example
    -------
    >>> capped_leakyReLU_func = physicsnemo.models.layers.CappedLeakyReLU()
    >>> input = torch.Tensor([[-2,-1],[0,1],[2,3]])
    >>> capped_leakyReLU_func(input)
    tensor([[-0.0200, -0.0100],
            [ 0.0000,  1.0000],
            [ 1.0000,  1.0000]])

    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        Parameters:
        ----------
        cap_value: float, optional
            Maximum that values will be capped at
        **kwargs:
             Keyword arguments to be passed to the `torch.nn.LeakyReLU` function
        """
        super().__init__()
        self.add_module("leaky_relu", torch.nn.LeakyReLU(**kwargs))
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.leaky_relu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x


def _capped_gelu(gelu: nn.Module, inputs: Tensor, cap: float) -> Tensor:
    return torch.clamp(gelu(inputs), max=cap)


# Compiled at module scope so every CappedGELU instance shares one compilation,
# and so instances hold no compiled callable of their own (keeps them picklable
# and deep-copyable).
_capped_gelu_compiled = torch.compile(_capped_gelu)


class CappedGELU(torch.nn.Module):
    """
    Implements a GELU with capped maximum value.

    Example
    -------
    >>> capped_gelu_func = physicsnemo.models.layers.CappedGELU()
    >>> input = torch.Tensor([[-2,-1],[0,1],[2,3]])
    >>> capped_gelu_func(input)
    tensor([[-0.0455, -0.1587],
            [ 0.0000,  0.8413],
            [ 1.0000,  1.0000]])

    """

    def __init__(self, cap_value=1.0, compile_forward=False, **kwargs):
        """
        Parameters:
        ----------
        cap_value: float, optional
            Maximum that values will be capped at
        compile_forward: bool, optional
            Fuse the GELU and the clamp into a single kernel via `torch.compile`.
            Costs a few seconds of compilation on the first call, so it is off by
            default; it is also redundant when the whole model is compiled.
        **kwargs:
             Keyword arguments to be passed to the `torch.nn.GELU` function
        """

        super().__init__()
        self.add_module("gelu", torch.nn.GELU(**kwargs))
        # Buffer is retained because existing checkpoints carry a `cap` entry.
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))
        self._cap_value = float(cap_value)
        self.compile_forward = compile_forward

    @property
    def cap_value(self) -> float:
        """Maximum that values will be capped at."""
        return self._cap_value

    @cap_value.setter
    def cap_value(self, value: float) -> None:
        # Used to keep cap_value in sync with the buffer
        self._cap_value = float(value)
        with torch.no_grad():
            self.cap.fill_(self._cap_value)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # `load_state_dict` copies into `cap` in place and so bypasses
        # __setattr__; the scalar that forward reads has to be refreshed here or
        # it would silently keep the value passed to __init__.
        self._cap_value = float(self.cap)

    def forward(self, inputs):
        if self.compile_forward:
            return _capped_gelu_compiled(self.gelu, inputs, self._cap_value)
        # A Python float keeps clamp on the faster scalar overload; passing the
        # `cap` buffer instead dispatches to clamp.Tensor, which is ~1.4x slower
        # on reduced-precision inputs.
        return torch.clamp(self.gelu(inputs), max=self._cap_value)


# Dictionary of activation functions
ACT2FN = {
    "relu": nn.ReLU,
    "leaky_relu": (nn.LeakyReLU, {"negative_slope": 0.1}),
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "elu": nn.ELU,
    "celu": (nn.CELU, {"alpha": 1.0}),
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "logsigmoid": nn.LogSigmoid,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": (nn.Threshold, {"threshold": 1.0, "value": 1.0}),
    "hardtanh": nn.Hardtanh,
    "identity": Identity,
    "stan": Stan,
    "squareplus": SquarePlus,
    "cappek_leaky_relu": CappedLeakyReLU,
    "capped_gelu": CappedGELU,
}


def get_activation(activation: str) -> nn.Module:
    """Returns an activation function given a string

    Parameters
    ----------
    activation : str
        String identifier for the desired activation function

    Returns
    -------
    Activation function

    Raises
    ------
    KeyError
        If the specified activation function is not found in the dictionary
    """
    try:
        activation = activation.lower()
        module = ACT2FN[activation]
        if isinstance(module, tuple):
            return module[0](**module[1])
        else:
            return module()
    except KeyError:
        raise KeyError(
            f"Activation function {activation} not found. Available options are: {list(ACT2FN.keys())}"
        )
