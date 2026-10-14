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


"""Public layer exports, importing optional implementations on first access."""

from importlib import import_module

_EXPORTS = {
    "CappedGELU": "activations",
    "CappedLeakyReLU": "activations",
    "Identity": "activations",
    "SquarePlus": "activations",
    "Stan": "activations",
    "get_activation": "activations",
    "ConvBlock": "conv_layers",
    "CubeEmbedding": "conv_layers",
    "DGMLayer": "dgm_layers",
    "FourierFilter": "fourier_layers",
    "FourierLayer": "fourier_layers",
    "GaborFilter": "fourier_layers",
    "Conv1dFCLayer": "fully_connected_layers",
    "Conv2dFCLayer": "fully_connected_layers",
    "Conv3dFCLayer": "fully_connected_layers",
    "ConvNdFCLayer": "fully_connected_layers",
    "ConvNdKernel1Layer": "fully_connected_layers",
    "FCLayer": "fully_connected_layers",
    "Mlp": "mlp_layers",
    "DownSample2D": "resample_layers",
    "DownSample3D": "resample_layers",
    "UpSample2D": "resample_layers",
    "UpSample3D": "resample_layers",
    "SirenLayer": "siren_layers",
    "SirenLayerType": "siren_layers",
    "SpectralConv1d": "spectral_layers",
    "SpectralConv2d": "spectral_layers",
    "SpectralConv3d": "spectral_layers",
    "SpectralConv4d": "spectral_layers",
    "DecoderLayer": "transformer_layers",
    "EncoderLayer": "transformer_layers",
    "FuserLayer": "transformer_layers",
    "SwinTransformer": "transformer_layers",
    "WeightFactLinear": "weight_fact",
    "WeightNormLinear": "weight_norm",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
