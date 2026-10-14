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

"""Load public HEALPix layers without eager optional configuration imports."""

from importlib import import_module

_EXPORTS = {
    **dict.fromkeys(
        ["AvgPool", "BasicConvBlock", "ConvGRUBlock", "ConvNeXtBlock",
         "DealiasedDownsample", "DoubleConvNeXtBlock", "Interpolate", "MaxPool",
         "Multi_SymmetricConvNeXtBlock", "SymmetricConvNeXtBlock",
         "TransposedConvUpsample", "SmoothedInterpolateConv"], "healpix_blocks"
    ),
    "UNetDecoder": "healpix_decoder",
    "UNetEncoder": "healpix_encoder",
    "HEALPixLayer": "healpix_layers",
    **dict.fromkeys(
        ["HEALPixFoldFaces", "HEALPixPadding", "HEALPixPaddingIsolatitude",
         "HEALPixPaddingv2", "HEALPixUnfoldFaces", "have_earth2grid",
         "make_hpx_padding_layer", "warn_deprecated_enable_healpixpad"],
        "healpix_paddings"
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Resolve the original public layer names only when explicitly requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
