# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Tuple

import torch
from torch import Tensor


def rexpand(a: Tensor, *dimensions: Tuple[int]) -> Tensor:
    """Expand tensor, adding new dimensions on right."""
    return a.view(a.shape + (1,) * len(dimensions)).expand(a.shape + tuple(dimensions))


def log_sum_exp_signs(value: Tensor, signs: Tensor, dim: int = 0, keepdim: bool = False) -> Tensor:
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.clamp(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim), min=1e15))
