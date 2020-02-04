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

from numbers import Number
from typing import Any, Tuple

import numpy as np
import scipy.special
import torch


class IveFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, v: Number, z: torch.Tensor) -> torch.Tensor:
        assert isinstance(v, Number), "v must be a scalar"

        ctx.save_for_backward(z)
        ctx.v = v
        z_cpu = z.double().detach().cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:  # v > 0
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        # else:
        #     print(v, type(v), np.isclose(v, 0))
        #     raise RuntimeError(f"v must be >= 0, it is {v}")

        return torch.tensor(output, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:
        z = ctx.saved_tensors[-1]
        return None, grad_output * (ive(ctx.v - 1, z) - ive(ctx.v, z) * (ctx.v + z) / z)


def ive(v: Number, z: torch.Tensor) -> torch.Tensor:
    return IveFunction.apply(v, z)
