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

from typing import Any, Optional, Tuple
import math

import torch

eps = 1e-8  # TODO: Move this lower for doubles?
max_norm = 85
ln_2: torch.Tensor = math.log(2)
ln_pi: torch.Tensor = math.log(math.pi)
ln_2pi: torch.Tensor = ln_2 + ln_pi


class LeakyClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None


def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)


class Atanh(torch.autograd.Function):
    """
    Numerically stable arctanh that never returns NaNs.
    x = clamp(x, min=-1+eps, max=1-eps)
    Returns atanh(x) = arctanh(x) = 0.5*(log(1+x)-log(1-x)).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        x = clamp(x, min=-1. + 4 * eps, max=1. - 4 * eps)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        return grad_output / (1 - x**2)


def atanh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable arctanh that never returns NaNs.

    :param x: The input tensor.
    :return: log(x + sqrt(max(x^2 - 1, eps))
    """
    return Atanh.apply(x)


class Acosh(torch.autograd.Function):
    """
    Numerically stable arccosh that never returns NaNs.
    Returns acosh(x) = arccosh(x) = log(x + sqrt(max(x^2 - 1, eps))).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        x = clamp(x, min=1 + eps)
        z = sqrt(x * x - 1.)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        z, = ctx.saved_tensors
        # z_ = clamp(z, min=eps)
        z_ = z
        return grad_output / z_


def acosh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable arccosh that never returns NaNs.

    :param x: The input tensor.
    :return: log(x + sqrt(max(x^2 - 1, eps))
    """
    return Acosh.apply(x)


def cosh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.cosh(x)


def sinh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.sinh(x)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=1e-9)  # Smaller epsilon due to precision around x=0.
    return torch.sqrt(x)


def logsinh(x: torch.Tensor) -> torch.Tensor:
    # torch.log(sinh(x))
    # return x + torch.log(clamp(1. - torch.exp(-2. * x), min=eps)) - ln_2
    x_exp = x.unsqueeze(dim=-1)
    signs = torch.cat((torch.ones_like(x_exp), -torch.ones_like(x_exp)), dim=-1)
    value = torch.cat((torch.zeros_like(x_exp), -2. * x_exp), dim=-1)
    return x + logsumexp_signs(value, dim=-1, signs=signs) - ln_2


def logcosh(x: torch.Tensor) -> torch.Tensor:
    # torch.log(cosh(x))
    # return x + torch.log(clamp(1. + torch.exp(-2. * x), min=eps)) - ln_2
    x_exp = x.unsqueeze(dim=-1)
    value = torch.cat((torch.zeros_like(x_exp), -2. * x_exp), dim=-1)
    return x + torch.logsumexp(value, dim=-1) - ln_2


def logsumexp_signs(value: torch.Tensor, dim: int = 0, keepdim: bool = False,
                    signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    if signs is None:
        signs = torch.ones_like(value)
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(clamp(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim), min=eps))


def e_i(i: int, shape: Tuple[int, ...], **kwargs: Any) -> torch.Tensor:
    e = torch.zeros(shape, **kwargs)
    e[..., i] = 1
    return e


def expand_proj_dims(x: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(x.shape[:-1] + torch.Size([1])).to(x.device).to(x.dtype)
    return torch.cat((zeros, x), dim=-1)
