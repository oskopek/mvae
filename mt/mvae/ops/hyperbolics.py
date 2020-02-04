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

from typing import Any, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .common import acosh, cosh, sinh, sqrt, logsinh, e_i, expand_proj_dims
from .manifold import RadiusManifold


class Hyperboloid(RadiusManifold):

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        return exp_map_mu0(expand_proj_dims(x), radius=self.radius)

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        return inverse_exp_map_mu0(x, radius=self.radius)

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        return parallel_transport_mu0(x, dst, radius=self.radius)

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        return inverse_parallel_transport_mu0(x, src, radius=self.radius)

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        return mu_0(shape, radius=self.radius, **kwargs)

    def sample_projection_mu0(self, x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return sample_projection_mu0(x, at_point, radius=self.radius)

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        return inverse_sample_projection_mu0(x_proj, at_point, radius=self.radius)

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        u = data[0]
        return _logdet(u, self.radius)

    @property
    def curvature(self) -> Tensor:
        return -super().curvature


def _logdet(u: Tensor, radius: Tensor) -> Tensor:
    # det [(\partial / \partial v) proj_{\mu}(v)] = (Rsinh(r) / r)^(n-1)
    r = lorentz_norm(u, dim=-1) / radius
    n = u.shape[-1] - 1

    logdet_partial = (n - 1) * (torch.log(radius) + logsinh(r) - torch.log(r))
    assert torch.isfinite(logdet_partial).all()
    return logdet_partial


def mu_0(shape: Tuple[int, ...], radius: Tensor, **kwargs: Any) -> Tensor:
    return e_i(i=0, shape=shape, **kwargs) * radius


def lorentz_product(x: Tensor, y: Tensor, keepdim: bool = False, dim: int = -1) -> Tensor:
    m = x * y
    if keepdim:
        ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
    else:
        ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
    return ret


def lorentz_norm(x: Tensor, **kwargs: Any) -> Tensor:
    product = lorentz_product(x, x, **kwargs)
    ret = sqrt(product)
    return ret


def parallel_transport_mu0(x: Tensor, dst: Tensor, radius: Tensor) -> Tensor:
    # PT_{mu0 -> dst}(x) = x + <dst, x>_L / (R^2 - <mu0, dst>_L) * (mu0+dst)
    denom = radius * (radius + dst[..., 0:1])  # lorentz_product(mu0, dst, keepdim=True) which is -dst[0]*radius
    lp = lorentz_product(dst, x, keepdim=True)
    coef = lp / denom
    right = torch.cat((dst[..., 0:1] + radius, dst[..., 1:]), dim=-1)  # mu0 + dst
    return x + coef * right


def inverse_parallel_transport_mu0(x: Tensor, src: Tensor, radius: Tensor) -> Tensor:
    # PT_{src -> mu0}(x) = x + <mu0, x>_L / (R^2 - <src, mu0>_L) * (src+mu0)
    denom = (radius + src[..., 0:1])  # lorentz_product(src, mu0, keepdim=True) which is -src[0]*radius
    lp = -x[..., 0:1]  # lorentz_product(mu0, x, keepdim=True) which is -x[0]*radius
    # coef = (lp * radius) / (radius * denom)
    coef = lp / denom
    right = torch.cat((src[..., 0:1] + radius, src[..., 1:]), dim=-1)  # mu0 + src
    return x + coef * right


def exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    x_norm = lorentz_norm(x, keepdim=True) / radius
    x_normed = x / x_norm
    ret = cosh(x_norm) * at_point + sinh(x_norm) * x_normed
    assert torch.isfinite(ret).all()
    return ret


def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    assert x[..., 0].allclose(torch.zeros_like(x[..., 0]))
    x = x[..., 1:]
    x_norm = torch.norm(x, p=2, keepdim=True, dim=-1) / radius
    x_normed = F.normalize(x, p=2, dim=-1) * radius
    ret = torch.cat((cosh(x_norm) * radius, sinh(x_norm) * x_normed), dim=-1)
    assert torch.isfinite(ret).all()
    return ret


def inverse_exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    alpha = -lorentz_product(at_point, x, keepdim=True) / (radius**2)
    coef = acosh(alpha) / sqrt(alpha**2 - 1)
    ret = coef * (x - alpha * at_point)
    return ret


def inverse_exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    alpha = x[..., 0:1] / radius  # -lorentz_product(x, mu0, keepdim=True) / R^2 .. -<x, mu0>_L = x[0] * R
    coef = acosh(alpha) / sqrt(alpha**2 - 1.)
    diff = torch.cat((x[..., 0:1] - alpha * radius, x[..., 1:]), dim=-1)
    return coef * diff


def sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x_expanded = expand_proj_dims(x)
    pt = parallel_transport_mu0(x_expanded, dst=at_point, radius=radius)
    x_proj = exp_map(pt, at_point=at_point, radius=radius)
    return x_proj, (pt, x)


def inverse_sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tensor]:
    unmapped = inverse_exp_map(x, at_point=at_point, radius=radius)
    unpt = inverse_parallel_transport_mu0(unmapped, src=at_point, radius=radius)
    return unmapped, unpt[..., 1:]


def lorentz_to_poincare(x: Tensor, radius: Tensor) -> Tensor:
    return radius * x[..., 1:] / (radius + x[..., 0:1])
