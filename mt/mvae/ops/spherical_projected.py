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

from .manifold import RadiusManifold
from .common import sqrt
from . import spherical as S
from .poincare import pm

MIN_NORM = 1e-15


class StereographicallyProjectedSphere(RadiusManifold):

    def __init__(self, radius: Tensor) -> None:
        super().__init__(radius)

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        return exp_map_mu0(x, radius=self.radius)

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        return inverse_exp_map_mu0(x, radius=self.radius)

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        return parallel_transport_mu0(x, dst, radius=self.radius)

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        return inverse_parallel_transport_mu0(x, src, radius=self.radius)

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        return mu_0(shape, **kwargs)

    def sample_projection_mu0(self, x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return sample_projection_mu0(x, at_point, radius=self.radius)

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        return inverse_sample_projection_mu0(x_proj, at_point, radius=self.radius)

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        b_mu = mu
        # b_std = std
        if len(z.shape) > 2:
            b_mu = mu.unsqueeze(dim=0).repeat((z.shape[0], 1, 1))
            # b_std = std.unsqueeze(dim=0).repeat((z.shape[0], 1, 1))

        # TODO-LATER: Derive this for D just like PVAE did for P. Actually, theirs doesn't work...
        # c = _c(self.radius)
        # d = mu.shape[-1]
        # dpc = spherical_projected_gyro_distance(b_mu, z, K=c)
        # assert torch.isfinite(dpc).all()
        # r = dpc / self.radius
        # log_r = torch.log(r)
        # log_sin_term = torch.log(torch.abs(torch.sin(r)))
        # assert torch.isfinite(log_r).all()
        # assert torch.isfinite(log_sin_term).all()
        #
        # n_logprob = data[-1]
        # # the value below is the whole logprob, not logdet, from the PVAE paper.. we need to subtract
        # # the original gaussian logprob from it
        # log_prob = -d * ln_2pi - d / 2. * torch.log(b_std) - dpc ** 2 / (2 * b_std ** 2) + (d - 1) * (
        #         log_r - log_sin_term)
        # log_prob = log_prob.sum(dim=-1)
        #
        # logdet_partial = -log_prob + n_logprob

        z_sphere = projected_to_spherical(z, self.radius)
        mu_sphere = projected_to_spherical(b_mu, self.radius)
        u, v = S.inverse_sample_projection_mu0(z_sphere, mu_sphere, radius=self.radius)
        logdet_partial = S._logdet(u, self.radius)
        assert torch.isfinite(logdet_partial).all()
        return logdet_partial


def spherical_projected_distance(x: Tensor, y: Tensor, K: Tensor, **kwargs: Any) -> Tensor:
    diff = x - y
    normxmy2 = torch.sum(diff * diff, dim=-1, keepdim=True)
    normx2 = torch.sum(x * x, dim=-1, keepdim=True)
    normy2 = torch.sum(y * y, dim=-1, keepdim=True)
    dist = 1. / sqrt(K) * torch.acos(torch.clamp(1 - 2 * K * normxmy2 / ((1 + K * normx2) * (1 + K * normy2)), max=1.0))
    assert torch.isfinite(dist).all()
    return dist


def spherical_projected_gyro_distance(x: Tensor, y: Tensor, K: Tensor, **kwargs: Any) -> Tensor:
    sqrt_K = sqrt(K)
    sm = mob_add(-x, y, K)
    normxy = torch.norm(sm, p=2, dim=-1, keepdim=True)
    return 2. / sqrt_K * torch.atan(sqrt_K * normxy)


def mob_add(x: Tensor, y: Tensor, K: Tensor) -> Tensor:
    # prod = torch.sum(x * y, dim=-1, keepdim=True)
    # normx2 = torch.sum(x * x, dim=-1, keepdim=True)
    # normy2 = torch.sum(y * y, dim=-1, keepdim=True)
    # denom = 1 - 2 * K * prod + K * K * normx2 * normy2
    # return ((1 - 2 * K * prod - K * normy2) * x + (1 + K * normx2) * y) / denom.clamp(min=MIN_NORM)
    return pm.mobius_add(x, y, c=-K)


def _c(radius: Tensor) -> Tensor:
    return 1 / radius**2


def mu_0(shape: Tuple[int, ...], **kwargs: Any) -> Tensor:
    return torch.zeros(shape, **kwargs)


def lambda_x_c(x: Tensor, c: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    return 2 / (1 + c * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp(min=MIN_NORM)


def lambda_x(x: Tensor, radius: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    return lambda_x_c(x, c=_c(radius), dim=dim, keepdim=keepdim)


def gyration(u: Tensor, v: Tensor, w: Tensor, c: Tensor) -> Tensor:
    # mupv = -mob_add(u, v, c)
    # vpw = mob_add(v, w, c)
    # upvpw = mob_add(u, vpw, c)
    # return mob_add(mupv, upvpw, c)
    return pm.gyration(u, v, w, c=-c)


def parallel_transport_mu0(x: Tensor, dst: Tensor, radius: Tensor) -> Tensor:
    return (2 / lambda_x(dst, radius)) * x


def inverse_parallel_transport_mu0(x: Tensor, src: Tensor, radius: Tensor) -> Tensor:
    return (lambda_x(src, radius) / 2) * x


def exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    r = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=MIN_NORM) / radius
    c = _c(radius)
    arg = r * lambda_x_c(at_point, c) / 2
    rhs = torch.tan(arg) * x / r
    assert torch.isfinite(rhs).all()
    return mob_add(at_point, rhs, c)


def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    r = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=MIN_NORM) / radius
    rhs = torch.tan(r) * x / r
    assert torch.isfinite(rhs).all()
    return rhs


def inverse_exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    c = _c(radius)
    mxpy = mob_add(-at_point, x, c)
    nmxpy = torch.norm(mxpy, p=2, dim=-1, keepdim=True).clamp(min=MIN_NORM) / radius
    normalized_mxpy = mxpy / nmxpy
    return 2 / lambda_x_c(at_point, c=c) * torch.atan(nmxpy) * normalized_mxpy


def inverse_exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    nx = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=MIN_NORM) / radius
    normx = x / nx
    return torch.atan(nx) * normx


def sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    v_ = x / lambda_x(at_point, radius=radius, dim=-1, keepdim=True)  # Corresponds to PT divided by 2.
    x_proj = exp_map(v_, at_point=at_point, radius=radius)
    return x_proj, (v_, x)


def inverse_sample_projection_mu0(x_proj: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tensor]:
    c = _c(radius)
    v_ = inverse_exp_map(x_proj, at_point=at_point, radius=radius)
    x = v_ * lambda_x_c(at_point, c=c, dim=-1, keepdim=True)  # Corresponds to PT multiplied by 2.
    return v_, x


def projected_to_spherical(y: Tensor, radius: Tensor) -> Tensor:
    yn2 = torch.norm(y, p=2, dim=-1, keepdim=True)**2
    r2 = radius * radius
    res = torch.cat((radius * (r2 - yn2), 2 * r2 * y), dim=-1) / (yn2 + r2)
    assert torch.isfinite(res).all()
    return res
