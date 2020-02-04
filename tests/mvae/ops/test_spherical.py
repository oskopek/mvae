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

from typing import Any

import numpy as np
import torch
from torch import tensor as t
import torch.nn.functional as F

import mt.mvae.ops.spherical as S
from mt.mvae.ops.common import eps

np.random.seed(42)
random_nums = np.random.random_sample(100) * 100 + 1
test_eps = 5e-6
low_prec_test_eps = 1e-4
radius = torch.tensor(2., dtype=torch.float32)


def is_in_hyp_space(x: torch.Tensor, radius: torch.Tensor, eps: float = eps) -> torch.Tensor:
    prod = torch.sum(x * x, dim=-1, keepdim=True)
    return prod.allclose(torch.ones_like(prod) * radius * radius, atol=eps)


def is_in_tangent_space(x: torch.Tensor, at_point: torch.Tensor, radius: torch.Tensor,
                        eps: float = eps) -> torch.Tensor:
    assert is_in_hyp_space(at_point, radius=radius)
    prod = torch.sum(x * at_point, dim=-1, keepdim=True)
    return prod.allclose(torch.zeros_like(prod), atol=eps)


def spherical_distance(x: torch.Tensor, y: torch.Tensor, radius: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    ndot = torch.sum(x * y, dim=-1, keepdim=True) / radius**2
    acos = torch.acos(torch.clamp(ndot, min=-1., max=1.))
    return radius * acos


def parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    xTy = torch.sum(src * dst, dim=-1, keepdim=True)
    return x - torch.sum(dst * x, dim=-1, keepdim=True) / (radius**2 + xTy) * (src + dst)


def inverse_parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                               radius: torch.Tensor) -> torch.Tensor:
    return parallel_transport(x, dst, src, radius=radius)


def test_mu_0() -> None:
    res = S.mu_0((3, 3), radius=radius, dtype=torch.float32)
    expected = t([[radius, 0, 0], [radius, 0, 0], [radius, 0, 0]])
    assert res.allclose(expected)


def test_is_in_hyp_space() -> None:
    assert is_in_hyp_space(radius * t([1., 0, 0]), radius)
    assert not is_in_hyp_space(radius * t([0.1, 2, 3]), radius)
    assert not is_in_hyp_space(radius * t([1, 1, np.sqrt(2)]), radius)
    assert not is_in_hyp_space(radius * t([1, 1, np.sqrt(2)]), radius)
    assert not is_in_hyp_space(radius * t([0, 2, -np.sqrt(2)]), radius)


def test_is_in_tangent_space() -> None:
    assert is_in_tangent_space(t([0., 2, 3]), radius * t([1., 0, 0]), radius)
    assert not is_in_tangent_space(t([0.1, 2, 3]), radius * t([1., 0, 0]), radius)
    # 0*2 + 2*1 - 2 = 0
    assert is_in_tangent_space(t([0, 2, -np.sqrt(2)]), _normalize(t([2, 1, np.sqrt(2)])), radius, eps=test_eps)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1) * radius


def test_spherical_distance() -> None:
    mu0 = S.mu_0((3,), radius)
    mu = _normalize(t([2., 1., np.sqrt(2)]))
    assert spherical_distance(mu0, mu0, radius).allclose(t(0.), atol=5e-4)
    dist = spherical_distance(mu, mu, radius)
    assert dist.allclose(t(0.), atol=5e-4)
    assert spherical_distance(mu0, mu, radius) == spherical_distance(mu, mu0, radius)


def test_parallel_transport() -> None:
    mu1 = _normalize(t([2., 1, np.sqrt(2)]))
    mu2 = _normalize(t([np.sqrt(5), 1, np.sqrt(3)]))
    assert is_in_hyp_space(mu1, radius)
    assert is_in_hyp_space(mu2, radius)

    u = t([0, 2, -np.sqrt(2)])
    assert is_in_tangent_space(u, at_point=mu1, radius=radius, eps=test_eps)

    assert parallel_transport(u, src=mu1, dst=mu1, radius=radius).allclose(u, atol=5e-4)

    pt_u = parallel_transport(u, src=mu1, dst=mu2, radius=radius)
    assert is_in_tangent_space(pt_u, at_point=mu2, radius=radius, eps=test_eps)
    u_ = parallel_transport(pt_u, src=mu2, dst=mu1, radius=radius)
    assert u.allclose(u_, atol=5e-4)
    u_inv = inverse_parallel_transport(pt_u, src=mu1, dst=mu2, radius=radius)
    assert u.allclose(u_inv)


def test_parallel_transport_batch() -> None:
    mu1 = _normalize(t([2., 1, np.sqrt(2)]))
    mu2 = _normalize(t([np.sqrt(5), 1, np.sqrt(3)]))
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])
    assert is_in_tangent_space(u, mu1, radius, eps=test_eps)
    assert is_in_tangent_space(u2, mu1, radius, eps=test_eps)

    U = torch.stack((u, u2), dim=0)
    res = parallel_transport(U, src=mu1, dst=mu2, radius=radius)
    U_ = inverse_parallel_transport(res, src=mu1, dst=mu2, radius=radius)
    assert U.allclose(U_)


def test_parallel_transport_mu0() -> None:
    mu0 = S.mu_0((3,), radius=radius)
    mu2 = _normalize(t([np.sqrt(5), 1, np.sqrt(3)]))
    u = t([0, 2, -np.sqrt(2)])
    assert is_in_tangent_space(u, mu0, radius, eps=test_eps)

    assert S.parallel_transport_mu0(u, dst=mu0, radius=radius).allclose(u)
    assert parallel_transport(u, mu0, mu0, radius).allclose(u)

    pt_u = S.parallel_transport_mu0(u, dst=mu2, radius=radius)
    pt_u_normal = parallel_transport(u, src=mu0, dst=mu2, radius=radius)
    assert pt_u_normal.allclose(pt_u)

    u_inv = S.inverse_parallel_transport_mu0(pt_u, src=mu2, radius=radius)
    u_inv_pt = inverse_parallel_transport(pt_u, src=mu0, dst=mu2, radius=radius)
    assert u.allclose(u_inv_pt)
    assert u_inv.allclose(u_inv_pt)
    assert u.allclose(u_inv)


def test_parallel_transport_mu0_batch() -> None:
    mu2 = _normalize(t([np.sqrt(5), 1, np.sqrt(3)]))
    assert is_in_hyp_space(mu2, radius)
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])
    assert is_in_tangent_space(u, S.mu_0((2, 3), radius=radius), radius, eps=test_eps)
    assert is_in_tangent_space(u2, S.mu_0((2, 3), radius=radius), radius, eps=test_eps)

    U = torch.stack((u, u2), dim=0)
    res = S.parallel_transport_mu0(U, dst=mu2, radius=radius)
    U_ = S.inverse_parallel_transport_mu0(res, src=mu2, radius=radius)
    assert U.allclose(U_)


def test_exp_map() -> None:
    mu = _normalize(t([2., 1, np.sqrt(2)]))
    u = 2 * t([0, 2, -np.sqrt(2)])

    assert is_in_tangent_space(u, at_point=mu, radius=radius, eps=test_eps)
    u_mapped = S.exp_map(u, at_point=mu, radius=radius)
    assert is_in_hyp_space(u_mapped, radius=radius, eps=test_eps)
    u_ = S.inverse_exp_map(u_mapped, at_point=mu, radius=radius)
    assert u.allclose(u_, atol=test_eps)

    assert spherical_distance(mu, u_mapped, radius).allclose(torch.norm(u, p=2))


def test_exp_map_mu0() -> None:
    mu0 = S.mu_0((3,), radius=radius)
    assert is_in_hyp_space(mu0, radius)
    u = t([0, 2, -np.sqrt(2)])
    assert is_in_tangent_space(u, mu0, radius)

    assert is_in_tangent_space(u, at_point=mu0, radius=radius, eps=test_eps)
    u_mapped = S.exp_map(u, at_point=mu0, radius=radius)
    u_mu0_mapped = S.exp_map_mu0(u, radius=radius)
    assert u_mapped.allclose(u_mu0_mapped)
    u_ = S.inverse_exp_map(u_mapped, at_point=mu0, radius=radius)
    u_mu0 = S.inverse_exp_map_mu0(u_mu0_mapped, radius=radius)
    assert u.allclose(u_, atol=test_eps)
    assert u.allclose(u_mu0, atol=test_eps)

    assert spherical_distance(mu0, u_mapped, radius).allclose(torch.norm(u, p=2))
    assert spherical_distance(mu0, u_mu0_mapped, radius).allclose(torch.norm(u, p=2))


def test_exp_map_large() -> None:
    mu = _normalize(t([2., 1, np.sqrt(2)]))
    u = 2 * t([0, 2, -np.sqrt(2)])
    assert is_in_tangent_space(u, at_point=mu, radius=radius, eps=test_eps)  # This should hold.
    u_mapped = S.exp_map(u, at_point=mu, radius=radius)
    assert is_in_hyp_space(u_mapped, radius, eps=test_eps)
    u_ = S.inverse_exp_map(u_mapped, at_point=mu, radius=radius)
    assert is_in_tangent_space(u_, at_point=mu, radius=radius, eps=test_eps)  # This should hold.
    assert u.allclose(u_, atol=low_prec_test_eps)


def test_exp_map_batch() -> None:
    mu = _normalize(t([2., 1, np.sqrt(2)]))
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])
    assert is_in_tangent_space(u, at_point=mu, radius=radius, eps=test_eps)
    assert is_in_tangent_space(u2, at_point=mu, radius=radius, eps=test_eps)

    U = torch.stack((u, u2), dim=0)
    U_mapped = S.exp_map(U, at_point=mu, radius=radius)
    U_ = S.inverse_exp_map(U_mapped, at_point=mu, radius=radius)
    assert U.allclose(U_, atol=1e-5)


def test_sample_projection() -> None:
    v = t([1., 2])
    expanded_v = t([0., 1, 2])
    assert not is_in_hyp_space(expanded_v, radius)

    mu0 = S.mu_0((3,), radius=radius)
    assert is_in_hyp_space(mu0, radius)
    assert is_in_tangent_space(expanded_v, at_point=mu0, radius=radius)

    mu = _normalize(t([2., 1, np.sqrt(2)]))
    assert is_in_hyp_space(mu, radius)

    v_proj, (pt, _) = S.sample_projection_mu0(v, at_point=mu, radius=radius)
    assert is_in_tangent_space(pt, at_point=mu, radius=radius, eps=test_eps)
    assert is_in_hyp_space(v_proj, radius=radius, eps=low_prec_test_eps)

    _, v_ = S.inverse_sample_projection_mu0(v_proj, at_point=mu, radius=radius)
    expanded_v_ = torch.cat((t([0.]), v_), dim=-1)
    assert v.allclose(v_, atol=test_eps)
    assert expanded_v.allclose(expanded_v_, atol=test_eps)
    assert not is_in_hyp_space(expanded_v_, radius)
    assert is_in_tangent_space(expanded_v_, at_point=mu0, radius=radius, eps=test_eps)

    radius2 = 1
    x = torch.tensor([[0.4106, -0.9035, 0.1229], [-0.8405, 0.5385, -0.0600]])
    at_point = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    unmapped = S.inverse_exp_map(x, at_point=at_point, radius=radius2)
    unpt = S.inverse_parallel_transport_mu0(unmapped, src=at_point, radius=radius2)
    unpt2 = parallel_transport(unmapped, src=at_point, dst=mu0, radius=radius2)
    assert unpt.allclose(unpt2)
    assert torch.isfinite(unpt).all()
