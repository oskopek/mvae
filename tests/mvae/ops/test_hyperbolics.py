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

import mt.mvae.ops.hyperbolics as H
from mt.mvae.ops.common import eps

from mt.mvae.ops.poincare import poincare_to_lorentz

np.random.seed(42)
random_nums = np.random.random_sample(100) * 100 + 1
test_eps = 5e-6
low_prec_test_eps = 1e-4
radius = torch.tensor(2., dtype=torch.get_default_dtype())


def is_in_hyp_space(x: torch.Tensor, eps: float = eps) -> torch.Tensor:
    prod = H.lorentz_product(x, x)
    prod += torch.ones_like(prod) * radius * radius
    return (torch.abs(prod) <= eps) & (x[..., 0] > 0)


def is_in_tangent_space(x: torch.Tensor, at_point: torch.Tensor, eps: float = eps) -> torch.Tensor:
    assert is_in_hyp_space(at_point)
    prod = H.lorentz_product(x, at_point)
    return prod.allclose(torch.zeros_like(prod), atol=eps)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return radius * H.acosh(-H.lorentz_product(x, y, **kwargs) / (radius**2))


def parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    coef = H.lorentz_product(dst, x, keepdim=True) / (radius**2 - H.lorentz_product(src, dst))
    return x + coef * (src + dst)


def inverse_parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return parallel_transport(x, dst, src)


def test_lorentz_product() -> None:
    assert H.lorentz_product(t([0, 0, 0]), t([3, 2, 1])) == t(0)
    assert H.lorentz_product(t([0, 0, 0]), t([0, 0, 0])) == t(0)
    assert H.lorentz_product(t([1, 2, 3]), t([3, 2, 1])) == t(4)
    assert H.lorentz_product(t([1, 2, 3]), t([0, 2, 1])) == t(7)
    assert H.lorentz_product(t([1, 2, 3]), t([0, 0, 0])) == t(0)


def test_lorentz_product_batch() -> None:
    x = t([[1, 2, 3], [1, 2, 3]])
    y = t([[3, 2, 1], [0, 2, 1]])
    assert H.lorentz_product(x, y).allclose(t([4, 7]))
    assert H.lorentz_product(x, y, keepdim=True).allclose(t([[4], [7]]))


def test_e_i() -> None:
    res = H.e_i(1, (3, 2), dtype=torch.int64)
    expected = t([[0, 1], [0, 1], [0, 1]])
    assert res.allclose(expected)


def test_mu_0() -> None:
    res = H.mu_0((3, 3), radius=radius, dtype=torch.int64)
    expected = radius * t([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    assert res.allclose(expected)


def test_is_in_hyp_space() -> None:
    assert is_in_hyp_space(radius * t([1., 0, 0]))
    assert not is_in_hyp_space(radius * t([0.1, 2, 3]))
    assert is_in_hyp_space(radius * t([2, 1, np.sqrt(2)]))
    assert not is_in_hyp_space(radius * t([0, 2, -np.sqrt(2)]))


def test_is_in_tangent_space() -> None:
    assert is_in_tangent_space(t([0., 2, 3]), radius * t([1., 0, 0]))
    assert not is_in_tangent_space(t([0.1, 2, 3]), radius * t([1., 0, 0]))
    # -0*2 + 2*1 - 2 = 0
    assert is_in_tangent_space(t([0, 2, -np.sqrt(2)]), radius * t([2, 1, np.sqrt(2)]), eps=test_eps)


def test_lorentz_norm() -> None:
    assert torch.isfinite(H.lorentz_norm(t([300., 0., 0.])))
    assert torch.isfinite(H.lorentz_norm(t([1., 0., 0.])))
    assert torch.isfinite(H.lorentz_norm(t([2., 1, np.sqrt(2)])))
    assert H.lorentz_norm(t([2., 1., 2.])).allclose(t(1.))


def test_lorentz_distance() -> None:
    mu0 = radius * t([1., 0, 0])
    mu = radius * t([2., 1., np.sqrt(2)])
    assert lorentz_distance(mu0, mu0).allclose(t(0.), atol=5e-4)
    assert lorentz_distance(mu, mu).allclose(t(0.), atol=5e-4)
    assert lorentz_distance(mu0, mu) == lorentz_distance(mu, mu0)


def test_parallel_transport() -> None:
    mu1 = radius * t([2., 1, np.sqrt(2)])
    mu2 = radius * t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])
    assert is_in_tangent_space(u, at_point=mu1, eps=test_eps)

    assert parallel_transport(u, src=mu1, dst=mu1).allclose(u, atol=5e-4)

    pt_u = parallel_transport(u, src=mu1, dst=mu2)
    assert is_in_tangent_space(pt_u, at_point=mu2)
    u_ = parallel_transport(pt_u, src=mu2, dst=mu1)
    assert u.allclose(u_, atol=5e-4)
    u_inv = inverse_parallel_transport(pt_u, src=mu1, dst=mu2)
    assert u.allclose(u_inv)


def test_parallel_transport_batch() -> None:
    mu1 = radius * t([2., 1, np.sqrt(2)])
    mu2 = radius * t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    U = torch.stack((u, u2), dim=0)
    res = parallel_transport(U, src=mu1, dst=mu2)
    U_ = inverse_parallel_transport(res, src=mu1, dst=mu2)
    assert U.allclose(U_)


def test_parallel_transport_mu0() -> None:
    mu0 = radius * t([1., 0, 0])
    mu2 = radius * t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])

    assert H.parallel_transport_mu0(u, dst=mu0, radius=radius).allclose(u)

    pt_u = H.parallel_transport_mu0(u, dst=mu2, radius=radius)
    assert parallel_transport(u, src=mu0, dst=mu2).allclose(pt_u)

    u_inv = H.inverse_parallel_transport_mu0(pt_u, src=mu2, radius=radius)
    assert u.allclose(u_inv)


def test_parallel_transport_mu0_batch() -> None:
    mu2 = radius * t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    U = torch.stack((u, u2), dim=0)
    res = H.parallel_transport_mu0(U, dst=mu2, radius=radius)
    U_ = H.inverse_parallel_transport_mu0(res, src=mu2, radius=radius)
    assert U.allclose(U_)


def test_exp_map() -> None:
    mu = radius * t([2., 1, np.sqrt(2)])
    u = t([0, 2, -np.sqrt(2)])

    assert is_in_tangent_space(u, at_point=mu, eps=test_eps)
    u_mapped = H.exp_map(u, at_point=mu, radius=radius)
    u_ = H.inverse_exp_map(u_mapped, at_point=mu, radius=radius)
    assert u.allclose(u_, atol=test_eps)

    assert lorentz_distance(mu, u_mapped).allclose(torch.norm(u, p=2))
    assert lorentz_distance(mu, u_mapped).allclose(H.lorentz_norm(u))


def test_exp_map_mu0() -> None:
    mu0 = H.mu_0((3,), radius=radius)
    u = t([0, 2, -np.sqrt(2)])

    assert is_in_tangent_space(u, at_point=mu0, eps=test_eps)
    u_mapped = H.exp_map(u, at_point=mu0, radius=radius)
    u_mu0_mapped = H.exp_map_mu0(u, radius=radius)
    assert u_mapped.allclose(u_mu0_mapped)
    u_ = H.inverse_exp_map(u_mapped, at_point=mu0, radius=radius)
    u_mu0 = H.inverse_exp_map_mu0(u_mu0_mapped, radius=radius)
    assert u.allclose(u_, atol=test_eps)
    assert u.allclose(u_mu0, atol=test_eps)

    assert lorentz_distance(mu0, u_mapped).allclose(torch.norm(u, p=2))
    assert lorentz_distance(mu0, u_mu0_mapped).allclose(torch.norm(u, p=2))
    assert lorentz_distance(mu0, u_mapped).allclose(H.lorentz_norm(u))
    assert lorentz_distance(mu0, u_mu0_mapped).allclose(H.lorentz_norm(u))


def test_exp_map_large() -> None:
    mu = radius * t([2., 1, np.sqrt(2)])
    u = 15 * t([0, 2, -np.sqrt(2)])
    assert is_in_tangent_space(u, at_point=mu, eps=test_eps)  # This should hold.
    u_mapped = H.exp_map(u, at_point=mu, radius=radius)
    u_ = H.inverse_exp_map(u_mapped, at_point=mu, radius=radius)
    assert u.allclose(u_, atol=low_prec_test_eps)


def test_exp_map_batch() -> None:
    mu = radius * t([2., 1, np.sqrt(2)])
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    U = torch.stack((u, u2), dim=0)
    U_mapped = H.exp_map(U, at_point=mu, radius=radius)
    U_ = H.inverse_exp_map(U_mapped, at_point=mu, radius=radius)
    assert U.allclose(U_, atol=1e-5)


def test_sample_projection() -> None:
    v = t([1., 2])
    expanded_v = t([0., 1, 2])
    assert not is_in_hyp_space(expanded_v)

    mu0 = radius * t([1., 0, 0])
    assert is_in_hyp_space(mu0)
    assert is_in_tangent_space(expanded_v, at_point=mu0)

    mu = radius * t([2., 1, np.sqrt(2)])
    assert is_in_hyp_space(mu)

    v_proj, (pt, _) = H.sample_projection_mu0(v, at_point=mu, radius=radius)
    assert is_in_tangent_space(pt, at_point=mu, eps=test_eps)
    assert is_in_hyp_space(v_proj, eps=low_prec_test_eps)

    _, v_ = H.inverse_sample_projection_mu0(v_proj, at_point=mu, radius=radius)
    expanded_v_ = torch.cat((t([0.]), v_), dim=-1)
    assert v.allclose(v_, atol=test_eps)
    assert expanded_v.allclose(expanded_v_, atol=test_eps)
    assert not is_in_hyp_space(expanded_v_)
    assert is_in_tangent_space(expanded_v_, at_point=mu0, eps=test_eps)


def test_poincare_to_lorentz_back() -> None:
    lorentz = radius * t([2., 1, np.sqrt(2)])
    poincare = H.lorentz_to_poincare(lorentz, radius=radius)
    lorentz_ = poincare_to_lorentz(poincare, radius=radius)
    assert lorentz.allclose(lorentz_)
    assert H.lorentz_product(lorentz_, lorentz_).allclose(-radius**2)
