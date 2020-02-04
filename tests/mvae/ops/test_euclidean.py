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

import mt.mvae.ops.euclidean as E
from mt.mvae.ops.common import eps

np.random.seed(42)
random_nums = np.random.random_sample(100) * 100 + 1
test_eps = 5e-6
low_prec_test_eps = 1e-4


def is_in_hyp_space(x: torch.Tensor, eps: float = eps) -> torch.Tensor:
    return (x == x).all()


def is_in_tangent_space(x: torch.Tensor, at_point: torch.Tensor, eps: float = eps) -> torch.Tensor:
    assert is_in_hyp_space(at_point)
    prod = x.dot(at_point)
    return prod.allclose(torch.zeros_like(prod), atol=eps)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return 2 * torch.norm(x - y, dim=-1, p=2, keepdim=True)


def parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return x


def inverse_parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return parallel_transport(x, dst, src)


def test_mu_0() -> None:
    res = E.mu_0((3, 3), dtype=torch.int64)
    expected = t([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert res.allclose(expected)


def test_is_in_hyp_space() -> None:
    assert is_in_hyp_space(t([1., 0, 0]))
    assert is_in_hyp_space(t([0.1, 2, 3]))
    assert is_in_hyp_space(t([1, 1, np.sqrt(2)]))
    assert is_in_hyp_space(t([1, 1, np.sqrt(2)]))
    assert is_in_hyp_space(t([0, 2, -np.sqrt(2)]))


def test_is_in_tangent_space() -> None:
    assert is_in_tangent_space(t([0., 2, 3]), t([1., 0, 0]))
    assert not is_in_tangent_space(t([0.1, 2, 3]), t([1., 0, 0]))
    # -0*2 + 2*1 - 2 = 0
    assert is_in_tangent_space(t([0, 2, -np.sqrt(2)]), t([2, 1, np.sqrt(2)]), eps=test_eps)


def test_lorentz_distance() -> None:
    mu0 = t([1., 0, 0])
    mu = t([2., 1., np.sqrt(2)])
    assert euclidean_distance(mu0, mu0).allclose(t(0.), atol=5e-4)
    assert euclidean_distance(mu, mu).allclose(t(0.), atol=5e-4)
    assert euclidean_distance(mu0, mu) == euclidean_distance(mu, mu0)


def test_parallel_transport() -> None:
    mu1 = t([2., 1, np.sqrt(2)]).double()
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)]).double()
    assert is_in_hyp_space(mu1)
    assert is_in_hyp_space(mu2)

    u = t([0, 2, -np.sqrt(2)]).double()
    assert is_in_tangent_space(u, at_point=mu1, eps=test_eps)

    assert parallel_transport(u, src=mu1, dst=mu1).allclose(u, atol=5e-4)

    pt_u = parallel_transport(u, src=mu1, dst=mu2)
    u_ = parallel_transport(pt_u, src=mu2, dst=mu1)
    assert u.allclose(u_, atol=5e-4)
    u_inv = inverse_parallel_transport(pt_u, src=mu1, dst=mu2)
    assert u.allclose(u_inv)


def test_parallel_transport_batch() -> None:
    mu1 = t([2., 1, np.sqrt(2)])
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    U = torch.stack((u, u2), dim=0)
    res = parallel_transport(U, src=mu1, dst=mu2)
    U_ = inverse_parallel_transport(res, src=mu1, dst=mu2)
    assert U.allclose(U_)


def test_parallel_transport_mu0() -> None:
    mu0 = t([0., 0, 0])
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])

    assert E.parallel_transport_mu0(u, dst=mu0).allclose(u)

    pt_u = E.parallel_transport_mu0(u, dst=mu2)
    assert parallel_transport(u, src=mu0, dst=mu2).allclose(pt_u)

    u_inv = E.inverse_parallel_transport_mu0(pt_u, src=mu2)
    assert u.allclose(u_inv)


def test_parallel_transport_mu0_batch() -> None:
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    U = torch.stack((u, u2), dim=0)
    res = E.parallel_transport_mu0(U, dst=mu2)
    U_ = E.inverse_parallel_transport_mu0(res, src=mu2)
    assert U.allclose(U_)


def test_exp_map() -> None:
    mu = t([1., 0.5, np.sqrt(2) / 2])
    u = t([0, 2, -np.sqrt(2)])

    assert is_in_tangent_space(u, at_point=mu, eps=test_eps)
    u_mapped = E.exp_map(u, at_point=mu)
    u_ = E.inverse_exp_map(u_mapped, at_point=mu)
    assert u.allclose(u_, atol=test_eps)

    assert euclidean_distance(mu, u_mapped).allclose(torch.norm(u, p=2))


def test_exp_map_mu0() -> None:
    mu0 = E.mu_0((3,))
    u = t([0, 2, -np.sqrt(2)])

    assert is_in_tangent_space(u, at_point=mu0, eps=test_eps)
    u_mapped = E.exp_map(u, at_point=mu0)
    u_mu0_mapped = E.exp_map_mu0(u)
    assert u_mapped.allclose(u_mu0_mapped)
    u_ = E.inverse_exp_map(u_mapped, at_point=mu0)
    u_mu0 = E.inverse_exp_map_mu0(u_mu0_mapped)
    assert u.allclose(u_, atol=test_eps)
    assert u.allclose(u_mu0, atol=test_eps)

    assert euclidean_distance(mu0, u_mapped).allclose(torch.norm(u, p=2))
    assert euclidean_distance(mu0, u_mu0_mapped).allclose(torch.norm(u, p=2))


def test_exp_map_large() -> None:
    mu = t([2., 1, np.sqrt(2)])
    mu = mu.double()
    u = 6.75 * t([0, 2, -np.sqrt(2)])
    u = u.double()
    assert is_in_tangent_space(u, at_point=mu, eps=test_eps)  # This should hold.
    u_mapped = E.exp_map(u, at_point=mu)
    u_ = E.inverse_exp_map(u_mapped, at_point=mu)
    assert u.allclose(u_, atol=low_prec_test_eps)


def test_exp_map_batch() -> None:
    mu = t([2., 1, np.sqrt(2)]).double()
    u = t([0, 2, -np.sqrt(2)]).double()
    u2 = t([0, 4, -2 * np.sqrt(2)]).double()
    assert is_in_tangent_space(u, at_point=mu, eps=test_eps)
    assert is_in_tangent_space(u2, at_point=mu, eps=test_eps)

    U = torch.stack((u, u2), dim=0)
    U_mapped = E.exp_map(U, at_point=mu)
    U_ = E.inverse_exp_map(U_mapped, at_point=mu)
    assert U.allclose(U_, atol=1e-5)


def test_sample_projection() -> None:
    v = t([0., 1, 2])

    mu0 = t([0., 0, 0])
    assert is_in_hyp_space(mu0)
    assert is_in_tangent_space(v, at_point=mu0)

    mu = t([2., 1, np.sqrt(2)])
    assert is_in_hyp_space(mu)

    v_proj, _ = E.sample_projection_mu0(v, at_point=mu)
    assert is_in_hyp_space(v_proj, eps=low_prec_test_eps)

    v_, _ = E.inverse_sample_projection_mu0(v_proj, at_point=mu)
    assert v.allclose(v_, atol=test_eps)
    assert is_in_tangent_space(v_, at_point=mu0, eps=test_eps)
