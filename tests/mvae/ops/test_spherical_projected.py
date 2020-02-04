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

import mt.mvae.ops.spherical as S
import mt.mvae.ops.spherical_projected as SP
from mt.mvae.ops.poincare import pm
from mt.mvae.ops.common import eps

import tests.mvae.ops.test_spherical as TS

np.random.seed(42)
random_nums = np.random.random_sample(100) * 100 + 1
test_eps = 5e-6
radius = torch.tensor(2., dtype=torch.float64)


def spherical_projected_distance_backprojection(x: torch.Tensor,
                                                y: torch.Tensor,
                                                radius: torch.Tensor = radius,
                                                **kwargs: Any) -> torch.Tensor:
    return TS.spherical_distance(SP.projected_to_spherical(x, radius), SP.projected_to_spherical(y, radius), radius)


def parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    c = SP._c(radius)
    return (SP.lambda_x_c(src, c) / SP.lambda_x_c(dst, c)) * SP.gyration(dst, -src, x, c)


def inverse_parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                               radius: torch.Tensor) -> torch.Tensor:
    return parallel_transport(x, dst, src, radius)


def is_in_hyp_space(x: torch.Tensor, eps: float = eps, radius: torch.Tensor = radius) -> torch.Tensor:
    return torch.ones_like(x, dtype=torch.uint8).all()


def is_in_tangent_space(x: torch.Tensor, at_point: torch.Tensor, eps: float = eps,
                        radius: torch.Tensor = radius) -> torch.Tensor:
    # TODO-LATER: This is most likely wrong, doesn't matter for VAE though, just for tests.
    assert is_in_hyp_space(at_point, radius=radius, eps=eps)
    prod = x.dot(at_point)
    return prod.allclose(torch.zeros_like(prod), atol=eps)


def test_mu_0() -> None:
    res = SP.mu_0((3, 3), dtype=torch.int64)
    expected = t([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert res.allclose(expected)


def test_is_in_hyp_space() -> None:
    assert is_in_hyp_space(radius * t([1., 0, 0]))
    assert is_in_hyp_space(radius * t([0.1, 2, 3]))
    assert is_in_hyp_space(radius * t([1, 1, np.sqrt(2)]))
    assert is_in_hyp_space(radius * t([1, 1, np.sqrt(2)]))
    assert is_in_hyp_space(radius * t([0, 2, -np.sqrt(2)]))


def test_is_in_tangent_space() -> None:
    # assert is_in_tangent_space(t([0., 2, 3]), radius * t([1., 0, 0]))
    # assert is_in_tangent_space(t([0.1, 2, 3]), radius * t([1., 0, 0]))
    # # -0*2 + 2*1 - 2 = 0
    # assert is_in_tangent_space(t([0, 2, -np.sqrt(2)]), t([2, 1, np.sqrt(2)]), eps=test_eps)
    pass


def test_spherical_projected_distance_backproj() -> None:
    mu0 = t([0., 0, 0])
    mu = t([2., 1., np.sqrt(2)])
    assert spherical_projected_distance_backprojection(mu0, mu0, radius).allclose(t(0.), atol=5e-4)
    assert spherical_projected_distance_backprojection(mu, mu, radius).allclose(t(0.), atol=5e-4)
    assert spherical_projected_distance_backprojection(mu0, mu, radius) == spherical_projected_distance_backprojection(
        mu, mu0, radius)


def test_spherical_projected_distance_normal() -> None:
    mu0 = t([0., 0, 0])
    mu = t([2., 1., np.sqrt(2)])
    K = SP._c(radius)
    assert SP.spherical_projected_distance(mu0, mu0, K).allclose(t(0.), atol=5e-4)
    assert SP.spherical_projected_distance(mu, mu, K).allclose(t(0.), atol=5e-4)
    assert SP.spherical_projected_distance(mu0, mu, K) == SP.spherical_projected_distance(mu, mu0, K)


def test_spherical_projected_distance_gyr() -> None:
    mu0 = t([0., 0, 0])
    mu = t([2., 1., np.sqrt(2)])
    K = SP._c(radius)
    assert SP.spherical_projected_gyro_distance(mu0, mu0, K).allclose(t(0.), atol=5e-4)
    assert SP.spherical_projected_gyro_distance(mu, mu, K).allclose(t(0.), atol=5e-4)
    assert SP.spherical_projected_gyro_distance(mu0, mu, K) == SP.spherical_projected_gyro_distance(mu, mu0, K)


def test_spherical_projected_distance_all() -> None:
    mu0 = t([0., 0, 0])
    mu = t([2., 1., np.sqrt(2)])
    K = SP._c(radius)
    gyr_dist = SP.spherical_projected_gyro_distance(mu0, mu, K)
    assert gyr_dist.allclose(SP.spherical_projected_distance(mu0, mu, K))
    backproj_dist = spherical_projected_distance_backprojection(mu0, mu, radius)
    assert gyr_dist.allclose(backproj_dist)


def test_mob_add() -> None:
    mu1 = t([2., 1, np.sqrt(2)]).double()
    mu2 = t([2., 0.6, np.sqrt(3)]).double()
    assert SP.mob_add(mu1, -mu1, 1.).allclose(torch.zeros_like(mu1))
    assert SP.mob_add(mu1, -mu1, 0.1).allclose(torch.zeros_like(mu1))
    assert SP.mob_add(mu1, -mu1, 10).allclose(torch.zeros_like(mu1))
    assert SP.mob_add(mu1, mu2, 1).allclose(pm.mobius_add(mu1, mu2, c=-1))


def test_gyration() -> None:
    mu1 = t([2., 1, np.sqrt(2)]).double()
    mu2 = t([2., 0.6, np.sqrt(3)]).double()
    u = t([1., 1., 1.]).double()
    assert SP.gyration(mu1, -mu1, u, 1.).allclose(u)
    assert SP.gyration(mu1, -mu1, u, 0.1).allclose(u)
    assert SP.gyration(mu1, -mu1, u, 10).allclose(u)
    spr = SP.gyration(mu1, mu2, u, c=1)
    poi = pm.gyration(mu1, mu2, u, c=-1)
    assert spr.allclose(poi)


def test_parallel_transport() -> None:
    mu1 = t([2., 1, np.sqrt(2)]).double()
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)]).double()
    assert is_in_hyp_space(mu1)
    assert is_in_hyp_space(mu2)

    u = t([1., 1., 1.]).double()
    # assert is_in_tangent_space(u, at_point=mu1, eps=test_eps)

    assert parallel_transport(u, src=mu1, dst=mu1, radius=radius).allclose(u, atol=test_eps)

    pt_u = parallel_transport(u, src=mu1, dst=mu2, radius=radius)
    # assert is_in_tangent_space(pt_u, at_point=mu2, eps=test_eps)

    u_ = parallel_transport(pt_u, src=mu2, dst=mu1, radius=radius)
    u_inv = inverse_parallel_transport(pt_u, src=mu1, dst=mu2, radius=radius)
    assert u_.allclose(u_inv)
    # assert is_in_tangent_space(u_, at_point=mu1, eps=test_eps)
    assert u.allclose(u_, atol=test_eps, rtol=test_eps)


def test_parallel_transport_batch() -> None:
    mu1 = t([2., 1, np.sqrt(2)]) / radius
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)]) / radius
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    U = torch.stack((u, u2), dim=0)
    res = parallel_transport(U, src=mu1, dst=mu2, radius=radius)
    U_ = inverse_parallel_transport(res, src=mu1, dst=mu2, radius=radius)
    assert U.allclose(U_, atol=test_eps)


def test_parallel_transport_mu0() -> None:
    mu0 = t([0., 0, 0])
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)]) / radius
    u = t([0, 2, -np.sqrt(2)])

    assert SP.parallel_transport_mu0(u, dst=mu0, radius=radius).allclose(u)

    pt_u = SP.parallel_transport_mu0(u, dst=mu2, radius=radius)
    assert parallel_transport(u, src=mu0, dst=mu2, radius=radius).allclose(pt_u, atol=test_eps)

    u_inv = SP.inverse_parallel_transport_mu0(pt_u, src=mu2, radius=radius)
    assert u.allclose(u_inv)


def test_parallel_transport_mu0_batch() -> None:
    mu2 = radius * t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    U = torch.stack((u, u2), dim=0)
    res = SP.parallel_transport_mu0(U, dst=mu2, radius=radius)
    U_ = SP.inverse_parallel_transport_mu0(res, src=mu2, radius=radius)
    assert U.allclose(U_)


def test_exp_map() -> None:
    mu = t([2., 1, np.sqrt(2)]) / radius
    u = t([0, 2, -np.sqrt(2)])

    assert is_in_tangent_space(u, at_point=mu, eps=test_eps)
    u_mapped = SP.exp_map(u, at_point=mu, radius=radius)
    u_ = SP.inverse_exp_map(u_mapped, at_point=mu, radius=radius)
    assert u.allclose(u_, atol=test_eps)

    c = SP._c(radius)
    assert SP.spherical_projected_distance(mu, u_mapped, c).allclose(SP.lambda_x_c(mu, c) * torch.norm(u, p=2))


def test_exp_map_mu0() -> None:
    mu0 = SP.mu_0((3,))
    u = t([0, 2, -np.sqrt(2)])

    assert is_in_tangent_space(u, at_point=mu0, eps=test_eps)
    u_mapped = SP.exp_map(u, at_point=mu0, radius=radius)
    u_mu0_mapped = SP.exp_map_mu0(u, radius=radius)
    assert u_mapped.allclose(u_mu0_mapped)
    u_ = SP.inverse_exp_map(u_mapped, at_point=mu0, radius=radius)
    u_mu0 = SP.inverse_exp_map_mu0(u_mu0_mapped, radius=radius)
    assert u.allclose(u_, atol=test_eps)
    assert u.allclose(u_mu0, atol=test_eps)

    K = SP._c(radius)
    assert SP.spherical_projected_distance(mu0, u_mapped, K).allclose(2 * torch.norm(u, p=2))
    assert SP.spherical_projected_distance(mu0, u_mu0_mapped, K).allclose(2 * torch.norm(u, p=2))


def test_exp_map_large() -> None:
    mu = t([2., 1, np.sqrt(2)])
    u = 2.5 * t([0, 2, -np.sqrt(2)])
    assert is_in_tangent_space(u, at_point=mu, eps=test_eps)  # This should hold.
    u_mapped = SP.exp_map(u, at_point=mu, radius=radius)
    u_ = SP.inverse_exp_map(u_mapped, at_point=mu, radius=radius)
    assert u.allclose(u_, atol=test_eps)


def test_exp_map_batch() -> None:
    mu = t([2., 1, np.sqrt(2)]).double() / radius
    u = t([0, 2, -np.sqrt(2)]).double() / radius
    u2 = t([0, 4, -2 * np.sqrt(2)]).double() / radius
    assert is_in_tangent_space(u, at_point=mu, eps=test_eps)
    assert is_in_tangent_space(u2, at_point=mu, eps=test_eps)

    U = torch.stack((u, u2), dim=0)
    U_mapped = SP.exp_map(U, at_point=mu, radius=radius)
    U_ = SP.inverse_exp_map(U_mapped, at_point=mu, radius=radius)
    assert U.allclose(U_, atol=test_eps)


def test_sample_projection() -> None:
    v = t([0., 1, 2])

    mu0 = t([0., 0, 0])
    assert is_in_hyp_space(mu0)
    assert is_in_tangent_space(v, at_point=mu0)

    mu = t([2., 1, np.sqrt(2)]) / radius
    assert is_in_hyp_space(mu)

    v_proj, _ = SP.sample_projection_mu0(v, at_point=mu, radius=radius)
    assert is_in_hyp_space(v_proj, eps=test_eps)

    _, v_ = SP.inverse_sample_projection_mu0(v_proj, at_point=mu, radius=radius)
    assert v.allclose(v_, atol=test_eps)
    assert is_in_tangent_space(v_, at_point=mu0, eps=test_eps)


def test_projections() -> None:
    mu0_d = t([0., 0])
    mu0_s = radius * t([1., 0, 0])

    assert SP.projected_to_spherical(mu0_d, radius).allclose(mu0_s)
    assert S.spherical_to_projected(mu0_s, radius).allclose(mu0_d)

    mu_d = t([1, np.sqrt(2)]) / radius
    assert S.spherical_to_projected(SP.projected_to_spherical(mu_d, radius), radius).allclose(mu_d)
    mu_s = t([2., 1, np.sqrt(2)])
    mu_s = mu_s / mu_s.norm() * radius
    torch.norm(mu_s)
    mu_s_in_d = S.spherical_to_projected(mu_s, radius)
    mu_s_ = SP.projected_to_spherical(mu_s_in_d, radius)
    assert mu_s_.allclose(mu_s)
