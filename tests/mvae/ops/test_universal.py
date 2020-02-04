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

import numpy as np
import pytest
import torch
from torch import tensor as t

from mt.mvae.ops import Euclidean
from mt.mvae.ops import StereographicallyProjectedSphere
import mt.mvae.ops.poincare as P
import mt.mvae.ops.spherical_projected as SP
import mt.mvae.ops.euclidean as E
from mt.mvae.ops import PoincareBall
import mt.mvae.ops.universal as U
from mt.mvae.ops.common import eps

import tests.mvae.ops.test_spherical_projected as TSP
import tests.mvae.ops.test_poincare as TP
import tests.mvae.ops.test_euclidean as TE

np.random.seed(42)
random_nums = np.random.random_sample(100) * 100 + 1
test_eps = 5e-6
low_prec_test_eps = 1e-4
curvatures = [-1. / 4, 0., 1. / 4]  # radii -2, 0, 2


def _create_manifold(curvature: float) -> U.Universal:
    t = torch.tensor(curvature)
    return U.Universal(lambda: t)


def lambda_x(x: torch.Tensor, manifold: U.Universal) -> torch.Tensor:
    if isinstance(manifold.manifold, Euclidean):
        return torch.ones(x.shape[:-1] + (1,), dtype=x.dtype)
    elif isinstance(manifold.manifold, PoincareBall):
        return P.pm.lambda_x(x, c=-manifold.curvature, keepdim=True)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return SP.lambda_x_c(x, c=manifold.curvature)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


def exp_map(x: torch.Tensor, at_point: torch.Tensor, manifold: U.Universal) -> torch.Tensor:
    if isinstance(manifold.manifold, Euclidean):
        return E.exp_map(x, at_point)
    elif isinstance(manifold.manifold, PoincareBall):
        return P.exp_map(x, at_point, radius=manifold.radius)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return SP.exp_map(x, at_point, radius=manifold.radius)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


def inverse_exp_map(x: torch.Tensor, at_point: torch.Tensor, manifold: U.Universal) -> \
        torch.Tensor:
    if isinstance(manifold.manifold, Euclidean):
        return E.inverse_exp_map(x, at_point)
    elif isinstance(manifold.manifold, PoincareBall):
        return P.inverse_exp_map(x, at_point, radius=manifold.radius)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return SP.inverse_exp_map(x, at_point, radius=manifold.radius)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


def inverse_sample_projection_mu0(x: torch.Tensor, at_point: torch.Tensor, manifold: U.Universal) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(manifold.manifold, Euclidean):
        return E.inverse_sample_projection_mu0(x, at_point)
    elif isinstance(manifold.manifold, PoincareBall):
        return P.inverse_sample_projection_mu0(x, at_point, radius=manifold.radius)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return SP.inverse_sample_projection_mu0(x, at_point, radius=manifold.radius)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


def is_in_hyp_space(x: torch.Tensor, manifold: U.Universal, eps: float = eps) -> torch.Tensor:
    if isinstance(manifold.manifold, Euclidean):
        return TE.is_in_hyp_space(x, eps=eps)
    elif isinstance(manifold.manifold, PoincareBall):
        return TP.is_in_hyp_space(x, radius=manifold.radius, eps=eps)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return TSP.is_in_hyp_space(x, radius=manifold.radius, eps=eps)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


def is_in_tangent_space(x: torch.Tensor, at_point: torch.Tensor, manifold: U.Universal,
                        eps: float = eps) -> torch.Tensor:
    if isinstance(manifold.manifold, Euclidean):
        return TE.is_in_tangent_space(x, at_point, eps=eps)
    elif isinstance(manifold.manifold, PoincareBall):
        return TP.is_in_tangent_space(x, at_point, radius=manifold.radius, eps=eps)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return TSP.is_in_tangent_space(x, at_point, radius=manifold.radius, eps=eps)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


def distance(x: torch.Tensor, y: torch.Tensor, manifold: U.Universal, **kwargs: Any) -> torch.Tensor:
    if isinstance(manifold.manifold, Euclidean):
        return TE.euclidean_distance(x, y, **kwargs)
    elif isinstance(manifold.manifold, PoincareBall):
        return P.poincare_distance(x, y, radius=manifold.radius, **kwargs)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return SP.spherical_projected_distance(x, y, K=manifold.curvature, **kwargs)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


def parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, manifold: U.Universal) -> torch.Tensor:
    if isinstance(manifold.manifold, Euclidean):
        return TE.parallel_transport(x, src, dst)
    elif isinstance(manifold.manifold, PoincareBall):
        return TP.parallel_transport(x, src, dst, radius=manifold.radius)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return TSP.parallel_transport(x, src, dst, radius=manifold.radius)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


def inverse_parallel_transport(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                               manifold: U.Universal) -> torch.Tensor:
    if isinstance(manifold.manifold, Euclidean):
        return TE.inverse_parallel_transport(x, src, dst)
    elif isinstance(manifold.manifold, PoincareBall):
        return TP.inverse_parallel_transport(x, src, dst, radius=manifold.radius)
    elif isinstance(manifold.manifold, StereographicallyProjectedSphere):
        return TSP.inverse_parallel_transport(x, src, dst, radius=manifold.radius)
    else:
        raise ValueError("Unknown manifold " + manifold.manifold.__class__.__name__)


@pytest.mark.parametrize("curvature", curvatures)
def test_mu_0(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    res = manifold.mu_0((3, 3), dtype=torch.int64)
    expected = t([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert res.allclose(expected)


@pytest.mark.parametrize("curvature", curvatures)
def test_is_in_hyp_space(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    radius = manifold.radius

    assert is_in_hyp_space(radius * t([1., 0, 0]), manifold=manifold)

    if curvature >= 0:
        assert is_in_hyp_space(radius * t([0.1, 2, 3]), manifold=manifold)
        assert is_in_hyp_space(radius * t([1, 1, np.sqrt(2)]), manifold=manifold)
        assert is_in_hyp_space(radius * t([1, 1, np.sqrt(2)]), manifold=manifold)
        assert is_in_hyp_space(radius * t([0, 2, -np.sqrt(2)]), manifold=manifold)
    else:
        assert not is_in_hyp_space(radius * t([0.1, 2, 3]), manifold=manifold)
        assert not is_in_hyp_space(radius * t([1, 1, np.sqrt(2)]), manifold=manifold)
        assert not is_in_hyp_space(radius * t([1, 1, np.sqrt(2)]), manifold=manifold)
        assert not is_in_hyp_space(radius * t([0, 2, -np.sqrt(2)]), manifold=manifold)


@pytest.mark.parametrize("curvature", curvatures)
def test_is_in_tangent_space(curvature: float) -> None:
    # manifold = _create_manifold(curvature)

    # assert is_in_tangent_space(t([0., 2, 3]), manifold.radius * t([1., 0, 0]), manifold=manifold)
    #
    # if curvature <= 0:
    #     assert not is_in_tangent_space(t([0.1, 2, 3]), manifold.radius * t([1., 0, 0]), manifold=manifold)
    # else:
    #     assert is_in_tangent_space(t([0.1, 2, 3]), manifold.radius * t([1., 0, 0]), manifold=manifold)
    #
    # assert is_in_tangent_space(t([0, 2, -np.sqrt(2)]), t([1, 0.5, np.sqrt(2) / 2]), eps=test_eps, manifold=manifold)
    # TODO-LATER: Projected spaces tangent spaces don't work like this.
    pass


@pytest.mark.parametrize("curvature", curvatures)
def test_distance(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu0 = manifold.radius * t([1., 0, 0])
    mu = manifold.radius * t([2., 1., np.sqrt(2)])
    assert distance(mu0, mu0, manifold=manifold).allclose(t(0.), atol=5e-4)
    assert distance(mu, mu, manifold=manifold).allclose(t(0.), atol=5e-4)
    assert distance(mu0, mu, manifold=manifold) == distance(mu, mu0, manifold=manifold)


@pytest.mark.parametrize("curvature", curvatures)
def test_parallel_transport(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu1 = t([2., 1, np.sqrt(2)]).double() / manifold.radius
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)]).double() / manifold.radius
    assert is_in_hyp_space(mu1, manifold=manifold)
    assert is_in_hyp_space(mu2, manifold=manifold)

    u = t([0, 2, -np.sqrt(2)]).double()
    # assert is_in_tangent_space(u, at_point=mu1, eps=test_eps, manifold=manifold)

    assert parallel_transport(u, src=mu1, dst=mu1, manifold=manifold).allclose(u, atol=5e-4)

    pt_u = parallel_transport(u, src=mu1, dst=mu2, manifold=manifold)
    # assert is_in_tangent_space(pt_u, at_point=mu2, eps=test_eps, manifold=manifold)
    u_ = parallel_transport(pt_u, src=mu2, dst=mu1, manifold=manifold)
    assert u.allclose(u_, atol=5e-4)
    u_inv = inverse_parallel_transport(pt_u, src=mu1, dst=mu2, manifold=manifold)
    assert u.allclose(u_inv)


@pytest.mark.parametrize("curvature", curvatures)
def test_parallel_transport_batch(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu1 = t([2., 1, np.sqrt(2)])
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    U = torch.stack((u, u2), dim=0)
    res = parallel_transport(U, src=mu1, dst=mu2, manifold=manifold)
    U_ = inverse_parallel_transport(res, src=mu1, dst=mu2, manifold=manifold)
    assert U.allclose(U_, atol=test_eps)


@pytest.mark.parametrize("curvature", curvatures)
def test_parallel_transport_mu0(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu0 = t([0., 0, 0])
    mu2 = t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])

    assert manifold.parallel_transport_mu0(u, dst=mu0).allclose(u)

    pt_u = manifold.parallel_transport_mu0(u, dst=mu2)
    assert parallel_transport(u, src=mu0, dst=mu2, manifold=manifold).allclose(pt_u)

    u_inv = manifold.inverse_parallel_transport_mu0(pt_u, src=mu2)
    assert u.allclose(u_inv)


@pytest.mark.parametrize("curvature", curvatures)
def test_parallel_transport_mu0_batch(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu2 = manifold.radius * t([np.sqrt(5), 1, np.sqrt(3)])
    u = t([0, 2, -np.sqrt(2)])
    u2 = t([0, 4, -2 * np.sqrt(2)])

    UU = torch.stack((u, u2), dim=0)
    res = manifold.parallel_transport_mu0(UU, dst=mu2)
    U_ = manifold.inverse_parallel_transport_mu0(res, src=mu2)
    assert UU.allclose(U_)


@pytest.mark.parametrize("curvature", curvatures)
def test_exp_map(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu = t([1., 0.5, np.sqrt(2) / 2])
    u = t([0, 2, -np.sqrt(2)])

    assert is_in_tangent_space(u, at_point=mu, eps=test_eps, manifold=manifold)
    u_mapped = exp_map(u, at_point=mu, manifold=manifold)
    u_ = inverse_exp_map(u_mapped, at_point=mu, manifold=manifold)
    assert u.allclose(u_, atol=test_eps)

    assert distance(mu, u_mapped, manifold=manifold).allclose(lambda_x(mu, manifold=manifold) * torch.norm(u, p=2))


@pytest.mark.parametrize("curvature", curvatures)
def test_exp_map_mu0(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu0 = manifold.mu_0((3,)).double()
    u = t([0, 2, -np.sqrt(2)]).double()

    assert is_in_tangent_space(u, at_point=mu0, eps=test_eps, manifold=manifold)
    u_mapped = exp_map(u, at_point=mu0, manifold=manifold)
    u_mu0_mapped = manifold.exp_map_mu0(u)
    assert u_mapped.allclose(u_mu0_mapped)
    u_ = inverse_exp_map(u_mapped, at_point=mu0, manifold=manifold)
    u_mu0 = manifold.inverse_exp_map_mu0(u_mu0_mapped)
    assert u.allclose(u_, atol=test_eps)
    assert u.allclose(u_mu0, atol=test_eps)

    assert distance(mu0, u_mapped, manifold=manifold).allclose(lambda_x(mu0, manifold) * torch.norm(u, p=2))
    assert distance(mu0, u_mu0_mapped, manifold=manifold).allclose(lambda_x(mu0, manifold) * torch.norm(u, p=2))


@pytest.mark.parametrize("curvature", curvatures)
def test_exp_map_large(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu = t([1., 0.5, np.sqrt(2) / 2]).double()
    u = 1.8 * t([0, 2, -np.sqrt(2)]).double()
    assert is_in_tangent_space(u, at_point=mu, eps=test_eps, manifold=manifold)  # This should hold.
    u_mapped = exp_map(u, at_point=mu, manifold=manifold)
    u_ = inverse_exp_map(u_mapped, at_point=mu, manifold=manifold)
    assert u.allclose(u_, atol=low_prec_test_eps)

    assert distance(mu, u_mapped, manifold=manifold).allclose(lambda_x(mu, manifold) * torch.norm(u, p=2))


@pytest.mark.parametrize("curvature", curvatures)
def test_exp_map_batch(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    mu = t([1., 0.5, np.sqrt(2) / 2]).double()
    u = t([0, 2, -np.sqrt(2)]).double()
    u2 = 0.9 * t([0, 4, -2 * np.sqrt(2)]).double()
    assert is_in_tangent_space(u, at_point=mu, eps=test_eps, manifold=manifold)
    assert is_in_tangent_space(u2, at_point=mu, eps=test_eps, manifold=manifold)

    UU = torch.stack((u, u2), dim=0)
    U_mapped = exp_map(UU, at_point=mu, manifold=manifold)
    U_ = inverse_exp_map(U_mapped, at_point=mu, manifold=manifold)
    assert UU.allclose(U_, atol=1e-5)

    assert distance(mu, U_mapped,
                    manifold=manifold).allclose(lambda_x(mu, manifold) * torch.norm(UU, p=2, dim=-1, keepdim=True))


@pytest.mark.parametrize("curvature", curvatures)
def test_sample_projection(curvature: float) -> None:
    manifold = _create_manifold(curvature)
    v = t([0., 1, 2])

    mu0 = t([0., 0, 0])
    assert is_in_hyp_space(mu0, manifold=manifold)
    assert is_in_tangent_space(v, at_point=mu0, manifold=manifold)

    mu = t([1., 0.5, np.sqrt(2) / 2])
    assert is_in_hyp_space(mu, manifold=manifold)

    v_proj, _ = manifold.sample_projection_mu0(v, at_point=mu)
    assert is_in_hyp_space(v_proj, eps=low_prec_test_eps, manifold=manifold)

    _, v_ = inverse_sample_projection_mu0(v_proj, at_point=mu, manifold=manifold)
    assert v.allclose(v_, atol=test_eps)
    assert is_in_tangent_space(v_, at_point=mu0, eps=test_eps, manifold=manifold)
