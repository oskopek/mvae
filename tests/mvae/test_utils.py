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

from typing import List

import pytest

import mt.mvae.utils as utils
from mt.mvae.components import Component, EuclideanComponent, HyperbolicComponent, SphericalComponent
from mt.mvae.components import StereographicallyProjectedSphereComponent, PoincareComponent
from mt.mvae.sampling import WrappedNormalProcedure as wn
from mt.mvae.sampling import EuclideanNormalProcedure as en
from mt.mvae.sampling import SphericalVmfProcedure as svmf
from mt.mvae.sampling import ProjectedSphericalVmfProcedure as pvmf
from mt.mvae.sampling import RiemannianNormalProcedure as rn


def component_type_and_dim_equal(xs: List[Component], ys: List[Component]) -> None:  # type: ignore
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        assert x.dim == y.dim
        assert type(x) == type(y)
        assert x.fixed_curvature == y.fixed_curvature
        if not isinstance(x, EuclideanComponent):  # Euclidean component is untrainable always.
            x.init_layers(10, True)
            assert x.manifold.curvature.requires_grad == (not x.fixed_curvature)
        if not isinstance(x, EuclideanComponent):  # Euclidean component is untrainable always.
            y.init_layers(10, True)
            assert y.manifold.curvature.requires_grad == (not y.fixed_curvature)


@pytest.mark.parametrize("fixed_curvature", [True, False])
def test_parse_components_empty(fixed_curvature: bool) -> None:
    assert utils.parse_components("", fixed_curvature) == []
    assert utils.parse_components("  ", fixed_curvature) == []


@pytest.mark.parametrize("fc", [True, False])
def test_parse_components_basic(fc: bool) -> None:
    component_type_and_dim_equal(utils.parse_components("d2", fc),
                                 [StereographicallyProjectedSphereComponent(2, fc, pvmf)])
    component_type_and_dim_equal(utils.parse_components("p2", fc), [PoincareComponent(2, fc, rn)])
    component_type_and_dim_equal(utils.parse_components("e1", fc), [EuclideanComponent(1, fc, en)])
    component_type_and_dim_equal(utils.parse_components("e2", fc), [EuclideanComponent(2, fc, en)])
    component_type_and_dim_equal(utils.parse_components(
        "3e2", fc), [EuclideanComponent(2, fc, en),
                     EuclideanComponent(2, fc, en),
                     EuclideanComponent(2, fc, en)])
    component_type_and_dim_equal(utils.parse_components("s2", fc), [SphericalComponent(2, fc, svmf)])
    component_type_and_dim_equal(utils.parse_components("h2", fc), [HyperbolicComponent(2, fc, wn)])
    component_type_and_dim_equal(utils.parse_components("1h2", fc), [HyperbolicComponent(2, fc, wn)])


@pytest.mark.parametrize("fixed_curvature", [True, False])
def test_canonical_name(fixed_curvature: bool) -> None:
    assert utils.canonical_name(utils.parse_components("3d2", fixed_curvature)) == "3d2"
    assert utils.canonical_name(utils.parse_components("3h3,2s2,1e1,e2", fixed_curvature)) == "e1,e2,3h3,2s2"
    for model in ["e1", "e1,e2,s3", "10e2,2h2,4s32"]:
        assert utils.canonical_name(utils.parse_components(model, fixed_curvature)) == model


@pytest.mark.parametrize("fc", [True, False])
def test_parse_components_products(fc: bool) -> None:

    component_type_and_dim_equal(utils.parse_components(
        "e1,e2", fc), [EuclideanComponent(1, fc, en), EuclideanComponent(2, fc, en)])
    component_type_and_dim_equal(
        utils.parse_components("e1,s2,e2", fc),
        [EuclideanComponent(1, fc, en),
         SphericalComponent(2, fc, svmf),
         EuclideanComponent(2, fc, en)])
    component_type_and_dim_equal(
        utils.parse_components("h3,s2,e1", fc),
        [HyperbolicComponent(3, fc, wn),
         SphericalComponent(2, fc, svmf),
         EuclideanComponent(1, fc, en)])
    component_type_and_dim_equal(utils.parse_components("3h3,2s2,e1", fc), [
        HyperbolicComponent(3, fc, wn),
        HyperbolicComponent(3, fc, wn),
        HyperbolicComponent(3, fc, wn),
        SphericalComponent(2, fc, svmf),
        SphericalComponent(2, fc, svmf),
        EuclideanComponent(1, fc, en)
    ])


def test_linear_betas_incr() -> None:
    betas = utils.linear_betas(1.0, 2.0, 30, 100)
    assert len(betas) == 100
    assert betas[0] == 1.
    assert betas[1] > 1.
    assert betas[28] < 2.
    for i in range(29, len(betas)):
        assert betas[i] == 2.


def test_linear_betas_decr() -> None:
    betas = utils.linear_betas(2.0, 1.0, 30, 100)
    assert len(betas) == 100
    assert betas[0] == 2.
    assert betas[1] < 2.
    assert betas[28] > 1.
    for i in range(29, len(betas)):
        assert betas[i] == 1.
