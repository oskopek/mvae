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

from typing import Tuple, Any

import torch
from torch import Tensor

from .manifold import Manifold


class Euclidean(Manifold):

    @property
    def radius(self) -> Tensor:
        return 0

    @property
    def curvature(self) -> Tensor:
        return 0

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        return exp_map_mu0(x)

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        return inverse_exp_map_mu0(x)

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        return parallel_transport_mu0(x, dst)

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        return inverse_parallel_transport_mu0(x, src)

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        return mu_0(shape, **kwargs)

    def __init__(self) -> None:
        super().__init__()

    def sample_projection_mu0(self, x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return sample_projection_mu0(x, at_point)

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        raise inverse_sample_projection_mu0(x_proj, at_point)

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        return torch.zeros_like(mu)


def mu_0(shape: Tuple[int, ...], **kwargs: Any) -> Tensor:
    return torch.zeros(shape, **kwargs)


def parallel_transport_mu0(x: Tensor, dst: Tensor) -> Tensor:
    return x


def inverse_parallel_transport_mu0(x: Tensor, src: Tensor) -> Tensor:
    return x


def exp_map(x: Tensor, at_point: Tensor) -> Tensor:
    return at_point + x / 2


def exp_map_mu0(x: Tensor) -> Tensor:
    return x / 2


def inverse_exp_map(x: Tensor, at_point: Tensor) -> Tensor:
    return 2 * (x - at_point)


def inverse_exp_map_mu0(x: Tensor) -> Tensor:
    return 2 * x


def sample_projection_mu0(x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    pt = parallel_transport_mu0(x, at_point)
    xx = exp_map(pt, at_point)
    return xx, (pt, x)


def inverse_sample_projection_mu0(x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
    pt = inverse_exp_map(x, at_point)
    xx = inverse_parallel_transport_mu0(pt, at_point)
    return pt, xx
