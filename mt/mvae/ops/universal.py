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

from .manifold import Manifold
from .poincare import PoincareBall
from .euclidean import Euclidean
from .spherical_projected import StereographicallyProjectedSphere
from ..ops.common import sqrt


class Universal(Manifold):

    @property
    def radius(self) -> Tensor:
        return torch.relu(1 / sqrt(self.curvature.abs()))

    @property
    def curvature(self) -> Tensor:
        return self._curvature()

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        return self.manifold.exp_map_mu0(x)

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        return self.manifold.inverse_exp_map_mu0(x)

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        return self.manifold.parallel_transport_mu0(x, dst)

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        return self.manifold.inverse_parallel_transport_mu0(x, src)

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        return self.manifold.mu_0(shape, **kwargs)

    def __init__(self, curvature: Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self._curvature = curvature
        self._manifolds = {
            -1: PoincareBall(lambda: self.radius),
            0: Euclidean(),
            1: StereographicallyProjectedSphere(lambda: self.radius)
        }
        self.eps = eps

    @property
    def manifold(self) -> Manifold:
        return self._manifolds[self._choice]

    @property
    def _choice(self) -> int:
        if self._curvature() < -self.eps:
            return -1
        elif self._curvature() > self.eps:
            return +1
        else:
            return 0

    def sample_projection_mu0(self, x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return self.manifold.sample_projection_mu0(x, at_point)

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        return self.manifold.inverse_sample_projection_mu0(x_proj, at_point)

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        return self.manifold.logdet(mu, std, z, data)
