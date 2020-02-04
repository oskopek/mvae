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

from typing import Any, Tuple, Callable

import torch
from torch import Tensor


class Manifold:

    def sample_projection_mu0(self, xexpo: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Projects the point x from the T_{x}M tangent space to the manifold M.
        :param x:
        :param at_point:
        :return:
        """
        raise NotImplementedError

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError

    @property
    def radius(self) -> Tensor:
        raise NotImplementedError

    @property
    def curvature(self) -> Tensor:
        raise NotImplementedError

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        raise NotImplementedError

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        raise NotImplementedError


class RadiusManifold(Manifold):

    def __init__(self, radius: Callable[[], Tensor]):
        super().__init__()
        self._radius = radius

    @property
    def curvature(self) -> Tensor:
        return 1. / self.radius.pow(2)

    @property
    def radius(self) -> Tensor:
        return torch.clamp(torch.relu(self._radius()), min=1e-8, max=1e8)
