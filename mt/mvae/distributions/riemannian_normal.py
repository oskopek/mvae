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

from numbers import Number

import torch
import torch.distributions
from torch import Tensor

from .wrapped_distributions import VaeDistribution
from .hyperbolic_radius import HyperbolicRadius
from .hyperspherical_uniform import HypersphericalUniform
from ..ops import Manifold, PoincareBall
from ..ops import poincare as P


class RiemannianNormal(torch.distributions.Distribution, VaeDistribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.interval(-1, 1),
        "scale": torch.distributions.constraints.positive
    }
    support = torch.distributions.constraints.interval(-1, 1)
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def stddev(self) -> Tensor:
        return self.scale  # Not actually the stddev, I think.

    def __init__(self, loc: Tensor, scale: Tensor, manifold: Manifold, validate_args: bool = None) -> None:
        if not isinstance(manifold, PoincareBall):
            raise NotImplementedError("Riemannian Normal currently only works for the Poincare ball.")

        if len(scale.shape) < 2 or scale.size(dim=-1) > 1:
            raise NotImplementedError(f"Illegal shape for scale: {scale.shape}. Need spherical covariance of dim=2.")

        assert not (torch.isnan(loc).any() or torch.isnan(scale).any())
        assert len(scale.shape) == 2
        self.loc = loc
        self.scale = torch.clamp(scale, min=.1)
        self.manifold = manifold
        self.dim = self.loc.size(-1)
        self.radius = HyperbolicRadius(self.dim, self.c, self.scale)
        self.direction = HypersphericalUniform(self.dim - 1, device=loc.device)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def c(self) -> Tensor:
        return P._c(self.manifold.radius)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        alpha = self.direction.sample(torch.Size(shape[:-1])).squeeze(dim=-2)
        radius = self.radius.rsample(sample_shape).squeeze(dim=0)
        res = P.exp_map_x_polar(self.loc.expand(shape), radius, alpha, c=self.c)
        return res

    def log_prob(self, value: Tensor) -> Tensor:
        loc = self.loc.expand(value.shape)
        radius_sq = P.poincare_distance_c(loc, value, c=self.c)**2
        res = -radius_sq / 2 / self.scale.pow(2) - self.direction.log_normalizer() - self.radius.log_normalizer
        assert res.size(-1) == 1
        assert torch.isfinite(res).all()
        return res.squeeze(dim=-1)
