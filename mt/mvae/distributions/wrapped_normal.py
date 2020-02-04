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

from typing import Any, Tuple, Optional

import torch
from torch import Tensor

from .wrapped_distributions import VaeDistribution, EuclideanNormal
from ..ops import Manifold, StereographicallyProjectedSphere, PoincareBall


class WrappedNormal(torch.distributions.Distribution, VaeDistribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real_vector,
        "scale": torch.distributions.constraints.positive
    }
    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, loc: Tensor, scale: Tensor, manifold: Manifold, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim = loc.shape[-1]

        # is projected?
        if isinstance(manifold, PoincareBall) or isinstance(manifold, StereographicallyProjectedSphere):
            tangent_dim = self.dim
        else:
            tangent_dim = self.dim - 1

        if scale.shape[-1] > 1 and scale.shape[-1] != tangent_dim:
            raise ValueError("Invalid scale dimension: neither isotropic nor elliptical.")

        if scale.shape[-1] == 1:  # repeat along last dim for (loc.shape[-1] - 1) times.
            s = [1] * len(scale.shape)
            s[-1] = tangent_dim
            scale = scale.repeat(s)  # Expand scalar scale to vector.

        # Loc has to be one dim bigger than scale or equal (in projected spaces).
        assert loc.shape[:-1] == scale.shape[:-1]
        assert tangent_dim == scale.shape[-1]

        self.loc = loc
        self.scale = scale
        self.manifold = manifold
        self.device = self.loc.device
        smaller_shape = self.loc.shape[:-1] + torch.Size([tangent_dim])
        self.normal = EuclideanNormal(torch.zeros(smaller_shape, device=self.device), scale, *args, **kwargs)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def stddev(self) -> Tensor:
        return self.scale

    def rsample_with_parts(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        # v ~ N(0, \Sigma)
        v_tilde = self.normal.rsample(shape)
        assert torch.isfinite(v_tilde).all()
        # u = PT_{mu_0 -> mu}([0, v_tilde])
        # z = exp_{mu}(u)
        z, helper_data = self.manifold.sample_projection_mu0(v_tilde, at_point=self.loc)
        assert torch.isfinite(z).all()
        return z, helper_data

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        z, _ = self.rsample_with_parts(sample_shape)
        return z

    def log_prob_from_parts(self, z: Tensor, data: Optional[Tuple[Tensor, ...]]) -> Tensor:
        if data is None:
            raise ValueError("Additional data cannot be empty for WrappedNormal.")

        # log(z) = log p(v) - log det [(\partial / \partial v) proj_{\mu}(v)]
        v = data[1]
        assert torch.isfinite(v).all()

        n_logprob = self.normal.log_prob(v)
        logdet = self.manifold.logdet(self.loc, self.scale, z, (*data, n_logprob))
        assert n_logprob.shape == logdet.shape
        log_prob = n_logprob - logdet
        assert torch.isfinite(log_prob).all()
        return log_prob

    def log_prob(self, z: Tensor) -> Tensor:
        """Should only be used for p_z, prefer log_prob_from_parts."""
        assert torch.isfinite(z).all()
        data = self.manifold.inverse_sample_projection_mu0(z, at_point=self.loc)
        return self.log_prob_from_parts(z, data)

    def rsample_log_prob(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tensor]:
        z, data = self.rsample_with_parts(shape)
        return z, self.log_prob_from_parts(z, data)
