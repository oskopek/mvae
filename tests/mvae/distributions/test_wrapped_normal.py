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

import numpy as np
import pytest
import torch

from mt.mvae import utils
from mt.mvae.distributions import WrappedNormal
from mt.mvae.ops import Hyperboloid

dims = [3, 6, 11, 21, 41]
scales = [100, 1 + 1e-2, 1 + 1e-15, 1]  # Variance is always >= on the diagonal.
radii = [1e-5, 1., 1e5]


def phg_distribution(dim: int, scale: float, radius: float, batch: int = 2) -> WrappedNormal:
    utils.set_seeds(42)
    loc = torch.tensor([1] + [0] * (dim - 1), dtype=torch.get_default_dtype())
    covariance_diag = torch.ones(dim - 1, dtype=torch.get_default_dtype()) * scale
    radius = torch.tensor(radius, dtype=torch.get_default_dtype())
    phg = WrappedNormal(loc.unsqueeze(0).repeat(batch, 1),
                        covariance_diag.unsqueeze(0).repeat(batch, 1),
                        manifold=Hyperboloid(lambda: radius))
    return phg


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("scale", scales)
@pytest.mark.parametrize("radius", radii)
def test_phg_sampling_nans(dim: int, scale: float, radius: float) -> None:
    batch = 2
    n_samples = 1000
    phg = phg_distribution(dim, scale, radius, batch=batch)
    for i in range(100):
        samples, log_prob = phg.rsample_log_prob(torch.Size([n_samples]))
        assert samples.shape == torch.Size([n_samples, batch, dim])
        assert torch.isfinite(samples).all()
        assert torch.isfinite(log_prob).all()
        assert (log_prob <= 0).all(), "Log probs too big."


def test_rsample_log_prob() -> None:
    batch_size = 10
    q = phg_distribution(dim=3, scale=10., radius=1.0, batch=batch_size)
    q.loc = torch.tensor([[2., 1, np.sqrt(2)]]).repeat((batch_size, 1))
    shape = torch.Size([100])

    z, log_prob_sample = q.rsample_log_prob(shape)
    assert torch.isfinite(log_prob_sample).all()
