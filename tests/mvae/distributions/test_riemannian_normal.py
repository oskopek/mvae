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

from typing import Type, Optional

import numpy as np
import pytest
import torch

from mt.mvae import utils
from mt.mvae.distributions import RiemannianNormal
from mt.mvae.ops import PoincareBall, Manifold

dims = [2, 6, 10, 40]
scales = [100, 1.01, 1.0001, 1., 0.9]  # Variance is always >= on the diagonal.
radii = [0.5, 1., 2.6]
manifold_types = [PoincareBall]  # , StereographicallyProjectedSphere] Doesn't work yet, as the math isn't there.
# dtypes = ["float32", "float64"]  # Float32 doesn't seem to be enough.
dtypes = ["float64"]

dtype_map = {"float32": torch.float32, "float64": torch.float64}


def rn_distribution(dim: int,
                    scale: float,
                    radius: float,
                    manifold_type: Type[Manifold],
                    batch: int = 2,
                    loc: Optional[torch.Tensor] = None) -> RiemannianNormal:
    utils.set_seeds(42)
    if loc is None:
        loc = torch.tensor([[0] * dim], dtype=torch.get_default_dtype()).repeat(batch, 1)
    scale = torch.tensor([[scale]], dtype=torch.get_default_dtype()).repeat(batch, 1)
    radius = torch.tensor(radius, dtype=torch.get_default_dtype())
    d = RiemannianNormal(loc, scale, manifold_type(lambda: radius))
    return d


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("scale", scales)
@pytest.mark.parametrize("radius", radii)
@pytest.mark.parametrize("manifold_type", manifold_types)
@pytest.mark.parametrize("dtype", dtypes)
def test_rn_sampling_nans(dim: int, scale: float, radius: float, manifold_type: Type[Manifold], dtype: str) -> None:
    orig = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype_map[dtype])
        batch = 2
        n_samples = 1000
        rn = rn_distribution(dim, scale, radius, manifold_type, batch=batch)
        for i in range(100):
            samples, log_prob = rn.rsample_log_prob(torch.Size([n_samples]))
            assert samples.shape == torch.Size([n_samples, batch, dim])
            assert torch.isfinite(samples).all()
            assert torch.isfinite(log_prob).all()
            assert (log_prob <= 0).all(), "Log probs too big."
    finally:
        torch.set_default_dtype(orig)


@pytest.mark.parametrize("manifold_type", manifold_types)
@pytest.mark.parametrize("dtype", dtypes)
def test_rsample_log_prob(manifold_type: Type[Manifold], dtype: str) -> None:
    orig = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype_map[dtype])
        batch_size = 10
        q = rn_distribution(dim=3,
                            scale=10.,
                            radius=1.0,
                            batch=batch_size,
                            manifold_type=manifold_type,
                            loc=torch.tensor([[2., 1, np.sqrt(2)]]).repeat((batch_size, 1)))
        shape = torch.Size([100])

        z, log_prob_sample = q.rsample_log_prob(shape)
        assert torch.isfinite(log_prob_sample).all()
        assert (log_prob_sample <= 0).all(), "Log probs too big."
    finally:
        torch.set_default_dtype(orig)
