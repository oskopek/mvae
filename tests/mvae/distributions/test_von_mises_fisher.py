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

import pytest
import torch

from mt.mvae import utils
from mt.mvae.distributions.von_mises_fisher import VonMisesFisher

dims = [2, 3, 4]
scales = [1e9, 1e5, 1e1, 1e0, 1e-5, 1e-15]


def vmf_distribution(dim: int, scale: float) -> VonMisesFisher:
    utils.set_seeds(42)
    loc = torch.tensor([[1.] + [0.] * (dim - 1)], dtype=torch.get_default_dtype())
    scale = torch.tensor([[scale]], dtype=torch.get_default_dtype())
    vmf = VonMisesFisher(loc, scale)
    return vmf


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("scale", scales)
def test_vmf_sampling_nans(dim: int, scale: float) -> None:
    vmf = vmf_distribution(dim, scale)
    shape = torch.Size([10])
    for i in range(100):
        samples = vmf.sample(shape)
        assert torch.isfinite(samples).all()
        assert torch.norm(samples, p=2, dim=-1).allclose(torch.ones(samples.shape[:-1]))
        log_prob = vmf.log_prob(samples)
        assert torch.isfinite(log_prob).all()
        # assert (log_prob <= 0).all() This does not hold, because it actually doesn't have to hold :) Math :)
        assert (log_prob < 1e20).all()
        assert (log_prob > -1e20).all()


# This does not depend on the mean (loc), just it's dimensionality.
@pytest.mark.parametrize("scale", scales)
def test_sampling_w3(scale: float) -> None:
    vmf = vmf_distribution(3, scale)
    w = vmf._sample_w3(shape=torch.Size([100]))
    assert (w.abs() <= 1).all()


# This does not depend on the mean (loc), just it's dimensionality.
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("scale", scales)
def test_sampling_w_rej(dim: int, scale: float) -> None:
    vmf = vmf_distribution(dim, scale)
    w = vmf._sample_w_rej(shape=torch.Size([100]))
    assert (w.abs() <= 1).all()
