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

from typing import Callable

import numpy as np
import pytest
import torch
from torch import tensor as t

import mt.mvae.ops.hyperbolics as H
from mt.mvae.ops.common import eps, logcosh, logsinh

np.random.seed(42)
test_eps = 5e-6
low_prec_test_eps = 1e-4


@pytest.mark.parametrize("fun", [
    H.acosh,
    H.sqrt,
    H.cosh,
    H.sinh,
    logsinh,
    logcosh,
])
@pytest.mark.parametrize("x", (np.random.random_sample(100) - 0.5) * 1000.)
def test_function_non_nan(fun: Callable[[torch.Tensor], torch.Tensor], x: float) -> None:
    res = fun(torch.tensor(x).float())
    assert torch.isfinite(res).all()


@pytest.mark.parametrize("x", np.random.random_sample(100) * 100 + 1)
def test_acosh(x: float) -> None:
    npres = np.arccosh(x)
    torchres = H.acosh(torch.tensor(x).float())
    torchres = np.float32(torchres)
    assert np.allclose(npres, torchres)


def test_acosh_constants() -> None:
    assert torch.isfinite(H.acosh(t(0.)))
    assert torch.isfinite(H.acosh(t(1. - eps)))
    assert t(0.).allclose(H.acosh(t(1.)), atol=5e-4)
