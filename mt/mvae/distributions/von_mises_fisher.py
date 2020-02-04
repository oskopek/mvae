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

import math
from typing import Optional, Tuple

import torch
import torch.distributions
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.kl import register_kl

from .wrapped_distributions import VaeDistribution, EuclideanUniform
from .hyperspherical_uniform import HypersphericalUniform
from ..ops.ive import ive
from ..ops import common as C
from ..ops import spherical_projected as SP
from ..ops import spherical as S


class VonMisesFisher(torch.distributions.Distribution, VaeDistribution):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.loc * (ive(self.p / 2, self.scale) / ive(self.p / 2 - 1, self.scale))

    @property
    def stddev(self) -> Tensor:
        return self.scale

    def __init__(self, loc: Tensor, scale: Tensor, validate_args: Optional[bool] = None) -> None:
        self.dtype = loc.dtype
        self.loc = loc
        assert loc.norm(p=2, dim=-1).allclose(torch.ones(loc.shape[:-1], device=loc.device))
        self.scale = scale
        assert (scale > 0).all()
        self.device = loc.device
        self.p = loc.shape[-1]

        self.uniform = EuclideanUniform(0, 1)
        self.hyperspherical_uniform_v = HypersphericalUniform(self.p - 2, device=self.device)

        # Pre-compute Householder transformation
        e1 = torch.tensor([1.] + [0.] * (loc.shape[-1] - 1), requires_grad=False, device=self.device)
        self.u = F.normalize(e1 - self.loc)

        super().__init__(self.loc.size(), validate_args=validate_args)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = sample_shape if isinstance(sample_shape, torch.Size) else torch.Size([sample_shape])

        # Sample w ~ g(w | kappa, m). Should be in [-1, 1], but not always is (numerical reasons).
        w = self._sample_w3(shape=shape) if self.p == 3 else self._sample_w_rej(shape=shape)

        # Sample v ~ U(S^(m-2))
        v = self.hyperspherical_uniform_v.sample(shape)

        w_ = C.sqrt(1 - w**2)
        x = torch.cat((w, w_ * v), dim=-1)
        z = self._householder_rotation(x)
        return z.to(dtype=self.dtype)

    def _sample_w3(self, shape: torch.Size) -> Tensor:
        shape = torch.Size(shape + torch.Size(self.scale.shape))
        u = self.uniform.sample(shape).to(self.device)

        log_u = torch.log(u)
        inv_log_u = torch.log(1 - u) - 2 * self.scale
        stack = torch.stack([log_u, inv_log_u], dim=0)
        w = 1 + stack.logsumexp(dim=0) / self.scale
        self.__w = torch.clamp(w, min=-1, max=1)  # Assure w is in [-1, 1].
        return self.__w

    def _sample_w_rej(self, shape: torch.Size) -> Tensor:
        c = torch.sqrt((4 * (self.scale**2)) + (self.p - 1)**2)
        b_true = (-2 * self.scale + c) / (self.p - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.p - 1) / (4 * self.scale)
        s = torch.min(torch.max(torch.tensor([0.], device=self.device), self.scale - 10),
                      torch.tensor([1.], device=self.device))
        b = b_app * s + b_true * (1 - s)

        a = (self.p - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.p - 1) * math.log(self.p - 1)

        self.__b, (self.__e, self.__w) = b, self._while_loop(b, a, d, shape)
        return self.__w

    def _while_loop(self, b: Tensor, a: Tensor, d: Tensor, shape: torch.Size) -> Tuple[Tensor, Tensor]:
        b, a, d = [e.repeat(*shape, *([1] * len(self.scale.shape))) for e in (b, a, d)]
        w, e, bool_mask = torch.zeros_like(b).to(self.device), torch.zeros_like(b).to(
            self.device), (torch.ones_like(b) == 1).to(self.device)

        shape = torch.Size(shape + torch.Size(self.scale.shape))

        while bool_mask.sum() != 0:
            e_ = torch.distributions.Beta((self.p - 1) / 2,
                                          (self.p - 1) / 2).sample(shape[:-1]).reshape(shape).to(self.device)
            u = self.uniform.sample(shape).to(self.device)

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.p - 1) * t.log() - t + d) > torch.log(u)
            reject = 1 - accept

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e, w

    def _householder_rotation(self, x: Tensor) -> Tensor:
        # z = Ux = (I - 2uu^T)x = x - 2(u^T x)u
        z = x - 2 * (x * self.u).sum(-1, keepdim=True) * self.u
        return z

    def entropy(self) -> Tensor:
        ive_ = ive((self.p / 2) - 1, self.scale)
        output = -self.scale * ive(self.p / 2, self.scale) / ive_
        return output.view(*(output.shape[:-1])) - self._c_p_kappa(self.scale, p=self.p, ive_precomp=ive_)

    def log_prob(self, x: Tensor) -> Tensor:
        assert torch.norm(x, p=2, dim=-1).allclose(torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device))
        expprob = self._log_unnormalized_prob(x)
        norm_const = self._c_p_kappa(self.scale, p=self.p)
        output = expprob + norm_const

        # Alternative equally good way to calculate it:
        # expprob2 = (self.loc * x).sum(dim=-1, keepdim=False)
        # ln_kappa = torch.log(self.scale)
        # ln2pi = math.log(math.pi) + math.log(2)
        # ln_ive_ = torch.log(ive(self.p / 2 - 1, self.scale))
        # norm_const2 = self.p * (ln_kappa - ln2pi) / 2. - ln_kappa - ln_ive_
        # output2 = self.scale * (expprob2 - 1) + norm_const2

        return output

    def _log_unnormalized_prob(self, x: Tensor) -> Tensor:  # log(e^(k*u^Tx)) = k*u^Tx
        output = self.scale * (self.loc * x).sum(dim=-1, keepdim=True)
        return output.view(*(output.shape[:-1]))

    @staticmethod
    def _c_p_kappa(kappa: Tensor, p: int, ive_precomp: Optional[Tensor] = None) -> Tensor:
        # https://en.wikipedia.org/wiki/Von_Mises–Fisher_distribution
        ln_kappa = torch.log(kappa)  # log(kappa)
        if ive_precomp is not None:
            ive_ = ive_precomp
        else:
            ive_ = ive(p / 2 - 1, kappa)
        ln_ive_ = torch.log(ive_)
        ln_iv_ = ln_ive_ + kappa
        ln2pi = math.log(2 * math.pi)

        output1 = p * (ln_kappa - ln2pi) / 2. - ln_kappa - ln_iv_  # Same as output3.
        # output2 = torch.log(kappa**(p/2 + 1) / ((2*math.pi)**(p/2) * (ive_ * math.exp(kappa))))  # Too imprecise.
        # output3 = (p / 2 - 1) * ln_kappa - (p / 2) * ln2pi - ln_iv_

        return output1.view(*(output1.shape[:-1]))


class RadiusVonMisesFisher(VonMisesFisher):

    def __init__(self, loc: Tensor, scale: Tensor, radius: Tensor, validate_args: Optional[bool] = None) -> None:
        self.radius = radius
        assert torch.norm(loc, p=2,
                          dim=-1).allclose(self.radius * torch.ones(loc.shape[:-1], dtype=loc.dtype, device=loc.device))

        self.unnormalized_loc = loc
        self.normalized_loc = F.normalize(loc, dim=-1, p=2)
        self.unnormalized_scale = scale

        super().__init__(self.normalized_loc, scale, validate_args)

    def log_prob(self, x: Tensor) -> Tensor:
        assert torch.norm(x, p=2,
                          dim=-1).allclose(self.radius * torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device))
        return super().log_prob(x / self.radius)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        rsampled = super().rsample(sample_shape)
        assert torch.norm(rsampled, p=2,
                          dim=-1).allclose(torch.ones(rsampled.shape[:-1], dtype=rsampled.dtype,
                                                      device=rsampled.device))
        return self.radius * rsampled


class RadiusProjectedVonMisesFisher(RadiusVonMisesFisher):

    def log_prob(self, x_proj: Tensor) -> Tensor:
        x = SP.projected_to_spherical(x_proj, self.radius)
        return super().log_prob(x)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        rsampled = super().rsample(sample_shape)
        return S.spherical_to_projected(rsampled, self.radius)


@register_kl(RadiusVonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf: RadiusVonMisesFisher, hyu: HypersphericalUniform) -> Tensor:
    kl = -vmf.entropy() + hyu.entropy()  # Doesn't depend on radius, only implicitly through scale.
    return kl
