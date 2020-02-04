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

from typing import Dict, Tuple, TypeVar, Type, Optional

import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.nn.functional as F

from ..ops import Manifold, PoincareBall, Hyperboloid, Sphere, StereographicallyProjectedSphere, Euclidean, Universal
from ..sampling import SamplingProcedure

Q = TypeVar('Q', bound=Distribution)
P = TypeVar('P', bound=Distribution)


class Component(torch.nn.Module):

    def forward(self, x: Tensor) -> Tuple[Q, P, Tuple[Tensor, ...]]:
        z_params = self.encode(x)
        q_z, p_z = self.reparametrize(*z_params)
        return q_z, p_z, z_params

    def __init__(self, dim: int, fixed_curvature: bool, sampling_procedure: Type[SamplingProcedure[Q, P]]) -> None:
        super().__init__()
        self.dim = dim
        self.fixed_curvature = fixed_curvature
        self._sampling_procedure_type = sampling_procedure
        self.sampling_procedure: SamplingProcedure[Q, P] = None
        self.manifold: Manifold = None

        self.fc_mean: torch.nn.Linear = None
        self.fc_logvar: torch.nn.Linear = None

    def init_layers(self, in_dim: int, scalar_parametrization: bool) -> None:
        self.manifold = self.create_manifold()
        self.sampling_procedure = self._sampling_procedure_type(self.manifold, scalar_parametrization)

        self.fc_mean = torch.nn.Linear(in_dim, self.mean_dim)

        if scalar_parametrization:
            self.fc_logvar = torch.nn.Linear(in_dim, 1)
        else:
            self.fc_logvar = torch.nn.Linear(in_dim, self.true_dim)

    @property
    def device(self) -> torch.device:
        return self.fc_mean.weight.device

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z_mean = self.fc_mean(x)
        assert torch.isfinite(z_mean).all()
        z_mean_h = self.manifold.exp_map_mu0(z_mean)
        assert torch.isfinite(z_mean_h).all()

        z_logvar = self.fc_logvar(x)
        assert torch.isfinite(z_logvar).all()
        # +eps prevents collapse
        std = F.softplus(z_logvar) + 1e-5
        # std = std / (self.manifold.radius**self.true_dim)  # TODO: Incorporate radius for (P)VMF
        assert torch.isfinite(std).all()
        return z_mean_h, std

    def reparametrize(self, z_mean: Tensor, z_logvar: Tensor) -> Tuple[Q, P]:
        return self.sampling_procedure.reparametrize(z_mean, z_logvar)

    def kl_loss(self, q_z: Q, p_z: P, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        return self.sampling_procedure.kl_loss(q_z, p_z, z, data)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(R^{self.dim})"

    def _shortcut(self) -> str:
        return f"{self.__class__.__name__.lower()[0]}{self.true_dim}"

    def summary_name(self, comp_idx: int) -> str:
        return f"comp_{comp_idx:03d}_{self._shortcut()}"

    def summaries(self, comp_idx: int, q_z: Q, prefix: str = "train") -> Dict[str, Tensor]:
        name = prefix + "/" + self.summary_name(comp_idx)
        return {
            name + "/mean/norm": torch.norm(q_z.mean, p=2, dim=-1),
            name + "/stddev/norm": torch.norm(q_z.stddev, p=2, dim=-1),
        }

    def create_manifold(self) -> Manifold:
        raise NotImplementedError

    @property
    def true_dim(self) -> int:
        raise NotImplementedError

    @property
    def mean_dim(self) -> int:
        return self.true_dim


class HyperbolicComponent(Component):

    def __init__(self,
                 dim: int,
                 fixed_curvature: bool,
                 sampling_procedure: Type[SamplingProcedure[Q, P]],
                 radius: float = 1.0) -> None:
        # Add one to the dimension here on purpose.
        super().__init__(dim + 1, fixed_curvature, sampling_procedure)
        self._nradius = torch.nn.Parameter(torch.tensor(radius), requires_grad=not fixed_curvature)

    def create_manifold(self) -> Manifold:
        return Hyperboloid(lambda: self._nradius)

    @property
    def true_dim(self) -> int:
        return self.dim - 1


class PoincareComponent(Component):

    def __init__(self,
                 dim: int,
                 fixed_curvature: bool,
                 sampling_procedure: Type[SamplingProcedure[Q, P]],
                 radius: float = 1.0) -> None:
        # Add one to the dimension here on purpose.
        super().__init__(dim, fixed_curvature, sampling_procedure)
        self._nradius = torch.nn.Parameter(torch.tensor(radius), requires_grad=not fixed_curvature)

    def create_manifold(self) -> Manifold:
        return PoincareBall(lambda: self._nradius)

    @property
    def true_dim(self) -> int:
        return self.dim


class SphericalComponent(Component):

    def __init__(self,
                 dim: int,
                 fixed_curvature: bool,
                 sampling_procedure: Type[SamplingProcedure[Q, P]],
                 radius: float = 1.0) -> None:
        super().__init__(dim + 1, fixed_curvature, sampling_procedure)  # Add one to the dimension here on purpose.
        self._pradius = torch.nn.Parameter(torch.tensor(radius), requires_grad=not fixed_curvature)

    def create_manifold(self) -> Manifold:
        return Sphere(lambda: self._pradius)

    @property
    def true_dim(self) -> int:
        return self.dim - 1


class StereographicallyProjectedSphereComponent(Component):

    def __init__(self,
                 dim: int,
                 fixed_curvature: bool,
                 sampling_procedure: Type[SamplingProcedure[Q, P]],
                 radius: float = 1.0) -> None:
        # Add one to the dimension here on purpose.
        super().__init__(dim, fixed_curvature, sampling_procedure)
        self._pradius = torch.nn.Parameter(torch.tensor(radius), requires_grad=not fixed_curvature)

    def create_manifold(self) -> Manifold:
        return StereographicallyProjectedSphere(lambda: self._pradius)

    @property
    def true_dim(self) -> int:
        return self.dim

    def _shortcut(self) -> str:
        return f"d{self.true_dim}"


class EuclideanComponent(Component):

    def __init__(self, dim: int, fixed_curvature: bool, sampling_procedure: Type[SamplingProcedure[Q, P]]) -> None:
        # Euclidean component always has fixed curvature.
        super().__init__(dim, fixed_curvature=True, sampling_procedure=sampling_procedure)

    def create_manifold(self) -> Manifold:
        return Euclidean()

    @property
    def true_dim(self) -> int:
        return self.dim


class ConstantComponent(Component):

    def __init__(self,
                 dim: int,
                 fixed_curvature: bool,
                 sampling_procedure: Type[SamplingProcedure[Q, P]],
                 const: Optional[Tensor] = None,
                 eps: Optional[Tensor] = None) -> None:
        # Constant component always has fixed curvature.
        super().__init__(dim, fixed_curvature=False, sampling_procedure=sampling_procedure)

    def create_manifold(self) -> Manifold:
        return Euclidean()

    @property
    def true_dim(self) -> int:
        return self.dim


class UniversalComponent(Component):

    def __init__(self,
                 dim: int,
                 fixed_curvature: bool,
                 sampling_procedure: Type[SamplingProcedure[Q, P]],
                 curvature: float = 0.0,
                 eps: float = 1e-6) -> None:
        super().__init__(dim, fixed_curvature, sampling_procedure)  # Add one to the dimension here on purpose.
        self._curvature = torch.nn.Parameter(torch.tensor(curvature), requires_grad=not fixed_curvature)
        self._eps = eps

    def create_manifold(self) -> Manifold:
        return Universal(lambda: self._curvature, eps=self._eps)

    @property
    def true_dim(self) -> int:
        return self.dim
