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

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Callable, Any, Optional

import numpy as np
import torch
import torch.backends.cudnn
from torch.optim import Optimizer

from .components import Component, ConstantComponent, EuclideanComponent, StereographicallyProjectedSphereComponent
from .components import SphericalComponent, UniversalComponent, HyperbolicComponent, PoincareComponent
from .sampling import WrappedNormalProcedure, EuclideanNormalProcedure, EuclideanConstantProcedure
from .sampling import UniversalSamplingProcedure
# from .sampling import SphericalVmfProcedure, ProjectedSphericalVmfProcedure, RiemannianNormalProcedure

space_creator_map = {
    "h": HyperbolicComponent,
    "u": UniversalComponent,
    "s": SphericalComponent,
    "d": StereographicallyProjectedSphereComponent,
    "p": PoincareComponent,
    "c": ConstantComponent,
    "e": EuclideanComponent,
}

sampling_procedure_map = {
    SphericalComponent: WrappedNormalProcedure,
    StereographicallyProjectedSphereComponent: WrappedNormalProcedure,
    EuclideanComponent: EuclideanNormalProcedure,
    ConstantComponent: EuclideanConstantProcedure,
    HyperbolicComponent: WrappedNormalProcedure,
    PoincareComponent: WrappedNormalProcedure,
    UniversalComponent: UniversalSamplingProcedure,
}


def setup_gpu(device: torch.device) -> None:
    if device != torch.device("cpu"):
        torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False, verbose=False)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, verbose=False)
    np.random.seed(seed)


def canonical_name(components: List[Component]) -> str:
    spaces_dims: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for component in components:
        component_type = component._shortcut()[0]
        spaces_dims[component_type][component.true_dim] += 1

    def _serialize_components(spaces_dims: Dict[str, Dict[int, int]]) -> Iterable[str]:
        for component_type in sorted(spaces_dims.keys()):
            typed_components = spaces_dims[component_type]
            for true_dim in sorted(typed_components.keys()):
                multiplier = typed_components[true_dim]
                yield f"{multiplier if multiplier > 1 else ''}{component_type}{true_dim}"

    return ",".join(_serialize_components(spaces_dims))


def parse_component_str(space_str: str) -> Tuple[int, str, int]:
    space_str = space_str.split("-")[0]
    i = 0
    multiplier = ""
    while i < len(space_str):
        if '0' <= space_str[i] <= '9':
            i += 1
        else:
            multiplier = space_str[:i]
            break

    space_type = ""
    while i < len(space_str):
        if 'a' <= space_str[i] <= 'z':
            i += 1
        else:
            space_type = space_str[len(multiplier):i]
            break

    dimension = space_str[i:]
    if not multiplier:
        multiplier = "1"
    return int(multiplier), space_type, int(dimension)


def parse_components(arg: str, fixed_curvature: bool) -> List[Component]:
    """
    Parses command line specifications of a model. The argument is a comma separated list of the letters "e", "h", "s",
    concatenated with dimensions of the respective component. As an example, a VAE with 2 dimensional euclidean
    component, 3 dimensional hyperbolic component, and 4 dimensional spherical component would be described by the
    string: "e2,h3,s4".

    :param arg: Model latent component description string.
    :param fixed_curvature: Whether to use fixed (-1, 0, 1) curvatures or learnable ones.
    :return: A list of components according to the description.
    """

    def _create_space(space_multiplier: int, space_type: str, dim: int,
                      fixed_curvature: bool) -> Iterable[Component]:  # type: ignore
        if space_multiplier < 1:
            raise ValueError(f"Space multiplier has to be at least 1, was: '{space_multiplier}'.")
        if dim < 1:
            raise ValueError(f"Dimension has to be at least 1, was: '{dim}'.")

        if space_type not in space_creator_map:
            raise NotImplementedError(f"Unknown latent space type '{space_type}'.")
        space_creator = space_creator_map[space_type]

        for _ in range(space_multiplier):
            yield space_creator(dim, fixed_curvature, sampling_procedure=sampling_procedure_map[space_creator])

    arg = arg.lower().strip()
    if not arg:
        return []

    space_strings = [space_str.strip() for space_str in arg.split(",")]
    spaces = [parse_component_str(space_str) for space_str in space_strings]

    components = []
    for multiplier, space, dim in spaces:
        for component in _create_space(multiplier, space, dim, fixed_curvature):
            components.append(component)
    return components


def linear_betas(start: float, end: float, end_epoch: int, epochs: int) -> np.ndarray:
    return np.concatenate((np.linspace(start, end, num=end_epoch), end * np.ones(
        (epochs - end_epoch,), dtype=np.float32)))


class CurvatureOptimizer(Optimizer):

    def __init__(self,
                 optimizer: Optimizer,
                 neg: Optional[Optimizer],
                 pos: Optional[Optimizer],
                 should_do_curvature_step: Callable[[], bool] = lambda: False) -> None:
        neg_params: List[Any] = []
        if neg is not None:
            neg_params = neg.param_groups
        pos_params: List[Any] = []
        if pos is not None:
            pos_params = pos.param_groups
        super().__init__(optimizer.param_groups + neg_params + pos_params, {})
        self.optimizer = optimizer
        self.pos = pos
        self.neg = neg
        self.curv_condition = should_do_curvature_step

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()
        if self.neg is not None:
            self.neg.zero_grad()
        if self.pos is not None:
            self.pos.zero_grad()

    def step(self, closure: Optional[Any] = None) -> None:
        self.optimizer.step(closure)
        if self.curv_condition():
            if self.pos is not None:
                self.pos.step(closure)
            if self.neg is not None:
                self.neg.step(closure)
