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

from typing import Dict, Iterable, List, Set
from collections import defaultdict

import argparse


def canonical_name(components: List[str]) -> str:
    spaces_dims: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for component in components:
        component_type = component[0]
        spaces_dims[component_type][int(component[1:])] += 1

    def _serialize_components(spaces_dims: Dict[str, Dict[int, int]]) -> Iterable[str]:
        for component_type in sorted(spaces_dims.keys()):
            typed_components = spaces_dims[component_type]
            for true_dim in sorted(typed_components.keys()):
                multiplier = typed_components[true_dim]
                yield f"{multiplier if multiplier > 1 else ''}{component_type}{true_dim}"

    return ",".join(_serialize_components(spaces_dims))


def all_models(dim: int, types: List[str], min_dim: int = 2) -> Iterable[str]:
    if dim < min_dim:
        return []

    for d in range(min_dim, dim):
        for model in all_models(dim - d, types):
            for t in types:
                yield f"{t}{d},{model}"
    for t in types:
        yield f"{t}{dim}"


def all_models_pub(dim: int = 5, types: List[str] = ["h", "s", "e"]) -> Iterable[str]:
    seen: Set[str] = set()
    for m in all_models(dim=dim, types=types, min_dim=5):
        cn = canonical_name(m.split(","))
        if cn in seen:
            continue
        seen.add(cn)
        yield cn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=5, help="latent space total dim")
    args = parser.parse_args()

    models = sorted(all_models_pub(dim=args.dim))
    for model in models:
        print(model)
