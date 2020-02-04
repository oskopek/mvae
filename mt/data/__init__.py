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

from typing import Any

from .image_reconstruction import CifarVaeDataset, MnistVaeDataset, OmniglotVaeDataset
from .synthetic import BdpVaeDataset
from .vae_dataset import VaeDataset

__all__ = [
    "BdpVaeDataset",
    "CifarVaeDataset",
    "MnistVaeDataset",
    "OmniglotVaeDataset",
    "VaeDataset",
    "create_dataset",
]


def create_dataset(dataset_type: str, *args: Any, **kwargs: Any) -> VaeDataset:
    if dataset_type == "bdp":
        return BdpVaeDataset(*args, **kwargs)
    elif dataset_type == "mnist":
        return MnistVaeDataset(*args, **kwargs)
    elif dataset_type == "omniglot":
        return OmniglotVaeDataset(*args, **kwargs)
    elif dataset_type == "cifar":
        return CifarVaeDataset(*args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'.")
