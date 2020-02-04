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

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader


class VaeDataset:

    def __init__(self, batch_size: int, in_dim: int, img_dims: Optional[Tuple[int, ...]]) -> None:
        self.batch_size = batch_size
        self._in_dim = in_dim
        self._img_dims = img_dims

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    @property
    def img_dims(self) -> Optional[Tuple[int, ...]]:
        return self._img_dims

    @property
    def in_dim(self) -> int:
        return self._in_dim

    def metrics(self, x_mb_: torch.Tensor, mode: str = "train") -> Dict[str, float]:
        return {}
