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

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .vae import ModelVAE
from ...data import VaeDataset
from ..components import Component


class FeedForwardVAE(ModelVAE):

    def __init__(self, h_dim: int, components: List[Component], dataset: VaeDataset,
                 scalar_parametrization: bool) -> None:
        super().__init__(h_dim, components, dataset, scalar_parametrization)

        self.in_dim = dataset.in_dim

        # 1 hidden layer encoder
        self.fc_e0 = nn.Linear(dataset.in_dim, h_dim)

        # 1 hidden layer decoder
        self.fc_d0 = nn.Linear(self.total_z_dim, h_dim)
        self.fc_logits = nn.Linear(h_dim, dataset.in_dim)

    def encode(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2
        bs, dim = x.shape
        assert dim == self.in_dim
        x = x.view(bs, self.in_dim)

        x = torch.relu(self.fc_e0(x))

        return x.view(bs, -1)

    def decode(self, concat_z: Tensor) -> Tensor:
        assert len(concat_z.shape) >= 2
        bs = concat_z.size(-2)

        x = torch.relu(self.fc_d0(concat_z))
        x = self.fc_logits(x)

        x = x.view(-1, bs, self.in_dim)  # flatten
        return x.squeeze(dim=0)  # in case we're not doing LL estimation
