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

from typing import List, Tuple

import functools
import torch
import torch.nn as nn
from torch import Tensor

from .vae import ModelVAE
from ...data import VaeDataset
from ..components import Component


class ConvolutionalVAE(ModelVAE):

    def __init__(self,
                 h_dim: int,
                 components: List[Component],
                 dataset: VaeDataset,
                 scalar_parametrization: bool,
                 img_dims: Tuple[int, int, int] = (3, 32, 32)) -> None:
        super().__init__(h_dim, components, dataset, scalar_parametrization)

        ks = 4
        st = 2
        pad = 1

        self.img_dims = img_dims
        self.img_dims_flat = functools.reduce((lambda x, y: x * y), self.img_dims)
        in_channels = img_dims[0]

        # encoder
        self.e0 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=ks, stride=st, padding=pad)
        self.e1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, stride=st, padding=pad)
        self.e2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=ks, stride=st, padding=pad)

        # decoder
        self.d0 = nn.Linear(self.total_z_dim, 2048)
        self.d1 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=ks, stride=st, padding=pad)
        self.d2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=ks, stride=st, padding=pad)
        self.d3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=ks, stride=st, padding=pad)

    def encode(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2
        bs, dim = x.shape
        assert dim == self.img_dims_flat
        x = x.view((bs,) + self.img_dims)  # make into (N, (C, H, W))
        x = torch.relu(self.e0(x))
        x = torch.relu(self.e1(x))
        x = torch.relu(self.e2(x))
        x = x.view(bs, -1)  # flatten
        return x

    def decode(self, concat_z: Tensor) -> Tensor:
        assert len(concat_z.shape) >= 2
        bs = concat_z.size(-2)

        x = torch.relu(self.d0(concat_z))
        x = x.view(-1, 128, 4, 4)
        x = torch.relu(self.d1(x))
        x = torch.relu(self.d2(x))
        x = self.d3(x)

        x = x.view(-1, bs, self.img_dims_flat)  # flatten
        return x.squeeze(dim=0)  # in case we're not doing LL estimation
