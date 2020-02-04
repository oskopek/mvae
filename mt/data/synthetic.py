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

from typing import Any, List, Tuple

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.distributions import Normal

from .vae_dataset import VaeDataset


class BinaryDiffusionProcessDataset(torch.utils.data.Dataset):
    """
    Implementation of a synthetic dataset by hierarchical diffusion.

    Args:
        dim: dimension of the input sample
        depth: depth of the tree; the root corresponds to the depth 0
        number_of_children: Number of children of each node in the tree
        number_of_siblings: Number of noisy observations obtained from the nodes of the tree
        sigma_children: noise
        param: integer by which :math:`\\sigma_{\\text{children}}` is divided at each deeper level of the tree
    """

    def __init__(self,
                 dim: int,
                 depth: int,
                 number_of_children: int = 2,
                 sigma_children: float = 1,
                 param: int = 1,
                 number_of_siblings: int = 1,
                 factor_sibling: float = 10) -> None:
        self.dim = int(dim)
        self.root = np.zeros(self.dim)
        self.depth = int(depth)
        self.sigma_children = sigma_children
        self.factor_sibling = factor_sibling
        self.param = param
        self.number_of_children = int(number_of_children)
        self.number_of_siblings = int(number_of_siblings)

        self.origin_data, self.origin_labels, self.data, self.labels = self.bst()

        # Normalise data (0 mean, 1 std)
        self.data -= np.mean(self.data, axis=0, keepdims=True)
        self.data /= np.std(self.data, axis=0, keepdims=True)

    def __len__(self) -> int:
        """
        this method returns the total number of samples/nodes
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates one sample
        """
        data, labels = self.data[idx], self.labels[idx]
        return torch.Tensor(data), torch.Tensor(labels)

    def get_children(self,
                     parent_value: np.ndarray,
                     parent_label: np.ndarray,
                     current_depth: int,
                     offspring: bool = True) -> List[Tuple[float, np.ndarray]]:
        """
        :param 1d-array parent_value
        :param 1d-array parent_label
        :param int current_depth
        :param  Boolean offspring: if True the parent node gives birth to number_of_children nodes
                                    if False the parent node gives birth to number_of_siblings noisy observations
        :return: list of 2-tuples containing the value and label of each child of a parent node
        :rtype: list of length number_of_children
        """
        if offspring:
            number_of_children = self.number_of_children
            sigma = self.sigma_children / (self.param**current_depth)
        else:
            number_of_children = self.number_of_siblings
            sigma = self.sigma_children / (self.factor_sibling * (self.param**current_depth))
        children = []
        for i in range(number_of_children):
            child_value = parent_value + np.random.randn(self.dim) * np.sqrt(sigma)
            child_label = np.copy(parent_label)
            if offspring:
                child_label[current_depth] = i + 1
            else:
                child_label[current_depth] = -i - 1
            children.append((child_value, child_label))
        return children

    def bst(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This method generates all the nodes of a level before going to the next level
        """
        queue = [(self.root, np.zeros(self.depth + 1), 0)]
        visited = []
        labels_visited = []
        values_clones = []
        labels_clones = []
        while len(queue) > 0:
            current_node, current_label, current_depth = queue.pop(0)
            visited.append(current_node)
            labels_visited.append(current_label)
            if current_depth < self.depth:
                children = self.get_children(current_node, current_label, current_depth)
                for child in children:
                    queue.append((child[0], child[1], current_depth + 1))
            if current_depth <= self.depth:
                clones = self.get_children(current_node, current_label, current_depth, False)
                for clone in clones:
                    values_clones.append(clone[0])
                    labels_clones.append(clone[1])
        length = int((self.number_of_children**(self.depth + 1) - 1) / (self.number_of_children - 1))
        images = np.concatenate([i for i in visited]).reshape(length, self.dim)
        labels_visited = np.concatenate([i for i in labels_visited]).reshape(length, self.depth + 1)[:, :self.depth]
        values_clones = np.concatenate([i for i in values_clones]).reshape(self.number_of_siblings * length, self.dim)
        labels_clones = np.concatenate([i for i in labels_clones]).reshape(self.number_of_siblings * length,
                                                                           self.depth + 1)
        return images, labels_visited, values_clones, labels_clones


class BdpVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(batch_size, in_dim=50, img_dims=None)

    def _load_synth(self, dataset: BinaryDiffusionProcessDataset, train: bool = True) -> DataLoader:
        return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        dataset = BinaryDiffusionProcessDataset(self.in_dim,
                                                5,
                                                number_of_children=2,
                                                sigma_children=1,
                                                param=1,
                                                number_of_siblings=5,
                                                factor_sibling=10)
        n_train = int(len(dataset) * 0.7)
        n_test = len(dataset) - n_train
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        train_loader = self._load_synth(train_dataset, train=True)
        test_loader = self._load_synth(test_dataset, train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return -Normal(x_mb_, torch.ones_like(x_mb_)).log_prob(x_mb)
