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

from typing import Dict, Tuple, Optional, Any

import json
import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader


class RepresentationDataset:
    """
    To load a dataset using this class, the data directory needs to have the following structure:
    * meta.txt -- JSON file with information about labels. E.g. for CIFAR-10:
        ```
        {"meta":{
            "labels": [0,1,2,3,4,5,6,7,8,9],
            "label_names": ["airplane","automobile","bird","cat","deer","dog","frog", "horse","ship","truck"]
        }}
        ```
    * train_data.pt -- Float Tensor of shape (N, D)
    * train_labels.pt -- Int Tensor of shape (N,)
    * test_data.pt -- Float Tensor of shape (N_Test, D)
    * test_labels.pt -- Int Tensor of shape (N_Test,)
    """

    def __init__(self, data_folder: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.data_folder = data_folder
        self.datasets: Dict[bool, torch.utils.data.TensorDataset] = {}

        if meta is None:
            meta_dict = self._read_meta()
        else:
            meta_dict = meta
        self.labels = meta_dict["labels"]
        self.label_names = meta_dict["label_names"]
        self.n_classes = len(self.labels)
        if len(self.labels) != len(self.label_names):
            raise ValueError(
                f"Labels and label names have to be of same length ({len(self.labels)} != {len(self.label_names)}).")

    def _read_meta(self) -> Dict[str, Any]:
        filepath = os.path.join(self.data_folder, "meta.txt")
        with open(filepath, "r") as file:
            meta_dict = json.load(file)
        return meta_dict["meta"]

    def _write_meta(self, d: Dict[str, Any]) -> None:
        filepath = os.path.join(self.data_folder, "meta.txt")
        meta_dict = {"meta": d}

        with open(filepath, "w") as file:
            file.write(json.dumps(meta_dict))

    def __len__(self) -> int:
        return self.datasets[False].tensors[0].size(0)

    @property
    def dim(self) -> int:
        return self.datasets[False].tensors[0].size(1)

    @staticmethod
    def _check_dimensions(data: Tensor, labels: Tensor) -> None:
        if data.size(0) != labels.size(0):
            raise ValueError("Data and labels have to be of same length.")
        if len(data.shape) != 2:
            raise ValueError("Data has to be two dimensional (length, dimensionality).")
        if len(labels.shape) != 1:
            raise ValueError("Labels have to be one dimensional.")

    def load(self, train: bool) -> None:
        train_str = "train" if train else "test"
        with torch.no_grad():
            data = torch.load(os.path.join(self.data_folder, f"{train_str}_data.pt"))
            data.requires_grad = False
            labels = torch.load(os.path.join(self.data_folder, f"{train_str}_labels.pt"))
            labels.requires_grad = False
            RepresentationDataset._check_dimensions(data, labels)
            self.datasets[train] = torch.utils.data.TensorDataset(data, labels)

    def create_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:

        def _create_loader(train: bool, batch_size: int) -> DataLoader:
            return DataLoader(dataset=self.datasets[train],
                              batch_size=batch_size,
                              num_workers=8,
                              pin_memory=True,
                              shuffle=train)

        train_loader = _create_loader(train=True, batch_size=batch_size)
        test_loader = _create_loader(train=False, batch_size=batch_size)
        return train_loader, test_loader
