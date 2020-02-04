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

import os
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from mt.data import VaeDataset, MnistVaeDataset
from mt.mvae.components import EuclideanComponent, SphericalComponent, HyperbolicComponent, Component, PoincareComponent
from mt.mvae.components import ConstantComponent, StereographicallyProjectedSphereComponent, UniversalComponent
from mt.mvae.models import FeedForwardVAE, Trainer
from mt.mvae.stats import EpochStats
from mt.mvae.sampling import EuclideanNormalProcedure as en
from mt.mvae.sampling import WrappedNormalProcedure as wn
from mt.mvae.sampling import SphericalVmfProcedure as svmf
from mt.mvae.sampling import ProjectedSphericalVmfProcedure as pvmf
from mt.mvae.sampling import RiemannianNormalProcedure as rn
from mt.mvae.sampling import UniversalSamplingProcedure as usp

DATA_TYPE = Tuple[torch.Tensor, torch.Tensor]
eps = 1e1  # TODO-LATER: This should be lower.


class FakeVaeDataset(VaeDataset):

    def __init__(self) -> None:
        super().__init__(batch_size=2, in_dim=2, img_dims=(-1, 1, 2, 2))

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")

    @staticmethod
    def _tensorize(data: List[Tuple[List, int]]) -> List[DATA_TYPE]:
        # Convert into a tensor and add the batch dimension.
        return [(torch.Tensor(x), torch.Tensor(y)) for x, y in data]

    def create_loaders(self) -> Tuple[List[DATA_TYPE], List[DATA_TYPE]]:

        class LenSetter:

            def __init__(self, len: int) -> None:
                self._len = len

            def __len__(self) -> int:
                return self._len

        class FakeDataLoader(list):

            def __init__(self, len: int, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)
                self.dataset = LenSetter(len)

        len = 1000
        train_data = [([[1, 0], [0, 1]], [0, 0])] * 5 + [([[0, 0], [1, 1]], [1, 1, 1])] * 5
        eval_data = [([[1, 0], [0, 1]], [0, 0])] * 5 + [([[0, 0], [1, 1]], [1, 1, 1])] * 5
        return FakeDataLoader(len, FakeVaeDataset._tensorize(train_data)), FakeDataLoader(
            len, FakeVaeDataset._tensorize(eval_data))


class MnistVaeDatasetForTest(MnistVaeDataset):

    def __init__(self, batch_size: int, data_folder: str, take_n: int) -> None:
        super().__init__(batch_size, data_folder)
        self.take_n = take_n

    def _get_dataset(self, data_folder: str, train: bool, download: bool, transform: Any) -> torch.utils.data.Dataset:
        return torch.utils.data.Subset(super()._get_dataset(data_folder, train, download, transform),
                                       list(range(self.take_n)))


class FakeVaeDimDataset(VaeDataset):

    def __init__(self, in_dim: int) -> None:
        super().__init__(batch_size=1, in_dim=in_dim, img_dims=None)

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        pass

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        pass


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def num_of_trainable_params(model: nn.Module) -> int:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def count_euclidean_vae_params(in_dim: int, h_dim: int, component_dims: List[int]) -> int:
    params = 0

    layers = [h_dim]
    prev_layer = in_dim
    for layer in layers:  # Encoder FC layers
        params += prev_layer * layer + layer
        prev_layer = layer

    for component in component_dims:  # Components
        params += component * prev_layer + component  # hidden layer to comp dim + bias
        params += 1 * prev_layer + 1  # hidden layer to 1 dim for variance + bias
        # params += 1  # 1 param for curvature

    prev_layer = sum(component_dims)
    for layer in layers[::-1] + [in_dim]:  # Decoder FC layers
        params += prev_layer * layer + layer
        prev_layer = layer
    return params


def test_num_of_params_euclidean() -> None:
    in_dim = 5
    h_dim = 10

    small_component = EuclideanComponent(dim=2, fixed_curvature=False, sampling_procedure=en)
    small_vae = FeedForwardVAE(h_dim,
                               dataset=FakeVaeDimDataset(in_dim),
                               components=[small_component],
                               scalar_parametrization=True)
    small_params = num_of_trainable_params(small_vae)
    small_params_expected = count_euclidean_vae_params(in_dim, h_dim, component_dims=[2])
    assert small_params == small_params_expected

    big_component = EuclideanComponent(dim=5, fixed_curvature=False, sampling_procedure=en)
    big_vae = FeedForwardVAE(h_dim,
                             dataset=FakeVaeDimDataset(in_dim),
                             components=[big_component],
                             scalar_parametrization=True)
    big_params = num_of_trainable_params(big_vae)
    big_params_expected = count_euclidean_vae_params(in_dim, h_dim, component_dims=[5])
    assert big_params == big_params_expected

    assert big_params > small_params
    assert len(list(small_vae.parameters())) == len(list(big_vae.parameters()))


@pytest.mark.parametrize("component_type", [SphericalComponent, HyperbolicComponent])
def test_num_of_params_fixed_curvature(component_type: Type[Component]) -> None:
    in_dim = 5
    h_dim = 10

    fixed_components = [
        component_type(dim=5, fixed_curvature=True, sampling_procedure=wn),
        component_type(dim=5, fixed_curvature=True, sampling_procedure=wn)
    ]
    fixed_vae = FeedForwardVAE(h_dim,
                               dataset=FakeVaeDimDataset(in_dim),
                               components=fixed_components,
                               scalar_parametrization=True)

    learn_components = [
        component_type(dim=5, fixed_curvature=False, sampling_procedure=wn),
        component_type(dim=5, fixed_curvature=False, sampling_procedure=wn)
    ]
    learn_vae = FeedForwardVAE(h_dim,
                               dataset=FakeVaeDimDataset(in_dim),
                               components=learn_components,
                               scalar_parametrization=True)

    fixed_params = num_of_trainable_params(fixed_vae)
    learn_params = num_of_trainable_params(learn_vae)

    def num_of_trainable_param_tensors(vae: FeedForwardVAE) -> int:
        return sum(1 for p in vae.parameters() if p.requires_grad)

    # One curvature param per component.
    assert learn_params - 2 == fixed_params
    assert num_of_trainable_param_tensors(learn_vae) - 2 == num_of_trainable_param_tensors(fixed_vae)


def test_num_of_layers() -> None:
    in_dim = 5
    h_dim = 10
    small_components = [
        EuclideanComponent(dim=2, fixed_curvature=True, sampling_procedure=en),
        EuclideanComponent(dim=2, fixed_curvature=True, sampling_procedure=en)
    ]
    big_components = [EuclideanComponent(dim=5, fixed_curvature=True, sampling_procedure=en)]

    small_vae = FeedForwardVAE(h_dim,
                               dataset=FakeVaeDimDataset(in_dim),
                               components=small_components,
                               scalar_parametrization=True)
    big_vae = FeedForwardVAE(h_dim,
                             dataset=FakeVaeDimDataset(in_dim),
                             components=big_components,
                             scalar_parametrization=True)

    # small_vae has more different parameter matrices
    assert len(list(small_vae.parameters())) > len(list(big_vae.parameters()))


models = {
    "e2": (lambda fc: [EuclideanComponent(2, fc, en)], False),
    "s2": (lambda fc: [SphericalComponent(2, fc, svmf)], True),
    "s2-wn": (lambda fc: [SphericalComponent(2, fc, wn)], False),
    "d2-wn": (lambda fc: [StereographicallyProjectedSphereComponent(2, fc, wn)], False),
    "d2-pvmf": (lambda fc: [StereographicallyProjectedSphereComponent(2, fc, pvmf)], True),
    "p2-rn": (lambda fc: [PoincareComponent(2, fc, rn)], True),
    "p2-wn": (lambda fc: [PoincareComponent(2, fc, wn)], False),
    "h2": (lambda fc: [HyperbolicComponent(2, fc, wn)], False),
    "s2,e2":
    (lambda fc: [SphericalComponent(2, fc, svmf), EuclideanComponent(2, fc, en)], True),
    "h2,s2,e2":
    (lambda fc: [HyperbolicComponent(2, fc, wn),
                 SphericalComponent(2, fc, wn),
                 EuclideanComponent(2, fc, en)], False),
    "h2,s2-vmf,e2":
    (lambda fc: [HyperbolicComponent(2, fc, wn),
                 SphericalComponent(2, fc, svmf),
                 EuclideanComponent(2, fc, en)], True),
    "d2-wn,e2,p2-wn": (lambda fc: [
        StereographicallyProjectedSphereComponent(2, fc, wn),
        EuclideanComponent(2, fc, en),
        PoincareComponent(2, fc, wn)
    ], False),
    "d2-pvmf,e2,p2-rn": (lambda fc: [
        StereographicallyProjectedSphereComponent(2, fc, pvmf),
        EuclideanComponent(2, fc, en),
        PoincareComponent(2, fc, rn)
    ], True),
    "h40": (lambda fc: [HyperbolicComponent(40, fc, wn)], False),
    # "s40-vmf": (lambda fc: [SphericalComponent(40, fc, svmf)], True),  # Works poorly.
    "s40-wn": (lambda fc: [SphericalComponent(40, fc, wn)], False),
    # "p20-rn": (lambda fc: [PoincareComponent(20, fc, rn)], True),  # Works poorly.
    "p40-wn": (lambda fc: [PoincareComponent(40, fc, wn)], False),
    # "d40-pvmf": (lambda fc: [StereographicallyProjectedSphereComponent(40, fc, pvmf)], True),  # Works poorly.
    "d40-wn": (lambda fc: [StereographicallyProjectedSphereComponent(40, fc, wn)], False),
    "u2": (lambda fc: [UniversalComponent(2, fc, usp)], False),
}
devices = ["cuda", "cpu"]


def _run_training(model: str, device: str, fixed_curvature: bool, likelihood_n: int = 10,
                  fixed_epochs: int = -1) -> Dict[int, EpochStats]:
    device = torch.device(device)
    components_lam, sp = models[model]
    components = components_lam(fixed_curvature)

    dataset = FakeVaeDataset()
    # dataset = MnistVaeDatasetForTest(batch_size=2, data_folder="./data", take_n=2)
    vae = FeedForwardVAE(dataset=dataset, h_dim=2, components=components, scalar_parametrization=sp).to(device)
    trainer = Trainer(model=vae, img_dims=dataset.img_dims, chkpt_dir=f"./chkpt/test/{model}")
    os.makedirs(trainer.chkpt_dir, exist_ok=True)

    optimizer = Adam(vae.parameters(), lr=1e-3)

    train_data, eval_data = dataset.create_loaders()
    # with torch.autograd.set_detect_anomaly(True):
    if fixed_epochs > 0:
        res = trainer.train_epochs(optimizer,
                                   train_data,
                                   eval_data,
                                   epochs=fixed_epochs,
                                   likelihood_n=likelihood_n,
                                   betas=[1.])
    else:
        res = trainer.train_stopping(optimizer,
                                     train_data,
                                     eval_data,
                                     warmup=1,
                                     lookahead=1,
                                     max_epochs=2,
                                     likelihood_n=likelihood_n,
                                     betas=[1.])
    if len(res) == 2:
        if not fixed_curvature:
            for component in vae.components:
                if not isinstance(component, EuclideanComponent) and not isinstance(component, UniversalComponent):
                    assert not component.manifold.curvature.allclose(torch.tensor(-1., device=device))
                    assert not component.manifold.curvature.allclose(torch.tensor(0., device=device))
                    assert not component.manifold.curvature.allclose(torch.tensor(1., device=device))
        else:
            for component in vae.components:
                if isinstance(component, EuclideanComponent) or isinstance(component, ConstantComponent):
                    assert component.manifold.curvature == 0
                elif isinstance(component, SphericalComponent) or isinstance(component,
                                                                             StereographicallyProjectedSphereComponent):
                    assert component.manifold.curvature.allclose(torch.tensor(0.01, device=device))
                elif isinstance(component, HyperbolicComponent) or isinstance(component, PoincareComponent):
                    assert component.manifold.curvature.allclose(torch.tensor(-0.01, device=device))
                elif isinstance(component, UniversalComponent):
                    assert True
                else:
                    raise ValueError(f"Unknown component type '{component}'.")
    return res


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("fixed_curvature", [True, False])
@pytest.mark.parametrize("fixed_epochs", [-1, 2])
def test_run_training_fixed_epochs(device: str, fixed_curvature: bool, fixed_epochs: int) -> None:
    if not torch.cuda.is_available() and device == "cuda":
        return  # Ignore test if CUDA is not available.

    test_results = _run_training("h2,s2,e2", device, fixed_curvature, fixed_epochs=fixed_epochs)
    assert len(test_results) == 1
    for epoch_results in test_results.values():
        for k, v in epoch_results.to_print().items():
            assert np.isfinite(v), f"{k} is not finite."


@pytest.mark.parametrize("model", models.keys())
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("fixed_curvature", [True, False])
def test_run_training(model: str, device: str, fixed_curvature: bool) -> None:
    if not torch.cuda.is_available() and device == "cuda":
        return  # Ignore test if CUDA is not available.

    test_results = _run_training(model, device, fixed_curvature, likelihood_n=5000, fixed_epochs=2)
    assert len(test_results) == 1
    for epoch_results in test_results.values():
        log_likelihood = float(epoch_results.log_likelihood)
        assert np.isfinite(log_likelihood).all()
        assert log_likelihood < eps
        assert log_likelihood > -1e4

        for k, v in epoch_results.to_print().items():
            assert np.isfinite(v), f"{k} is not finite."
            if k in {"bce", "kl", "mi", "beta"}:
                assert v > -eps, f"Value of '{k}' should be positive, not '{v}'."
            elif k in {"cov_norm"}:
                pass  # Can be of any sign and value.
            else:
                assert v <= eps, f"Value of '{k}' should be non-positive, not '{v}'."
