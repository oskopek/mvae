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

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import matplotlib.pyplot
import torch
from torch import Tensor
from tensorboardX import SummaryWriter
import torchvision

EpochStatsType = Dict[str, float]


class Stats:

    def __init__(self,
                 chkpt_dir: str,
                 img_dims: Optional[Tuple[int, ...]] = None,
                 global_step: int = 0,
                 epoch: int = 0,
                 train_statistics: bool = False,
                 show_embeddings: int = 0,
                 export_embeddings: int = 0,
                 test_every: int = 0):
        self._summary_writer = SummaryWriter(chkpt_dir)
        self._global_step = global_step
        self._epoch = epoch
        self._img_dims = img_dims
        self.train_statistics = train_statistics
        self.show_embeddings = show_embeddings
        self.export_embeddings = export_embeddings
        self.test_every = test_every
        self.test_epochs = 0

    @property
    def img_dims(self) -> Optional[Tuple[int, ...]]:
        return self._img_dims

    @property
    def global_step(self) -> int:
        return self._global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        self._global_step = value

    def _step(self, epoch: bool = False) -> int:
        if epoch:
            return self.epoch
        else:
            return self.global_step

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = value

    def add_scalar(self, tag: str, scalar_value: Any, epoch: bool = False) -> None:
        self._summary_writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self._step(epoch))

    @staticmethod
    def _show_images(x: Tensor, dims: Tuple[int, ...]) -> Tensor:
        return torchvision.utils.make_grid(x.view(*dims))

    def add_images(self, tag: str, imgs: Tensor, dims: Optional[Tuple[int, ...]] = None, epoch: bool = False) -> None:
        if dims is None:
            dims = self._img_dims

        if dims is None:
            warnings.warn("No img dims specified, will not show reconstructions in TensorBoard.")
            return

        img_tensor = Stats._show_images(imgs, dims)
        self._summary_writer.add_image(tag=tag, img_tensor=img_tensor, global_step=self._step(epoch))

    def add_histogram(self, tag: str, values: Tensor, epoch: bool = False) -> None:
        self._summary_writer.add_histogram(tag=tag, values=values, global_step=self._step(epoch))

    def add_embedding(self, tag: str, mat: Tensor, metadata: List[str], epoch: bool = False) -> None:
        self._summary_writer.add_embedding(tag=tag, mat=mat, metadata=metadata, global_step=self._step(epoch))

    def add_figure(self, tag: str, figure: matplotlib.pyplot.figure, epoch: bool = False) -> None:
        self._summary_writer.add_figure(tag=tag, figure=figure, global_step=self._step(epoch))


def _to_print(stats: Union["BatchStatsFloat", "EpochStats"]) -> EpochStatsType:
    return {
        "bce": stats.bce,
        "kl": stats.kl,
        "elbo": stats.elbo,
        "ll": 0.0 if stats.log_likelihood is None else stats.log_likelihood,
        "mi": 0.0 if stats.mutual_info is None else stats.mutual_info,
        "cov_norm": 0.0 if stats.cov_norm is None else stats.cov_norm,
        "beta": stats.beta
    }


class BatchStatsFloat:

    def __init__(self, bce: Tensor, kl: Tensor, elbo: Tensor, log_likelihood: Optional[Tensor],
                 mutual_info: Optional[Tensor], cov_norm: Optional[Tensor], component_kl: List[Tensor],
                 beta: float) -> None:
        self.bce = bce.item()
        self.kl = kl.item()
        self.elbo = elbo.item()
        self.log_likelihood = None if log_likelihood is None else log_likelihood.item()
        self.mutual_info = None if mutual_info is None else mutual_info.item()
        self.cov_norm = None if cov_norm is None else cov_norm.item()
        self.component_kl = [x.item() for x in component_kl]
        self.beta = beta

    def summaries(self, stats: Stats, prefix: str = "train/batch") -> None:
        stats.add_scalar(prefix + "/bce", self.bce)
        stats.add_scalar(prefix + "/kl", self.kl)
        stats.add_scalar(prefix + "/elbo", self.elbo)
        if self.log_likelihood is not None:
            stats.add_scalar(prefix + "/log_likelihood", self.log_likelihood)
        if self.mutual_info is not None:
            stats.add_scalar(prefix + "/mutual_info", self.mutual_info)
        if self.cov_norm is not None:
            stats.add_scalar(prefix + "/cov_norm", self.cov_norm)

    def to_print(self) -> EpochStatsType:
        return _to_print(self)


class BatchStats:

    def __init__(self,
                 bce: Tensor,
                 component_kl: List[Tensor],
                 beta: float,
                 log_likelihood: Optional[Tensor] = None,
                 mutual_info: Optional[Tensor] = None,
                 cov_norm: Optional[Tensor] = None) -> None:
        self._beta = beta

        self._bce = bce
        self._component_kl = component_kl
        self._component_kl_mean = [x.sum(dim=0) for x in component_kl]
        self._log_likelihood = log_likelihood
        self._mutual_info = mutual_info
        self._cov_norm = cov_norm

        self._kl_val = self._kl()
        self._elbo_val = self._elbo(beta)

    @property
    def bce(self) -> Tensor:
        return self._bce.sum(dim=0)

    @property
    def component_kl(self) -> List[Tensor]:
        return self._component_kl_mean

    @property
    def log_likelihood(self) -> Optional[Tensor]:
        return None if self._log_likelihood is None else self._log_likelihood.sum(dim=0)

    @property
    def mutual_info(self) -> Optional[Tensor]:
        return None if self._mutual_info is None else self._mutual_info.sum(dim=0)

    @property
    def cov_norm(self) -> Optional[Tensor]:
        return None if self._cov_norm is None else self._cov_norm.sum(dim=0)

    @property
    def kl(self) -> Tensor:
        return self._kl_val.sum(dim=-1)

    @property
    def elbo(self) -> Tensor:
        return self._elbo_val.sum(dim=-1)

    @property
    def beta(self) -> float:
        return self._beta

    def _kl(self) -> Tensor:
        return torch.sum(torch.cat([x.unsqueeze(dim=-1) for x in self._component_kl], dim=-1), dim=-1)

    def _elbo(self, beta: float) -> Tensor:
        assert self._bce.shape == self._kl_val.shape
        return (-self._bce - beta * self._kl_val).sum(dim=0)

    def convert_to_float(self) -> BatchStatsFloat:
        return BatchStatsFloat(self.bce,
                               self.kl,
                               self.elbo,
                               self.log_likelihood,
                               self.mutual_info,
                               self.cov_norm,
                               self.component_kl,
                               beta=self.beta)


class EpochStats:

    def __init__(self, bs: List[BatchStatsFloat], length: int) -> None:
        assert len(bs) > 0

        self.bce = 0.
        self.kl = 0.
        self.elbo = 0.
        self.log_likelihood = 0.
        self.mutual_info = 0.
        self.cov_norm = 0.
        self.component_kl = [0. for _ in bs[0].component_kl]

        self.beta = bs[0].beta
        assert sum(self.beta == b.beta for b in bs) == len(bs)  # Assert all betas in epoch are the same.

        for batch in bs:
            self.bce += batch.bce
            self.kl += batch.kl
            self.elbo += batch.elbo
            if batch.log_likelihood:
                self.log_likelihood += batch.log_likelihood
            if batch.mutual_info:
                self.mutual_info += batch.mutual_info
            if batch.cov_norm:
                self.cov_norm += batch.cov_norm
            for i in range(len(self.component_kl)):
                self.component_kl[i] += batch.component_kl[i]

        self.bce /= length
        self.kl /= length
        self.elbo /= length
        self.log_likelihood /= length
        self.mutual_info /= length
        self.cov_norm /= length
        for i in range(len(self.component_kl)):
            self.component_kl[i] /= length

    def to_print(self) -> EpochStatsType:
        return _to_print(self)

    def summaries(self, stats: Stats, prefix: str = "train/epoch") -> None:
        stats.add_scalar(prefix + "/beta", self.beta, epoch=True)
        stats.add_scalar(prefix + "/bce", self.bce, epoch=True)
        stats.add_scalar(prefix + "/kl", self.kl, epoch=True)
        stats.add_scalar(prefix + "/elbo", self.elbo, epoch=True)
        for i in range(len(self.component_kl)):
            stats.add_scalar(prefix + f"/kl_comp_{i}", self.component_kl[i], epoch=True)
        if self.log_likelihood:
            stats.add_scalar(prefix + "/log_likelihood", self.log_likelihood, epoch=True)
        if self.mutual_info:
            stats.add_scalar(prefix + "/mutual_info", self.mutual_info, epoch=True)
        if self.cov_norm:
            stats.add_scalar(prefix + "/cov_norm", self.cov_norm, epoch=True)
