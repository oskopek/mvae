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
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from .vae import ModelVAE
from ..components import StereographicallyProjectedSphereComponent, PoincareComponent
from ..components import SphericalComponent, HyperbolicComponent
from ..ops import hyperbolics as H
from ..stats import Stats, EpochStats
from ...visualization.utils import plot_poincare_embeddings
from ..utils import CurvatureOptimizer


class Trainer:

    def __init__(self,
                 model: ModelVAE,
                 img_dims: Optional[Tuple[int, ...]],
                 chkpt_dir: str = "./chkpt",
                 train_statistics: bool = False,
                 show_embeddings: int = 0,
                 export_embeddings: int = 0,
                 test_every: int = 0) -> None:
        self.model = model
        self.chkpt_dir = chkpt_dir
        self.stats = Stats(chkpt_dir=chkpt_dir,
                           img_dims=img_dims,
                           show_embeddings=show_embeddings,
                           export_embeddings=export_embeddings,
                           train_statistics=train_statistics,
                           test_every=test_every)

    @property
    def epoch(self) -> int:
        return self.stats.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.stats.epoch = value

    def _load_epoch(self, epoch: int) -> None:
        self.model.load_state_dict(torch.load(os.path.join(self.chkpt_dir, f"{epoch}.chkpt")))
        self.model.to(self.model.device)

    def _save_epoch(self, epoch: int) -> None:
        torch.save(self.model.state_dict(), os.path.join(self.chkpt_dir, f"{epoch}.chkpt"))

    def _delete_epoch(self, epoch: int) -> None:
        path = os.path.join(self.chkpt_dir, f"{epoch}.chkpt")
        if os.path.isfile(path):
            os.remove(path)

    def _update_checkpoints(self, lookahead: int) -> None:
        delete_epoch = self.epoch - lookahead - 1
        if delete_epoch >= 0:
            self._delete_epoch(delete_epoch)
        self._save_epoch(self.epoch)

    @staticmethod
    def _should_stop(results: Dict[int, EpochStats], epoch: int, lookahead: int, max_epoch: int) -> Optional[int]:

        def _get_elbo(epoch: int) -> float:
            return float(results[epoch].elbo)

        cur_stop_step = epoch - lookahead
        lookahead_interval = range(cur_stop_step + 1, epoch + 1)  # (cur_stop_step, epoch]
        elbos = np.asarray([_get_elbo(i) for i in lookahead_interval])
        max_elbo_i = np.argmax(elbos)
        max_elbo = elbos[max_elbo_i]
        if max_elbo < _get_elbo(cur_stop_step):
            return cur_stop_step
        elif epoch == max_epoch:
            return cur_stop_step + 1 + max_elbo_i
        else:
            return None

    def get_beta(self, betas: Optional[Sequence[float]]) -> float:
        if betas is None:
            return 1.0
        elif self.epoch >= len(betas):
            return betas[-1]
        else:
            return betas[self.epoch]

    def train_stopping(self,
                       optimizer: Any,
                       train_data: DataLoader,
                       eval_data: DataLoader,
                       betas: Sequence[float],
                       warmup: int = 5,
                       lookahead: int = 2,
                       likelihood_n: int = 500,
                       max_epochs: int = 1000) -> Dict[int, EpochStats]:
        assert warmup >= lookahead

        train_results = dict()
        test_results = dict()

        # Warmup
        for _ in range(warmup):
            beta = self.get_beta(betas)
            train_results[self.epoch] = self._train_epoch(optimizer, train_data, beta=beta)
            self._update_checkpoints(lookahead)
            self.epoch += 1
            self._try_test_during_train(test_results, eval_data, likelihood_n, betas)

        # Early stopping active
        stop_epoch = None
        for _ in range(warmup, max_epochs):
            beta = self.get_beta(betas)
            train_results[self.epoch] = self._train_epoch(optimizer, train_data, beta=beta)

            stop_epoch = Trainer._should_stop(train_results, self.epoch, lookahead, max_epoch=max_epochs - 1)
            self._update_checkpoints(lookahead)
            if stop_epoch:
                break
            self.epoch += 1
            self._try_test_during_train(test_results, eval_data, likelihood_n, betas)

        if not stop_epoch:
            warnings.warn("Did not stop using early stopping.")
        self._load_epoch(stop_epoch)
        last_epoch = self.epoch
        self.epoch = stop_epoch
        print(f"Stopped at epoch: {stop_epoch}. Deleting epochs [{stop_epoch + 1}, {last_epoch}] and "
              f"[{last_epoch - lookahead},{stop_epoch - 1}].")
        for epoch in range(stop_epoch + 1, last_epoch + 1):
            self._delete_epoch(epoch)
        for epoch in range(last_epoch - lookahead, stop_epoch):
            self._delete_epoch(epoch)
        with torch.set_grad_enabled(False):
            test_results[stop_epoch] = self._test_epoch(eval_data, likelihood_n=likelihood_n, beta=self.get_beta(betas))

        return test_results

    def _try_test_during_train(self, test_results: Mapping[int, EpochStats], eval_data: DataLoader, likelihood_n: int,
                               betas: Optional[Sequence[float]]) -> None:
        if self.stats.test_every > 0 and self.epoch % self.stats.test_every == 0:
            test_results[self.epoch - 1] = self._test_epoch(eval_data,
                                                            likelihood_n=likelihood_n,
                                                            beta=self.get_beta(betas))

    def train_epochs(self,
                     optimizer: Any,
                     train_data: DataLoader,
                     eval_data: DataLoader,
                     betas: Optional[Sequence[float]],
                     epochs: int = 300,
                     likelihood_n: int = 500) -> Dict[int, EpochStats]:
        test_results = dict()
        for _ in range(epochs):
            beta = self.get_beta(betas)
            self._train_epoch(optimizer, train_data, beta=beta)
            self.epoch += 1
            self._try_test_during_train(test_results, eval_data, likelihood_n, betas)

        with torch.set_grad_enabled(False):
            test_results[self.epoch - 1] = self._test_epoch(eval_data,
                                                            likelihood_n=likelihood_n,
                                                            beta=self.get_beta(betas))

        self._save_epoch(self.epoch)
        return test_results

    def _train_epoch(self, optimizer: torch.optim.Optimizer, train_data: DataLoader, beta: float) -> EpochStats:
        print(f"\tTrainEpoch {self.epoch}:\t", end="")
        self.model.train()

        if self.epoch < 10:
            for c in self.model.components:
                if isinstance(c, StereographicallyProjectedSphereComponent) or isinstance(c, SphericalComponent):
                    c._pradius.data = torch.ones_like(c._pradius.data) * (11 - self.epoch)
                elif isinstance(c, PoincareComponent) or isinstance(c, HyperbolicComponent):
                    c._nradius.data = torch.ones_like(c._nradius.data) * (11 - self.epoch)

        batch_stats = []
        for x_mb, y_mb in train_data:
            stats, (reparametrized, _, _) = self.model.train_step(optimizer, x_mb, beta=beta)

            if self.stats.train_statistics:
                stats.summaries(self.stats, prefix="train/batch")
                for i, (component, r) in enumerate(zip(self.model.components, reparametrized)):
                    self.stats.add_scalar(f"train/batch/{component.summary_name(i)}/curvature",
                                          component.manifold.curvature)
                    for key, val in component.summaries(i, r.q_z, prefix="train/batch").items():
                        self.stats.add_histogram(tag=key, values=val)

            self.stats.global_step += 1
            batch_stats.append(stats)

        epoch_stats = EpochStats(batch_stats, length=len(train_data.dataset))
        epoch_stats.summaries(self.stats, prefix="train/epoch")
        epoch_dict = epoch_stats.to_print()
        for i, component in enumerate(self.model.components):
            name = f"{component.summary_name(i)}/curvature"
            epoch_dict[name] = float(component.manifold.curvature)
            self.stats.add_scalar(f"train/epoch/{name}", component.manifold.curvature, epoch=True)
        print(epoch_dict, flush=True)

        return epoch_stats

    def _test_epoch(self, test_data: DataLoader, likelihood_n: int, beta: float) -> EpochStats:
        print(f"\tEpoch {self.epoch}:\t", end="")
        self.model.eval()

        show_embeddings = self.stats.show_embeddings > 0 and self.stats.test_epochs % self.stats.show_embeddings == 0
        if show_embeddings:
            embeddings: List[List[torch.Tensor]] = [[] for _ in self.model.components]
            total_embeddings = []
            labels = []

        image_summary = True
        batch_stats = []
        histograms: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for batch_idx, (x_mb, y_mb) in enumerate(test_data):
            x_mb = x_mb.to(self.model.device)
            reparametrized, concat_z, x_mb_ = self.model(x_mb)
            stats = self.model.compute_batch_stats(x_mb, x_mb_, reparametrized, likelihood_n=likelihood_n, beta=beta)
            batch_stats.append(stats.convert_to_float())

            for i, (component, r) in enumerate(zip(self.model.components, reparametrized)):
                for key, val in component.summaries(i, r.q_z, prefix="eval/batch").items():
                    histograms[key].append(val)
                if show_embeddings and batch_idx % 10 == 0:
                    embeddings[i].append(r.q_z.loc)

            if show_embeddings and batch_idx % 10 == 0:
                total_embeddings.append(concat_z)
                labels.append(y_mb)

            if image_summary:
                self.stats.add_images("eval/x_input", x_mb, epoch=True)
                self.stats.add_images("eval/x_recon", torch.sigmoid(x_mb_), epoch=True)
                image_summary = False

        for key, val_lst in histograms.items():
            self.stats.add_histogram(tag=key, values=torch.cat(val_lst, dim=0), epoch=True)

        if show_embeddings:
            total_embeddings_cat = torch.cat(total_embeddings, dim=0)
            labels_cat = torch.flatten(torch.cat(labels, dim=0)).tolist()
            self.stats.add_embedding(tag="eval/total_embeddings",
                                     mat=total_embeddings_cat,
                                     metadata=labels_cat,
                                     epoch=True)
            for i, component in enumerate(self.model.components):
                tag = f"eval/{component.summary_name(i)}/embeddings"
                embeddings_cat = torch.cat(embeddings[i], dim=0)
                if component.manifold.curvature < 0:
                    if isinstance(component, HyperbolicComponent):
                        embeddings_poincare = H.lorentz_to_poincare(embeddings_cat, radius=component.manifold.radius)
                    else:
                        embeddings_poincare = embeddings_cat
                    fig = plot_poincare_embeddings(embeddings_poincare, labels_cat)
                    self.stats.add_figure(tag=tag + "_proj", figure=fig, epoch=True)
                self.stats.add_embedding(tag=tag, mat=embeddings_cat, metadata=labels_cat, epoch=True)
            self._save_epoch(self.epoch)  # Save an epoch that has embeddings, because the TF projector needs it.

        epoch_stats = EpochStats(batch_stats, length=len(test_data.dataset))
        epoch_stats.summaries(self.stats, prefix="eval/epoch")
        epoch_dict = epoch_stats.to_print()
        for i, component in enumerate(self.model.components):
            name = f"{component.summary_name(i)}/curvature"
            epoch_dict[name] = float(component.manifold.curvature)
            self.stats.add_scalar(f"eval/epoch/{name}", component.manifold.curvature, epoch=True)
        print(epoch_dict, flush=True)

        export_embeddings = self.stats.export_embeddings > 0 \
            and self.stats.test_epochs % self.stats.export_embeddings == 0
        if export_embeddings:
            self._export_representations(test_data)  # TODO: merge this method with _export_representations.

        self.model.train()
        self.stats.test_epochs += 1
        return epoch_stats

    def _export_representations(self, data: DataLoader, mode: str = "eval") -> None:
        print(f"\tExporting {mode} representations...")
        self.model.eval()

        repr_folder = os.path.join(self.chkpt_dir, "repr")
        os.makedirs(repr_folder, exist_ok=True)

        def _filename(component: str) -> str:
            return os.path.join(repr_folder, f"{mode}_{component}_{self.epoch}.pt")

        embeddings: List[List[torch.Tensor]] = [[] for _ in self.model.components]
        total_embeddings = []
        labels = []
        for batch_idx, (x_mb, y_mb) in enumerate(data):
            x_mb = x_mb.to(self.model.device)
            reparametrized, concat_z, x_mb_ = self.model(x_mb)

            for i, (component, r) in enumerate(zip(self.model.components, reparametrized)):
                embeddings[i].append(r.q_z.loc.to("cpu"))
            total_embeddings.append(concat_z.to("cpu"))
            labels.append(y_mb.to("cpu"))

        total_embeddings_cat = torch.cat(total_embeddings, dim=0)
        torch.save(total_embeddings_cat, _filename("total"))
        labels_cat = torch.cat(labels, dim=0)
        torch.save(labels_cat, _filename("labels"))
        for i, component in enumerate(self.model.components):
            embeddings_cat = torch.cat(embeddings[i], dim=0)
            torch.save(embeddings_cat, _filename(component.summary_name(i)))

    def build_optimizer(self, learning_rate: float, fixed_curvature: bool) -> torch.optim.Optimizer:

        def ncurvature_param_cond(key: str) -> bool:
            return "nradius" in key or "curvature" in key

        def pcurvature_param_cond(key: str) -> bool:
            return "pradius" in key

        net_params = [
            v for key, v in self.model.named_parameters()
            if not ncurvature_param_cond(key) and not pcurvature_param_cond(key)
        ]
        neg_curv_params = [v for key, v in self.model.named_parameters() if ncurvature_param_cond(key)]
        pos_curv_params = [v for key, v in self.model.named_parameters() if pcurvature_param_cond(key)]
        curv_params = neg_curv_params + pos_curv_params

        net_optimizer = torch.optim.Adam(net_params, lr=learning_rate)
        if not fixed_curvature and not curv_params:
            warnings.warn("Fixed curvature disabled, but found no curvature parameters. Did you mean to set "
                          "fixed=True, or not?")
        if not pos_curv_params:
            c_opt_pos = None
        else:
            c_opt_pos = torch.optim.SGD(pos_curv_params, lr=1e-4)

        if not neg_curv_params:
            c_opt_neg = None
        else:
            c_opt_neg = torch.optim.SGD(neg_curv_params, lr=1e-4)

        def condition() -> bool:
            return (not fixed_curvature) and (self.epoch >= 10) and (self.stats.global_step % 1 == 0)

        return CurvatureOptimizer(net_optimizer, neg=c_opt_neg, pos=c_opt_pos, should_do_curvature_step=condition)
