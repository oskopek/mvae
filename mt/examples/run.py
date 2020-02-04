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

import argparse
import datetime
import os

import torch

from ..data import create_dataset
from ..mvae import utils
from ..mvae.models import Trainer, FeedForwardVAE, ConvolutionalVAE
from ..utils import str2bool


def main() -> None:
    parser = argparse.ArgumentParser(description="M-VAE runner.")
    parser.add_argument("--device", type=str, default="cuda", help="Whether to use cuda or cpu.")
    parser.add_argument("--data", type=str, default="./data", help="Data directory.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lookahead", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--model", type=str, default="h2,s2,e2", help="Model latent space description.")
    parser.add_argument("--universal", type=str2bool, default=False, help="Universal training scheme.")
    parser.add_argument("--dataset",
                        type=str,
                        default="mnist",
                        help="Which dataset to run on. Options: 'mnist', 'bdp', 'cifar', 'omniglot'.")
    parser.add_argument("--h_dim", type=int, default=400, help="Hidden layer dimension.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--show_embeddings", type=int, default=None, help="Show embeddings every N epochs.")
    parser.add_argument("--train_statistics",
                        type=str2bool,
                        default=False,
                        help="Show Tensorboard statistics for training.")
    parser.add_argument(
        "--scalar_parametrization",
        type=str2bool,
        default=False,
        help="Use a spheric covariance matrix (single scalar) if true, or elliptic (diagonal covariance matrix) if "
        "false.")
    parser.add_argument("--fixed_curvature",
                        type=str2bool,
                        default=True,
                        help="Whether to fix curvatures to (-1, 0, 1).")
    parser.add_argument("--doubles", type=str2bool, default=True, help="Use float32 or float64. Default float32.")
    parser.add_argument("--beta_start", type=float, default=1.0, help="Beta-VAE beginning value.")
    parser.add_argument("--beta_end", type=float, default=1.0, help="Beta-VAE end value.")
    parser.add_argument("--beta_end_epoch", type=int, default=1, help="Beta-VAE end epoch (0 to epochs-1).")
    parser.add_argument("--likelihood_n",
                        type=int,
                        default=500,
                        help="How many samples to use for LL estimation. Value 0 disables LL estimation.")
    args = parser.parse_args()

    if args.seed:
        print("Using pre-set random seed:", args.seed)
        utils.set_seeds(args.seed)

    if not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA is not available.")
    args.device = torch.device(args.device)
    utils.setup_gpu(args.device)
    print("Running on:", args.device, flush=True)

    if args.doubles:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    dataset = create_dataset(args.dataset, args.batch_size, args.data)
    print("#####")
    cur_time = datetime.datetime.utcnow().isoformat()
    components = utils.parse_components(args.model, args.fixed_curvature)
    model_name = utils.canonical_name(components)
    print(f"VAE Model: {model_name}; Epochs: {args.epochs}; Time: {cur_time}; Fixed curvature: {args.fixed_curvature}; "
          f"Dataset: {args.dataset}")
    print("#####", flush=True)
    chkpt_dir = f"./chkpt/vae-{args.dataset}-{model_name}-{cur_time}"
    os.makedirs(chkpt_dir)

    if args.dataset == "cifar":
        model_cls = ConvolutionalVAE
    elif args.dataset in ["mnist", "omniglot", "bdp"]:
        model_cls = FeedForwardVAE  # type: ignore
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")
    model = model_cls(h_dim=args.h_dim,
                      components=components,
                      dataset=dataset,
                      scalar_parametrization=args.scalar_parametrization).to(args.device)

    trainer = Trainer(model,
                      img_dims=dataset.img_dims,
                      chkpt_dir=chkpt_dir,
                      train_statistics=args.train_statistics,
                      show_embeddings=args.show_embeddings)
    optimizer = trainer.build_optimizer(learning_rate=args.learning_rate, fixed_curvature=args.fixed_curvature)
    train_loader, test_loader = dataset.create_loaders()
    betas = utils.linear_betas(args.beta_start, args.beta_end, end_epoch=args.beta_end_epoch, epochs=args.epochs)

    if args.universal:
        # Pre-training:
        trainer.train_epochs(optimizer=optimizer,
                             train_data=train_loader,
                             eval_data=test_loader,
                             epochs=args.epochs // 2,
                             betas=betas,
                             likelihood_n=0)

        # Choose signs:
        eps = 1e-5
        cn = len(model.components) // 3
        signs = [-1] * cn + [1] * cn + [0] * (len(model.components) - 2 * cn)
        print("Chosen signs:", signs)
        for i, component in enumerate(model.components):
            component._curvature.data += signs[i] * eps
            component._curvature.requires_grad = False

        # ... and continue without learning curvature for an epoch:
        trainer.train_epochs(optimizer=optimizer,
                             train_data=train_loader,
                             eval_data=test_loader,
                             epochs=10,
                             betas=betas,
                             likelihood_n=0)

        # ... then unfix it:
        for component in model.components:
            component._curvature.requires_grad = True
        trainer.train_stopping(optimizer=optimizer,
                               train_data=train_loader,
                               eval_data=test_loader,
                               warmup=args.lookahead + 1,
                               lookahead=args.lookahead,
                               betas=betas,
                               likelihood_n=args.likelihood_n,
                               max_epochs=args.epochs)
    else:
        trainer.train_stopping(optimizer=optimizer,
                               train_data=train_loader,
                               eval_data=test_loader,
                               warmup=args.warmup,
                               lookahead=args.lookahead,
                               betas=betas,
                               likelihood_n=args.likelihood_n,
                               max_epochs=args.epochs)
    print(flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    # with torch.autograd.set_detect_anomaly(True):
    main()
