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
from collections import defaultdict

import numpy as np

import torch

from ..data import create_dataset
from ..mvae import utils
from ..mvae.models import FeedForwardVAE, ConvolutionalVAE
from ..utils import str2bool

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="M-VAE runner.")
    parser.add_argument("--data", type=str, default="./data", help="Data directory.")
    parser.add_argument("--batch_size", type=int, default=150, help="Batch size.")
    parser.add_argument("--model_a", type=str, default="e72", help="Model latent space description.")
    parser.add_argument("--chkpt_a",
                        type=str,
                        default="/home/oskopek/git/mt/vae-mnist-e72-2019-08-28T13:21:06.740534/497.chkpt",
                        help="Model latent space description.")
    parser.add_argument("--dataset",
                        type=str,
                        default="mnist",
                        help="Which dataset to run on. Options: 'mnist', 'bdp', 'cifar', 'omniglot'.")
    parser.add_argument("--img_size", type=str, default="(28,28)")
    parser.add_argument("--h_dim_a", type=int, default=400, help="Hidden layer dimension.")
    parser.add_argument(
        "--scalar_parametrization_a",
        type=str2bool,
        default=False,
        help="Use a spheric covariance matrix (single scalar) if true, or elliptic (diagonal covariance matrix) if "
        "false.")
    parser.add_argument("--doubles", type=str2bool, default=True, help="Use float32 or float64. Default float32.")
    args = parser.parse_args()

    args.device = torch.device("cpu")
    utils.setup_gpu(args.device)
    print("Running on:", args.device, flush=True)

    if args.doubles:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    dataset = create_dataset(args.dataset, args.batch_size, args.data)
    print("#####")
    components_a = utils.parse_components(args.model_a, False)
    model_name_a = utils.canonical_name(components_a)

    if args.dataset == "cifar":
        model_cls = ConvolutionalVAE
    elif args.dataset in ["mnist", "omniglot", "bdp"]:
        model_cls = FeedForwardVAE  # type: ignore
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")
    model_a = model_cls(h_dim=args.h_dim_a,
                        components=components_a,
                        dataset=dataset,
                        scalar_parametrization=args.scalar_parametrization_a).to(args.device)
    model_a.load_state_dict(torch.load(args.chkpt_a, map_location="cpu"))
    train_loader, test_loader = dataset.create_loaders()

    model_a.eval()

    n_img = 3
    n_chan = 1
    img_dim = eval(args.img_size)
    classes = defaultdict(lambda: [])

    for xs, ys in test_loader:
        ys = ys.detach().numpy()
        for i, cls in enumerate(ys):
            classes[cls].append(xs[i])
        stacked = [torch.stack(classes[cls][:n_img], dim=0) for cls in sorted(classes.keys())]
        print([s.shape for s in stacked])
        orig = torch.stack(stacked, dim=0)
        orig_shape = orig.shape
        orig = orig.reshape(-1, orig.size(-1))
        _, _, recon_a = model_a(orig)
        recon_a = torch.sigmoid(recon_a).detach()
        break

    orig = orig.reshape(orig_shape[0], orig_shape[1], n_chan, img_dim[0], img_dim[1]) \
        .transpose(2, 3).transpose(3, 4).reshape(orig_shape[0], orig_shape[1] * img_dim[0], img_dim[1], n_chan) \
        .transpose(0, 1).reshape(orig_shape[1] * img_dim[0], -1, n_chan)
    recon_a = recon_a.reshape(orig_shape[0], orig_shape[1], n_chan, img_dim[0], img_dim[1]) \
        .transpose(2, 3).transpose(3, 4).reshape(orig_shape[0], orig_shape[1] * img_dim[0], img_dim[1], n_chan) \
        .transpose(0, 1).reshape(orig_shape[1] * img_dim[0], -1, n_chan)

    print(orig.shape)

    orig = orig.squeeze().detach().numpy()
    recon_a = recon_a.squeeze().detach().numpy()

    def save(img: np.ndarray, name: str) -> None:
        imgg = Image.fromarray((img * 255).astype('uint8'))
        imgg = imgg.resize(size=(imgg.size[0] * 10, imgg.size[1] * 10), resample=Image.NEAREST)
        imgg = imgg.convert(mode="RGB")
        imgg.save(f"samples-{args.dataset}-{name}.png")

    save(orig, "orig")
    save(recon_a, model_name_a)

    # for idx, component_i in enumerate([1, len(model_a.components) // 2, len(model_a.components) - 2]):
    for idx, component_i in enumerate([0, len(model_a.components) // 2, len(model_a.components) - 1]):
        linspace = np.linspace(-2, 2, num=8, dtype=np.float64)
        ds = np.zeros((len(linspace)**2, model_a.total_z_dim)).astype(np.float64)
        for i in range(len(linspace)):
            for j in range(len(linspace)):
                ds[i * len(linspace) + j, 2 * component_i] = linspace[i]
                ds[i * len(linspace) + j, 2 * component_i + 1] = linspace[j]

        ds = torch.tensor(ds, dtype=torch.float64)
        print(ds.shape)
        recon_a = model_a.decode(ds)
        recon_a = torch.sigmoid(recon_a).detach()

        recon_a = recon_a.reshape(len(linspace), len(linspace), n_chan, img_dim[0],
                                  img_dim[1]).transpose(2, 3).transpose(3, 4).transpose(1, 2).reshape(
                                      len(linspace) * img_dim[0],
                                      len(linspace) * img_dim[1], n_chan)

        recon_a = recon_a.squeeze().detach().numpy()
        name = model_name_a + "-interpolation-comp_" + str(
            component_i) + "-" + model_a.components[component_i]._shortcut()
        if hasattr(model_a.components[component_i], "_curvature"):
            name += "-curv" + str(model_a.components[component_i]._curvature.item())
        save(recon_a, name)


if __name__ == "__main__":
    # with torch.autograd.set_detect_anomaly(True):
    main()
