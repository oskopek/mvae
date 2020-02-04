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
import os
from typing import Tuple

import bokeh.io
import bokeh.models
import bokeh.palettes
import bokeh.plotting
from bokeh.plotting import ColumnDataSource
import bokeh.resources
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from google import protobuf
from tensorflow.contrib.tensorboard.plugins.projector import projector_config_pb2

from . import utils
from ..mvae.ops.hyperbolics import lorentz_to_poincare


def pca_dim_red(x: np.ndarray, keep_dim: int) -> np.ndarray:
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=keep_dim)
    x = pca.fit_transform(x)
    return x


def to_spherical_coords(inp: np.ndarray) -> np.ndarray:
    x = inp[..., 0]
    y = inp[..., 1]
    z = inp[..., 2]
    # points are assumed normalized (R=1) for visualization
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    return np.stack((theta, phi), axis=-1)


def sphere_proj_2d(inp: np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
    x = inp[..., 0]
    y = inp[..., 1]
    z = inp[..., 2]
    coef = np.sqrt(2 / (1 - z)) / 2
    X = coef * x
    Y = coef * y
    return np.stack((X, Y), axis=-1)


def _plot_latent_space(embeddings: np.ndarray, labels: np.ndarray, type: str) -> bokeh.plotting.figure:
    assert len(embeddings.shape) == 2
    assert len(labels.shape) == 1
    assert embeddings.shape[0] == labels.shape[0]

    if embeddings.shape[1] > 2:
        if type == "h":
            embeddings = lorentz_to_poincare(embeddings)
        elif type == "s":
            embeddings = sphere_proj_2d(embeddings)

    if embeddings.shape[1] > 2:
        embeddings = pca_dim_red(embeddings, keep_dim=2)

    if not embeddings.shape[1] <= 2:
        raise ValueError("Embeddings have to be 2D or lower. If they aren't apply PCA or a projection first.")

    embeddings /= np.linalg.norm(embeddings, axis=-1).max(axis=0)

    x = embeddings[:, 0]
    if embeddings.shape[1] == 1:  # 1D embeddings
        y = np.zeros((embeddings.shape[0],))
    else:
        y = embeddings[:, 1]

    palette = bokeh.palettes.Category20[20]
    colormap = {v: palette[k] for k, v in enumerate(set(labels))}

    tooltips = [("x", "@x"), ("y", "@y"), ("label", "@label")]
    p = bokeh.plotting.figure(tools="pan,crosshair,reset,save,wheel_zoom",
                              tooltips=tooltips,
                              toolbar_location="above",
                              plot_height=utils.BASE_HEIGHT,
                              sizing_mode="stretch_width")

    items = []
    for label, color in colormap.items():
        mask = labels == label
        df = pd.DataFrame({"x": x[mask], "y": y[mask], "label": labels[mask]})
        points = p.scatter(x="x",
                           y="y",
                           fill_color=color,
                           fill_alpha=0.8,
                           line_color=None,
                           size=15,
                           source=ColumnDataSource(df),
                           name=str(int(label)))
        renderers = [points]
        items.append(bokeh.models.LegendItem(label=str(int(label)), renderers=renderers))

    legend = bokeh.models.Legend(items=items, location="center", orientation="vertical", click_policy="hide")
    p.add_layout(legend, "right")

    return p


def read_tsv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter='\t')


def plot_latent_space(embedding_info: projector_config_pb2.EmbeddingInfo,
                      folder: str) -> Tuple[bokeh.plotting.figure, str]:
    name = embedding_info.tensor_name.replace("/", "_").replace(":", "-")
    embeddings = read_tsv(os.path.join(folder, embedding_info.tensor_path))
    labels = read_tsv(os.path.join(folder, embedding_info.metadata_path))
    assert embeddings.shape[0] == labels.shape[0]

    space_type = "e"
    if "total" not in name:
        space_type = name.split("_")[-2][0]

    print(f"Plotting {name} as space type {space_type}...")
    figure = _plot_latent_space(embeddings, labels, type=space_type)

    return figure, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to checkpoint folder.", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise ValueError(f"Input folder doesn't exist: '{args.path}'.")

    projector_meta_path = os.path.join(args.path, "projector_config.pbtxt")
    if not os.path.isfile(projector_meta_path):
        raise ValueError(f"Projector metadata file doesn't exist: '{projector_meta_path}'.")

    with open(projector_meta_path) as f:
        txt = f.read()

    projector_config = protobuf.text_format.Parse(txt, projector_config_pb2.ProjectorConfig())  # type: ignore

    def to_key(embedding_info: projector_config_pb2.EmbeddingInfo) -> Tuple[str, int]:
        name, checkpoint = embedding_info.tensor_name.split(":")
        return name, int(checkpoint)

    embeddings = {to_key(e): e for e in projector_config.embeddings}
    last_embeddings = [v for k, v in embeddings.items() if k[1] >= max(k[1] for k in embeddings.keys())]
    for embedding_info in last_embeddings:
        figure, name = plot_latent_space(embedding_info, args.path)
        utils.export_plots(figure,
                           filename=os.path.join(args.path, name),
                           title=f"Latent space {name}",
                           box=True,
                           x_range_start=-1,
                           x_range_end=1,
                           y_range_start=-1,
                           y_range_end=1)


if __name__ == "__main__":
    main()
