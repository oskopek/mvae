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

from typing import Dict, List, Optional, Tuple

import bokeh
import bokeh.io
import bokeh.plotting
import bokeh.resources
import matplotlib.pyplot as plt
import torch

from ..mvae.utils import parse_component_str

BASE_HEIGHT = 800
HEIGHT = 1500
WIDTH = 2 * HEIGHT


def texify_components(signature: str, to_latex: Optional[Dict[str, str]] = None) -> str:
    cmap = {"s": 1, "h": -1, "p": -1, "d": 1}
    if to_latex is None:
        to_latex = {"s": "\\sph", "h": "\\hyp", "e": "\\euc", "p": "\\poi", "u": "\\uni", "d": "\\spr"}

    def _texify_component(component: Tuple[int, str, int]) -> str:
        assert to_latex is not None
        multiplier, space_type, dim = component
        assert multiplier >= 1
        texified = to_latex[space_type] + "^{" + str(dim) + "}"
        return texified if multiplier == 1 else f"({texified})" + "^{" + str(multiplier) + "}"

    components = (parse_component_str(component_str.strip()) for component_str in signature.split(","))
    texified = map(_texify_component, components)
    res = " \\times ".join(texified)
    if signature.endswith("fixed"):
        for k, val in to_latex.items():
            if k in cmap:
                c = cmap[k]
                cstr = "{" + str(int(c)) + "}"
                res = res.replace(val, f"{val}_{cstr}")
    return res


def plot_poincare_embeddings(embeddings_poincare: torch.Tensor, labels: List[int]) -> plt.figure:
    if embeddings_poincare.shape[-1] > 2:
        fig = plt.figure(0)
        plt.text(0, 0, "Unavailable for dim > 2", size="large")
        return fig

    assert len(embeddings_poincare.shape) == 2
    data: Dict[int, Tuple[List[float], List[float]]] = {l: ([], []) for l in labels}
    for label, row in zip(labels, embeddings_poincare.tolist()):
        data[label][0].append(float(row[0]))
        data[label][1].append(float(row[1]))

    fig, ax = plt.subplots()
    colors = ["black", "red", "yellow", "blue", "magenta", "cyan", "green", "orange", "darkred", "lime"]
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    for i, color in zip(labels, colors):
        ax.scatter(data[i][0], data[i][1], c=color, label=str(i), edgecolors="none")

    ax.legend()
    ax.grid(True)

    return fig


def set_font_size(p: bokeh.plotting.figure, font_size: int = 30, small_font_size: int = 20) -> None:
    font_size_s = f"{font_size}pt"
    small_font_size_s = f"{small_font_size}pt"
    if p.title is not None:
        p.title.text_font_size = font_size_s
    p.xaxis.axis_label_text_font_size = font_size_s
    p.xaxis.major_label_text_font_size = small_font_size_s
    p.yaxis.axis_label_text_font_size = font_size_s
    p.yaxis.major_label_text_font_size = small_font_size_s
    if p.legend:
        p.legend.label_text_font_size = small_font_size_s
        p.legend.glyph_height = font_size
        p.legend.glyph_width = font_size


def export_plots(p: bokeh.plotting.figure,
                 filename: str,
                 title: str,
                 width: int = WIDTH,
                 height: int = HEIGHT,
                 box: bool = False,
                 show_title: bool = False,
                 y_range_start: Optional[float] = None,
                 y_range_end: Optional[float] = None,
                 x_range_start: Optional[float] = None,
                 x_range_end: Optional[float] = None) -> None:
    # HTML
    if not show_title:
        p.title = None
    bokeh.plotting.save(p, title=title, filename=filename + ".html", resources=bokeh.resources.CDN)

    # PNG
    if y_range_start:
        p.y_range.start = y_range_start
    if y_range_end:
        p.y_range.end = y_range_end
    if x_range_start:
        p.x_range.start = x_range_start
    if x_range_end:
        p.x_range.end = x_range_end

    set_font_size(p)
    p.sizing_mode = "fixed"
    p.width = width
    p.height = height
    if box:
        p.width = height
    p.toolbar_location = None
    bokeh.io.export_png(p, filename=filename + ".png", height=HEIGHT, width=WIDTH)

    # SVG:
    # p.output_backend = "svg"
    # bokeh.io.export_svgs(p, filename=filename + ".svg")
    #
    # os.system(f"inkscape --without-gui --export-pdf={filename}.pdf {filename}.svg")
