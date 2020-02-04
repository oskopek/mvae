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
import glob
import itertools
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

import bokeh.io
import bokeh.models
import bokeh.palettes
import bokeh.plotting
import bokeh.resources
import numpy as np
import pandas as pd

from . import read_log, utils

BASE_HEIGHT = 800
HEIGHT = 1500
WIDTH = 2 * HEIGHT

stat_title = {
    "ll": "Log Likelihood",
    "elbo": "Evidence Lower BOund",
    "kl": "Kullback-Leibler Divergence",
    "bce": "Binary Cross-Entropy",
    "mi": "Mutual Information",
    "cov_norm": "Frobenius norm of the cross-covariance matrix"
}

axis_title = {
    "ll": "ln(likelihood)",
    "elbo": "ELBO",
    "kl": "KL",
    "bce": "BCE",
    "mi": "MI",
    "cov_norm": "Cross-Covariance norm"
}


def set_font_size(p: bokeh.plotting.figure, font_size: int = 40, small_font_size: int = 35) -> None:
    font_size_s = f"{font_size}pt"
    small_font_size_s = f"{small_font_size}pt"
    if p.title is not None:
        p.title.text_font_size = font_size_s
        p.title.text_color = "black"

    p.xaxis.axis_line_width = 5
    p.xaxis.major_tick_line_width = 5
    p.xaxis.minor_tick_line_width = 5
    p.yaxis.axis_line_width = 5
    p.yaxis.major_tick_line_width = 5
    p.yaxis.minor_tick_line_width = 5

    p.xaxis.axis_label_text_font_size = font_size_s
    p.xaxis.major_label_text_color = "black"
    p.xaxis.axis_label_text_color = "black"
    p.xaxis.major_label_text_font_size = small_font_size_s
    p.yaxis.axis_label_text_font_size = font_size_s
    p.yaxis.major_label_text_font_size = small_font_size_s
    p.yaxis.major_label_text_color = "black"
    p.yaxis.axis_label_text_color = "black"
    if p.legend:
        p.legend.label_text_font_size = small_font_size_s
        p.legend.label_text_color = "black"
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
                 y_range_end: Optional[float] = None) -> None:
    # HTML
    if not show_title:
        p.title = None
    bokeh.plotting.save(p, title=title, filename=filename + ".html", resources=bokeh.resources.CDN)

    # PNG
    if y_range_start:
        p.y_range.start = y_range_start
    if y_range_end:
        p.y_range.end = y_range_end

    set_font_size(p)
    p.sizing_mode = "fixed"
    p.width = width
    if box:
        p.height = width
    else:
        p.height = height
    p.toolbar_location = None
    bokeh.io.export_png(p, filename=filename + ".png", height=HEIGHT, width=WIDTH)

    # SVG:
    # p.output_backend = "svg"
    # bokeh.io.export_svgs(p, filename=filename + ".svg")
    #
    # os.system(f"inkscape --without-gui --export-pdf={filename}.pdf {filename}.svg")


def box_whiskers_plot(df: pd.DataFrame, out_folder: str, statistic: str = "ll", subtitle: str = "") -> None:
    title = f"{stat_title[statistic]} ({subtitle})"

    # find the quartiles and IQR for each category
    groups = df.groupby('model')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    cats = sorted(df['model'].unique())
    p = bokeh.plotting.figure(title=title,
                              x_range=cats,
                              x_axis_label="Model",
                              tools="",
                              toolbar_location=None,
                              y_axis_label=axis_title[statistic],
                              plot_height=BASE_HEIGHT,
                              sizing_mode="stretch_width")

    p.xaxis.major_label_orientation = math.pi / 4  # math.pi / 6
    p.xaxis.major_label_standoff = 20
    p.yaxis.major_label_standoff = 15

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper[statistic] = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, statistic]), upper[statistic])]
    lower[statistic] = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, statistic]), lower[statistic])]

    lw = 2

    # stems
    p.segment(cats, upper[statistic], cats, q3[statistic], line_color="black", line_width=lw)
    p.segment(cats, lower[statistic], cats, q1[statistic], line_color="black", line_width=lw)

    # boxes
    p.vbar(cats, 0.7, q2[statistic], q3[statistic], line_width=lw, fill_color="#E08E79", line_color="black")
    p.vbar(cats, 0.7, q1[statistic], q2[statistic], line_width=lw, fill_color="#3B8686", line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower[statistic], 0.2, 0.001, line_color="black", line_width=lw, fill_color="black")
    p.rect(cats, upper[statistic], 0.2, 0.001, line_color="black", line_width=lw, fill_color="black")

    # outliers
    def outliers(group: pd.DataFrame) -> pd.Series:
        cat = group.name
        return group[(group[statistic] > upper.loc[cat][statistic]) |
                     (group[statistic] < lower.loc[cat][statistic])][statistic]

    out = groups.apply(outliers).dropna()
    if not out.empty:
        outx = []
        outy = []
        for keys in out.index:
            outx.append(keys[0])
            outy.append(out[keys[0]][keys[1]])
        p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    export_plots(p, filename=os.path.join(out_folder, f"model_boxplot_{statistic}"), title=title, box=False)


def line_plot(run_table: pd.DataFrame,
              std_table: pd.DataFrame,
              out_folder: str,
              statistic: str = "ll",
              color_column: str = "run",
              y_range_start: Optional[float] = None,
              y_range_end: Optional[float] = None,
              subtitle: str = "") -> None:
    title = f"{stat_title[statistic]} ({subtitle})"

    # create a new plot
    tooltips = [('x', f'@x'), ("y", "@y"), ("label", "@label")]
    if std_table is not None:
        tooltips.append(("Stddev", f"@{statistic}_std"))
    p = bokeh.plotting.figure(tools="pan,crosshair,reset,save,wheel_zoom",
                              title=title,
                              x_axis_label="Epoch",
                              y_axis_label=axis_title[statistic],
                              tooltips=tooltips,
                              toolbar_location="above",
                              plot_height=BASE_HEIGHT,
                              sizing_mode="stretch_width")

    colors = bokeh.palettes.Category20[20]
    items = []
    for color, line in zip(colors, sorted(run_table[color_column].unique())):
        df = pd.DataFrame(run_table[run_table[color_column] == line])
        if std_table is not None:
            std = std_table[std_table[color_column] == line]
            std_key = f"{statistic}_std"
            df[std_key] = std[statistic]
            df["lower"] = df[statistic] - df[std_key]
            df["upper"] = df[statistic] + df[std_key]

        source = bokeh.models.ColumnDataSource(df)
        plotted_line = p.line(x="epoch", y=statistic, source=source, color=color, line_width=2)
        # plotted_points = p.scatter(x="epoch", y=statistic, source=source, color=color)
        renderers = [plotted_line]  # plotted_points
        if std_table is not None:
            band = bokeh.models.Band(base="epoch",
                                     lower="lower",
                                     upper="upper",
                                     source=source,
                                     level='underlay',
                                     fill_alpha=0.1,
                                     fill_color=color)
            p.add_layout(band)
            callback = bokeh.models.CustomJS(args=dict(band=band),
                                             code="""
            if (band.visible == false)
                band.visible = true;
            else
                band.visible = false; """)
            plotted_line.js_on_change('visible', callback)

        items.append(bokeh.models.LegendItem(label=line, renderers=renderers))

    legend = bokeh.models.Legend(items=items, location="center", orientation="vertical", click_policy="hide")
    p.add_layout(legend, "right")
    p.x_range.start = 0
    p.x_range.end = run_table["epoch"].max()

    # show the results
    export_plots(p,
                 filename=os.path.join(out_folder, f"{color_column}_lineplot_{statistic}"),
                 title=title,
                 y_range_start=y_range_start,
                 y_range_end=y_range_end)


def curvature_line_plot(run_table: pd.DataFrame,
                        std_table: pd.DataFrame,
                        out_folder: str,
                        color_column: str = "run",
                        subtitle: str = "") -> None:

    def filter_nan_cols(df: pd.DataFrame, ccval: str) -> pd.DataFrame:
        df = df[df[color_column] == ccval]
        df = df.dropna(axis=1, how="all")
        df = df[sorted([col for col in df if col.endswith("/curvature") or col.lower() == "epoch"])]
        return df

    tooltips = [('Run-Component', f'@label'), ("Epoch", "@epoch"), ("Curvature", f"@curvature")]
    if std_table is not None:
        tooltips.append(("StdDev", f"@curvature_std"))
    title = f"Learned curvature of components ({subtitle})"
    p = bokeh.plotting.figure(
        tools="pan,crosshair,reset,save,wheel_zoom",
        #  title=title,
        x_axis_label="Epoch",
        y_axis_label="Curvature",
        tooltips=tooltips,
        toolbar_location="above",
        plot_height=BASE_HEIGHT,
        sizing_mode="stretch_width")

    items = []
    colors = bokeh.palettes.Category20[20]
    color_iter = itertools.cycle(colors)

    for ccval in run_table[color_column].unique():
        if "fixed" in ccval:
            continue
        df = filter_nan_cols(run_table, ccval)
        if std_table is not None:
            std = filter_nan_cols(std_table, ccval)

        for curvature_col in [col for col in df if col.endswith("/curvature")]:
            label = f"{ccval}-{curvature_col[:curvature_col.find('/')]}"
            curvature_df = df.rename(index=str, columns={curvature_col: "curvature"})
            curvature_df['label'] = label
            if std_table is not None and curvature_col in std:
                std_key = f"curvature_std"
                curvature_std = std.rename(index=str, columns={curvature_col: std_key})
                curvature_df[std_key] = curvature_std[std_key]
                curvature_df["lower"] = curvature_df["curvature"] - curvature_df[std_key]
                curvature_df["upper"] = curvature_df["curvature"] + curvature_df[std_key]

            source = bokeh.models.ColumnDataSource(curvature_df)
            color = next(color_iter)
            plotted_line = p.line(x="epoch", y="curvature", source=source, color=color)
            plotted_points = p.scatter(x="epoch", y="curvature", source=source, color=color)
            renderers = [plotted_line, plotted_points]
            if std_table is not None and curvature_col in std:
                band = bokeh.models.Band(base="epoch",
                                         lower="lower",
                                         upper="upper",
                                         source=source,
                                         level='underlay',
                                         fill_alpha=0.1,
                                         fill_color=color)
                p.add_layout(band)
                callback = bokeh.models.CustomJS(args=dict(band=band),
                                                 code="""
                if (band.visible == false)
                    band.visible = true;
                else
                    band.visible = false; """)
                plotted_line.js_on_change('visible', callback)

            items.append(bokeh.models.LegendItem(label=label, renderers=renderers))

    legend = bokeh.models.Legend(items=items, location="center", orientation="vertical", click_policy="hide")
    p.add_layout(legend, "right")
    p.x_range.start = 0
    p.x_range.end = run_table["epoch"].max()

    export_plots(p, filename=os.path.join(out_folder, f"{color_column}_lineplot_curvature"), title=title)


def models_latex_table(mean: pd.DataFrame, std: pd.DataFrame, show_curvature: bool = False) -> str:

    def _last(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("model").last().reset_index()

    model_mean = _last(mean)
    model_std = _last(std)
    rows = [[f"{x} ${utils.texify_components(x)}$" for x in model_mean["model"]]]
    for key in ["ll"]:  # , "elbo", "bce", "kl"]:  # "mi", "cov_norm"]:
        if key not in model_mean.keys():
            warnings.warn(f"Key {key} not in data frame, skipping.")
            continue
        rows.append([])
        for m, s in zip(model_mean[key], model_std[key]):
            rows[-1].append(f"${m:0.2f}$" + "{\\scriptsize" + f"$\\pm {s:0.2f}$" + "}")

    def filter_nan_cols(df: pd.DataFrame, model: str) -> Tuple[np.ndarray, ...]:
        df = df[df["model"] == model]
        df = df.dropna(axis=1)
        npa = df[sorted([col for col in df if col.endswith("/curvature")])].to_numpy()
        return tuple(*npa)

    def print_tuple(t: Tuple[np.ndarray, ...], fmt: str = "0.3f") -> str:
        s = ", ".join((("{:" + fmt + "}").format(x) for x in t))
        return f"({s})"

    if show_curvature:
        rows.append([])
        for model in model_mean["model"]:
            model_mean_ = filter_nan_cols(model_mean, model)
            model_std_ = filter_nan_cols(model_std, model)

            if len(model_mean_) > 0:
                rows[-1].append(f"${print_tuple(model_mean_)} \\pm {print_tuple(model_std_)}$")

        curvature = True
        if not rows[-1]:
            curvature = False
            del rows[-1]

    rows_transposed = sorted(list(zip(*rows)), key=lambda x: x[0])
    rows_str = "\n".join(" & ".join(str(element) for element in row) + "\\\\" for row in rows_transposed)

    if show_curvature and curvature:
        return """
      \\begin{tabular}{l|rrrrrl}
        \\toprule
        \\textbf{Model} & LL & ELBO & BCE & KL & Curvature\\\\
        \\midrule
        """ + rows_str + """
        \\bottomrule
      \\end{tabular}
        """
    else:
        return """
              \\begin{tabular}{l|rrrrr}
                \\toprule
                \\textbf{Model} & LL & ELBO & BCE & KL \\\\
                \\midrule
                """ + rows_str + """
                \\bottomrule
              \\end{tabular}
                """


def merge_runs(runs: Dict[str, List[Tuple[datetime, pd.DataFrame]]]) -> pd.DataFrame:
    total_dfs = []
    for model, model_runs in runs.items():
        model_dfs = []
        for i, (time, df) in enumerate(sorted(model_runs, key=lambda x: x[0])):
            df["time"] = [time] * len(df)
            df["run"] = [f"{model} (run {i})"] * len(df)
            model_dfs.append(df)
        model_df = pd.concat(model_dfs, sort=True)
        model_df["model"] = [model] * len(model_df)
        total_dfs.append(model_df)
    return pd.concat(total_dfs, sort=False)


def mean_models(run_table: pd.DataFrame, by: List[str] = ["epoch", "model"]) -> pd.DataFrame:
    return run_table.groupby(by=by).mean().reset_index()


def std_models(run_table: pd.DataFrame, by: List[str] = ["epoch", "model"]) -> pd.DataFrame:
    return run_table.groupby(by=by).std().reset_index()


def read_runs(pattern: str) -> Dict[str, List[Tuple[datetime, pd.DataFrame, pd.DataFrame]]]:
    """
    :param pattern: the glob of logs to read.
    :return: A dictionary of modelname: [(time, eval_df, train_df)]
    """
    runs: Dict[str, List[Tuple[datetime, pd.DataFrame, pd.DataFrame]]] = defaultdict(list)
    for filename in glob.iglob(pattern, recursive=True):
        try:
            model_name, time, df, df_train = read_log.log_to_pd(filename)
        except NameError:  # For NaNs
            print(f"Couldn't read file '{filename}', skipping (probably a failed run).")
            continue

        print(f"Found model: '{model_name}' at time '{time}'.")
        runs[model_name].append((time, df, df_train))
    if not runs:
        raise ValueError(f"No runs found for glob '{pattern}'.")
    return runs


def early_stopping(
        dft: pd.DataFrame,
        dfe: pd.DataFrame,
        lookahead: int = 50,
        warmup: int = 500,  # don't do ES, we already do it during training
        stat: str = "elbo",
        difference: float = 0.0) -> pd.DataFrame:
    for run in dft["run"].unique():
        if "u" in run:
            continue  # Ignore universal model runs.
        r = dft[dft["run"] == run].set_index("epoch").reset_index()

        epochs = list(r["epoch"])
        stats = list(r[stat])
        assert len(epochs) == len(stats)

        should_stop = max(zip(epochs, stats), key=lambda x: x[1])[0]
        for epoch, val in zip(epochs, stats):
            if epoch < warmup:
                continue
            if epoch + lookahead >= len(stats):
                break
            if val >= max(stats[epoch + 1:epoch + lookahead + 1]) + difference:
                should_stop = epoch
                break
        # remove the runs of run which are above should_stop epochs
        dfe = dfe[(dfe["epoch"] <= should_stop) | (dfe["run"] != run)]

    return dfe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default="lsf.*", help="Which folder to search for log files.")
    parser.add_argument("--plot", type=str, default="runs", help="Type of plots ('runs', 'models').")
    parser.add_argument("--out_dir", type=str, default="./plots", help="Output dir.")
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--statistics", type=str, default="ll,mi,bce")
    args = parser.parse_args()

    statistics = [stat.strip() for stat in args.statistics.split(",")]

    bounds = {
        "ll": {
            # const
            "bdp_z6_const_1000": (-55., -60.1),
            "mnist_z6_const_200": (-96.5, -101.5),
            "mnist_z6_const_300": (-95.5, -99.0),
            "mnist_z6_const_500": (-95.5, -99.0),
            "mnist_z15_const_200": (-77., -87.1),
            "mnist_z30_const_200": (-75., -86.1),
            # learn
            "bdp_z6_learn_1000": (-55., -60.1),
            "mnist_z6_learn_300": (-95.5, -99.0)
        },
        "mi": {
            # const
            "bdp_z6_const_1000": (7., 3.),
            "mnist_z6_const_200": (13., 9.),
            "mnist_z6_const_300": (13., 9.),
            "mnist_z6_const_500": (13., 9.),
            "mnist_z15_const_200": (21., 16.),
            "mnist_z30_const_200": (26., 17.),
            # learn
            "bdp_z6_learn_1000": (7., 3.),
            "mnist_z6_learn_300": (13.0, 9.)
        }
    }

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    runs = read_runs(args.glob)
    df = merge_runs({k: [(dt, dfev) for dt, dfev, _ in runs[k]] for k in runs})
    df_train = merge_runs({k: [(dt, dftr) for dt, _, dftr in runs[k]] for k in runs})

    if args.plot == "runs":  # Single runs.
        color_column = "run"
        subtitle = "Run comparison"
        std = None
    elif args.plot == "models":  # Average runs of a model.
        subtitle = "Comparison across runs"

        df = early_stopping(df_train, df, warmup=100, lookahead=50, stat="elbo", difference=0.0)
        by_runs = df.groupby("run").last().reset_index()
        for statistic in statistics:
            box_whiskers_plot(by_runs, args.out_dir, statistic=statistic, subtitle=subtitle)

        std = std_models(by_runs, by=["model"])
        df = mean_models(by_runs, by=["model"])
        color_column = "model"

        table_str = models_latex_table(df, std, show_curvature=False)
        table_path = os.path.join(args.out_dir, "model_recon_table.tex")
        with open(table_path, "w") as f:
            print(table_str, file=f)
        print("Saved to", table_path)
    else:
        raise NotImplementedError(f"Invalid plot type {args.plot}.")

    for statistic in statistics:
        start = None
        end = None
        if statistic in bounds:
            if args.exp in bounds[statistic]:
                end, start = bounds[statistic][args.exp]
        line_plot(df,
                  std,
                  args.out_dir,
                  statistic=statistic,
                  color_column=color_column,
                  subtitle=subtitle,
                  y_range_start=start,
                  y_range_end=end)

    # Filter weird numbers
    for run in df_train["run"].unique():
        if (df_train[df_train["run"] == run].loc[:, df_train.columns.str.endswith("curvature")].abs() >
                1e1).any().any():
            print("Removing", run)
            df_train = df_train[df_train["run"] != run]

    dft_mean = mean_models(df_train)
    dft_std = std_models(df_train)
    curvature_line_plot(dft_mean, dft_std, args.out_dir, color_column=color_column, subtitle=subtitle)


if __name__ == "__main__":
    main()
