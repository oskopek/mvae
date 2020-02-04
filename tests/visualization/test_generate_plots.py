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

import mt.visualization.generate_plots as generate_plots

glob = "./tests/test_data/run*.txt"


def test_generate_table() -> None:
    run_dict = generate_plots.read_runs(glob)
    eval_dict = {k: [(v[0], v[1]) for v in run_dict[k]] for k in run_dict}
    df = generate_plots.merge_runs(eval_dict)

    std = generate_plots.std_models(df)
    df = generate_plots.mean_models(df)
    table_str = generate_plots.models_latex_table(df, std)
    assert "nan" not in table_str
    assert "\\euc^{2} \\times \\hyp_{-1}^{2} \\times \\sph_{1}^{2}" in table_str
