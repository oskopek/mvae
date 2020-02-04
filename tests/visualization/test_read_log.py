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

import pytest

import mt.visualization.read_log as read_log


@pytest.mark.parametrize("file", ["run1.txt", "run2.txt"])
def test_log_to_pd(file: str) -> None:
    name, time, df, df_train = read_log.log_to_pd(f"./tests/test_data/{file}")
    assert name == "e2,h2,s2-fixed"
    assert read_log.from_iso_format("2019-05-25T20:48:00.0") < time < read_log.from_iso_format("2019-05-25T20:50:00.0")
    assert df.shape == (500, 9)
    assert df.shape == df_train.shape
    assert set(df.keys()) == {
        "epoch", "bce", "kl", "elbo", "ll", "mi", "comp_000_e2/curvature", "comp_001_h2/curvature",
        "comp_002_s2/curvature"
    }
