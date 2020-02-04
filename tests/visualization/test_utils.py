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

import mt.visualization.utils as utils


def test_texify_components() -> None:
    assert utils.texify_components("e2,h2,s2") == "\\euc^{2} \\times \\hyp^{2} \\times \\sph^{2}"
    assert utils.texify_components("e2,2h2,3s2") == "\\euc^{2} \\times (\\hyp^{2})^{2} \\times (\\sph^{2})^{3}"
    assert utils.texify_components("d2,2u2,3p2") == "\\spr^{2} \\times (\\uni^{2})^{2} \\times (\\poi^{2})^{3}"


def test_texify_components_custom_to_latex() -> None:
    to_latex = {"s": "\\mathbb{S}", "h": "\\mathbb{H}", "e": "\\mathbb{R}"}
    texified = utils.texify_components("e2,2h2,3s2", to_latex=to_latex)
    assert texified == "\\mathbb{R}^{2} \\times (\\mathbb{H}^{2})^{2} \\times (\\mathbb{S}^{2})^{3}"
