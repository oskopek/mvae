#!/bin/bash

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

set -e

prepend () {
    while read line; do echo "    $line"; done
}


dir="$HOME/git/mvae-paper/results-fin/checkpoints"
mod="mt.visualization.latent_space"
for folder in `ls -A $dir/`; do
    folder=$(basename $folder)
    echo $folder
    python -m "$mod" --path "$dir/$folder/" 2>/dev/null | prepend
done

res="$HOME/git/mvae-paper"
model="mnist_z6_const_200"
for suffix in "png" "html"; do
    mv "$dir/e6/eval_comp_000_e6_embeddings-00200.$suffix" "$res/plots/$model/e6_latent_space.$suffix"
    mv "$dir/h6/eval_comp_000_h6_embeddings-00200.$suffix" "$res/plots/$model/h6_latent_space.$suffix"
    mv "$dir/s6/eval_comp_000_s6_embeddings-00200.$suffix" "$res/plots/$model/s6_latent_space.$suffix"
    mv "$dir/e2,h2,s2/eval_comp_000_e2_embeddings-00200.$suffix" "$res/plots/$model/e2h2s2_latent_space_e2.$suffix"
    mv "$dir/e2,h2,s2/eval_comp_001_h2_embeddings-00200.$suffix" "$res/plots/$model/e2h2s2_latent_space_h2.$suffix"
    mv "$dir/e2,h2,s2/eval_comp_002_s2_embeddings-00200.$suffix" "$res/plots/$model/e2h2s2_latent_space_s2.$suffix"
    mv "$dir/e2,h2,s2/eval_total_embeddings-00200.$suffix" "$res/plots/$model/e2h2s2_latent_space_total.$suffix"
done
