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

######## SETUP #########

# Job parameters (only Leonhard).
mem=512
time_per_job=${1:-"24:00"}
#cpu="rusage[mem=${mem}]"
gpu="rusage[mem=${mem}, ngpus_excl_p=1]"
nodes=8

# Job parameters (only local).
gpuid=0

# Model parameters.
epochs=${2:-"300"}
zdim=${3:-"6"}  # 6, 15, 30, 50
dataset=${4:-"mnist"}  # "bdp", "omniglot", "mnist", "cifar"
h_dim=${5:-"400"}
fixed_curvature=${6:-"False"}
likelihood_n=${7:-500}
beta_start=${8:-"1.0"}
beta_end=${9:-"1.0"}
beta_end_epoch=${10:-1}

######## END OF SETUP #########

gpujob_local () {
    gpuid="$1"
    CUDA_VISIBLE_DEVICES="${gpuid}" python -O -m mt.examples.run --device=cuda ${@:2} 2>&1 >"${dataset}.fixed${fixed_curvature}.${model}.${start_time}.log"
}

gpujub_leo () {
    bsub -W ${time_per_job} -n ${nodes} -R "${gpu}" "python -O -m mt.examples.run --device=cuda ${@}" 2>&1
}

if [[ $zdim -eq 12 ]]; then
    models="6e2,6h2,6s2
    18e2
    6e2,6p2,6d2"
elif [[ $zdim -eq 144 ]]; then
    models="24e2,24h2,24s2
    72e2
    72h2
    72s2
    72p2"
#    24e2,24p2,24d2
#    72d2
#    72u2"
else
    echo "Unknown z_dim: ${zdim}."
    exit 1
fi

# Job scheduling.
for model in ${models}; do
    args="--model=${model} \
          --epochs=${epochs} \
          --dataset=${dataset} \
          --fixed_curvature=${fixed_curvature} \
          --h_dim=${h_dim} \
          --beta_start=${beta_start} \
          --beta_start=${beta_end} \
          --beta_end_epoch=${beta_end_epoch} \
          --likelihood_n=${likelihood_n} \
          --doubles=True \
         "

    start_time="$(date -u '+%s%N')"
    if [[ $(hostname -d) == "leonhard.ethz.ch" ]]; then
        gpujub_leo $args
        sleep 2s
    else
        gpujob_local ${gpuid} $args
    fi
done
