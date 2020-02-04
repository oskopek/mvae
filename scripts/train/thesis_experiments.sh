#!/usr/bin/env bash

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

runs=2

h_dim=400

doubles=True

declare -A epoch_map
epoch_map[bdp]=1000
epoch_map[mnist]=300
epoch_map[omniglot]=300
epoch_map[cifar]=200

declare -A z_dim_map
z_dim_map[bdp]="12"
z_dim_map[mnist]="6"
z_dim_map[omniglot]="6"
z_dim_map[cifar]="144"

declare -A likelihood_n_map
likelihood_n_map["bdp"]=500
likelihood_n_map["mnist"]=500
likelihood_n_map["omniglot"]=500
likelihood_n_map["cifar"]=500

declare -A beta_start_map
beta_start_map["bdp"]="1.0"
beta_start_map["mnist"]="1.0"
beta_start_map["omniglot"]="1.0"
beta_start_map["cifar"]="4.0"

beta_end="1.0"
beta_end_epoch=50

for fixed_curvature in "True" "False"; do
    for dataset in "bdp"; do
        epochs=${epoch_map[$dataset]}
        time="24:00"
        beta_start=${beta_start_map[$dataset]}
        likelihood_n=${likelihood_n_map[$dataset]}

        for z_dim in ${z_dim_map[$dataset]}; do
            echo -e "Dataset: $dataset;\tZ: $z_dim;\tH: $h_dim;\tFixed: $fixed_curvature;\tEpochs: $epochs;\tTime; $time; LikelihoodN: $likelihood_n; Beta: $beta_start -> $beta_end ($beta_end_epoch);\tRuns: $runs;\tDouble: $doubles"

            for run in `seq 1 $runs`; do
                ./scripts/train/run_experiments.sh "$time" "$epochs" "$z_dim" "$dataset" "$h_dim" "$fixed_curvature" "$likelihood_n" "$beta_start" "$beta_end" "$beta_end_epoch" "$doubles"
            done
            echo ""
        done
    done
done
