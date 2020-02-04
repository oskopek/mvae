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

for lsf in $(find . -maxdepth 1 -type f -name 'lsf.o*'); do
    dataset=$(head -n 3 "$lsf" | grep -Eo 'dataset=[^ ]*' | cut -d '=' -f 2)
    model=$(head -n 3 "$lsf" | grep -Eo 'model=[^ ]*' | cut -d '=' -f 2)
    fixed=$(head -n 3 "$lsf" | grep -Eo 'fixed_curvature=[^ ]*' | cut -d '=' -f 2)
    suffix=$(basename $lsf)
    mv -v $lsf $dataset.fixed${fixed}.$model.$suffix
done
