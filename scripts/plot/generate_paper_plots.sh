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

dir="$HOME/git/mvae-paper"
mod="mt.visualization.generate_plots"

rm -f plots/*
for folder in $(find $dir/results-fin2/ -type d -name '*_*_redacted'); do
    folder=$(basename $folder)
    echo $folder
    python -m "$mod" --plot models --glob "$dir/results-fin2/$folder/*" --exp $folder --statistics "ll" | prepend
    rm -rf "$dir/plots/$folder"
    nonred=$(echo $folder | sed -E 's@/[^/]*$@@')
    cp -r plots/ "$dir/plots/$nonred"
done

