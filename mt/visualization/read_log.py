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

import datetime
import os
from numbers import Number
from typing import Dict, List, Tuple

import pandas as pd


def from_iso_format(time: str) -> datetime.datetime:
    return datetime.datetime.strptime(time, "%Y-%m-%dt%H:%M:%S.%f")


def log_to_pd(logfile: str) -> Tuple[str, datetime.datetime, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(logfile):
        raise ValueError(f"Cannot read logs from non-existent file: '{logfile}'")
    d: Dict[str, List[Number]] = {}
    d_train: Dict[str, List[Number]] = {}
    name = None
    time = None
    with open(logfile, "r") as f:
        for line in f:
            line = line.strip().lower()
            if "vae model" in line:
                split = line.strip().split(";")
                name = split[0].split(":")[-1].strip()
                str_time = split[2].split(":", maxsplit=1)[-1].strip()
                time = from_iso_format(str_time)

                if len(split) < 4:
                    fixed_curvature = True
                else:
                    fixed_curvature = split[3].split(":", maxsplit=1)[-1].strip() == "true"
                if fixed_curvature:
                    name += "-fixed"

                continue

            if not line.endswith("}"):
                continue

            if line.startswith("epoch "):
                train = False
            elif line.startswith("trainepoch "):
                train = True
            else:
                continue

            idx = line.find(":")
            epoch = int(line[:idx].strip().split(" ")[1])
            dictionary = eval(line[idx + 1:].strip())
            dictionary["epoch"] = epoch
            for key in dictionary:
                if key not in d:
                    d[key] = []
                    d_train[key] = []
            for key in d:
                if train:
                    d_train[key].append(dictionary[key])
                else:
                    d[key].append(dictionary[key])

    if name is None or time is None:
        raise ValueError(f"Unable to parse name ({name}) or time ({time}) from logs.")
    return name, time, pd.DataFrame(d), pd.DataFrame(d_train)
