# Copyright 2022 The ml_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setuptool-based build for ml_dtypes."""

import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "ml_dtypes._custom_floats",
            [
                "ml_dtypes/_src/dtypes.cc",
                "ml_dtypes/_src/numpy.cc",
            ],
            include_dirs=[
                ".",
                "ml_dtypes",
                np.get_include(),
            ],
            extra_compile_args=[
                "-std=c++17",
                "-DEIGEN_MPL2_ONLY",
            ],
        )
    ]
)