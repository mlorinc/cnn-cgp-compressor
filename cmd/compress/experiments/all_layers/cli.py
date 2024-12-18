# Copyright 2024 Mari�n Lorinc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     LICENSE.txt file
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# cli.py: Define CLI arguments for all_layers experiment.

import argparse

thresholds = [250, 150, 100, 50, 25, 15, 10, 0]

def get_argument_parser(parser: argparse._SubParsersAction):
    parser.add_argument("--layer-names", nargs="+", default=["conv1", "conv2"], help="List of the input CNN layer names")
    parser.add_argument("--kernel-dimension", type=int, default=5, help="Convolution kernel size")
    parser.add_argument("--kernel-core-dimension", type=int, default=3, help="Convolution core kernel size")
    parser.add_argument("--prefix", default="", help="Prefix for experiment names")
    parser.add_argument("--suffix", default="", help="Suffix for experiment names")
    parser.add_argument("--mse-thresholds", nargs="+", type=float, default=thresholds, help="List of MSE thresholds")
    parser.add_argument("--rows", type=int, default=30, help="Number of rows per filter")
    parser.add_argument("--cols", type=int, default=7, help="Number of columns per layer")
    return parser
