# Copyright 2024 Marián Lorinc
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
# cli.py: Define CLI arguments for single_channel experiment.

import argparse

layer_name = "conv1"
thresholds = [0, 1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100, 200]
rows_per_filter = 5
channel = 0

def get_argument_parser(parser: argparse._SubParsersAction):
    parser.add_argument("--layer-name", default=layer_name, help="Name of the layer")
    parser.add_argument("--channel", type=int, default=channel, help="Channel index")
    parser.add_argument("--prefix", default="", help="Prefix for experiment names")
    parser.add_argument("--suffix", default="", help="Suffix for experiment names")
    parser.add_argument("--mse-thresholds", nargs="+", type=int, default=thresholds, help="List of MSE thresholds")
    parser.add_argument("--rows-per-filter", type=int, default=rows_per_filter, help="CGP rows used per filter for circuit design")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows per filter")
    parser.add_argument("--cols", type=int, default=None, help="Number of columns per layer")
    return parser
