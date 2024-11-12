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
# cli.py: Define CLI arguments for reversed_single_filter experiment.

import argparse
import experiments.grid_size.cli as gs

def get_argument_parser(parser: argparse._SubParsersAction):
    parser.add_argument("--prefix", default="", help="Prefix for experiment names")
    parser.add_argument("--suffix", default="", help="Suffix for experiment names")
    parser.add_argument("-n", default=5, type=int, help="Amount of filters to be tested")
    parser.add_argument("--layer-names", nargs="+", default=["conv1", "conv2"], help="List of CNN layer names")
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=gs.default_grid_sizes, help="List of grid sizes (rows, columns)")
    parser.add_argument("--reuse", type=str, default=None, help="Reuse experiment configuration from the other experiment")
    parser.add_argument("--name-format", type=str, default=None, help="Name format of the resued experiments")
    return parser
