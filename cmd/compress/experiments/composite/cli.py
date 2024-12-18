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
# cli.py: Define CLI arguments for composite experiments.

import argparse

def get_argument_parser(parser: argparse._SubParsersAction):
    parser.add_argument("-b", "--batches", default=None, type=int, help="Split single experiment into multiple smaller batches")    
    return parser