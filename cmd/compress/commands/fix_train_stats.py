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
# fix_train_statistics.py: Used to recalculate missing statistics. Eventually this got deprecated.

from commands.factory.experiment import create_all_experiment
def fix_train_statistics(args):
    for experiment in create_all_experiment(args):
        experiment = experiment.get_statistics_fix_env()
        print("fixing experiment: " + experiment.get_name())
        experiment.evaluate_chromosome_in_statistics()
