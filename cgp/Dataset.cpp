// Copyright 2024 Mari�n Lorinc
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     LICENSE.txt file
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Dataset.cpp: Class dataset utlity implementation which are essential to CGP training.

#include "Dataset.h"

using namespace cgp;

const std::vector<weight_input_t>& cgp::get_dataset_input(const dataset_t& dataset)
{
	return std::get<0>(dataset);
}

const std::vector<weight_output_t>& cgp::get_dataset_output(const dataset_t& dataset)
{
	return std::get<1>(dataset);
}

const std::vector<int>& cgp::get_dataset_no_care(const dataset_t& dataset)
{
	return std::get<2>(dataset);
}

const std::array<int, 256>& cgp::get_dataset_needed_quant_values(const dataset_t& dataset)
{
	return std::get<3>(dataset);
}

const std::array<int, 256>& cgp::get_dataset_usage_vector(const dataset_t& dataset)
{
	return std::get<4>(dataset);
}

static std::vector<weight_input_t> get_dataset_input_helper(const dataset_t& dataset)
{
	return std::get<0>(dataset);
}

static std::vector<weight_output_t> get_dataset_output_output(const dataset_t& dataset)
{
	return std::get<1>(dataset);
}

void cgp::delete_dataset(const dataset_t& dataset)
{
	auto in = get_dataset_input_helper(dataset);
	auto out = get_dataset_output_output(dataset);
	for (int i = 0; i < in.size(); i++)
	{
		if (in[i] != nullptr) delete[] in[i];
		if (out[i] != nullptr) delete[] out[i];
		in[i] = nullptr;
		out[i] = nullptr;
	}
}
