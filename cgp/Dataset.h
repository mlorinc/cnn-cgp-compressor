// Copyright 2024 Marián Lorinc
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
// Dataset.h: Class dataset utlity declarations which are essential to CGP training.

#pragma once

#include "Configuration.h"
#include <vector>
#include <array>

namespace cgp
{
	using weight_input_t = CGPConfiguration::weight_input_t;
	using weight_output_t = CGPConfiguration::weight_output_t;
	using dataset_t = std::tuple<std::vector<weight_input_t>, std::vector<weight_output_t>, std::vector<int>, std::array<int, 256>, std::array<int, 256>>;

	/// <summary>
	/// Get all dataset input wieghts.
	/// </summary>
	/// <param name="dataset">The dataset to get inputs from.</param>
	/// <returns>Vector of byte arrays.</returns>
	const std::vector<weight_input_t>& get_dataset_input(const dataset_t& dataset);

	/// <summary>
	/// Get all dataset output wieghts.
	/// </summary>
	/// <param name="dataset">The dataset to get outputs from.</param>
	/// <returns>Vector of byte arrays.</returns>	
	const std::vector<weight_output_t>& get_dataset_output(const dataset_t& dataset);

	/// <summary>
	/// Get no care values after which fitness evaluation must ignore output weights.
	/// </summary>
	/// <param name="dataset">The dataset to get the values from.</param>
	/// <returns>Vector of no care values.</returns>	
	const std::vector<int>& get_dataset_no_care(const dataset_t& dataset);

	/// <summary>
	/// Get weights indicating how convolution weights are important in multiplex optimsiation.
	/// </summary>
	/// <param name="dataset">The dataset to get the values from.</param>
	/// <returns>Array of weights.</returns>	
	const std::array<int, 256>& get_dataset_needed_quant_values(const dataset_t& dataset);

	/// <summary>
	/// Get usage map with values only 0 (unused) or 1 (used).
	/// </summary>
	/// <param name="dataset">The dataset to get the values from.</param>
	/// <returns>Array of usage.</returns>		
	const std::array<int, 256>& get_dataset_usage_vector(const dataset_t& dataset);
	void delete_dataset(const dataset_t& dataset);
}
