#pragma once

#include "Configuration.h"
#include <vector>
#include <array>

namespace cgp
{
	using weight_input_t = CGPConfiguration::weight_input_t;
	using weight_output_t = CGPConfiguration::weight_output_t;
	using dataset_t = std::tuple<std::vector<weight_input_t>, std::vector<weight_output_t>, std::vector<int>, std::array<int, 256>>;

	const std::vector<weight_input_t>& get_dataset_input(const dataset_t& dataset);
	const std::vector<weight_output_t>& get_dataset_output(const dataset_t& dataset);
	const std::vector<int>& get_dataset_no_care(const dataset_t& dataset);
	const std::array<int, 256>& get_dataset_needed_quant_values(const dataset_t& dataset);
	void delete_dataset(const dataset_t& dataset);
}
