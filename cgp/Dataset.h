#pragma once

#include "Configuration.h"

namespace cgp
{
	using weight_input_t = CGPConfiguration::weight_input_t;
	using weight_output_t = CGPConfiguration::weight_output_t;
	using dataset_t = std::tuple<std::vector<weight_input_t>, std::vector<weight_output_t>>;

	const std::vector<weight_input_t>& get_dataset_input(const dataset_t& dataset);
	const std::vector<weight_output_t>& get_dataset_output(const dataset_t& dataset);
	void delete_dataset(const dataset_t& dataset);
}
