#pragma once

#include <fstream>
#include <ios>
#include <string>
#include <vector>
#include "Cgp.h"

namespace cgp
{
	using weight_value_t = cgp::CGPConfiguration::weight_value_t;
	using weight_repr_value_t = cgp::CGPConfiguration::weight_repr_value_t;
	std::shared_ptr<std::istream> get_input(const std::string& in, std::shared_ptr<std::istream> default_input = nullptr, std::ios_base::openmode mode = std::ios::in);
	std::shared_ptr<std::ostream> get_output(const std::string& out, std::shared_ptr<std::ostream> default_output = nullptr, std::ios_base::openmode mode = std::ios::out | std::ios::trunc);
	void log_human(std::ostream& stream, size_t run, size_t generation, cgp::CGP& cgp_model);
	void log_csv(std::ostream& stream, size_t run, size_t generation, cgp::CGP& cgp_model, const std::string& timestmap);
	void log_weights(std::ostream& stream, const std::vector<std::shared_ptr<weight_value_t[]>>& inputs, cgp::CGP& cgp_model);
	std::shared_ptr<weight_value_t[]> load_input(std::istream& in, size_t input_size);
	std::shared_ptr<weight_value_t[]> load_output(std::istream& in, size_t output_size, weight_repr_value_t& min, weight_repr_value_t& max);
}
