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
