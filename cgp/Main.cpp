// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"

using weight_value_t = cgp::CGPConfiguration::weight_value_t;

std::shared_ptr<std::istream> get_input(cgp::CGPConfiguration& config) {
	std::shared_ptr<std::istream> input;
	if (config.input_file() == "-")
	{
		input.reset(&std::cin, [](...) {});
	}
	else
	{
		auto file = new std::ifstream(config.input_file());

		if (!file->is_open())
		{
			delete file;
			throw std::runtime_error("could not open input file " + config.input_file());
		}
		input.reset(file);
	}

	return input;
}

std::shared_ptr<std::ostream> get_output(cgp::CGPConfiguration& config) {
	std::shared_ptr<std::ostream> output;
	if (config.input_file() == "-")
	{
		output.reset(&std::cout, [](...) {});
	}
	else
	{
		auto file = new std::ofstream(config.output_file());

		if (!file->is_open())
		{
			delete file;
			throw std::runtime_error("could not open output file " + config.input_file());
		}
		output.reset(file);
	}

	return output;
}

int main(int argc, const char** args) {
	// Asserts floating point compatibility at compile time
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
	std::vector<std::string> arguments(args + 1, args + argc);
	std::vector<std::shared_ptr<weight_value_t[]>> inputs, outputs;
	size_t pairs_to_approximate = 0;
	size_t input_size = 0;
	size_t output_size = 0;
	weight_repr_value_t weight;
	weight_repr_value_t min = std::numeric_limits<weight_repr_value_t>::max();
	weight_repr_value_t max = std::numeric_limits<weight_repr_value_t>::min();

	cgp::CGP cgp_model;
	cgp_model
		.col_count(20)
		.row_count(50)
		.look_back_parameter(20)
		.mutation_max(0.15)
		.function_count(function_count)
		.function_input_arity(2)
		.function_output_arity(1)
		.input_count(input_size)
		.output_count(output_size)
		.population_max(100)
		.generation_count(900000000)
		.periodic_log_frequency(2500);
	cgp_model.set_from_arguments(arguments);

	auto in = get_input(cgp_model);
	auto out = get_output(cgp_model);

	// Get number of pairs
	if (!(*in >> pairs_to_approximate))
	{
		std::cerr << "invalid input; expected number of input/output pairs" << std::endl;
		return 1;
	}


	for (size_t i = 0; i < pairs_to_approximate; i++)
	{
		// Read two values from the standard input
		if (!(*in >> input_size >> output_size))
		{
			std::cerr << "invalid input size and output size" << std::endl;
			return 2;
		}

		std::shared_ptr<weight_value_t[]> input(new weight_value_t[input_size]);
		std::shared_ptr<weight_value_t[]> output(new weight_value_t[output_size]);

		inputs.push_back(input);
		outputs.push_back(output);

		std::cerr << "loading values" << std::endl;
		for (size_t i = 0; i < input_size; i++)
		{
			if (*in >> weight)
			{
				input[i] = static_cast<weight_value_t>(weight);
			}
			else {
				// todo change error message
				std::cerr << "invalit input weight: expecting double value; got " << weight << std::endl;
				return 3;
			}
		}

		for (size_t i = 0; i < output_size; i++)
		{
			if (*in >> weight)
			{
				output[i] = static_cast<weight_value_t>(weight);
				min = std::min(min, weight);
				max = std::max(max, weight);
			}
			else {
				// todo change error message
				std::cerr << "invalit output weight: expecting double value; got " << weight << std::endl;
				return 4;
			}
		}
	}

	std::cerr << "loaded values" << std::endl;
	cgp_model.build();
	std::cerr << "chromosome size: " << cgp_model.get_serialized_chromosome_size() << std::endl;
	std::cerr << "input count: " << cgp_model.input_count() << std::endl;
	std::cerr << "output count: " << cgp_model.output_count() << std::endl;
	std::cerr << "min value: " << min << std::endl;
	std::cerr << "max value: " << max << std::endl;
	auto generation_stop = 50 * cgp_model.periodic_log_frequency();
	for (size_t i = 0; i < cgp_model.generation_count(); i++)
	{
		cgp_model.evaluate(inputs, outputs);

		if (i % cgp_model.periodic_log_frequency() == 0) {
			std::cerr << "[" << (i + 1) << "] MSE: " << cgp_model.get_best_error_fitness() << std::endl;
		}

		if (cgp_model.get_best_error_fitness() == 0 || cgp_model.get_generations_without_change() > generation_stop)
		{
			break;
		}
		cgp_model.mutate();
	}


	std::ofstream chromosome_file("evolution.chr", std::ios::binary);

	// Check whether the file is successfully opened
	if (chromosome_file.is_open()) {
		chromosome_file << *cgp_model.get_best_chromosome() << std::endl;
		std::cerr << *cgp_model.get_best_chromosome() << std::endl;
		std::cerr << "saved file" << std::endl;
	}
	else {
		// Handle file opening failure
		std::cerr << "error: unable to open file 'evolution.chr'" << std::endl;
	}

	*out << "chromosome size: " << cgp_model.get_serialized_chromosome_size() << std::endl;
	*out << "weights: " << std::endl;
	std::copy(cgp_model.get_best_chromosome()->begin_output(), cgp_model.get_best_chromosome()->end_output(), std::ostream_iterator<weight_repr_value_t>(*out, ", "));
	*out << std::endl << "exitting program" << std::endl;
	return 0;
}
