// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"

using weight_value_t = cgp::CGPConfiguration::weight_value_t;

std::shared_ptr<std::istream> get_input(const std::string& in, std::shared_ptr<std::istream> default_input = nullptr, std::ios_base::openmode mode = std::ios::in) {
	std::shared_ptr<std::istream> stream;
	if (in == "-")
	{
		stream.reset(&std::cin, [](...) {});
	}
	else if (in.empty())
	{
		if (default_input == nullptr)
		{
			throw std::invalid_argument("default input must not be null when input file is not provided");
		}

		stream = default_input;
	}
	else
	{
		auto file = new std::ifstream(in, mode);

		if (!file->is_open())
		{
			delete file;
			throw std::ofstream::failure("could not open input file " + in);
		}
		stream.reset(file);
	}

	return stream;
}

std::shared_ptr<std::ostream> get_output(const std::string& out, std::shared_ptr<std::ostream> default_output = nullptr, std::ios_base::openmode mode = std::ios::out) {
	std::shared_ptr<std::ostream> stream;
	if (out == "-")
	{
		stream.reset(&std::cout, [](...) {});
	}
	else if (out.empty())
	{
		if (default_output == nullptr)
		{
			throw std::invalid_argument("default input must not be null when output file is not provided");
		}

		stream = default_output;
	}
	else
	{
		auto file = new std::ofstream(out, mode);

		if (!file->is_open())
		{
			delete file;
			throw std::ofstream::failure("could not open output file " + out);
		}
		stream.reset(file);
	}

	return stream;
}

int main(int argc, const char** args) {
	// Asserts floating point compatibility at compile time
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
	std::vector<std::string> arguments(args + 1, args + argc);
	std::vector<std::shared_ptr<weight_value_t[]>> inputs, outputs;
	std::shared_ptr<double[]> energy_costs = std::make_shared<double[]>(function_count);
	size_t pairs_to_approximate = 0;
	size_t input_size = 0;
	size_t output_size = 0;
	weight_repr_value_t weight;
	weight_repr_value_t min = std::numeric_limits<weight_repr_value_t>::max();
	weight_repr_value_t max = std::numeric_limits<weight_repr_value_t>::min();

	for (size_t i = 0; i < function_count; i++)
	{
		energy_costs[i] = 1;
	}

	cgp::CGP cgp_model(5);
	cgp_model
		.function_energy_costs(energy_costs)
		.col_count(50)
		.row_count(30)
		.look_back_parameter(50)
		.mutation_max(0.15)
		.function_count(function_count)
		.function_input_arity(2)
		.function_output_arity(1)
		.input_count(input_size)
		.output_count(output_size)
		.population_max(4)
		.generation_count(90000000)
		.periodic_log_frequency(2500)
		.chromosome_output_file("evolution.chr")
		.input_file("C:\\Users\\Majo\\source\\repos\\TorchCompresser\\train.data");
	cgp_model.set_from_arguments(arguments);

	auto in = get_input(cgp_model.input_file());
	auto out = get_output(cgp_model.output_file());

	// Get number of pairs
	if (!(*in >> pairs_to_approximate))
	{
		std::cerr << "invalid input; expected number of input/output pairs" << std::endl;
		return 1;
	}

	// Read two values from the standard input
	if (!(*in >> input_size >> output_size))
	{
		std::cerr << "invalid input size and output size" << std::endl;
		return 2;
	}

	cgp_model.input_count(input_size);
	cgp_model.output_count(output_size);

	for (size_t i = 0; i < pairs_to_approximate; i++)
	{
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
	auto generation_stop = 1e6;

	if (inputs.size() > 1) {
		for (size_t i = 0; i < cgp_model.generation_count(); i++)
		{
			cgp_model.evaluate(inputs, outputs);

			if (i % cgp_model.periodic_log_frequency() == 0) {
				std::cerr << "[" << (i + 1) << "] MSE: " << cgp_model.get_best_error_fitness() << ", Energy: " << cgp_model.get_best_energy_fitness() << std::endl;
			}

			if (cgp_model.get_best_error_fitness() == 0 || cgp_model.get_generations_without_change() > generation_stop)
			{
				break;
			}
			cgp_model.mutate();
		}
	}
	else {
		auto input = inputs[0];
		auto output = outputs[0];
		for (size_t i = 0; i < cgp_model.generation_count(); i++)
		{
			cgp_model.evaluate(input, output);

			if (i % cgp_model.periodic_log_frequency() == 0) {
				std::cerr << "[" << (i + 1) << "] MSE: " << cgp_model.get_best_error_fitness() << ", Energy: " << cgp_model.get_best_energy_fitness() << std::endl;
			}

			if (cgp_model.get_best_error_fitness() == 0 || cgp_model.get_generations_without_change() > generation_stop)
			{
				break;
			}
			cgp_model.mutate();
		}
	}


	try {
		auto chromosome_out = get_output(cgp_model.chromosome_output_file(), nullptr, std::ios::binary);
		std::cerr << *cgp_model.get_best_chromosome() << std::endl;
		*chromosome_out << *cgp_model.get_best_chromosome() << std::endl;
		std::cerr << "saved file" << std::endl;
	}
	catch (const std::ofstream::failure& e)
	{
		std::cerr << "file I/O error: " << e.what() << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
	}


	std::cerr << "chromosome size: " << cgp_model.get_serialized_chromosome_size() << std::endl;
	*out << "weights: " << std::endl;
	std::copy(cgp_model.get_best_chromosome()->begin_output(), cgp_model.get_best_chromosome()->end_output(), std::ostream_iterator<weight_repr_value_t>(*out, ", "));
	std::cerr << std::endl << "exitting program" << std::endl;
	return 0;
}
