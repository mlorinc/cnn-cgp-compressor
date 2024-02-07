// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"

int main(int argc, const char** args) {
	// Asserts floating point compatibility at compile time
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

	size_t layers_to_approximate = 0;
	size_t input_size = 0;
	size_t output_size = 0;
	double weight;

	// Read three values from the standard input
	if (!(std::cin >> layers_to_approximate >> input_size >> output_size)) {
		std::cerr << "invalid input; 3 values required: number of layers to approximate, input size and output size" << std::endl;
		return 1;
	}

	std::shared_ptr<double[]> input(new double[input_size]);
	std::shared_ptr<double[]> output(new double[output_size]);
	double min = std::numeric_limits<double>::infinity(), max = -std::numeric_limits<double>::infinity();

	std::cout << "loading values" << std::endl;
	for (size_t i = 0; i < input_size; i++)
	{
		if (std::cin >> weight)
		{
			input[i] = weight;
		}
		else {
			std::cerr << "invalit input weight: expecting double value; got " << weight << std::endl;
			return 1;
		}
	}

	for (size_t i = 0; i < output_size; i++)
	{
		if (std::cin >> weight)
		{
			output[i] = weight;
			min = std::min(min, weight);
			max = std::max(max, weight);
		}
		else {
			std::cerr << "invalit output weight: expecting double value; got " << weight << std::endl;
			return 1;
		}
	}

	std::cout << "loaded values" << std::endl;

	cgp::CGP cgp_model(output, output_size, min, max);

	cgp_model
		.col_count(20)
		.row_count(50)
		/*.col_count(5)
		.row_count(5)*/
		.look_back_parameter(50)
		.mutation_max(static_cast<uint16_t>(cgp_model.chromosome_size() * 0.15))
		.function_count(14)
		.function_input_arity(2)
		.function_output_arity(1)
		.input_count(input_size)
		.output_count(output_size)
		.population_max(100)
		.generation_count(900000000)
		.periodic_log_frequency(2500);
	cgp_model.build();

	double max_mse = std::max(std::abs(min), std::abs(max));
	max_mse = output_size * max_mse;
	max_mse = max_mse * max_mse;
	max_mse /= output_size;

	std::cout << "chromosome size: " << cgp_model.get_serialized_chromosome_size() << std::endl;
	std::cout << "input count: " << cgp_model.input_count() << std::endl;
	std::cout << "output count: " << cgp_model.output_count() << std::endl;
	std::cout << "min value: " << min << std::endl;
	std::cout << "max value: " << max << std::endl;
	std::cout << "highest possible MSE: " << max_mse << std::endl;
	auto generation_stop = 50 * cgp_model.periodic_log_frequency();
	for (size_t i = 0; i < cgp_model.generation_count(); i++)
	{
		cgp_model.evaluate(input);

		if (i % cgp_model.periodic_log_frequency() == 0) {
			std::cout << "[" << (i + 1) << "] MSE: " << cgp_model.get_best_fitness() << std::endl;
		}

		if (cgp_model.get_best_fitness() == 0 || cgp_model.get_generations_without_change() > generation_stop)
		{
			break;
		}
		cgp_model.mutate();
	}


	std::ofstream chromosome_file("evolution.chr", std::ios::binary);

	// Check whether the file is successfully opened
	if (chromosome_file.is_open()) {
		chromosome_file << *cgp_model.get_best_chromosome() << std::endl;
		std::cout << *cgp_model.get_best_chromosome() << std::endl;
		std::cout << "saved file" << std::endl;
	}
	else {
		// Handle file opening failure
		std::cerr << "error: unable to open file 'evolution.chr'" << std::endl;
	}

	std::cout << "chromosome size: " << cgp_model.get_serialized_chromosome_size() << std::endl;
	std::cout << "weights: " << std::endl;
	std::copy(cgp_model.get_best_chromosome()->begin_output(), cgp_model.get_best_chromosome()->end_output(), std::ostream_iterator<double>(std::cout, ", "));
	std::cout << std::endl << "exitting program" << std::endl;
	return 0;
}
