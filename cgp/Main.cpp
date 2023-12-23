// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"

int main(int argc, const char** args) {
	std::vector<double> numbers = {
		0.1346,  0.0341, -0.0078,  0.0476,  0.0381,
		0.0844, 0.0747,
		0.1176, -0.1313,
		0.0583, -0.1564,
		-0.1313, -0.0860, -0.0552, -0.1569, -0.0733
	};


	std::shared_ptr<double[]> input(new double[9]);
	input[0] = 0.0725;
	input[1] = 0.1039;
	input[2] = 0.0855;
	input[3] = 0.1212;
	input[4] = 0.0435;
	input[5] = 0.1364;
	input[6] = 0.0936;
	input[7] = 0.0342;
	input[8] = -0.0307;

	cgp::CGP cgp_model(numbers);

	cgp_model
		/*.col_count(30)
		.row_count(10)*/
		.col_count(5)
		.row_count(5)
		.look_back_parameter(5)
		.mutation_max(static_cast<uint16_t>(cgp_model.chromosome_size() * 0.15))
		.function_count(11)
		.function_input_arity(2)
		.function_output_arity(2)
		.input_count(9)
		.output_count(numbers.size())
		.population_max(100)
		.generation_count(900000000)
		.periodic_log_frequency(2500);
	cgp_model.build();

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

	std::cout << "weights: " << std::endl;
	std::copy(cgp_model.get_best_chromosome()->begin_output(), cgp_model.get_best_chromosome()->end_output(), std::ostream_iterator<double>(std::cout, ", "));
	std::cout << std::endl << "exitting program" << std::endl;
	return 0;
}
