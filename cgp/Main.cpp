// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"
using namespace cgp;

std::shared_ptr<CGP> init_cgp(const std::string& cgp_file)
{
	std::shared_ptr<double[]> energy_costs = std::make_shared<double[]>(function_count);

	for (size_t i = 0; i < function_count; i++)
	{
		energy_costs[i] = 1;
	}

	std::ifstream cgp_in(cgp_file);
	auto cgp_model = std::make_shared<CGP>(cgp_in);
	cgp_in.close();
	cgp_model->function_energy_costs(energy_costs);
	return cgp_model;
}

int evaluate(std::vector<std::string>& arguments, size_t inputs_to_evaluate, const std::string& cgp_file, const std::string& solution = "")
{
	std::vector<std::shared_ptr<weight_value_t[]>> inputs;

	auto cgp_model_pointer = init_cgp(cgp_file);
	CGP& cgp_model = *cgp_model_pointer;
	cgp_model.set_from_arguments(arguments);

	auto in = get_input(cgp_model.input_file());
	auto out = get_output(cgp_model.output_file());

	// Read two values from the standard input
	if (cgp_model.input_count() == 0 || cgp_model.output_count() == 0)
	{
		std::cerr << "invalid input size and output size" << std::endl;
		return 2;
	}

	for (size_t i = 0; i < inputs_to_evaluate; i++)
	{
		std::cerr << "loading values" << std::endl;
		inputs.push_back(load_input(*in, cgp_model.input_count()));
	}

	if (!solution.empty())
	{
		cgp_model.restore(solution);
	}
	cgp_model.dump(std::cerr);
	log_weights(*out, inputs, cgp_model);

	std::cerr << std::endl << "exitting program" << std::endl;
	return 0;
}

int train(std::vector<std::string>& arguments, size_t pairs_to_approximate, const std::string& cgp_file)
{
	std::vector<std::shared_ptr<weight_value_t[]>> inputs, outputs;
	weight_repr_value_t min = std::numeric_limits<weight_repr_value_t>::max();
	weight_repr_value_t max = std::numeric_limits<weight_repr_value_t>::min();

	auto cgp_model_pointer = init_cgp(cgp_file);
	CGP& cgp_model = *cgp_model_pointer;
	cgp_model.set_from_arguments(arguments);

	auto in = get_input(cgp_model.input_file());
	auto stats_out = get_output(cgp_model.cgp_statistics_file());

	// Read two values from the standard input
	if (cgp_model.input_count() == 0 || cgp_model.output_count() == 0)
	{
		std::cerr << "invalid input size and output size" << std::endl;
		return 2;
	}

	for (size_t i = 0; i < pairs_to_approximate; i++)
	{
		std::cerr << "loading values" << std::endl;
		inputs.push_back(load_input(*in, cgp_model.input_count()));
		outputs.push_back(load_output(*in, cgp_model.output_count(), min, max));
	}


	auto generation_stop = 500000;
	cgp_model.build_indices();
	cgp_model.dump(std::cerr);
	cgp_model.generate_population();
	for (size_t run = 0; run < cgp_model.number_of_runs(); run++)
	{
		for (size_t i = 0; i < cgp_model.generation_count(); i++)
		{
			cgp_model.evaluate(inputs, outputs);
			if (cgp_model.get_generations_without_change() == 0)
			{
				log_csv(*stats_out, run, i, cgp_model);
				//log_human(std::cerr, run, i, cgp_model);
			}
			if (i % cgp_model.periodic_log_frequency() == 0) {
				log_human(std::cerr, run, i, cgp_model);
			}

			if (cgp_model.get_best_error_fitness() <= 0 || cgp_model.get_generations_without_change() > generation_stop)
			{
				log_human(std::cerr, run, i, cgp_model);
				break;
			}
			cgp_model.mutate();
		}

		std::cerr << "chromosome size: " << cgp_model.get_serialized_chromosome_size() << std::endl;
		std::cerr << "weights: " << std::endl;
		std::copy(cgp_model.get_best_chromosome()->begin_output(), cgp_model.get_best_chromosome()->end_output(), std::ostream_iterator<weight_repr_value_t>(std::cerr, ", "));
		std::cerr << std::endl;

		auto out = get_output(cgp_model.output_file() + "." + std::to_string(run));
		cgp_model.dump(*out);
		cgp_model.reset();
		cgp_model.generate_population();
	}

	std::cerr << std::endl << "exitting program" << std::endl;
	return 0;
}

int main(int argc, const char** args) {
	// Asserts floating point compatibility at compile time
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
	std::vector<std::string> arguments(args + 1, args + argc);
	//std::vector<std::string> arguments{"train", "1", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\cmd\\compress\\experiments\\single_filter\\config.cgp", "--output-file", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\cmd\\compress\\experiment_results\\single_filter\\data.cgp"};
	//std::vector<std::string> arguments{"evaluate", "1", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\cmd\\compress\\experiment_results\\single_filter\\data.cgp.1", "--output-file", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\cmd\\compress\\experiment_results\\single_filter\\inferred_weights.0"};

	try
	{
		if (arguments.size() >= 1 && arguments[0] == "evaluate")
		{
			if (arguments.size() < 2)
			{
				std::cerr << "missing argument for input quantity right after " + arguments[0] + "." << std::endl;
				return 12;
			}
			size_t quantity = parse_integer_argument(arguments[1]);

			if (arguments.size() < 3)
			{
				std::cerr << "missing argument for cgp configuraiton file" << std::endl;
				return 12;
			}
			std::string cgp_file = arguments[2];

			arguments.erase(arguments.begin(), arguments.begin() + 3);
			return evaluate(arguments, quantity, cgp_file);
		}
		else if (arguments.size() >= 1 && arguments[0] == "evaluate:inline")
		{
			if (arguments.size() < 2)
			{
				std::cerr << "missing argument for input quantity right after " + arguments[0] + "." << std::endl;
				return 12;
			}
			size_t quantity = parse_integer_argument(arguments[1]);

			if (arguments.size() < 3)
			{
				std::cerr << "missing argument for cgp configuraiton file" << std::endl;
				return 12;
			}
			std::string cgp_file = arguments[2];

			if (arguments.size() < 4)
			{
				std::cerr << "missing argument for serialized chromosome solution" << std::endl;
				return 13;
			}

			std::string solution = arguments[3];

			if (solution.empty())
			{
				std::cerr << "solution must not be empty string" << std::endl;
				return 14;
			}

			arguments.erase(arguments.begin(), arguments.begin() + 4);
			return evaluate(arguments, quantity, cgp_file, solution);
		}
		else if (arguments.size() >= 1 && arguments[0] == "train")
		{
			if (arguments.size() < 2)
			{
				std::cerr << "missing argument for input quantity right after " + arguments[0] + "." << std::endl;
				return 12;
			}

			size_t quantity = parse_integer_argument(arguments[1]);

			if (arguments.size() < 3)
			{
				std::cerr << "missing argument for cgp configuraiton file" << std::endl;
				return 12;
			}
			std::string cgp_file = arguments[2];

			arguments.erase(arguments.begin(), arguments.begin() + 3);
			return train(arguments, quantity, cgp_file);
		}
		else if (arguments.size() >= 1)
		{
			std::cerr << "invalid first argument " + arguments[0] + "; expected evaluate or train." << std::endl;
			return 10;
		}
		else {
			std::cerr << "missing first argument; expected evaluate or train." << std::endl;
			return 11;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 42;
	}
}
