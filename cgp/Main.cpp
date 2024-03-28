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

std::shared_ptr<std::ostream> get_output(const std::string& out, std::shared_ptr<std::ostream> default_output = nullptr, std::ios_base::openmode mode = std::ios::out | std::ios::trunc) {
	std::shared_ptr<std::ostream> stream;
	if (out == "-")
	{
		stream.reset(&std::cout, [](...) {});
	}
	else if (out == "+")
	{
		stream.reset(&std::cerr, [](...) {});
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

void log_human(std::ostream& stream, size_t run, size_t generation, cgp::CGP& cgp_model)
{
	stream << "[" << (run + 1) << ", " << (generation + 1) << "] MSE: " << cgp_model.get_best_error_fitness() << ", Energy: " << cgp_model.get_best_energy_fitness() << std::endl;
}

void log_csv(std::ostream& stream, size_t run, size_t generation, cgp::CGP& cgp_model)
{
	//",\"" << *cgp_model.get_best_chromosome() << "\""
	stream << (run + 1) << "," << (generation + 1) << "," << cgp_model.get_best_error_fitness() << "," << cgp_model.get_best_energy_fitness() << ",\"" << cgp_model.get_best_chromosome()->to_string() << "\"" << std::endl;
}

void log_weights(std::ostream& stream, const std::vector<std::shared_ptr<weight_value_t[]>>& inputs, cgp::CGP& cgp_model)
{
	for (const auto& in : inputs) {
		auto weights = cgp_model.get_best_chromosome()->get_weights(in);
		std::copy(weights.get(), weights.get() + cgp_model.output_count(), std::ostream_iterator<weight_repr_value_t>(stream, " "));
		stream << std::endl;
	}
}

std::shared_ptr<weight_value_t[]> load_input(std::istream& in, size_t input_size)
{
	weight_repr_value_t weight;
	std::shared_ptr<weight_value_t[]> input = std::make_shared<weight_value_t[]>(input_size);

	for (size_t i = 0; i < input_size; i++)
	{
		if (in >> weight)
		{
			input[i] = static_cast<weight_value_t>(weight);
		}
		else {
			throw std::invalid_argument("invalit input weight: expecting double value; got " + std::to_string(weight));
		}
	}
	std::copy(input.get(), input.get() + input_size, std::ostream_iterator<weight_repr_value_t>(std::cout, " "));
	std::cout << std::endl;
	return input;
}

std::shared_ptr<weight_value_t[]> load_output(std::istream& in, size_t output_size, weight_repr_value_t& min, weight_repr_value_t& max)
{
	weight_repr_value_t weight;
	std::shared_ptr<weight_value_t[]> output = std::make_shared<weight_value_t[]>(output_size);

	for (size_t i = 0; i < output_size; i++)
	{
		if (in >> weight)
		{
			output[i] = static_cast<weight_value_t>(weight);
			min = std::min(min, weight);
			max = std::max(max, weight);
		}
		else {
			throw std::invalid_argument("invalit output weight: expecting double value; got " + std::to_string(weight));
		}
	}
	return output;
}

int evaluate(std::vector<std::string>& arguments, size_t inputs_to_evaluate, const std::string &cgp_file)
{
	std::vector<std::shared_ptr<weight_value_t[]>> inputs;
	std::shared_ptr<double[]> energy_costs = std::make_shared<double[]>(function_count);

	for (size_t i = 0; i < function_count; i++)
	{
		energy_costs[i] = 1;
	}

	std::ifstream cgp_in(cgp_file);
	cgp::CGP cgp_model(cgp_in);
	cgp_in.close();
	cgp_model.function_energy_costs(energy_costs);
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

	cgp_model.dump(std::cerr);
	log_weights(*out, inputs, cgp_model);
	
	std::cerr << std::endl << "exitting program" << std::endl;
	return 0;
}

int train(std::vector<std::string> &arguments, size_t pairs_to_approximate, const std::string& cgp_file)
{
	std::vector<std::shared_ptr<weight_value_t[]>> inputs, outputs;
	weight_repr_value_t min = std::numeric_limits<weight_repr_value_t>::max();
	weight_repr_value_t max = std::numeric_limits<weight_repr_value_t>::min();
	std::shared_ptr<double[]> energy_costs = std::make_shared<double[]>(function_count);

	for (size_t i = 0; i < function_count; i++)
	{
		energy_costs[i] = 1;
	}

	std::ifstream cgp_in(cgp_file);
	cgp::CGP cgp_model(cgp_in);
	cgp_in.close();
	cgp_model.set_from_arguments(arguments);
	cgp_model.function_energy_costs(energy_costs);

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
	//std::vector<std::string> arguments(args + 1, args + argc);
	//std::vector<std::string> arguments{"train", "1", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\cmd\\compress\\experiments\\single_filter\\config.cgp", "--output-file", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\cmd\\compress\\experiment_results\\single_filter\\data.cgp"};
	std::vector<std::string> arguments{"evaluate", "1", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\cmd\\compress\\experiment_results\\single_filter\\data.cgp.1", "--output-file", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\cmd\\compress\\experiment_results\\single_filter\\inferred_weights.0"};

	try 
	{
		if (arguments.size() >= 1 && arguments[0] == "evaluate")
		{
			if (arguments.size() < 2)
			{
				std::cerr << "missing argument for input quantity right after " + arguments[0] + "." << std::endl;
				return 12;
			}
			size_t quantity = cgp::parse_integer_argument(arguments[1]);

			if (arguments.size() < 3)
			{
				std::cerr << "missing argument for cgp configuraiton file" << std::endl;
				return 12;
			}
			std::string cgp_file = arguments[2];

			arguments.erase(arguments.begin(), arguments.begin() + 3);
			return evaluate(arguments, quantity, cgp_file);
		}
		else if (arguments.size() >= 1 && arguments[0] == "train")
		{
			if (arguments.size() < 2)
			{
				std::cerr << "missing argument for input quantity right after " + arguments[0] + "." << std::endl;
				return 12;
			}

			size_t quantity = cgp::parse_integer_argument(arguments[1]);

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
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return 42;
	}
}
