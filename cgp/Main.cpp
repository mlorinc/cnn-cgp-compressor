// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"
using namespace cgp;

std::shared_ptr<CGP> init_cgp(const std::string& cgp_file)
{
	std::ifstream cgp_in(cgp_file);
	auto cgp_model = std::make_shared<CGP>(cgp_in);
	cgp_in.close();

	std::shared_ptr<double[]> energy_costs = std::make_shared<double[]>(cgp_model->function_count());

	for (size_t i = 0; i < cgp_model->function_count(); i++)
	{
		energy_costs[i] = 1;
	}

	cgp_model->function_energy_costs(energy_costs);
	return cgp_model;
}

static std::string format_timestamp(std::chrono::milliseconds ms) {
	auto secs = std::chrono::duration_cast<std::chrono::seconds>(ms);
	ms -= std::chrono::duration_cast<std::chrono::milliseconds>(secs);
	auto mins = std::chrono::duration_cast<std::chrono::minutes>(secs);
	secs -= std::chrono::duration_cast<std::chrono::seconds>(mins);
	auto hour = std::chrono::duration_cast<std::chrono::hours>(mins);
	mins -= std::chrono::duration_cast<std::chrono::minutes>(hour);

	std::stringstream ss;
	ss << std::setw(2) << std::setfill('0') << hour.count() << ":" << mins.count() << ":" << secs.count() << "." << ms.count();
	return ss.str();
}

int evaluate(std::vector<std::string>& arguments, const std::string& cgp_file, const std::string& solution = "")
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

	weight_repr_value_t min, max;
	for (size_t i = 0; i < cgp_model.dataset_size(); i++)
	{
		std::cerr << "loading values" << std::endl;
		inputs.push_back(load_input(*in, cgp_model.input_count()));
		// Ignore output
		load_output(*in, cgp_model.output_count(), min, max);
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

int train(std::vector<std::string>& arguments, const std::string& cgp_file)
{
	std::vector<std::shared_ptr<weight_value_t[]>> inputs, outputs;
	weight_repr_value_t min = std::numeric_limits<weight_repr_value_t>::max();
	weight_repr_value_t max = std::numeric_limits<weight_repr_value_t>::min();

	auto cgp_model_pointer = init_cgp(cgp_file);
	CGP& cgp_model = *cgp_model_pointer;
	cgp_model.set_from_arguments(arguments);

	auto in = get_input(cgp_model.input_file());
	auto stats_out = get_output(
		cgp_model.cgp_statistics_file(),
		nullptr, 
		(cgp_model.start_generation() == 0 && cgp_model.start_run() == 0) ?
		(std::ios::out) : (std::ios::out | std::ios::trunc)
		);

	// Read two values from the standard input
	if (cgp_model.input_count() == 0 || cgp_model.output_count() == 0)
	{
		std::cerr << "invalid input size and output size" << std::endl;
		return 2;
	}

	if ((cgp_model.start_generation() != 0 || cgp_model.start_run() != 0) && !cgp_model.get_best_chromosome())
	{
		std::cerr << "cannot resume evolution without starting chromosome" << std::endl;
		return 3;
	}

	for (size_t i = 0; i < cgp_model.dataset_size(); i++)
	{
		std::cerr << "loading values" << std::endl;
		inputs.push_back(load_input(*in, cgp_model.input_count()));
		outputs.push_back(load_output(*in, cgp_model.output_count(), min, max));
	}


	auto generation_stop = 125000;
	cgp_model.build_indices();
	cgp_model.dump(std::cerr);
	std::cerr << "invalid_value: " << CGPConfiguration::invalid_value << std::endl
		<< "no_care_value: " << CGPConfiguration::no_care_value << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	cgp_model.generate_population();

	for (size_t run = cgp_model.start_run(); run < cgp_model.number_of_runs(); run++)
	{
		for (size_t i = (run != cgp_model.start_run()) ? (0) : (cgp_model.start_generation()), log_counter = cgp_model.periodic_log_frequency(); i < cgp_model.generation_count(); i++)
		{
			cgp_model.evaluate(inputs, outputs);
			if (cgp_model.get_generations_without_change() == 0)
			{
				auto now = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
				log_csv(*stats_out, run, i, cgp_model, format_timestamp(duration));
				log_human(std::cerr, run, i, cgp_model);
				log_counter = 0;
				start = std::chrono::high_resolution_clock::now();
			}
			else if (log_counter == cgp_model.periodic_log_frequency())
			{
				log_human(std::cerr, run, i, cgp_model);
				log_counter = 0;
			}
			else
			{
				log_counter++;
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

	try
	{
		if (arguments.size() >= 1 && arguments[0] == "evaluate")
		{
			if (arguments.size() < 2)
			{
				std::cerr << "missing argument for cgp configuraiton file" << std::endl;
				return 12;
			}
			std::string cgp_file = arguments[1];

			arguments.erase(arguments.begin(), arguments.begin() + 2);
			return evaluate(arguments, cgp_file);
		}
		else if (arguments.size() >= 1 && arguments[0] == "evaluate:inline")
		{
			if (arguments.size() < 2)
			{
				std::cerr << "missing argument for cgp configuraiton file" << std::endl;
				return 12;
			}
			std::string cgp_file = arguments[1];

			if (arguments.size() < 3)
			{
				std::cerr << "missing argument for serialized chromosome solution" << std::endl;
				return 13;
			}

			std::string solution = arguments[2];

			if (solution.empty())
			{
				std::cerr << "solution must not be empty string" << std::endl;
				return 14;
			}

			arguments.erase(arguments.begin(), arguments.begin() + 3);
			return evaluate(arguments, cgp_file, solution);
		}
		else if (arguments.size() >= 1 && arguments[0] == "train")
		{
			if (arguments.size() < 2)
			{
				std::cerr << "missing argument for cgp configuraiton file" << std::endl;
				return 12;
			}
			std::string cgp_file = arguments[1];

			arguments.erase(arguments.begin(), arguments.begin() + 2);
			return train(arguments, cgp_file);
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
