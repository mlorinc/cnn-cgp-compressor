// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"
using namespace cgp;

static std::shared_ptr<CGP> init_cgp(const std::string& cgp_file, const std::vector<std::string>& arguments)
{
	std::ifstream cgp_in(cgp_file);
	auto cgp_model = std::make_shared<CGP>(cgp_in, arguments);
	cgp_in.close();

	if (cgp_model->gate_parameters_input_file().empty())
	{
		throw std::invalid_argument("missing " + CGPConfiguration::GATE_PARAMETERS_FILE_LONG + " in file or as CLI argument");
	}

	CGPInputStream gate_parameter_loader(cgp_model, cgp_model->gate_parameters_input_file());
	gate_parameter_loader.load_gate_parameters();
	return cgp_model;
}

static std::string format_timestamp(std::chrono::milliseconds ms) {
	auto secs = std::chrono::duration_cast<std::chrono::seconds>(ms);

	std::stringstream ss;
	ss << ms.count();
	return ss.str();
}

static int evaluate(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, size_t run, size_t generation, std::string solution = "")
{
	CGPOutputStream out(cgp_model, cgp_model->output_file(), {{"run", std::to_string(run+1)}, {"generation", std::to_string(generation+1)}});
	CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file(), { {"run", std::to_string(run+1)}, {"generation", std::to_string(generation+1)} });
	CGPOutputStream logger(cgp_model, "-");
	if (!solution.empty())
	{
		cgp_model->restore(solution);
	}

	logger.dump();
	out.log_weights(get_dataset_input(dataset));
	stats_out.log_csv(run, generation, cgp_model->get_best_chromosome(), get_dataset_input(dataset), get_dataset_output(dataset));
	return 0;
}

static int evaluate(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset)
{
	CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file());
	CGPOutputStream logger(cgp_model, "-");
	for (size_t i = cgp_model->start_run(); i < cgp_model->number_of_runs(); i++)
	{
		CGPOutputStream out(cgp_model, cgp_model->output_file(), { {"run", std::to_string(i+1)}});
		logger.dump();
		out.log_weights(get_dataset_input(dataset));
		stats_out.log_csv(i, cgp_model->get_evolution_steps_made(), cgp_model->get_best_chromosome(), get_dataset_input(dataset), get_dataset_output(dataset));
	}

	return 0;
}

static int train(std::shared_ptr<CGP> cgp_model, const dataset_t &dataset)
{
	bool new_evolution = cgp_model->start_generation() == 0 && cgp_model->start_run() == 0;
	auto mode = (new_evolution) ? (std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::app);
	CGPOutputStream logger(cgp_model, "-");

	cgp_model->build_indices();
	logger.dump();
	auto start = std::chrono::high_resolution_clock::now();
	cgp_model->generate_population();
	auto generation_stop = cgp_model->patience();
	for (size_t run = cgp_model->start_run(); run < cgp_model->number_of_runs(); run++)
	{
		CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file(), mode, { {"run", std::to_string(run+1)} });
		size_t i = (run != cgp_model->start_run()) ? (0) : (cgp_model->start_generation());
		std::cout << "performin run " << run+1 << " and starting from generation " << i << std::endl;
		for (size_t log_counter = cgp_model->periodic_log_frequency(); i < cgp_model->generation_count(); i++)
		{
			cgp_model->evaluate(get_dataset_input(dataset), get_dataset_output(dataset));
			if (cgp_model->get_generations_without_change() == 0)
			{
				auto now = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
				stats_out.log_csv(run, i, format_timestamp(duration));
				logger.log_human(run, i);
				log_counter = 0;
				start = std::chrono::high_resolution_clock::now();
			}
			else if (log_counter >= cgp_model->periodic_log_frequency())
			{
				logger.log_human(run, i);
				log_counter = 0;
			}
			else
			{
				log_counter++;
			}

			if ((cgp_model->get_best_error_fitness() <= cgp_model->mse_early_stop() && cgp_model->get_best_energy_fitness() <= cgp_model->energy_early_stop()) || cgp_model->get_generations_without_change() > generation_stop)
			{
				logger.log_human(run, i);
				break;
			}
			cgp_model->mutate();
		}

		std::cout << "chromosome size: " << cgp_model->get_serialized_chromosome_size() << std::endl
			<< "finished evolution after " << i << " generations" << std::endl;

		CGPOutputStream out(cgp_model, cgp_model->output_file(), { {"run", std::to_string(run+1)}});
		out.dump_all();
		std::cout << "resetting cgp" << std::endl;
		cgp_model->reset();
		std::cout << "resetted cgp" << std::endl << "generating population" << std::endl;
		cgp_model->generate_population();
		std::cout << "generated population" << std::endl;
	}

	std::cerr << std::endl << "exitting program" << std::endl;
	return 0;
}

int main(int argc, const char** args) {
	// Asserts floating point compatibility at compile time
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
	std::vector<std::string> arguments(args + 1, args + argc);

	if (arguments.size() < 1)
	{
		std::cerr << "missing command: evaluate | train" << std::endl;
		return 12;
	}

	if (arguments.size() < 2)
	{
		std::cerr << "missing cgp configuration file" << std::endl;
		return 13;
	}

	std::string command = arguments[0];
	std::string cgp_file = arguments[1];
	arguments.erase(arguments.begin(), arguments.begin() + 2);
	auto cgp_model_pointer = init_cgp(cgp_file, arguments);

	// Read two values from the standard input
	if (cgp_model_pointer->input_count() == 0 || cgp_model_pointer->output_count() == 0)
	{
		std::cerr << "invalid input size and output size" << std::endl;
		return 2;
	}
	
	CGPInputStream in(cgp_model_pointer, cgp_model_pointer->input_file());
	auto train_dataset = in.load_train_data();
	in.close();

	try
	{
		if (command == "evaluate")
		{
			return evaluate(cgp_model_pointer, train_dataset);
		}
		else if (command == "train")
		{
			return train(cgp_model_pointer, train_dataset);
		}
		else
		{
			std::cerr << "unknown command: " + command << std::endl;
			return 13;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 42;
	}
}
