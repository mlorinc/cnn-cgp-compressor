// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"
#include "StringTemplate.h"
using namespace cgp;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

static std::shared_ptr<CGP> init_cgp(const std::string& cgp_file, const std::vector<std::string>& arguments)
{
	std::ifstream cgp_in(cgp_file);

	if (!cgp_in.is_open())
	{
		throw std::ios_base::failure("could not open file: " + cgp_file);
	}

	auto cgp_model = std::make_shared<CGP>(cgp_in, arguments);
	cgp_in.close();

	if (cgp_model->gate_parameters_input_file().empty())
	{
		throw std::invalid_argument("missing " + CGPConfiguration::GATE_PARAMETERS_FILE_LONG + " in file or as CLI argument");
	}

	CGPInputStream gate_parameter_loader(cgp_model, cgp_model->gate_parameters_input_file());
	auto parameters = gate_parameter_loader.load_gate_parameters();
	cgp_model->function_costs(std::move(parameters));
	return cgp_model;
}

static std::string format_timestamp(double t) {
	return std::to_string(t);
}

static int evaluate(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, size_t run, size_t generation, std::string solution = "")
{
	CGPOutputStream out(cgp_model, cgp_model->output_file(), { {"run", std::to_string(run + 1)}, {"generation", std::to_string(generation + 1)} });
	CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file(), { {"run", std::to_string(run + 1)}, {"generation", std::to_string(generation + 1)} });
	CGPOutputStream logger(cgp_model, "-");
	if (!solution.empty())
	{
		cgp_model->restore(solution);
	}

	logger.dump();
	out.log_weights(get_dataset_input(dataset));
	stats_out.log_csv(run, generation, cgp_model->get_best_chromosome(), dataset);
	return 0;
}

static int evaluate(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset)
{
	CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file(), std::ios::app);
	CGPOutputStream logger(cgp_model, "-");
	for (size_t i = cgp_model->start_run(); i < cgp_model->number_of_runs(); i++)
	{
		CGPOutputStream out(cgp_model, cgp_model->output_file(), { {"run", std::to_string(i + 1)} });
		logger.dump();
		out.log_weights(get_dataset_input(dataset));
		stats_out.log_csv(i, cgp_model->get_evolution_steps_made(), cgp_model->get_best_chromosome(), dataset);
	}

	return 0;
}

static int perform_evolution(const double start_time, std::shared_ptr<cgp::CGP>& cgp_model, const cgp::dataset_t& dataset, cgp::CGPOutputStream& logger, const int& run, int i, cgp::CGPOutputStream& stats_out, int& log_counter, const int& generation_stop)
{
	auto now = omp_get_wtime();
	const CGP::solution_t& solution = cgp_model->evaluate(dataset);
	if (cgp_model->get_generations_without_change() == 0)
	{
		logger.log_human(run, i, solution);
		stats_out.log_csv(run, i, format_timestamp(now - start_time), solution, true);
		log_counter = 0;
	}
	else if (log_counter >= cgp_model->periodic_log_frequency())
	{
		logger.log_human(run, i);
		stats_out.log_csv(run, i, format_timestamp(now - start_time), true);
		log_counter = 0;
	}
	else
	{
		log_counter++;
	}

	if (
		(CGP::get_error(solution) <= cgp_model->mse_early_stop() &&
			CGP::get_energy(solution) <= cgp_model->energy_early_stop() &&
			CGP::get_delay(solution) == 0 &&
			CGP::get_depth(solution) <= 1 &&
			CGP::get_gate_count(solution) <= 1) ||
		(cgp_model->get_generations_without_change() > generation_stop))
	{
		logger.log_human(run, i);
		stats_out.log_csv(run, i, format_timestamp(now - start_time), true);
		return 0;
	}
	cgp_model->mutate();
	return 1;
}

static int train(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, int run)
{
	int i = (cgp_model->start_run() == 0) ? (0) : (cgp_model->start_generation());
	int generation_stop = cgp_model->patience();
	int start_run = (run == -1) ? (cgp_model->start_run()) : (run);
	int end_run = (run == -1) ? (cgp_model->number_of_runs()) : (run + 1);
	CGPOutputStream logger(cgp_model, "-");

	cgp_model->build_indices();
	logger.dump();
	cgp_model->generate_population();
	for (int run = start_run; run < end_run; run++)
	{
		double start_time = omp_get_wtime();
		auto mode = (run == start_run && cgp_model->start_generation() == 0) ? (std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::app);
		CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file(), mode, { {"run", std::to_string(run + 1)} });
		std::cout << "performin run " << run + 1 << " and starting from generation " << (i+1) << std::endl;
		for (int log_counter = cgp_model->periodic_log_frequency(), continue_evolution = 1; continue_evolution && i < cgp_model->generation_count(); i++)
		{
			continue_evolution = perform_evolution(start_time, cgp_model, dataset, logger, run, i, stats_out, log_counter, generation_stop);
		}

		std::cout << "finished evolution after " << (i + 1) << " generations" << std::endl;

		CGPOutputStream out(cgp_model, cgp_model->output_file(), { {"run", std::to_string(run + 1)} });
		out.dump_all();
		cgp_model->reset();
		cgp_model->generate_population();
		i = 0;
	}

	std::cerr << std::endl << "exitting program" << std::endl;
	return 0;
}

static int train(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset)
{
	return train(cgp_model, dataset, -1);
}

int main(int argc, const char** args) {
	// Asserts floating point compatibility at compile time
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
	std::vector<std::string> arguments(args + 1, args + argc);

	if (arguments.size() == 0)
	{
		std::cerr << "missing command: evaluate | train" << std::endl;
		return 12;
	}

	if (arguments.size() < 2)
	{
		std::cerr << "missing cgp configuration file" << std::endl;
		return 13;
	}

	dataset_t train_dataset;
	int code = 0;
	try
	{
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
		train_dataset = in.load_train_data();
		in.close();

		if (command == "evaluate")
		{
			code = evaluate(cgp_model_pointer, train_dataset);
		}
		else if (command == "train")
		{
			code = train(cgp_model_pointer, train_dataset);
		}
		else
		{
			std::cerr << "unknown command: " + command << std::endl;
			code = 13;
		}
	}
	catch (const StringTemplateError& e)
	{
		std::cerr << e.get_message() << std::endl;
		code = 42;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		code = 43;
	}
	delete_dataset(train_dataset);
	return code;
}
