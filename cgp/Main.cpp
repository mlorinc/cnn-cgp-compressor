// Cgp.cpp : Defines the entry point for the application.

#include "Main.h"

#ifdef __DISABLE_COUT
constexpr auto LOGGER_OUTPUT_FILE = ("#");
#else
constexpr auto LOGGER_OUTPUT_FILE = ("-");
#endif

#ifdef _MEASURE_LEARNING_RATE
#error "Critical error when compiling CGP. Learning should be disabled."
#endif // _MEASURE_LEARNING_RATE


using namespace cgp;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

static dataset_t init_dataset(const std::string& cgp_file, const std::vector<std::string>& arguments)
{
	std::ifstream config_in(cgp_file);
	auto config = std::make_shared<CGPConfiguration>();

	if (!config_in.is_open())
	{
		throw std::ios_base::failure("could not open file: " + cgp_file);
	}

	config->load(config_in);
	config_in.close();
	config->set_from_arguments(arguments);

	CGPInputStream in(config, config->input_file());
	dataset_t dataset = in.load_train_data();
	in.close();
	return dataset;
}

static std::shared_ptr<CGP> init_cgp(const std::string& cgp_file, const std::vector<std::string>& arguments, const dataset_t& dataset)
{
	std::ifstream cgp_in(cgp_file);
	if (!cgp_in.is_open())
	{
		throw std::ios_base::failure("could not open file: " + cgp_file);
	}

	auto cgp_model = std::make_shared<CGP>(cgp_in, arguments, dataset);
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

static int evaluate_chromosome(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, std::string chromosome)
{
	auto chrom = std::make_shared<Chromosome>(*cgp_model, chromosome);
	auto solution = cgp_model->evaluate(dataset, chrom);
	CGPOutputStream out(cgp_model, cgp_model->output_file(), std::ios::trunc);
	out.log_csv(0, 0, "");
	return 0;
}

static int evaluate_chromosomes(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset)
{
	std::unordered_map<std::string, std::string> template_args{};
	CGPInputStream in(cgp_model, cgp_model->cgp_statistics_file());
	size_t counter = 1;
	while (!in.eof() && !in.fail())
	{
		std::string chromosome;
		std::getline(in.get_stream(), chromosome);

		template_args["run"] = std::to_string(counter);
		auto chrom = std::make_shared<Chromosome>(*cgp_model, chromosome);
		auto solution = cgp_model->evaluate(dataset, chrom);

		CGPOutputStream out(cgp_model, cgp_model->output_file(), std::ios::app, template_args);
		out.log_csv(counter, counter, "");
		counter++;
	}

	return 0;
}

static int evaluate(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, std::function<bool(const CGPCSVRow&)> predicate = nullptr)
{
	for (int i = 0; i < cgp_model->number_of_runs(); i++)
	{
		std::unordered_map<std::string, std::string> template_args{ {"run", std::to_string(i + 1)} };
		CGPInputStream in(cgp_model, cgp_model->cgp_statistics_file(), template_args);
		CGPOutputStream out(cgp_model, cgp_model->output_file(), std::ios::app, template_args);

		while (!in.eof() && !in.fail())
		{
			auto row = in.read_csv_line();

			if (!row.ok)
			{
				out << row.raw_line;
				continue;
			}

			if (predicate == nullptr || predicate(row))
			{
				auto chrom = std::make_shared<Chromosome>(*cgp_model, row.chromosome);
				auto solution = cgp_model->evaluate(dataset, chrom);

				out.log_csv(
					row.run,
					row.generation,
					row.timestamp,
					solution,
					true);
			}
		}
	}

	return 0;
}

template<typename T>
bool early_stop_check(const T value, T early_stop_value, T nan)
{
	return early_stop_value == nan || value <= early_stop_value;
}

static int perform_evolution(const double start_time, const double now, std::shared_ptr<cgp::CGP>& cgp_model, const cgp::dataset_t& dataset, cgp::CGPOutputStream& logger, const size_t run, size_t i, cgp::CGPOutputStream& stats_out, size_t& log_counter, size_t generation_stop)
{
	const CGP::solution_t& solution = cgp_model->evaluate(dataset);
	const bool log_chromosome = CGP::get_error(solution) <= cgp_model->mse_chromosome_logging_threshold();

	if (cgp_model->get_generations_without_change() == 0)
	{
		logger.log_human(run, i, solution);
		stats_out.log_csv(run, i, format_timestamp(now - start_time), solution, log_chromosome);
		log_counter = 0;
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
	if ((early_stop_check(CGP::get_error(solution), cgp_model->mse_early_stop(), cgp_model->error_nan) &&
		early_stop_check(CGP::get_energy(solution), cgp_model->energy_early_stop(), cgp_model->energy_nan) &&
		early_stop_check(CGP::get_delay(solution), cgp_model->delay_early_stop(), cgp_model->delay_nan) &&
		early_stop_check(CGP::get_depth(solution), cgp_model->depth_early_stop(), cgp_model->depth_nan) &&
		early_stop_check(CGP::get_gate_count(solution), cgp_model->gate_count_early_stop(), cgp_model->gate_count_nan)))
	{
		return 0;
	}

	return 1;
}

static int train(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, int run)
{
#ifdef _MEASURE_LEARNING_RATE
	const double LR = cgp_model->learning_rate();
	const double LR_PERIOD = 0.001;
#endif
	size_t patience = cgp_model->patience();
	if (patience == 0)
	{
		throw std::invalid_argument("patience cannot be 0");
	}

	size_t start_run = (run == -1) ? (cgp_model->start_run()) : (run);
	size_t generation = cgp_model->start_generation();
	int end_run = (run == -1) ? (cgp_model->number_of_runs()) : (run + 1);
	auto mode = (generation == 0) ? (std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::app);
	CGPOutputStream logger(cgp_model, LOGGER_OUTPUT_FILE);
	CGPOutputStream event_logger(cgp_model, "-");

	cgp_model->build_indices();
	logger.dump();
	cgp_model->generate_population();
	for (size_t run = start_run; run < end_run; run++)
	{
		double start_time = omp_get_wtime();
		std::unordered_map<std::string, std::string> template_args{ {"run",  std::to_string(run + 1)} };
		CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file(), mode, template_args);

#ifdef _MEASURE_LEARNING_RATE
		OutputStream lr_stream(cgp_model->learning_rate_file(), mode, template_args);
		OutputStream lr_period_stream(cgp_model->learning_rate_file() + ".period", mode, template_args);
		if (cgp_model->start_generation() == 0)
		{
			stats_out.log_csv_header();
			lr_stream << "lr_error,lr_energy,lr_delay,lr_gate_count" << std::endl;
			lr_period_stream << "lr_error,lr_energy,lr_delay,lr_gate_count" << std::endl;
		}
		Learning lr(LR, patience, cgp_model->mse_threshold(), 5);
#else
		if (cgp_model->start_generation() == 0)
		{
			stats_out.log_csv_header();
		}
#endif

		event_logger << "performin run " << run + 1 << " and starting from generation " << (generation + 1) << std::endl;
		for (size_t log_counter = cgp_model->periodic_log_frequency(); generation < cgp_model->generation_count(); generation++)
		{
			auto now = omp_get_wtime();
			int continue_evolution = perform_evolution(start_time, now, cgp_model, dataset, logger, run, generation, stats_out, log_counter, patience);

			if (continue_evolution == 0)
			{
				event_logger << "early stopping because target fitness values were satisfied" << std::endl;
				logger.log_human(run, generation);
				stats_out.log_csv(run, generation, format_timestamp(now - start_time), true);
				break;
			}

#ifdef _MEASURE_LEARNING_RATE
			const CGP::solution_t best_solution = cgp_model->get_best_solution();
			if (lr.tick(best_solution))
			{
				const LearningRates lr_rates = lr.get_average_learning_rates();
				lr_stream << lr_rates.error << "," << lr_rates.energy << "," << lr_rates.delay << "," << lr_rates.gate_count << std::endl;
				logger << "[GLOBAL] LR: " << LR << ", LR Error: " << lr_rates.error << ", LR Energy: " << lr_rates.energy << ", LR Delay: " << lr_rates.delay << ", LR Count: " << lr_rates.gate_count << std::endl;
			}

			if (lr.finished_period())
			{
				const LearningRates lr_rates = lr.get_average_period_learning_rates();
				lr_period_stream << lr_rates.error << "," << lr_rates.energy << "," << lr_rates.delay << "," << lr_rates.gate_count << std::endl;
				logger << "[AVERAGE PERIOD] LR: " << LR << ", LR Error: " << lr_rates.error << ", LR Energy: " << lr_rates.energy << ", LR Delay: " << lr_rates.delay << ", LR Count: " << lr_rates.gate_count << std::endl;

				if (lr.is_stagnated() && lr.is_periodicaly_stagnated(LR_PERIOD))
				{
					event_logger << "no significant change made ... early stopping" << std::endl;
					logger.log_human(run, generation);
					stats_out.log_csv(run, generation, format_timestamp(now - start_time), true);
					break;
				}
			}
#else
			if (cgp_model->get_generations_without_change() > patience)
			{
				event_logger << "early stopping because of no change between " << patience << " generations" << std::endl;
				logger.log_human(run, generation);
				stats_out.log_csv(run, generation, format_timestamp(now - start_time), true);
				break;
			}
#endif
			cgp_model->mutate();
		}
		event_logger << "finished evolution after " << (generation + 1) << " generations" << std::endl;
		stats_out.close();


		CGPOutputStream out(cgp_model, cgp_model->output_file(), template_args);
		out.dump_all();
		out.close();

		CGPOutputStream weights_out(cgp_model, cgp_model->train_weights_file(), template_args);
		weights_out.log_weights(get_dataset_input(dataset));
		weights_out.close();

		cgp_model->reset();
		cgp_model->generate_population();
		generation = 0;
		mode = std::ios::out | std::ios::app;
	}

	event_logger << std::endl << "exitting program with code 0" << std::endl;
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
#if defined __DATE__ && defined __TIME__
	std::cout << "starting cgp optimiser with compiled " << __DATE__ << " at " << __TIME__ << std::endl;
#endif // _COMPILE_TIME


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
		int consumed_arguments = 2;
		std::string command = arguments[0];
		std::string cgp_file = arguments[1];
		std::string chromosome;

		if (command == "evaluate:chromosome")
		{
			if (arguments.size() < 3)
			{
				std::cerr << "missing chromosome" << std::endl;
				return 14;
			}
			chromosome = arguments[2];
			consumed_arguments++;
		}

		arguments.erase(arguments.begin(), arguments.begin() + consumed_arguments);
		auto train_dataset = init_dataset(cgp_file, arguments);
		auto cgp_model = init_cgp(cgp_file, arguments, train_dataset);

		// Read two values from the standard input
		if (cgp_model->input_count() == 0 || cgp_model->output_count() == 0)
		{
			std::cerr << "invalid input size and output size" << std::endl;
			return 2;
		}

		if (command == "evaluate:chromosome")
		{
			code = evaluate_chromosome(cgp_model, train_dataset, chromosome);
		}
		else if (command == "evaluate:all")
		{
			code = evaluate(cgp_model, train_dataset);
		}
		else if (command == "evaluate:chrmosomes")
		{
			code = evaluate_chromosomes(cgp_model, train_dataset);
		}
		else if (command == "train")
		{
			code = train(cgp_model, train_dataset);
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
