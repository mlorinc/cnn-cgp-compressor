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


#if defined(__SINGLE_MULTIPLEX)
#define evaluate_cgp(cgp_model, dataset) cgp_model->evaluate_single_multiplexed(dataset)
#elif defined(__MULTI_MULTIPLEX)
#define evaluate_cgp(cgp_model, dataset) cgp_model->evaluate_multi_multiplexed(dataset)
#else
#define evaluate_cgp(cgp_model, dataset) cgp_model->evaluate(dataset)
#endif

using namespace cgp;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

bool stop_flag = false;
int code = 0;

void signal_handler(int signum)
{
	if (signum == SIGTERM) {
		std::cerr << "received sigterm ... terminating" << std::endl;
		stop_flag = true;
		code = 33;
	}
	else {
		std::cerr << "received unknown signal: " << signum << std::endl;
	}
}

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
	cgp_model->calculate_energy_threshold();
	return cgp_model;
}

static std::string format_timestamp(double t) {
	return std::to_string(t);
}

static int evaluate_chromosome(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, std::string chromosome)
{
	auto chrom = std::make_shared<Chromosome>(*cgp_model, chromosome);
	CGPOutputStream weights(cgp_model, cgp_model->train_weights_file(), std::ios::trunc);
	weights.log_weights(chrom, get_dataset_input(dataset));
	chrom->tight_layout();
	auto solution = cgp_model->evaluate(dataset, chrom);
	CGPOutputStream out(cgp_model, cgp_model->output_file(), std::ios::trunc);
	out.log_csv(0, 0, "", solution, true);
	out.log_human(0, 0, solution);
	return 0;
}

static int evaluate_chromosomes(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, std::string gate_stats_file = "#")
{
	CGPInputStream in(cgp_model, cgp_model->cgp_statistics_file());
	size_t counter = 1;
	std::string chromosome;
	std::getline(in.get_stream(), chromosome);
	CGPOutputStream out(cgp_model, cgp_model->output_file(), std::ios::trunc);
	out.log_csv_header();
	while (!in.eof() && !in.fail())
	{
		std::unordered_map<std::string, std::string> template_args{ {"run", std::to_string(counter)} };

		auto chrom = std::make_shared<Chromosome>(*cgp_model, chromosome);
		auto solution = cgp_model->evaluate(dataset, chrom);
		out.log_csv(counter, counter, "", solution);
		counter++;

		CGPOutputStream weight_logger(cgp_model, cgp_model->train_weights_file(), std::ios::app, template_args);
		weight_logger.log_weights(chrom, get_dataset_input(dataset));

		CGPOutputStream gate_stats_logger(cgp_model, gate_stats_file, std::ios::trunc, template_args);
		gate_stats_logger.log_gate_statistics(chrom);
		std::getline(in.get_stream(), chromosome);
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
		CGPOutputStream weight_logger(cgp_model, cgp_model->train_weights_file(), std::ios::app, template_args);

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
				weight_logger.log_weights(chrom, get_dataset_input(dataset));
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
	const CGP::solution_t& solution = evaluate_cgp(cgp_model, dataset);
	const bool log_chromosome = cgp_model->mse_chromosome_logging_threshold() != CGPConfiguration::error_nan && CGP::get_error(solution) <= cgp_model->mse_chromosome_logging_threshold();

	if (cgp_model->get_generations_without_change() == 0)
	{
		logger.log_human(run, i, solution);
		stats_out.log_csv(run, i, format_timestamp(now - start_time), solution, log_chromosome);
		log_counter = 0;
	}
	else if (log_counter >= cgp_model->periodic_log_frequency())
	{
		logger.log_human(run, i, true);
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

static size_t train_run(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, int run, size_t generation)
{
	size_t patience = cgp_model->patience();
	if (patience == 0)
	{
		throw std::invalid_argument("patience cannot be 0");
	}

	generation = (generation == std::numeric_limits<size_t>::max()) ? (cgp_model->start_generation()) : (generation);
	auto mode = (generation == 0) ? (std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::app);
	CGPOutputStream logger(cgp_model, LOGGER_OUTPUT_FILE);
	CGPOutputStream event_logger(cgp_model, "-");

	double start_time = omp_get_wtime();
	std::unordered_map<std::string, std::string> template_args{ {"run",  std::to_string(run + 1)} };
	CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file(), mode, template_args);

	event_logger << "performin run " << run + 1 << " and starting from generation " << (generation + 1) << std::endl;
	for (size_t log_counter = cgp_model->periodic_log_frequency(); !stop_flag && generation < cgp_model->generation_count(); generation++)
	{
		auto now = omp_get_wtime();
		int continue_evolution = perform_evolution(start_time, now, cgp_model, dataset, logger, run, generation, stats_out, log_counter, patience);

		if (continue_evolution == 0 || cgp_model->get_generations_without_change() > patience)
		{
			return generation;
		}

		cgp_model->mutate(dataset);
	}

	event_logger << "finished evolution after " << (generation + 1) << " generations" << std::endl;
	stats_out.close();

	return generation;
}

static int train_generic(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, int run, bool remove_multiplex_on_patience, bool remove_multiplex_on_result, size_t generation = std::numeric_limits<size_t>::max())
{
	size_t start_run = (run == -1) ? (cgp_model->start_run()) : (run);
	size_t end_run = (run == -1) ? (cgp_model->number_of_runs()) : (run + 1);
	generation = (generation == std::numeric_limits<size_t>::max()) ? (cgp_model->start_generation()) : (generation);

	if (generation != 0 && !cgp_model->get_best_chromosome())
	{
		throw std::invalid_argument("cannot resume evolution without starting chromosome");
	}
	CGPOutputStream logger(cgp_model, LOGGER_OUTPUT_FILE);
	logger.dump();
	logger.close();

	double start_time = omp_get_wtime();
	for (size_t run = start_run; run < end_run; run++)
	{
		std::unordered_map<std::string, std::string> template_args{ {"run",  std::to_string(run + 1)} };
		cgp_model->reset();
		cgp_model->build_indices();
		cgp_model->generate_population(dataset);

		auto generation = train_run(cgp_model, dataset, run, 0);

		if (cgp_model->mse_threshold() < cgp_model->get_best_error_fitness())
		{
			cgp_model->set_generations_without_change(0);
			cgp_model->mse_threshold(cgp_model->get_best_error_fitness());
		}
		
		if (remove_multiplex_on_patience)
		{
			CGPOutputStream event_logger(cgp_model, "-");
			cgp_model->remove_multiplexing(dataset);
			event_logger << "early stopping patience multiplexing removed" << std::endl;
		}
		
		generation = train_run(cgp_model, dataset, run, generation + 1);

		auto time_taken = omp_get_wtime() - start_time;
		CGPOutputStream stats_out(cgp_model, cgp_model->cgp_statistics_file(), std::ios::app, template_args);
		stats_out.log_csv(run, generation, format_timestamp(time_taken), true);


		CGPOutputStream event_logger(cgp_model, "-");
		event_logger << "early stopping because of no change between " << cgp_model->patience() << " generations" << std::endl;
		if (remove_multiplex_on_result)
		{
			cgp_model->remove_multiplexing(dataset);
			event_logger << "early stopping final multiplexing removed" << std::endl;
		}

		logger.log_human(run, generation);
		stats_out.log_csv(run, generation, format_timestamp(time_taken), true);

		CGPOutputStream out(cgp_model, cgp_model->output_file(), std::ios::app, template_args);
		out.dump_all();
		out.close();

		CGPOutputStream weights_out(cgp_model, cgp_model->train_weights_file(), std::ios::app, template_args);
		weights_out.log_weights(get_dataset_input(dataset));
		weights_out.close();
	}
	return 0;
}

static int train_with_dynamic_threshold(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, int run, size_t generation = std::numeric_limits<size_t>::max())
{

	return train_generic(cgp_model, dataset, run, false, false, generation);
}

static int train_multi_multiplex(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, int run, size_t generation = std::numeric_limits<size_t>::max())
{

	return train_generic(cgp_model, dataset, run, true, false, generation);
}

static int train_single_multiplex(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, int run, size_t generation = std::numeric_limits<size_t>::max())
{

	return train_generic(cgp_model, dataset, run, false, true, generation);
}


static int train(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset, int run)
{
#if defined(__SINGLE_MULTIPLEX)
	return train_single_multiplex(cgp_model, dataset, run);
#elif defined(__MULTI_MULTIPLEX)
	return train_multi_multiplex(cgp_model, dataset, run);
#else
	return train_with_dynamic_threshold(cgp_model, dataset, run);
#endif
}

static int train(std::shared_ptr<CGP> cgp_model, const dataset_t& dataset)
{
	return train(cgp_model, dataset, -1);
}

int main(int argc, const char** args) {
	//assert(("should not abort", false));
	// Asserts floating point compatibility at compile time
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
	srand(time(NULL));
	std::vector<std::string> arguments(args + 1, args + argc);
	//std::vector<std::string> arguments
	//{
	//"train", "C:/Users/Majo/source/repos/TorchCompresser/local_experiments/layer_bypass/mse_0.0_350_100/train_cgp.config"
	//};

	//std::vector<std::string> arguments
	//{
	//"train", "C:/Users/Majo/source/repos/TorchCompresser/local_experiments/worst_case/mse_0_256_20/train_cgp.config"
	//};

	//std::vector<std::string> arguments
	//{
	//"train", "C:/Users/Majo/source/repos/TorchCompresser/local_experiments/all_layers/mse_0.0_256_10/train_cgp.config"
	//};

	//std::vector<std::string> arguments
	//{ "evaluate:chromosomes", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\data_store\\all_layers\\mse_0_50_10\\train_cgp.config", "C:\\Users\\Majo\\source\\repos\\TorchCompresser\\data_store\\all_layers\\mse_0_50_10\\gate_statistics\\statistics.1.{run}.txt", "--mse-threshold", "0", "--row-count", "50", "--col-count", "10", "--look-back-parameter", "10", "--input-file", "c:/users/majo/source/repos/torchcompresser/data_store/all_layers/mse_0_50_10/train.data", "--cgp-statistics-file", "c:/users/majo/source/repos/torchcompresser/data_store/all_layers/mse_0_50_10/chromosomes.txt", "--output-file", "c:/users/majo/source/repos/torchcompresser/data_store/all_layers/mse_0_50_10/evaluate_statistics/statistics.1.csv", "--train-weights-file", "c:/users/majo/source/repos/torchcompresser/data_store/all_layers/mse_0_50_10/all_weights/weights.1.{run}.txt", "--gate-parameters-file", "c:/users/majo/source/repos/torchcompresser/data_store/all_layers/mse_0_50_10/gate_parameters.txt" }
	//;

	std::cout << "... INIT CONFIGURATION CHECK ..." << std::endl;

#if defined __DATE__ && defined __TIME__
	std::cout << "starting cgp optimiser with compiled " << __DATE__ << " at " << __TIME__ << std::endl;
#endif // _COMPILE_TIME

#ifdef __SINGLE_MULTIPLEX
	std::cout << "SINGLE_MULTIPLEX: on" << std::endl;
#else
	std::cout << "SINGLE_MULTIPLEX: off" << std::endl;
#endif

#ifdef __SINGLE_OUTPUT_ARITY 
	std::cout << "SINGLE_OUTPUT_ARITY: on" << std::endl;
#else
	std::cout << "SINGLE_OUTPUT_ARITY: off" << std::endl;
#endif

#ifdef __NO_POW_SOLUTIONS 
	std::cout << "NO_POW_SOLUTIONS: on" << std::endl;
#else
	std::cout << "NO_POW_SOLUTIONS: off" << std::endl;
#endif

#ifdef __NO_DIRECT_SOLUTIONS 
	std::cout << "NO_DIRECT_SOLUTIONS: on" << std::endl;
#else
	std::cout << "NO_DIRECT_SOLUTIONS: off" << std::endl;
#endif

#ifdef _DEPTH_ENABLED 
	std::cout << "DEPTH_ENABLED: on" << std::endl;
#else
	std::cout << "DEPTH_ENABLED: off" << std::endl;
#endif

#ifdef _DISABLE_ROW_COL_STATS 
	std::cout << "DISABLE_ROW_COL_STATS: on" << std::endl;
#else
	std::cout << "DISABLE_ROW_COL_STATS: off" << std::endl;
#endif

#ifdef __MULTI_MULTIPLEX 
	std::cout << "MULTI_MULTIPLEX: on" << std::endl;
#else
	std::cout << "MULTI_MULTIPLEX: off" << std::endl;
#endif

#ifdef NDEBUG 
	std::cout << "NDEBUG: on" << std::endl;
#else
	std::cout << "NDEBUG: off" << std::endl;
#endif

#ifdef __ABSOLUTE_ERROR_METRIC 
	std::cout << "ABSOLUTE_ERROR_METRIC: on" << std::endl;
#else
	std::cout << "ABSOLUTE_ERROR_METRIC: off" << std::endl;
#endif

#ifdef __MEAN_SQUARED_ERROR_METRIC 
	std::cout << "MEAN_SQUARED_ERROR_METRIC: on" << std::endl;
#else
	std::cout << "MEAN_SQUARED_ERROR_METRIC: off" << std::endl;
#endif

#if !defined(__MEAN_SQUARED_ERROR_METRIC) && !defined(__ABSOLUTE_ERROR_METRIC) 
	std::cout << "SQUARED_ERROR_METRIC: on" << std::endl;
#else
	std::cout << "SQUARED_ERROR_METRIC: off" << std::endl;
#endif

#ifdef __VIRTUAL_SELECTOR 
	std::cout << "VIRTUAL_SELECTOR: on" << std::endl;
#else
	std::cout << "VIRTUAL_SELECTOR: off" << std::endl;
#endif

#ifdef __OLD_NOOP_OPERATION_SUPPORT 
	std::cout << "OLD_NOOP_OPERATION_SUPPORT: on" << std::endl;
#else
	std::cout << "OLD_NOOP_OPERATION_SUPPORT: off" << std::endl;
#endif

	std::cout << "... CONFIGURATION CHECK DONE ..." << std::endl;

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
	try
	{
		int consumed_arguments = 2;
		std::string command = arguments[0];
		std::string cgp_file = arguments[1];
		std::string chromosome, gate_statistics_file = "#";

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

		if (command == "evaluate:chromosomes")
		{
			if (arguments.size() < 3)
			{
				std::cerr << "missing gate statistics file" << std::endl;
				return 14;
			}
			gate_statistics_file = arguments[2];
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
			code = evaluate(cgp_model, train_dataset, [cgp_model](const CGPCSVRow& row) -> bool { return row.error <= cgp_model->mse_chromosome_logging_threshold(); });
		}
		else if (command == "evaluate:chromosomes")
		{
			code = evaluate_chromosomes(cgp_model, train_dataset, gate_statistics_file);
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
	OutputStream event_logger("-");
	event_logger << std::endl << "exitting program with code " << code << std::endl;
	return code;
}
