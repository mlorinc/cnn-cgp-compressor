#include "Configuration.h"
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace cgp {

	const std::string CGPConfiguration::error_nan_string = "inf";
	const std::string CGPConfiguration::quantized_energy_nan_string = "inf";
	const std::string CGPConfiguration::energy_nan_string = "inf";
	const std::string CGPConfiguration::quantized_delay_nan_string = "inf";
	const std::string CGPConfiguration::delay_nan_string = "inf";
	const std::string CGPConfiguration::depth_nan_string = "nan";
	const std::string CGPConfiguration::gate_count_nan_string = "nan";
	const std::string CGPConfiguration::area_nan_string = "inf";

	const std::string CGPConfiguration::PERIODIC_LOG_FREQUENCY_LONG = "--periodic-log-frequency";

	const std::string CGPConfiguration::FUNCTION_INPUT_ARITY_LONG = "--function-input-arity";
	const std::string CGPConfiguration::FUNCTION_INPUT_ARITY_SHORT = "-fi";

	const std::string CGPConfiguration::FUNCTION_OUTPUT_ARITY_LONG = "--function-output-arity";
	const std::string CGPConfiguration::FUNCTION_OUTPUT_ARITY_SHORT = "-fo";

	const std::string CGPConfiguration::OUTPUT_COUNT_LONG = "--output-count";
	const std::string CGPConfiguration::OUTPUT_COUNT_SHORT = "-oc";

	const std::string CGPConfiguration::INPUT_COUNT_LONG = "--input-count";
	const std::string CGPConfiguration::INPUT_COUNT_SHORT = "-ic";

	const std::string CGPConfiguration::CGPConfiguration::POPULATION_MAX_LONG = "--population-max";
	const std::string CGPConfiguration::CGPConfiguration::POPULATION_MAX_SHORT = "-p";

	const std::string CGPConfiguration::MUTATION_MAX_LONG = "--mutation-max";
	const std::string CGPConfiguration::MUTATION_MAX_SHORT = "-m";

	const std::string CGPConfiguration::ROW_COUNT_LONG = "--row-count";
	const std::string CGPConfiguration::ROW_COUNT_SHORT = "-r";

	const std::string CGPConfiguration::COL_COUNT_LONG = "--col-count";
	const std::string CGPConfiguration::COL_COUNT_SHORT = "-c";

	const std::string CGPConfiguration::LOOK_BACK_PARAMETER_LONG = "--look-back-parameter";
	const std::string CGPConfiguration::LOOK_BACK_PARAMETER_SHORT = "-l";

	const std::string CGPConfiguration::GENERATION_COUNT_LONG = "--generation-count";
	const std::string CGPConfiguration::GENERATION_COUNT_SHORT = "-g";

	const std::string CGPConfiguration::NUMBER_OF_RUNS_LONG = "--number-of-runs";
	const std::string CGPConfiguration::NUMBER_OF_RUNS_SHORT = "-n";

	const std::string CGPConfiguration::FUNCTION_COUNT_LONG = "--function-count";
	const std::string CGPConfiguration::FUNCTION_COUNT_SHORT = "-fc";

	const std::string CGPConfiguration::INPUT_FILE_LONG = "--input-file";
	const std::string CGPConfiguration::INPUT_FILE_SHORT = "-i";

	const std::string CGPConfiguration::OUTPUT_FILE_LONG = "--output-file";
	const std::string CGPConfiguration::OUTPUT_FILE_SHORT = "-o";

	const std::string CGPConfiguration::CGP_STATISTICS_FILE_LONG = "--cgp-statistics-file";
	const std::string CGPConfiguration::CGP_STATISTICS_FILE_SHORT = "-s";

	const std::string CGPConfiguration::GATE_PARAMETERS_FILE_LONG = "--gate-parameters-file";
	const std::string CGPConfiguration::GATE_PARAMETERS_FILE_SHORT = "-fp";

	const std::string CGPConfiguration::TRAIN_WEIGHTS_FILE_LONG = "--train-weights-file";

	const std::string CGPConfiguration::MSE_THRESHOLD_LONG = "--mse-threshold";
	const std::string CGPConfiguration::MSE_THRESHOLD_SHORT = "-mse";

	const std::string CGPConfiguration::DATASET_SIZE_LONG = "--dataset-size";
	const std::string CGPConfiguration::DATASET_SIZE_SHORT = "-d";

	const std::string CGPConfiguration::START_GENERATION_LONG = "--start-generation";

	const std::string CGPConfiguration::START_RUN_LONG = "--start-run";

	const std::string CGPConfiguration::STARTING_SOLUTION_LONG = "--starting-solution";

	const std::string CGPConfiguration::PATIENCE_LONG = "--patience";

	const std::string CGPConfiguration::MSE_EARLY_STOP_LONG = "--mse-early-stop";

	const std::string CGPConfiguration::ENERGY_EARLY_STOP_LONG = "--energy-early-stop";

	const std::string CGPConfiguration::DELAY_EARLY_STOP_LONG = "--delay-early-stop";

	const std::string CGPConfiguration::DEPTH_EARLY_STOP_LONG = "--depth-early-stop";

	const std::string CGPConfiguration::GATE_COUNT_EARLY_STOP_LONG = "--gate-count-early-stop";

	const std::string CGPConfiguration::MSE_CHROMOSOME_LOGGING_THRESHOLD_LONG = "--mse-chromosome-logging-threshold";

	const std::string CGPConfiguration::LEARNING_RATE_FILE_LONG = "--learning-rate-file";

	const std::string CGPConfiguration::LEARNING_RATE_LONG = "--learning-rate";

	
	long long parse_integer_argument(const std::string& arg) {
		try {
			return std::stoll(arg);
		}
		catch (const std::invalid_argument& e) {
			// Handle invalid argument (not an integer).
			throw CGPConfigurationInvalidArgument("invalid integer argument for " + arg);
		}
		catch (const std::out_of_range& e) {
			// Handle out of range argument.
			throw CGPConfigurationOutOfRange("integer argument out of range for " + arg);
		}
	}

	long double parse_decimal_argument(const std::string& arg) {
		try {
			return std::stold(arg);
		}
		catch (const std::invalid_argument& e) {
			// Handle invalid argument (not an integer).
			throw CGPConfigurationInvalidArgument("invalid decimal argument for " + arg);
		}
		catch (const std::out_of_range& e) {
			// Handle out of range argument.
			throw CGPConfigurationOutOfRange("decimal argument out of range for " + arg);
		}
	}

	const std::string error_to_string(CGPConfiguration::error_t value)
	{
		return (value != CGPConfiguration::error_nan) ? (std::to_string(value)) : (CGPConfiguration::error_nan_string);
	}

	const std::string quantized_energy_to_string(CGPConfiguration::quantized_energy_t value)
	{
		return (value != CGPConfiguration::quantized_energy_nan) ? (std::to_string(value)) : (CGPConfiguration::quantized_energy_nan_string);
	}

	const std::string energy_to_string(CGPConfiguration::energy_t value)
	{
		return (value != CGPConfiguration::energy_nan) ? (std::to_string(value)) : (CGPConfiguration::energy_nan_string);
	}

	const std::string quantized_delay_to_string(CGPConfiguration::quantized_delay_t value)
	{
		return (value != CGPConfiguration::quantized_delay_nan) ? (std::to_string(value)) : (CGPConfiguration::quantized_delay_nan_string);
	}

	const std::string delay_to_string(CGPConfiguration::delay_t value)
	{
		return (value != CGPConfiguration::delay_nan) ? (std::to_string(value)) : (CGPConfiguration::delay_nan_string);
	}

	const std::string depth_to_string(CGPConfiguration::depth_t value)
	{
		return (value != CGPConfiguration::depth_nan) ? (std::to_string(value)) : (CGPConfiguration::depth_nan_string);
	}

	const std::string gate_count_to_string(CGPConfiguration::gate_count_t value)
	{
		return (value != CGPConfiguration::gate_count_nan) ? (std::to_string(value)) : (CGPConfiguration::gate_count_nan_string);
	}

	const std::string weight_to_string(CGPConfiguration::weight_value_t value)
	{
		return std::to_string(static_cast<CGPConfiguration::weight_repr_value_t>(value));
	}

	const std::string area_to_string(CGPConfiguration::area_t value)
	{
		return (value != CGPConfiguration::area_nan) ? (std::to_string(value)) : (CGPConfiguration::area_nan_string);
	}

	CGPConfiguration::error_t string_to_error(const std::string& value) {
		return (value == CGPConfiguration::error_nan_string) ? CGPConfiguration::error_nan : std::stoull(value);
	}

	CGPConfiguration::quantized_energy_t string_to_quantized_energy(const std::string& value) {
		return (value == CGPConfiguration::quantized_energy_nan_string) ? CGPConfiguration::quantized_energy_nan : std::stoull(value);
	}

	CGPConfiguration::energy_t string_to_energy(const std::string& value) {
		return (value == CGPConfiguration::energy_nan_string) ? CGPConfiguration::energy_nan : std::stold(value);
	}

	CGPConfiguration::area_t string_to_area(const std::string& value) {
		return (value == CGPConfiguration::area_nan_string) ? CGPConfiguration::area_nan : std::stold(value);
	}

	CGPConfiguration::quantized_delay_t string_to_quantized_delay(const std::string& value) {
		return (value == CGPConfiguration::quantized_delay_nan_string) ? CGPConfiguration::quantized_delay_nan : std::stoull(value);
	}

	CGPConfiguration::delay_t string_to_delay(const std::string& value) {
		return (value == CGPConfiguration::delay_nan_string) ? CGPConfiguration::delay_nan : std::stold(value);
	}

	CGPConfiguration::depth_t string_to_depth(const std::string& value) {
		return (value == CGPConfiguration::depth_nan_string) ? CGPConfiguration::depth_nan : std::stoul(value);
	}

	CGPConfiguration::gate_count_t string_to_gate_count(const std::string& value) {
		return (value == CGPConfiguration::gate_count_nan_string) ? CGPConfiguration::gate_count_nan : std::stoi(value);
	}

	CGPConfiguration::gate_parameters_t CGPConfiguration::get_default_gate_parameters()
	{
		return std::make_tuple(quantized_energy_nan, energy_nan, area_nan, quantized_delay_nan, delay_nan);
	}

	CGPConfiguration::quantized_energy_t CGPConfiguration::get_quantized_energy_parameter(const gate_parameters_t& params)
	{
		return std::get<0>(params);
	}

	CGPConfiguration::energy_t CGPConfiguration::get_energy_parameter(const gate_parameters_t& params)
	{
		return std::get<1>(params);
	}

	CGPConfiguration::area_t CGPConfiguration::get_area_parameter(const gate_parameters_t& params)
	{
		return std::get<2>(params);
	}

	CGPConfiguration::quantized_delay_t CGPConfiguration::get_quantized_delay_parameter(const gate_parameters_t& params)
	{
		return std::get<3>(params);
	}

	CGPConfiguration::delay_t CGPConfiguration::get_delay_parameter(const gate_parameters_t& params)
	{
		return std::get<4>(params);
	}

	void CGPConfiguration::set_quantized_energy_parameter(gate_parameters_t& params, quantized_energy_t energy)
	{
		std::get<0>(params) = energy;
	}

	void CGPConfiguration::set_energy_parameter(gate_parameters_t& params, energy_t energy)
	{
		std::get<1>(params) = energy;
	}

	void CGPConfiguration::set_area_parameter(gate_parameters_t& params, area_t area)
	{
		std::get<2>(params) = area;
	}

	void CGPConfiguration::set_quantized_delay_parameter(gate_parameters_t& params, quantized_delay_t delay)
	{
		std::get<3>(params) = delay;
	}

	void CGPConfiguration::set_delay_parameter(gate_parameters_t& params, delay_t delay)
	{
		std::get<4>(params) = delay;
	}

	void CGPConfiguration::set_quantized_energy_parameter(gate_parameters_t& params, const std::string& energy)
	{
		set_quantized_energy_parameter(params, (energy == quantized_energy_nan_string) ? (quantized_energy_nan) : (std::stoull(energy)));
	}

	void CGPConfiguration::set_energy_parameter(gate_parameters_t& params, const std::string& energy)
	{
		set_energy_parameter(params, (energy == energy_nan_string) ? (energy_nan) : (std::stold(energy)));
	}

	void CGPConfiguration::set_quantized_delay_parameter(gate_parameters_t& params, const std::string& delay)
	{
		set_quantized_delay_parameter(params, (delay == quantized_delay_nan_string) ? (quantized_delay_nan) : (std::stoull(delay)));
	}

	void CGPConfiguration::set_delay_parameter(gate_parameters_t& params, const std::string& delay)
	{
		set_delay_parameter(params, (delay == delay_nan_string) ? (delay_nan) : (std::stold(delay)));
	}

	void CGPConfiguration::set_area_parameter(gate_parameters_t& params, const std::string& area)
	{
		set_area_parameter(params, (area == area_nan_string) ? (area_nan) : (std::stold(area)));
	}

	CGPConfiguration& CGPConfiguration::start_generation(decltype(start_generation_value) value)
	{
		start_generation_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::start_run(decltype(start_run_value) value)
	{
		start_run_value = value;
		return *this;
	}

	CGPConfiguration::CGPConfiguration()
	{
		max_genes_to_mutate_value = static_cast<int>(std::ceil(chromosome_size() * mutation_max()));
	}

	CGPConfiguration::CGPConfiguration(const std::vector<std::string>& arguments)
	{
		set_from_arguments(arguments);
	}

	void CGPConfiguration::set_from_arguments(const std::vector<std::string>& arguments)
	{
		for (size_t i = 0; i < arguments.size(); i++)
		{
			try {
				if (arguments[i] == FUNCTION_INPUT_ARITY_LONG || arguments[i] == FUNCTION_INPUT_ARITY_SHORT) {
					function_input_arity(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == FUNCTION_OUTPUT_ARITY_LONG || arguments[i] == FUNCTION_OUTPUT_ARITY_SHORT) {
					function_output_arity(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == OUTPUT_COUNT_LONG || arguments[i] == OUTPUT_COUNT_SHORT) {
					output_count(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == INPUT_COUNT_LONG || arguments[i] == INPUT_COUNT_SHORT) {
					input_count(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == POPULATION_MAX_LONG || arguments[i] == POPULATION_MAX_SHORT) {
					population_max(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == MUTATION_MAX_LONG || arguments[i] == MUTATION_MAX_SHORT) {
					mutation_max(parse_decimal_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == LEARNING_RATE_LONG) {
					learning_rate(parse_decimal_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == ROW_COUNT_LONG || arguments[i] == ROW_COUNT_SHORT) {
					row_count(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == COL_COUNT_LONG || arguments[i] == COL_COUNT_SHORT) {
					col_count(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == LOOK_BACK_PARAMETER_LONG || arguments[i] == LOOK_BACK_PARAMETER_SHORT) {
					look_back_parameter(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == GENERATION_COUNT_LONG || arguments[i] == GENERATION_COUNT_SHORT) {
					generation_count(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == NUMBER_OF_RUNS_LONG || arguments[i] == NUMBER_OF_RUNS_SHORT) {
					number_of_runs(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == FUNCTION_COUNT_LONG || arguments[i] == FUNCTION_COUNT_SHORT) {
					function_count(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == INPUT_FILE_LONG || arguments[i] == INPUT_FILE_SHORT) {
					input_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == OUTPUT_FILE_LONG || arguments[i] == OUTPUT_FILE_SHORT) {
					output_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == CGP_STATISTICS_FILE_LONG || arguments[i] == CGP_STATISTICS_FILE_SHORT) {
					cgp_statistics_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == LEARNING_RATE_FILE_LONG) {
					learning_rate_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == GATE_PARAMETERS_FILE_LONG || arguments[i] == GATE_PARAMETERS_FILE_SHORT) {
					gate_parameters_input_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == TRAIN_WEIGHTS_FILE_LONG) {
					train_weights_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == MSE_THRESHOLD_LONG || arguments[i] == MSE_THRESHOLD_SHORT) {
					mse_threshold(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == DATASET_SIZE_LONG || arguments[i] == DATASET_SIZE_SHORT) {
					dataset_size(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == START_GENERATION_LONG) {
					start_generation(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == START_RUN_LONG) {
					start_run(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == STARTING_SOLUTION_LONG) {
					starting_solution(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == PATIENCE_LONG) {
					patience(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == MSE_EARLY_STOP_LONG) {
					mse_early_stop(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == ENERGY_EARLY_STOP_LONG) {
					energy_early_stop(parse_decimal_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == DELAY_EARLY_STOP_LONG) {
					delay_early_stop(parse_decimal_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == DEPTH_EARLY_STOP_LONG) {
					depth_early_stop(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == GATE_COUNT_EARLY_STOP_LONG) {
					gate_count_early_stop(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == MSE_CHROMOSOME_LOGGING_THRESHOLD_LONG) {
					mse_chromosome_logging_threshold(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == PERIODIC_LOG_FREQUENCY_LONG) {
					periodic_log_frequency(parse_integer_argument(arguments.at(i + 1)));
					i += 1;
				}
				else {
					throw CGPConfigurationInvalidArgument("unknown argument " + arguments[i]);
				}
			}
			catch (const std::out_of_range& e)
			{
				throw CGPConfigurationOutOfRange("missing argument for " + arguments[i]);
			}
		}
		max_genes_to_mutate_value = static_cast<int>(std::ceil(chromosome_size() * mutation_max()));
	}

	decltype(CGPConfiguration::function_input_arity_value) CGPConfiguration::function_input_arity() const {
		return function_input_arity_value;
	}

	int CGPConfiguration::pin_map_size() const {
#ifdef __VIRTUAL_SELECTOR
		return row_count() * col_count() * function_output_arity() + output_count() * dataset_size();
#else
		return row_count() * col_count() * function_output_arity() + output_count();
#endif // __VIRTUAL_SELECTOR
	}

	int CGPConfiguration::block_chromosome_size() const
	{
		return function_input_arity() + 1;
	}

	int CGPConfiguration::blocks_chromosome_size() const {
		return row_count() * col_count() * block_chromosome_size();
	}

	int CGPConfiguration::chromosome_size() const {
#ifdef __VIRTUAL_SELECTOR
		return blocks_chromosome_size() + output_count() * dataset_size();
#else
		return blocks_chromosome_size() + output_count();
#endif // __VIRTUAL_SELECTOR
	}

	const decltype(CGPConfiguration::function_costs_value)& CGPConfiguration::function_costs() const
	{
		return function_costs_value;
	}

	decltype(CGPConfiguration::function_output_arity_value) CGPConfiguration::function_output_arity() const {
		return function_output_arity_value;
	}


	decltype(CGPConfiguration::output_count_val) CGPConfiguration::output_count() const {
		return output_count_val;
	}


	decltype(CGPConfiguration::output_count_val) CGPConfiguration::input_count() const {
		return input_count_val;
	}


	decltype(CGPConfiguration::population_max_value) CGPConfiguration::population_max() const {
		return population_max_value;
	}


	decltype(CGPConfiguration::mutation_max_value) CGPConfiguration::mutation_max() const {
		return mutation_max_value;
	}


	decltype(CGPConfiguration::max_genes_to_mutate_value) CGPConfiguration::max_genes_to_mutate() const
	{
		return max_genes_to_mutate_value;
	}

	decltype(CGPConfiguration::row_count_value) CGPConfiguration::row_count() const {
		return row_count_value;
	}


	decltype(CGPConfiguration::col_count_value) CGPConfiguration::col_count() const {
		return col_count_value;
	}


	decltype(CGPConfiguration::look_back_parameter_value) CGPConfiguration::look_back_parameter() const {
		return look_back_parameter_value;
	}


	decltype(CGPConfiguration::generation_count_value) CGPConfiguration::generation_count() const {
		return generation_count_value;
	}


	decltype(CGPConfiguration::number_of_runs_value) CGPConfiguration::number_of_runs() const {
		return number_of_runs_value;
	}


	decltype(CGPConfiguration::function_count_value) CGPConfiguration::function_count() const {
		return function_count_value;
	}


	decltype(CGPConfiguration::periodic_log_frequency_value) CGPConfiguration::periodic_log_frequency() const {
		return periodic_log_frequency_value;
	}

	decltype(CGPConfiguration::input_file_value) CGPConfiguration::input_file() const
	{
		return input_file_value;
	}

	decltype(CGPConfiguration::output_file_value) CGPConfiguration::output_file() const
	{
		return output_file_value;
	}

	decltype(CGPConfiguration::cgp_statistics_file_value) CGPConfiguration::cgp_statistics_file() const
	{
		return cgp_statistics_file_value;
	}

	decltype(CGPConfiguration::learning_rate_file_value) CGPConfiguration::learning_rate_file() const
	{
		return learning_rate_file_value;
	}

	decltype(CGPConfiguration::mse_threshold_value) CGPConfiguration::mse_threshold() const
	{
		return mse_threshold_value;
	}

	decltype(CGPConfiguration::dataset_size_value) CGPConfiguration::dataset_size() const
	{
		return dataset_size_value;
	}

	decltype(CGPConfiguration::starting_solution_value) CGPConfiguration::starting_solution() const
	{
		return starting_solution_value;
	}

	decltype(CGPConfiguration::patience_value) CGPConfiguration::patience() const
	{
		return patience_value;
	}

	decltype(CGPConfiguration::learning_rate_value) CGPConfiguration::learning_rate() const
	{
		return learning_rate_value;
	}

	decltype(CGPConfiguration::start_generation_value) CGPConfiguration::start_generation() const
	{
		return start_generation_value;
	}

	decltype(CGPConfiguration::start_run_value) CGPConfiguration::start_run() const
	{
		return start_run_value;
	}

	decltype(CGPConfiguration::mse_early_stop_value) CGPConfiguration::mse_early_stop() const {
		return mse_early_stop_value;
	}

	decltype(CGPConfiguration::energy_early_stop_value) CGPConfiguration::energy_early_stop() const {
		return energy_early_stop_value;
	}

	decltype(CGPConfiguration::delay_early_stop_value) CGPConfiguration::delay_early_stop() const {
		return delay_early_stop_value;
	}

	decltype(CGPConfiguration::depth_early_stop_value) CGPConfiguration::depth_early_stop() const {
		return depth_early_stop_value;
	}

	decltype(CGPConfiguration::gate_count_early_stop_value) CGPConfiguration::gate_count_early_stop() const {
		return gate_count_early_stop_value;
	}

	decltype(CGPConfiguration::expected_value_min_value) CGPConfiguration::expected_value_min() const
	{
		return expected_value_min_value;
	}

	decltype(CGPConfiguration::expected_value_max_value) CGPConfiguration::expected_value_max() const
	{
		return expected_value_max_value;
	}

	decltype(CGPConfiguration::gate_parameters_input_file_value) CGPConfiguration::gate_parameters_input_file() const
	{
		return gate_parameters_input_file_value;
	}

	decltype(CGPConfiguration::train_weights_file_value) CGPConfiguration::train_weights_file() const
	{
		return train_weights_file_value;
	}

	decltype(CGPConfiguration::max_multiplexer_bit_variant_value) CGPConfiguration::max_multiplexer_bit_variant() const
	{
		return max_multiplexer_bit_variant_value;
	}

	decltype(CGPConfiguration::mse_chromosome_logging_threshold_value) CGPConfiguration::mse_chromosome_logging_threshold() const
	{
		return mse_chromosome_logging_threshold_value;
	}

	CGPConfiguration& CGPConfiguration::learning_rate_file(decltype(learning_rate_file_value) value)
	{
		learning_rate_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::function_input_arity(decltype(function_input_arity_value) value) {
		function_input_arity_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::function_output_arity(decltype(function_output_arity_value) value) {
		function_output_arity_value = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::output_count(decltype(output_count_val) value) {
		output_count_val = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::input_count(decltype(input_count_val) value) {
		input_count_val = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::population_max(decltype(population_max_value) value) {
		population_max_value = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::mutation_max(decltype(mutation_max_value) value) {
		mutation_max_value = value;
		max_genes_to_mutate_value = static_cast<int>(std::ceil(chromosome_size() * value));
		return *this;
	}


	CGPConfiguration& CGPConfiguration::row_count(decltype(row_count_value) value) {
		row_count_value = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::col_count(decltype(col_count_value) value) {
		col_count_value = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::look_back_parameter(decltype(look_back_parameter_value) value) {
		look_back_parameter_value = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::generation_count(decltype(generation_count_value) value) {
		generation_count_value = value;
		periodic_log_frequency(static_cast<uint32_t>(generation_count_value / 2.0));
		return *this;
	}


	CGPConfiguration& CGPConfiguration::number_of_runs(decltype(number_of_runs_value) value) {
		number_of_runs_value = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::function_count(decltype(function_count_value) value) {
		function_count_value = value;
		return *this;
	}


	CGPConfiguration& CGPConfiguration::periodic_log_frequency(decltype(periodic_log_frequency_value) value) {
		periodic_log_frequency_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::function_costs(decltype(function_costs_value)&& value)
	{
		function_costs_value = std::move(value);
		return *this;
	}

	CGPConfiguration& CGPConfiguration::input_file(decltype(input_file_value) value)
	{
		input_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::cgp_statistics_file(decltype(cgp_statistics_file_value) value)
	{
		cgp_statistics_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::output_file(decltype(output_file_value) value)
	{
		output_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::mse_threshold(decltype(mse_threshold_value) value)
	{
		mse_threshold_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::dataset_size(decltype(dataset_size_value) value)
	{
		dataset_size_value = value;
		max_multiplexer_bit_variant_value = std::min(max_hardware_multiplexer_bit_variant, static_cast<int>(std::ceil(std::log2(value))));
		return *this;
	}

	CGPConfiguration& CGPConfiguration::starting_solution(decltype(starting_solution_value) value)
	{
		starting_solution_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::patience(decltype(patience_value) value)
	{
		patience_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::mse_early_stop(decltype(mse_early_stop_value) value) {
		mse_early_stop_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::energy_early_stop(decltype(energy_early_stop_value) value) {
		energy_early_stop_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::delay_early_stop(decltype(delay_early_stop_value) value) {
		delay_early_stop_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::depth_early_stop(decltype(depth_early_stop_value) value) {
		depth_early_stop_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::gate_count_early_stop(decltype(gate_count_early_stop_value) value) {
		gate_count_early_stop_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::expected_value_min(decltype(expected_value_min_value) value)
	{
		expected_value_min_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::expected_value_max(decltype(expected_value_max_value) value)
	{
		expected_value_max_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::gate_parameters_input_file(decltype(gate_parameters_input_file_value) value)
	{
		gate_parameters_input_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::train_weights_file(decltype(train_weights_file_value) value)
	{
		train_weights_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::mse_chromosome_logging_threshold(decltype(mse_chromosome_logging_threshold_value) value)
	{
		mse_chromosome_logging_threshold_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::learning_rate(decltype(learning_rate_value) value)
	{
		learning_rate_value = value;
		return *this;
	}

	void CGPConfiguration::dump(std::ostream& out) const
	{
		// Serialize each variable to the file
		out << "function_input_arity: " << static_cast<int>(function_input_arity()) << std::endl;
		out << "function_output_arity: " << static_cast<int>(function_output_arity()) << std::endl;
		out << "output_count: " << output_count() << std::endl;
		out << "input_count: " << input_count() << std::endl;
		out << "population_max: " << population_max() << std::endl;
		out << "mutation_max: " << mutation_max() << std::endl;
		out << "learning_rate: " << learning_rate() << std::endl;
		out << "row_count: " << row_count() << std::endl;
		out << "col_count: " << col_count() << std::endl;
		out << "look_back_parameter: " << look_back_parameter() << std::endl;
		out << "generation_count: " << generation_count() << std::endl;
		out << "number_of_runs: " << number_of_runs() << std::endl;
		out << "function_count: " << static_cast<int>(function_count()) << std::endl;
		out << "periodic_log_frequency: " << periodic_log_frequency() << std::endl;
		if (!input_file().empty()) out << "input_file: " << input_file() << std::endl;
		if (!output_file().empty()) out << "output_file: " << output_file() << std::endl;
		if (!cgp_statistics_file().empty()) out << "cgp_statistics_file: " << cgp_statistics_file() << std::endl;
		if (!learning_rate_file().empty()) out << "learning_rate_file: " << learning_rate_file() << std::endl;
		if (!gate_parameters_input_file().empty()) out << "gate_parameters_file: " << gate_parameters_input_file() << std::endl;
		if (!train_weights_file().empty() && train_weights_file()[0] != '#') out << "train_weights_file: " << train_weights_file() << std::endl;
		if (!starting_solution().empty()) out << "starting_solution: " << starting_solution() << std::endl;
		out << "mse_threshold: " << error_to_string(mse_threshold()) << std::endl;
		out << "mse_chromosome_logging_threshold: " << error_to_string(mse_chromosome_logging_threshold()) << std::endl;
		out << "dataset_size: " << dataset_size() << std::endl;
		out << "patience: " << patience() << std::endl;
		out << "mse_early_stop: " << error_to_string(mse_early_stop()) << std::endl;
		out << "energy_early_stop: " << energy_to_string(energy_early_stop()) << std::endl;
		out << "delay_early_stop: " << delay_to_string(delay_early_stop()) << std::endl;
		out << "depth_early_stop: " << depth_to_string(depth_early_stop()) << std::endl;
		out << "gate_count_early_stop: " << gate_count_to_string(gate_count_early_stop()) << std::endl;
		out << "expected_value_min: " << weight_to_string(expected_value_min()) << std::endl;
		out << "expected_value_max: " << weight_to_string(expected_value_max()) << std::endl;
	}

	std::map<std::string, std::string> CGPConfiguration::load(std::istream& in, const std::vector<std::string>& arguments)
	{
		std::string line;
		std::map<std::string, std::string> remaining_data;

		// Read each line from the file and parse it to extract variable name and value
		while (std::getline(in, line)) {
			std::string key = line;
			std::string value = line;

			// Remove leading/trailing whitespaces from key and value
			key.erase(0, key.find_first_not_of(" \t\r\n"));
			key.erase(key.find_first_of(":"));
			value.erase(0, value.find_first_of(":") + 1);
			value.erase(0, value.find_first_not_of(" \t\r\n"));
			value.erase(value.find_last_not_of(" \t\r\n") + 1);

			// Set the variable based on the key and value
			if (key == "function_input_arity") function_input_arity(std::stoi(value));
			else if (key == "function_output_arity") function_output_arity(std::stoi(value));
			else if (key == "output_count") output_count(std::stoi(value));
			else if (key == "input_count") input_count(std::stoi(value));
			else if (key == "row_count") row_count(std::stoi(value));
			else if (key == "col_count") col_count(std::stoi(value));
			else if (key == "look_back_parameter") look_back_parameter(std::stoi(value));
			else if (key == "population_max") population_max(std::stoi(value));
			else if (key == "learning_rate") learning_rate(std::stod(value));
			else if (key == "generation_count") generation_count(std::stoull(value));
			else if (key == "number_of_runs") number_of_runs(std::stoi(value));
			else if (key == "function_count") function_count(std::stoi(value));
			else if (key == "periodic_log_frequency") periodic_log_frequency(std::stoull(value));
			else if (key == "input_file") input_file(value);
			else if (key == "output_file") output_file(value);
			else if (key == "cgp_statistics_file") cgp_statistics_file(value);
			else if (key == "learning_rate_file") learning_rate_file(value);
			else if (key == "gate_parameters_file") gate_parameters_input_file(value);
			else if (key == "train_weights_file") train_weights_file(value);
			else if (key == "starting_solution") starting_solution(value);
			else if (key == "mse_threshold") mse_threshold(string_to_error(value));
			else if (key == "mse_chromosome_logging_threshold") mse_chromosome_logging_threshold(string_to_error(value));
			else if (key == "dataset_size") dataset_size(std::stoi(value));
			else if (key == "patience") patience(std::stoi(value));
			else if (key == "mse_early_stop") mse_early_stop(string_to_error(value));
			else if (key == "energy_early_stop") energy_early_stop(string_to_energy(value));
			else if (key == "delay_early_stop") delay_early_stop(string_to_delay(value));
			else if (key == "depth_early_stop") depth_early_stop(string_to_depth(value));
			else if (key == "gate_count_early_stop") gate_count_early_stop(string_to_gate_count(value));
			else if (key == "expected_value_min") expected_value_min(std::stoi(value));
			else if (key == "expected_value_max") expected_value_max(std::stoi(value));
			else if (key == "mutation_max") mutation_max(std::stof(value));
			
			else if (!key.empty() && key != "start_generation" && key != "start_run")
			{
				remaining_data[key] = value;
			}
			else if (key == "start_generation" || key == "start_run")
			{
				throw CGPConfigurationInvalidArgument("invalid attribute: " + key);
			}
		}
		max_genes_to_mutate_value = static_cast<int>(std::ceil(chromosome_size() * mutation_max()));
		return remaining_data;
	}
}
