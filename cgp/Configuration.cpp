#include "Configuration.h"
#include <cstdlib>
#include <sstream>

namespace cgp {
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

	const std::string CGPConfiguration::MSE_THRESHOLD_LONG = "--mse-threshold";
	const std::string CGPConfiguration::MSE_THRESHOLD_SHORT = "-mse";

	const std::string CGPConfiguration::DATASET_SIZE_LONG = "--dataset-size";
	const std::string CGPConfiguration::DATASET_SIZE_SHORT = "-d";

	const std::string CGPConfiguration::START_GENERATION_LONG = "--start-generation";

	const std::string CGPConfiguration::START_RUN_LONG = "--start-run";

	const std::string CGPConfiguration::STARTING_SOLUTION_LONG = "--starting_solution";

	const std::string CGPConfiguration::PATIENCE_LONG = "--patience";

	const std::string CGPConfiguration::MSE_EARLY_STOP_LONG = "--mse-early-stop";

	const std::string CGPConfiguration::ENERGY_EARLY_STOP_LONG = "--energy-early-stop";

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
		max_genes_to_mutate_value = chromosome_size() * mutation_max();
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
				else if (arguments[i] == MSE_THRESHOLD_LONG || arguments[i] == MSE_THRESHOLD_SHORT) {
					mse_threshold(parse_decimal_argument(arguments.at(i + 1)));
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
					mse_early_stop(parse_decimal_argument(arguments.at(i + 1)));
					i += 1;
				}
				else if (arguments[i] == ENERGY_EARLY_STOP_LONG) {
					energy_early_stop(parse_decimal_argument(arguments.at(i + 1)));
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
		max_genes_to_mutate_value = chromosome_size() * mutation_max();
	}

	decltype(CGPConfiguration::function_input_arity_value) CGPConfiguration::function_input_arity() const {
		return function_input_arity_value;
	}

	uint16_t CGPConfiguration::pin_map_size() const {
		return static_cast<uint16_t>(row_count()) * col_count() * function_output_arity() + output_count() + input_count();
	}

	size_t CGPConfiguration::blocks_chromosome_size() const {
		return row_count() * col_count() * (function_input_arity() + 1);
	}

	size_t CGPConfiguration::chromosome_size() const {
		return blocks_chromosome_size() + output_count();
	}

	decltype(CGPConfiguration::function_energy_costs_value) CGPConfiguration::function_energy_costs() const
	{
		return function_energy_costs_value;
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

	decltype(CGPConfiguration::mse_threshold_value) CGPConfiguration::mse_threshold() const
	{
		return mse_threshold_value;
	}

	inline decltype(CGPConfiguration::dataset_size_value) CGPConfiguration::dataset_size() const
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

	decltype(CGPConfiguration::mse_early_stop_value) CGPConfiguration::mse_early_stop() const
	{
		return mse_early_stop_value;
	}

	decltype(CGPConfiguration::start_generation_value) CGPConfiguration::start_generation() const
	{
		return start_generation_value;
	}

	decltype(CGPConfiguration::start_run_value) CGPConfiguration::start_run() const
	{
		return start_run_value;
	}

	decltype(CGPConfiguration::energy_early_stop_value) CGPConfiguration::energy_early_stop() const
	{
		return energy_early_stop_value;
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
		max_genes_to_mutate_value = chromosome_size() * value;
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

	CGPConfiguration& CGPConfiguration::function_energy_costs(decltype(function_energy_costs_value) value)
	{
		function_energy_costs_value = value;
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

	CGPConfiguration& CGPConfiguration::mse_early_stop(decltype(mse_early_stop_value) value)
	{
		mse_early_stop_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::energy_early_stop(decltype(energy_early_stop_value) value)
	{
		energy_early_stop_value = value;
		return *this;
	}

	void CGPConfiguration::dump(std::ostream &out) const
	{
		// Serialize each variable to the file
		out << "function_input_arity: " << static_cast<int>(function_input_arity()) << std::endl;
		out << "function_output_arity: " << static_cast<int>(function_output_arity()) << std::endl;
		out << "output_count: " << output_count() << std::endl;
		out << "input_count: " << input_count() << std::endl;
		out << "population_max: " << population_max() << std::endl;
		out << "mutation_max: " << mutation_max() << std::endl;
		out << "row_count: " << row_count() << std::endl;
		out << "col_count: " << col_count() << std::endl;
		out << "look_back_parameter: " << look_back_parameter() << std::endl;
		out << "generation_count: " << generation_count() << std::endl;
		out << "number_of_runs: " << number_of_runs() << std::endl;
		out << "function_count: " << static_cast<int>(function_count()) << std::endl;
		out << "periodic_log_frequency: " << periodic_log_frequency() << std::endl;
		out << "input_file: " << input_file() << std::endl;
		out << "output_file: " << output_file() << std::endl;
		out << "cgp_statistics_file: " << cgp_statistics_file() << std::endl;
		out << "starting_solution: " << starting_solution() << std::endl;
		out << "mse_threshold: " << mse_threshold() << std::endl;
		out << "dataset_size: " << dataset_size() << std::endl;
		out << "patience: " << patience() << std::endl;
		out << "mse_early_stop: " << mse_early_stop() << std::endl;
		out << "energy_early_stop: " << energy_early_stop() << std::endl;
	}

	std::map<std::string, std::string> CGPConfiguration::load(std::istream &in)
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
			else if (key == "output_count") output_count(std::stoull(value));
			else if (key == "input_count") input_count(std::stoull(value));
			else if (key == "population_max") population_max(std::stoi(value));
			else if (key == "mutation_max") mutation_max(std::stod(value));
			else if (key == "row_count") row_count(std::stoi(value));
			else if (key == "col_count") col_count(std::stoi(value));
			else if (key == "look_back_parameter") look_back_parameter(std::stoi(value));
			else if (key == "generation_count") generation_count(std::stoull(value));
			else if (key == "number_of_runs") number_of_runs(std::stoull(value));
			else if (key == "function_count") function_count(std::stoi(value));
			else if (key == "periodic_log_frequency") periodic_log_frequency(std::stoull(value));
			else if (key == "input_file") input_file(value);
			else if (key == "output_file") output_file(value);
			else if (key == "cgp_statistics_file") cgp_statistics_file(value);
			else if (key == "starting_solution") starting_solution(value);
			else if (key == "mse_threshold") mse_threshold(std::stod(value));
			else if (key == "dataset_size") dataset_size(std::stoull(value));
			else if (key == "patience") patience(std::stoull(value));
			else if (key == "mse_early_stop") mse_early_stop(std::stold(value));
			else if (key == "energy_early_stop") energy_early_stop(std::stold(value));
			else if (!key.empty() && key != "start_generation" && key != "start_run")
			{
				remaining_data[key] = value;
			}
			else if (key == "start_generation" || key == "start_run")
			{
				throw CGPConfigurationInvalidArgument("invalid attribute: " + key);
			}
		}
		max_genes_to_mutate_value = chromosome_size() * mutation_max();
		return remaining_data;
	}
}
