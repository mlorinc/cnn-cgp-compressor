#include "Configuration.h"
#include <cstdlib>

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
	const std::string CGPConfiguration::CGPConfiguration::POPULATION_MAX_SHORT = "-pm";

	const std::string CGPConfiguration::MUTATION_MAX_LONG = "--mutation-max";
	const std::string CGPConfiguration::MUTATION_MAX_SHORT = "-mm";

	const std::string CGPConfiguration::ROW_COUNT_LONG = "--row-count";
	const std::string CGPConfiguration::ROW_COUNT_SHORT = "-rc";

	const std::string CGPConfiguration::COL_COUNT_LONG = "--col-count";
	const std::string CGPConfiguration::COL_COUNT_SHORT = "-cc";

	const std::string CGPConfiguration::LOOK_BACK_PARAMETER_LONG = "--look-back-parameter";
	const std::string CGPConfiguration::LOOK_BACK_PARAMETER_SHORT = "-lbp";

	const std::string CGPConfiguration::GENERATION_COUNT_LONG = "--generation-count";
	const std::string CGPConfiguration::GENERATION_COUNT_SHORT = "-gc";

	const std::string CGPConfiguration::NUMBER_OF_RUNS_LONG = "--number-of-runs";
	const std::string CGPConfiguration::NUMBER_OF_RUNS_SHORT = "-nr";

	const std::string CGPConfiguration::FUNCTION_COUNT_LONG = "--function-count";
	const std::string CGPConfiguration::FUNCTION_COUNT_SHORT = "-fc";

	const std::string CGPConfiguration::INPUT_FILE_LONG = "--input-file";
	const std::string CGPConfiguration::INPUT_FILE_SHORT = "-i";

	const std::string CGPConfiguration::OUTPUT_FILE_LONG = "--output-file";
	const std::string CGPConfiguration::OUTPUT_FILE_SHORT = "-o";

	const std::string CGPConfiguration::CHROMOSOME_OUTPUT_FILE_LONG = "--chromosome-output-file";
	const std::string CGPConfiguration::CHROMOSOME_OUTPUT_FILE_SHORT = "-co";

	const std::string CGPConfiguration::CGP_METADATA_OUTPUT_FILE_LONG = "--cgp-metadata-output-file";
	const std::string CGPConfiguration::CGP_METADATA_OUTPUT_FILE_SHORT = "-cmo";

	const std::string CGPConfiguration::CHROMOSOME_INPUT_FILE_LONG = "--chromosome-input-file";
	const std::string CGPConfiguration::CHROMOSOME_INPUT_FILE_SHORT = "-ci";

	const std::string CGPConfiguration::CGP_METADATA_INPUT_FILE_LONG = "--cgp-metadata-input-file";
	const std::string CGPConfiguration::CGP_METADATA_INPUT_FILE_SHORT = "-cmi";

	static long long parse_integer_argument(const std::string& arg) {
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

	static long double parse_decimal_argument(const std::string& arg) {
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

	CGPConfiguration::CGPConfiguration(const std::vector<std::string>& arguments) : CGPConfiguration()
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
				else if (arguments[i] == CHROMOSOME_OUTPUT_FILE_LONG || arguments[i] == CHROMOSOME_OUTPUT_FILE_SHORT) {
					chromosome_output_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == CGP_METADATA_OUTPUT_FILE_LONG || arguments[i] == CGP_METADATA_OUTPUT_FILE_SHORT) {
					cgp_metadata_output_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == CHROMOSOME_INPUT_FILE_LONG || arguments[i] == CHROMOSOME_INPUT_FILE_SHORT) {
					chromosome_input_file(arguments.at(i + 1));
					i += 1;
				}
				else if (arguments[i] == CGP_METADATA_INPUT_FILE_LONG || arguments[i] == CGP_METADATA_INPUT_FILE_SHORT) {
					cgp_metadata_input_file(arguments.at(i + 1));
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


	decltype(CGPConfiguration::periodic_log_value) CGPConfiguration::periodic_log() const {
		return periodic_log_value;
	}

	decltype(CGPConfiguration::input_file_value) CGPConfiguration::input_file() const
	{
		return input_file_value;
	}

	decltype(CGPConfiguration::output_file_value) CGPConfiguration::output_file() const
	{
		return output_file_value;
	}

	decltype(CGPConfiguration::chromosome_output_file_value) CGPConfiguration::chromosome_output_file() const
	{
		return chromosome_output_file_value;
	}

	decltype(CGPConfiguration::cgp_metadata_output_file_value) CGPConfiguration::cgp_metadata_output_file() const
	{
		return cgp_metadata_output_file_value;
	}

	decltype(CGPConfiguration::chromosome_input_file_value) CGPConfiguration::chromosome_input_file() const
	{
		return chromosome_input_file_value;
	}

	decltype(CGPConfiguration::cgp_metadata_input_file_value) CGPConfiguration::cgp_metadata_input_file() const
	{
		return cgp_metadata_input_file_value;
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


	CGPConfiguration& CGPConfiguration::periodic_log(decltype(periodic_log_value) value) {
		periodic_log_value = value;
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

	CGPConfiguration& CGPConfiguration::chromosome_output_file(decltype(chromosome_output_file_value) value)
	{
		chromosome_output_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::cgp_metadata_output_file(decltype(cgp_metadata_output_file_value) value)
	{
		cgp_metadata_output_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::chromosome_input_file(decltype(chromosome_input_file_value) value)
	{
		chromosome_input_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::cgp_metadata_input_file(decltype(cgp_metadata_input_file_value) value)
	{
		cgp_metadata_input_file_value = value;
		return *this;
	}

	CGPConfiguration& CGPConfiguration::output_file(decltype(output_file_value) value)
	{
		output_file_value = value;
		return *this;
	}
}
