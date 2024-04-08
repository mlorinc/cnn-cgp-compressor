#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <map>

namespace cgp {
	long long parse_integer_argument(const std::string& arg);
	long double parse_decimal_argument(const std::string& arg);

	class CGPConfigurationInvalidArgument : public std::invalid_argument
	{
	public:
		CGPConfigurationInvalidArgument(const std::string& message)
			: std::invalid_argument(message) {}
	};

	class CGPConfigurationOutOfRange : public std::out_of_range
	{
	public:
		CGPConfigurationOutOfRange(const std::string& message)
			: std::out_of_range(message) {}
	};

	/// <summary>
	/// Configuration class for Cartesian Genetic Programming (CGP).
	/// </summary>
	class CGPConfiguration
	{
	public:
		/// <summary>
		/// Type alias for dimension values, represented as unsigned 16-bit integers.
		/// </summary>
		using dimension_t = uint16_t;

		/// <summary>
		/// Type alias for error values, represented as double-precision floating-point numbers.
		/// </summary>
		using error_t = double;

		/// <summary>
		/// Type alias for energy values, represented as double-precision floating-point numbers.
		/// </summary>
		using energy_t = double;

		/// <summary>
		/// Type alias for delay values, represented as double-precision floating-point numbers.
		/// </summary>
		using delay_t = double;

		/// <summary>
		/// Type alias for depth values, represented as dimension_t.
		/// </summary>
		using depth_t = dimension_t;

		/// <summary>
		/// Type alias for gate count values, represented as size_t.
		/// </summary>
		using gate_count_t = size_t;

		/// <summary>
		/// Type alias for gate parameters, represented as a tuple of energy and delay values.
		/// </summary>
		using gate_parameters_t = std::tuple<energy_t, delay_t>;

		/// <summary>
		/// Type alias for gene values, represented as unsigned 16-bit integers.
		/// </summary>
		using gene_t = uint16_t;

		/// <summary>
		/// Value representing NaN for error_t.
		/// </summary>
		static constexpr error_t error_nan = (std::numeric_limits<error_t>::has_infinity) ? (std::numeric_limits<error_t>::infinity()) : (std::numeric_limits<error_t>::max());

		/// <summary>
		/// Value representing NaN for energy_t.
		/// </summary>
		static constexpr energy_t energy_nan = (std::numeric_limits<energy_t>::has_infinity) ? (std::numeric_limits<energy_t>::infinity()) : (std::numeric_limits<energy_t>::max());

		/// <summary>
		/// Value representing NaN for delay_t.
		/// </summary>
		static constexpr delay_t delay_nan = (std::numeric_limits<delay_t>::has_infinity) ? (std::numeric_limits<delay_t>::infinity()) : (std::numeric_limits<delay_t>::max());

		/// <summary>
		/// Value representing NaN for depth_t.
		/// </summary>
		static constexpr depth_t depth_nan = (std::numeric_limits<depth_t>::has_infinity) ? (std::numeric_limits<depth_t>::infinity()) : (std::numeric_limits<depth_t>::max());

		/// <summary>
		/// Value representing NaN for gate_count_t.
		/// </summary>
		static constexpr gate_count_t gate_count_nan = (std::numeric_limits<gate_count_t>::has_infinity) ? (std::numeric_limits<gate_count_t>::infinity()) : (std::numeric_limits<gate_count_t>::max());

		/// <summary>
		/// Retrieve the default gate parameters.
		/// </summary>
		/// <returns>A tuple containing default energy and delay parameters.</returns>
		static gate_parameters_t get_default_gate_parameters();

		/// <summary>
		/// Get the energy parameter from the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <returns>The energy parameter.</returns>
		static energy_t get_energy_parameter(const gate_parameters_t& params);

		/// <summary>
		/// Get the delay parameter from the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <returns>The delay parameter.</returns>
		static delay_t get_delay_parameter(const gate_parameters_t& params);

		/// <summary>
		/// Set the energy parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="energy">The energy parameter to set.</param>
		static void set_energy_parameter(gate_parameters_t& params, energy_t energy);

		/// <summary>
		/// Set the delay parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="delay">The delay parameter to set.</param>
		static void set_delay_parameter(gate_parameters_t& params, delay_t delay);

		/// <summary>
		/// String representation of error_nan.
		/// </summary>
		static const std::string error_nan_string;

		/// <summary>
		/// String representation of energy_nan.
		/// </summary>
		static const std::string energy_nan_string;

		/// <summary>
		/// String representation of delay_nan.
		/// </summary>
		static const std::string delay_nan_string;

		/// <summary>
		/// String representation of depth_nan.
		/// </summary>
		static const std::string depth_nan_string;

		/// <summary>
		/// String representation of gate_count_nan.
		/// </summary>
		static const std::string gate_count_nan_string;


		static const std::string PERIODIC_LOG_FREQUENCY_LONG;

		static const std::string FUNCTION_INPUT_ARITY_LONG;
		static const std::string FUNCTION_INPUT_ARITY_SHORT;

		static const std::string FUNCTION_OUTPUT_ARITY_LONG;
		static const std::string FUNCTION_OUTPUT_ARITY_SHORT;

		static const std::string OUTPUT_COUNT_LONG;
		static const std::string OUTPUT_COUNT_SHORT;

		static const std::string INPUT_COUNT_LONG;
		static const std::string INPUT_COUNT_SHORT;

		static const std::string POPULATION_MAX_LONG;
		static const std::string POPULATION_MAX_SHORT;

		static const std::string MUTATION_MAX_LONG;
		static const std::string MUTATION_MAX_SHORT;

		static const std::string ROW_COUNT_LONG;
		static const std::string ROW_COUNT_SHORT;

		static const std::string COL_COUNT_LONG;
		static const std::string COL_COUNT_SHORT;

		static const std::string LOOK_BACK_PARAMETER_LONG;
		static const std::string LOOK_BACK_PARAMETER_SHORT;

		static const std::string GENERATION_COUNT_LONG;
		static const std::string GENERATION_COUNT_SHORT;

		static const std::string NUMBER_OF_RUNS_LONG;
		static const std::string NUMBER_OF_RUNS_SHORT;

		static const std::string FUNCTION_COUNT_LONG;
		static const std::string FUNCTION_COUNT_SHORT;

		static const std::string INPUT_FILE_LONG;
		static const std::string INPUT_FILE_SHORT;

		static const std::string OUTPUT_FILE_LONG;
		static const std::string OUTPUT_FILE_SHORT;

		static const std::string CGP_STATISTICS_FILE_LONG;
		static const std::string CGP_STATISTICS_FILE_SHORT;

		static const std::string MSE_THRESHOLD_LONG;
		static const std::string MSE_THRESHOLD_SHORT;

		static const std::string DATASET_SIZE_LONG;
		static const std::string DATASET_SIZE_SHORT;

		static const std::string START_GENERATION_LONG;

		static const std::string START_RUN_LONG;

		static const std::string STARTING_SOLUTION_LONG;

		static const std::string PATIENCE_LONG;

		static const std::string MSE_EARLY_STOP_LONG;

		static const std::string ENERGY_EARLY_STOP_LONG;

#ifndef CNN_FP32_WEIGHTS
		/// <summary>
		/// Type alias for inferred weight used internally by the CGP, represented as a signed 16-bit integer
		/// to prevent overflow when multiplying.
		/// </summary>
		using weight_value_t = int16_t;

		/// <summary>
		/// Type alias for hardware inferred weight, represented as a signed 8-bit integer.
		/// </summary>
		using weight_actual_value_t = int8_t;

		/// <summary>
		/// Type alias for serialization, represented as a signed integer.
		/// Used to distinguish char from int8_t.
		/// </summary>
		using weight_repr_value_t = int;

		/// <summary>
		/// Value representing an invalid value of arbitary weight value.
		/// This happens when multiplexer is incorrectly used.
		/// </summary>
		static constexpr weight_value_t invalid_value = std::numeric_limits<weight_value_t>::max();

		/// <summary>
		/// Value representing a weight that is not considered during error evaluation.
		/// </summary>
		static constexpr weight_value_t no_care_value = std::numeric_limits<weight_value_t>::min();
#else
		/// <summary>
		/// Type alias for inferred weight, represented as a double-precision floating-point number.
		/// </summary>
		using weight_value_t = double;

		/// <summary>
		/// Type alias for inferred weight, represented as a double-precision floating-point number.
		/// </summary>
		using weight_actual_value_t = double;

		/// <summary>
		/// Type alias for serialization, represented as a double-precision floating-point number.
		/// </summary>
		using weight_repr_value_t = double;

		/// <summary>
		/// Value representing an invalid weight, represented as positive infinity.
		/// </summary>
		static constexpr weight_value_t invalid_value = std::numeric_limits<weight_value_t>::infinity();

		/// <summary>
		/// Value representing a weight that is not considered during error evaluation.
		/// </summary>
		static constexpr weight_value_t no_care_value = std::numeric_limits<weight_value_t>::min();
#endif // !CNN_FP32_WEIGHTS


	private:
		/// <summary>
		/// Default value for the input arity of functions.
		/// </summary>
		uint8_t function_input_arity_value = 2;

		/// <summary>
		/// Default value for the output arity of functions.
		/// </summary>
		uint8_t function_output_arity_value = 1;

		/// <summary>
		/// Default value for the number of output pins in the CGP.
		/// </summary>
		size_t output_count_val = 1;

		/// <summary>
		/// Default value for the number of input pins in the CGP.
		/// </summary>
		size_t input_count_val = 2;

		/// <summary>
		/// Default value for the maximum population size in the CGP algorithm.
		/// </summary>
		uint16_t population_max_value = 5;

		/// <summary>
		/// Default value for the maximum mutation value in the CGP algorithm.
		/// </summary>
		double mutation_max_value = 0.15;

		/// <summary>
		/// Variable limiting maximum number of genes that can be mutated.
		/// </summary>
		size_t max_genes_to_mutate_value;

		/// <summary>
		/// Default value for the number of rows in the CGP grid.
		/// </summary>
		dimension_t row_count_value = 5;

		/// <summary>
		/// Default value for the number of columns in the CGP grid.
		/// </summary>
		dimension_t col_count_value = 5;

		/// <summary>
		/// Default value for the look-back parameter in the CGP algorithm.
		/// </summary>
		uint16_t look_back_parameter_value = 1;

		/// <summary>
		/// Default value for the maximum number of generations in the CGP algorithm.
		/// </summary>
		uint64_t generation_count_value = 5000;

		/// <summary>
		/// Default value for the number of runs in the CGP algorithm.
		/// </summary>
		uint32_t number_of_runs_value = 10;

		/// <summary>
		/// Default value for the number of functions in the CGP algorithm.
		/// </summary>
		uint8_t function_count_value = 33;

		/// <summary>
		/// Default value for the log frequency in the CGP algorithm.
		/// </summary>
		size_t periodic_log_frequency_value = 100000;

		/// <summary>
		/// Array of energy costs for various operations.
		/// </summary>
		std::shared_ptr<gate_parameters_t[]> function_costs_value;

		/// <summary>
		/// A path to a file with input data.
		/// </summary>
		std::string input_file_value = "-";

		/// <summary>
		/// A path to a file to create which contains output of the CGP process.
		/// </summary>
		std::string output_file_value = "-";

		/// <summary>
		/// A path where resulting chromosome array will be saved.
		/// </summary>
		std::string chromosome_output_file_value = "-";

		/// <summary>
		/// A path where CGP statistics will be saved.
		/// </summary>
		std::string cgp_statistics_file_value = "+";

		/// <summary>
		/// A path where chromosome array is saved.
		/// </summary>
		std::string chromosome_input_file_value = "";

		/// <summary>
		/// Mean Squared Error threshold after optimisation is focused on minimising energy.
		/// </summary>
		error_t mse_threshold_value = 5;

		/// <summary>
		/// The CGP dataset size.
		/// </summary>
		size_t dataset_size_value = 1;

		/// <summary>
		/// Used in case CGP evolution is resumed.
		/// </summary>
		size_t start_generation_value = 0;

		/// <summary>
		/// Used in case CGP evolution is resumed.
		/// </summary>
		size_t start_run_value = 0;

		/// <summary>
		/// Used in case CGP evolution is resumed.
		/// </summary>
		std::string starting_solution_value;

		/// <summary>
		/// Value indicating after how many generations CGP will come to stop.
		/// </summary>
		size_t patience_value = 125000;

		/// <summary>
		/// Value indicating stop condition for parameter of approximation error. Defaultly
		/// that stop condition is delegated to energy parameter.
		/// </summary>
		error_t mse_early_stop_value = -1;

		/// <summary>
		/// Value indicating stop condition for parameter of energy usage. By default
		/// it never stops until patience runs out.
		/// </summary>
		energy_t energy_early_stop_value = 0;

		/// <summary>
		/// Minimum expected value in the dataset.
		/// </summary>
		weight_value_t expected_value_min_value;

		/// <summary>
		/// Maximum expected value in the dataset.
		/// </summary>
		weight_value_t expected_value_max_value;

		/// <summary>
		/// Sets the starting generation value for Cartesian Genetic Programming (CGP) configuration.
		/// </summary>
		/// <param name="value">The starting generation value to set.</param>
		/// <returns>A reference to the modified CGPConfiguration object.</returns>
		CGPConfiguration& start_generation(decltype(start_generation_value) value);

		/// <summary>
		/// Sets the starting run value for Cartesian Genetic Programming (CGP) configuration.
		/// </summary>
		/// <param name="value">The starting run value to set.</param>
		/// <returns>A reference to the modified CGPConfiguration object.</returns>
		CGPConfiguration& start_run(decltype(start_run_value) value);
	public:
		CGPConfiguration();
		CGPConfiguration(const std::vector<std::string>& arguments);

		/// <summary>
		/// Sets configuration parameters according to given command line arguments.
		/// </summary>
		void set_from_arguments(const std::vector<std::string>& arguments);

		/// <summary>
		/// Gets the input arity of functions.
		/// </summary>
		inline decltype(function_input_arity_value) function_input_arity() const;

		/// <summary>
		/// Gets the output arity of functions.
		/// </summary>
		inline decltype(function_output_arity_value) function_output_arity() const;

		/// <summary>
		/// Gets the number of output pins in the CGP configuration.
		/// </summary>
		inline decltype(output_count_val) output_count() const;

		/// <summary>
		/// Gets the number of input pins in the CGP configuration.
		/// </summary>
		inline decltype(input_count_val) input_count() const;

		/// <summary>
		/// Gets the maximum population size in the CGP algorithm.
		/// </summary>
		inline decltype(population_max_value) population_max() const;

		/// <summary>
		/// Gets the maximum mutation value in the CGP algorithm.
		/// </summary>
		inline decltype(mutation_max_value) mutation_max() const;

		/// <summary>
		/// Gets maximum number of genes that can be mutated.
		/// </summary>
		decltype(max_genes_to_mutate_value) max_genes_to_mutate() const;

		/// <summary>
		/// Gets the number of rows in the CGP grid.
		/// </summary>
		inline decltype(row_count_value) row_count() const;

		/// <summary>
		/// Gets the number of columns in the CGP grid.
		/// </summary>
		inline decltype(col_count_value) col_count() const;

		/// <summary>
		/// Gets the look-back parameter in the CGP algorithm.
		/// </summary>
		inline decltype(look_back_parameter_value) look_back_parameter() const;

		/// <summary>
		/// Gets the maximum number of generations in the CGP algorithm.
		/// </summary>
		inline decltype(generation_count_value) generation_count() const;

		/// <summary>
		/// Gets the number of runs in the CGP algorithm.
		/// </summary>
		inline decltype(number_of_runs_value) number_of_runs() const;

		/// <summary>
		/// Gets the number of functions in the CGP algorithm.
		/// </summary>
		inline decltype(function_count_value) function_count() const;

		/// <summary>
		/// Gets the log frequency in the CGP algorithm.
		/// </summary>
		inline decltype(periodic_log_frequency_value) periodic_log_frequency() const;

		/// <summary>
		/// Gets a file path in which input data are located.
		/// </summary>
		inline decltype(input_file_value) input_file() const;

		/// <summary>
		/// Gets a file path in which output data will be stored.
		/// </summary>
		inline decltype(output_file_value) output_file() const;

		/// <summary>
		/// Gets the path where CGP statistics will be saved.
		/// </summary>
		inline decltype(cgp_statistics_file_value) cgp_statistics_file() const;

		/// <summary>
		/// Calculates the size of the pin map based on row and column counts.
		/// </summary>
		uint16_t pin_map_size() const;

		/// <summary>
		/// Calculates the size of the chromosome blocks.
		/// </summary>
		inline size_t blocks_chromosome_size() const;

		/// <summary>
		/// Calculates the total size of the chromosome.
		/// </summary>
		inline size_t chromosome_size() const;

		/// <summary>
		/// Gets array of energy costs for various operations.
		/// </summary>
		decltype(function_costs_value) function_costs() const;

		/// <summary>
		/// Get Mean Squared Error threshold after optimisation is focused on minimising energy.
		/// </summary>
		inline decltype(mse_threshold_value) mse_threshold() const;

		/// <summary>
		/// Get dataset size of the CGP.
		/// </summary>
		inline decltype(dataset_size_value) dataset_size() const;

		/// <summary>
		/// Get start generation number.
		/// </summary>
		decltype(start_generation_value) start_generation() const;

		/// <summary>
		/// Get start runnumber.
		/// </summary>
		decltype(start_run_value) start_run() const;

		/// <summary>
		/// Get chromosome in string form.
		/// </summary>
		decltype(starting_solution_value) starting_solution() const;

		/// <summary>
		/// Get value indicating after how many generations CGP will come to stop.
		/// </summary>
		decltype(patience_value) patience() const;

		/// <summary>
		/// Get value indicating stop condition for parameter of approximation error. Defaultly
		/// that stop condition is delegated to energy parameter.
		/// </summary>
		decltype(mse_early_stop_value) mse_early_stop() const;

		/// <summary>
		/// Get value indicating stop condition for parameter of energy usage. By default
		/// it never stops until patience runs out.
		/// </summary>
		decltype(energy_early_stop_value) energy_early_stop() const;

		/// <summary>
		/// Get the minimum expected value in the dataset.
		/// </summary>
		decltype(expected_value_min_value) expected_value_min() const;

		/// <summary>
		/// Get the maximum expected value in the dataset.
		/// </summary>
		decltype(expected_value_max_value) expected_value_max() const;

		/// <summary>
		/// Sets the input arity of functions in the CGP configuration.
		/// </summary>
		CGPConfiguration& function_input_arity(decltype(function_input_arity_value));

		/// <summary>
		/// Sets the output arity of functions in the CGP configuration.
		/// </summary>
		CGPConfiguration& function_output_arity(decltype(function_output_arity_value));

		/// <summary>
		/// Sets the number of output pins in the CGP configuration.
		/// </summary>
		CGPConfiguration& output_count(decltype(output_count_val));

		/// <summary>
		/// Sets the number of input pins in the CGP configuration.
		/// </summary>
		CGPConfiguration& input_count(decltype(input_count_val));

		/// <summary>
		/// Sets the maximum population size in the CGP algorithm.
		/// </summary>
		CGPConfiguration& population_max(decltype(population_max_value));

		/// <summary>
		/// Sets the maximum mutation value in the CGP algorithm.
		/// </summary>
		CGPConfiguration& mutation_max(decltype(mutation_max_value));

		/// <summary>
		/// Sets the number of rows in the CGP grid.
		/// </summary>
		CGPConfiguration& row_count(decltype(row_count_value));

		/// <summary>
		/// Sets the number of columns in the CGP grid.
		/// </summary>
		CGPConfiguration& col_count(decltype(col_count_value));

		/// <summary>
		/// Sets the look-back parameter in the CGP algorithm.
		/// </summary>
		CGPConfiguration& look_back_parameter(decltype(look_back_parameter_value));

		/// <summary>
		/// Sets the maximum number of generations in the CGP algorithm.
		/// </summary>
		CGPConfiguration& generation_count(decltype(generation_count_value));

		/// <summary>
		/// Sets the number of runs in the CGP algorithm.
		/// </summary>
		CGPConfiguration& number_of_runs(decltype(number_of_runs_value));

		/// <summary>
		/// Sets the number of functions in the CGP algorithm.
		/// </summary>
		CGPConfiguration& function_count(decltype(function_count_value));

		/// <summary>
		/// Sets the log frequency in the CGP algorithm.
		/// </summary>
		CGPConfiguration& periodic_log_frequency(decltype(periodic_log_frequency_value));

		/// <summary>
		/// Sets array of energy costs for various operations in the CGP algorithm.
		/// </summary>
		CGPConfiguration& function_costs(decltype(function_costs_value));

		/// <summary>
		/// Sets the file path in which input data are located for the CGP algorithm.
		/// </summary>
		CGPConfiguration& input_file(decltype(input_file_value));

		/// <summary>
		/// Sets the file path in which output data will be stored for the CGP algorithm.
		/// </summary>
		CGPConfiguration& output_file(decltype(input_file_value));

		/// <summary>
		/// Sets the path where CGP metadata will be saved.
		/// </summary>
		CGPConfiguration& cgp_statistics_file(decltype(cgp_statistics_file_value));

		/// <summary>
		/// Sets the Mean Squared Error threshold after optimization is focused on minimizing energy.
		/// </summary>
		CGPConfiguration& mse_threshold(decltype(mse_threshold_value));

		/// <summary>
		/// Sets the dataset size of the CGP.
		/// </summary>
		CGPConfiguration& dataset_size(decltype(dataset_size_value));

		/// <summary>
		/// Sets the starting chromosome for the CGP.
		/// </summary>
		CGPConfiguration& starting_solution(decltype(starting_solution_value) value);

		/// <summary>
		/// Sets the value indicating after how many generations CGP will come to a stop.
		/// </summary>
		CGPConfiguration& patience(decltype(patience_value) value);

		/// <summary>
		/// Sets the value indicating the stop condition for the parameter of approximation error. 
		/// By default, that stop condition is delegated to the energy parameter.
		/// </summary>
		CGPConfiguration& mse_early_stop(decltype(mse_early_stop_value) value);

		/// <summary>
		/// Sets the value indicating the stop condition for the parameter of energy usage. 
		/// By default, it never stops until patience runs out.
		/// </summary>
		CGPConfiguration& energy_early_stop(decltype(energy_early_stop_value) value);

		/// <summary>
		/// Sets the minimum expected value in the dataset.
		/// </summary>
		/// <param name="value">The minimum expected value to set.</param>
		CGPConfiguration& expected_value_min(decltype(expected_value_min_value) value);

		/// <summary>
		/// Sets the maximum expected value in the dataset.
		/// </summary>
		/// <param name="value">The maximum expected value to set.</param>
		CGPConfiguration& expected_value_max(decltype(expected_value_max_value) value);

		/// <summary>
		/// Saves the configuration to a file.
		/// </summary>
		virtual void dump(std::ostream& out) const;

		/// <summary>
		/// Loads the configuration from a file.
		/// </summary>
		virtual std::map<std::string, std::string> load(std::istream& in, const std::vector<std::string>& arguments = {});

	};

	std::string error_to_string(CGPConfiguration::error_t value);
	std::string energy_to_string(CGPConfiguration::energy_t value);
	std::string delay_to_string(CGPConfiguration::delay_t value);
	std::string depth_to_string(CGPConfiguration::depth_t value);
	std::string gate_count_to_string(CGPConfiguration::gate_count_t value);
	std::string weight_to_string(CGPConfiguration::weight_value_t value);
}