// Copyright 2024 Marián Lorinc
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     LICENSE.txt file
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Configuration.h: Class declaration for holding all neccesary information and configuration properties for CGP algorithm.

//#define __VIRTUAL_SELECTOR 1
//#define __ABSOLUTE_ERROR_METRIC 1
//#define __SINGLE_MULTIPLEX 1
//#define __MULTI_MULTIPLEX 1
//#define __SINGLE_OUTPUT_ARITY 1

#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <map>


namespace cgp {
	enum CGPOperator {
		REVERSE_MAX_A = 0,
		ADD = 1,
		SUB = 2,
		MUL = 3,
		NEG = 4,
		REVERSE_MIN_B = 5,
		QUARTER = 6,
		HALF = 7,
		BIT_AND = 8,
		BIT_OR = 9,
		BIT_XOR = 10,
		BIT_NEG = 11,
		DOUBLE = 12,
		BIT_INC = 13,
		BIT_DEC = 14,
		R_SHIFT_3 = 15,
		R_SHIFT_4 = 16,
		R_SHIFT_5 = 17,
		L_SHIFT_2 = 18,
		L_SHIFT_3 = 19,
		L_SHIFT_4 = 20,
		L_SHIFT_5 = 21,
		ONE_CONST = 22,
		MINUS_ONE_CONST = 23,
		ZERO_CONST = 24,
		EXPECTED_VALUE_MIN = 25,
		EXPECTED_VALUE_MAX = 26,
		MUX = 27,
		DEMUX = 28,
		ID = 100
	};

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
		using dimension_t = int;

		/// <summary>
		/// Type alias for error values.
		/// </summary>
#ifndef __ERROR_T
		using error_t = uint64_t;
#else
		using error_t = __ERROR_T;
		#warning "Setting error_t to " #__ERROR_T "."
#endif
		/// <summary>
		/// Type alias for energy values which are primary use for computation.
		/// </summary>
		using quantized_energy_t = uint64_t;

		/// <summary>
		/// Type alias for real energy values.
		/// </summary>
		using energy_t = double;

		/// <summary>
		/// Type alias for delay values, represented as double-precision floating-point numbers.
		/// </summary>
		using quantized_delay_t = uint64_t;

		/// <summary>
		/// Type alias for real delay values.
		/// </summary>
		using delay_t = double;

		/// <summary>
		/// Type alias for depth values, represented as dimension_t.
		/// </summary>
		using depth_t = int;

		/// <summary>
		/// Type alias for gate count values, represented as size_t.
		/// </summary>
		using gate_count_t = int;

		/// <summary>
		/// Type alias for area values, represented as double-precision floating-point numbers.
		/// </summary>
		using area_t = double;

		/// <summary>
		/// Type alias for gate parameters, represented as a tuple of energy and delay values.
		/// </summary>
		using gate_parameters_t = std::tuple<quantized_energy_t, energy_t, area_t, quantized_delay_t, delay_t>;

		/// <summary>
		/// Type alias for gene values, represented as unsigned 16-bit integers.
		/// </summary>
		using gene_t = int;

		/// <summary>
		/// Value representing NaN for error_t.
		/// </summary>
		static constexpr error_t error_nan = (std::numeric_limits<error_t>::has_infinity) ? (std::numeric_limits<error_t>::infinity()) : (std::numeric_limits<error_t>::max());

		/// <summary>
		/// Value representing NaN for quantized_energy_t.
		/// </summary>
		static constexpr quantized_energy_t quantized_energy_nan = (std::numeric_limits<quantized_energy_t>::has_infinity) ? (std::numeric_limits<quantized_energy_t>::infinity()) : (std::numeric_limits<quantized_energy_t>::max());

		/// <summary>
		/// Value representing NaN for energy_t.
		/// </summary>
		static constexpr energy_t energy_nan = (std::numeric_limits<energy_t>::has_infinity) ? (std::numeric_limits<energy_t>::infinity()) : (std::numeric_limits<energy_t>::max());

		/// <summary>
		/// Value representing NaN for quantized_delay_t.
		/// </summary>
		static constexpr quantized_delay_t quantized_delay_nan = (std::numeric_limits<quantized_delay_t>::has_infinity) ? (std::numeric_limits<quantized_delay_t>::infinity()) : (std::numeric_limits<quantized_delay_t>::max());

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
		/// Value representing NaN for area_t.
		/// </summary>
		static constexpr area_t area_nan = (std::numeric_limits<area_t>::has_infinity) ? (std::numeric_limits<area_t>::infinity()) : (std::numeric_limits<area_t>::max());

		/// <summary>
		/// The biggest possible multiplexer bit variant.
		/// </summary>
		static constexpr int max_hardware_multiplexer_bit_variant = 4;

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
		static quantized_energy_t get_quantized_energy_parameter(const gate_parameters_t& params);

		/// <summary>
		/// Get the delay parameter from the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <returns>The delay parameter.</returns>
		static quantized_delay_t get_quantized_delay_parameter(const gate_parameters_t& params);

		/// <summary>
		/// Get the real energy parameter from the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <returns>The energy parameter.</returns>
		static energy_t get_energy_parameter(const gate_parameters_t& params);

		/// <summary>
		/// Get the real delay parameter from the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <returns>The delay parameter.</returns>
		static delay_t get_delay_parameter(const gate_parameters_t& params);

		/// <summary>
		/// Get the area parameter from the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <returns>The area parameter.</returns>
		static area_t get_area_parameter(const gate_parameters_t& params);

		/// <summary>
		/// Set the energy parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="energy">The energy parameter to set.</param>
		static void set_quantized_energy_parameter(gate_parameters_t& params, quantized_energy_t energy);

		/// <summary>
		/// Set the delay parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="delay">The delay parameter to set.</param>
		static void set_quantized_delay_parameter(gate_parameters_t& params, quantized_delay_t delay);

		/// <summary>
		/// Set the real energy parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="energy">The energy parameter to set.</param>
		static void set_energy_parameter(gate_parameters_t& params, energy_t energy);

		/// <summary>
		/// Set the real delay parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="delay">The delay parameter to set.</param>
		static void set_delay_parameter(gate_parameters_t& params, delay_t delay);

		/// <summary>
		/// Set the area parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="delay">The area parameter to set.</param>
		static void set_area_parameter(gate_parameters_t& params, area_t area);

		/// <summary>
		/// Set the energy parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="energy">The energy parameter to set.</param>
		static void set_energy_parameter(gate_parameters_t& params, const std::string &energy);

		/// <summary>
		/// Set the real energy parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="energy">The energy parameter to set.</param>
		static void set_quantized_energy_parameter(gate_parameters_t& params, const std::string& energy);

		/// <summary>
		/// Set the delay parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="delay">The delay parameter to set.</param>
		static void set_quantized_delay_parameter(gate_parameters_t& params, const std::string &delay);

		/// <summary>
		/// Set the real delay parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="energy">The energy parameter to set.</param>
		static void set_delay_parameter(gate_parameters_t& params, const std::string& delay);

		/// <summary>
		/// Set the area parameter in the given gate parameters.
		/// </summary>
		/// <param name="params">The gate parameters tuple.</param>
		/// <param name="delay">The area parameter to set.</param>
		static void set_area_parameter(gate_parameters_t& params, const std::string &area);

		/// <summary>
		/// String representation of error_nan.
		/// </summary>
		static const std::string error_nan_string;

		/// <summary>
		/// String representation of quantized_energy_nan.
		/// </summary>
		static const std::string quantized_energy_nan_string;

		/// <summary>
		/// String representation of energy_nan.
		/// </summary>
		static const std::string energy_nan_string;

		/// <summary>
		/// String representation of quantized_delay_nan.
		/// </summary>
		static const std::string quantized_delay_nan_string;

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

		/// <summary>
		/// String representation of gate_count_nan.
		/// </summary>
		static const std::string area_nan_string;

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

		static const std::string GATE_PARAMETERS_FILE_LONG;
		static const std::string GATE_PARAMETERS_FILE_SHORT;

		static const std::string TRAIN_WEIGHTS_FILE_LONG;

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

		static const std::string DELAY_EARLY_STOP_LONG;

		static const std::string DEPTH_EARLY_STOP_LONG;

		static const std::string GATE_COUNT_EARLY_STOP_LONG;

		static const std::string MSE_CHROMOSOME_LOGGING_THRESHOLD_LONG;

		static const std::string LEARNING_RATE_FILE_LONG;

		static const std::string LEARNING_RATE_LONG;

#ifndef CNN_FP32_WEIGHTS
		/// <summary>
		/// Type alias for inferred weight used internally by the CGP, represented as a signed integer
		/// to prevent overflow when multiplying.
		/// </summary>
		using weight_value_t = int8_t;

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
		using weight_input_t = weight_value_t*;
		using weight_output_t = weight_value_t*;
	private:
		/// <summary>
		/// Default value for the input arity of functions.
		/// </summary>
		int function_input_arity_value = 2;

		/// <summary>
		/// Default value for the output arity of functions.
		/// </summary>
		int function_output_arity_value = 1;

		/// <summary>
		/// Default value for the number of output pins in the CGP.
		/// </summary>
		int output_count_val = 1;

		/// <summary>
		/// Default value for the number of input pins in the CGP.
		/// </summary>
		int input_count_val = 2;

		/// <summary>
		/// Default value for the maximum population size in the CGP algorithm.
		/// </summary>
		int population_max_value = 0;

		/// <summary>
		/// Default value for the maximum mutation value in the CGP algorithm.
		/// </summary>
		float mutation_max_value = 0.15f;

		/// <summary>
		/// Variable limiting maximum number of genes that can be mutated.
		/// </summary>
		int max_genes_to_mutate_value;

		/// <summary>
		/// Default value for the number of rows in the CGP grid.
		/// </summary>
		int row_count_value = 5;

		/// <summary>
		/// Default value for the number of columns in the CGP grid.
		/// </summary>
		int col_count_value = 5;

		/// <summary>
		/// Default value for the look-back parameter in the CGP algorithm.
		/// </summary>
		int look_back_parameter_value = 1;

		/// <summary>
		/// Default value for the maximum number of generations in the CGP algorithm.
		/// </summary>
		uint64_t generation_count_value = std::numeric_limits<uint64_t>::max();

		/// <summary>
		/// Default value for the number of runs in the CGP algorithm.
		/// </summary>
		int number_of_runs_value = 30;

		/// <summary>
		/// Default value for the number of functions in the CGP algorithm.
		/// </summary>
		int function_count_value = 30;

		/// <summary>
		/// Default value for the log frequency in the CGP algorithm.
		/// </summary>
		size_t periodic_log_frequency_value = 100000;

		/// <summary>
		/// Array of energy costs for various operations.
		/// </summary>
		std::unique_ptr<gate_parameters_t[]> function_costs_value;

		/// <summary>
		/// A path to a file with input data.
		/// </summary>
		std::string input_file_value = "-";

		/// <summary>
		/// A path to a file to create which contains output of the CGP process.
		/// </summary>
		std::string output_file_value = "-";

		/// <summary>
		/// A path where CGP statistics will be saved.
		/// </summary>
		std::string cgp_statistics_file_value = "+";

		/// <summary>
		/// A path where CGP learning will be saved.
		/// </summary>
		std::string learning_rate_file_value = "";

		/// <summary>
		/// A path where gate parameters are stored.
		/// </summary>
		std::string gate_parameters_input_file_value = "";

		/// <summary>
		/// A path where trained weights parameters will be stored.
		/// </summary>
		std::string train_weights_file_value = "#train_weights_file_value";

		/// <summary>
		/// Mean Squared Error threshold after optimisation is focused on minimising energy.
		/// </summary>
		error_t mse_threshold_value = 5;

		/// <summary>
		/// The CGP dataset size.
		/// </summary>
		int dataset_size_value = 1;

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
		int patience_value = 125000;

		/// <summary>
		/// Value indicating stop condition for parameter of approximation error. By default
		/// perfect solution is assumed.
		/// </summary>
		error_t mse_early_stop_value = 0;

		/// <summary>
		/// Value indicating stop condition for parameter of energy usage. By default
		/// perfect solution is assumed.
		/// </summary>
		energy_t energy_early_stop_value = 0;

		/// <summary>
		/// Value indicating stop condition for parameter of delay. By default
		/// it is ignored.
		/// </summary>
		delay_t delay_early_stop_value = delay_nan;

		/// <summary>
		/// Value indicating stop condition for parameter of depth. By default
		/// it is ignored.
		/// </summary>
		depth_t depth_early_stop_value = depth_nan;

		/// <summary>
		/// Value indicating stop condition for parameter of gate count. By default
		/// it is ignored.
		/// </summary>
		gate_count_t gate_count_early_stop_value = gate_count_nan;

		/// <summary>
		/// Logging threshold when chromosomes with error less than value 
		/// will start being printed in CSV logs as serialized strings.
		/// By default every chromosome is serialized and logged.
		/// </summary>
		error_t mse_chromosome_logging_threshold_value = error_nan;

		/// <summary>
		/// Minimum expected value in the dataset.
		/// </summary>
		weight_value_t expected_value_min_value = std::numeric_limits<weight_actual_value_t>::min();

		/// <summary>
		/// Maximum expected value in the dataset.
		/// </summary>
		weight_value_t expected_value_max_value = std::numeric_limits<weight_actual_value_t>::max();

		/// <summary>
		/// Maximum supported multiplexer bit variant.
		/// </summary>
		int max_multiplexer_bit_variant_value = 0;

		/// <summary>
		/// Learning rate threshold after which the training process is terminated.
		/// </summary>
		double learning_rate_value = 0.05;

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
		/// Learning rate threshold after which the training process is terminated.
		/// </summary>
		decltype(learning_rate_value) learning_rate() const;

		/// <summary>
		/// A path where CGP learning will be saved.
		/// </summary>
		decltype(learning_rate_file_value) learning_rate_file() const;

		/// <summary>
		/// Gets the input arity of functions.
		/// </summary>
		decltype(function_input_arity_value) function_input_arity() const;

		/// <summary>
		/// Gets the output arity of functions.
		/// </summary>
		decltype(function_output_arity_value) function_output_arity() const;

		/// <summary>
		/// Gets the number of output pins in the CGP configuration.
		/// </summary>
		decltype(output_count_val) output_count() const;

		/// <summary>
		/// Gets the number of input pins in the CGP configuration.
		/// </summary>
		decltype(input_count_val) input_count() const;

		/// <summary>
		/// Gets the maximum population size in the CGP algorithm.
		/// </summary>
		decltype(population_max_value) population_max() const;

		/// <summary>
		/// Gets the maximum mutation value in the CGP algorithm.
		/// </summary>
		decltype(mutation_max_value) mutation_max() const;

		/// <summary>
		/// Gets maximum number of genes that can be mutated.
		/// </summary>
		decltype(max_genes_to_mutate_value) max_genes_to_mutate() const;

		/// <summary>
		/// Gets the number of rows in the CGP grid.
		/// </summary>
		decltype(row_count_value) row_count() const;

		/// <summary>
		/// Gets the number of columns in the CGP grid.
		/// </summary>
		decltype(col_count_value) col_count() const;

		/// <summary>
		/// Gets the look-back parameter in the CGP algorithm.
		/// </summary>
		decltype(look_back_parameter_value) look_back_parameter() const;

		/// <summary>
		/// Gets the maximum number of generations in the CGP algorithm.
		/// </summary>
		decltype(generation_count_value) generation_count() const;

		/// <summary>
		/// Gets the number of runs in the CGP algorithm.
		/// </summary>
		decltype(number_of_runs_value) number_of_runs() const;

		/// <summary>
		/// Gets the number of functions in the CGP algorithm.
		/// </summary>
		decltype(function_count_value) function_count() const;

		/// <summary>
		/// Gets the log frequency in the CGP algorithm.
		/// </summary>
		decltype(periodic_log_frequency_value) periodic_log_frequency() const;

		/// <summary>
		/// Gets a file path in which input data are located.
		/// </summary>
		decltype(input_file_value) input_file() const;

		/// <summary>
		/// Gets a file path in which output data will be stored.
		/// </summary>
		decltype(output_file_value) output_file() const;

		/// <summary>
		/// Gets the path where CGP statistics will be saved.
		/// </summary>
		decltype(cgp_statistics_file_value) cgp_statistics_file() const;

		/// <summary>
		/// Calculates the size of the pin map based on row and column counts.
		/// </summary>
		int pin_map_size() const;

		/// <summary>
		/// Calculates the start of pin section.
		/// </summary>
		int pin_map_pins_start() const;

		/// <summary>
		/// Calculates the start of input pin section.
		/// </summary>
		int pin_map_input_start(int selector) const;

		/// <summary>
		/// Calculates the start of output pin section.
		/// </summary>
		int pin_map_output_start() const;

		/// <summary>
		/// Calculates the size of a chromosome block.
		/// </summary>
		int block_chromosome_size() const;

		/// <summary>
		/// Calculates the size of the chromosome blocks.
		/// </summary>
		int blocks_chromosome_size() const;

		/// <summary>
		/// Calculates the total size of the chromosome.
		/// </summary>
		int chromosome_size() const;

		/// <summary>
		/// Gets array of energy costs for various operations.
		/// </summary>
		const decltype(function_costs_value)& function_costs() const;

		/// <summary>
		/// Get Mean Squared Error threshold after optimisation is focused on minimising energy.
		/// </summary>
		decltype(mse_threshold_value) mse_threshold() const;

		/// <summary>
		/// Get dataset size of the CGP.
		/// </summary>
		decltype(dataset_size_value) dataset_size() const;

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
		/// Get value indicating stop condition for parameter of approximation error. By default
		/// perfect solution is assumed.
		/// </summary>
		decltype(mse_early_stop_value) mse_early_stop() const;

		/// <summary>
		/// Get value indicating stop condition for parameter of energy usage. By default
		/// perfect solution is assumed.
		/// </summary>
		decltype(energy_early_stop_value) energy_early_stop() const;

		/// <summary>
		/// Get value indicating stop condition for parameter of delay. By default
		/// it is ignored
		/// </summary>
		decltype(delay_early_stop_value) delay_early_stop() const;

		/// <summary>
		/// Get value indicating stop condition for parameter of depth. By default
		/// it is ignored
		/// </summary>
		decltype(depth_early_stop_value) depth_early_stop() const;

		/// <summary>
		/// Get value indicating stop condition for parameter of gate count. By default
		/// it is ignored
		/// </summary>
		decltype(gate_count_early_stop_value) gate_count_early_stop() const;

		/// <summary>
		/// Get the minimum expected value in the dataset.
		/// </summary>
		decltype(expected_value_min_value) expected_value_min() const;

		/// <summary>
		/// Get the maximum expected value in the dataset.
		/// </summary>
		decltype(expected_value_max_value) expected_value_max() const;

		/// <summary>
		/// A path where gate parameters are stored.
		/// </summary>
		decltype(gate_parameters_input_file_value) gate_parameters_input_file() const;

		/// <summary>
		/// A path where trained weights parameters will be stored.
		/// </summary>
		decltype(train_weights_file_value) train_weights_file() const;

		/// <summary>
		/// Maximum supported multiplexer bit variant.
		/// </summary>
		decltype(max_multiplexer_bit_variant_value) max_multiplexer_bit_variant() const;

		/// <summary>
		/// Get logging threshold when chromosomes with error less than value 
		/// will start being printed in CSV logs as serialized strings.
		/// By default every chromosome is serialized and logged.
		/// </summary>
		decltype(mse_chromosome_logging_threshold_value) mse_chromosome_logging_threshold() const;

		/// <summary>
		/// Sets learning rate threshold after which the training process is terminated.
		/// </summary>
		CGPConfiguration& learning_rate(decltype(learning_rate_value) value);

		/// <summary>
		/// Sets a path where CGP learning will be saved.
		/// </summary>
		CGPConfiguration& learning_rate_file(decltype(learning_rate_file_value));

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
		CGPConfiguration& function_costs(decltype(function_costs_value)&&);

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
		/// Sets value indicating stop condition for parameter of approximation error. By default
		/// perfect solution is assumed.
		/// </summary>
		CGPConfiguration& mse_early_stop(decltype(mse_early_stop_value) value);

		/// <summary>
		/// Sets value indicating stop condition for parameter of energy usage. By default
		/// perfect solution is assumed.
		/// </summary>
		CGPConfiguration& energy_early_stop(decltype(energy_early_stop_value) value);

		/// <summary>
		/// Sets value indicating stop condition for parameter of delay. By default
		/// perfect solution is assumed.
		/// </summary>
		CGPConfiguration& delay_early_stop(decltype(delay_early_stop_value) value);

		/// <summary>
		/// Sets value indicating stop condition for parameter of depth. By default
		/// perfect solution is assumed.
		/// </summary>
		CGPConfiguration& depth_early_stop(decltype(depth_early_stop_value) value);

		/// <summary>
		/// Sets value indicating stop condition for parameter of gate count. By default
		/// perfect solution is assumed.
		/// </summary>
		CGPConfiguration& gate_count_early_stop(decltype(gate_count_early_stop_value) value);

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
		/// Sets a path where gate parameters are stored.
		/// </summary>
		CGPConfiguration& gate_parameters_input_file(decltype(gate_parameters_input_file_value) value);

		/// <summary>
		/// A path where trained weights parameters will be stored.
		/// </summary>
		CGPConfiguration& train_weights_file(decltype(train_weights_file_value) value);

		/// <summary>
		/// Set logging threshold when chromosomes with error less than value 
		/// will start being printed in CSV logs as serialized strings.
		/// By default every chromosome is serialized and logged.
		/// </summary>
		CGPConfiguration& mse_chromosome_logging_threshold(decltype(mse_chromosome_logging_threshold_value) value);

		/// <summary>
		/// Saves the configuration to a file.
		/// </summary>
		virtual void dump(std::ostream& out) const;

		/// <summary>
		/// Loads the configuration from a file.
		/// </summary>
		virtual std::map<std::string, std::string> load(std::istream& in, const std::vector<std::string>& arguments = {});

	};

	const std::string error_to_string(CGPConfiguration::error_t value);
	const std::string quantized_energy_to_string(CGPConfiguration::quantized_energy_t value);
	const std::string energy_to_string(CGPConfiguration::energy_t value);
	const std::string quantized_delay_to_string(CGPConfiguration::quantized_delay_t value);
	const std::string delay_to_string(CGPConfiguration::delay_t value);
	const std::string depth_to_string(CGPConfiguration::depth_t value);
	const std::string gate_count_to_string(CGPConfiguration::gate_count_t value);
	const std::string weight_to_string(CGPConfiguration::weight_value_t value);
	const std::string area_to_string(CGPConfiguration::area_t value);
	CGPConfiguration::error_t string_to_error(const std::string& value);
	CGPConfiguration::quantized_energy_t string_to_quantized_energy(const std::string& value);
	CGPConfiguration::energy_t string_to_energy(const std::string& value);
	CGPConfiguration::area_t string_to_area(const std::string& value);
	CGPConfiguration::quantized_delay_t string_to_quantized_delay(const std::string& value);
	CGPConfiguration::delay_t string_to_delay(const std::string& value);
	CGPConfiguration::depth_t string_to_depth(const std::string& value);
	CGPConfiguration::gate_count_t string_to_gate_count(const std::string& value);
}