#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <map>

//#ifndef CNN_FP32_WEIGHTS
//constexpr int function_count = 31;
//using weight_repr_value_t = int;
//#else
//constexpr int function_count = 13;
//using weight_repr_value_t = double;
//#endif // !CNN_FP32_WEIGHTS

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
	/// Configuration struct for Cartesian Genetic Programming (CGP).
	/// </summary>
	struct CGPConfiguration
	{
	public:
		using gate_parameters_t = std::tuple<double, double>;
		using gene_t = uint16_t;
		static gate_parameters_t get_default_gate_parameters();
		static double get_energy_parameter(const gate_parameters_t &params);
		static double get_delay_parameter(const gate_parameters_t& params);
		static void set_energy_parameter(gate_parameters_t& params, double energy);
		static void set_delay_parameter(gate_parameters_t& params, double delay);

#ifndef CNN_FP32_WEIGHTS
		// Type definition for inferred weight.
		using weight_value_t = int16_t;
		// Type definition for hardware inferred weight.
		using weight_actual_value_t = int8_t;
		// Type definition for serialization.
		using weight_repr_value_t = int;
		static const weight_value_t invalid_value = std::numeric_limits<weight_value_t>::max();
		static const weight_value_t no_care_value = std::numeric_limits<weight_value_t>::min();
#else
		// Type definition for inferred weight.
		using weight_value_t = double;
		// Type definition for inferred weight.
		using weight_actual_value_t = double;
		using weight_repr_value_t = double;
		static const weight_value_t invalid_value = std::numeric_limits<weight_value_t>::infinity();
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
		uint16_t row_count_value = 5;

		/// <summary>
		/// Default value for the number of columns in the CGP grid.
		/// </summary>
		uint16_t col_count_value = 5;

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
		double mse_threshold_value = 5;

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
		double mse_early_stop_value = -1;

		/// <summary>
		/// Value indicating stop condition for parameter of energy usage. By default
		/// it never stops until patience runs out.
		/// </summary>
		double energy_early_stop_value = 0;

		CGPConfiguration& start_generation(decltype(start_generation_value) value);
		CGPConfiguration& start_run(decltype(start_run_value) value);
	public:
		CGPConfiguration();
		CGPConfiguration(const std::vector<std::string>& arguments);

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
		/// Sets the input arity of functions.
		/// </summary>
		CGPConfiguration& function_input_arity(decltype(function_input_arity_value));

		/// <summary>
		/// Sets the output arity of functions.
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
		/// Sets array of energy costs for various operations.
		/// </summary>
		CGPConfiguration& function_costs(decltype(function_costs_value));

		/// <summary>
		/// Sets a file path in which input data are located.
		/// </summary>
		CGPConfiguration& input_file(decltype(input_file_value));

		/// <summary>
		/// Sets a file path in which output data will be stored.
		/// </summary>
		CGPConfiguration& output_file(decltype(input_file_value));

		/// <summary>
		/// Sets the path where CGP metadata will be saved.
		/// </summary>
		CGPConfiguration& cgp_statistics_file(decltype(cgp_statistics_file_value));

		/// <summary>
		/// Set Mean Squared Error threshold after optimisation is focused on minimising energy.
		/// </summary>
		CGPConfiguration& mse_threshold(decltype(mse_threshold_value));

		/// <summary>
		/// Set dataset size of the CGP.
		/// </summary>
		CGPConfiguration& dataset_size(decltype(dataset_size_value));

		/// <summary>
		/// Sets starting chromosome.
		/// </summary>
		CGPConfiguration& starting_solution(decltype(starting_solution_value) value);

		/// <summary>
		/// Set value indicating after how many generations CGP will come to stop.
		/// </summary>
		CGPConfiguration& patience(decltype(patience_value) value);

		/// <summary>
		/// Set value indicating stop condition for parameter of approximation error. Defaultly
		/// that stop condition is delegated to energy parameter.
		/// </summary>
		CGPConfiguration& mse_early_stop(decltype(mse_early_stop_value) value);

		/// <summary>
		/// Set value indicating stop condition for parameter of energy usage. By default
		/// it never stops until patience runs out.
		/// </summary>
		CGPConfiguration& energy_early_stop(decltype(energy_early_stop_value) value);

		/// <summary>
		/// Save configuration to file.
		/// </summary>
		virtual void dump(std::ostream &out) const;

		/// <summary>
		/// Load configuration from file.
		/// </summary>
		virtual std::map<std::string, std::string> load(std::istream& in, const std::vector<std::string>& arguments = {});
	};
}