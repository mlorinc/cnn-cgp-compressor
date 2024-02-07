#pragma once
#include <cstdint>

namespace cgp {
	/// <summary>
	/// Configuration struct for Cartesian Genetic Programming (CGP).
	/// </summary>
	struct CGPConfiguration
	{
	private:
		// Private member variables representing default configuration values

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
		uint16_t mutation_max_value = 3;

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
		uint8_t function_count_value = 4;

		/// <summary>
		/// Default value for the log frequency in the CGP algorithm.
		/// </summary>
		uint32_t periodic_log_frequency_value = static_cast<uint32_t>(generation_count_value / 2.0);

		/// <summary>
		/// Default value for enabling periodic logging in the CGP algorithm.
		/// </summary>
		bool periodic_log_value = true;

	public:
		// Type definition for the gene.
		using gene_t = uint16_t;

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
		/// Gets whether periodic logging is enabled in the CGP algorithm.
		/// </summary>
		decltype(periodic_log_value) periodic_log() const;

		// Additional public methods for calculating derived values

		/// <summary>
		/// Calculates the size of the pin map based on row and column counts.
		/// </summary>
		uint16_t pin_map_size() const;

		/// <summary>
		/// Calculates the size of the chromosome blocks.
		/// </summary>
		size_t blocks_chromosome_size() const;

		/// <summary>
		/// Calculates the total size of the chromosome.
		/// </summary>
		size_t chromosome_size() const;

		// Public setter methods for modifying configuration parameters

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
		/// Sets whether periodic logging is enabled in the CGP algorithm.
		/// </summary>
		CGPConfiguration& periodic_log(decltype(periodic_log_value));
	};
}