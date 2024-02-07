#pragma once

#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "Configuration.h"

namespace cgp {
	/// <summary>
	/// Chromosome class representing an individual in Cartesian Genetic Programming (CGP).
	/// </summary>
	class Chromosome {
	private:
		using gene_t = CGPConfiguration::gene_t;

		// Reference to the CGP configuration used for chromosome setup.
		const CGPConfiguration& cgp_configuration;

		// Pointers to the start and end positions of the chromosome output.
		gene_t* output_start, * output_end;

		// Pointers to the start and end positions of the output pins in the pin map.
		double* output_pin_start, * output_pin_end;

		// Array containing tuples specifying the minimum and maximum pin indices for possible output connections base on look back parameter.
		const std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies;

		// Minimum expected value in the dataset.
		const double expected_value_min;

		// Maximum expected value in the dataset.
		const double expected_value_max;

		// Shared pointer to the chromosome array.
		std::shared_ptr<gene_t[]> chromosome;

		// Shared pointer to the pin map array.
		std::shared_ptr<double[]> pin_map;

		// Shared pointer to the input array.
		std::shared_ptr<double[]> input;

		// Flag indicating whether the chromosome needs evaluation.
		bool need_evaluation = true;

		// Private method to check if a given position in the chromosome represents a function.
		bool is_function(size_t position) const;

		// Private method to check if a given position in the chromosome represents an input.
		bool is_input(size_t position) const;

		// Private method to check if a given position in the chromosome represents an output.
		bool is_output(size_t position) const;

		// Private method for setting up the initial state of the chromosome.
		void setup();

	public:
		friend std::ostream& operator<<(std::ostream& os, const Chromosome& chromosome);

		/// <summary>
		/// Constructor for the Chromosome class.
		/// </summary>
		/// <param name="cgp_configuration">Reference to the CGP configuration.</param>
		/// <param name="minimum_output_indicies">Array containing tuples specifying the minimum and maximum pin indices for possible output connections base on look back parameter.</param>
		/// <param name="expected_value_min">Minimum expected value in the dataset.</param>
		/// <param name="expected_value_max">Maximum expected value in the dataset.</param>
		Chromosome(const CGPConfiguration& cgp_configuration, std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies, double expected_value_min, double expected_value_max);

		/// <summary>
		/// Copy constructor for the Chromosome class.
		/// </summary>
		/// <param name="that">Reference to the chromosome to be copied.</param>
		Chromosome(const Chromosome& that);

		/// <summary>
		/// Getter for the pointer to the chromosome outputs.
		/// </summary>
		/// <returns>Pointer to the chromosome outputs.</returns>
		gene_t* get_outputs() const;

		/// <summary>
		/// Getter for the pointer to the inputs of a specific block in the chromosome.
		/// </summary>
		/// <param name="row">Row index of the block.</param>
		/// <param name="column">Column index of the block.</param>
		/// <returns>Pointer to the block inputs.</returns>
		gene_t* get_block_inputs(size_t row, size_t column) const;

		/// <summary>
		/// Getter for the pointer to the function represented by a specific block in the chromosome.
		/// </summary>
		/// <param name="row">Row index of the block.</param>
		/// <param name="column">Column index of the block.</param>
		/// <returns>Pointer to the block function.</returns>
		gene_t* get_block_function(size_t row, size_t column) const;

		/// <summary>
		/// Getter for the shared pointer to the chromosome array.
		/// </summary>
		/// <returns>Shared pointer to the chromosome array.</returns>
		std::shared_ptr<gene_t[]> get_chromosome() const;

		/// <summary>
		/// Method to perform mutation on the chromosome.
		/// </summary>
		/// <returns>Shared pointer to the mutated chromosome.</returns>
		std::shared_ptr<Chromosome> mutate();

		/// <summary>
		/// Method to set the input for the chromosome.
		/// </summary>
		/// <param name="input">Shared pointer to the input array.</param>
		void set_input(std::shared_ptr<double[]> input);

		/// <summary>
		/// Method to evaluate the chromosome based on its inputs.
		/// </summary>
		void evaluate();

		/// <summary>
		/// Getter for the pointer to the beginning of the output array.
		/// </summary>
		/// <returns>Pointer to the beginning of the output array.</returns>
		double* begin_output();

		/// <summary>
		/// Getter for the pointer to the end of the output array.
		/// </summary>
		/// <returns>Pointer to the end of the output array.</returns>
		double* end_output();

		/// <summary>
		/// Convert the Chromosome to a string representation which can be used in cgpviewer.exe.
		/// </summary>
		/// <returns>The string representation of the Chromosome.</returns>
		std::string to_string() const;

		/// <summary>
		/// Calculate size of the gene.
		/// </summary>
		/// <returns>The size of gene.</returns>
		size_t get_serialized_chromosome_size() const;

		/// <summary>
		/// Copy assignment operator for the Chromosome class.
		/// </summary>
		/// <param name="that">Reference to the chromosome to be assigned.</param>
		/// <returns>Reference to the assigned chromosome.</returns>
		Chromosome& operator=(const Chromosome& that);
	};

	std::string to_string(const cgp::Chromosome& chromosome);
}
