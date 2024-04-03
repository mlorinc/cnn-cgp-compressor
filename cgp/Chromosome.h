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
		using weight_value_t = CGPConfiguration::weight_value_t;
		using weight_actual_value_t = CGPConfiguration::weight_actual_value_t;
		using gate_parameters_t = CGPConfiguration::gate_parameters_t;

		// Reference to the CGP configuration used for chromosome setup.
		CGPConfiguration &cgp_configuration;

		// Pointers to the start and end positions of the chromosome output.
		gene_t* output_start, * output_end;

		// Pointers to the start and end positions of the output pins in the pin map.
		weight_value_t* output_pin_start, * output_pin_end;

		// Array containing tuples specifying the minimum and maximum pin indices for possible output connections base on look back parameter.
		const std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies;

		// Minimum expected value in the dataset.
		const weight_value_t expected_value_min;

		// Maximum expected value in the dataset.
		const weight_value_t expected_value_max;

		// Shared pointer to the chromosome array.
		std::shared_ptr<gene_t[]> chromosome;

		// Shared pointer to the pin map array.
		std::unique_ptr<weight_value_t[]> pin_map;

		// Shared pointer to the function energy map array.
		std::unique_ptr<gate_parameters_t[]> gate_parameters_map;

		// Shared pointer to the function energy visit map array.
		std::unique_ptr<bool[]> gate_visit_map;

		// Shared pointer to the input array.
		std::shared_ptr<weight_value_t[]> input;

		// Flag indicating whether the chromosome needs evaluation.
		bool need_evaluation = true;

		// Flag indicating whether the genotype needs energy evaluation.
		bool need_energy_evaluation = true;

		// Cached energy consumption value.
		double estimated_energy_consumptation = std::numeric_limits<double>::infinity();

		// Cached delay value.
		double estimated_largest_delay = 0;

		// Cached depth value.
		size_t estimated_largest_depth = 0;

		// Cached phenotype node count value. By node it is understood as one digital gate.
		size_t phenotype_node_count = 0;

		// Cached the lowest used row.
		size_t top_row = 0;

		// Cached the highest used row.
		size_t bottom_row = 0;

		// Private method to check if a given position in the chromosome represents a function.
		bool is_function(size_t position) const;

		// Private method to check if a given position in the chromosome represents an input.
		bool is_input(size_t position) const;

		// Private method to check if a given position in the chromosome represents an output.
		bool is_output(size_t position) const;

		// Private method for setting up the initial state of the chromosome.
		void setup_chromosome();

		// Private method for allocating pin and energy arrays (maps). Furthermore, chromosome array is allocated.
		void setup_maps();

		// Private method for allocating pin and energy arrays (maps). Chromosome is reused.
		void setup_maps(decltype(chromosome) chromosome);

		void setup_maps(Chromosome &&that);

		// Private method for setting up iterator pointers.
		void setup_iterators();
		void setup_iterators(Chromosome&& that);
	public:
		friend std::ostream& operator<<(std::ostream& os, const Chromosome& chromosome);
		/// <summary>
		/// Constructor for the Chromosome class.
		/// </summary>
		/// <param name="cgp_configuration">Reference to the CGP configuration.</param>
		/// <param name="minimum_output_indicies">Array containing tuples specifying the minimum and maximum pin indices for possible output connections base on look back parameter.</param>
		/// <param name="expected_value_min">Minimum expected value in the dataset.</param>
		/// <param name="expected_value_max">Maximum expected value in the dataset.</param>
		Chromosome(CGPConfiguration& cgp_configuration, const std::shared_ptr<std::tuple<int, int>[]> &minimum_output_indicies, weight_actual_value_t expected_value_min, weight_actual_value_t expected_value_max);
		
		
		/// <summary>
		/// Constructor for the Chromosome class using string chromosome representation.
		/// </summary>
		/// <param name="cgp_configuration">Reference to the CGP configuration.</param>
		/// <param name="minimum_output_indicies">Array containing tuples specifying the minimum and maximum pin indices for possible output connections base on look back parameter.</param>
		/// <param name="expected_value_min">Minimum expected value in the dataset.</param>
		/// <param name="expected_value_max">Maximum expected value in the dataset.</param>
		/// <param name="serialized_chromosome">Serialized chromosome to be parsed.</param>
		Chromosome(CGPConfiguration& cgp_configuration, const std::shared_ptr<std::tuple<int, int>[]> &minimum_output_indicies, weight_actual_value_t expected_value_min, weight_actual_value_t expected_value_max, const std::string &serialized_chromosome);

		///// <summary>
		///// Constructor for the Chromosome class using string chromosome representation.
		///// </summary>
		///// <param name="serialized_chromosome">Serialized chromosome to be parsed.</param>
		//Chromosome(const std::string& serialized_chromosome);

		/// <summary>
		/// Copy constructor for the Chromosome class.
		/// </summary>
		/// <param name="that">Reference to the chromosome to be copied.</param>
		Chromosome(const Chromosome& that) noexcept;

		/// <summary>
		/// Move constructor for the Chromosome class.
		/// </summary>
		/// <param name="that">Reference to the chromosome to be move.</param>
		Chromosome(Chromosome&& that) noexcept;

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
		/// Getter for the pointer to the inputs of a specific block in the chromosome.
		/// </summary>
		/// <param name="index">Digital gate index. Indexing start from top-left position, continues down, finally moves to the next column. Repeat until the end is reached.</param>
		/// <returns>Pointer to the block inputs.</returns>
		gene_t* get_block_inputs(size_t index) const;

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
		std::shared_ptr<Chromosome> mutate() const;

		/// <summary>
		/// Method to set the input for the chromosome.
		/// </summary>
		/// <param name="input">Shared pointer to the input array.</param>
		void set_input(std::shared_ptr<weight_value_t[]> input);

		/// <summary>
		/// Method to evaluate the chromosome based on its inputs.
		/// </summary>
		void evaluate(size_t selector);

		/// <summary>
		/// Getter for the pointer to the beginning of the output array.
		/// </summary>
		/// <returns>Pointer to the beginning of the output array.</returns>
		weight_value_t* begin_output();

		/// <summary>
		/// Getter for the pointer to the end of the output array.
		/// </summary>
		/// <returns>Pointer to the end of the output array.</returns>
		weight_value_t* end_output();

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
		/// Estimate energy used by phenotype digital circuit.
		/// </summary>
		/// <returns>Energy estimation.</returns>
		decltype(estimated_energy_consumptation) get_estimated_energy_usage();

		/// <summary>
		/// Estimate largest delay by phenotype digital circuit.
		/// </summary>
		/// <returns>Largest delay.</returns>
		decltype(estimated_largest_delay) get_estimated_largest_delay();

		/// <summary>
		/// Estimate largest depth by phenotype digital circuit.
		/// </summary>
		/// <returns>Largest depth.</returns>
		decltype(estimated_largest_depth) get_estimated_largest_depth();

		/// <summary>
		/// Get quantity of used digital gates used by phenotype.
		/// </summary>
		/// <returns>Qunatity of used digital gates.</returns>
		decltype(phenotype_node_count) get_node_count();

		/// <summary>
		/// Get the highest row used by phenotype.
		/// </summary>
		/// <returns>Qunatity of used digital gates.</returns>
		decltype(top_row) get_top_row();

		/// <summary>
		/// Get the lowest row used by phenotype.
		/// </summary>
		/// <returns>Qunatity of used digital gates.</returns>
		decltype(bottom_row) get_bottom_row();

		/// <summary>
		/// Infer unknown weights using CGP genotype and return array of weights.
		/// </summary>
		/// <param name="input">Shared pointer to an array of input values.</param>
		/// <param name="selector">Selector set to multipexor and de-multiplexor gates.</param>
		/// <returns>Shared pointer to an array of infered weights</returns>
		std::shared_ptr<weight_value_t[]> get_weights(const std::shared_ptr<weight_value_t[]> input, size_t selector = 0);

		/// <summary>
		/// Infer unknown weights using CGP genotype and return vector of weights arrays.
		/// </summary>
		/// <param name="input">Vector of shared pointers to an array of input values.</param>
		/// <returns>Vector of shared pointers to an array of infered weights associated with specific inputs</returns>
		std::vector<std::shared_ptr<weight_value_t[]>> get_weights(const std::vector<std::shared_ptr<weight_value_t[]>>& input);
	};

	std::string to_string(const cgp::Chromosome& chromosome);
}
