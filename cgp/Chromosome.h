#pragma once

#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "Configuration.h"
#include "Dataset.h"

namespace cgp {
	/// <summary>
	/// Chromosome class representing an individual in Cartesian Genetic Programming (CGP).
	/// </summary>
	class Chromosome {
	public:
		using weight_input_t = CGPConfiguration::weight_input_t;
		using weight_output_t = CGPConfiguration::weight_output_t;

		/// <summary>
		/// Type definition for inferred weight in Cartesian Genetic Programming.
		/// </summary>
		using weight_value_t = CGPConfiguration::weight_value_t;

		/// <summary>
		/// Type definition for hardware inferred weight in Cartesian Genetic Programming.
		/// </summary>
		using weight_actual_value_t = CGPConfiguration::weight_actual_value_t;

		/// <summary>
		/// Type alias for error values, represented as double-precision floating-point numbers.
		/// </summary>
		using error_t = CGPConfiguration::error_t;

		/// <summary>
		/// Type alias for energy values, represented as double-precision floating-point numbers.
		/// </summary>
		using quantized_energy_t = CGPConfiguration::quantized_energy_t; using energy_t = CGPConfiguration::energy_t;

		/// <summary>
		/// Type alias for area values, represented as double-precision floating-point numbers.
		/// </summary>
		using area_t = CGPConfiguration::area_t;

		/// <summary>
		/// Type alias for delay values, represented as double-precision floating-point numbers.
		/// </summary>
		using quantized_delay_t = CGPConfiguration::quantized_delay_t; using delay_t = CGPConfiguration::delay_t;

		/// <summary>
		/// Type alias for depth values, represented as dimension_t.
		/// </summary>
		using depth_t = CGPConfiguration::dimension_t;

		/// <summary>
		/// Type alias for gate count values, represented as size_t.
		/// </summary>
		using gate_count_t = CGPConfiguration::gate_count_t;

		/// <summary>
		/// Type alias for gate parameters, represented as a tuple of energy and delay values.
		/// </summary>
		using gate_parameters_t = CGPConfiguration::gate_parameters_t;

		/// <summary>
		/// Type alias for gene values, represented as unsigned 16-bit integers.
		/// </summary>
		using gene_t = CGPConfiguration::gene_t;

		/// <summary>
		/// String representation for NaN chromosome in Cartesian Genetic Programming.
		/// </summary>
		static const std::string nan_chromosome_string;

		static bool is_mux(int func);
		static bool is_demux(int func);
	private:
		/// <summary>
		/// Reference to the CGP configuration used for chromosome setup.
		/// </summary>
		const CGPConfiguration& cgp_configuration;

		/// <summary>
		/// Pointer to the start position of the chromosome output.
		/// </summary>
		gene_t* output_start, *absolute_output_start;

		/// <summary>
		/// Pointer to the end position of the chromosome output.
		/// </summary>
		gene_t* output_end, *absolute_output_end;

		/// <summary>
		/// Pointer to the start position of the output pins in the pin map.
		/// </summary>
		weight_value_t* output_pin_start, *absolute_pin_start;

		/// <summary>
		/// Pointer to the end position of the output pins in the pin map.
		/// </summary>
		weight_value_t* output_pin_end, *absolute_pin_end;

		/// <summary>
		/// Array containing tuples specifying the minimum and maximum pin indices for possible output connections based on the look-back parameter.
		/// </summary>
		const std::unique_ptr<std::tuple<int, int>[]> &minimum_output_indicies;

		/// <summary>
		/// Unique pointer to the chromosome array.
		/// </summary>
		std::unique_ptr<gene_t[]> chromosome;

		/// <summary>
		/// Unique pointer to locked output array.
		/// </summary>
		std::unique_ptr<bool[]> locked_outputs;

		/// <summary>
		/// Unique pointer to the pin map array.
		/// </summary>
		std::unique_ptr<weight_value_t[]> pin_map;

		/// <summary>
		/// Unique pointer to the function energy visit map array.
		/// </summary>
		std::unique_ptr<bool[]> gate_visit_map;

		/// <summary>
		/// Shared pointer to the input array.
		/// </summary>
		const weight_value_t* input = nullptr;

		/// <summary>
		/// Current selector.
		/// </summary>
		int selector = 0;

		/// <summary>
		/// Flag indicating whether the chromosome needs evaluation.
		/// </summary>
		bool need_evaluation = true;

		/// <summary>
		/// Flag indicating whether the genotype needs energy evaluation.
		/// </summary>
		bool need_energy_evaluation = true;

		/// <summary>
		/// Flag indicating whether the genotype needs delay evaluation.
		/// </summary>
		bool need_delay_evaluation = true;

		/// <summary>
		/// Flag indicating whether the genotype needs depth evaluation.
		/// </summary>
		bool need_depth_evaluation = true;

		/// <summary>
		/// Cached energy consumption value.
		/// </summary>
		quantized_energy_t estimated_quantized_energy_consumption = CGPConfiguration::quantized_energy_nan;

		/// <summary>
		/// Cached energy consumption value.
		/// </summary>
		energy_t estimated_energy_consumption = CGPConfiguration::energy_nan;

		/// <summary>
		/// Cached area consumption value.
		/// </summary>
		area_t estimated_area_utilisation = CGPConfiguration::area_nan;

		/// <summary>
		/// Cached delay value.
		/// </summary>
		quantized_delay_t estimated_quantized_delay = CGPConfiguration::quantized_delay_nan;

		/// <summary>
		/// Cached delay value.
		/// </summary>
		delay_t estimated_delay = CGPConfiguration::delay_nan;

		/// <summary>
		/// Cached depth value.
		/// </summary>
		depth_t estimated_depth = CGPConfiguration::depth_nan;

		/// <summary>
		/// Cached phenotype node count value. By node, it is understood as one digital gate.
		/// </summary>
		gate_count_t phenotype_node_count = CGPConfiguration::gate_count_nan;

		/// <summary>
		/// Cached the lowest used row.
		/// </summary>
		int top_row = 0;

		/// <summary>
		/// Cached the highest used row.
		/// </summary>
		int bottom_row = 0;

		/// <summary>
		/// Cached the lowest used column.
		/// </summary>
		int first_col = 0;

		/// <summary>
		/// Cached the highest used column.
		/// </summary>
		int last_col = 0;

		/// <summary>
		/// Predicate to check if a given position in the chromosome represents a function.
		/// </summary>
		/// <param name="position">The position in the chromosome.</param>
		/// <returns>True if the position represents a function, otherwise false.</returns>
		bool is_function(int position) const;

		/// <summary>
		/// Predicate to check if a given position in the chromosome represents an input.
		/// </summary>
		/// <param name="position">The position in the chromosome.</param>
		/// <returns>True if the position represents an input, otherwise false.</returns>
		bool is_input(int position) const;

		/// <summary>
		/// Predicate to check if a given position in the chromosome represents an output.
		/// </summary>
		/// <param name="position">The position in the chromosome.</param>
		/// <returns>True if the position represents an output, otherwise false.</returns>
		bool is_output(int position) const;

		int get_output_position(int chromosome_position) const;

		/// <summary>
		/// Method for setting up the initial state of the chromosome randomly.
		/// </summary>
		void setup_chromosome();

		/// <summary>
		/// Method for setting up the initial state of the chromosome based on a serialized chromosome.
		/// </summary>
		/// <param name="serialized_chromosome">Serialized chromosome to initialize the chromosome state.</param>
		void setup_chromosome(const std::string& serialized_chromosome);

		/// <summary>
		/// Method for allocating required maps in order to perform evaluating.
		/// </summary>
		void setup_maps();

		/// <summary>
		/// Method for allocating required maps in order to perform evaluating.
		/// Given chromosome is reused, otherwise new chromosome array is created.
		/// </summary>
		/// <param name="chromosome">The chromosome to be reused or null.</param>
		void setup_maps(const decltype(chromosome)& chromosome);

		/// <summary>
		/// Method for allocating required maps in order to perform evaluating.
		/// Given chromosome is disposed, and all maps and iterators moved to this one.
		/// </summary>
		/// <param name="chromosome">The chromosome to be disposed.</param>
		void setup_maps(Chromosome&& that);

		/// <summary>
		/// Method for setting up output iterator pointer.
		/// </summary>
		void setup_output_iterators(int selector);

		weight_value_t plus(int a, int b);
		weight_value_t minus(int a, int b);
		weight_value_t mul(int a, int b);
		weight_value_t bit_rshift(weight_value_t a, weight_value_t b);
		weight_value_t bit_lshift(weight_value_t a, weight_value_t b);
		weight_value_t bit_and(weight_value_t a, weight_value_t b);
		weight_value_t bit_or(weight_value_t a, weight_value_t b);
		weight_value_t bit_xor(weight_value_t a, weight_value_t b);
		weight_value_t bit_neg(weight_value_t a);
		weight_value_t neg(weight_value_t a);

		void mutate_genes(std::shared_ptr<Chromosome> that) const;
		weight_value_t get_pin_value(int index) const;
		bool move_block_to_the_start(int gate_index);
		void move_gate(int src_gate_index, int dst_gate_index);
		int get_gate_index_from_output_pin(int pin) const;
		int get_gate_index_from_input_pin(int pin) const;
		void invalidate();
	public:
		friend std::ostream& operator<<(std::ostream& os, const Chromosome& chromosome);
		/// <summary>
		/// Constructor for the Chromosome class.
		/// </summary>
		/// <param name="cgp_configuration">Reference to the CGP configuration.</param>
		/// <param name="minimum_output_indicies">Array containing tuples specifying the minimum and maximum pin indices for possible output connections base on look back parameter.</param>
		Chromosome(const CGPConfiguration& cgp_configuration, const std::unique_ptr<std::tuple<int, int>[]> &minimum_output_indicies);
		
		
		/// <summary>
		/// Constructor for the Chromosome class using string chromosome representation.
		/// </summary>
		/// <param name="cgp_configuration">Reference to the CGP configuration.</param>
		/// <param name="minimum_output_indicies">Array containing tuples specifying the minimum and maximum pin indices for possible output connections base on look back parameter.</param>
		/// <param name="serialized_chromosome">Serialized chromosome to be parsed.</param>
		Chromosome(const CGPConfiguration& cgp_configuration, const std::unique_ptr<std::tuple<int, int>[]> &minimum_output_indicies, const std::string &serialized_chromosome);

		/// <summary>
		/// Constructs a Chromosome object using a string chromosome representation.
		/// </summary>
		/// <param name="cgp_configuration">The CGP configuration to be used.</param>
		/// <param name="serialized_chromosome">The serialized chromosome to be parsed.</param>
		Chromosome(const CGPConfiguration& cgp_configuration, const std::string& serialized_chromosome);


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
		/// Getter for the pointer to the specific block in the chromosome.
		/// </summary>
		/// <param name="index">Digital gate index. Indexing start from top-left position, continues down, finally moves to the next column. Repeat until the end is reached.</param>
		/// <returns>Pointer to the block inputs.</returns>
		gene_t* get_block(int index) const;

		/// <summary>
		/// Getter for the pointer to the inputs of a specific block in the chromosome.
		/// </summary>
		/// <param name="row">Row index of the block.</param>
		/// <param name="column">Column index of the block.</param>
		/// <returns>Pointer to the block inputs.</returns>
		gene_t* get_block_inputs(int row, int column) const;

		/// <summary>
		/// Getter for the pointer to the inputs of a specific block in the chromosome.
		/// </summary>
		/// <param name="index">Digital gate index. Indexing start from top-left position, continues down, finally moves to the next column. Repeat until the end is reached.</param>
		/// <returns>Pointer to the block inputs.</returns>
		gene_t* get_block_inputs(int index) const;

		/// <summary>
		/// Getter for the pointer to the function represented by a specific block in the chromosome.
		/// </summary>
		/// <param name="row">Row index of the block.</param>
		/// <param name="column">Column index of the block.</param>
		/// <returns>Pointer to the block function.</returns>
		gene_t* get_block_function(int row, int column) const;

		/// <summary>
		/// Getter for the pointer to the function represented by a specific block in the chromosome.
		/// </summary>
		/// <param name="gate_index">Index of the specific block.</param>
		/// <returns>Pointer to the block function.</returns>
		gene_t* get_block_function(int gate_index) const;

		/// <summary>
		/// Getter for the shared pointer to the chromosome array.
		/// </summary>
		/// <returns>Shared pointer to the chromosome array.</returns>
		const std::unique_ptr<gene_t[]>& get_chromosome() const;

		/// <summary>
		/// Method to perform mutation on the chromosome.
		/// </summary>
		/// <returns>Shared pointer to the mutated chromosome.</returns>
		std::shared_ptr<Chromosome> mutate() const;

		/// <summary>
		/// Method to perform mutation on the chromosome while reusing given chromosome.
		/// </summary>
		/// <returns>Shared pointer to the mutated chromosome.</returns>
		std::shared_ptr<Chromosome> mutate(std::shared_ptr<Chromosome> that);

		/// <summary>
		/// Method to set the input for the chromosome.
		/// </summary>
		/// <param name="input">Shared pointer to the input array.</param>
		void set_input(const weight_value_t *input, int selector);

		/// <summary>
		/// Method to evaluate the chromosome based on its inputs.
		/// </summary>
		void evaluate();

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
		int get_serialized_chromosome_size() const;

		void tight_layout();

		/// <summary>
		/// Estimate energy used by phenotype digital circuit.
		/// </summary>
		/// <returns>Energy estimation.</returns>
		decltype(estimated_energy_consumption) get_estimated_energy_usage();

		/// <summary>
		/// Estimate energy used by phenotype digital circuit.
		/// </summary>
		/// <returns>Energy estimation.</returns>
		decltype(estimated_quantized_energy_consumption) get_estimated_quantized_energy_usage();

		/// <summary>
		/// Estimate area used by phenotype digital circuit.
		/// </summary>
		/// <returns>Area estimation.</returns>
		decltype(estimated_area_utilisation) get_estimated_area_usage();

		/// <summary>
		/// Estimate largest delay by phenotype digital circuit.
		/// </summary>
		/// <returns>Largest delay.</returns>
		decltype(estimated_quantized_delay) get_estimated_quantized_delay();

		/// <summary>
		/// Estimate largest delay by phenotype digital circuit.
		/// </summary>
		/// <returns>Largest delay.</returns>
		decltype(estimated_delay) get_estimated_delay();

		/// <summary>
		/// Estimate largest depth by phenotype digital circuit.
		/// </summary>
		/// <returns>Largest depth.</returns>
		decltype(estimated_depth) get_estimated_depth();

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
		/// Get the first column used by phenotype.
		/// </summary>
		/// <returns>Qunatity of used digital gates.</returns>
		decltype(first_col) get_first_column();

		/// <summary>
		/// Get the last column used by phenotype.
		/// </summary>
		/// <returns>Qunatity of used digital gates.</returns>
		decltype(last_col) get_last_column();

		/// <summary>
		/// Infer unknown weights using CGP genotype and return array of weights.
		/// </summary>
		/// <param name="input">Shared pointer to an array of input values.</param>
		/// <param name="selector">Selector set to multipexor and de-multiplexor gates.</param>
		/// <returns>Shared pointer to an array of infered weights</returns>
		weight_output_t get_weights(const weight_input_t &input, int selector = 0);

		/// <summary>
		/// Infer unknown weights using CGP genotype and return vector of weights arrays.
		/// </summary>
		/// <param name="input">Vector of shared pointers to an array of input values.</param>
		/// <returns>Vector of shared pointers to an array of infered weights associated with specific inputs</returns>
		std::vector<weight_output_t> get_weights(const std::vector<weight_input_t>& input);

		void find_direct_solutions(const dataset_t &dataset);
	};

	std::string to_string(const Chromosome& chromosome);
	std::string to_string(const std::shared_ptr<Chromosome> &chromosome);
}
