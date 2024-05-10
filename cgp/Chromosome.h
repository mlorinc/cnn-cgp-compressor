#pragma once

#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <map>
#include "Configuration.h"
#include "Dataset.h"

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
		static gate_parameters_t id_gate_parameters;

		int max_genes_to_mutate;
		int chromosome_size;

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
		/// Unique pointer to locked gates array.
		/// </summary>
		int locked_nodes_index;

		/// <summary>
		/// Unique pointer to the pin map array.
		/// </summary>
		std::unique_ptr<weight_value_t[]> pin_map;

		/// <summary>
		/// Unique pointer to the function energy visit map array.
		/// </summary>
		std::unique_ptr<bool[]> gate_visit_map;

		std::unique_ptr<depth_t[]> depth_distance_map;
		std::unique_ptr<quantized_delay_t[]> quantized_delay_distance_map;

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
		/// Flag indicating multiplexing.
		/// </summary>
		bool multiplexing = false;

		/// <summary>
		/// Multiplexed gates start index.
		/// </summary>
		int start_id_index;

		int id_count;

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

		size_t output_count;

		bool is_locked_node(int gate_index) const;

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
		void setup_maps(const Chromosome& chromosome);

		/// <summary>
		/// Method for allocating required maps in order to perform evaluating.
		/// Given chromosome is disposed, and all maps and iterators moved to this one.
		/// </summary>
		/// <param name="chromosome">The chromosome to be disposed.</param>
		void setup_maps(Chromosome&& that);

		/// <summary>
		/// Method for setting up output iterator pointer.
		/// </summary>
		void setup_output_iterators(int selector, size_t output_count);

		void update_mutation_variables();

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

		int get_column(int gate_index) const;
		int get_row(int gate_index) const;

		/// <summary>
		/// Mutate genes of the chromosome represented by the provided pointer to a chromosome.
		/// </summary>
		/// <param name="that">A pointer to the chromosome to mutate genes.</param>
		/// <returns>True if mutation was successful, false otherwise.</returns>
		bool mutate_genes(std::shared_ptr<Chromosome> that) const;

		/// <summary>
		/// Get the value of the pin at the specified index.
		/// </summary>
		/// <param name="index">The index of the pin.</param>
		/// <returns>The value of the pin.</returns>
		weight_value_t get_pin_value(int index) const;

		/// <summary>
		/// Get the output value of the gate at the specified index.
		/// </summary>
		/// <param name="gate_index">The index of the gate.</param>
		/// <returns>The output value of the gate.</returns>
		weight_value_t get_gate_output(int gate_index) const;

		/// <summary>
		/// Get the used pin of the gate's output at the specified index.
		/// </summary>
		/// <param name="gate_index">The index of the gate.</param>
		/// <returns>The used pin of the gate's output.</returns>
		int get_gate_output_used_pin(int gate_index) const;

		/// <summary>
		/// Get the input value of the gate at the specified index and pin.
		/// </summary>
		/// <param name="gate_index">The index of the gate.</param>
		/// <param name="pin">The index of the input pin.</param>
		/// <returns>The input value of the gate.</returns>
		weight_value_t get_gate_input(int gate_index, int pin = 0) const;

		/// <summary>
		/// Get the used pin of the gate's input at the specified index.
		/// </summary>
		/// <param name="gate_index">The index of the gate.</param>
		/// <returns>The used pin of the gate's input.</returns>
		int get_gate_input_used_pin(int gate_index) const;

		/// <summary>
		/// Move the gate at the gate index to the first free gate block.
		/// </summary>
		/// <param name="gate_index">The index of the gate.</param>
		/// <returns>True if the block was successfully moved, false otherwise.</returns>
		bool move_block_to_the_start(int gate_index);

		/// <summary>
		/// Move the gate from the source index to the destination index.
		/// </summary>
		/// <param name="src_gate_index">The source index of the gate.</param>
		/// <param name="dst_gate_index">The destination index of the gate.</param>
		void move_gate(int src_gate_index, int dst_gate_index);

		/// <summary>
		/// Get the gate index from the given output pin.
		/// </summary>
		/// <param name="pin">The output pin.</param>
		/// <returns>The gate index corresponding to the output pin.</returns>
		int get_gate_index_from_output_pin(int pin) const;

		/// <summary>
		/// Get the gate index from the given input pin.
		/// </summary>
		/// <param name="pin">The input pin.</param>
		/// <returns>The gate index corresponding to the input pin.</returns>
		int get_gate_index_from_input_pin(int pin) const;

		/// <summary>
		/// Get the output pin from the gate index and pin.
		/// </summary>
		/// <param name="gate_index">The index of the gate.</param>
		/// <param name="pin">The index of the output pin.</param>
		/// <returns>The output pin corresponding to the gate index and pin.</returns>
		int get_output_pin_from_gate_index(int gate_index, int pin = 0) const;

		/// <summary>
		/// Invalidate the chromosome, indicating that it needs evaluation.
		/// </summary>
		void invalidate();

		/// <summary>
		/// Find a free index starting from the specified index.
		/// </summary>
		/// <param name="from">The index to start searching from.</param>
		/// <returns>The index of the first free position found, or -1 if none is found.</returns>
		int find_free_index(int from) const;

		/// <summary>
		/// Get the cost of the specified function.
		/// </summary>
		/// <param name="function">The function to get the cost for.</param>
		/// <returns>The cost of the function.</returns>
		gate_parameters_t get_function_cost(gene_t function) const;

		/// <summary>
		/// Copy input pins from the source gate to the destination gate.
		/// </summary>
		/// <param name="src_index">The index of the source gate.</param>
		/// <param name="dst_index">The index of the destination gate.</param>
		void copy_gate_input_pins(int src_index, int dst_index);

		/// <summary>
		/// Copy input pins from the source gate to the destination gate, starting from the specified pins.
		/// </summary>
		/// <param name="src_index">The index of the source gate.</param>
		/// <param name="dst_index">The index of the destination gate.</param>
		/// <param name="src_pin">The index of the starting input pin in the source gate.</param>
		/// <param name="dst_pin">The index of the starting input pin in the destination gate.</param>
		void copy_gate_input_pins(int src_index, int dst_index, int src_pin, int dst_pin);

		/// <summary>
		/// Copy the function of the source gate to the destination gate.
		/// </summary>
		/// <param name="src_index">The index of the source gate.</param>
		/// <param name="dst_index">The index of the destination gate.</param>
		void copy_gate_function(int src_index, int dst_index);

		/// <summary>
		/// Convert the multiplexed value to its index.
		/// </summary>
		/// <param name="value">The multiplexed value.</param>
		/// <returns>The index corresponding to the multiplexed value.</returns>
		int mulitplexed_value_to_index(int value) const;

		/// <summary>
		/// Convert the multiplexed value to its relative index.
		/// </summary>
		/// <param name="value">The multiplexed value.</param>
		/// <returns>The relative index corresponding to the multiplexed value.</returns>
		int mulitplexed_value_to_relative_index(int value) const;

		/// <summary>
		/// Convert the multiplexed index to its value.
		/// </summary>
		/// <param name="index">The multiplexed index.</param>
		/// <returns>The value corresponding to the multiplexed index.</returns>
		int mulitplexed_index_to_value(int index) const;

		/// <summary>
		/// Convert the relative multiplexed index to its value.
		/// </summary>
		/// <param name="index">The relative multiplexed index.</param>
		/// <returns>The value corresponding to the relative multiplexed index.</returns>
		int relative_mulitplexed_index_to_value(int index) const;

		/// <summary>
		/// Get the ID output for the given value.
		/// </summary>
		/// <param name="value">The value to get the ID output for.</param>
		/// <returns>The ID output corresponding to the value.</returns>
		int get_id_output_for(int value) const;

		int get_function_input_arity(int gate_index) const;
		int get_function_output_arity(int gate_index) const;
		int clip_pin(int pin) const;
		bool is_used_pin(int pin) const;

		void wire_multiplexed_id_to_output(const dataset_t& dataset);
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
		std::shared_ptr<Chromosome> mutate(std::shared_ptr<Chromosome> that, const dataset_t& dataset);

		/// <summary>
		/// Swap visits maps.
		/// </summary>
		void swap_visit_map(Chromosome &chromosome);

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
		/// Getter for the pointer to the end of the multiplexed output array.
		/// </summary>
		/// <returns>Pointer to the end of the output array.</returns>
		weight_value_t* end_multiplexed_output();

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

		/// <summary>
		/// Shrink CGP circuit grid to minimal size.
		/// </summary>
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
		/// <returns>Quantity of used digital gates.</returns>
		decltype(phenotype_node_count) get_node_count();

		/// <summary>
		/// Get the highest row used by phenotype.
		/// </summary>
		/// <returns>Quantity of used digital gates.</returns>
		decltype(top_row) get_top_row();

		/// <summary>
		/// Get the lowest row used by phenotype.
		/// </summary>
		/// <returns>Quantity of used digital gates.</returns>
		decltype(bottom_row) get_bottom_row();

		/// <summary>
		/// Get the first column used by phenotype.
		/// </summary>
		/// <returns>Quantity of used digital gates.</returns>
		decltype(first_col) get_first_column();

		/// <summary>
		/// Get the last column used by phenotype.
		/// </summary>
		/// <returns>Quantity of used digital gates.</returns>
		decltype(last_col) get_last_column();

		/// <summary>
		/// Get whether the chromosome is multiplexed.
		/// </summary>
		/// <returns>True when multiplexed, false otherwise.</returns>
		decltype(multiplexing) is_multiplexing() const;

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

		/// <summary>
		/// Find direct solutions for the given dataset.
		/// </summary>
		/// <param name="dataset">The dataset to find direct solutions for.</param>
		void find_direct_solutions(const dataset_t& dataset);

		/// <summary>
		/// Add helper powers of 2 circuits for the given dataset.
		/// </summary>
		/// <param name="dataset">The dataset to add circuits to.</param>
		void add_2pow_circuits(const dataset_t& dataset);

		/// <summary>
		/// Apply multiplexing optimisation.
		/// </summary>
		/// <param name="dataset">The dataset to build initial circuit.</param>
		void use_multiplexing(const dataset_t& dataset);

		/// <summary>
		/// Remove multiplexing from the chromosome.
		/// </summary>
		void remove_multiplexing(const dataset_t& dataset);

		/// <summary>
		/// Perform corrections on the chromosome based on the given dataset and threshold.
		/// </summary>
		/// <param name="dataset">The dataset to perform corrections with.</param>
		/// <param name="threshold">The threshold value for corrections (default is determined automatically).</param>
		/// <param name="zero_energy_only">Whether to consider only zero energy corrections (default is false).</param>
		void perform_corrections(const dataset_t& dataset, const int threshold = 512, const bool zero_energy_only = false);

		/// <summary>
		/// Get the relative ID output from the given index.
		/// </summary>
		/// <param name="index">The index to get the relative ID output for.</param>
		/// <returns>The relative ID output.</returns>
		int get_relative_id_output_from_index(int index) const;

		bool needs_evaluation() const;

		/// <summary>
		/// Collect gate statistics and their quantity.
		/// </summary>
		/// <returns>Map mapping function number to statistics.</returns>
		std::map<int, int> get_gate_statistics();
	};

	std::string to_string(const Chromosome& chromosome);
	std::string to_string(const std::shared_ptr<Chromosome> &chromosome);
}
