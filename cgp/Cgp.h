#pragma once

#include <array>
#include <vector>
#include <memory>
#include <map>
#include <omp.h>
#include "Configuration.h"
#include "Chromosome.h"

namespace cgp {
	/// <summary>
	/// Cartesian Genetic Programming (CGP) class, derived from CGPConfiguration.
	/// </summary>
	class CGP : public CGPConfiguration {
	public:
		// A candidate solution tuple in format: error fitness, (energy fitness, largest delay), depth, and the genotype.
		using solution_t = std::tuple<error_t, energy_t, area_t, delay_t, depth_t, gate_count_t, std::shared_ptr<Chromosome>>;

		/// <summary>
		/// Get the error value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to retrieve the error value from</param>
		/// <returns>The error value of the solution</returns>
		static error_t get_error(const solution_t solution);

		/// <summary>
		/// Get the energy value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to retrieve the energy value from</param>
		/// <returns>The energy value of the solution</returns>
		static energy_t get_energy(const solution_t solution);

		/// <summary>
		/// Get the area value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to retrieve the area value from</param>
		/// <returns>The area value of the solution</returns>
		static area_t get_area(const solution_t solution);

		/// <summary>
		/// Get the delay value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to retrieve the delay value from</param>
		/// <returns>The delay value of the solution</returns>
		static delay_t get_delay(const solution_t solution);

		/// <summary>
		/// Get the depth value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to retrieve the depth value from</param>
		/// <returns>The depth value of the solution</returns>
		static depth_t get_depth(const solution_t solution);

		/// <summary>
		/// Get the gate count value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to retrieve the gate count value from</param>
		/// <returns>The gate count value of the solution</returns>
		static gate_count_t get_gate_count(const solution_t solution);

		/// <summary>
		/// Ensure that the energy value of the given solution is valid.
		/// </summary>
		/// <param name="solution">The solution to ensure the energy value for</param>
		/// <returns>The ensured energy value of the solution</returns>
		static decltype(CGP::get_energy(solution_t())) ensure_energy(solution_t& solution);

		/// <summary>
		/// Ensure that the area value of the given solution is valid.
		/// </summary>
		/// <param name="solution">The solution to ensure the area value for</param>
		/// <returns>The ensured area value of the solution</returns>
		static decltype(CGP::get_area(solution_t())) ensure_area(solution_t& solution);

		/// <summary>
		/// Ensure that the delay value of the given solution is valid.
		/// </summary>
		/// <param name="solution">The solution to ensure the delay value for</param>
		/// <returns>The ensured delay value of the solution</returns>
		static decltype(CGP::get_delay(solution_t())) ensure_delay(solution_t& solution);

		/// <summary>
		/// Ensure that the depth value of the given solution is valid.
		/// </summary>
		/// <param name="solution">The solution to ensure the depth value for</param>
		/// <returns>The ensured depth value of the solution</returns>
		static decltype(CGP::get_depth(solution_t())) ensure_depth(solution_t& solution);

		/// <summary>
		/// Ensure that the gate count value of the given solution is valid.
		/// </summary>
		/// <param name="solution">The solution to ensure the gate count value for</param>
		/// <returns>The ensured gate count value of the solution</returns>
		static decltype(CGP::get_gate_count(solution_t())) ensure_gate_count(solution_t& solution);

		/// <summary>
		/// Get the chromosome associated with the given solution.
		/// </summary>
		/// <param name="solution">The solution to retrieve the chromosome from</param>
		/// <returns>The chromosome associated with the solution</returns>
		static std::shared_ptr<Chromosome> get_chromosome(const solution_t solution);
	protected:
		/// <summary>
		/// Get default solution with invalid values.
		/// </summary>
		/// <returns>A new invalid solution.</returns>
		static solution_t get_default_solution();

		/// <summary>
		/// Create a new solution with assigned values.
		/// </summary>
		/// <param name="chromosome">The chromosome associated with the solution</param>
		/// <param name="error">The error value of the solution</param>
		/// <param name="energy">The energy value of the solution (default: energy_nan)</param>
		/// <param name="delay">The delay value of the solution (default: delay_nan)</param>
		/// <param name="depth">The depth value of the solution (default: Chromosome::depth_nan)</param>
		/// <param name="gate_count">The gate count value of the solution (default: Chromosome::gate_count_nan)</param>
		/// <returns>The newly created solution</returns>
		static solution_t create_solution(
			std::shared_ptr<Chromosome> chromosome,
			error_t error,
			energy_t energy = energy_nan,
			area_t area = area_nan,
			delay_t delay = delay_nan,
			depth_t depth = depth_nan,
			gate_count_t gate_count = gate_count_nan);

		/// <summary>
		/// Set the error value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to set the error value for</param>
		/// <param name="value">The value to set as the error value</param>
		/// <returns>The updated error value of the solution</returns>
		static decltype(CGP::get_error(solution_t())) set_error(solution_t& solution, decltype(CGP::get_error(solution_t())) value);

		/// <summary>
		/// Set the energy value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to set the energy value for</param>
		/// <param name="value">The value to set as the energy value</param>
		/// <returns>The updated energy value of the solution</returns>
		static decltype(CGP::get_energy(solution_t())) set_energy(solution_t& solution, decltype(CGP::get_energy(solution_t())) value);

		/// <summary>
		/// Set the area value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to set the area value for</param>
		/// <param name="value">The value to set as the area value</param>
		/// <returns>The updated area value of the solution</returns>
		static decltype(CGP::get_area(solution_t())) set_area(solution_t& solution, decltype(CGP::get_area(solution_t())) value);

		/// <summary>
		/// Set the delay value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to set the delay value for</param>
		/// <param name="value">The value to set as the delay value</param>
		/// <returns>The updated delay value of the solution</returns>
		static decltype(CGP::get_delay(solution_t())) set_delay(solution_t& solution, decltype(CGP::get_delay(solution_t())) value);

		/// <summary>
		/// Set the depth value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to set the depth value for</param>
		/// <param name="value">The value to set as the depth value</param>
		/// <returns>The updated depth value of the solution</returns>
		static decltype(CGP::get_depth(solution_t())) set_depth(solution_t& solution, decltype(CGP::get_depth(solution_t())) value);

		/// <summary>
		/// Set the gate count value of the given solution.
		/// </summary>
		/// <param name="solution">The solution to set the gate count value for</param>
		/// <param name="value">The value to set as the gate count value</param>
		/// <returns>The updated gate count value of the solution</returns>
		static decltype(CGP::get_gate_count(solution_t())) set_gate_count(solution_t& solution, decltype(CGP::get_gate_count(solution_t())) value);

		/// <summary>
		/// The default solution format string.
		/// </summary>
		static const std::string default_solution_format;

		/// <summary>
		/// The best solution found by the CGP.
		/// </summary>
		solution_t best_solution;

		/// <summary>
		/// Collection of chromosomes representing individuals in the population.
		/// </summary>
		std::vector<std::shared_ptr<Chromosome>> chromosomes;

		/// <summary>
		/// Array containing tuples specifying the minimum and maximum pin indices for output connections.
		/// </summary>
		std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies;

		/// <summary>
		/// Count of generations without improvement in the best fitness.
		/// </summary>
		size_t generations_without_change;

		/// <summary>
		/// Best solutions found by each thread during parallel execution.
		/// </summary>
		std::unique_ptr<solution_t[]> best_solutions;
		decltype(omp_get_max_threads()) best_solutions_size;

		/// <summary>
		/// Counter for the total evolution steps made during the process.
		/// </summary>
		size_t evolution_steps_made;

		/// <summary>
		/// Additional configuration attributes lefto by CGPConfiguration::load.
		/// </summary>
		std::map<std::string, std::string> other_config_attribitues;

		/// <summary>
		/// Calculate the accuracy (error) fitness of a chromosome.
		/// </summary>
		/// <param name="chrom">The chromosome for which to calculate the fitness.</param>
		/// <param name="expected_output">The expected output values against which to compare the chromosome's output.</param>
		/// <returns>The error fitness value.</returns>
		error_t error_fitness(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output);

		/// <summary>
		/// Calculate the accuracy (error) fitness of a chromosome without aggregation.
		/// </summary>
		/// <param name="chrom">The chromosome for which to calculate the fitness.</param>
		/// <param name="expected_output">The expected output values against which to compare the chromosome's output.</param>
		/// <returns>The error fitness value.</returns>
		error_t error_fitness_without_aggregation(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output);

		/// <summary>
		/// Calculate the energy fitness of a chromosome.
		/// </summary>
		/// <param name="chrom">The chromosome for which to calculate the fitness.</param>
		/// <returns>The energy fitness value.</returns>
		energy_t get_energy_fitness(Chromosome& chrom);

		/// <summary>
		/// Calculate the area fitness of a chromosome.
		/// </summary>
		/// <param name="chrom">The chromosome for which to calculate the fitness.</param>
		/// <returns>The area fitness value.</returns>
		area_t get_area_fitness(Chromosome& chrom);

		/// <summary>
		/// Calculate the delay fitness of a chromosome.
		/// </summary>
		/// <param name="chrom">The chromosome for which to calculate the fitness.</param>
		/// <returns>The delay fitness value.</returns>
		delay_t get_delay_fitness(Chromosome& chrom);

		/// <summary>
		/// Calculate the depth fitness of a chromosome.
		/// </summary>
		/// <param name="chrom">The chromosome for which to calculate the fitness.</param>
		/// <returns>The depth fitness value.</returns>
		depth_t get_depth_fitness(Chromosome& chrom);

		/// <summary>
		/// Calculate the gate count fitness of a chromosome.
		/// </summary>
		/// <param name="chrom">The chromosome for which to calculate the fitness.</param>
		/// <returns>The gate count fitness value.</returns>
		gate_count_t get_gate_count(Chromosome& chrom);

		/// <summary>
		/// Calculate the Mean Squared Error (MSE) metric for made predictions.
		/// </summary>
		/// <param name="predictions">The predictions made by the chromosome.</param>
		/// <param name="expected_output">The expected output values.</param>
		/// <returns>The MSE metric value.</returns>
		error_t mse(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const;

		/// <summary>
		/// Calculate the Mean Squared Error (MSE) metric for made predictions without division.
		/// </summary>
		/// <param name="predictions">The predictions made by the chromosome.</param>
		/// <param name="expected_output">The expected output values.</param>
		/// <returns>The MSE metric value without being divided.</returns>
		error_t mse_without_division(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const;

		/// <summary>
		/// Analyze chromosome and calculate MSE metric for made predictions. It does not fill up other fitnesses due to performance.
		/// </summary>
		/// <param name="chrom">The chromosome to analyze.</param>
		/// <param name="input">The input data.</param>
		/// <param name="expected_output">The expected output data.</param>
		/// <returns>The solution containing the chromosome and MSE metric. Other attributes can be accesed from the chromosome instance.</returns>
		solution_t analyse_chromosome(std::shared_ptr<Chromosome> chrom, const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output);

		/// <summary>
		/// Analyze chromosome and calculate MSE metric for made predictions. It does not fill up other fitnesses due to performance.
		/// </summary>
		/// <param name="chrom">The chromosome to analyze.</param>
		/// <param name="input">The input data.</param>
		/// <param name="expected_output">The expected output data.</param>
		/// <param name="selector">Optional parameter for selector.</param>
		/// <returns>The solution containing the analyzed chromosome and MSE metric. Other attributes can be accesed from the chromosome instance.</returns>
		solution_t analyse_chromosome(std::shared_ptr<Chromosome> chrom, const std::shared_ptr<weight_value_t[]> input, const std::shared_ptr<weight_value_t[]> expected_output, size_t selector = 0);

		/// <summary>
		/// Determine whether candidate solution A is better than B.
		/// </summary>
		/// <param name="a">The first solution candidate.</param>
		/// <param name="b">The second solution candidate.</param>
		/// <returns>A tuple indicating if A dominates B and if the mutation was neutral.</returns>
		std::tuple<bool, bool> dominates(solution_t& a, solution_t& b) const;

		/// <summary>
		/// Set the best solution from the given string.
		/// </summary>
		/// <param name="solution">The string representing the solution.</param>
		/// <param name="format">The format of the solution containing parameters.</param>
		void set_best_solution(const std::string& solution, std::string format);

		/// <summary>
		/// Create a solution from the given string.
		/// </summary>
		/// <param name="solution">The string representing the solution.</param>
		/// <param name="format">The format of the string.</param>
		/// <returns>The created solution.</returns>
		solution_t create_solution(std::string solution, std::string format);

	public:
		/// <summary>
		/// Constructor for CGP class.
		/// </summary>
		/// <param name="expected_min_value">Minimum expected value in the dataset.</param>
		/// <param name="expected_max_value">Maximum expected value in the dataset.</param>
		CGP(const weight_actual_value_t expected_min_value, const weight_actual_value_t expected_max_value);

		/// <summary>
		/// Constructor for CGP class.
		/// </summary>
		CGP();

		/// <summary>
		/// Constructor for CGP class using text stream to initialize variables.
		/// </summary>
		/// <param name="in">Input stream containing serialized form of the CGP class.</param>
		/// <param name="arguments">CGP arguments entered from CLI by the user.</param>
		CGP(std::istream& in, const std::vector<std::string>& arguments = {});

		/// <summary>
		/// Destructor for CGP class.
		/// </summary>
		~CGP();

		/// <summary>
		/// Get number of evolutions preformed.
		/// </summary>
		decltype(evolution_steps_made) get_evolution_steps_made() const;

		/// <summary>
		/// Build the initial pin indices.
		/// </summary>
		void build_indices();

		/// <summary>
		/// Build the initial population.
		/// </summary>
		void generate_population();

		/// <summary>
		/// Mutate the current population.
		/// </summary>
		void mutate();

		/// <summary>
		/// Evaluate the fitness of the population based on the given inputs and expected outputs.
		/// </summary>
		/// <param name="input">Vector reference to shared array pointer of input values.</param>
		/// <param name="expected_output">Vector reference to shared array pointer of expected output values.</param>
		void evaluate(const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output);

		/// <summary>
		/// Evaluate the fitness of the chromosome based on the given inputs and expected outputs. Compared to the train variant,
		/// this method returns all fitness values in the solution.
		/// </summary>
		/// <param name="input">Reference to a vector of shared pointers to arrays containing input values.</param>
		/// <param name="expected_output">Reference to a vector of shared pointers to arrays containing expected output values.</param>
		/// <param name="chromosome">The chromosome to evaluate.</param>
		/// <returns>A solution containing all fitness values.</returns>
		solution_t evaluate(const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output, std::shared_ptr<Chromosome> chromosome);

		/// <summary>
		/// Get the current best error fitness value.
		/// </summary>
		/// <returns>Current best error fitness value.</returns>
		error_t get_best_error_fitness() const;

		/// <summary>
		/// Get the current best energy fitness value.
		/// </summary>
		/// <returns>Current best energy fitness value.</returns>
		energy_t get_best_energy_fitness();

		/// <summary>
		/// Get the current best area fitness value.
		/// </summary>
		/// <returns>Current best area fitness value.</returns>
		area_t get_best_area_fitness();

		/// <summary>
		/// Get the current best delay fitness value.
		/// </summary>
		/// <returns>Current best delay fitness value.</returns>
		delay_t get_best_delay_fitness();

		/// <summary>
		/// Get the current best depth value.
		/// </summary>
		/// <returns>Current best depth value.</returns>
		depth_t get_best_depth();

		/// <summary>
		/// Get the current best gate count.
		/// </summary>
		/// <returns>Current best gate count value.</returns>
		gate_count_t get_best_gate_count();

		/// <summary>
		/// Get the chromosome with the lowest fitness value.
		/// </summary>
		/// <returns>Best chromosome.</returns>
		std::shared_ptr<Chromosome> get_best_chromosome() const;

		/// <summary>
		/// Set evoluton to the specific point.
		/// </summary>
		/// <param name="chromosome">Starting chromosome for evolution.</param>
		/// <param name="input">Shared array pointer of input values.</param>
		/// <param name="expected_output">Shared array pointer of expected output values.</param>
		/// <param name="mutations_made">Mutations made prior obtaining given chromosome.</param>
		void restore(
			std::shared_ptr<Chromosome> chromosome,
			const std::shared_ptr<weight_value_t[]> input,
			const std::shared_ptr<weight_value_t[]> expected_output,
			const size_t mutations_made = std::numeric_limits<size_t>::max()
		);

		/// <summary>
		/// Set evoluton to the specific point.
		/// </summary>
		/// <param name="chromosome">Starting chromosome for evolution.</param>
		/// <param name="input">Vector reference to shared array pointer of input values.</param>
		/// <param name="expected_output">Vector reference to shared array pointer of expected output values.</param>
		/// <param name="mutations_made">Mutations made prior obtaining given chromosome.</param>
		void restore(
			std::shared_ptr<Chromosome> chromosome,
			const std::vector<std::shared_ptr<weight_value_t[]>>& input,
			const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output,
			const size_t mutations_made = std::numeric_limits<size_t>::max()
		);

		/// <summary>
		/// Set evoluton to the specific point from given serialized solution string.
		/// </summary>
		/// <param name="solution">Serialized solution string containing error fitness, energy fitness and serialized chromosome.</param>
		/// <param name="mutations_made">Mutations made prior obtaining given chromosome.</param>
		void restore(
			const std::string& solution,
			const std::string& solution_format = CGP::default_solution_format,
			const size_t mutations_made = std::numeric_limits<size_t>::max()
		);

		/// <summary>
		/// Reset the CGP algorithm to initial state.
		/// </summary>
		void reset();

		/// <summary>
		/// Get number of generations without improvement in the best fitness.
		/// </summary>
		/// <returns>Number of generations without change.</returns>
		decltype(generations_without_change) get_generations_without_change() const;

		/// <summary>
		/// Calculate size of the gene.
		/// </summary>
		/// <returns>The size of gene.</returns>
		size_t get_serialized_chromosome_size() const;

		/// <summary>
		/// Dump the CGP experiment parameters to the specified output stream.
		/// </summary>
		/// <param name="out">The output stream to which the parameters will be dumped.</param>
		void dump(std::ostream& out) const override;
		void dump_all(std::ostream& out);

		/// <summary>
		/// Load the CGP experiment parameters from the specified input stream.
		/// </summary>
		/// <param name="in">The input stream from which to load the parameters.</param>
		/// <param name="arguments">Optional arguments for loading.</param>
		/// <returns>A map containing additional unprocessed information.</returns>
		std::map<std::string, std::string> load(std::istream& in, const std::vector<std::string>& arguments = {}) override;

	};
}
