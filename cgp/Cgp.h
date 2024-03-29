#pragma once

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
	private:
		using gene_t = CGPConfiguration::gene_t;
		// A candidate solution tuple in format: error fitness, energy fitness and the genotype.
		using solution_t = std::tuple<double, double, std::shared_ptr<Chromosome>>;

		solution_t best_solution;

		// Collection of chromosomes representing individuals in the population.
		std::vector<std::shared_ptr<Chromosome>> chromosomes;

		// Array containing tuples specifying the minimum and maximum pin indices for output connections.
		std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies;

		// Count of generations without improvement in the best fitness.
		size_t generations_without_change;

		// Best solutions found by each thread during parallel execution.
		std::map<decltype(omp_get_thread_num()), solution_t> best_solutions;

		// Counter for the total evolution steps made during the process.
		size_t evolution_steps_made;

		// Minimum expected value in the dataset.
		weight_value_t expected_value_min;

		// Maximum expected value in the dataset.
		weight_value_t expected_value_max;

		std::map<std::string, std::string> other_config_attribitues;

		// Calculate the accuracy(error) fitness of a chromosome.
		double error_fitness(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output);

		// Calculate the accuracy(error) fitness of a chromosome without aggregation.
		double error_fitness_without_aggregation(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output);

		// Calculate the energy fitness of a chromosome.
		double energy_fitness(Chromosome& chrom);

		solution_t analyse_chromosome(std::shared_ptr<Chromosome> chrom, const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output);
		solution_t analyse_chromosome(std::shared_ptr<Chromosome> chrom, const std::shared_ptr<weight_value_t[]> input, const std::shared_ptr<weight_value_t[]> expected_output);

		// Calculate MSE metric for made predictions
		double mse(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const;

		// Calculate MSE metric for made predictions
		double mse_without_division(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const;

		// Determine whether candidate solution A is better than B.
		bool dominates(solution_t a, solution_t b) const;

		/// <summary>
		/// Set the best solution from given string.
		/// </summary>
		/// <returns>Best chromosome.</returns>
		void set_best_solution(const std::string &solution);
	public:
		/// <summary>
		/// Constructor for CGP class.
		/// </summary>
		/// <param name="expected_min_value">Minimum expected value in the dataset.</param>
		/// <param name="expected_max_value">Maximum expected value in the dataset.</param>
		/// <param name="mse_threshold">Mean Squared Error threshold after optimisation is focused on minimising energy.</param>
		CGP(const weight_actual_value_t expected_min_value, const weight_actual_value_t expected_max_value, const double mse_threshold = 0);

		/// <summary>
		/// Constructor for CGP class.
		/// </summary>
		/// <param name="mse_threshold">Mean Squared Error threshold after optimisation is focused on minimising energy.</param>
		CGP(const double mse_threshold = 0);

		/// <summary>
		/// Constructor for CGP class using text stream to initialize variables.
		/// </summary>
		/// <param name="in">Input stream containing serialized form of the CGP class.</param>
		CGP(std::istream &in);

		/// <summary>
		/// Destructor for CGP class.
		/// </summary>
		~CGP();

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
		/// Evaluate the fitness of the population based on the given input and expected output.
		/// </summary>
		/// <param name="input">Shared pointer to an array of input values.</param>
		/// <param name="expected_output">Shared pointer to an array of expected output values.</param>
		void evaluate(const std::shared_ptr<weight_value_t[]> input, const std::shared_ptr<weight_value_t[]> expected_output);

		/// <summary>
		/// Evaluate the fitness of the population based on the given inputs and expected outputs.
		/// </summary>
		/// <param name="input">Vector reference to shared array pointer of input values.</param>
		/// <param name="expected_output">Vector reference to shared array pointer of expected output values.</param>
		void evaluate(const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output);

		/// <summary>
		/// Get the current best error fitness value.
		/// </summary>
		/// <returns>Current best error fitness value.</returns>
		double get_best_error_fitness() const;

		/// <summary>
		/// Get the current best energy fitness value.
		/// </summary>
		/// <returns>Current best energy fitness value.</returns>
		double get_best_energy_fitness() const;

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
			const std::string &solution,
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

		void dump(std::ostream& out) const override;
		void load(std::istream& in) override;
	};
}
