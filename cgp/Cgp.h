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

		// Mean Squared Error threshold after optimisation is focused on minimising energy.
		double mse_threshold;

		// Count of generations without improvement in the best fitness.
		size_t generations_without_change;

		// Best solutions found by each thread during parallel execution.
		std::map<decltype(omp_get_thread_num()), solution_t> best_solutions;

		// Counter for the total evolution steps made during the process.
		size_t evolution_steps_made;

		// Shared pointer to array of expected values used for fitness evaluation.
		const std::shared_ptr<weight_value_t[]> expected_values;

		// Number of elements in expected_values.
		const size_t expected_values_size;

		// Minimum expected value in the dataset.
		weight_value_t expected_value_min;

		// Maximum expected value in the dataset.
		weight_value_t expected_value_max;

		// Calculate the fitness of a chromosome.
		double error_fitness(Chromosome& chrom);
		double energy_fitness(Chromosome& chrom);

		solution_t analyse_chromosome(std::shared_ptr<Chromosome> chrom);

		// Calculate MSE metric for made predictions
		double mse(const weight_value_t* predictions) const;

		// Determine whether candidate solution A is better than B.
		bool dominates(solution_t a, solution_t b) const;
	public:
		/// <summary>
		/// Constructor for CGP class.
		/// </summary>
		/// <param name="expected_values">Array of expected values for fitness evaluation.</param>
		/// <param name="expected_values_size">Number of expected values.</param>
		/// <param name="expected_min_value">Minimum expected value in the dataset.</param>
		/// <param name="expected_max_value">Maximum expected value in the dataset.</param>
		/// <param name="mse_threshold">Mean Squared Error threshold after optimisation is focused on minimising energy.</param>
		CGP(const std::shared_ptr<weight_value_t[]> expected_values, const size_t expected_values_size, const weight_value_t expected_min_value, const weight_value_t expected_max_value, const double mse_threshold = 0);

		/// <summary>
		/// Destructor for CGP class.
		/// </summary>
		~CGP();

		/// <summary>
		/// Build the initial population.
		/// </summary>
		void build();

		/// <summary>
		/// Mutate the current population.
		/// </summary>
		void mutate();

		/// <summary>
		/// Evaluate the fitness of the population based on the given input.
		/// </summary>
		/// <param name="input">Shared pointer to an array of input values.</param>
		void evaluate(const std::shared_ptr<weight_value_t[]> input);

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
		/// Get number of generations without improvement in the best fitness.
		/// </summary>
		/// <returns>Number of generations without change.</returns>
		decltype(generations_without_change) get_generations_without_change() const;

		/// <summary>
		/// Calculate size of the gene.
		/// </summary>
		/// <returns>The size of gene.</returns>
		size_t get_serialized_chromosome_size() const;
	};
}
