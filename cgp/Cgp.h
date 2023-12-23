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
		// Collection of chromosomes representing individuals in the population.
		std::vector<std::shared_ptr<Chromosome>> chromosomes;

		// Array containing tuples specifying the minimum and maximum pin indices for output connections.
		std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies;

		// Current best fitness value achieved during the evolutionary process.
		double best_fitness;

		// Count of generations without improvement in the best fitness.
		size_t generations_without_change;

		// Best chromosome with the lowest fitness value (Mean Squared Error).
		std::shared_ptr<Chromosome> best_chromosome;

		// Best solutions found by each thread during parallel execution.
		std::map<decltype(omp_get_thread_num()), std::tuple<double, std::shared_ptr<Chromosome>>> best_solutions;

		// Counter for the total evolution steps made during the process.
		size_t evolution_steps_made;

		// Reference to the vector of expected values used for fitness evaluation.
		const std::vector<double>& expected_values;

		// Minimum expected value in the dataset.
		double expected_value_min;

		// Maximum expected value in the dataset.
		double expected_value_max;

		// Calculate the fitness of a chromosome.
		double fitness(Chromosome& chrom);

	public:
		/// <summary>
		/// Constructor for CGP class.
		/// </summary>
		/// <param name="expected_values">Vector of expected values for fitness evaluation.</param>
		CGP(const std::vector<double>& expected_values);

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
		void evaluate(const std::shared_ptr<double[]> input);

		/// <summary>
		/// Get the current best fitness value.
		/// </summary>
		/// <returns>Current best fitness value.</returns>
		decltype(best_fitness) get_best_fitness() const;

		/// <summary>
		/// Get the chromosome with the lowest fitness value.
		/// </summary>
		/// <returns>Best chromosome.</returns>
		decltype(best_chromosome) get_best_chromosome() const;

		/// <summary>
		/// Get number of generations without improvement in the best fitness.
		/// </summary>
		/// <returns>Number of generations without change.</returns>
		decltype(generations_without_change) get_generations_without_change() const;
	};
}
