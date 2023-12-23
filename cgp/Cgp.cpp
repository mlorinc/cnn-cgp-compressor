#include "Cgp.h"
#include <algorithm>
#include <execution>
#include <limits>
#include <omp.h>

using namespace cgp;

CGP::CGP(const std::vector<double>& expected_values) : best_fitness(std::numeric_limits<double>::infinity()), expected_values(expected_values) {
	auto min_max = std::minmax_element(expected_values.begin(), expected_values.end());
	expected_value_min = *min_max.first;
	expected_value_max = *min_max.second;
	generations_without_change = 0;
}

CGP::~CGP() {}

void CGP::build() {
	const uint8_t input_column_pins = input_count();
	const uint8_t fn_arity = function_output_arity();
	minimum_output_indicies = std::make_shared<std::tuple<int, int>[]>(col_count() + 1);

	// Calculate pin available according to defined L parameter.
	// Skip input pins (+1) and include output pins (+1)
#pragma omp parallel for
	for (int col = 1; col < col_count() + 2; col++) {
		int first_col = std::max(0, col - look_back_parameter());
		decltype(row_count()) min_pin, max_pin;

		// Input pins can be used, however they quantity might be different than row_count()
		if (first_col == 0)
		{
			min_pin = 0;
			max_pin = (look_back_parameter() - 1) * row_count() * fn_arity + input_column_pins;
		}
		else {
			min_pin = std::max(0, first_col - 1) * row_count() * fn_arity + input_column_pins;
			max_pin = min_pin + look_back_parameter() * row_count() * fn_arity;
		}


		minimum_output_indicies[col - 1] = std::make_tuple(min_pin, max_pin);
	}

	// Create new population
#pragma omp barrier
	for (size_t i = 0; i < population_max(); i++)
	{
		auto chromosome = std::make_shared<Chromosome>(Chromosome(*this, minimum_output_indicies, expected_value_min, expected_value_max));
		chromosomes.push_back(chromosome);
	}
}

// MSE loss function implementations;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
double mse(const double* predictions, const std::vector<double>& actual)
{
	int count = actual.size();
	double sum = 0;

#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < count; i++)
	{
		double diff = predictions[i] - actual[i];
		sum += diff * diff;
	}

	return sum / count;
}

double CGP::fitness(Chromosome& chrom) {
	chrom.evaluate();
	return mse(chrom.begin_output(), expected_values); // / actual.size();
}

void CGP::mutate() {
	for (int i = 0; i < chromosomes.size(); i++)
	{
		std::shared_ptr<Chromosome> chrom = chromosomes[i];
		chrom = best_chromosome->mutate();
		chromosomes[i] = chrom;
	}
}


void CGP::evaluate(const std::shared_ptr<double[]> input) {
#pragma omp parallel for
	for (int i = 0; i < chromosomes.size(); i++) {
		auto& chromosome = chromosomes[i];
		chromosome->set_input(input);
		double fit = fitness(*chromosome);

		// The following section will set fitness and the chromosome
		// to best_solutions map if the fitness is better than the best fitness
		// known. Map collection was employed to make computation more efficient
		// on CPU with multiple cores, and to avoid mutex blocking.
		auto thread_id = omp_get_thread_num();
		auto best_solution = best_solutions.find(thread_id);

		if (best_solution != best_solutions.end())
		{
			double best_fitness = std::get<0>(best_solution->second);
			auto& best_chromosome = std::get<1>(best_solution->second);
			if (fit <= best_fitness)
			{
				best_solution->second = std::make_tuple(fit, chromosome);
			}
		}
		else {
			best_solutions[thread_id] = std::make_tuple(fit, chromosome);
		}
	}

#pragma omp barrier
	generations_without_change++;

	// Find the best chromosome accros threads and save it to the best chromosome variable
	// including its fitness.
	for (auto it = best_solutions.begin(), end = best_solutions.end(); it != end; it++)
	{
		double& thread_best_fitness = std::get<0>(it->second);
		auto& thread_best_chromosome = std::get<1>(it->second);
		if (thread_best_fitness <= best_fitness)
		{
			best_fitness = thread_best_fitness;
			best_chromosome = thread_best_chromosome;

			if (thread_best_fitness != best_fitness) {
				generations_without_change = 0;
			}
		}
	}
}

decltype(CGP::best_fitness) CGP::get_best_fitness() const {
	return best_fitness;
}

decltype(CGP::best_chromosome) CGP::get_best_chromosome() const {
	return best_chromosome;
}

decltype(CGP::generations_without_change) CGP::get_generations_without_change() const {
	return generations_without_change;
}
