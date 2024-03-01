#include "Cgp.h"
#include <algorithm>
#include <execution>
#include <limits>
#include <omp.h>

using namespace cgp;

CGP::CGP(const std::shared_ptr<weight_value_t[]> expected_values, const size_t expected_values_size, const weight_value_t expected_min_value, const weight_value_t expected_max_value, const double mse_threshold) :
	best_solution(std::make_tuple(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), nullptr)),
	expected_values(expected_values),
	expected_values_size(expected_values_size),
	expected_value_min(expected_min_value),
	expected_value_max(expected_max_value),
	generations_without_change(0),
	evolution_steps_made(0),
	mse_threshold(mse_threshold) {
}

CGP::~CGP() {}

void CGP::build() {
	const uint8_t input_column_pins = input_count();
	const uint8_t fn_arity = function_output_arity();
	minimum_output_indicies = std::make_shared<std::tuple<int, int>[]>(col_count() + 1);
	int l_back = look_back_parameter();

	// Calculate pin available according to defined L parameter.
	// Skip input pins (+1) and include output pins (+1)
#pragma omp parallel for
	for (int col = 1; col < col_count() + 2; col++) {
		int first_col = std::max(0, col - l_back);
		decltype(row_count()) min_pin, max_pin;

		// Input pins can be used, however they quantity might be different than row_count()
		if (first_col == 0)
		{
			min_pin = 0;
			max_pin = (col - 1) * row_count() * fn_arity + input_column_pins;
		}
		else {
			min_pin = std::max(0, first_col - 1) * row_count() * fn_arity + input_column_pins;
			max_pin = min_pin + l_back * row_count() * fn_arity;
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
double CGP::mse(const weight_value_t* predictions) const
{
	double sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < expected_values_size; i++)
	{
		double diff = predictions[i] - expected_values[i];
		sum += diff * diff;
	}

	// Overflow detected, discard result completely
	if (sum < 0) {
		return std::numeric_limits<double>::infinity();
	}

	return sum / expected_values_size;
}

CGP::solution_t CGP::analyse_chromosome(std::shared_ptr<Chromosome> chrom)
{
	chrom->evaluate();
	auto mse = error_fitness(*chrom);

	if (mse <= mse_threshold)
	{
		return std::make_tuple(mse, energy_fitness(*chrom), chrom);
	}
	else {
		return std::make_tuple(mse, std::numeric_limits<double>::infinity(), chrom);
	}
}

double CGP::energy_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_energy_usage();
}

double CGP::error_fitness(Chromosome& chrom) {
	return mse(chrom.begin_output());
}

void CGP::mutate() {
	auto best_chromosome = get_best_chromosome();
	for (int i = 0; i < chromosomes.size(); i++)
	{
		std::shared_ptr<Chromosome> chrom = chromosomes[i];
		chrom = best_chromosome->mutate();
		chromosomes[i] = chrom;
	}
	evolution_steps_made++;
}

bool CGP::dominates(solution_t a, solution_t b) const
{
	double a_error_fitness = std::get<0>(a);
	double a_energy_fitness = std::get<1>(a);

	double b_error_fitness = std::get<0>(b);
	double b_energy_fitness = std::get<1>(b);

	// Allow neutral evolution for error in case energies are the same
	return ((mse_threshold < a_error_fitness && a_error_fitness <= b_error_fitness) ||
		(mse_threshold >= a_error_fitness && a_energy_fitness < b_energy_fitness) ||
		(mse_threshold >= a_error_fitness && a_energy_fitness == b_energy_fitness && a_error_fitness <= b_error_fitness));
}

void CGP::evaluate(const std::shared_ptr<weight_value_t[]> input) {
#pragma omp parallel for
	for (int i = 0; i < chromosomes.size(); i++) {
		auto& chromosome = chromosomes[i];
		chromosome->set_input(input);
		solution_t chromosome_result = analyse_chromosome(chromosome);

		// The following section will set fitness and the chromosome
		// to best_solutions map if the fitness is better than the best fitness
		// known. Map collection was employed to make computation more efficient
		// on CPU with multiple cores, and to avoid mutex blocking.
		auto thread_id = omp_get_thread_num();
		auto best_solution = best_solutions.find(thread_id);

		if (best_solution != best_solutions.end())
		{
			// Allow neutral evolution for error
			if (dominates(chromosome_result, best_solution->second)) {
				best_solution->second = chromosome_result;
			}
		}
		else {
			// Create default
			best_solutions[thread_id] = chromosome_result;
		}
	}

#pragma omp barrier
	generations_without_change++;

	// Find the best chromosome accros threads and save it to the best chromosome variable
	// including its fitness.
	for (auto it = best_solutions.begin(), end = best_solutions.end(); it != end; it++)
	{
		auto thread_solution = it->second;
		double thread_best_error_fitness = std::get<0>(thread_solution);
		double thread_best_energy_fitness = std::get<1>(thread_solution);
		double best_error_fitness = std::get<0>(best_solution);
		double best_energy_fitness = std::get<1>(best_solution);

		// Allow neutral evolution for error
		if (dominates(thread_solution, best_solution)) {
			if (thread_best_error_fitness != best_error_fitness || thread_best_energy_fitness != best_energy_fitness) {
				generations_without_change = 0;
			}
			best_solution = thread_solution;
		}
	}
}

double CGP::get_best_error_fitness() const {
	return std::get<0>(best_solution);
}

double CGP::get_best_energy_fitness() const {
	return std::get<1>(best_solution);
}

std::shared_ptr<Chromosome> CGP::get_best_chromosome() const {
	return std::get<2>(best_solution);
}

decltype(CGP::generations_without_change) CGP::get_generations_without_change() const {
	return generations_without_change;
}

size_t CGP::get_serialized_chromosome_size() const
{
	// chromosome size + input information + output information
	return chromosome_size() * sizeof(gene_t) + 2 * sizeof(gene_t);
}