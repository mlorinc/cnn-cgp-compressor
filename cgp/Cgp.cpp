#include "Cgp.h"
#include <algorithm>
#include <execution>
#include <limits>
#include <omp.h>
#include <fstream>
#include <sstream>

using namespace cgp;

CGP::CGP(const weight_actual_value_t expected_min_value, const weight_actual_value_t expected_max_value, const double mse_threshold) :
	best_solution(std::make_tuple(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), nullptr)),
	expected_value_min(expected_min_value),
	expected_value_max(expected_max_value),
	generations_without_change(0),
	evolution_steps_made(0)
{
	this->mse_threshold(mse_threshold);
}

CGP::CGP(const double mse_threshold) :
	best_solution(std::make_tuple(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), nullptr)),
	expected_value_min(std::numeric_limits<weight_actual_value_t>::min()),
	expected_value_max(std::numeric_limits<weight_actual_value_t>::max()),
	generations_without_change(0),
	evolution_steps_made(0)
{
	this->mse_threshold(mse_threshold);
}

CGP::CGP(std::istream& in) : CGP()
{
	load(in);
}

CGP::~CGP() {}

void CGP::reset()
{
	best_solution = std::make_tuple(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), nullptr);
	generations_without_change = 0;
	evolution_steps_made = 0;
	best_solutions.clear();
	chromosomes.clear();
}

void CGP::dump(std::ostream& out) const
{
	CGPConfiguration::dump(out);
	std::string chromosome = (std::get<2>(best_solution) != nullptr) ? (std::get<2>(best_solution)->to_string()) : ("non-existent");
	out << "best_solution: " << std::get<0>(best_solution) << " " << std::get<1>(best_solution) << " " << chromosome << std::endl;
	out << "evolution_steps_made: " << evolution_steps_made << std::endl;
	out << "expected_value_min: " << static_cast<weight_repr_value_t>(expected_value_min) << std::endl;
	out << "expected_value_max: " << static_cast<weight_repr_value_t>(expected_value_max) << std::endl;
	for (auto it = other_config_attribitues.begin(), end = other_config_attribitues.end(); it != end; it++)
	{
		out << it->first << ": " << it->second << std::endl;
	}
}

void CGP::load(std::istream& in)
{
	CGPConfiguration::load(in);
	in.clear();
	in.seekg(0);
	std::string line;
	std::string best_solution_string;
	
	other_config_attribitues.clear();
	while (std::getline(in, line)) {
		std::string key = line;
		std::string value = line;

		key.erase(0, key.find_first_not_of(" \t\r\n"));
		key.erase(key.find_first_of(":"));
		value.erase(0, value.find_first_of(":") + 1);
		value.erase(0, value.find_first_not_of(" \t\r\n"));
		value.erase(value.find_last_not_of(" \t\r\n") + 1);

		if (key == "best_solution") {
			best_solution_string = value;
		}
		else if (key == "evolution_steps_made") {
			evolution_steps_made = std::stoull(value);
		}
		else if (key == "expected_value_min") {
			expected_value_min = std::stoul(value);
		}
		else if (key == "expected_value_max") {
			expected_value_max = std::stoul(value);
		}
		else if (!key.empty())
		{
			other_config_attribitues[key] = value;
		}
	}
	build_indices();
	if (!best_solution_string.empty())
	{
		set_best_solution(best_solution_string);
	}
}

void CGP::build_indices()
{
	const auto input_column_pins = input_count();
	const auto fn_arity = function_output_arity();
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
			min_pin = (first_col - 1) * row_count() * fn_arity + input_column_pins;
			max_pin = min_pin + (col - first_col) * row_count() * fn_arity;
		}


		minimum_output_indicies[col - 1] = std::make_tuple(min_pin, max_pin);
	}
#pragma omp barrier
	return;
}

void CGP::generate_population()
{
	chromosomes.clear();
	const auto& best_chromosome = std::get<2>(best_solution);
	// Create new population
	for (size_t i = 0; i < population_max(); i++)
	{
		auto chromosome = (best_chromosome != nullptr) ? (best_chromosome->mutate()) : (std::make_shared<Chromosome>(Chromosome(*this, minimum_output_indicies, expected_value_min, expected_value_max)));
		chromosomes.push_back(chromosome);
	}
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
double CGP::mse_without_division(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const
{
	double sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < output_count(); i++)
	{
		double diff = predictions[i] - expected_output[i];
		sum += diff * diff;
	}

	// Overflow detected, discard result completely
	if (sum < 0) {
		return std::numeric_limits<double>::infinity();
	}

	return sum;
}

void CGP::set_best_solution(const std::string& solution)
{
	std::istringstream iss(solution);
	double mse;
	std::string chrom, energy;
	iss >> mse >> energy >> chrom;
	auto chromosome = std::make_shared<Chromosome>(*this, minimum_output_indicies, expected_value_min, expected_value_max, chrom);
	best_solution = std::make_tuple(mse, (energy == "inf") ? (std::numeric_limits<double>::infinity()) : std::stold(energy), chromosome);
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
double CGP::mse(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const
{
	return mse_without_division(predictions, expected_output) / output_count();
}

CGP::solution_t CGP::analyse_chromosome(std::shared_ptr<Chromosome> chrom, const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output)
{
	auto end = input.end();
	decltype(mse(nullptr, nullptr)) mse_accumulator = 0;
	size_t selector = 0;
	for (auto input_it = input.begin(), output_it = expected_output.begin(); input_it != end; input_it++, output_it++)
	{
		chrom->set_input(*input_it);
		chrom->evaluate(selector++);
		mse_accumulator += error_fitness_without_aggregation(*chrom, *output_it);
		if (mse_accumulator < 0) [[unlikely]]
			{
				mse_accumulator = std::numeric_limits<double>::infinity();
				break;
			}
	}
	mse_accumulator /= output_count() * input.size();
	if (mse_accumulator <= mse_threshold())
	{
		return std::make_tuple(mse_accumulator, energy_fitness(*chrom), chrom);
	}
	else {
		return std::make_tuple(mse_accumulator, std::numeric_limits<double>::infinity(), chrom);
	}
}

CGP::solution_t CGP::analyse_chromosome(std::shared_ptr<Chromosome> chrom, const std::shared_ptr<weight_value_t[]> input, const std::shared_ptr<weight_value_t[]> expected_output, size_t selector)
{
	chrom->set_input(input);
	chrom->evaluate(selector);
	auto mse = error_fitness(*chrom, expected_output);

	if (mse <= mse_threshold())
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

double CGP::error_fitness(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output) {
	return mse(chrom.begin_output(), expected_output);
}

double CGP::error_fitness_without_aggregation(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output) {
	return mse_without_division(chrom.begin_output(), expected_output);
}

void CGP::mutate() {
	auto best_chromosome = get_best_chromosome();
	for (int i = 0; i < chromosomes.size(); i++)
	{
		chromosomes[i] = best_chromosome->mutate();
	}
	evolution_steps_made++;
}

bool CGP::dominates(solution_t a, solution_t b) const
{
	double a_error_fitness = std::get<0>(a);
	double a_energy_fitness = std::get<1>(a);

	double b_error_fitness = std::get<0>(b);
	double b_energy_fitness = std::get<1>(b);

	auto mse_t = mse_threshold();
	// Allow neutral evolution for error in case energies are the same
	return ((mse_t < a_error_fitness && a_error_fitness <= b_error_fitness) ||
		(mse_t >= a_error_fitness && a_energy_fitness < b_energy_fitness) ||
		(mse_t >= a_error_fitness && a_energy_fitness == b_energy_fitness && a_error_fitness <= b_error_fitness));
}

void CGP::evaluate(const std::shared_ptr<weight_value_t[]> input, const std::shared_ptr<weight_value_t[]> expected_output)
{
	return evaluate(std::vector{ input }, std::vector{ expected_output });
}

void CGP::evaluate(const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output) {
#pragma omp parallel for
	for (int i = 0; i < chromosomes.size(); i++) {
		auto& chromosome = chromosomes[i];
		solution_t chromosome_result = analyse_chromosome(chromosome, input, expected_output);

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
		const auto &thread_solution = it->second;
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
	// for some reason slows down openMP extremely
	//best_solutions.clear();
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

void CGP::restore(
	const std::string& solution,
	const size_t mutations_made
)
{
	generations_without_change = 0;
	evolution_steps_made = (mutations_made != std::numeric_limits<size_t>::max()) ? (mutations_made) : (evolution_steps_made);
	set_best_solution(solution);
}

void CGP::restore(
	std::shared_ptr<Chromosome> chromosome,
	const std::shared_ptr<weight_value_t[]> input,
	const std::shared_ptr<weight_value_t[]> expected_output,
	const size_t mutations_made
)
{
	generations_without_change = 0;
	evolution_steps_made = (mutations_made != std::numeric_limits<size_t>::max()) ? (mutations_made) : (evolution_steps_made);
	best_solution = analyse_chromosome(chromosome, input, expected_output);
}


void CGP::restore(std::shared_ptr<Chromosome> chromosome,
	const std::vector<std::shared_ptr<weight_value_t[]>>& input,
	const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output,
	const size_t mutations_made
)
{
	generations_without_change = 0;
	evolution_steps_made = (mutations_made != std::numeric_limits<size_t>::max()) ? (mutations_made) : (evolution_steps_made);
	best_solution = analyse_chromosome(chromosome, input, expected_output);
}
