#include "Cgp.h"
#include <algorithm>
#include <execution>
#include <limits>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace cgp;

const std::string CGP::default_solution_format = "error energy chromosome";

CGP::CGP(const weight_actual_value_t expected_min_value, const weight_actual_value_t expected_max_value) :
	best_solution(CGP::get_default_solution()),
	expected_value_min(expected_min_value),
	expected_value_max(expected_max_value),
	generations_without_change(0),
	evolution_steps_made(0),
	chromosomes(std::vector<std::shared_ptr<Chromosome>>(omp_get_max_threads(), nullptr))
{
	best_solutions_size = omp_get_max_threads();
	best_solutions = std::make_unique<solution_t[]>(best_solutions_size);
}

CGP::CGP() :
	best_solution(CGP::get_default_solution()),
	expected_value_min(std::numeric_limits<weight_actual_value_t>::min()),
	expected_value_max(std::numeric_limits<weight_actual_value_t>::max()),
	generations_without_change(0),
	evolution_steps_made(0),
	chromosomes(std::vector<std::shared_ptr<Chromosome>>(omp_get_max_threads(), nullptr))
{
	best_solutions_size = omp_get_max_threads();
	best_solutions = std::make_unique<solution_t[]>(best_solutions_size);
}

CGP::CGP(std::istream& in, const std::vector<std::string>& arguments) : CGP()
{
	load(in, arguments);
}

CGP::~CGP() {}

void CGP::reset()
{
	best_solution = CGP::get_default_solution();
	generations_without_change = 0;
	evolution_steps_made = 0;
	best_solutions_size = omp_get_max_threads();
	best_solutions = std::make_unique<solution_t[]>(best_solutions_size);
}

void CGP::dump(std::ostream& out) const
{
	CGPConfiguration::dump(out);
	const auto& chrom = CGP::get_chromosome(best_solution);
	std::string chromosome = (chrom) ? (chrom->to_string()) : ("non-existent");
	out << "best_solution_format: error energy delay depth gate_count chromosome" << std::endl;
	out << "best_solution: " 
		<< CGP::get_error(best_solution) << " "
		<< CGP::get_energy(best_solution) << " "
		<< CGP::get_delay(best_solution) << " "
		<< ((CGP::get_depth(best_solution) != std::numeric_limits<size_t>::max()) ? (std::to_string(CGP::get_depth(best_solution))) : ("inf")) << " "
		<< ((CGP::get_gate_count(best_solution) != std::numeric_limits<size_t>::max()) ? (std::to_string(CGP::get_gate_count(best_solution))) : ("inf")) << " "
		<< chromosome << std::endl;
	out << "evolution_steps_made: " << evolution_steps_made << std::endl;
	out << "expected_value_min: " << static_cast<weight_repr_value_t>(expected_value_min) << std::endl;
	out << "expected_value_max: " << static_cast<weight_repr_value_t>(expected_value_max) << std::endl;
	for (auto it = other_config_attribitues.cbegin(), end = other_config_attribitues.cend(); it != end; it++)
	{
		out << it->first << ": " << it->second << std::endl;
	}
}

std::map<std::string, std::string> CGP::load(std::istream& in, const std::vector<std::string>& arguments)
{
	auto remaining_data = CGPConfiguration::load(in);
	std::string best_solution_string;
	std::string best_solution_format = default_solution_format;
	
	other_config_attribitues.clear();
	for (auto it = remaining_data.cbegin(), end = remaining_data.cend(); it != end;) {
		const std::string &key = it->first;
		const std::string &value = it->second;

		if (key == "best_solution") {
			best_solution_string = value;
			remaining_data.erase(it++);
		}
		else if (key == "evolution_steps_made") {
			evolution_steps_made = std::stoull(value);
			remaining_data.erase(it++);
		}
		else if (key == "expected_value_min") {
			expected_value_min = std::stoul(value);
			remaining_data.erase(it++);
		}
		else if (key == "expected_value_max") {
			expected_value_max = std::stoul(value);
			remaining_data.erase(it++);
		}
		else if (key == "best_solution_format") {
			best_solution_format = value;
		}
		else {
			++it;
		}
	}
	set_from_arguments(arguments);
	build_indices();
	if (!starting_solution().empty())
	{
		set_best_solution(starting_solution(), best_solution_format);
	}
	else if (!best_solution_string.empty())
	{
		set_best_solution(best_solution_string, best_solution_format);
	}
	other_config_attribitues = remaining_data;
	return remaining_data;
}

void CGP::build_indices()
{
	const auto input_column_pins = input_count();
	const auto fn_arity = function_output_arity();
	const auto row_count = this->row_count();
	const auto col_count = this->col_count();
	minimum_output_indicies = std::make_shared<std::tuple<int, int>[]>(col_count + 1);
	int l_back = look_back_parameter();

	// Calculate pin available according to defined L parameter.
	// Skip input pins (+1) and include output pins (+1)
	for (int col = 1; col < col_count + 2; col++) {
		int first_col = std::max(0, col - l_back);
		decltype(this->row_count()) min_pin, max_pin;

		// Input pins can be used, however they quantity might be different than row_count()
		if (first_col == 0)
		{
			min_pin = 0;
			max_pin = (col - 1) * row_count * fn_arity + input_column_pins;
		}
		else {
			min_pin = (first_col - 1) * row_count * fn_arity + input_column_pins;
			max_pin = min_pin + (col - first_col) * row_count * fn_arity;
		}

		minimum_output_indicies[col - 1] = std::make_tuple(min_pin, max_pin);
	}
	return;
}

void CGP::generate_population()
{
	const auto& best_chromosome = get_best_chromosome();
	const int end = population_max();
	int i;
	#pragma omp parallel for default(shared) private(i)
	// Create new population
	for (i = 0; i < end; i++)
	{
		chromosomes[i] = (best_chromosome) ? (best_chromosome->mutate()) : (std::make_shared<Chromosome>(*this, minimum_output_indicies, expected_value_min, expected_value_max));;
	}
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
double CGP::mse_without_division(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const
{
	double sum = 0;
// #pragma omp parallel for reduction(+:sum)
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

void CGP::set_best_solution(const std::string& solution, std::string format_string)
{
	std::istringstream iss(solution), format(format_string);
	std::string attribute_name, attribute_value;

	std::string chrom;
	double error, energy, delay;
	size_t depth, gate_count;

	while (format >> attribute_name)
	{
		if (iss >> attribute_value)
		{
			if (attribute_name == "error")
			{
				error = (attribute_value == "inf") ? (std::numeric_limits<double>::infinity()) : std::stold(attribute_value);
			}
			else if (attribute_name == "energy")
			{
				energy = (attribute_value == "inf") ? (std::numeric_limits<double>::infinity()) : std::stold(attribute_value);
			}
			else if (attribute_name == "delay")
			{
				delay = (attribute_value == "inf") ? (std::numeric_limits<double>::infinity()) : std::stold(attribute_value);
			}
			else if (attribute_name == "depth")
			{
				depth = (attribute_value == "inf") ? (std::numeric_limits<size_t>::max()) : std::stoull(attribute_value);
			}
			else if (attribute_name == "gate_count")
			{
				gate_count = (attribute_value == "inf") ? (std::numeric_limits<size_t>::max()) : std::stoull(attribute_value);
			}
			else
			{
				throw std::invalid_argument("unknown fitness parameter " + attribute_name);
			}
		}
		else
		{
			throw std::invalid_argument("missing value for " + attribute_name);
		}
	}

	if (!format.eof())
	{
		throw std::invalid_argument("all attributes were not consumed: " + format.str());
	}

	if (!iss.eof())
	{
		throw std::invalid_argument("all attribute values were not consumed: " + iss.str());
	}

	auto chromosome = std::make_shared<Chromosome>(*this, minimum_output_indicies, expected_value_min, expected_value_max, chrom);
	best_solution = CGP::create_solution(
		chromosome,
		error,
		energy,
		delay,
		depth,
		gate_count
	);
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
double CGP::mse(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const
{
	double sum = 0;
	int i = 0;
	int end = output_count();
	//#pragma omp parallel for reduction(+:sum) default(shared) private(i, end)
	for (; i < end; i++)
	{
		if (CGPConfiguration::no_care_value == expected_output[i]) break;
		const double diff = predictions[i] - expected_output[i];
		sum += diff * diff;
	}

	// Overflow detected, discard result completely
	if (sum < 0) {
		return std::numeric_limits<double>::infinity();
	}

	return sum / i;
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
		mse_accumulator += error_fitness(*chrom, *output_it);
		if (mse_accumulator < 0) [[unlikely]]
			{
				mse_accumulator = std::numeric_limits<double>::infinity();
				break;
			}
	}
	mse_accumulator /= input.size();
	if (mse_accumulator <= mse_threshold())
	{
		return CGP::create_solution(chrom, mse_accumulator, get_energy_fitness(*chrom), get_delay_fitness(*chrom), get_depth_fitness(*chrom), get_gate_count(*chrom));
	}
	else {
		return CGP::create_solution(chrom, mse_accumulator);
	}
}

CGP::solution_t CGP::analyse_chromosome(std::shared_ptr<Chromosome> chrom, const std::shared_ptr<weight_value_t[]> input, const std::shared_ptr<weight_value_t[]> expected_output, size_t selector)
{
	chrom->set_input(input);
	chrom->evaluate(selector);
	auto mse = error_fitness(*chrom, expected_output);

	if (mse <= mse_threshold())
	{
		return CGP::create_solution(chrom, mse, get_energy_fitness(*chrom), get_delay_fitness(*chrom), get_depth_fitness(*chrom), get_gate_count(*chrom));
	}
	else {
		return CGP::create_solution(chrom, mse);
	}
}

double CGP::get_energy_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_energy_usage();
}

double CGP::get_delay_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_largest_delay();
}

size_t CGP::get_depth_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_largest_depth();
}

size_t CGP::get_gate_count(Chromosome& chrom)
{
	return chrom.get_node_count();
}

double CGP::error_fitness(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output) {
	return mse(chrom.begin_output(), expected_output);
}

double CGP::error_fitness_without_aggregation(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output) {
	return mse_without_division(chrom.begin_output(), expected_output);
}

inline CGP::solution_t CGP::get_default_solution()
{
	return std::make_tuple(
		std::numeric_limits<double>::infinity(),
		std::numeric_limits<double>::infinity(),
		std::numeric_limits<double>::infinity(),
		std::numeric_limits<size_t>::max(),
		std::numeric_limits<size_t>::max(),
		nullptr
	);
}

inline CGP::solution_t CGP::create_solution(
	std::shared_ptr<Chromosome> chromosome,
	double error,
	double energy,
	double delay,
	size_t depth,
	size_t gate_count)
{
	return std::make_tuple(
		error,
		energy,
		delay,
		depth,
		gate_count,
		chromosome
	);
}

inline double CGP::get_error(const solution_t solution)
{
	return std::get<0>(solution);
}

inline double CGP::get_energy(const solution_t solution)
{
	return std::get<1>(solution);
}

inline double CGP::get_delay(const solution_t solution)
{
	return std::get<2>(solution);
}

inline size_t CGP::get_depth(const solution_t solution)
{
	return std::get<3>(solution);
}

inline size_t CGP::get_gate_count(const solution_t solution)
{
	return std::get<4>(solution);
}

inline std::shared_ptr<Chromosome> CGP::get_chromosome(const solution_t solution)
{
	return std::get<5>(solution);
}

void CGP::mutate() {
	auto best_chromosome = get_best_chromosome();
	const int end = population_max();
	int i;
	#pragma omp parallel for default(shared) private(i)
	for (i = 0; i < end; i++)
	{
		chromosomes[i] = best_chromosome->mutate();
	}
	evolution_steps_made++;
}

std::tuple<bool, bool> CGP::dominates(solution_t a, solution_t b) const
{
	const auto& a_error_fitness = CGP::get_error(a);
	const auto& a_energy_fitness = CGP::get_energy(a);
	const auto& a_delay = CGP::get_delay(a);
	const auto& a_chromosome = CGP::get_chromosome(a);

	const auto& b_error_fitness = CGP::get_error(b);
	const auto& b_energy_fitness = CGP::get_energy(b);
	const auto& b_delay = CGP::get_delay(b);
	const auto& b_chromosome = CGP::get_chromosome(b);

	if (!a_chromosome)
	{
		return  std::make_tuple(false, false);
	}

	const auto mse_t = mse_threshold();
	const bool error_domination = 
		mse_t < a_error_fitness && a_error_fitness <= b_error_fitness;
	const bool energy_domination = 
		mse_t >= a_error_fitness && a_energy_fitness < b_energy_fitness;
	const bool more_precise_same_energy_domination = 
		mse_t >= a_error_fitness && a_energy_fitness == b_energy_fitness && a_error_fitness < b_error_fitness;
	const bool delay_domination = 
		mse_t >= a_error_fitness && a_energy_fitness == b_energy_fitness && a_error_fitness == b_error_fitness && a_delay <= b_delay;
	const bool dominates = 
		!b_chromosome || error_domination || energy_domination || more_precise_same_energy_domination || delay_domination;
	const bool not_neutral = 
		(a_energy_fitness != b_energy_fitness || a_error_fitness != b_error_fitness || a_delay != b_delay);

	return std::make_tuple(dominates, not_neutral);
}

void CGP::evaluate(const std::shared_ptr<weight_value_t[]> input, const std::shared_ptr<weight_value_t[]> expected_output)
{
	return evaluate(std::vector{ input }, std::vector{ expected_output });
}

void CGP::evaluate(const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output) {
	const int end = population_max();
	int i;
	#pragma omp parallel for default(shared) private(i)
	for (i = 0; i < end; i++) {
		auto& chromosome = chromosomes[i];
		solution_t chromosome_result = analyse_chromosome(chromosome, input, expected_output);

		// The following section will set fitness and the chromosome
		// to best_solutions map if the fitness is better than the best fitness
		// known. Map collection was employed to make computation more efficient
		// on CPU with multiple cores, and to avoid mutex blocking.
		auto thread_id = omp_get_thread_num();
		auto &best_thread_solution = best_solutions[thread_id];

		if (std::get<0>(dominates(chromosome_result, best_thread_solution))) {
			best_thread_solution = std::move(chromosome_result);
		}
	}

	generations_without_change++;
	// Find the best chromosome accros threads and save it to the best chromosome variable
	// including its fitness.
	const auto threads_count = omp_get_max_threads();
	for (int i = 0; i < threads_count; i++)
	{
		const auto &best_thread_solution = best_solutions[i];
		auto result = dominates(best_thread_solution, best_solution);
		// Allow neutral evolution for error
		if (std::get<0>(result)) {
			if (std::get<1>(result)) {
				generations_without_change = 0;
			}
			best_solution = best_thread_solution;
		}
	}
}

double CGP::get_best_error_fitness() const {
	return std::get<0>(best_solution);
}

double CGP::get_best_energy_fitness() const {
	return std::get<1>(best_solution);
}

double CGP::get_best_delay_fitness() const
{
	return std::get<2>(best_solution);
}

size_t CGP::get_best_depth() const
{
	return std::get<3>(best_solution);
}

size_t CGP::get_best_gate_count() const
{
	return std::get<4>(best_solution);
}

std::shared_ptr<Chromosome> CGP::get_best_chromosome() const {
	return std::get<5>(best_solution);
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
	const std::string& solution_format,
	const size_t mutations_made
)
{
	generations_without_change = 0;
	evolution_steps_made = (mutations_made != std::numeric_limits<size_t>::max()) ? (mutations_made) : (evolution_steps_made);
	set_best_solution(solution, solution_format);
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
