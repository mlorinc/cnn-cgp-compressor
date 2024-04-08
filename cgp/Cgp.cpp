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
	generations_without_change(0),
	evolution_steps_made(0),
	chromosomes(std::vector<std::shared_ptr<Chromosome>>(omp_get_max_threads(), nullptr))
{
	best_solutions_size = omp_get_max_threads();
	best_solutions = std::make_unique<solution_t[]>(best_solutions_size);
}

CGP::CGP() :
	best_solution(CGP::get_default_solution()),
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
		<< error_to_string(get_error(best_solution)) << " "
		<< energy_to_string(get_energy(best_solution)) << " "
		<< delay_to_string(get_delay(best_solution)) << " "
		<< depth_to_string(get_depth(best_solution)) << " "
		<< gate_count_to_string(get_gate_count(best_solution)) << " "
		<< chromosome << std::endl;
	out << "evolution_steps_made: " << evolution_steps_made << std::endl;
	for (auto it = other_config_attribitues.cbegin(), end = other_config_attribitues.cend(); it != end; it++)
	{
		out << it->first << ": " << it->second << std::endl;
	}
}

void CGP::dump_all(std::ostream& out)
{
	get_best_error_fitness();
	get_best_energy_fitness();
	get_best_delay_fitness();
	get_best_depth();
	get_best_gate_count();
	dump(out);
}

std::map<std::string, std::string> CGP::load(std::istream& in, const std::vector<std::string>& arguments)
{
	auto remaining_data = CGPConfiguration::load(in);
	std::string best_solution_string;
	std::string best_solution_format = default_solution_format;

	other_config_attribitues.clear();
	for (auto it = remaining_data.cbegin(), end = remaining_data.cend(); it != end;) {
		const std::string& key = it->first;
		const std::string& value = it->second;

		if (key == "best_solution") {
			best_solution_string = value;
			remaining_data.erase(it++);
		}
		else if (key == "evolution_steps_made") {
			evolution_steps_made = std::stoull(value);
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

	if ((start_generation() != 0 || start_run() != 0) && !best_chromosome)
	{
		throw std::invalid_argument("cannot resume evolution without starting chromosome");
	}

	const int end = population_max();
	int i;
#pragma omp parallel for default(shared) private(i)
	// Create new population
	for (i = 0; i < end; i++)
	{
		chromosomes[i] = (best_chromosome) ? (best_chromosome->mutate()) : (std::make_shared<Chromosome>(*this, minimum_output_indicies));;
	}
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
CGP::error_t CGP::mse_without_division(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const
{
	CGP::error_t sum = 0;
	// #pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < output_count(); i++)
	{
		if (expected_output[i] == no_care_value) break;
		error_t diff = predictions[i] - expected_output[i];
		sum += diff * diff;
	}

	// Overflow detected, discard result completely
	if (sum < 0) {
		return error_nan;
	}

	return sum;
}

void CGP::set_best_solution(const std::string& solution, std::string format_string)
{
	best_solution = create_solution(solution, format_string);
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
CGP::error_t CGP::mse(const weight_value_t* predictions, const std::shared_ptr<weight_value_t[]> expected_output) const
{
	error_t sum = 0;
	int i = 0;
	int end = output_count();
	//#pragma omp parallel for reduction(+:sum) default(shared) private(i, end)
	for (; i < end; i++)
	{
		if (expected_output[i] == no_care_value) break;
		const error_t diff = predictions[i] - expected_output[i];
		sum += diff * diff;
	}

	// Overflow detected, discard result completely
	if (sum < 0) {
		return error_nan;
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
	return CGP::create_solution(chrom, mse_accumulator /= input.size());
}

CGP::solution_t CGP::analyse_chromosome(std::shared_ptr<Chromosome> chrom, const std::shared_ptr<weight_value_t[]> input, const std::shared_ptr<weight_value_t[]> expected_output, size_t selector)
{
	chrom->set_input(input);
	chrom->evaluate(selector);
	return CGP::create_solution(chrom, error_fitness(*chrom, expected_output));
}

CGP::energy_t CGP::get_energy_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_energy_usage();
}

CGP::area_t CGP::get_area_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_area_usage();
}

CGP::delay_t CGP::get_delay_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_largest_delay();
}

CGP::depth_t CGP::get_depth_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_largest_depth();
}

CGP::gate_count_t CGP::get_gate_count(Chromosome& chrom)
{
	return chrom.get_node_count();
}

CGP::error_t CGP::error_fitness(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output) {
	return mse(chrom.begin_output(), expected_output);
}

CGP::error_t CGP::error_fitness_without_aggregation(Chromosome& chrom, const std::shared_ptr<weight_value_t[]> expected_output) {
	return mse_without_division(chrom.begin_output(), expected_output);
}

CGP::solution_t CGP::get_default_solution()
{
	return std::make_tuple(
		error_nan,
		energy_nan,
		area_nan,
		delay_nan,
		depth_nan,
		gate_count_nan,
		nullptr
	);
}

CGP::solution_t CGP::create_solution(
	std::shared_ptr<Chromosome> chromosome,
	error_t error,
	energy_t energy,
	area_t area,
	delay_t delay,
	depth_t depth,
	gate_count_t gate_count)
{
	return std::make_tuple(
		error,
		energy,
		area,
		delay,
		depth,
		gate_count,
		chromosome
	);
}

CGP::solution_t CGP::create_solution(std::string solution, std::string format)
{
	std::istringstream iss(solution), format_stream(format);
	std::string attribute_name, attribute_value;

	std::string chrom;
	error_t error = error_nan;
	area_t area = area_nan;
	energy_t energy = energy_nan;
	delay_t delay = delay_nan;
	depth_t depth = depth_nan;
	gate_count_t gate_count = gate_count_nan;

	while (format_stream >> attribute_name)
	{
		if (iss >> attribute_value)
		{
			if (attribute_name == "error")
			{
				error = (attribute_value == error_nan_string) ? (error_nan) : std::stold(attribute_value);
			}
			else if (attribute_name == "energy")
			{
				energy = (attribute_value == energy_nan_string) ? (energy_nan) : std::stold(attribute_value);
			}
			else if (attribute_name == "area")
			{
				area = (attribute_value == area_nan_string) ? (area_nan) : std::stold(attribute_value);
			}
			else if (attribute_name == "delay")
			{
				delay = (attribute_value == delay_nan_string) ? (delay_nan) : std::stold(attribute_value);
			}
			else if (attribute_name == "depth")
			{
				depth = (attribute_value == depth_nan_string) ? (depth_nan) : std::stoull(attribute_value);
			}
			else if (attribute_name == "gate_count")
			{
				gate_count = (attribute_value == gate_count_nan_string) ? (gate_count_nan) : std::stoull(attribute_value);
			}
			else if (attribute_name == "chromosome")
			{
				chrom = attribute_value;
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

	if (!format_stream.eof())
	{
		throw std::invalid_argument("all attributes were not consumed: " + format_stream.str());
	}

	if (!iss.eof())
	{
		throw std::invalid_argument("all attribute values were not consumed: " + iss.str());
	}

	auto chromosome = std::make_shared<Chromosome>(*this, minimum_output_indicies, chrom);
	return CGP::create_solution(
		chromosome,
		error,
		energy,
		area,
		delay,
		depth,
		gate_count
	);
}

CGP::error_t CGP::get_error(const solution_t solution)
{
	return std::get<0>(solution);
}

CGP::energy_t CGP::get_energy(const solution_t solution)
{
	return std::get<1>(solution);
}

CGP::area_t CGP::get_area(const solution_t solution)
{
	return std::get<2>(solution);
}

CGP::delay_t CGP::get_delay(const solution_t solution)
{
	return std::get<3>(solution);
}

CGP::depth_t CGP::get_depth(const solution_t solution)
{
	return std::get<4>(solution);
}

CGP::gate_count_t CGP::get_gate_count(const solution_t solution)
{
	return std::get<5>(solution);
}

std::shared_ptr<Chromosome> CGP::get_chromosome(const solution_t solution)
{
	return std::get<6>(solution);
}

decltype(CGP::get_error(CGP::solution_t())) CGP::set_error(solution_t& solution, decltype(CGP::get_error(solution_t())) value)
{
	std::get<0>(solution) = value;
	return value;
}

decltype(CGP::get_energy(CGP::solution_t())) CGP::set_energy(solution_t& solution, decltype(CGP::get_energy(solution_t())) value)
{
	std::get<1>(solution) = value;
	return value;
}

decltype(CGP::get_area(CGP::solution_t())) CGP::set_area(solution_t& solution, decltype(CGP::get_area(solution_t())) value)
{
	std::get<2>(solution) = value;
	return value;
}

decltype(CGP::get_delay(CGP::solution_t())) CGP::set_delay(solution_t& solution, decltype(CGP::get_delay(solution_t())) value)
{
	std::get<3>(solution) = value;
	return value;
}

decltype(CGP::get_depth(CGP::solution_t())) CGP::set_depth(solution_t& solution, decltype(CGP::get_depth(solution_t())) value)
{
	std::get<4>(solution) = value;
	return value;
}

decltype(CGP::get_gate_count(CGP::solution_t())) CGP::set_gate_count(solution_t& solution, decltype(CGP::get_gate_count(solution_t())) value)
{
	std::get<5>(solution) = value;
	return value;
}

decltype(CGP::get_energy(CGP::solution_t())) CGP::ensure_energy(solution_t& solution)
{
	if (CGP::get_energy(solution) == energy_nan)
	{
		CGP::set_energy(solution, CGP::get_chromosome(solution)->get_estimated_energy_usage());
	}

	return CGP::get_energy(solution);
}

decltype(CGP::get_area(CGP::solution_t())) CGP::ensure_area(solution_t& solution)
{
	if (CGP::get_area(solution) == area_nan)
	{
		CGP::set_area(solution, CGP::get_chromosome(solution)->get_estimated_area_usage());
	}

	return CGP::get_area(solution);
}

decltype(CGP::get_delay(CGP::solution_t())) CGP::ensure_delay(solution_t& solution)
{
	if (CGP::get_delay(solution) == delay_nan)
	{
		CGP::set_delay(solution, CGP::get_chromosome(solution)->get_estimated_largest_delay());
	}

	return CGP::get_delay(solution);
}

decltype(CGP::get_depth(CGP::solution_t())) CGP::ensure_depth(solution_t& solution)
{
	if (CGP::get_depth(solution) == depth_nan)
	{
		CGP::set_depth(solution, CGP::get_chromosome(solution)->get_estimated_largest_depth());
	}

	return CGP::get_depth(solution);
}

decltype(CGP::get_gate_count(CGP::solution_t())) CGP::ensure_gate_count(solution_t& solution)
{
	if (CGP::get_gate_count(solution) == gate_count_nan)
	{
		CGP::set_gate_count(solution, CGP::get_chromosome(solution)->get_node_count());
	}

	return CGP::get_gate_count(solution);
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

std::tuple<bool, bool> CGP::dominates(solution_t& a, solution_t& b) const
{
	const auto& a_chromosome = CGP::get_chromosome(a);
	const auto& b_chromosome = CGP::get_chromosome(b);

	if (!a_chromosome)
	{
		return std::make_tuple(false, true);
	}

	if (!b_chromosome)
	{
		return std::make_tuple(true, false);
	}

	const auto mse_t = mse_threshold();
	const auto& a_error_fitness = CGP::get_error(a);
	const auto& b_error_fitness = CGP::get_error(b);
	if (mse_t < a_error_fitness)
	{
		const bool dominates = a_error_fitness <= b_error_fitness;
		const bool neutral = a_error_fitness == a_error_fitness;
		return std::make_tuple(dominates, neutral);
	}
	else
	{
		const auto& a_energy_fitness = CGP::ensure_energy(a);
		const auto& b_energy_fitness = CGP::ensure_energy(b);

		if (a_energy_fitness < b_energy_fitness)
		{
			return std::make_tuple(true, false);
		}

		if (a_energy_fitness == b_energy_fitness && a_error_fitness < b_error_fitness)
		{
			return std::make_tuple(true, false);
		}

		const auto& a_delay = CGP::ensure_delay(a);
		const auto& b_delay = CGP::ensure_delay(b);

		if (a_energy_fitness == b_energy_fitness && a_error_fitness == b_error_fitness && a_delay <= b_delay)
		{
			return std::make_tuple(true, a_delay == b_delay);
		}
	}

	return std::make_tuple(false, true);
}

void CGP::evaluate(const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output)
{
	const int end = population_max();
	int i;
#pragma omp parallel for default(shared) private(i)
	for (i = 0; i < end; i++) {
		auto& chromosome = chromosomes[i];
		auto chromosome_result = analyse_chromosome(chromosome, input, expected_output);

		// The following section will set fitness and the chromosome
		// to best_solutions map if the fitness is better than the best fitness
		// known. Map collection was employed to make computation more efficient
		// on CPU with multiple cores, and to avoid mutex blocking.
		auto thread_id = omp_get_thread_num();
		auto& best_thread_solution = best_solutions[thread_id];

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
		auto& best_thread_solution = best_solutions[i];
		auto result = dominates(best_thread_solution, best_solution);
		// Allow neutral evolution for error
		if (std::get<0>(result)) {
			// check whether mutation was not neutral
			if (!std::get<1>(result)) {
				generations_without_change = 0;
			}
			best_solution = best_thread_solution;
		}
	}
}

CGP::solution_t CGP::evaluate(const std::vector<std::shared_ptr<weight_value_t[]>>& input, const std::vector<std::shared_ptr<weight_value_t[]>>& expected_output, std::shared_ptr<Chromosome> chromosome)
{
	auto solution = analyse_chromosome(chromosome, input, expected_output);

	return CGP::create_solution(
		CGP::get_chromosome(solution),
		CGP::get_error(solution),
		CGP::ensure_energy(solution),
		CGP::ensure_delay(solution),
		CGP::ensure_depth(solution),
		CGP::ensure_gate_count(solution)
	);
}

CGP::error_t CGP::get_best_error_fitness() const {
	return get_energy(best_solution);
}

CGP::energy_t CGP::get_best_energy_fitness() {
	return ensure_energy(best_solution);
}

CGP::area_t CGP::get_best_area_fitness() {
	return ensure_area(best_solution);
}

CGP::delay_t CGP::get_best_delay_fitness()
{
	return ensure_delay(best_solution);
}

decltype(CGP::evolution_steps_made) cgp::CGP::get_evolution_steps_made() const
{
	return evolution_steps_made;
}

CGP::depth_t CGP::get_best_depth()
{
	return ensure_depth(best_solution);
}

CGP::gate_count_t CGP::get_best_gate_count()
{
	return ensure_gate_count(best_solution);
}

std::shared_ptr<Chromosome> CGP::get_best_chromosome() const {
	return std::get<6>(best_solution);
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
