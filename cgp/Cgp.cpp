#include "Cgp.h"
#include <algorithm>
#include <limits>
#include <omp.h>
#include <fstream>
#include <sstream>

using namespace cgp;

CGP::CGP(int population) :
	best_solution(CGP::get_default_solution()),
	generations_without_change(0),
	evolution_steps_made(0)
{
	prepare_population_structures(population);
}

CGP::CGP() : CGP(population_max()) {}

CGP::CGP(std::istream& in, int population, const std::vector<std::string>& arguments) : CGP(population)
{
	load(in, arguments);
}

CGP::CGP(std::istream& in, const std::vector<std::string>& arguments) : CGP(in, arguments, {})
{
	load(in, arguments);
	prepare_population_structures(population_max());
}

CGP::CGP(std::istream& in, const std::vector<std::string>& arguments, const dataset_t& dataset) :
	best_solution(CGP::get_default_solution()),
	generations_without_change(0),
	evolution_steps_made(0)
{
	load(in, arguments, dataset);
	prepare_population_structures(population_max());
}

CGP::~CGP() {}

void CGP::prepare_population_structures(int population)
{
	if (population == 0)
	{
		population_max(omp_get_max_threads());
	}
	else
	{
		population_max(std::min(population, omp_get_max_threads()));
	}
	chromosomes = std::vector<std::shared_ptr<Chromosome>>(population_max(), nullptr);
	best_solutions = std::make_unique<solution_t[]>(population_max());
}

void CGP::reset()
{
	best_solution = CGP::get_default_solution();
	generations_without_change = 0;
	evolution_steps_made = 0;
	best_solutions = std::make_unique<solution_t[]>(population_max());
}

void CGP::dump(std::ostream& out) const
{
	CGPConfiguration::dump(out);
	const auto& chrom = CGP::get_chromosome(best_solution);
	std::string chromosome = (chrom) ? (chrom->to_string()) : (Chromosome::nan_chromosome_string);

	out << "best_solution: error:"
		<< error_to_string(get_error(best_solution))
		<< " quantized_energy:" << quantized_energy_to_string(get_quantized_energy(best_solution))
		<< " energy:" << energy_to_string(get_energy(best_solution))
		<< " area:" << area_to_string(get_area(best_solution))
		<< " quantized_delay:" << quantized_delay_to_string(get_quantized_delay(best_solution))
		<< " delay:" << delay_to_string(get_delay(best_solution))
		<< " depth:" << depth_to_string(get_depth(best_solution))
		<< " gate_count:" << gate_count_to_string(get_gate_count(best_solution))
		<< " chromosome:" << chromosome << std::endl;
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
	return load(in, arguments, {});
}

std::map<std::string, std::string> CGP::load(std::istream& in, const std::vector<std::string>& arguments, const dataset_t& dataset)
{
	auto remaining_data = CGPConfiguration::load(in);

	other_config_attribitues.clear();
	for (auto it = remaining_data.cbegin(), end = remaining_data.cend(); it != end;) {
		const std::string& key = it->first;
		const std::string& value = it->second;

		if (key == "evolution_steps_made") {
			evolution_steps_made = std::stoull(value);
			remaining_data.erase(it++);
		}
		else if (key == "")
		{

		}
		else {
			++it;
		}
	}

	set_from_arguments(arguments);
	build_indices();
	std::string chrom = starting_solution();
	
	if (!chrom.empty())
	{
		set_best_chromosome(chrom, dataset);
	}

	return remaining_data;
}

void CGP::build_indices()
{
	const auto input_column_pins = input_count();
	const auto fn_arity = function_output_arity();
	const auto row_count = this->row_count();
	const auto col_count = this->col_count();
	minimum_output_indicies = std::make_unique<std::tuple<int, int>[]>(col_count + 1);
	int l_back = look_back_parameter();

	// Calculate pin available according to defined L parameter.
	// Skip input pins (+1) and include output pins (+1)
	for (auto col = 1; col < col_count + 2; col++) {
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
#pragma omp parallel for default(shared)
	// Create new population
	for (int i = 0; i < end; i++)
	{
		chromosomes[i] = (best_chromosome) ? (best_chromosome->mutate(chromosomes[i])) : (std::make_shared<Chromosome>(*this, minimum_output_indicies));
	}
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
CGP::error_t CGP::mse_without_division(const weight_value_t* predictions, const weight_output_t& expected_output, const int no_care, const int layer) const
{
	CGP::error_t sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < no_care; i++)
	{
		const int a = predictions[i];
		const int b = expected_output[i];
		const int delta = a - b;
		const int delta_squared = delta * delta;
		sum += delta_squared;
	}
	return sum;
}

void CGP::set_best_chromosome(const std::string& solution, const dataset_t& dataset)
{
	if (minimum_output_indicies == nullptr)
	{
		build_indices();
	}

	auto chromosome = std::make_shared<Chromosome>(*this, minimum_output_indicies, solution);
	best_solution = evaluate(dataset, chromosome);
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
CGP::error_t CGP::mse(const weight_value_t* predictions, const weight_output_t& expected_output, const int no_care, const int layer) const
{
	error_t sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < no_care; i++)
	{
		const int a = predictions[i];
		const int b = expected_output[i];
		const int delta = a - b;
		const int delta_squared = delta * delta;
		sum += delta_squared;
	}

	return sum / no_care;
}

CGP::solution_t CGP::analyse_chromosome(std::shared_ptr<Chromosome> chrom, const dataset_t& dataset)
{
	const auto& input = get_dataset_input(dataset);
	const auto& expected_output = get_dataset_output(dataset);
	const auto& no_cares = get_dataset_no_care(dataset);
	const size_t end = input.size();
	error_t mse_accumulator = 0;
	for (int i = 0; i < end; i++)
	{
		chrom->set_input(input[i], i);
		chrom->evaluate();
		mse_accumulator += error_fitness_without_aggregation(*chrom, expected_output[i], no_cares[i], i);
	}

	return CGP::create_solution(chrom, mse_accumulator);
}

CGP::solution_t CGP::analyse_chromosome(std::shared_ptr<Chromosome> chrom, const weight_input_t& input, const weight_output_t& expected_output, const int no_care, int selector)
{
	chrom->set_input(input, selector);
	chrom->evaluate();
	return CGP::create_solution(chrom, error_fitness_without_aggregation(*chrom, expected_output, no_care, selector));
}

CGP::quantized_energy_t CGP::get_energy_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_quantized_energy_usage();
}

CGP::area_t CGP::get_area_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_area_usage();
}

CGP::quantized_delay_t CGP::get_delay_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_quantized_delay();
}

CGP::depth_t CGP::get_depth_fitness(Chromosome& chrom)
{
	return chrom.get_estimated_depth();
}

CGP::gate_count_t CGP::get_gate_count(Chromosome& chrom)
{
	return chrom.get_node_count();
}

CGP::error_t CGP::error_fitness(Chromosome& chrom, const weight_output_t &expected_output, const int no_care, const int layer) {
	return mse(chrom.begin_output(), expected_output, no_care, layer);
}

CGP::error_t CGP::error_fitness_without_aggregation(Chromosome& chrom, const weight_output_t& expected_output, const int no_care, const int layer) {
	return mse_without_division(chrom.begin_output(), expected_output, no_care, layer);
}

CGP::solution_t CGP::get_default_solution()
{
	return std::make_tuple(
		error_nan,
		quantized_energy_nan,
		energy_nan,
		area_nan,
		quantized_delay_nan,
		delay_nan,
		depth_nan,
		gate_count_nan,
		nullptr
	);
}

CGP::solution_t CGP::create_solution(
	std::shared_ptr<Chromosome> chromosome,
	error_t error,
	quantized_energy_t quantized_energy,
	energy_t energy,
	area_t area,
	quantized_delay_t quantized_delay,
	delay_t delay,
	depth_t depth,
	gate_count_t gate_count)
{
	return std::make_tuple(
		error,
		quantized_energy,
		energy,
		area,
		quantized_delay,
		delay,
		depth,
		gate_count,
		chromosome
	);
}

CGP::error_t CGP::get_error(const solution_t solution)
{
	return std::get<0>(solution);
}

CGP::quantized_energy_t CGP::get_quantized_energy(const solution_t solution)
{
	return std::get<1>(solution);
}

CGP::energy_t CGP::get_energy(const solution_t solution)
{
	return std::get<2>(solution);
}

CGP::area_t CGP::get_area(const solution_t solution)
{
	return std::get<3>(solution);
}

CGP::quantized_delay_t CGP::get_quantized_delay(const solution_t solution)
{
	return std::get<4>(solution);
}

CGP::delay_t CGP::get_delay(const solution_t solution)
{
	return std::get<5>(solution);
}

CGP::depth_t CGP::get_depth(const solution_t solution)
{
	return std::get<6>(solution);
}

CGP::gate_count_t CGP::get_gate_count(const solution_t solution)
{
	return std::get<7>(solution);
}

std::shared_ptr<Chromosome> CGP::get_chromosome(const solution_t solution)
{
	return std::get<8>(solution);
}

std::shared_ptr<Chromosome> CGP::get_best_chromosome() const {
	return std::get<8>(best_solution);
}

decltype(CGP::get_error(CGP::solution_t())) CGP::set_error(solution_t& solution, decltype(CGP::get_error(solution_t())) value)
{
	std::get<0>(solution) = value;
	return value;
}

decltype(CGP::get_quantized_energy(CGP::solution_t())) CGP::set_quantized_energy(solution_t& solution, decltype(CGP::get_quantized_energy(solution_t())) value)
{
	std::get<1>(solution) = value;
	return value;
}

decltype(CGP::get_energy(CGP::solution_t())) CGP::set_energy(solution_t& solution, decltype(CGP::get_energy(solution_t())) value)
{
	std::get<2>(solution) = value;
	return value;
}

decltype(CGP::get_area(CGP::solution_t())) CGP::set_area(solution_t& solution, decltype(CGP::get_area(solution_t())) value)
{
	std::get<3>(solution) = value;
	return value;
}

decltype(CGP::get_quantized_delay(CGP::solution_t())) CGP::set_quantized_delay(solution_t& solution, decltype(CGP::get_quantized_delay(solution_t())) value)
{
	std::get<4>(solution) = value;
	return value;
}

decltype(CGP::get_delay(CGP::solution_t())) CGP::set_delay(solution_t& solution, decltype(CGP::get_delay(solution_t())) value)
{
	std::get<5>(solution) = value;
	return value;
}

decltype(CGP::get_depth(CGP::solution_t())) CGP::set_depth(solution_t& solution, decltype(CGP::get_depth(solution_t())) value)
{
	std::get<6>(solution) = value;
	return value;
}

decltype(CGP::get_gate_count(CGP::solution_t())) CGP::set_gate_count(solution_t& solution, decltype(CGP::get_gate_count(solution_t())) value)
{
	std::get<7>(solution) = value;
	return value;
}

decltype(CGP::get_quantized_energy(CGP::solution_t())) CGP::ensure_quantized_energy(solution_t& solution)
{
	if (CGP::get_quantized_energy(solution) == quantized_energy_nan)
	{
		CGP::set_quantized_energy(solution, CGP::get_chromosome(solution)->get_estimated_quantized_energy_usage());
	}

	return CGP::get_quantized_energy(solution);
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

decltype(CGP::get_quantized_delay(CGP::solution_t())) CGP::ensure_quantized_delay(solution_t& solution)
{
	if (CGP::get_quantized_delay(solution) == quantized_delay_nan)
	{
		CGP::set_quantized_delay(solution, CGP::get_chromosome(solution)->get_estimated_quantized_delay());
	}

	return CGP::get_quantized_delay(solution);
}

decltype(CGP::get_delay(CGP::solution_t())) CGP::ensure_delay(solution_t& solution)
{
	if (CGP::get_delay(solution) == delay_nan)
	{
		CGP::set_delay(solution, CGP::get_chromosome(solution)->get_estimated_delay());
	}

	return CGP::get_delay(solution);
}

decltype(CGP::get_depth(CGP::solution_t())) CGP::ensure_depth(solution_t& solution)
{
	if (CGP::get_depth(solution) == depth_nan)
	{
		CGP::set_depth(solution, CGP::get_chromosome(solution)->get_estimated_depth());
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
#pragma omp parallel for default(shared)
	for (int i = 0; i < end; i++)
	{
		chromosomes[i] = best_chromosome->mutate(chromosomes[i]);
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
	if (mse_t < a_error_fitness || mse_t < b_error_fitness)
	{
		return std::make_tuple(a_error_fitness <= b_error_fitness, a_error_fitness == b_error_fitness);
	}
	else
	{
		const auto& a_energy_fitness = CGP::ensure_quantized_energy(a);
		const auto& b_energy_fitness = CGP::ensure_quantized_energy(b);

		// Area is calculated with energy, so put area into to the solution too
		ensure_area(a); ensure_area(b);
		// Real energy detto
		ensure_energy(a); ensure_energy(b);

		if (a_energy_fitness != b_energy_fitness)
		{
			return std::make_tuple(a_energy_fitness < b_energy_fitness, false);
		}

		if (a_error_fitness != b_error_fitness)
		{
			return std::make_tuple(a_error_fitness < b_error_fitness, false);
		}

		const auto& a_delay = CGP::ensure_quantized_delay(a);
		const auto& b_delay = CGP::ensure_quantized_delay(b);

		// Load real delays too
		ensure_delay(a); ensure_delay(b);
		if (a_delay != b_delay)
		{
			return std::make_tuple(a_delay < b_delay, false);
		}

		const auto& a_gate_count = CGP::ensure_gate_count(a);
		const auto& b_gate_count = CGP::ensure_gate_count(b);

		if (a_gate_count != b_gate_count)
		{
			return std::make_tuple(a_gate_count < b_gate_count, false);
		}

		#ifdef _DEPTH_ENABLED
		const auto& a_depth = CGP::ensure_depth(a);
		const auto& b_depth = CGP::ensure_depth(b);
		return std::make_tuple(a_depth <= b_depth, a_depth == b_depth);
		#else
		return std::make_tuple(false, true);
		#endif
	}
}

CGP::solution_t CGP::evaluate(const dataset_t &dataset)
{
	const int end = population_max();
#pragma omp parallel for default(shared)
	for (int i = 0; i < end; i++) {
		auto& chromosome = chromosomes[i];
		auto chromosome_result = analyse_chromosome(chromosome, dataset);

		// The following section will set fitness and the chromosome
		// to best_solutions map if the fitness is better than the best fitness
		// known. Map collection was employed to make computation more efficient
		// on CPU with multiple cores, and to avoid mutex blocking.
		if (std::get<0>(dominates(chromosome_result, best_solution))) {
			best_solutions[i] = chromosome_result;
		}
	}

	std::shared_ptr<Chromosome> last_chromosome = CGP::get_chromosome(best_solution);
	generations_without_change++;
	best_solution_changed = false;
	// Find the best chromosome accros threads and save it to the best chromosome variable
	// including its fitness.
	for (int i = 0; i < end; i++)
	{
		auto result = dominates(best_solutions[i], best_solution);
		// Allow neutral evolution for error
		if (std::get<0>(result)) {
			// check whether mutation was not neutral
			if (!std::get<1>(result)) {
				generations_without_change = 0;
			}
			best_solution = std::move(best_solutions[i]);
			best_solution_changed = true;
		}
	}

	if (last_chromosome != CGP::get_chromosome(best_solution))
	{
		CGP::ensure_quantized_energy(best_solution);
		CGP::ensure_energy(best_solution);
		CGP::ensure_area(best_solution);
		CGP::ensure_quantized_delay(best_solution);
		CGP::ensure_delay(best_solution);
		CGP::ensure_depth(best_solution);
		CGP::ensure_gate_count(best_solution);
	}

	return best_solution;
}

CGP::solution_t CGP::evaluate(const dataset_t &dataset, std::shared_ptr<Chromosome> chromosome)
{
	auto solution = analyse_chromosome(chromosome, dataset);

	return CGP::create_solution(
		chromosome,
		CGP::get_error(solution),
		CGP::ensure_quantized_energy(solution),
		CGP::ensure_energy(solution),
		CGP::ensure_area(solution),
		CGP::ensure_quantized_delay(solution),
		CGP::ensure_delay(solution),
		CGP::ensure_depth(solution),
		CGP::ensure_gate_count(solution)
	);
}

CGP::error_t CGP::get_best_error_fitness() const {
	return get_error(best_solution);
}

CGP::quantized_energy_t CGP::get_best_energy_fitness() {
	return ensure_quantized_energy(best_solution);
}

CGP::area_t CGP::get_best_area_fitness() {
	return ensure_area(best_solution);
}

CGP::quantized_delay_t CGP::get_best_delay_fitness()
{
	return ensure_quantized_delay(best_solution);
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

CGP::solution_t cgp::CGP::get_best_solution() const
{
	return CGP::create_solution(
		std::make_shared<Chromosome>(*CGP::get_chromosome(best_solution)),
		CGP::get_error(best_solution),
		CGP::get_quantized_energy(best_solution),
		CGP::get_energy(best_solution),
		CGP::get_area(best_solution),
		CGP::get_quantized_delay(best_solution),
		CGP::get_delay(best_solution),
		CGP::get_depth(best_solution),
		CGP::get_gate_count(best_solution)
	);
}

decltype(CGP::generations_without_change) CGP::get_generations_without_change() const {
	return generations_without_change;
}

decltype(CGP::best_solution_changed) CGP::has_best_solution_changed() const
{
	return best_solution_changed;
}

int CGP::get_serialized_chromosome_size() const
{
	// chromosome size + input information + output information
	return chromosome_size() * sizeof(gene_t) + 2 * sizeof(gene_t);
}

void CGP::restore(std::shared_ptr<Chromosome> chromosome,
	const dataset_t& dataset,
	const size_t mutations_made
)
{
	generations_without_change = 0;
	evolution_steps_made = (mutations_made != std::numeric_limits<size_t>::max()) ? (mutations_made) : (evolution_steps_made);
	best_solution = evaluate(dataset, chromosome);
}
