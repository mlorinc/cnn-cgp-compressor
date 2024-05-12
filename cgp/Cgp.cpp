#include "Cgp.h"
#include <algorithm>
#include <limits>
#include <omp.h>
#include <fstream>
#include <sstream>
#include "Assert.h"

using namespace cgp;

#define best_solution (chromosomes[0])

#if defined(__MEAN_SQUARED_ERROR_METRIC)
#define _error_mx_fitness(chrom, dataset, i) mx_mse_error(chrom->begin_output(), weights)
#elif defined(__ABSOLUTE_ERROR_METRIC)
#define _error_mx_fitness(chrom, dataset, i) mx_ae_error(chrom->begin_output(), weights)
#else
#define _error_mx_fitness(chrom, dataset, i) mx_se_error(chrom->begin_output(), weights)
#endif

#if defined(__MEAN_SQUARED_ERROR_METRIC)
#define _error_normal_fitness(chrom, dataset, i) mse_error(chrom->begin_output(), get_dataset_output(dataset)[i], get_dataset_no_care(dataset)[i])
#elif defined(__ABSOLUTE_ERROR_METRIC)
#define _error_normal_fitness(chrom, dataset, i) ae_error(chrom->begin_output(), get_dataset_output(dataset)[i], get_dataset_no_care(dataset)[i])
#else
#define _error_normal_fitness(chrom, dataset, i) se_error(chrom->begin_output(), get_dataset_output(dataset)[i], get_dataset_no_care(dataset)[i])
#endif


#ifdef __SINGLE_MULTIPLEX
#define _error_fitness(chrom, dataset, i) _error_mx_fitness(chrom, dataset, i)
#else
#define _error_fitness(chrom, dataset, i) _error_normal_fitness(chrom, dataset, i)
#endif


CGP::CGP(int population) :
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
		population_max(omp_get_max_threads() / 2 + 1);
	}
	else
	{
		population_max(std::min(population, omp_get_max_threads()) / 2 + 1);
	}
	reset();
}

void CGP::reset()
{
	chromosomes = std::vector<solution_t>(population_max());
	generations_without_change = 0;
	evolution_steps_made = 0;
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

bool cgp::CGP::is_multiplexing() const
{
	const auto& chromosome = get_chromosome(best_solution);
	return chromosome && chromosome->is_multiplexing();
}

void cgp::CGP::perform_correction(const dataset_t& dataset, bool only_id)
{
	return;
	auto chromosome = get_chromosome(best_solution);

	if (!chromosome->is_multiplexing())
	{
		return;
	}

	if (dataset_size() == 1)
	{
		// Find better connections than found in grid, however do not tamper with energy if
		// the threshold is not zero
		chromosome->perform_corrections(dataset, 512, true, only_id);
		analyse_solution(best_solution, dataset);
	}
}

void cgp::CGP::remove_multiplexing(const dataset_t& dataset)
{
#pragma omp parallel for
	for (int i = 0; i < chromosomes.size(); i++)
	{
		get_chromosome(chromosomes[i])->remove_multiplexing(dataset);
	}

	solution_t new_solution = evaluate(dataset, get_chromosome(best_solution));
	best_solution = std::move(new_solution);
}

void cgp::CGP::set_generations_without_change(decltype(generations_without_change) new_value)
{
	generations_without_change = new_value;
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

void CGP::generate_population(const dataset_t& dataset)
{
	const auto& best_chromosome = get_best_chromosome();
	const int end = population_max();
	// Create new population
	#pragma omp parallel for
	for (int i = 0; i < end; i++)
	{
		if (best_chromosome)
		{
			get_chromosome(chromosomes[i]) = best_chromosome->mutate(rand() * omp_get_thread_num() * (i + 1) * (i + 1));
		}
		else
		{
			chromosomes[i] = create_solution(std::make_shared<Chromosome>(*this, minimum_output_indicies), error_nan);
#ifndef __NO_POW_SOLUTIONS
			//chromosomes[i]->add_2pow_circuits(dataset);
#endif // !__NO_DIRECT_SOLUTIONS
#if defined(__SINGLE_MULTIPLEX) || defined(__MULTIPLE_MULTIPLEX)
			get_chromosome(chromosomes[i])->use_multiplexing(dataset);
#endif
			//#ifndef __NO_DIRECT_SOLUTIONS
			//				get_chromosome(chromosomes[i])->find_direct_solutions(dataset);
			//#endif // !__NO_DIRECT_SOLUTIONS
		}
	}

	use_quantity_weights(dataset);
}

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
CGP::error_t CGP::se_error(const weight_value_t* predictions, const weight_output_t& expected_output, const int no_care) const
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

// MSE loss function implementation;
// for reference see https://en.wikipedia.org/wiki/Mean_squared_error
CGP::error_t CGP::mse_error(const weight_value_t* predictions, const weight_output_t& expected_output, const int no_care) const
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

cgp::CGP::error_t cgp::CGP::ae_error(const weight_value_t* predictions, const weight_output_t& expected_output, const int no_care) const
{
	CGP::error_t sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < no_care; i++)
	{
		const int a = predictions[i];
		const int b = expected_output[i];
		const int delta = std::abs(a - b);
		sum += delta;
	}
	return sum;
}

CGP::error_t CGP::mx_mse_error(const weight_value_t* predictions, const std::array<int, 256>& weights) const
{
	return mx_se_error(predictions, weights) / 256;
}

CGP::error_t CGP::mx_ae_error(const weight_value_t* predictions, const std::array<int, 256>& weights) const
{
	CGP::error_t sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < 256; i++)
	{
		const int a = predictions[i];
		const int b = i - 128;
		const int delta = std::abs(a - b);
		sum += delta * weights[i];
	}
	return sum;
}
CGP::error_t CGP::mx_se_error(const weight_value_t* predictions, const std::array<int, 256>& weights) const
{
	CGP::error_t sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < 256; i++)
	{
		const int a = predictions[i];
		const int b = i - 128;
		const int delta = a - b;
		const int delta_squared = delta * delta;
		sum += delta_squared * weights[i];
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

void CGP::analyse_solution(solution_t& solution, const dataset_t& dataset)
{
	auto chromosome = get_chromosome(solution);
	if (!chromosome->needs_evaluation())
	{
		return;
	}

	const auto& input = get_dataset_input(dataset);
	const int end = input.size();
	error_t mse_accumulator = 0;

	for (int i = 0; i < end; i++)
	{
		chromosome->set_input(input[i], i);
		chromosome->evaluate();
		mse_accumulator += _error_fitness(chromosome, dataset, i);
	}

	set_error(solution, mse_accumulator);
}

CGP::solution_t CGP::analyse_chromosome(std::shared_ptr<Chromosome> chrom, const dataset_t& dataset)
{
	const auto& input = get_dataset_input(dataset);
	const int end = input.size();
	error_t mse_accumulator = 0;

	for (int i = 0; i < end; i++)
	{
		chrom->set_input(input[i], i);
		chrom->evaluate();
		mse_accumulator += _error_normal_fitness(chrom, dataset, i);
	}

	return CGP::create_solution(chrom, mse_accumulator);
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

void CGP::set_error(solution_t& solution, decltype(CGP::get_error(solution_t())) value)
{
	std::get<0>(solution) = value;
}

void CGP::set_quantized_energy(solution_t& solution, decltype(CGP::get_quantized_energy(solution_t())) value)
{
	std::get<1>(solution) = value;
}

void CGP::set_energy(solution_t& solution, decltype(CGP::get_energy(solution_t())) value)
{
	std::get<2>(solution) = value;
}

void CGP::set_area(solution_t& solution, decltype(CGP::get_area(solution_t())) value)
{
	std::get<3>(solution) = value;
}

void CGP::set_quantized_delay(solution_t& solution, decltype(CGP::get_quantized_delay(solution_t())) value)
{
	std::get<4>(solution) = value;
}

void CGP::set_delay(solution_t& solution, decltype(CGP::get_delay(solution_t())) value)
{
	std::get<5>(solution) = value;
}

void CGP::set_depth(solution_t& solution, decltype(CGP::get_depth(solution_t())) value)
{
	std::get<6>(solution) = value;
}

void CGP::set_gate_count(solution_t& solution, decltype(CGP::get_gate_count(solution_t())) value)
{
	std::get<7>(solution) = value;
}

void CGP::set_chromosome(solution_t& solution, std::shared_ptr<Chromosome> value)
{
	std::get<8>(solution) = value;
}

decltype(CGP::get_depth(CGP::solution_t())) CGP::ensure_depth(solution_t& solution)
{
	auto value = CGP::get_chromosome(solution)->get_estimated_depth();
	set_depth(solution, value);
	return value;
}

decltype(CGP::get_gate_count(CGP::solution_t())) CGP::ensure_gate_count(solution_t& solution)
{
	auto value = CGP::get_chromosome(solution)->get_node_count();
	set_gate_count(solution, value);
	return value;
}

decltype(CGP::get_quantized_energy(CGP::solution_t())) CGP::ensure_quantized_energy(solution_t& solution)
{
	auto value = CGP::get_chromosome(solution)->get_estimated_quantized_energy_usage();
	set_quantized_energy(solution, value);
	return value;
}

decltype(CGP::get_energy(CGP::solution_t())) CGP::ensure_energy(solution_t& solution)
{
	auto value = CGP::get_chromosome(solution)->get_estimated_energy_usage();
	set_energy(solution, value);
	return value;
}

decltype(CGP::get_area(CGP::solution_t())) CGP::ensure_area(solution_t& solution)
{
	auto value = CGP::get_chromosome(solution)->get_estimated_area_usage();
	set_area(solution, value);
	return value;
}

decltype(CGP::get_quantized_delay(CGP::solution_t())) CGP::ensure_quantized_delay(solution_t& solution)
{
	auto value = CGP::get_chromosome(solution)->get_estimated_quantized_delay();
	set_quantized_delay(solution, value);
	return value;
}

decltype(CGP::get_delay(CGP::solution_t())) CGP::ensure_delay(solution_t& solution)
{
	auto value = CGP::get_chromosome(solution)->get_estimated_delay();
	set_delay(solution, value);
	return value;
}


void CGP::mutate(const dataset_t& dataset) {
	auto best_chromosome = get_best_chromosome();
	const int end = population_max();
#pragma omp parallel for default(shared)
	for (int i = 1; i < end; i++)
	{
		auto chromosome = get_chromosome(chromosomes[i]);
		best_chromosome->mutate(chromosome, dataset);
		if (chromosome->needs_evaluation())
		{
			set_error(chromosomes[i], error_nan);
		}
		else
		{
			set_error(chromosomes[i], get_error(best_solution));
		}
	}
	evolution_steps_made++;
}

void cgp::CGP::calculate_energy_threshold()
{
	if (function_count() > CGPOperator::MUX)
	{
		auto mux_cost = get_quantized_energy_parameter(function_costs()[CGPOperator::MUX]);
		energy_threshold = mux_cost * output_count();
	}
	else
	{
		energy_threshold = 0;
	}
}

std::tuple<bool, bool> CGP::dominates(solution_t& a) const
{
	const auto& a_chromosome = CGP::get_chromosome(a);
	const auto& b_chromosome = CGP::get_chromosome(best_solution);
	assert(a_chromosome); assert(b_chromosome); assert(a_chromosome != b_chromosome);

	const auto mse_t = mse_threshold();
	const auto& a_error_fitness = CGP::get_error(a);
	const auto& b_error_fitness = CGP::get_error(best_solution);
	if (mse_t < a_error_fitness || mse_t < b_error_fitness)
	{
		return std::make_tuple(a_error_fitness <= b_error_fitness, a_error_fitness == b_error_fitness);
	}
	else
	{

		quantized_energy_t a_energy_fitness;
		quantized_delay_t a_delay;
		quantized_energy_t b_energy_fitness = CGP::get_quantized_energy(best_solution); assert(b_energy_fitness != quantized_energy_nan);
		quantized_delay_t b_delay = CGP::get_quantized_delay(best_solution); assert(b_delay != quantized_delay_nan);
#pragma omp parallel sections
		{
#pragma omp section
			{
				a_energy_fitness = CGP::ensure_quantized_energy(a);
				// Those values are calculate as part of the process already
				ensure_area(a); ensure_energy(a); ensure_gate_count(a);
			}

#pragma omp section
			{
				a_delay = CGP::ensure_quantized_delay(a);
				// Load real delays too
				ensure_delay(a);
			}
		}


#pragma omp barrier
		if (a_energy_fitness != b_energy_fitness)
		{
			return std::make_tuple(a_energy_fitness < b_energy_fitness, false);
		}

		if (a_error_fitness != b_error_fitness)
		{
			return std::make_tuple(a_error_fitness < b_error_fitness, false);
		}


		if (a_delay != b_delay)
		{
			return std::make_tuple(a_delay < b_delay, false);
		}

		const auto& a_gate_count = CGP::get_gate_count(a);
		const auto& b_gate_count = CGP::get_gate_count(best_solution); assert(b_gate_count != gate_count_nan);

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

std::tuple<bool, bool> CGP::dominates(solution_t& a, solution_t& b) const
{
	const auto& a_chromosome = CGP::get_chromosome(a);
	const auto& b_chromosome = CGP::get_chromosome(b);
	assert(a_chromosome); assert(b_chromosome); assert(a_chromosome != b_chromosome);

	const auto mse_t = mse_threshold();
	const auto& a_error_fitness = CGP::get_error(a);
	const auto& b_error_fitness = CGP::get_error(b);
	if (mse_t < a_error_fitness || mse_t < b_error_fitness)
	{
		return std::make_tuple(a_error_fitness <= b_error_fitness, a_error_fitness == b_error_fitness);
	}
	else
	{

		quantized_energy_t a_energy_fitness, b_energy_fitness;
#pragma omp parallel sections
		{
#pragma omp section
			{
				a_energy_fitness = CGP::ensure_quantized_energy(a);
				// Those values are calculate as part of the process already
				ensure_area(a); ensure_energy(a); ensure_gate_count(a);
			}

#pragma omp section
			{
				b_energy_fitness = CGP::ensure_quantized_energy(b);
				// Those values are calculate as part of the process already
				ensure_area(b); ensure_energy(b); ensure_gate_count(b);
			}
		}


#pragma omp barrier
		if (a_energy_fitness != b_energy_fitness)
		{
			return std::make_tuple(a_energy_fitness < b_energy_fitness, false);
		}

		if (a_error_fitness != b_error_fitness)
		{
			return std::make_tuple(a_error_fitness < b_error_fitness, false);
		}


		quantized_delay_t a_delay, b_delay;
#pragma omp parallel sections
		{
#pragma omp section
			{
				a_delay = CGP::ensure_quantized_delay(a);
				// Load real delays too
				ensure_delay(a);
			}

#pragma omp section
			{
				b_delay = CGP::ensure_quantized_delay(b);
				// Load real delays too
				ensure_delay(b);
			}
		}

#pragma omp barrier
		if (a_delay != b_delay)
		{
			return std::make_tuple(a_delay < b_delay, false);
		}

		const auto& a_gate_count = CGP::get_gate_count(a);
		const auto& b_gate_count = CGP::get_gate_count(b);

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

CGP::solution_t CGP::evaluate(const dataset_t& dataset)
{
	const int end = population_max();
#pragma omp parallel for default(shared)
	for (int i = 1; i < end; i++) {
		analyse_solution(chromosomes[i], dataset);
	}

	generations_without_change++;
	int max_i = 0;
	// Find the best chromosome accros threads and save it to the best chromosome variable
	// including its fitness.
	for (int i = 1; i < end; i++)
	{
		auto result = dominates(chromosomes[i], best_solution);
		// Allow neutral evolution for error
		if (std::get<0>(result)) {
			// check whether mutation was not neutral
			if (!std::get<1>(result)) {
				generations_without_change = 0;
				std::swap(best_solution, chromosomes[i]);
			}
			max_i = i;
		}
	}

	if (generations_without_change != 0 && max_i != 0)
	{
		std::swap(best_solution, chromosomes[max_i]);
	}

	return best_solution;
}

CGP::solution_t CGP::evaluate_single_multiplexed(const dataset_t& dataset)
{
	return evaluate(dataset);
}

CGP::solution_t CGP::evaluate_multi_multiplexed(const dataset_t& dataset)
{
	auto solution = evaluate_single_multiplexed(dataset);

	if (get_error(best_solution) == 0 && get_energy(best_solution) <= energy_threshold)
	{
		remove_multiplexing(dataset);
		set_generations_without_change(0);
	}

	return solution;
}

CGP::solution_t CGP::evaluate(const dataset_t& dataset, std::shared_ptr<Chromosome> chromosome)
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

void cgp::CGP::use_equal_weights(const dataset_t& dataset)
{
	weights = get_dataset_usage_vector(dataset);
}

void cgp::CGP::use_quantity_weights(const dataset_t& dataset)
{
	weights = get_dataset_needed_quant_values(dataset);
}
