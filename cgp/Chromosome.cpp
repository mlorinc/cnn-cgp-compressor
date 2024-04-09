#include "Chromosome.h"
#include <iostream>
#include <numeric>
#include <sstream>
#include <stack>
#include <cassert>

using namespace cgp;

const std::string Chromosome::nan_chromosome_string = "null";

Chromosome::Chromosome(const CGPConfiguration& cgp_configuration, const std::shared_ptr<std::tuple<int, int>[]>& minimum_output_indicies) :
	cgp_configuration(cgp_configuration),
	minimum_output_indicies(minimum_output_indicies)
{
	setup_maps();
	setup_iterators();
	setup_chromosome();
}

cgp::Chromosome::Chromosome(const CGPConfiguration& cgp_config, const std::shared_ptr<std::tuple<int, int>[]>& minimum_output_indicies, const std::string& serialized_chromosome) :
	cgp_configuration(cgp_config),
	minimum_output_indicies(minimum_output_indicies)
{
	setup_chromosome(serialized_chromosome);
}

Chromosome::Chromosome(const CGPConfiguration& cgp_configuration, const std::string& serialized_chromosome) : Chromosome(cgp_configuration, nullptr, serialized_chromosome)
{
}

Chromosome::Chromosome(const Chromosome& that) noexcept :
	cgp_configuration(that.cgp_configuration),
	minimum_output_indicies(that.minimum_output_indicies) {
	input = that.input;
	need_evaluation = that.need_evaluation;
	need_energy_evaluation = that.need_energy_evaluation;
	need_delay_evaluation = that.need_delay_evaluation;
	need_depth_evaluation = that.need_depth_evaluation;
	setup_maps(that.chromosome);
	setup_iterators();
	estimated_energy_consumption = that.estimated_energy_consumption;
	estimated_area_utilisation = that.estimated_area_utilisation;
	estimated_largest_delay = that.estimated_largest_delay;
	estimated_largest_depth = that.estimated_largest_depth;
	phenotype_node_count = that.phenotype_node_count;

	if (!need_evaluation) [[likely]]
		{
			std::copy(that.output_pin_start, that.output_pin_end, output_pin_start);
		}
}

Chromosome::Chromosome(Chromosome&& that) noexcept :
	cgp_configuration(that.cgp_configuration),
	minimum_output_indicies(that.minimum_output_indicies) {
	input = std::move(that.input);
	need_evaluation = that.need_evaluation;
	need_energy_evaluation = that.need_energy_evaluation;
	need_delay_evaluation = that.need_delay_evaluation;
	need_depth_evaluation = that.need_depth_evaluation;
	setup_maps(std::move(that));
	setup_iterators(std::move(that));
	estimated_energy_consumption = that.estimated_energy_consumption;
	estimated_area_utilisation = that.estimated_area_utilisation;
	estimated_largest_delay = that.estimated_largest_delay;
	estimated_largest_depth = that.estimated_largest_depth;
	phenotype_node_count = that.phenotype_node_count;
}

bool Chromosome::is_function(size_t position) const
{
	return !is_output(position) && !is_input(position);
}


bool Chromosome::is_input(size_t position) const
{
	return !is_output(position) && position % (cgp_configuration.function_input_arity() + 1) != cgp_configuration.function_input_arity();
}


bool Chromosome::is_output(size_t position) const
{
	return position >= cgp_configuration.blocks_chromosome_size();
}


void Chromosome::setup_chromosome()
{
	auto ite = chromosome.get();
	const auto end = cgp_configuration.row_count() * cgp_configuration.col_count();

	for (auto i = 0; i < end; i++) {
		int column_index = static_cast<int>(i / cgp_configuration.row_count());
		// Possible input connections according to L parameter
		const auto column_values = minimum_output_indicies[column_index];
		auto min = std::get<0>(column_values);
		auto max = std::get<1>(column_values);

		// Block inputs
		for (auto j = 0; j < cgp_configuration.function_input_arity(); j++)
		{
			*ite++ = (rand() % (max - min)) + min;
		}
		// Block function
		*ite++ = rand() % cgp_configuration.function_count();
	}

	// Connect outputs
	const auto& output_values = minimum_output_indicies[cgp_configuration.col_count()];
	auto min = std::get<0>(output_values);
	auto max = std::get<1>(output_values);
	for (int i = 0; i < cgp_configuration.output_count(); i++) {
		*ite++ = (rand() % (max - min)) + min;
	}
}

void Chromosome::setup_chromosome(const std::string& serialized_chromosome)
{
	std::istringstream input(serialized_chromosome);

	char discard;
	size_t input_count, output_count, col_count, row_count, function_input_arity, l_back;

	// {n,n,n,n,n,n,5}
	input >> discard >> input_count >> discard >> output_count >> discard >> col_count >> discard >> row_count
		>> discard >> function_input_arity >> discard >> l_back >> discard >> discard >> discard;

	setup_maps();
	setup_iterators();
	size_t block_size = cgp_configuration.function_input_arity() + 1;
	size_t value;
	auto it = chromosome.get();
	for (size_t i = 0; i < col_count * row_count; i++)
	{
		// ([n]i,i,...,f)
		input >> discard >> discard >> input_count >> discard;

		for (size_t i = 0; i < block_size; i++)
		{
			input >> *it++ >> discard;
		}
	}

	// (n,n,...,n)
	input >> discard;
	for (size_t i = 0; i < output_count; i++)
	{
		input >> *it++ >> discard;
	}

	auto check_chromosome = to_string();
	assert(("Chromosome::Chromosome serialized chromosome does not correspond to built chromosome", check_chromosome == serialized_chromosome));
}

void Chromosome::setup_maps()
{
	setup_maps(nullptr);
}

void Chromosome::setup_maps(decltype(chromosome) chromosome)
{
	if (chromosome == nullptr)
	{
		chromosome = std::make_shared<gene_t[]>(cgp_configuration.chromosome_size());
	}

	this->chromosome = chromosome;
	pin_map = std::make_unique<weight_value_t[]>(cgp_configuration.pin_map_size());
	gate_parameters_map = std::make_unique<gate_parameters_t[]>(cgp_configuration.row_count() * cgp_configuration.col_count());
	gate_visit_map = std::make_unique<bool[]>(cgp_configuration.row_count() * cgp_configuration.col_count());
}

void Chromosome::setup_maps(Chromosome &&that)
{
	chromosome = std::move(that.chromosome);
	pin_map = std::move(that.pin_map);
	gate_parameters_map = std::move(that.gate_parameters_map);
	gate_visit_map = std::move(that.gate_visit_map);
}

void cgp::Chromosome::setup_iterators()
{
	output_start = chromosome.get() + cgp_configuration.blocks_chromosome_size();
	output_end = output_start + cgp_configuration.output_count();
	output_pin_start = pin_map.get() + cgp_configuration.row_count() * cgp_configuration.col_count() * cgp_configuration.function_output_arity();
	output_pin_end = output_pin_start + cgp_configuration.output_count();
}

void cgp::Chromosome::setup_iterators(Chromosome&& that)
{
	std::swap(output_start, that.output_start);
	std::swap(output_end, that.output_end);
	std::swap(output_pin_start, that.output_pin_start);
	std::swap(output_pin_end, that.output_pin_end);
}

Chromosome::weight_value_t cgp::Chromosome::plus(weight_value_t a, weight_value_t b)
{
	auto c = a + b;
	if (c > cgp_configuration.expected_value_max())
	{
		return c - 256;
	}
	else if (c < cgp_configuration.expected_value_min())
	{
		return 256 + c;
	}
	else
	{
		return c;
	}
}

Chromosome::weight_value_t cgp::Chromosome::minus(weight_value_t a, weight_value_t b)
{
	return plus(a, -b);
}

Chromosome::weight_value_t cgp::Chromosome::mul(weight_value_t a, weight_value_t b)
{
	auto c = a * b;
	int period = std::abs(c) / 128;
	int result = std::abs(c) % 128;
	if (c > cgp_configuration.expected_value_max())
	{
		return (period % 2 == 0) ? (result) : (cgp_configuration.expected_value_min() + result);
	}
	else if (c < cgp_configuration.expected_value_min())
	{
		return (period % 2 == 0) ? (-result) : (cgp_configuration.expected_value_max() + 1 - result);
	}
	else
	{
		return c;
	}
}

Chromosome::weight_value_t Chromosome::bit_shift(weight_value_t a)
{
	return a & CGPConfiguration::bit_shift_mask;
}

Chromosome::gene_t* Chromosome::get_outputs() const {
	return output_start;
}

Chromosome::gene_t* Chromosome::get_block_inputs(size_t row, size_t column) const {
	return get_block_inputs(row * column);
}

Chromosome::gene_t* cgp::Chromosome::get_block_inputs(size_t index) const
{
	return chromosome.get() + index * (cgp_configuration.function_input_arity() + 1);
}


Chromosome::gene_t* Chromosome::get_block_function(size_t row, size_t column) const {
	return chromosome.get() + (row * column) * (cgp_configuration.function_input_arity() + 1) + cgp_configuration.function_input_arity();
}


std::shared_ptr<Chromosome::gene_t[]> Chromosome::get_chromosome() const
{
	return chromosome;
}

void Chromosome::mutate_genes(std::shared_ptr<Chromosome> that) const
{
	// Number of genes to mutate
	auto genes_to_mutate = (rand() % cgp_configuration.max_genes_to_mutate()) + 1;
	for (auto i = 0; i < genes_to_mutate; i++) {
		// Select a random gene
		auto random_gene_index = rand() % cgp_configuration.chromosome_size();
		auto random_number = rand();

		if (is_input(random_gene_index))
		{
			int gate_index = random_gene_index / (cgp_configuration.function_input_arity() + 1);
			int column_index = (int)(gate_index / cgp_configuration.row_count());
			const auto& column_values = minimum_output_indicies[column_index];
			auto min = std::get<0>(column_values);
			auto max = std::get<1>(column_values);

			that->chromosome[random_gene_index] = (random_number % (max - min)) + min;
		}
		else if (is_output(random_gene_index)) {
			const auto& output_values = minimum_output_indicies[cgp_configuration.col_count()];
			auto min = std::get<0>(output_values);
			auto max = std::get<1>(output_values);
			that->chromosome[random_gene_index] = (random_number % (max - min)) + min;
		}
		else {
			that->chromosome[random_gene_index] = random_number % (cgp_configuration.function_count());
		}
	}
	that->need_evaluation = true;
	that->need_energy_evaluation = true;
	that->need_delay_evaluation = true;
	that->need_depth_evaluation = true;
}

std::shared_ptr<Chromosome> Chromosome::mutate() const
{
	auto chrom = std::make_shared<Chromosome>(*this);
	chrom->chromosome = std::make_shared<Chromosome::gene_t[]>(cgp_configuration.chromosome_size());
	std::copy(chromosome.get(), chromosome.get() + cgp_configuration.chromosome_size(), chrom->chromosome.get());
	chrom->setup_iterators();
	mutate_genes(chrom);
	chrom->input = nullptr;
	return chrom;
}

std::shared_ptr<Chromosome> cgp::Chromosome::mutate(std::shared_ptr<Chromosome> that)
{
	if (this == that.get())
	{
		return mutate();
	}

	std::copy(chromosome.get(), chromosome.get() + cgp_configuration.chromosome_size(), that->chromosome.get());
	mutate_genes(that);
	return that;
}

void Chromosome::set_input(std::shared_ptr<weight_value_t[]> input)
{
	if (this->input == input)
	{
		return;
	}

	this->input = input;
	need_evaluation = true;
	for (size_t i = 0; i < cgp_configuration.input_count(); i++)
	{
		pin_map[i] = input[i];
	}
}

inline static void set_value(CGPConfiguration::weight_value_t& target, const CGPConfiguration::weight_value_t& value)
{
	target = (value != CGPConfiguration::invalid_value) ? (value) : (CGPConfiguration::invalid_value);
}

bool Chromosome::is_mux(int func)
{
#ifdef CNN_FP32_WEIGHTS
	return func == 10;
#else
	return func == 28;
#endif
}

bool Chromosome::is_demux(int func)
{
#ifdef CNN_FP32_WEIGHTS
	return func == 11;
#else
	return func == 29;
#endif
}

void Chromosome::evaluate(size_t selector)
{
	assert(("Chromosome::evaluate cannot be called without calling Chromosome::set_input before", input != nullptr));

	if (!need_evaluation)
	{
		return;
	}

	auto output_pin = pin_map.get() + cgp_configuration.input_count();
	auto input_pin = chromosome.get();
	auto func = chromosome.get() + cgp_configuration.function_input_arity();
	auto reference_costs = cgp_configuration.function_costs();
	const auto expected_value_min = cgp_configuration.expected_value_min();
	const auto expected_value_max = cgp_configuration.expected_value_max();
	size_t used_pin = 0;
	for (size_t i = 0; i < cgp_configuration.col_count() * cgp_configuration.row_count(); i++)
	{
		auto block_output_pins = output_pin;
		gate_parameters_map[i] = reference_costs[*func];
		gate_visit_map[i] = false;
		switch (*func) {
		case 0:
			set_value(block_output_pins[0], pin_map[input_pin[0]]);
			break;
		case 1:
			set_value(block_output_pins[0], minus(expected_value_max, pin_map[input_pin[0]]));
			break;
		case 2:
			set_value(block_output_pins[0], plus(pin_map[input_pin[0]], pin_map[input_pin[1]]));
			break;
		case 3:
			set_value(block_output_pins[0], minus(pin_map[input_pin[0]], pin_map[input_pin[1]]));
			break;
		case 4:
			set_value(block_output_pins[0], mul(pin_map[input_pin[0]], pin_map[input_pin[1]]));
			break;
		case 5:
			set_value(block_output_pins[0], -pin_map[input_pin[0]]);
			break;
		case 6:
			set_value(block_output_pins[0], plus(expected_value_min, pin_map[input_pin[0]]));
			break;
#ifdef CNN_FP32_WEIGHTS
		case 7:
			set_value(block_output_pins[0], pin_map[input_pin[0]] * 0.25);
			break;
#else
		case 7:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] >> 2));
			break;
#endif
#ifdef CNN_FP32_WEIGHTS
		case 8:
			set_value(block_output_pins[0], pin_map[input_pin[0]] * 0.5);
			break;
#else
		case 8:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] >> 1));
			break;
#endif
#ifndef CNN_FP32_WEIGHTS
		case 9:
			set_value(block_output_pins[0], pin_map[input_pin[0]] & pin_map[input_pin[1]]);
			break;
		case 10:
			set_value(block_output_pins[0], pin_map[input_pin[0]] | pin_map[input_pin[1]]);
			break;
		case 11:
			set_value(block_output_pins[0], pin_map[input_pin[0]] ^ pin_map[input_pin[1]]);
			break;
		case 12:
			set_value(block_output_pins[0], ~(pin_map[input_pin[0]]));
			break;
		case 13:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] << 1));
			break;
		case 14:
			set_value(block_output_pins[0], plus(pin_map[input_pin[0]], 1));
			break;
		case 15:
			set_value(block_output_pins[0], minus(pin_map[input_pin[0]], 1));
			break;
		case 16:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] >> 3));
			break;
		case 17:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] >> 4));
			break;
		case 18:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] >> 5));
			break;
		case 19:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] << 2));
			break;
		case 20:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] << 3));
			break;
		case 21:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] << 4));
			break;
		case 22:
			set_value(block_output_pins[0], bit_shift(pin_map[input_pin[0]] << 5));
			break;
		case 23:
			set_value(block_output_pins[0], 1);
			break;
		case 24:
			set_value(block_output_pins[0], -1);
			break;
		case 25:
			set_value(block_output_pins[0], 0);
			break;
		case 26:
			set_value(block_output_pins[0], expected_value_min);
			break;
		case 27:
			set_value(block_output_pins[0], expected_value_max);
			break;
		// multiplexor
		case 28:
			set_value(block_output_pins[0], pin_map[input_pin[selector]]);
			break;
		// de-multiplexor
		case 29:
			set_value(block_output_pins[selector], pin_map[input_pin[0]]);
			used_pin = selector;
			break;
#else
		case 9:
			set_value(block_output_pins[0], pin_map[input_pin[0]] * 1.05);
			break;
		// multiplexor
		case 10:
			set_value(block_output_pins[0], pin_map[input_pin[selector]]);
			break;
		// de-multiplexor
		case 11:
			set_value(block_output_pins[selector], pin_map[input_pin[0]]);
			used_pin = selector;
			break;
#endif
		default:
			block_output_pins[0] = 0xffffffff;
			break;
		}

		if (is_demux(*func)) [[unlikely]]
		{
			for (size_t i = 0; i < cgp_configuration.function_output_arity(); i++)
			{
				set_value(block_output_pins[i], (i == used_pin) ? (block_output_pins[i]) : (0));
			}
		}
		else [[likely]]
		{
			// speed up evolution of operator + multiplexor; shortcircuit pins
			for (size_t i = 0; i < cgp_configuration.function_output_arity(); i++)
			{
				set_value(block_output_pins[i], block_output_pins[selector]);
			}
		}

		output_pin += cgp_configuration.function_output_arity();
		input_pin += cgp_configuration.function_input_arity() + 1;
		func += cgp_configuration.function_input_arity() + 1;
	}

	auto it = output_start;
	auto pin_it = output_pin_start;
	for (; it != output_end; it++, pin_it++)
	{
		*pin_it = pin_map[*it];
	}
	need_evaluation = false;
}

Chromosome::weight_value_t* Chromosome::begin_output()
{
	assert(("Chromosome::begin_output cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	return output_pin_start;
}


Chromosome::weight_value_t* Chromosome::end_output()
{
	assert(("Chromosome::end_output cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	return output_pin_end;
}

std::string Chromosome::to_string() const
{
	std::ostringstream result;

	result << "{" << static_cast<unsigned>(cgp_configuration.input_count()) << "," << static_cast<unsigned>(cgp_configuration.output_count()) << ","
		<< static_cast<unsigned>(cgp_configuration.col_count()) << "," << cgp_configuration.row_count() << ","
		<< static_cast<unsigned>(cgp_configuration.function_input_arity()) << "," << cgp_configuration.look_back_parameter() << ",5}";

	for (int i = 0; i < cgp_configuration.blocks_chromosome_size(); ++i) {
		if (i % (cgp_configuration.function_input_arity() + 1) == 0) {
			result << "(["
				<< (i / (cgp_configuration.function_input_arity() + 1)) + static_cast<int>(cgp_configuration.input_count())
				<< "]";
		}

		result << chromosome[i];

		((i + 1) % (cgp_configuration.function_input_arity() + 1) == 0) ? result << ")" : result << ",";
	}

	result << "(";

	for (int i = cgp_configuration.blocks_chromosome_size(); i < cgp_configuration.chromosome_size(); ++i) {
		if (i > cgp_configuration.blocks_chromosome_size()) {
			result << ",";
		}

		result << chromosome[i];
	}

	result << ")";

	return result.str();
}

size_t cgp::Chromosome::get_serialized_chromosome_size() const
{
	// chromosome size + input information + output information
	return cgp_configuration.chromosome_size() * sizeof(gene_t) + 2 * sizeof(gene_t);
}

decltype(Chromosome::estimated_energy_consumption) cgp::Chromosome::get_estimated_energy_usage()
{
	assert(("Chromosome::estimate_energy_usage cannot be called without calling Chromosome::evaluate before", !need_evaluation));

	if (!need_energy_evaluation)
	{
		return estimated_energy_consumption;
	}


	auto pin_it = output_pin_start;
	std::stack<gene_t> pins_to_visit;

	for (auto it = output_start; it != output_end; it++, pin_it++)
	{
		pins_to_visit.push(*it);
	}

	estimated_energy_consumption = 0;
	estimated_area_utilisation = 0;
	phenotype_node_count = 0;
	bottom_row = cgp_configuration.row_count();
	top_row = 0;

	while (!pins_to_visit.empty())
	{
		gene_t pin = pins_to_visit.top();
		pins_to_visit.pop();

		// if is CGP input pin
		if (pin < cgp_configuration.input_count())
		{
			continue;
		}

		const int gate_index = (pin - cgp_configuration.input_count()) / cgp_configuration.function_output_arity();

		if (!gate_visit_map[gate_index])
		{
			gate_visit_map[gate_index] = true;
			phenotype_node_count += 1;
			estimated_energy_consumption += CGPConfiguration::get_energy_parameter(gate_parameters_map[gate_index]);
			estimated_area_utilisation += CGPConfiguration::get_area_parameter(gate_parameters_map[gate_index]);
			gene_t* inputs = get_block_inputs(gate_index);

			for (size_t i = 0; i < cgp_configuration.function_input_arity(); i++)
			{
				pins_to_visit.push(inputs[i]);
			}
		}

		int row = gate_index / cgp_configuration.col_count();
		top_row = std::max(row, top_row);
		bottom_row = std::min(row, bottom_row);
	}

	need_energy_evaluation = false;
	return estimated_energy_consumption;
}

decltype(Chromosome::estimated_energy_consumption) Chromosome::get_estimated_area_usage()
{
	assert(("Chromosome::get_estimated_area_usage cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_energy_usage();
	return estimated_area_utilisation;
}

decltype(Chromosome::estimated_largest_delay) Chromosome::get_estimated_largest_delay()
{
	assert(("Chromosome::get_estimated_largest_delay cannot be called without calling Chromosome::evaluate before", !need_evaluation));

	if (!need_delay_evaluation)
	{
		return estimated_largest_delay;
	}

	auto pin_it = output_pin_start;
	std::stack<gene_t> pins_to_visit;
	std::stack<delay_t> current_delays;
	auto distance_map = std::make_unique<delay_t[]>(cgp_configuration.col_count() * cgp_configuration.row_count());

	estimated_largest_delay = 0;
	for (auto it = output_start; it != output_end; it++, pin_it++)
	{
		pins_to_visit.push(*it);
		current_delays.push(0);
	}
		
	while (!pins_to_visit.empty())
	{
		gene_t pin = pins_to_visit.top();
		delay_t current_delay = current_delays.top();
		pins_to_visit.pop();
		current_delays.pop();

		// if is CGP input pin
		if (pin < cgp_configuration.input_count())
		{
			estimated_largest_delay = std::max(estimated_largest_delay, current_delay);
			continue;
		}

		const auto gate_index = (pin - cgp_configuration.input_count()) / cgp_configuration.function_output_arity();
		const auto new_cost = current_delay + CGPConfiguration::get_delay_parameter(gate_parameters_map[gate_index]);

		if (new_cost > distance_map[gate_index])
		{
			distance_map[gate_index] = new_cost;
			gene_t* inputs = get_block_inputs(gate_index);
			for (size_t i = 0; i < cgp_configuration.function_input_arity(); i++)
			{
				pins_to_visit.push(inputs[i]);
				current_delays.push(new_cost);
			}
		}
	}

	need_delay_evaluation = false;
	return estimated_largest_delay;
}

decltype(Chromosome::estimated_largest_depth) Chromosome::get_estimated_largest_depth()
{
	assert(("Chromosome::get_estimated_largest_depth cannot be called without calling Chromosome::evaluate before", !need_evaluation));

	if (!need_depth_evaluation)
	{
		return estimated_largest_depth;
	}

	auto pin_it = output_pin_start;
	std::stack<gene_t> pins_to_visit;
	std::stack<delay_t> current_depths;
	auto distance_map = std::make_unique<delay_t[]>(cgp_configuration.col_count() * cgp_configuration.row_count());

	estimated_largest_depth = 0;
	for (auto it = output_start; it != output_end; it++, pin_it++)
	{
		pins_to_visit.push(*it);
		current_depths.push(0);
	}

	while (!pins_to_visit.empty())
	{
		gene_t pin = pins_to_visit.top();
		depth_t current_depth = current_depths.top();
		pins_to_visit.pop();
		current_depths.pop();

		// if is CGP input pin
		if (pin < cgp_configuration.input_count())
		{
			estimated_largest_depth = std::max(estimated_largest_depth, current_depth);
			continue;
		}

		const auto gate_index = (pin - cgp_configuration.input_count()) / cgp_configuration.function_output_arity();
		const auto new_cost = current_depth + 1;

		if (new_cost > distance_map[gate_index])
		{
			distance_map[gate_index] = new_cost;
			gene_t* inputs = get_block_inputs(gate_index);
			for (size_t i = 0; i < cgp_configuration.function_input_arity(); i++)
			{
				pins_to_visit.push(inputs[i]);
				current_depths.push(new_cost);
			}
		}
	}

	need_depth_evaluation = false;
	return estimated_largest_depth;
}

decltype(Chromosome::phenotype_node_count) cgp::Chromosome::get_node_count()
{
	assert(("Chromosome::get_node_count cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_energy_usage();
	return phenotype_node_count;
}

decltype(Chromosome::top_row) Chromosome::get_top_row()
{
	assert(("Chromosome::get_top_row cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_energy_usage();
	return top_row;
}

decltype(Chromosome::bottom_row) Chromosome::get_bottom_row()
{
	assert(("Chromosome::get_bottom_row cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_energy_usage();
	return bottom_row;
}

std::shared_ptr<Chromosome::weight_value_t[]> Chromosome::get_weights(const std::shared_ptr<weight_value_t[]> input, size_t selector)
{
	set_input(input);
	evaluate(selector);
	auto weights = std::make_shared<weight_value_t[]>(cgp_configuration.output_count());
	std::copy(begin_output(), end_output(), weights.get());
	return weights;
}

std::vector<std::shared_ptr<Chromosome::weight_value_t[]>> Chromosome::get_weights(const std::vector<std::shared_ptr<weight_value_t[]>>& input)
{
	std::vector<std::shared_ptr<Chromosome::weight_value_t[]>> weight_vector(input.size());
	size_t selector = 0;
	for (const auto& in : input)
	{
		weight_vector.push_back(get_weights(in, selector++));
	}
	return weight_vector;
}

std::ostream& cgp::operator<<(std::ostream& os, const Chromosome& chromosome)
{
	os << chromosome.to_string();
	return os;
}

std::string cgp::to_string(const cgp::Chromosome& chromosome)
{
	return chromosome.to_string();
}

std::string cgp::to_string(const std::shared_ptr<Chromosome>& chromosome)
{
	return (chromosome) ? (to_string(*chromosome)) : Chromosome::nan_chromosome_string;
}
