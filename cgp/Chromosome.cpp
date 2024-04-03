#include "Chromosome.h"
#include <iostream>
#include <numeric>
#include <sstream>
#include <stack>
#include <cassert>

using namespace cgp;

Chromosome::Chromosome(CGPConfiguration& cgp_configuration, const std::shared_ptr<std::tuple<int, int>[]>& minimum_output_indicies, weight_actual_value_t expected_value_min, weight_actual_value_t expected_value_max) :
	cgp_configuration(cgp_configuration),
	minimum_output_indicies(minimum_output_indicies),
	expected_value_min(expected_value_min),
	expected_value_max(expected_value_max)
{
	setup_maps();
	setup_iterators();
	setup_chromosome();
}

cgp::Chromosome::Chromosome(CGPConfiguration& cgp_config, const std::shared_ptr<std::tuple<int, int>[]>& minimum_output_indicies, weight_actual_value_t expected_value_min, weight_actual_value_t expected_value_max, const std::string& serialized_chromosome) :
	cgp_configuration(cgp_config),
	minimum_output_indicies(minimum_output_indicies),
	expected_value_min(expected_value_min),
	expected_value_max(expected_value_max)
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

Chromosome::Chromosome(const Chromosome& that) noexcept :
	cgp_configuration(that.cgp_configuration),
	minimum_output_indicies(that.minimum_output_indicies),
	expected_value_min(that.expected_value_min),
	expected_value_max(that.expected_value_max) {
	input = that.input;
	need_evaluation = that.need_evaluation;
	need_energy_evaluation = that.need_energy_evaluation;
	setup_maps(that.chromosome);
	setup_iterators();
	estimated_energy_consumptation = that.estimated_energy_consumptation;
	phenotype_node_count = that.phenotype_node_count;

	if (!need_evaluation) [[likely]]
		{
			std::copy(that.output_pin_start, that.output_pin_end, output_pin_start);
		}
}

Chromosome::Chromosome(Chromosome&& that) noexcept :
	cgp_configuration(that.cgp_configuration),
	minimum_output_indicies(that.minimum_output_indicies),
	expected_value_min(that.expected_value_min),
	expected_value_max(that.expected_value_max) {
	input = std::move(that.input);
	need_evaluation = that.need_evaluation;
	need_energy_evaluation = that.need_energy_evaluation;
	setup_maps(std::move(that));
	setup_iterators(std::move(that));
	estimated_energy_consumptation = that.estimated_energy_consumptation;
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
	output_start = that.output_start;
	output_end = that.output_end;
	output_pin_start = that.output_pin_start;
	output_pin_end = that.output_pin_end;
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

std::shared_ptr<Chromosome> Chromosome::mutate() const
{
	auto chrom = std::make_shared<Chromosome>(*this);
	chrom->chromosome = std::make_shared<Chromosome::gene_t[]>(cgp_configuration.chromosome_size());
	std::copy(chromosome.get(), chromosome.get() + cgp_configuration.chromosome_size(), chrom->chromosome.get());
	chrom->setup_iterators();

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

			chrom->chromosome[random_gene_index] = (random_number % (max - min)) + min;
		}
		else if (is_output(random_gene_index)) {
			const auto& output_values = minimum_output_indicies[cgp_configuration.col_count()];
			auto min = std::get<0>(output_values);
			auto max = std::get<1>(output_values);
			chrom->chromosome[random_gene_index] = (random_number % (max - min)) + min;
		}
		else {
			chrom->chromosome[random_gene_index] = random_number % cgp_configuration.function_count();
		}
	}
	chrom->need_evaluation = true;
	chrom->need_energy_evaluation = true;
	chrom->input = nullptr;

	return chrom;
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

inline static bool is_mux(int func)
{
#ifdef CNN_FP32_WEIGHTS
	return func == 13;
#else
	return func == 31;
#endif
}

inline static bool is_demux(int func)
{
#ifdef CNN_FP32_WEIGHTS
	return func == 14;
#else
	return func == 32;
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
			set_value(block_output_pins[0], expected_value_max - pin_map[input_pin[0]]);
			break;
		case 2:
			set_value(block_output_pins[0], expected_value_max + pin_map[input_pin[0]]);
			break;
		case 3:
			set_value(block_output_pins[0], pin_map[input_pin[0]] + pin_map[input_pin[1]]);
			break;
		case 4:
			set_value(block_output_pins[0], pin_map[input_pin[0]] - pin_map[input_pin[1]]);
			break;
		case 5:
			set_value(block_output_pins[0], pin_map[input_pin[0]] * pin_map[input_pin[1]]);
			break;
		case 6:
			set_value(block_output_pins[0], -pin_map[input_pin[0]]);
			break;
		case 7:
			set_value(block_output_pins[0], std::max(expected_value_min, std::min(pin_map[input_pin[0]], expected_value_max)));
			break;
		case 8:
			set_value(block_output_pins[0], expected_value_min - pin_map[input_pin[0]]);
			break;
		case 9:
			set_value(block_output_pins[0], expected_value_min + pin_map[input_pin[0]]);
			break;
#ifdef CNN_FP32_WEIGHTS
		case 10:
			set_value(block_output_pins[0], pin_map[input_pin[0]] * 0.25);
			break;
#else
		case 10:
			set_value(block_output_pins[0], pin_map[input_pin[0]] >> 2);
			break;
#endif
#ifdef CNN_FP32_WEIGHTS
		case 11:
			set_value(block_output_pins[0], pin_map[input_pin[0]] * 0.5);
			break;
#else
		case 11:
			set_value(block_output_pins[0], pin_map[input_pin[0]] >> 1);
			break;
#endif
#ifndef CNN_FP32_WEIGHTS
		case 12:
			set_value(block_output_pins[0], pin_map[input_pin[0]] & pin_map[input_pin[1]]);
			break;
		case 13:
			set_value(block_output_pins[0], pin_map[input_pin[0]] | pin_map[input_pin[1]]);
			break;
		case 14:
			set_value(block_output_pins[0], pin_map[input_pin[0]] ^ pin_map[input_pin[1]]);
			break;
		case 15:
			set_value(block_output_pins[0], ~(pin_map[input_pin[0]]));
			break;
		case 16:
			set_value(block_output_pins[0], pin_map[input_pin[0]] << 1);
			break;
		case 17:
			set_value(block_output_pins[0], pin_map[input_pin[0]] + 1);
			break;
		case 18:
			set_value(block_output_pins[0], pin_map[input_pin[0]] - 1);
			break;
		case 19:
			set_value(block_output_pins[0], pin_map[input_pin[0]] >> 3);
			break;
		case 20:
			set_value(block_output_pins[0], pin_map[input_pin[0]] >> 4);
			break;
		case 21:
			set_value(block_output_pins[0], pin_map[input_pin[0]] >> 5);
			break;
		case 22:
			set_value(block_output_pins[0], pin_map[input_pin[0]] << 2);
			break;
		case 23:
			set_value(block_output_pins[0], pin_map[input_pin[0]] << 3);
			break;
		case 24:
			set_value(block_output_pins[0], pin_map[input_pin[0]] << 4);
			break;
		case 25:
			set_value(block_output_pins[0], pin_map[input_pin[0]] << 5);
			break;
		case 26:
			set_value(block_output_pins[0], 1);
			break;
		case 27:
			set_value(block_output_pins[0], -1);
			break;
		case 28:
			set_value(block_output_pins[0], 0);
			break;
		case 29:
			set_value(block_output_pins[0], expected_value_min);
			break;
		case 30:
			set_value(block_output_pins[0], expected_value_max);
			break;
		// multiplexor
		case 31:
			set_value(block_output_pins[0], pin_map[input_pin[selector]]);
			break;
		// de-multiplexor
		case 32:
			set_value(block_output_pins[selector], pin_map[input_pin[0]]);
			used_pin = selector;
			break;
#else
		case 12:
			set_value(block_output_pins[0], pin_map[input_pin[0]] * 1.05);
			break;
		// multiplexor
		case 13:
			set_value(block_output_pins[0], pin_map[input_pin[selector]]);
			break;
		// de-multiplexor
		case 14:
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
				block_output_pins[i] = (i == used_pin) ? (block_output_pins[i]) : (CGPConfiguration::invalid_value);
			}
			// Prevent overflows hence undefined behaviour
			set_value(block_output_pins[used_pin], std::max(expected_value_min, std::min(block_output_pins[used_pin], expected_value_max)));
		}
		else [[likely]]
		{
			// speed up evolution of operator + multiplexor; shortciruit pins
			for (size_t i = 0; i < cgp_configuration.function_output_arity(); i++)
			{
				set_value(block_output_pins[i], std::max(expected_value_min, std::min(block_output_pins[selector], expected_value_max)));
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

decltype(Chromosome::estimated_energy_consumptation) cgp::Chromosome::get_estimated_energy_usage()
{
	assert(("Chromosome::estimate_energy_usage cannot be called without calling Chromosome::evaluate before", !need_evaluation));

	if (!need_energy_evaluation)
	{
		return estimated_energy_consumptation;
	}


	auto pin_it = output_pin_start;
	std::stack<gene_t> pins_to_visit;
	std::stack<decltype(estimated_largest_delay)> delays;
	std::stack<decltype(estimated_largest_depth)> depths;
	for (auto it = output_start; it != output_end; it++, pin_it++)
	{
		pins_to_visit.push(*it);
		delays.push(0);
		depths.push(0);
	}

	estimated_energy_consumptation = 0;
	estimated_largest_delay = 0;
	estimated_largest_depth = 0;
	phenotype_node_count = 0;
	//bottom_row = cgp_configuration.row_count();
	//top_row = 0;

	while (!pins_to_visit.empty())
	{
		auto pin = pins_to_visit.top();
		auto current_largest_delay = delays.top();
		auto current_largest_depth = depths.top();
		depths.pop();
		delays.pop();
		pins_to_visit.pop();

		// if is CGP input pin
		if (pin < cgp_configuration.input_count())
		{
			estimated_largest_delay = std::max(current_largest_delay, estimated_largest_delay);
			estimated_largest_depth = std::max(current_largest_depth, estimated_largest_depth);
			continue;
		}

		auto gate_index = (pin - cgp_configuration.input_count()) / cgp_configuration.function_output_arity();

		if (gate_visit_map[gate_index])
		{
			continue;
		}

		//auto row = gate_index / cgp_configuration.col_count();
		//top_row = std::max(row, top_row);
		//bottom_row = std::min(row, bottom_row);
		gate_visit_map[gate_index] = true;
		phenotype_node_count += 1;
		estimated_energy_consumptation += CGPConfiguration::get_energy_parameter(gate_parameters_map[gate_index]);
		current_largest_delay += CGPConfiguration::get_delay_parameter(gate_parameters_map[gate_index]);
		current_largest_depth += 1;
		gene_t* inputs = get_block_inputs(gate_index);

		for (size_t i = 0; i < cgp_configuration.function_input_arity(); i++)
		{
			pins_to_visit.push(inputs[i]);
			delays.push(current_largest_delay);
			depths.push(current_largest_depth);
		}
	}

	need_energy_evaluation = false;
	return estimated_energy_consumptation;
}

decltype(Chromosome::estimated_largest_delay) Chromosome::get_estimated_largest_delay()
{
	assert(("Chromosome::get_estimated_largest_delay cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_energy_usage();
	return estimated_largest_delay;
}

decltype(Chromosome::estimated_largest_depth) Chromosome::get_estimated_largest_depth()
{
	assert(("Chromosome::get_estimated_largest_depth cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_energy_usage();
	return estimated_largest_delay;
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
