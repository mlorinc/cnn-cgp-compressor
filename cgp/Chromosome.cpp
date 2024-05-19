#include "Chromosome.h"
#include "Assert.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stack>
#include <iostream>
#include <set>
#include <map>
#include <cmath>

//#if defined(__VIRTUAL_SELECTOR) || defined(__SINGLE_OUTPUT_ARITY)
//#define _clip_pin(x) (x)
//#else
//#define _clip_pin(x) (clip_pin(x))
//#endif
#define _clip_pin(x) (x)


#if defined(__VIRTUAL_SELECTOR)
#define _update_mutation_variables() ((void)0)
#else
#define _update_mutation_variables() (update_mutation_variables())
#endif



using namespace cgp;

Chromosome::gate_parameters_t Chromosome::id_gate_parameters = std::make_tuple(0, 0, 0, 0, 0);
const std::string Chromosome::nan_chromosome_string = "null";

Chromosome::Chromosome(const CGPConfiguration& cgp_configuration, const std::unique_ptr<std::tuple<int, int>[]>& minimum_output_indicies) :
	cgp_configuration(cgp_configuration),
	minimum_output_indicies(minimum_output_indicies)
{
	output_count = cgp_configuration.output_count();
	chromosome_size = cgp_configuration.chromosome_size();
	setup_maps();
	setup_chromosome();
	setup_output_iterators(0, output_count);
	rng.set_seed(rand());
}

cgp::Chromosome::Chromosome(const CGPConfiguration& cgp_config, const std::unique_ptr<std::tuple<int, int>[]>& minimum_output_indicies, const std::string& serialized_chromosome) :
	cgp_configuration(cgp_config),
	minimum_output_indicies(minimum_output_indicies)
{
	output_count = cgp_configuration.output_count();
	chromosome_size = cgp_configuration.chromosome_size();
	setup_chromosome(serialized_chromosome);
	rng.set_seed(rand());
}

Chromosome::Chromosome(const CGPConfiguration& cgp_configuration, const std::string& serialized_chromosome) : Chromosome(cgp_configuration, nullptr, serialized_chromosome)
{
	rng.set_seed(rand());
}

Chromosome::Chromosome(const Chromosome& that) noexcept :
	cgp_configuration(that.cgp_configuration),
	minimum_output_indicies(that.minimum_output_indicies) {
	need_evaluation = that.need_evaluation;
	need_energy_evaluation = that.need_energy_evaluation;
	need_delay_evaluation = that.need_delay_evaluation;
	need_depth_evaluation = that.need_depth_evaluation;
	selector = that.selector;
	output_count = that.output_count;
	estimated_quantized_energy_consumption = that.estimated_quantized_energy_consumption;
	estimated_energy_consumption = that.estimated_energy_consumption;
	estimated_area_utilisation = that.estimated_area_utilisation;
	estimated_quantized_delay = that.estimated_quantized_delay;
	estimated_delay = that.estimated_delay;
	estimated_depth = that.estimated_depth;
	phenotype_node_count = that.phenotype_node_count;
	input = that.input;
	start_id_index = that.start_id_index;
	id_count = that.id_count;
	start_mux_index = that.start_mux_index;
	mux_count = that.mux_count;
	multiplexing = that.multiplexing;
	chromosome_size = that.chromosome_size;
	max_genes_to_mutate = that.max_genes_to_mutate;
	locked_nodes_index = that.locked_nodes_index;
	setup_maps(that);
	setup_output_iterators(selector, output_count);
	rng.set_seed(rand());
}

uint64_t Chromosome::get_random_number()
{
	return rng.generate();
}

int cgp::Chromosome::get_column(int gate_index) const
{
	return gate_index / cgp_configuration.row_count();
}

int cgp::Chromosome::get_row(int gate_index) const
{
	return gate_index % cgp_configuration.row_count();
}

Chromosome::Chromosome(Chromosome&& that) noexcept :
	cgp_configuration(that.cgp_configuration),
	minimum_output_indicies(that.minimum_output_indicies) {
	need_evaluation = that.need_evaluation;
	need_energy_evaluation = that.need_energy_evaluation;
	need_delay_evaluation = that.need_delay_evaluation;
	need_depth_evaluation = that.need_depth_evaluation;
	selector = that.selector;
	output_count = that.output_count;
	estimated_quantized_energy_consumption = that.estimated_quantized_energy_consumption;
	estimated_energy_consumption = that.estimated_energy_consumption;
	estimated_area_utilisation = that.estimated_area_utilisation;
	estimated_quantized_delay = that.estimated_quantized_delay;
	estimated_delay = that.estimated_delay;
	estimated_depth = that.estimated_depth;
	phenotype_node_count = that.phenotype_node_count;
	input = std::move(that.input);
	start_id_index = that.start_id_index;
	id_count = that.id_count;
	start_mux_index = that.start_mux_index;
	mux_count = that.mux_count;
	multiplexing = that.multiplexing;
	chromosome_size = that.chromosome_size;
	max_genes_to_mutate = that.max_genes_to_mutate;
	mutable_genes_count = that.mutable_genes_count;
	locked_nodes_index = that.locked_nodes_index;
	rng = std::move(that.rng);
	setup_maps(std::move(that));
}

bool Chromosome::is_function(int position) const
{
	return !is_output(position) && !is_input(position);
}


bool Chromosome::is_input(int position) const
{
	return !is_output(position) && position % (cgp_configuration.function_input_arity() + 1) != cgp_configuration.function_input_arity();
}


bool Chromosome::is_output(int position) const
{
	return position >= cgp_configuration.blocks_chromosome_size();
}

int Chromosome::get_output_position(int chromosome_position) const
{
	return chromosome_position - cgp_configuration.blocks_chromosome_size();
}

void Chromosome::setup_chromosome()
{
	if (!minimum_output_indicies)
	{
		return;
	}

	auto ite = chromosome.get();
	const int end = cgp_configuration.row_count() * cgp_configuration.col_count();

	for (auto i = 0; i < end; i++) {
		int column_index = static_cast<int>(i / cgp_configuration.row_count());
		// Possible input connections according to L parameter
		const auto& column_values = minimum_output_indicies[column_index];
		auto min = std::get<0>(column_values);
		auto max = std::get<1>(column_values);
		// Block function
		ite[cgp_configuration.function_input_arity()] = get_random_number() % cgp_configuration.function_count();
		// Block inputs
		for (auto j = 0; j < cgp_configuration.function_input_arity(); j++)
		{
			*ite++ = _clip_pin((get_random_number() % (max - min)) + min);
		}
		// Block function
		/**ite++ = rand() % cgp_configuration.function_count();*/
		ite++;
	}

	// Connect outputs
	const auto& output_values = minimum_output_indicies[cgp_configuration.col_count()];
	auto min = std::get<0>(output_values);
	auto max = std::get<1>(output_values);

#ifdef __VIRTUAL_SELECTOR
	for (int i = 0; i < output_count * cgp_configuration.dataset_size(); i++) {
#else
	for (int i = 0; i < output_count; i++) {
#endif
		*ite++ = _clip_pin((get_random_number() % (max - min)) + min);
	}
}

void Chromosome::setup_chromosome(const std::string & serialized_chromosome)
{
	std::istringstream input(serialized_chromosome);

	char discard;
	size_t input_count, output_count, col_count, row_count, function_input_arity, dataset_size, l_back, number_discard;

	// {n,n,n,n,n,n,5}
	input >> discard >> input_count >> discard >> output_count >> discard >> col_count >> discard >> row_count
		>> discard >> function_input_arity >> discard >> l_back >> discard >> dataset_size >> discard;

	if (input.fail())
	{
		throw std::invalid_argument("invalid format of the chromosome header\n" + serialized_chromosome);
	}

	setup_maps();
	setup_chromosome();
	setup_output_iterators(0, output_count);

	size_t block_size = cgp_configuration.block_chromosome_size();
	auto it = chromosome.get();

	for (size_t i = 0; i < row_count * col_count; i++)
	{
		// ([n]i,i,...,f)
		if (!(input >> discard >> discard >> number_discard >> discard))
		{
			throw std::invalid_argument("invalid format of the gate ID");
		}

		for (size_t k = 0; k < block_size; k++)
		{
			if (!(input >> *it++ >> discard))
			{
				throw std::invalid_argument("invalid format of the chromosome gate definition\n" + serialized_chromosome);
			}
		}

#if defined(__OLD_NOOP_OPERATION_SUPPORT)
		if (it[target_block + block_size - 1] == 0)
		{
			it[target_block + block_size - 1] = CGPOperator::ID;
		}
		else
		{
			// Back then ID function was assgined 0
			it[target_block + block_size - 1] -= 1;
		}
#endif // __OLD_NOOP_OPERATION_SUPPORT
	}

	// (n,n,...,n)
	input >> discard;
	gene_t pin;
	for (auto it = absolute_output_start; it != absolute_output_end; it++)
	{
		if (!(input >> pin >> discard))
		{
			throw std::invalid_argument("invalid format of the chromosome output\n" + serialized_chromosome);
		}
		*it = _clip_pin(pin);
		assert(0 <= *it && *it < input_count + row_count * col_count * cgp_configuration.function_output_arity());
	}

#ifdef __CGP_DEBUG 
	assert(("Chromosome::Chromosome serialized chromosome does not correspond to built chromosome", to_string() == serialized_chromosome));
#endif
}

void Chromosome::setup_maps()
{
	_update_mutation_variables();
	int nodes_count = cgp_configuration.row_count() * cgp_configuration.col_count();
	chromosome.reset(new gene_t[cgp_configuration.chromosome_size()]);
	quantized_delay_distance_map.reset(new quantized_delay_t[nodes_count]);
	depth_distance_map = std::make_unique<depth_t[]>(nodes_count);
	pin_map.reset(new weight_value_t[cgp_configuration.pin_map_size()]);
	gate_visit_map.reset(new bool[nodes_count]);
	absolute_output_start = chromosome.get() + cgp_configuration.blocks_chromosome_size();
	absolute_pin_start = pin_map.get() + cgp_configuration.pin_map_output_start();
	output_start = absolute_output_start;
	output_pin_start = absolute_pin_start;

#if defined(__VIRTUAL_SELECTOR)
	const int output_size = cgp_configuration.output_count() * cgp_configuration.dataset_size();
	estimated_quantized_energy_consumption_array = std::make_unique<quantized_energy_t[]>(output_size);
	estimated_energy_consumption_array = std::make_unique<energy_t[]>(output_size);
	estimated_area_utilisation_array = std::make_unique<area_t[]>(output_size);
	phenotype_node_count_array = std::make_unique<gate_count_t[]>(output_size);
#else
	estimated_quantized_energy_consumption_array = std::make_unique<quantized_energy_t[]>(cgp_configuration.output_count());
	estimated_energy_consumption_array = std::make_unique<energy_t[]>(cgp_configuration.output_count());
	estimated_area_utilisation_array = std::make_unique<area_t[]>(cgp_configuration.output_count());
	phenotype_node_count_array = std::make_unique<gate_count_t[]>(cgp_configuration.output_count());
#endif // __VIRTUAL_SELECTOR


}

bool cgp::Chromosome::is_locked_node(int gate_index) const
{
	return locked_nodes_index <= gate_index;
}

void Chromosome::setup_maps(const Chromosome & that)
{
#if defined(__VIRTUAL_SELECTOR)
	int output_size = output_count * cgp_configuration.dataset_size();
#else
	int output_size = output_count;
#endif
	int nodes_count = cgp_configuration.row_count() * cgp_configuration.col_count();
	setup_maps();

	std::copy(that.chromosome.get(), that.chromosome.get() + chromosome_size, chromosome.get());

	if (!that.need_energy_evaluation)
	{
		std::copy(that.gate_visit_map.get(), that.gate_visit_map.get() + nodes_count, gate_visit_map.get());
	}

	if (!that.need_evaluation)
	{
		std::copy(that.absolute_pin_start, that.absolute_pin_start + output_size, absolute_pin_start);
	}
}

void Chromosome::setup_maps(Chromosome && that)
{
	chromosome = std::move(that.chromosome);
	pin_map = std::move(that.pin_map);
	gate_visit_map = std::move(that.gate_visit_map);
	quantized_delay_distance_map = std::move(that.quantized_delay_distance_map);
	depth_distance_map = std::move(that.depth_distance_map);
	std::swap(output_start, that.output_start);
	std::swap(output_end, that.output_end);
	std::swap(output_pin_start, that.output_pin_start);
	std::swap(output_pin_end, that.output_pin_end);
	std::swap(absolute_output_start, that.absolute_output_start);
	std::swap(absolute_output_end, that.absolute_output_end);
	std::swap(absolute_pin_start, that.absolute_pin_start);
	std::swap(absolute_pin_end, that.absolute_pin_end);
}

void cgp::Chromosome::setup_output_iterators(int selector, size_t output_count)
{
	this->output_count = output_count;
#if defined(__VIRTUAL_SELECTOR)
	const int output_size = output_count * cgp_configuration.dataset_size();
	output_start = absolute_output_start + selector * output_count;
	output_pin_start = absolute_pin_start + selector * output_count;
	output_pin_end = output_pin_start + output_count;
	absolute_pin_end = absolute_pin_start + output_size;
	absolute_output_end = absolute_output_start + output_size;
	max_genes_to_mutate = cgp_configuration.max_genes_to_mutate() + 1;
	mutable_genes_count = chromosome_size;
#else
	output_pin_end = output_pin_start + output_count;
	absolute_pin_end = absolute_pin_start + output_count;
	absolute_output_end = absolute_output_start + output_count;
	_update_mutation_variables();
#endif
	output_end = output_start + output_count;
}

void cgp::Chromosome::update_mutation_variables()
{
#ifndef __VIRTUAL_SELECTOR
	if (is_multiplexing())
	{
		chromosome_size = cgp_configuration.blocks_chromosome_size() + output_count;
		mutable_genes_count = chromosome_size - id_count;
#ifdef __MULTI_MULTIPLEX
		// Do not mutate MUX
		mutable_genes_count -= (cgp_configuration.row_count() * cgp_configuration.col_count() - start_mux_index) * cgp_configuration.block_chromosome_size();
#endif

		max_genes_to_mutate = static_cast<int>(std::floor(mutable_genes_count * cgp_configuration.mutation_max())) + 1;
	}
	else
	{
		chromosome_size = cgp_configuration.chromosome_size();
		mutable_genes_count = chromosome_size;
		max_genes_to_mutate = cgp_configuration.max_genes_to_mutate() + 1;
	}
#endif
}

Chromosome::weight_value_t cgp::Chromosome::plus(int a, int b)
{
	int c = a + b;
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

Chromosome::weight_value_t cgp::Chromosome::minus(int a, int b)
{
	return plus(a, -b);
}

Chromosome::weight_value_t cgp::Chromosome::mul(int a, int b)
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

Chromosome::weight_value_t Chromosome::bit_rshift(weight_value_t a, weight_value_t b)
{
	return a >> b;
}

Chromosome::weight_value_t Chromosome::bit_lshift(weight_value_t a, weight_value_t b)
{
	return a << b;
}

Chromosome::weight_value_t Chromosome::bit_and(weight_value_t a, weight_value_t b)
{
	return a & b;
}

Chromosome::weight_value_t Chromosome::bit_or(weight_value_t a, weight_value_t b)
{
	return a | b;
}

Chromosome::weight_value_t Chromosome::bit_xor(weight_value_t a, weight_value_t b)
{
	return a ^ b;
}
Chromosome::weight_value_t Chromosome::bit_neg(weight_value_t a)
{
	return ~a;
}

Chromosome::weight_value_t cgp::Chromosome::neg(weight_value_t a)
{
	return (a != -128) ? (-a) : (-128);
}

Chromosome::gene_t* Chromosome::get_outputs() const {
	return output_start;
}

Chromosome::gene_t* Chromosome::get_block(int index) const
{
	return chromosome.get() + index * (cgp_configuration.function_input_arity() + 1);
}

Chromosome::gene_t* Chromosome::get_block_inputs(int row, int column) const {
	return get_block_inputs(row * column);
}

Chromosome::gene_t* cgp::Chromosome::get_block_inputs(int index) const
{
	return get_block(index);
}


Chromosome::gene_t* Chromosome::get_block_function(int row, int column) const {
	return get_block_function(row * column);
}

Chromosome::gene_t* Chromosome::get_block_function(int gate_index) const
{
	return chromosome.get() + gate_index * cgp_configuration.block_chromosome_size() + cgp_configuration.function_input_arity();
}

const std::unique_ptr<Chromosome::gene_t[]>& Chromosome::get_chromosome() const
{
	return chromosome;
}

bool Chromosome::mutate_genes(std::shared_ptr<Chromosome> that) const
{
	// Only and only if gate_visit_map was fille
	int genes_to_mutate = that->rng.generate() % max_genes_to_mutate;
	bool neutral_mutation = !need_gate_visit_map;

	for (int i = 0; i < genes_to_mutate; i++) {
		// Select a random gene
		auto random_gene_index = that->rng.generate() % mutable_genes_count;
		auto random_number = that->rng.generate();

		if (is_input(random_gene_index))
		{
			int gate_index = random_gene_index / (cgp_configuration.function_input_arity() + 1);
			int column_index = get_column(gate_index);
			const auto& column_values = minimum_output_indicies[column_index];
			auto min = std::get<0>(column_values);
			auto max = std::get<1>(column_values);
			auto new_pin = _clip_pin((random_number % (max - min)) + min);

			neutral_mutation = neutral_mutation && (new_pin == chromosome[random_gene_index] || !gate_visit_map[gate_index]);
			that->chromosome[random_gene_index] = new_pin;
		}
		else if (is_output(random_gene_index)) {
			assert(!multiplexing);
			const auto& output_values = minimum_output_indicies[cgp_configuration.col_count()];
			auto min = std::get<0>(output_values);
			auto max = std::get<1>(output_values);
			auto new_pin = _clip_pin((random_number % (max - min)) + min);

			neutral_mutation = neutral_mutation && chromosome[random_gene_index] == new_pin;
			that->chromosome[random_gene_index] = new_pin;
		}
		else {
			int gate_index = random_gene_index / (cgp_configuration.function_input_arity() + 1);
			if (is_locked_node(gate_index))
			{
				continue;
			}

			int func = random_number % (cgp_configuration.function_count() + 1);

			if (func == cgp_configuration.function_count())
			{
				func = CGPOperator::ID;
			}

			neutral_mutation = neutral_mutation && (func == chromosome[random_gene_index] || !gate_visit_map[gate_index]);
			that->chromosome[random_gene_index] = func;
		}
	}
	if (!neutral_mutation)
	{
		that->invalidate();
	}
	else
	{
		that->need_gate_visit_map = true;
		that->need_evaluation = need_evaluation;
		that->need_energy_evaluation = need_energy_evaluation;
		that->need_delay_evaluation = need_delay_evaluation;
		that->need_depth_evaluation = need_depth_evaluation;
		that->estimated_quantized_energy_consumption = estimated_quantized_energy_consumption;
		that->estimated_energy_consumption = estimated_energy_consumption;
		that->estimated_area_utilisation = estimated_area_utilisation;
		that->estimated_quantized_delay = estimated_quantized_delay;
		that->estimated_delay = estimated_delay;
		that->estimated_depth = estimated_depth;
		that->phenotype_node_count = phenotype_node_count;
		that->top_row = top_row;
		that->bottom_row = bottom_row;
		that->first_col = first_col;
		that->last_col = last_col;
	}

	return neutral_mutation;
}

std::shared_ptr<Chromosome> Chromosome::mutate(uint64_t seed)
{
	auto chrom = std::make_shared<Chromosome>(*this);
	chrom->rng.set_seed(seed);
	mutate_genes(chrom);
	return chrom;
}

bool cgp::Chromosome::mutate(std::shared_ptr<Chromosome> that, const dataset_t & dataset)
{
	// Actually, do not care anymore about having stricly lambda offsprings ...
	assert(this != that.get());

	std::copy(chromosome.get(), chromosome.get() + chromosome_size, that->chromosome.get());
	return mutate_genes(that);
}

void cgp::Chromosome::swap_visit_map(Chromosome & chromosome)
{
	std::swap(gate_visit_map, chromosome.gate_visit_map);
}

void Chromosome::set_input(const weight_value_t * input, int selector)
{
	bool valid = this->input == input && this->selector == selector;
	this->input = input;
	this->selector = selector;
	if (!valid)
	{
		need_evaluation = true;
	}
#if defined(__VIRTUAL_SELECTOR)
	setup_output_iterators(selector, output_count);
#endif // __VIRTUAL_SELECTOR
}

inline static void set_value(CGPConfiguration::weight_value_t & target, const CGPConfiguration::weight_value_t & value)
{
	target = value;
}

bool Chromosome::is_mux(int func)
{
#if defined(CNN_FP32_WEIGHTS)
	return func == 9;
#else
	return func == CGPOperator::MUX;
#endif
}

bool Chromosome::is_demux(int func)
{
#ifdef CNN_FP32_WEIGHTS
	return func == 10;
#else
	return func == CGPOperator::DEMUX;
#endif
}

Chromosome::weight_value_t Chromosome::get_pin_value(int index) const
{
	int new_index = index - cgp_configuration.input_count();
	if (new_index < 0)
	{
		return input[index];
	}

	return pin_map[new_index];
}

Chromosome::weight_value_t Chromosome::get_gate_output(int gate_index) const
{
	return get_pin_value(get_gate_output_used_pin(gate_index));
}

Chromosome::weight_value_t Chromosome::get_gate_input(int gate_index, int pin) const
{
	assert(("pin is out of bounds", pin < cgp_configuration.function_input_arity()));
	assert(("pin is negative", pin < 0));
	if (pin != 0)
	{
		return get_pin_value(get_block_inputs(gate_index)[pin]);
	}
	else
	{
		return get_pin_value(get_gate_input_used_pin(gate_index));
	}
}

int Chromosome::get_gate_output_used_pin(int gate_index) const
{
#ifndef __SINGLE_OUTPUT_ARITY
	const auto func = *get_block_function(gate_index);

	if (func != CGPOperator::DEMUX)
	{
		return get_output_pin_from_gate_index(gate_index);
	}
	else if (func == CGPOperator::DEMUX)
	{
		return get_output_pin_from_gate_index(gate_index, selector);
	}
	else
	{
		throw std::runtime_error("unexpected state");
	}
#else
	return get_output_pin_from_gate_index(gate_index);
#endif
}

int Chromosome::get_gate_input_used_pin(int gate_index) const
{
#ifndef __SINGLE_OUTPUT_ARITY
	const auto func = *get_block_function(gate_index);

	if (func != CGPOperator::MUX)
	{
		return get_block_inputs(gate_index)[0];
	}
	else if (func == CGPOperator::MUX)
	{
		return get_block_inputs(gate_index)[selector];
	}
	else
	{
		throw std::runtime_error("unexpected state");
	}
#else
	return get_block_inputs(gate_index)[0];
#endif
}

void cgp::Chromosome::evaluate_single_from_pins(gene_t * input_pin, weight_output_t output, gene_t function)
{
	const auto expected_value_min = cgp_configuration.expected_value_min();
	const auto expected_value_max = cgp_configuration.expected_value_max();
	switch (function) {
	case CGPOperator::REVERSE_MAX_A:
		set_value(output[0], minus(expected_value_max, get_pin_value(input_pin[0])));
		break;
	case CGPOperator::ADD:
		set_value(output[0], plus(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
		break;
	case CGPOperator::SUB:
		set_value(output[0], minus(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
		break;
	case CGPOperator::MUL:
		set_value(output[0], mul(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
		break;
	case CGPOperator::NEG:
		set_value(output[0], neg(get_pin_value(input_pin[0])));
		break;
	case CGPOperator::REVERSE_MIN_B:
		set_value(output[0], plus(expected_value_min, get_pin_value(input_pin[0])));
		break;
#ifdef CNN_FP32_WEIGHTS
	case 6:
		set_value(output[0], get_pin_value(input_pin[0]) * 0.25);
		break;
#else
	case CGPOperator::QUARTER:
		set_value(output[0], bit_rshift(get_pin_value(input_pin[0]), 2));
		break;
#endif
#ifdef CNN_FP32_WEIGHTS
	case 7:
		set_value(output[0], get_pin_value(input_pin[0]) * 0.5);
		break;
#else
	case CGPOperator::HALF:
		set_value(output[0], bit_rshift(get_pin_value(input_pin[0]), 1));
		break;
#endif
#ifndef CNN_FP32_WEIGHTS
	case CGPOperator::BIT_AND:
		set_value(output[0], bit_and(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
		break;
	case CGPOperator::BIT_OR:
		set_value(output[0], bit_or(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
		break;
	case CGPOperator::BIT_XOR:
		set_value(output[0], bit_xor(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
		break;
	case CGPOperator::BIT_NEG:
		set_value(output[0], bit_neg(get_pin_value(input_pin[0])));
		break;
	case CGPOperator::DOUBLE:
		set_value(output[0], bit_lshift(get_pin_value(input_pin[0]), 1));
		break;
	case CGPOperator::BIT_INC:
		set_value(output[0], plus(get_pin_value(input_pin[0]), 1));
		break;
	case CGPOperator::BIT_DEC:
		set_value(output[0], minus(get_pin_value(input_pin[0]), 1));
		break;
	case CGPOperator::R_SHIFT_3:
		set_value(output[0], bit_rshift(get_pin_value(input_pin[0]), 3));
		break;
	case CGPOperator::R_SHIFT_4:
		set_value(output[0], bit_rshift(get_pin_value(input_pin[0]), 4));
		break;
	case CGPOperator::R_SHIFT_5:
		set_value(output[0], bit_rshift(get_pin_value(input_pin[0]), 5));
		break;
	case CGPOperator::L_SHIFT_2:
		set_value(output[0], bit_lshift(get_pin_value(input_pin[0]), 2));
		break;
	case CGPOperator::L_SHIFT_3:
		set_value(output[0], bit_lshift(get_pin_value(input_pin[0]), 3));
		break;
	case CGPOperator::L_SHIFT_4:
		set_value(output[0], bit_lshift(get_pin_value(input_pin[0]), 4));
		break;
	case CGPOperator::L_SHIFT_5:
		set_value(output[0], bit_lshift(get_pin_value(input_pin[0]), 5));
		break;
	case CGPOperator::ONE_CONST:
		set_value(output[0], 1);
		break;
	case CGPOperator::MINUS_ONE_CONST:
		set_value(output[0], -1);
		break;
	case CGPOperator::ZERO_CONST:
		set_value(output[0], 0);
		break;
	case CGPOperator::EXPECTED_VALUE_MIN:
		set_value(output[0], expected_value_min);
		break;
	case CGPOperator::EXPECTED_VALUE_MAX:
		set_value(output[0], expected_value_max);
		break;
		// multiplexor
	case CGPOperator::MUX:
		set_value(output[0], get_pin_value(input_pin[selector]));
		break;
		// de-multiplexor
	case CGPOperator::DEMUX:
		for (int j = 0; j < cgp_configuration.function_output_arity(); j++)
		{
			set_value(output[j], (j == selector) ? (get_pin_value(input_pin[0])) : (0));
		}
		break;
	case CGPOperator::ID:
#ifdef __SINGLE_OUTPUT_ARITY
		set_value(output[0], get_pin_value(input_pin[0]));
#else
		for (int j = 0; j < cgp_configuration.function_output_arity(); j++)
		{
			set_value(output[j], get_pin_value(input_pin[j]));
		}
#endif
		break;
#else
	case 8:
		set_value(output[0], get_pin_value(input_pin[0]) * 1.05);
		break;
		// multiplexor
	case 9:
		set_value(output[0], get_pin_value(input_pin[selector]));
		break;
		// de-multiplexor
	case 10:
		set_value(output[selector], get_pin_value(input_pin[0]));
		break;
#endif
	default:
		throw std::runtime_error("this Chromosome::evaluate branch is not implemented! branch: " + std::to_string(function));
		break;
	}
}

void cgp::Chromosome::evaluate_single_from_values(weight_input_t input, weight_output_t output, gene_t function)
{
	const auto expected_value_min = cgp_configuration.expected_value_min();
	const auto expected_value_max = cgp_configuration.expected_value_max();
	switch (function) {
	case CGPOperator::REVERSE_MAX_A:
		set_value(output[0], minus(expected_value_max, input[0]));
		break;
	case CGPOperator::ADD:
		set_value(output[0], plus(input[0], input[1]));
		break;
	case CGPOperator::SUB:
		set_value(output[0], minus(input[0], input[1]));
		break;
	case CGPOperator::MUL:
		set_value(output[0], mul(input[0], input[1]));
		break;
	case CGPOperator::NEG:
		set_value(output[0], neg(input[0]));
		break;
	case CGPOperator::REVERSE_MIN_B:
		set_value(output[0], plus(expected_value_min, input[0]));
		break;
#ifdef CNN_FP32_WEIGHTS
	case 6:
		set_value(output[0], input[0] * 0.25);
		break;
#else
	case CGPOperator::QUARTER:
		set_value(output[0], bit_rshift(input[0], 2));
		break;
#endif
#ifdef CNN_FP32_WEIGHTS
	case 7:
		set_value(output[0], input[0] * 0.5);
		break;
#else
	case CGPOperator::HALF:
		set_value(output[0], bit_rshift(input[0], 1));
		break;
#endif
#ifndef CNN_FP32_WEIGHTS
	case CGPOperator::BIT_AND:
		set_value(output[0], bit_and(input[0], input[1]));
		break;
	case CGPOperator::BIT_OR:
		set_value(output[0], bit_or(input[0], input[1]));
		break;
	case CGPOperator::BIT_XOR:
		set_value(output[0], bit_xor(input[0], input[1]));
		break;
	case CGPOperator::BIT_NEG:
		set_value(output[0], bit_neg(input[0]));
		break;
	case CGPOperator::DOUBLE:
		set_value(output[0], bit_lshift(input[0], 1));
		break;
	case CGPOperator::BIT_INC:
		set_value(output[0], plus(input[0], 1));
		break;
	case CGPOperator::BIT_DEC:
		set_value(output[0], minus(input[0], 1));
		break;
	case CGPOperator::R_SHIFT_3:
		set_value(output[0], bit_rshift(input[0], 3));
		break;
	case CGPOperator::R_SHIFT_4:
		set_value(output[0], bit_rshift(input[0], 4));
		break;
	case CGPOperator::R_SHIFT_5:
		set_value(output[0], bit_rshift(input[0], 5));
		break;
	case CGPOperator::L_SHIFT_2:
		set_value(output[0], bit_lshift(input[0], 2));
		break;
	case CGPOperator::L_SHIFT_3:
		set_value(output[0], bit_lshift(input[0], 3));
		break;
	case CGPOperator::L_SHIFT_4:
		set_value(output[0], bit_lshift(input[0], 4));
		break;
	case CGPOperator::L_SHIFT_5:
		set_value(output[0], bit_lshift(input[0], 5));
		break;
	case CGPOperator::ONE_CONST:
		set_value(output[0], 1);
		break;
	case CGPOperator::MINUS_ONE_CONST:
		set_value(output[0], -1);
		break;
	case CGPOperator::ZERO_CONST:
		set_value(output[0], 0);
		break;
	case CGPOperator::EXPECTED_VALUE_MIN:
		set_value(output[0], expected_value_min);
		break;
	case CGPOperator::EXPECTED_VALUE_MAX:
		set_value(output[0], expected_value_max);
		break;
		// multiplexor
	case CGPOperator::MUX:
		set_value(output[0], input[selector]);
		break;
		// de-multiplexor
	case CGPOperator::DEMUX:
		for (int j = 0; j < cgp_configuration.function_output_arity(); j++)
		{
			set_value(output[j], (j == selector) ? (input[0]) : (0));
		}
		break;
	case CGPOperator::ID:
#ifdef __SINGLE_OUTPUT_ARITY
		set_value(output[0], input[0]);
#else
		for (int j = 0; j < cgp_configuration.function_output_arity(); j++)
		{
			set_value(output[j], input[j]);
		}
#endif
		break;
#else
	case 8:
		set_value(output[0], input[0] * 1.05);
		break;
		// multiplexor
	case 9:
		set_value(output[0], input[selector]);
		break;
		// de-multiplexor
	case 10:
		set_value(output[selector], input[0]);
		break;
#endif
	default:
		throw std::runtime_error("this Chromosome::evaluate branch is not implemented! branch: " + std::to_string(function));
		break;
	}
}

void Chromosome::evaluate()
{
	assert(("Chromosome::evaluate cannot be called without calling Chromosome::set_input before", input != nullptr));

	if (!need_evaluation)
	{
		return;
	}

	auto output_pin = pin_map.get();
	auto input_pin = chromosome.get();
	auto func = chromosome.get() + cgp_configuration.function_input_arity();

	for (int i = 0; i < cgp_configuration.col_count() * cgp_configuration.row_count(); i++)
	{
		gate_visit_map[i] = !need_gate_visit_map && gate_visit_map[i];
		evaluate_single_from_pins(input_pin, output_pin, *func);

#ifndef __SINGLE_OUTPUT_ARITY
		// Shortcircuit pins if multi-output is used
		if (*func != CGPOperator::DEMUX)
		{
			int src_pin = ((*func) != CGPOperator::MUX) ? (0) : (selector);
			for (int i = 0; i < cgp_configuration.function_output_arity(); i++)
			{
				set_value(output_pin[i], output_pin[src_pin]);
			}
		}
#endif

		output_pin += cgp_configuration.function_output_arity();
		input_pin += cgp_configuration.function_input_arity() + 1;
		func += cgp_configuration.function_input_arity() + 1;
	}

	auto it = output_start;
	auto pin_it = output_pin_start;
#pragma omp parallel for
	for (int i = 0; i < output_end - output_start; i++)
	{
		//assert(0 <= *it && *it < cgp_configuration.row_count() * cgp_configuration.col_count() * cgp_configuration.function_output_arity() + cgp_configuration.input_count());
		pin_it[i] = get_pin_value(it[i]);
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

int cgp::Chromosome::get_function_input_arity(int gate_index) const
{
	try {
		return get_function_input_arity_2(*get_block_function(gate_index));
	}
	catch (const std::runtime_error& e)
	{
		std::string new_message = std::string(e.what()) + ", from gate index " + std::to_string(gate_index);
		throw std::runtime_error(new_message);
	}
}

int cgp::Chromosome::get_function_input_arity_2(gene_t func) const
{
	switch (func) {
	case CGPOperator::REVERSE_MAX_A:
		return 1;
		break;
	case CGPOperator::ADD:
		return 2;
		break;
	case CGPOperator::SUB:
		return 2;
		break;
	case CGPOperator::MUL:
		return 2;
		break;
	case CGPOperator::NEG:
		return 1;
		break;
	case CGPOperator::REVERSE_MIN_B:
		return 1;
		break;
	case CGPOperator::QUARTER:
		return 1;
		break;
	case CGPOperator::HALF:
		return 1;
		break;
	case CGPOperator::BIT_AND:
		return 2;
		break;
	case CGPOperator::BIT_OR:
		return 2;
		break;
	case CGPOperator::BIT_XOR:
		return 2;
		break;
	case CGPOperator::BIT_NEG:
		return 1;
		break;
	case CGPOperator::DOUBLE:
		return 1;
		break;
	case CGPOperator::BIT_INC:
		return 1;
		break;
	case CGPOperator::BIT_DEC:
		return 1;
		break;
	case CGPOperator::R_SHIFT_3:
		return 1;
		break;
	case CGPOperator::R_SHIFT_4:
		return 1;
		break;
	case CGPOperator::R_SHIFT_5:
		return 1;
		break;
	case CGPOperator::L_SHIFT_2:
		return 1;
		break;
	case CGPOperator::L_SHIFT_3:
		return 1;
		break;
	case CGPOperator::L_SHIFT_4:
		return 1;
		break;
	case CGPOperator::L_SHIFT_5:
		return 1;
		break;
	case CGPOperator::ONE_CONST:
		return 0;
		break;
	case CGPOperator::MINUS_ONE_CONST:
		return 0;
		break;
	case CGPOperator::ZERO_CONST:
		return 0;
		break;
	case CGPOperator::EXPECTED_VALUE_MIN:
		return 0;
		break;
	case CGPOperator::EXPECTED_VALUE_MAX:
		return 0;
		break;
		// multiplexor
	case CGPOperator::MUX:
		return cgp_configuration.function_input_arity();
		break;
		// de-multiplexor
	case CGPOperator::DEMUX:
		return 1;
		break;
	case CGPOperator::ID:
		return 1;
		break;
	default:
		throw std::runtime_error("this Chromosome::get_function_input_arity branch is not implemented! branch: " + std::to_string(func));
		break;
	}
}

int cgp::Chromosome::get_function_output_arity(int gate_index) const
{
	return get_function_output_arity2(*get_block_function(gate_index));
}

int cgp::Chromosome::get_function_output_arity2(gene_t func) const
{
	assert(func <= CGPOperator::DEMUX || func == CGPOperator::ID);
	switch (func) {
	case CGPOperator::REVERSE_MAX_A:
		return 1;
		break;
	case CGPOperator::ADD:
		return 1;
		break;
	case CGPOperator::SUB:
		return 1;
		break;
	case CGPOperator::MUL:
		return 1;
		break;
	case CGPOperator::NEG:
		return 1;
		break;
	case CGPOperator::REVERSE_MIN_B:
		return 1;
		break;
	case CGPOperator::QUARTER:
		return 1;
		break;
	case CGPOperator::HALF:
		return 1;
		break;
	case CGPOperator::BIT_AND:
		return 1;
		break;
	case CGPOperator::BIT_OR:
		return 1;
		break;
	case CGPOperator::BIT_XOR:
		return 1;
		break;
	case CGPOperator::BIT_NEG:
		return 1;
		break;
	case CGPOperator::DOUBLE:
		return 1;
		break;
	case CGPOperator::BIT_INC:
		return 1;
		break;
	case CGPOperator::BIT_DEC:
		return 1;
		break;
	case CGPOperator::R_SHIFT_3:
		return 1;
		break;
	case CGPOperator::R_SHIFT_4:
		return 1;
		break;
	case CGPOperator::R_SHIFT_5:
		return 1;
		break;
	case CGPOperator::L_SHIFT_2:
		return 1;
		break;
	case CGPOperator::L_SHIFT_3:
		return 1;
		break;
	case CGPOperator::L_SHIFT_4:
		return 1;
		break;
	case CGPOperator::L_SHIFT_5:
		return 1;
		break;
	case CGPOperator::ONE_CONST:
		return 1;
		break;
	case CGPOperator::MINUS_ONE_CONST:
		return 1;
		break;
	case CGPOperator::ZERO_CONST:
		return 1;
		break;
	case CGPOperator::EXPECTED_VALUE_MIN:
		return 1;
		break;
	case CGPOperator::EXPECTED_VALUE_MAX:
		return 1;
		break;
		// multiplexor
	case CGPOperator::MUX:
		return 1;
		break;
		// de-multiplexor
	case CGPOperator::DEMUX:
		return cgp_configuration.function_output_arity();
		break;
	case CGPOperator::ID:
		return 1;
		break;
	default:
		throw std::runtime_error("this Chromosome::get_function_output_arity branch is not implemented! branch: " + std::to_string(func));
		break;
	}
}

Chromosome::weight_value_t* Chromosome::end_multiplexed_output()
{
	assert(("Chromosome::end_multiplexed_output cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	const auto end = begin_output() + id_count;
	return end;
}

std::string Chromosome::to_string() const
{
	std::ostringstream result;

	result << "{" << static_cast<unsigned>(cgp_configuration.input_count()) << "," << static_cast<unsigned>(cgp_configuration.output_count()) << ","
		<< static_cast<unsigned>(cgp_configuration.col_count()) << "," << cgp_configuration.row_count() << ","
		<< static_cast<unsigned>(cgp_configuration.function_input_arity()) << "," << cgp_configuration.look_back_parameter() << "," << static_cast<unsigned>(cgp_configuration.dataset_size()) << "}";

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

int cgp::Chromosome::get_serialized_chromosome_size() const
{
	// chromosome size + input information + output information
	return cgp_configuration.chromosome_size() * sizeof(gene_t) + 2 * sizeof(gene_t);
}

void cgp::Chromosome::move_gate(int src_gate_index, int dst_gate_index)
{
	auto src = get_block(src_gate_index);
	auto dst = get_block(dst_gate_index);

	auto dst_temp = std::make_unique<gene_t[]>(cgp_configuration.block_chromosome_size());
	std::copy(dst, dst + cgp_configuration.block_chromosome_size(), dst_temp.get());
	std::copy(src, src + cgp_configuration.block_chromosome_size(), dst);
	std::copy(dst_temp.get(), dst_temp.get() + cgp_configuration.block_chromosome_size(), src);

	const auto output_src_pin_min = src_gate_index * cgp_configuration.function_output_arity() + cgp_configuration.input_count();
	const auto output_src_pin_max = output_src_pin_min + cgp_configuration.function_output_arity();
	const auto output_dst_pin_min = dst_gate_index * cgp_configuration.function_output_arity() + cgp_configuration.input_count();
	const auto output_dst_pin_max = output_dst_pin_min + cgp_configuration.function_output_arity();

#pragma omp parallel for 
	for (int i = std::min(src_gate_index, dst_gate_index) + 1; i < cgp_configuration.row_count() * cgp_configuration.col_count(); i++)
	{
		if (i == src_gate_index || i == dst_gate_index)
		{
			continue;
		}

		auto dependent_inputs = get_block_inputs(i);
		for (int j = 0; j < cgp_configuration.function_input_arity(); j++)
		{
			if (output_src_pin_min <= dependent_inputs[j] && dependent_inputs[j] < output_src_pin_max)
			{
				dependent_inputs[j] = output_dst_pin_min + (dependent_inputs[j] - output_src_pin_min);
			}
			// just connect it randomly, it does not matter
			else if (output_dst_pin_min <= dependent_inputs[j] && dependent_inputs[j] < output_dst_pin_max)
			{
				dependent_inputs[j] = output_src_pin_min + (dependent_inputs[j] - output_dst_pin_min);
			}
		}
	}

	for (auto it = absolute_output_start; it != absolute_output_end; it++)
	{
		if (output_src_pin_min <= *it && *it < output_src_pin_max)
		{
			*it = output_dst_pin_min + (*it - output_src_pin_min);
		}
		else if (output_dst_pin_min <= *it && *it < output_dst_pin_max)
		{
			throw std::runtime_error("should not ever happened");
		}
	}

	std::swap(gate_visit_map[src_gate_index], gate_visit_map[dst_gate_index]);
}

bool cgp::Chromosome::move_block_to_the_start(int gate_index)
{
	// Populate visit map
	get_estimated_energy_usage();

	// if already on first row ... stop
	if (gate_index < cgp_configuration.row_count())
	{
		return false;
	}

	auto inputs = get_block_inputs(gate_index);
	int max_pin = -1;
	for (int i = 0; i < cgp_configuration.function_input_arity(); i++)
	{
		max_pin = std::max(max_pin, inputs[i]);
	}

	if (max_pin == -1)
	{
		return false;
	}

	int start_gate_index;

	if (max_pin < cgp_configuration.input_count())
	{
		start_gate_index = 0;
	}
	else
	{
		const int index = get_gate_index_from_output_pin(max_pin);
		const int col = get_column(index) + 1;
		start_gate_index = col * cgp_configuration.row_count();
	}
	const int end_index = get_column(gate_index) * cgp_configuration.row_count();
	int target_gate_index;
	for (target_gate_index = start_gate_index; target_gate_index < end_index && gate_visit_map[target_gate_index]; target_gate_index++)
	{

	}

	if (target_gate_index == end_index)
	{
		return false;
	}

	move_gate(gate_index, target_gate_index);
	return true;
}

void cgp::Chromosome::tight_layout()
{
	while (true)
	{
		bool left_moves = false;
		for (int i = 0; i < cgp_configuration.row_count() * cgp_configuration.col_count(); i++)
		{
			if (!gate_visit_map[i])
			{
				continue;
			}
			bool result = move_block_to_the_start(i);
			left_moves = result || left_moves;
		}
		if (!left_moves)
		{
			break;
		}
	}
	invalidate();
}

int cgp::Chromosome::get_gate_index_from_output_pin(int pin) const
{
	assert(("pin is not gate pin, but input pin", pin >= cgp_configuration.input_count()));
#ifndef __SINGLE_OUTPUT_ARITY
	return (pin - cgp_configuration.input_count()) / cgp_configuration.function_output_arity();
#else
	return pin - cgp_configuration.input_count();
#endif
}

int cgp::Chromosome::get_gate_index_from_input_pin(int pin) const
{
	assert(("pin is not gate pin, but input pin", pin >= cgp_configuration.input_count()));
	return (pin - cgp_configuration.input_count()) / cgp_configuration.function_input_arity();
}

int cgp::Chromosome::get_output_pin_from_gate_index(int gate_index, int pin) const
{
	assert(0 <= pin && 0 <= gate_index && gate_index < cgp_configuration.row_count() * cgp_configuration.col_count());
#ifndef __SINGLE_OUTPUT_ARITY
	return gate_index * cgp_configuration.function_output_arity() + cgp_configuration.input_count() + pin;
#else
	return gate_index + cgp_configuration.input_count() + pin;
#endif
}

void cgp::Chromosome::invalidate()
{
	need_evaluation = true;
	need_energy_evaluation = true;
	need_delay_evaluation = true;
	need_depth_evaluation = true;
	need_gate_visit_map = true;
	estimated_quantized_energy_consumption = CGPConfiguration::quantized_energy_nan;
	estimated_energy_consumption = CGPConfiguration::energy_nan;
	estimated_area_utilisation = CGPConfiguration::area_nan;
	estimated_quantized_delay = CGPConfiguration::quantized_delay_nan;
	estimated_delay = CGPConfiguration::delay_nan;
	estimated_depth = CGPConfiguration::depth_nan;
	phenotype_node_count = CGPConfiguration::gate_count_nan;
}

void cgp::Chromosome::invalidate_visit_map()
{
	need_energy_evaluation = true;
	need_delay_evaluation = true;
	need_depth_evaluation = true;
	need_gate_visit_map = true;
}

decltype(Chromosome::estimated_quantized_energy_consumption) cgp::Chromosome::get_estimated_quantized_energy_usage()
{
	//assert(("Chromosome::estimate_energy_usage cannot be called without calling Chromosome::evaluate before", !need_evaluation));

	if (!need_energy_evaluation)
	{
		return estimated_quantized_energy_consumption;
	}

	const int length = absolute_output_end - absolute_output_start;

#pragma omp parallel for
	for (int i = 0; i < length; i++)
	{
		std::stack<gene_t> pins_to_visit;
		pins_to_visit.push(absolute_output_start[i]);

		estimated_quantized_energy_consumption_array[i] = 0;
		estimated_energy_consumption_array[i] = 0;
		estimated_area_utilisation_array[i] = 0;
		phenotype_node_count_array[i] = 0;

#ifndef _DISABLE_ROW_COL_STATS
		int bottom_row = cgp_configuration.row_count();
		int top_row = 0;
		int first_col = cgp_configuration.col_count();
		int last_col = 0;
#endif

		while (!pins_to_visit.empty())
		{
			gene_t pin = pins_to_visit.top();
			pins_to_visit.pop();

			// if is CGP input pin
			if (pin < cgp_configuration.input_count())
			{
				continue;
			}

			const int gate_index = get_gate_index_from_output_pin(pin);
			bool can_visit = false;
#pragma omp critical		
			if (!gate_visit_map[gate_index])
			{
				gate_visit_map[gate_index] = true;
				can_visit = true;
			}

			if (can_visit)
			{
#ifndef _DISABLE_ROW_COL_STATS
				int row = get_row(gate_index);
				int col = get_column(gate_index);
				top_row = std::min(row, top_row);
				bottom_row = std::max(row, bottom_row);
				first_col = std::min(col, first_col);
				last_col = std::max(col, last_col);
#endif
				gene_t func = *get_block_function(gate_index);
				auto function_cost = get_function_cost(func);

				phenotype_node_count_array[i] += 1;
				estimated_energy_consumption_array[i] += CGPConfiguration::get_energy_parameter(function_cost);
				estimated_quantized_energy_consumption_array[i] += CGPConfiguration::get_quantized_energy_parameter(function_cost);
				estimated_area_utilisation_array[i] += CGPConfiguration::get_area_parameter(function_cost);

				gene_t* inputs = get_block_inputs(gate_index);
				const auto arity = get_function_input_arity(gate_index);
				for (int j = 0; j < arity; j++)
				{
					pins_to_visit.push(inputs[j]);
				}

				// Just for documentation block, the semantic is here
				//if (arity == 0)
				//{
				//	continue;
				//}
			}
		}
	}

	quantized_energy_t estimated_quantized_energy_consumption = 0;
	energy_t estimated_energy_consumption = 0;
	area_t estimated_area_utilisation = 0;
	gate_count_t phenotype_node_count = 0;

#pragma omp parallel for reduction(+:phenotype_node_count) reduction(+:estimated_energy_consumption) reduction(+:estimated_quantized_energy_consumption) reduction(+:estimated_area_utilisation)
	for (int i = 0; i < length; i++)
	{
		phenotype_node_count += phenotype_node_count_array[i];
		estimated_energy_consumption += estimated_energy_consumption_array[i];
		estimated_quantized_energy_consumption += estimated_quantized_energy_consumption_array[i];
		estimated_area_utilisation += estimated_area_utilisation_array[i];
	}

	this->estimated_quantized_energy_consumption = estimated_quantized_energy_consumption;
	this->estimated_energy_consumption = estimated_energy_consumption;
	this->estimated_area_utilisation = estimated_area_utilisation;
	this->phenotype_node_count = phenotype_node_count;

	need_energy_evaluation = false;
	need_gate_visit_map = false;
	return estimated_quantized_energy_consumption;
}

decltype(Chromosome::estimated_energy_consumption) Chromosome::get_estimated_energy_usage()
{
	//assert(("Chromosome::get_estimated_energy_usage cannot be called without calling Chromosome::evaluate before", !need_energy_evaluation || !need_evaluation));
	get_estimated_quantized_energy_usage();
	return estimated_energy_consumption;
}

decltype(Chromosome::estimated_area_utilisation) Chromosome::get_estimated_area_usage()
{
	assert(("Chromosome::get_estimated_area_usage cannot be called without calling Chromosome::evaluate before", !need_energy_evaluation || !need_evaluation));
	get_estimated_quantized_energy_usage();
	return estimated_area_utilisation;
}

decltype(Chromosome::estimated_delay) Chromosome::get_estimated_delay()
{
	assert(("Chromosome::get_estimated_delay cannot be called without calling Chromosome::evaluate before", !need_delay_evaluation || !need_evaluation));
	get_estimated_quantized_delay();
	return estimated_delay;
}

decltype(Chromosome::estimated_quantized_delay) Chromosome::get_estimated_quantized_delay()
{
	assert(("Chromosome::get_estimated_quantized_delay cannot be called without calling Chromosome::evaluate before", !need_delay_evaluation || !need_evaluation));

	if (!need_delay_evaluation)
	{
		return estimated_quantized_delay;
	}

	const int grid_size = cgp_configuration.col_count() * cgp_configuration.row_count();

	for (int i = 0; i < grid_size; i++)
	{
		quantized_delay_distance_map[i] = std::numeric_limits<quantized_delay_t>::max();
	}

	quantized_delay_t estimated_quantized_delay = 0;
	delay_t estimated_delay = 0;

	const int length = absolute_output_end - absolute_output_start;

#pragma omp parallel for shared(estimated_quantized_delay) shared(estimated_delay)
	for (int i = 0; i < length; i++) {

		std::stack<gene_t> pins_to_visit;
		std::stack<std::tuple<quantized_delay_t, delay_t>> current_delays;

		pins_to_visit.push(absolute_output_start[i]);
		current_delays.push(std::make_tuple(0, 0));

		while (!pins_to_visit.empty())
		{
			gene_t pin = pins_to_visit.top();
			std::tuple<quantized_delay_t, delay_t> parameters = current_delays.top();
			quantized_delay_t current_quantized_delay = std::get<0>(parameters);
			delay_t current_delay = std::get<1>(parameters);
			pins_to_visit.pop();
			current_delays.pop();


			// if is CGP input pin
			if (pin < cgp_configuration.input_count() || get_function_input_arity(get_gate_index_from_output_pin(pin)) == 0)
			{
				// then it is a constant node
				if (pin >= cgp_configuration.input_count())
				{
					const int gate_index = get_gate_index_from_output_pin(pin);
					gene_t func = *get_block_function(gate_index);
					current_quantized_delay += CGPConfiguration::get_quantized_delay_parameter(get_function_cost(func));
					current_delay += current_delay + CGPConfiguration::get_delay_parameter(get_function_cost(func));
				}

#pragma omp critical
				{
					estimated_quantized_delay = std::max(estimated_quantized_delay, current_quantized_delay);
					estimated_delay = std::max(estimated_delay, current_delay);
				}
				continue;
			}

			const int gate_index = get_gate_index_from_output_pin(pin);
			gene_t func = *get_block_function(gate_index);
			auto function_cost = get_function_cost(func);
			const auto new_quantized_cost = current_quantized_delay + CGPConfiguration::get_quantized_delay_parameter(function_cost);
			const auto new_real_cost = current_delay + CGPConfiguration::get_delay_parameter(function_cost);

			bool can_visit = false;

#pragma omp critical
			if (new_quantized_cost > quantized_delay_distance_map[gate_index] || quantized_delay_distance_map[gate_index] == std::numeric_limits<quantized_delay_t>::max())
			{
				can_visit = true;
				quantized_delay_distance_map[gate_index] = new_quantized_cost;
			}

			if (can_visit)
			{
				const auto arity = get_function_input_arity(gate_index);
				gene_t* inputs = get_block_inputs(gate_index);
				for (int j = 0; j < arity; j++)
				{
					pins_to_visit.push(inputs[j]);
					current_delays.push(std::make_tuple(new_quantized_cost, new_real_cost));
				}
			}
		}
	}
	this->estimated_quantized_delay = estimated_quantized_delay;
	this->estimated_delay = estimated_delay;
	need_delay_evaluation = false;
	return estimated_quantized_delay;
}

decltype(Chromosome::estimated_depth) Chromosome::get_estimated_depth()
{
#ifdef _DEPTH_DISABLED
	return CGPConfiguration::depth_nan;
#endif

	assert(("Chromosome::get_estimated_depth cannot be called without calling Chromosome::evaluate before", !need_depth_evaluation || !need_evaluation));

	if (!need_depth_evaluation)
	{
		return estimated_depth;
	}

	std::stack<gene_t> pins_to_visit;
	std::stack<depth_t> current_depths;

	for (int i = 0; i < cgp_configuration.col_count() * cgp_configuration.row_count(); i++)
	{
		depth_distance_map[i] = 0;
	}

	estimated_depth = 0;
	for (auto it = absolute_output_start; it != absolute_output_end; it++)
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
			estimated_depth = std::max(estimated_depth, current_depth);
			continue;
		}

		const int gate_index = get_gate_index_from_output_pin(pin);
		const auto new_cost = current_depth + 1;

		if (new_cost > depth_distance_map[gate_index])
		{
			depth_distance_map[gate_index] = new_cost;

			gene_t* inputs = get_block_inputs(gate_index);
			auto arity = get_function_input_arity(gate_index);
			for (int i = 0; i < arity; i++)
			{
				pins_to_visit.push(inputs[i]);
				current_depths.push(new_cost);
			}

			if (arity == 0)
			{
				estimated_depth = std::max(estimated_depth, current_depth);
				continue;
			}
		}
	}

	need_depth_evaluation = false;
	return estimated_depth;
}

decltype(Chromosome::phenotype_node_count) cgp::Chromosome::get_node_count()
{
	assert(("Chromosome::get_node_count cannot be called without calling Chromosome::evaluate before", !need_energy_evaluation || !need_evaluation));
	get_estimated_quantized_energy_usage();
	return phenotype_node_count;
}

decltype(Chromosome::top_row) Chromosome::get_top_row()
{
	assert(("Chromosome::get_top_row cannot be called without calling Chromosome::evaluate before", !need_energy_evaluation || !need_evaluation));
	get_estimated_quantized_energy_usage();
	return top_row;
}

decltype(Chromosome::bottom_row) Chromosome::get_bottom_row()
{
	assert(("Chromosome::get_bottom_row cannot be called without calling Chromosome::evaluate before", !need_energy_evaluation || !need_evaluation));
	get_estimated_quantized_energy_usage();
	return bottom_row;
}

int cgp::Chromosome::clip_pin(int pin) const
{
	assert(pin >= 0);
	//assert(cgp_configuration.function_output_arity() > 1);
	if (pin < cgp_configuration.input_count())
	{
		return pin;
	}

	auto pin_gate_index = get_gate_index_from_output_pin(pin);
	auto gate_base_pin = get_output_pin_from_gate_index(pin_gate_index);
	auto new_relative_pin = pin - gate_base_pin;
	int new_pin = gate_base_pin + std::min(get_function_output_arity(pin_gate_index) - 1, new_relative_pin);

	assert(gate_base_pin <= new_pin && new_pin < gate_base_pin + cgp_configuration.function_output_arity());
	return new_pin;
}

bool cgp::Chromosome::is_used_pin(int pin) const
{
	if (pin < cgp_configuration.input_count())
	{
		return true;
	}
	auto pin_gate_index = get_gate_index_from_output_pin(pin);
	auto gate_base_pin = get_output_pin_from_gate_index(pin_gate_index);
	auto new_relative_pin = pin - gate_base_pin;

	return new_relative_pin < get_function_output_arity(pin_gate_index);
}

decltype(Chromosome::first_col) Chromosome::get_first_column()
{
	assert(("Chromosome::get_first_column cannot be called without calling Chromosome::evaluate before", !need_energy_evaluation || !need_evaluation));
	get_estimated_quantized_energy_usage();
	return first_col;
}

decltype(Chromosome::last_col) Chromosome::get_last_column()
{
	assert(("Chromosome::get_last_column cannot be called without calling Chromosome::evaluate before", !need_energy_evaluation || !need_evaluation));
	get_estimated_quantized_energy_usage();
	return last_col;
}

decltype(cgp::Chromosome::multiplexing) cgp::Chromosome::is_multiplexing() const
{
	return multiplexing;
}

Chromosome::weight_output_t Chromosome::get_weights(const weight_input_t & input, int selector)
{
	set_input(input, selector);
	evaluate();
	auto weights = new weight_value_t[output_count];
	std::copy(begin_output(), end_output(), weights);
	return weights;
}

std::vector<Chromosome::weight_output_t> Chromosome::get_weights(const std::vector<weight_input_t>&input)
{
	std::vector<weight_output_t> weight_vector(input.size());
	for (int i = 0; i < input.size(); i++)
	{
		weight_vector.push_back(get_weights(input[i], i));
	}
	return weight_vector;
}

void cgp::Chromosome::find_direct_solutions(const dataset_t & dataset)
{
	const auto& inputs = get_dataset_input(dataset);
	const auto& no_care = get_dataset_no_care(dataset);
	const auto& outputs = get_dataset_output(dataset);
	gene_t* chromosome_outputs = get_outputs();

	for (int i = 0; i < inputs.size(); i++)
	{
		for (int j = 0; j < cgp_configuration.input_count(); j++)
		{
			for (int k = 0; k < no_care[i]; k++)
			{
				int output = 256;
				bool can_have_direct_solution = true;
				for (int l = 0; l < outputs.size(); l++)
				{
					if (outputs[l][k] != output && output != 256)
					{
						can_have_direct_solution = false;
						break;
					}
					output = outputs[l][k];
				}

				if (can_have_direct_solution && inputs[i][j] == outputs[i][k])
				{
#ifdef __VIRTUAL_SELECTOR
					chromosome_outputs[cgp_configuration.dataset_size() * i + k] = j;
					break;
#else
					chromosome_outputs[k] = j;
					break;
#endif
				}
			}
		}
	}
}

void cgp::Chromosome::add_2pow_circuits(const dataset_t & dataset)
{
	assert(("add_2pow_circuits does not support datasets with multiple combinatios", cgp_configuration.dataset_size() == 1));
	const auto& inputs = get_dataset_input(dataset);
	const auto& no_care = get_dataset_no_care(dataset);
	const auto& outputs = get_dataset_output(dataset);
	std::vector<int> values;
	std::map<int, int> pin_mapping;
	values.push_back(0);
	values.push_back(-128);
	for (int i = 1; i < 128; i *= 2)
	{
		values.push_back(i);
	}

	for (int i = 0; i < inputs.size(); i++)
	{
		for (int j = 0; j < cgp_configuration.input_count(); j++)
		{
			pin_mapping[inputs[i][j]] = j;
		}
	}

	assert(("grid is way too small to use add_2pow_circuits", values.size() < cgp_configuration.row_count() * cgp_configuration.col_count()));

	int free_index = find_free_index(0);
	bool one_skip_col = false;
	if (pin_mapping.find(0) == pin_mapping.end())
	{
		*get_block_function(free_index) = CGPOperator::ZERO_CONST;
		pin_mapping[0] = get_output_pin_from_gate_index(free_index);
		free_index = find_free_index(free_index);
	}

	if (pin_mapping.find(1) == pin_mapping.end())
	{
		*get_block_function(free_index) = CGPOperator::ONE_CONST;
		pin_mapping[1] = get_output_pin_from_gate_index(free_index);
		free_index = find_free_index(free_index);
		one_skip_col = true;
	}

	if (pin_mapping.find(-128) == pin_mapping.end())
	{
		*get_block_function(free_index) = CGPOperator::EXPECTED_VALUE_MIN;
		pin_mapping[-128] = get_output_pin_from_gate_index(free_index);
	}

	int index = (one_skip_col) ? (find_free_index(cgp_configuration.row_count())) : (find_free_index(free_index));

	for (int num : values)
	{
		if (num <= 1 || pin_mapping.find(num) != pin_mapping.end()) continue;
		index = find_free_index(index);
		int col = get_column(index);
		int next_col_index;
		switch (num)
		{
		case 2:
			pin_mapping[num] = index * cgp_configuration.function_output_arity() + cgp_configuration.input_count();
			*get_block_function(index) = CGPOperator::DOUBLE;
			*get_block_inputs(index) = pin_mapping[1];
			break;
		case 4:
			pin_mapping[num] = index * cgp_configuration.function_output_arity() + cgp_configuration.input_count();
			*get_block_function(index) = CGPOperator::L_SHIFT_2;
			*get_block_inputs(index) = pin_mapping[1];
			break;
		case 8:
			pin_mapping[num] = index * cgp_configuration.function_output_arity() + cgp_configuration.input_count();
			*get_block_function(index) = CGPOperator::L_SHIFT_3;
			*get_block_inputs(index) = pin_mapping[1];
			break;
		case 16:
			pin_mapping[num] = index * cgp_configuration.function_output_arity() + cgp_configuration.input_count();
			*get_block_function(index) = CGPOperator::L_SHIFT_4;
			*get_block_inputs(index) = pin_mapping[1];
			break;
		case 32:
			pin_mapping[num] = index * cgp_configuration.function_output_arity() + cgp_configuration.input_count();
			*get_block_function(index) = CGPOperator::L_SHIFT_5;
			*get_block_inputs(index) = pin_mapping[1];
			break;
		case 64:
			next_col_index = find_free_index((col + 1) * cgp_configuration.row_count());
			pin_mapping[num] = next_col_index * cgp_configuration.function_output_arity() + cgp_configuration.input_count();
			*get_block_function(next_col_index) = CGPOperator::DOUBLE;
			*get_block_inputs(next_col_index) = pin_mapping[32];
			break;
		default:
			throw std::invalid_argument("unexpected num in switch branch: " + std::to_string(num));
			break;
		}
	}

	gene_t* chromosome_outputs = get_outputs();
	for (int i = 0; i < outputs.size(); i++)
	{
		for (int j = 0; j < no_care[i]; j++)
		{
			auto result = pin_mapping.find(outputs[i][j]);

			if (result != pin_mapping.end())
			{
#ifdef __VIRTUAL_SELECTOR
				chromosome_outputs[cgp_configuration.dataset_size() * i + j] = result->second;
#else
				chromosome_outputs[j] = result->second;
#endif
			}
		}
	}
}

int cgp::Chromosome::find_free_index(int from) const
{
	for (int i = from; i < cgp_configuration.row_count() * cgp_configuration.col_count(); i++)
	{
		if (!gate_visit_map[i] && !is_locked_node(i))
		{
			return i;
		}
	}
	throw std::invalid_argument("could not find free index after: " + std::to_string(from));
}

int cgp::Chromosome::mulitplexed_value_to_index(int value) const
{
	assert(("cannot use mulitplexed_value_to_index without multiplexing mode", multiplexing));
	return value + 128 + start_id_index;
}

int cgp::Chromosome::mulitplexed_value_to_relative_index(int value) const
{
	assert(("cannot use relative_mulitplexed_value_to_index without multiplexing mode", multiplexing));
	return value + 128;
}

int cgp::Chromosome::mulitplexed_index_to_value(int index) const
{
	assert(("cannot use mulitplexed_value_to_index without multiplexing mode", multiplexing));
	return index - start_id_index - 128;
}

int cgp::Chromosome::relative_mulitplexed_index_to_value(int index) const
{
	assert(("cannot use relative_mulitplexed_index_to_value without multiplexing mode", multiplexing));
	return index - 128;
}

int cgp::Chromosome::get_relative_id_output_from_index(int index) const
{
	assert(("cannot use get_relative_mx_output_from_index without multiplexing mode", multiplexing));
	assert(("index out of range", 0 <= index && index < 256));
	return get_gate_output(start_id_index + index);
}

void cgp::Chromosome::cross_rng(Chromosome& other)
{
	rng.set_seed(rng.generate() * other.rng.generate());
	other.rng.set_seed(rng.generate() * other.rng.generate());
}

int cgp::Chromosome::get_id_output_for(int value) const
{
	assert(("cannot use get_id_output_for without multiplexing mode", multiplexing));
	return get_gate_output(mulitplexed_value_to_index(value));
}

std::map<int, int> cgp::Chromosome::get_gate_statistics()
{
	std::map<int, int> stats;
	get_estimated_energy_usage();

	for (int i = 0; i < CGPOperator::DEMUX + 1; i++)
	{
		stats[i] = 0;
	}
	stats[CGPOperator::ID] = 0;
	for (int i = 0; i < cgp_configuration.row_count() * cgp_configuration.col_count(); i++)
	{
		if (gate_visit_map[i])
		{
			stats[*get_block_function(i)] += 1;
		}
	}
	return stats;
}

bool cgp::Chromosome::needs_evaluation() const
{
	return need_evaluation;
}

bool cgp::Chromosome::needs_energy_evaluation() const
{
	return need_energy_evaluation;
}

bool cgp::Chromosome::needs_delay_evaluation() const
{
	return need_delay_evaluation;
}

void cgp::Chromosome::use_multiplexing(const dataset_t & dataset)
{
	if (multiplexing)
	{
		return;
	}

	multiplexing = true;
	const auto& outputs = get_dataset_output(dataset);
	const auto& inputs = get_dataset_input(dataset);
	const auto grid_size = cgp_configuration.row_count() * cgp_configuration.col_count();
	gene_t function = (cgp_configuration.dataset_size() == 1) ? (CGPOperator::ID) : (CGPOperator::MUX);
	id_count = 256;
	mux_count = (function == CGPOperator::ID) ? (0) : (output_count);
	start_mux_index = grid_size - mux_count;

	if (mux_count == 0)
	{
		start_id_index = grid_size - id_count;
	}
	else
	{
		start_id_index = (get_column(start_mux_index) - 1) * cgp_configuration.row_count() - id_count;
	}

	locked_nodes_index = start_id_index;

	for (int id_i = start_id_index; id_i < start_id_index + id_count; id_i++)
	{
		*get_block_function(id_i) = CGPOperator::ID;

		int num = mulitplexed_index_to_value(id_i);
		for (int dataset_i = 0; dataset_i < cgp_configuration.dataset_size(); dataset_i++)
		{
			for (int input_i = 0; input_i < cgp_configuration.input_count(); input_i++)
			{
				if (inputs[dataset_i][input_i] == num)
				{
					get_block_inputs(id_i)[dataset_i] = input_i;
					break;
				}
			}
		}
	}

	if (mux_count != 0)
	{
		// Connect MUXs to 256 IDs
		for (int mux_i = start_mux_index; mux_i < start_mux_index + id_count; mux_i++)
		{
			int id_index = mux_i - start_mux_index + start_id_index;
			*get_block_function(mux_i) = CGPOperator::MUX;

			for (int dataset_i = 0; dataset_i < cgp_configuration.dataset_size(); dataset_i++)
			{
				get_block_inputs(mux_i)[dataset_i] = get_output_pin_from_gate_index(id_index, dataset_i);
			}
		}
	}

	// Connect first 256 outputs to either ID or MUX
	const int output_facade_index = (mux_count != 0) ? (start_mux_index) : (start_id_index);
	for (int j = 0; j < id_count; j++)
	{
		get_outputs()[j] = get_output_pin_from_gate_index(output_facade_index + j);
	}
	
	setup_output_iterators(selector, id_count);
	invalidate();
}

void cgp::Chromosome::remove_multiplexing(const dataset_t & dataset)
{
	if (!multiplexing)
	{
		return;
	}

	const auto& outputs = get_dataset_output(dataset);
	locked_nodes_index = std::numeric_limits<decltype(locked_nodes_index)>::max();
	gene_t function = (mux_count == 0) ? (CGPOperator::ID) : (CGPOperator::MUX);
	multiplexing = false;
	setup_output_iterators(selector, cgp_configuration.output_count());

	if (function == CGPOperator::ID) {
		for (int j = 0; j < output_count; j++)
		{
			int pin = get_output_pin_from_gate_index(mulitplexed_value_to_index(outputs[0][j]));
			assert(pin >= cgp_configuration.input_count());
			int id_index = get_gate_index_from_output_pin(pin);

			if (*get_block_function(id_index) != CGPOperator::ID)
			{
				get_outputs()[j] = pin;
			}
			else 
			{
				auto inputs = get_block_inputs(id_index);
				get_outputs()[j] = inputs[0];
			}
		}
	}
	else
	{
		for (int dataset_i = 0; dataset_i < cgp_configuration.dataset_size(); dataset_i++) {
			for (int output_i = 0; output_i < output_count; output_i++)
			{
				int mux_id = start_mux_index + output_i;
				int id_index = mulitplexed_value_to_index(outputs[dataset_i][output_i]);
				// Connect MUX pin to specific ID input (ID is redundant)
				get_block_inputs(mux_id)[dataset_i] = get_block_inputs(id_index)[dataset_i];
				// Connect Output to MUX
				get_outputs()[output_i] = get_output_pin_from_gate_index(mux_id, 0);
			}
		}
	}
	id_count = 0;
	mux_count = 0;
	invalidate();
}

void cgp::Chromosome::copy_gate_input_pins(int src_index, int dst_index)
{
	auto src_inputs = get_block_inputs(src_index);
	auto dst_inputs = get_block_inputs(dst_index);
	for (int i = 0; i < cgp_configuration.function_input_arity(); i++)
	{
		dst_inputs[i] = src_inputs[i];
	}
}

void cgp::Chromosome::copy_gate_input_pins(int src_index, int dst_index, int src_pin, int dst_pin)
{
	auto src_inputs = get_block_inputs(src_index);
	auto dst_inputs = get_block_inputs(dst_index);
	dst_inputs[dst_pin] = src_inputs[src_pin];
}

void cgp::Chromosome::copy_gate_function(int src_index, int dst_index)
{
	auto src_function = get_block_function(src_index);
	auto dst_function = get_block_function(dst_index);
	*dst_function = *src_function;
}

void cgp::Chromosome::perform_corrections(const dataset_t & dataset, const int threshold, const bool zero_energy_only, const bool only_id)
{
	if (cgp_configuration.dataset_size() > 1)
	{
		throw std::invalid_argument("phenotype cannot be corrected when dataset is larger than 1");
	}
	need_evaluation = true;
	evaluate();
	get_estimated_energy_usage();

	bool valid = true;
	const auto& inputs = get_dataset_input(dataset);

	for (int i = start_id_index; i < start_id_index + id_count; i++)
	{
		const int expected_value = mulitplexed_index_to_value(i);
		const int col = get_column(i);

		// The pin is correct, nothing to do
		if (expected_value == get_id_output_for(expected_value))
		{
			continue;
		}

		const int actual_value = get_id_output_for(expected_value);
		const int delta = actual_value - expected_value;
		const int abs_delta = std::abs(delta);
		int best_delta = std::min(abs_delta, threshold), best_pin_index = -1, best_function = -1;
		quantized_energy_t best_energy = CGPConfiguration::get_quantized_energy_parameter(get_function_cost(*get_block_function(i)));
		auto input = std::make_unique<weight_value_t[]>(cgp_configuration.function_input_arity());
		auto output = std::make_unique<weight_value_t[]>(cgp_configuration.function_output_arity());

		std::vector<gene_t> functions;
		functions.push_back(CGPOperator::ID);

		if (!only_id)
		{
			for (int func = 0; func < cgp_configuration.function_count(); func++)
			{
				functions.push_back(func);
			}
		}

		for (int gate_index = 0, end = cgp_configuration.row_count() * (col - 1); (best_energy != 0 || best_delta != 0) && gate_index < end; gate_index++)
		{
			const auto& value = get_gate_output(gate_index);
			const int pin_index = get_gate_output_used_pin(gate_index);
			for (const auto& func : functions)
			{
				if (get_function_input_arity_2(func) == 1)
				{
					input[0] = value;
				}
				else if (get_function_input_arity_2(func) == 2)
				{
					input[0] = actual_value;
					input[1] = value;
				}
				else if (get_function_input_arity_2(func) == 0)
				{
					// pass
				}
				else
				{
					throw std::runtime_error("unsupported function " + std::to_string(func));
				}

				evaluate_single_from_values(input.get(), output.get(), func);
				const int delta = std::abs(output[0] - expected_value);
				const auto energy = CGPConfiguration::get_quantized_energy_parameter(get_function_cost(func));
				if (delta <= best_delta && energy <= best_energy && !(zero_energy_only && energy != 0) && gate_visit_map[gate_index])
				{
					best_delta = delta;
					best_energy = energy;
					best_pin_index = pin_index;
					best_function = func;
				}
			}
		}
		// If there is only 1 data pairing, an input shortcircuit optimisation can be done
		for (int j = 0; j < cgp_configuration.input_count() && (best_energy != 0 || best_delta != 0); j++)
		{
			for (const auto& func : functions)
			{
				if (get_function_input_arity_2(func) == 1)
				{
					input[0] = inputs[0][j];
				}
				else if (get_function_input_arity_2(func) == 2)
				{
					input[0] = actual_value;
					input[1] = inputs[0][j];
				}
				else if (get_function_input_arity_2(func) == 0)
				{
					// pass
				}
				else
				{
					throw std::runtime_error("unsupported function " + std::to_string(func));
				}

				evaluate_single_from_values(input.get(), output.get(), func);
				const int delta = std::abs(output[0] - expected_value);
				const auto energy = CGPConfiguration::get_quantized_energy_parameter(get_function_cost(func));
				if (delta <= best_delta && energy <= best_energy && !(zero_energy_only && energy != 0))
				{
					best_delta = delta;
					best_energy = energy;
					best_pin_index = j;
					best_function = func;
				}
			}
		}

		if (best_pin_index != -1)
		{
			*get_block_function(i) = best_function;
			auto arity = get_function_input_arity_2(best_function);
			if (arity == 1 || arity == 0)
			{
				get_block_inputs(i)[0] = best_pin_index;
			}
			else if (arity == 2)
			{
				get_block_inputs(i)[1] = best_pin_index;
			}
			else
			{
				throw std::runtime_error("unsupported function arity in correction");
			}
			valid = false;
		}
	}
	if (!valid)
	{
		invalidate();
	}
}

cgp::Chromosome::gate_parameters_t cgp::Chromosome::get_function_cost(gene_t function) const
{
	if (function == CGPOperator::ID)
	{
		return id_gate_parameters;
	}

#if defined(__OLD_NOOP_OPERATION_SUPPORT)
	// todo: remove after virtual selector is retired
	// Skip old ID function at index 0
	function++;
#endif // __OLD_NOOP_OPERATION_SUPPORT

	const auto& costs = cgp_configuration.function_costs();
	return costs[function];
}

std::ostream& cgp::operator<<(std::ostream & os, const Chromosome & chromosome)
{
	os << chromosome.to_string();
	return os;
}

std::string cgp::to_string(const cgp::Chromosome & chromosome)
{
	return chromosome.to_string();
}

std::string cgp::to_string(const std::shared_ptr<Chromosome>&chromosome)
{
	return (chromosome) ? (to_string(*chromosome)) : Chromosome::nan_chromosome_string;
}

#pragma once

#include "Chromosome.h"

Chromosome::quantized_energy_t Chromosome::get_raw_quantized_energy() {
	return estimated_quantized_energy_consumption;
}

Chromosome::energy_t Chromosome::get_raw_energy() {
	return estimated_energy_consumption;
}

Chromosome::area_t Chromosome::get_raw_area() {
	return estimated_area_utilisation;
}

Chromosome::quantized_delay_t Chromosome::get_raw_quantized_delay() {
	return estimated_quantized_delay;
}

Chromosome::delay_t Chromosome::get_raw_delay() {
	return estimated_delay;
}

Chromosome::gate_count_t Chromosome::get_raw_gate_count() {
	return phenotype_node_count;
}

void cgp::RandomNumberGenerator::set_seed(uint64_t seed)
{
	this->seed = seed;
}

uint64_t cgp::RandomNumberGenerator::get_seed()
{
	return seed;
}

uint64_t cgp::RandomNumberGenerator::generate()
{
	constexpr uint64_t multiplier = 1664525;
	constexpr uint64_t increment = 1013904223;
	//const uint64_t modulus = 4294967296;

	seed = multiplier * seed + increment;

	return seed;
}
