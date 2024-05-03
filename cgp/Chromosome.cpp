#include "Chromosome.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stack>
#include <cassert>
#include <iostream>

using namespace cgp;

const std::string Chromosome::nan_chromosome_string = "null";

Chromosome::Chromosome(const CGPConfiguration& cgp_configuration, const std::unique_ptr<std::tuple<int, int>[]>& minimum_output_indicies) :
	cgp_configuration(cgp_configuration),
	minimum_output_indicies(minimum_output_indicies)
{
	setup_maps();
	setup_output_iterators(0);
	setup_chromosome();
}

cgp::Chromosome::Chromosome(const CGPConfiguration& cgp_config, const std::unique_ptr<std::tuple<int, int>[]>& minimum_output_indicies, const std::string& serialized_chromosome) :
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
	need_energy_evaluation = that.need_energy_evaluation;
	need_delay_evaluation = that.need_delay_evaluation;
	need_depth_evaluation = that.need_depth_evaluation;
	selector = that.selector;
	setup_maps(that.chromosome);
	setup_output_iterators(that.selector);
	estimated_quantized_energy_consumption = that.estimated_quantized_energy_consumption;
	estimated_energy_consumption = that.estimated_energy_consumption;
	estimated_area_utilisation = that.estimated_area_utilisation;
	estimated_quantized_delay = that.estimated_quantized_delay;
	estimated_delay = that.estimated_delay;
	estimated_depth = that.estimated_depth;
	phenotype_node_count = that.phenotype_node_count;
	input = that.input;
	invalidate();

	//if (!need_evaluation)
	//{
	//	std::copy(that.absolute_pin_start, that.absolute_pin_end, absolute_pin_start);
	//}
}

Chromosome::Chromosome(Chromosome&& that) noexcept :
	cgp_configuration(that.cgp_configuration),
	minimum_output_indicies(that.minimum_output_indicies) {
	need_evaluation = that.need_evaluation;
	need_energy_evaluation = that.need_energy_evaluation;
	need_delay_evaluation = that.need_delay_evaluation;
	need_depth_evaluation = that.need_depth_evaluation;
	selector = that.selector;
	setup_maps(std::move(that));
	estimated_quantized_energy_consumption = that.estimated_quantized_energy_consumption;
	estimated_energy_consumption = that.estimated_energy_consumption;
	estimated_area_utilisation = that.estimated_area_utilisation;
	estimated_quantized_delay = that.estimated_quantized_delay;
	estimated_delay = that.estimated_delay;
	estimated_depth = that.estimated_depth;
	phenotype_node_count = that.phenotype_node_count;
	input = std::move(that.input);
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
	for (int i = 0; i < cgp_configuration.output_count() * cgp_configuration.dataset_size(); i++) {
		*ite++ = (rand() % (max - min)) + min;
	}
}

void Chromosome::setup_chromosome(const std::string& serialized_chromosome)
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
	setup_output_iterators(0);
	
	size_t block_size = cgp_configuration.function_input_arity() + 1;
	auto it = chromosome.get();
	auto target_row = cgp_configuration.row_count();
	auto target_col = cgp_configuration.col_count();

	if (target_row != row_count || target_col != col_count)
	{
		setup_chromosome();
	}

	for (size_t j = 0; j < col_count; j++)
	{
		for (size_t i = 0; i < row_count; i++)
		{
			// ([n]i,i,...,f)
			if (!(input >> discard >> discard >> number_discard >> discard))
			{
				throw std::invalid_argument("invalid format of the gate ID");
			}

			auto target_block = (j * target_row + i) * block_size;
			for (size_t k = 0; k < block_size; k++)
			{
				if (!(input >> it[target_block + k] >> discard))
				{
					throw std::invalid_argument("invalid format of the chromosome gate definition\n" + serialized_chromosome);
				}
			}
		}
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
		
		int gate_index = (pin - input_count) / cgp_configuration.function_output_arity();
		int pin_delta = (pin - input_count) - gate_index * cgp_configuration.function_output_arity();
		int row = gate_index % row_count;
		int col = gate_index / row_count;
		int transformed_index = col * target_row + row;
		*it = input_count + transformed_index * cgp_configuration.function_output_arity() + pin_delta;
	}

#ifdef __CGP_DEBUG 
	auto check_chromosome = to_string();
	assert(("Chromosome::Chromosome serialized chromosome does not correspond to built chromosome", check_chromosome == serialized_chromosome));
#endif
}

void Chromosome::setup_maps()
{
	setup_maps(nullptr);
}

void Chromosome::setup_maps(const decltype(chromosome)& chromosome)
{
	int locket_output_size = cgp_configuration.output_count() * cgp_configuration.dataset_size();
	this->chromosome = std::make_unique<gene_t[]>(cgp_configuration.chromosome_size());
	locked_outputs = std::make_unique<bool[]>(locket_output_size);
	std::copy(locked_outputs.get(), locked_outputs.get() + locket_output_size, locked_outputs.get());
	if (chromosome != nullptr)
	{
		std::copy(chromosome.get(), chromosome.get() + cgp_configuration.chromosome_size(), this->chromosome.get());
	}
	else
	{
		setup_chromosome();
	}

	pin_map = std::make_unique<weight_value_t[]>(cgp_configuration.pin_map_size());
	gate_visit_map = std::make_unique<bool[]>(cgp_configuration.row_count() * cgp_configuration.col_count());
	absolute_output_start = this->chromosome.get() + cgp_configuration.blocks_chromosome_size();
	absolute_output_end = absolute_output_start + cgp_configuration.output_count() * cgp_configuration.dataset_size();
	absolute_pin_start = pin_map.get() + cgp_configuration.row_count() * cgp_configuration.col_count() * cgp_configuration.function_output_arity();
	absolute_pin_end = absolute_pin_start + cgp_configuration.output_count() * cgp_configuration.dataset_size();
}

void Chromosome::setup_maps(Chromosome&& that)
{
	chromosome = std::move(that.chromosome);
	pin_map = std::move(that.pin_map);
	gate_visit_map = std::move(that.gate_visit_map);
	locked_outputs = std::move(that.locked_outputs);
	std::swap(output_start, that.output_start);
	std::swap(output_end, that.output_end);
	std::swap(output_pin_start, that.output_pin_start);
	std::swap(output_pin_end, that.output_pin_end);
	std::swap(absolute_output_start, that.absolute_output_start);
	std::swap(absolute_output_end, that.absolute_output_end);
	std::swap(absolute_pin_start, that.absolute_pin_start);
	std::swap(absolute_pin_end, that.absolute_pin_end);
}

void cgp::Chromosome::setup_output_iterators(int selector)
{
	output_start = chromosome.get() + cgp_configuration.blocks_chromosome_size() + selector * cgp_configuration.output_count();
	output_end = output_start + cgp_configuration.output_count();
	output_pin_start = absolute_pin_start + selector * cgp_configuration.output_count();
	output_pin_end = output_pin_start + cgp_configuration.output_count();
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
	return chromosome.get() + gate_index * (cgp_configuration.function_input_arity() + 1) + cgp_configuration.function_input_arity();
}

const std::unique_ptr<Chromosome::gene_t[]>& Chromosome::get_chromosome() const
{
	return chromosome;
}

void Chromosome::mutate_genes(std::shared_ptr<Chromosome> that) const
{
	// Number of genes to mutate
	int genes_to_mutate = (rand() % cgp_configuration.max_genes_to_mutate()) + 1;
	for (int i = 0; i < genes_to_mutate; i++) {
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
			// output is locked, try again
			if (locked_outputs[get_output_position(random_gene_index)])
			{
				i--;
				continue;
			}
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
	that->estimated_quantized_energy_consumption = CGPConfiguration::quantized_energy_nan;
	that->estimated_energy_consumption = CGPConfiguration::energy_nan;
	that->estimated_area_utilisation = CGPConfiguration::area_nan;
	that->estimated_quantized_delay = CGPConfiguration::quantized_delay_nan;
	that->estimated_delay = CGPConfiguration::delay_nan;
	that->estimated_depth = CGPConfiguration::depth_nan;
	that->phenotype_node_count = CGPConfiguration::gate_count_nan;
}

std::shared_ptr<Chromosome> Chromosome::mutate() const
{
	auto chrom = std::make_shared<Chromosome>(*this);
	mutate_genes(chrom);
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

void Chromosome::set_input(const weight_value_t* input, int selector)
{
	this->input = input;
	this->selector = selector;
	invalidate();
	setup_output_iterators(selector);
}

inline static void set_value(CGPConfiguration::weight_value_t& target, const CGPConfiguration::weight_value_t& value)
{
	target = (value != CGPConfiguration::invalid_value) ? (value) : (CGPConfiguration::invalid_value);
}

bool Chromosome::is_mux(int func)
{
#ifdef CNN_FP32_WEIGHTS
	return func == 9;
#else
	return func == 27;
#endif
}

bool Chromosome::is_demux(int func)
{
#ifdef CNN_FP32_WEIGHTS
	return func == 10;
#else
	return func == 28;
#endif
}

Chromosome::weight_value_t Chromosome::get_pin_value(int index) const
{
	return (index < cgp_configuration.input_count()) ? (input[index]) : (pin_map[index - cgp_configuration.input_count()]);
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
	const auto expected_value_min = cgp_configuration.expected_value_min();
	const auto expected_value_max = cgp_configuration.expected_value_max();
	int used_pin = 0;
	for (int i = 0; i < cgp_configuration.col_count() * cgp_configuration.row_count(); i++)
	{
		auto block_output_pins = output_pin;
		gate_visit_map[i] = false;
		switch (*func) {
		case 0:
			set_value(block_output_pins[0], minus(expected_value_max, get_pin_value(input_pin[0])));
			break;
		case 1:
			set_value(block_output_pins[0], plus(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
			break;
		case 2:
			set_value(block_output_pins[0], minus(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
			break;
		case 3:
			set_value(block_output_pins[0], mul(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
			break;
		case 4:
			set_value(block_output_pins[0], neg(get_pin_value(input_pin[0])));
			break;
		case 5:
			set_value(block_output_pins[0], plus(expected_value_min, get_pin_value(input_pin[0])));
			break;
#ifdef CNN_FP32_WEIGHTS
		case 6:
			set_value(block_output_pins[0], get_pin_value(input_pin[0]) * 0.25);
			break;
#else
		case 6:
			set_value(block_output_pins[0], bit_rshift(get_pin_value(input_pin[0]), 2));
			break;
#endif
#ifdef CNN_FP32_WEIGHTS
		case 7:
			set_value(block_output_pins[0], get_pin_value(input_pin[0]) * 0.5);
			break;
#else
		case 7:
			set_value(block_output_pins[0], bit_rshift(get_pin_value(input_pin[0]), 1));
			break;
#endif
#ifndef CNN_FP32_WEIGHTS
		case 8:
			set_value(block_output_pins[0], bit_and(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
			break;
		case 9:
			set_value(block_output_pins[0], bit_or(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
			break;
		case 10:
			set_value(block_output_pins[0], bit_xor(get_pin_value(input_pin[0]), get_pin_value(input_pin[1])));
			break;
		case 11:
			set_value(block_output_pins[0], bit_neg(get_pin_value(input_pin[0])));
			break;
		case 12:
			set_value(block_output_pins[0], bit_lshift(get_pin_value(input_pin[0]), 1));
			break;
		case 13:
			set_value(block_output_pins[0], plus(get_pin_value(input_pin[0]), 1));
			break;
		case 14:
			set_value(block_output_pins[0], minus(get_pin_value(input_pin[0]), 1));
			break;
		case 15:
			set_value(block_output_pins[0], bit_rshift(get_pin_value(input_pin[0]), 3));
			break;
		case 16:
			set_value(block_output_pins[0], bit_rshift(get_pin_value(input_pin[0]), 4));
			break;
		case 17:
			set_value(block_output_pins[0], bit_rshift(get_pin_value(input_pin[0]), 5));
			break;
		case 18:
			set_value(block_output_pins[0], bit_lshift(get_pin_value(input_pin[0]), 2));
			break;
		case 19:
			set_value(block_output_pins[0], bit_lshift(get_pin_value(input_pin[0]), 3));
			break;
		case 20:
			set_value(block_output_pins[0], bit_lshift(get_pin_value(input_pin[0]), 4));
			break;
		case 21:
			set_value(block_output_pins[0], bit_lshift(get_pin_value(input_pin[0]), 5));
			break;
		case 22:
			set_value(block_output_pins[0], 1);
			break;
		case 23:
			set_value(block_output_pins[0], -1);
			break;
		case 24:
			set_value(block_output_pins[0], 0);
			break;
		case 25:
			set_value(block_output_pins[0], expected_value_min);
			break;
		case 26:
			set_value(block_output_pins[0], expected_value_max);
			break;
			// multiplexor
		case 27:
			set_value(block_output_pins[0], get_pin_value(input_pin[selector]));
			break;
			// de-multiplexor
		case 28:
			set_value(block_output_pins[selector], get_pin_value(input_pin[0]));
			used_pin = selector;
			break;
#else
		case 8:
			set_value(block_output_pins[0], get_pin_value(input_pin[0]) * 1.05);
			break;
			// multiplexor
		case 9:
			set_value(block_output_pins[0], get_pin_value(input_pin[selector]));
			break;
			// de-multiplexor
		case 10:
			set_value(block_output_pins[selector], get_pin_value(input_pin[0]));
			used_pin = selector;
			break;
#endif
		default:
			throw std::runtime_error("this Chromosome::evaluate branch is not implemented! branch: " + std::to_string(*func));
			break;
		}

		if (is_demux(*func))
		{
			for (int i = 0; i < cgp_configuration.function_output_arity(); i++)
			{
				set_value(block_output_pins[i], (i == used_pin) ? (block_output_pins[i]) : (0));
			}
		}
		else
		{
			// speed up evolution of operator + multiplexor; shortcircuit pins
			for (int i = 0; i < cgp_configuration.function_output_arity(); i++)
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
		*pin_it = get_pin_value(*it);
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
		const int col = index / cgp_configuration.row_count() + 1;
		start_gate_index = col * cgp_configuration.row_count();
	}
	const int end_index = (gate_index / cgp_configuration.row_count()) * cgp_configuration.row_count();
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
	return (pin - cgp_configuration.input_count()) / cgp_configuration.function_output_arity();
}

int cgp::Chromosome::get_gate_index_from_input_pin(int pin) const
{
	return (pin - cgp_configuration.input_count()) / cgp_configuration.function_input_arity();
}

void cgp::Chromosome::invalidate()
{
	need_evaluation = true;
	need_energy_evaluation = true;
	need_delay_evaluation = true;
	need_depth_evaluation = true;
}

decltype(Chromosome::estimated_quantized_energy_consumption) cgp::Chromosome::get_estimated_quantized_energy_usage()
{
	assert(("Chromosome::estimate_energy_usage cannot be called without calling Chromosome::evaluate before", !need_evaluation));

	if (!need_energy_evaluation)
	{
		return estimated_quantized_energy_consumption;
	}

	std::stack<gene_t> pins_to_visit;

	for (auto it = absolute_output_start; it != absolute_output_end; it++)
	{
		pins_to_visit.push(*it);
	}

	estimated_energy_consumption = 0;
	estimated_quantized_energy_consumption = 0;
	estimated_area_utilisation = 0;
	phenotype_node_count = 0;
	bottom_row = cgp_configuration.row_count();
	top_row = 0;
	first_col = cgp_configuration.col_count();
	last_col = 0;
	const auto& reference_costs = cgp_configuration.function_costs();

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
		if (!gate_visit_map[gate_index])
		{
#ifndef _DISABLE_ROW_COL_STATS
			int row = gate_index / cgp_configuration.col_count();
			int col = gate_index / cgp_configuration.row_count();
			top_row = std::min(row, top_row);
			bottom_row = std::max(row, bottom_row);
			first_col = std::min(col, first_col);
			last_col = std::max(col, last_col);
#endif
			gene_t func = *get_block_function(gate_index);
			gate_visit_map[gate_index] = true;
			phenotype_node_count += 1;
			estimated_energy_consumption += CGPConfiguration::get_energy_parameter(reference_costs[func]);
			estimated_quantized_energy_consumption += CGPConfiguration::get_quantized_energy_parameter(reference_costs[func]);
			estimated_area_utilisation += CGPConfiguration::get_area_parameter(reference_costs[func]);
			gene_t* inputs = get_block_inputs(gate_index);

			for (int i = 0; i < cgp_configuration.function_input_arity(); i++)
			{
				pins_to_visit.push(inputs[i]);
			}
		}
	}

	need_energy_evaluation = false;
	return estimated_quantized_energy_consumption;
}

decltype(Chromosome::estimated_energy_consumption) Chromosome::get_estimated_energy_usage()
{
	assert(("Chromosome::get_estimated_energy_usage cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_quantized_energy_usage();
	return estimated_energy_consumption;
}

decltype(Chromosome::estimated_area_utilisation) Chromosome::get_estimated_area_usage()
{
	assert(("Chromosome::get_estimated_area_usage cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_quantized_energy_usage();
	return estimated_area_utilisation;
}

decltype(Chromosome::estimated_delay) Chromosome::get_estimated_delay()
{
	assert(("Chromosome::get_estimated_delay cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_quantized_delay();
	return estimated_delay;
}

decltype(Chromosome::estimated_quantized_delay) Chromosome::get_estimated_quantized_delay()
{
	assert(("Chromosome::get_estimated_quantized_delay cannot be called without calling Chromosome::evaluate before", !need_evaluation));

	if (!need_delay_evaluation)
	{
		return estimated_quantized_delay;
	}

	std::stack<gene_t> pins_to_visit;
	std::stack<std::tuple<quantized_delay_t, delay_t>> current_delays;
	auto distance_map = std::make_unique<quantized_delay_t[]>(cgp_configuration.col_count() * cgp_configuration.row_count());

	estimated_quantized_delay = 0;
	estimated_delay = 0;
	for (auto it = absolute_output_start; it != absolute_output_end; it++)
	{
		pins_to_visit.push(*it);
		current_delays.push(std::make_tuple(0, 0));
	}

	const auto& reference_costs = cgp_configuration.function_costs();
	while (!pins_to_visit.empty())
	{
		gene_t pin = pins_to_visit.top();
		std::tuple<quantized_delay_t, delay_t> parameters = current_delays.top();
		quantized_delay_t current_quantized_delay = std::get<0>(parameters);
		delay_t current_delay = std::get<1>(parameters);
		pins_to_visit.pop();
		current_delays.pop();

		// if is CGP input pin
		if (pin < cgp_configuration.input_count())
		{
			estimated_quantized_delay = std::max(estimated_quantized_delay, current_quantized_delay);
			estimated_delay = std::max(estimated_delay, current_delay);
			continue;
		}

		const int gate_index = get_gate_index_from_output_pin(pin);
		gene_t func = *get_block_function(gate_index);
		const auto new_quantized_cost = current_quantized_delay + CGPConfiguration::get_quantized_delay_parameter(reference_costs[func]);
		const auto new_real_cost = current_delay + CGPConfiguration::get_delay_parameter(reference_costs[func]);

		if (new_quantized_cost > distance_map[gate_index])
		{
			distance_map[gate_index] = new_quantized_cost;
			gene_t* inputs = get_block_inputs(gate_index);
			for (int i = 0; i < cgp_configuration.function_input_arity(); i++)
			{
				pins_to_visit.push(inputs[i]);
				current_delays.push(std::make_tuple(new_quantized_cost, new_real_cost));
			}
		}
	}

	need_delay_evaluation = false;
	return estimated_quantized_delay;
}

decltype(Chromosome::estimated_depth) Chromosome::get_estimated_depth()
{
#ifdef _DEPTH_DISABLED
	return CGPConfiguration::depth_nan;
#endif

	assert(("Chromosome::get_estimated_depth cannot be called without calling Chromosome::evaluate before", !need_evaluation));

	if (!need_depth_evaluation)
	{
		return estimated_depth;
	}

	std::stack<gene_t> pins_to_visit;
	std::stack<depth_t> current_depths;
	auto distance_map = std::make_unique<quantized_delay_t[]>(cgp_configuration.col_count() * cgp_configuration.row_count());

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

		if (new_cost > distance_map[gate_index])
		{
			distance_map[gate_index] = new_cost;
			gene_t* inputs = get_block_inputs(gate_index);
			for (int i = 0; i < cgp_configuration.function_input_arity(); i++)
			{
				pins_to_visit.push(inputs[i]);
				current_depths.push(new_cost);
			}
		}
	}

	need_depth_evaluation = false;
	return estimated_depth;
	}

decltype(Chromosome::phenotype_node_count) cgp::Chromosome::get_node_count()
{
	assert(("Chromosome::get_node_count cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_quantized_energy_usage();
	return phenotype_node_count;
}

decltype(Chromosome::top_row) Chromosome::get_top_row()
{
	assert(("Chromosome::get_top_row cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_quantized_energy_usage();
	return top_row;
}

decltype(Chromosome::bottom_row) Chromosome::get_bottom_row()
{
	assert(("Chromosome::get_bottom_row cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_quantized_energy_usage();
	return bottom_row;
}

decltype(Chromosome::first_col) Chromosome::get_first_column()
{
	assert(("Chromosome::get_first_column cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_quantized_energy_usage();
	return first_col;
}

decltype(Chromosome::last_col) Chromosome::get_last_column()
{
	assert(("Chromosome::get_last_column cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_quantized_energy_usage();
	return last_col;
}

Chromosome::weight_output_t Chromosome::get_weights(const weight_input_t& input, int selector)
{
	set_input(input, selector);
	evaluate();
	auto weights = new weight_value_t[cgp_configuration.output_count()];
	std::copy(begin_output(), end_output(), weights);
	return weights;
}

std::vector<Chromosome::weight_output_t> Chromosome::get_weights(const std::vector<weight_input_t>& input)
{
	std::vector<weight_output_t> weight_vector(input.size());
	for (int i = 0; i < input.size(); i++)
	{
		weight_vector.push_back(get_weights(input[i], i));
	}
	return weight_vector;
}

void cgp::Chromosome::find_direct_solutions(const dataset_t& dataset)
{
	const auto& inputs = get_dataset_input(dataset);
	const auto& no_care = get_dataset_no_care(dataset);
	const auto& outputs = get_dataset_output(dataset);
	gene_t *chromosome_outputs = get_outputs();

	for (int i = 0; i < inputs.size(); i++)
	{
		for (int j = 0; j < cgp_configuration.input_count(); j++)
		{
			for (int k = 0; k < no_care[i]; k++)
			{
				if (inputs[i][j] == outputs[i][k])
				{
					locked_outputs[cgp_configuration.dataset_size() * i + k] = true;
					chromosome_outputs[cgp_configuration.dataset_size() * i + k] = j;
					break;
				}
			}
		}
	}
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
