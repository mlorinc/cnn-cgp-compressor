#include "Chromosome.h"
#include <iostream>
#include <numeric>
#include <sstream>
#include <stack>
#include <cassert>

using namespace cgp;

Chromosome::Chromosome(CGPConfiguration& cgp_configuration, std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies, weight_actual_value_t expected_value_min, weight_actual_value_t expected_value_max) :
	cgp_configuration(cgp_configuration),
	minimum_output_indicies(minimum_output_indicies),
	expected_value_min(expected_value_min),
	expected_value_max(expected_value_max)
{
	setup_maps();
	setup_iterators();
	setup_chromosome();
}

cgp::Chromosome::Chromosome(CGPConfiguration& cgp_config, std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies, weight_actual_value_t expected_value_min, weight_actual_value_t expected_value_max, const std::string& serialized_chromosome) :
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

	cgp_configuration
		.input_count(input_count)
		.output_count(output_count)
		.col_count(col_count)
		.row_count(row_count)
		.function_input_arity(function_input_arity)
		.function_output_arity(1)
		.look_back_parameter(l_back);
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

Chromosome::Chromosome(const Chromosome& that) :
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
	energy_map = std::make_unique<double[]>(cgp_configuration.row_count() * cgp_configuration.col_count());
	energy_visit_map = std::make_unique<bool[]>(cgp_configuration.row_count() * cgp_configuration.col_count());
}

void cgp::Chromosome::setup_iterators()
{
	output_start = chromosome.get() + cgp_configuration.blocks_chromosome_size();
	output_end = output_start + cgp_configuration.output_count();
	output_pin_start = pin_map.get() + cgp_configuration.row_count() * cgp_configuration.col_count() * cgp_configuration.function_output_arity();
	output_pin_end = output_pin_start + cgp_configuration.output_count();
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


void Chromosome::evaluate()
{
	assert(("Chromosome::evaluate cannot be called without calling Chromosome::set_input before", input != nullptr));

	if (!need_evaluation)
	{
		return;
	}

	auto output_pin = pin_map.get() + cgp_configuration.input_count();
	auto input_pin = chromosome.get();
	auto func = chromosome.get() + cgp_configuration.function_input_arity();
	auto reference_energy_map = cgp_configuration.function_energy_costs();
	for (size_t i = 0; i < cgp_configuration.col_count() * cgp_configuration.row_count(); i++)
	{
		auto block_output_pins = output_pin;
		energy_map[i] = reference_energy_map[*func];
		energy_visit_map[i] = false;
		switch (*func) {
		case 0:
			block_output_pins[0] = pin_map[input_pin[0]];
			break;
		case 1:
			block_output_pins[0] = expected_value_max - pin_map[input_pin[0]];
			break;
		case 2:
			block_output_pins[0] = expected_value_max + pin_map[input_pin[0]];
			break;
		case 3:
			block_output_pins[0] = pin_map[input_pin[0]] + pin_map[input_pin[1]];
			break;
		case 4:
			block_output_pins[0] = pin_map[input_pin[0]] - pin_map[input_pin[1]];
			break;
		case 5:
			block_output_pins[0] = pin_map[input_pin[0]] * pin_map[input_pin[1]];
			break;
		case 6:
			block_output_pins[0] = -pin_map[input_pin[0]];
			break;
		case 7:
			block_output_pins[0] = std::max(expected_value_min, std::min(pin_map[input_pin[0]], expected_value_max));
			break;
		case 8:
			block_output_pins[0] = expected_value_min - pin_map[input_pin[0]];
			break;
		case 9:
			block_output_pins[0] = expected_value_min + pin_map[input_pin[0]];
			break;
		case 10:
			block_output_pins[0] = pin_map[input_pin[0]] * 0.25;
			break;
		case 11:
			block_output_pins[0] = pin_map[input_pin[0]] * 1.5;
			break;
		case 12:
			block_output_pins[0] = pin_map[input_pin[0]] * 1.05;
			break;
		case 13:
			block_output_pins[0] = pin_map[input_pin[0]] * 0.5;
			break;
#ifndef CNN_FP32_WEIGHTS
		case 14:
			block_output_pins[0] = pin_map[input_pin[0]] & pin_map[input_pin[1]];
			break;
		case 15:
			block_output_pins[0] = pin_map[input_pin[0]] | pin_map[input_pin[1]];
			break;
		case 16:
			block_output_pins[0] = pin_map[input_pin[0]] ^ pin_map[input_pin[1]];
			break;
		case 17:
			block_output_pins[0] = ~(pin_map[input_pin[0]]);
			break;
		case 18:
			block_output_pins[0] = pin_map[input_pin[0]] >> 1;
			break;
		case 19:
			block_output_pins[0] = pin_map[input_pin[0]] << 1;
			break;
		case 20:
			block_output_pins[0] = pin_map[input_pin[0]] + 1;
			break;
		case 21:
			block_output_pins[0] = pin_map[input_pin[0]] - 1;
			break;
#endif
		default:
			block_output_pins[0] = 0xffffffff;
			break;
		}
		// Prevent overflows hence undefined behaviour
		//block_output_pins[0] = std::max(expected_value_min, std::min(block_output_pins[0], expected_value_max));
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
	for (auto it = output_start; it != output_end; it++, pin_it++)
	{
		pins_to_visit.push(*it);
	}

	estimated_energy_consumptation = 0;
	phenotype_node_count = 0;
	while (!pins_to_visit.empty())
	{
		auto pin = pins_to_visit.top();
		pins_to_visit.pop();

		// if is CGP input pin
		if (pin < cgp_configuration.input_count())
		{
			continue;
		}

		auto gate_index = (pin - cgp_configuration.input_count()) / cgp_configuration.function_output_arity();

		if (energy_visit_map[gate_index])
		{
			continue;
		}

		energy_visit_map[gate_index] = true;
		phenotype_node_count += 1;
		estimated_energy_consumptation += energy_map[gate_index];
		gene_t* inputs = get_block_inputs(gate_index);

		for (size_t i = 0; i < cgp_configuration.function_input_arity(); i++)
		{
			pins_to_visit.push(inputs[i]);
		}
	}

	need_energy_evaluation = false;
	return estimated_energy_consumptation;
}

decltype(Chromosome::phenotype_node_count) cgp::Chromosome::get_node_count()
{
	assert(("Chromosome::get_node_count cannot be called without calling Chromosome::evaluate before", !need_evaluation));
	get_estimated_energy_usage();
	return phenotype_node_count;
}

std::shared_ptr<Chromosome::weight_value_t[]> Chromosome::get_weights(const std::shared_ptr<weight_value_t[]> input)
{
	set_input(input);
	evaluate();
	auto weights = std::make_shared<weight_value_t[]>(cgp_configuration.output_count());
	std::copy(begin_output(), end_output(), weights.get());
	return weights;
}

std::vector<std::shared_ptr<Chromosome::weight_value_t[]>> Chromosome::get_weights(const std::vector<std::shared_ptr<weight_value_t[]>>& input)
{
	std::vector<std::shared_ptr<Chromosome::weight_value_t[]>> weight_vector(input.size());
	for(const auto &in : input)
	{
		weight_vector.push_back(get_weights(in));
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
