#include "Chromosome.h"
#include <iostream>
#include <numeric>
#include <sstream>

using namespace cgp;

Chromosome::Chromosome(const CGPConfiguration& cgp_configuration, std::shared_ptr<std::tuple<int, int>[]> minimum_output_indicies, double expected_value_min, double expected_value_max) :
	cgp_configuration(cgp_configuration),
	minimum_output_indicies(minimum_output_indicies),
	expected_value_min(expected_value_min),
	expected_value_max(expected_value_max)
{
	chromosome = std::make_unique<gene_t[]>(cgp_configuration.chromosome_size());
	pin_map = std::make_unique<double[]>(cgp_configuration.pin_map_size());


	output_start = chromosome.get() + cgp_configuration.blocks_chromosome_size();
	output_end = output_start + cgp_configuration.output_count();
	output_pin_start = pin_map.get() + cgp_configuration.row_count() * cgp_configuration.col_count() * cgp_configuration.function_output_arity();
	output_pin_end = output_pin_start + cgp_configuration.output_count();

	setup();
}

Chromosome::Chromosome(const Chromosome& that) :
	cgp_configuration(that.cgp_configuration), minimum_output_indicies(that.minimum_output_indicies), expected_value_min(that.expected_value_min), expected_value_max(that.expected_value_max) {
	input = that.input;
	need_evaluation = that.need_evaluation;
	pin_map = that.pin_map;
	chromosome = that.chromosome;
	output_start = that.output_start;
	output_end = that.output_end;
	output_pin_start = that.output_pin_start;
	output_pin_end = that.output_pin_end;
}

Chromosome& Chromosome::operator=(const Chromosome& that) {
	input = that.input;
	need_evaluation = that.need_evaluation;
	pin_map = that.pin_map;
	chromosome = that.chromosome;
	output_start = that.output_start;
	output_end = that.output_end;
	output_pin_start = that.output_pin_start;
	output_pin_end = that.output_pin_end;
	return *this;
}

bool Chromosome::is_function(size_t position) const
{
	return !is_output(position) && !is_input(position);
}


bool Chromosome::is_input(size_t position) const
{
	return !is_output(position) && position % 3 != 2;
}


bool Chromosome::is_output(size_t position) const
{
	return position >= cgp_configuration.blocks_chromosome_size();
}


void Chromosome::setup()
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

Chromosome::gene_t* Chromosome::get_outputs() const {
	return output_start;
}

Chromosome::gene_t* Chromosome::get_block_inputs(size_t row, size_t column) const {
	return chromosome.get() + (row * column) * (cgp_configuration.function_input_arity() + 1);
}


Chromosome::gene_t* Chromosome::get_block_function(size_t row, size_t column) const {
	return chromosome.get() + (row * column) * (cgp_configuration.function_input_arity() + 1) + cgp_configuration.function_input_arity();
}


std::shared_ptr<Chromosome::gene_t[]> Chromosome::get_chromosome() const
{
	return chromosome;
}


std::shared_ptr<Chromosome> Chromosome::mutate()
{
	auto chrom = std::make_shared<Chromosome>(Chromosome(*this));
	chrom->chromosome = std::make_unique<Chromosome::gene_t[]>(cgp_configuration.chromosome_size());
	chrom->pin_map = std::make_unique<double[]>(cgp_configuration.pin_map_size());

	std::copy(pin_map.get(), output_pin_end, chrom->pin_map.get());
	std::copy(chromosome.get(), chromosome.get() + cgp_configuration.chromosome_size(), chrom->chromosome.get());


	chrom->output_start = chrom->chromosome.get() + cgp_configuration.blocks_chromosome_size();
	chrom->output_end = chrom->output_start + cgp_configuration.output_count();
	chrom->output_pin_start = chrom->pin_map.get() + cgp_configuration.row_count() * cgp_configuration.col_count() * cgp_configuration.function_output_arity();
	chrom->output_pin_end = chrom->output_pin_start + cgp_configuration.output_count();

	// Number of genes to mutate
	auto genes_to_mutate = (rand() % cgp_configuration.mutation_max()) + 1;
	for (auto i = 0; i < genes_to_mutate; i++) {
		// Select a random gene
		auto random_gene_index = rand() % cgp_configuration.chromosome_size();
		auto random_number = rand();

		if (is_input(random_gene_index))
		{
			int column_index = (int)(i / cgp_configuration.row_count());
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
	return chrom;
}


void Chromosome::set_input(std::shared_ptr<double[]> input)
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
	if (!need_evaluation)
	{
		return;
	}

	auto output_pin = pin_map.get() + cgp_configuration.input_count();
	auto input_pin = chromosome.get();
	auto func = chromosome.get() + cgp_configuration.function_input_arity();
	for (size_t col = 0; col < cgp_configuration.col_count(); col++)
	{
		for (size_t row = 0; row < cgp_configuration.row_count(); row++)
		{
			auto block_output_pins = output_pin;

			switch (*func) {
			case 0:
				block_output_pins[0] = pin_map[input_pin[0]];
				block_output_pins[1] = pin_map[input_pin[1]];
				break;
			case 1:
				block_output_pins[0] = pin_map[input_pin[1]];
				block_output_pins[1] = pin_map[input_pin[1]];
				break;
			case 2:
				block_output_pins[0] = pin_map[input_pin[0]] + pin_map[input_pin[1]];
				block_output_pins[1] = std::max(expected_value_min, std::min(block_output_pins[0], expected_value_max));
				break;
			case 3:
				block_output_pins[0] = pin_map[input_pin[0]] - pin_map[input_pin[1]];
				block_output_pins[1] = std::max(expected_value_min, std::min(block_output_pins[0], expected_value_max));
				break;
			case 4:
				block_output_pins[0] = pin_map[input_pin[0]] * pin_map[input_pin[1]];
				block_output_pins[1] = std::max(expected_value_min, std::min(block_output_pins[0], expected_value_max));
				break;

			case 5:
				block_output_pins[0] = -pin_map[input_pin[0]];
				block_output_pins[1] = -pin_map[input_pin[1]];
				break;
			case 6:
				block_output_pins[0] = std::max(expected_value_min, std::min(pin_map[input_pin[0]], expected_value_max));
				block_output_pins[1] = std::max(expected_value_min, std::min(pin_map[input_pin[1]], expected_value_max));
				break;

			case 7:
				block_output_pins[0] = 0.001;
				block_output_pins[1] = 0.010;
				break;
			case 8:
				block_output_pins[0] = 0.5;
				block_output_pins[1] = 0.05;
				break;
			case 9:
				block_output_pins[0] = 1.0 / pin_map[input_pin[0]];
				block_output_pins[1] = 1.0 / pin_map[input_pin[1]];
				break;
			case 10:
				block_output_pins[0] = 1.5;
				block_output_pins[1] = 1.05;
				break;
			default:;
				block_output_pins[0] = 0xffffffff;
			}
			output_pin += cgp_configuration.function_output_arity();
			input_pin += cgp_configuration.function_input_arity() + 1;
			func += cgp_configuration.function_input_arity() + 1;
		}
	}

	auto it = output_start;
	auto pin_it = output_pin_start;
	for (; it != output_end; it++, pin_it++)
	{
		*pin_it = pin_map[*it];
	}
	need_evaluation = false;
}

double* Chromosome::begin_output()
{
	return output_pin_start;
}


double* Chromosome::end_output()
{
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

std::ostream& cgp::operator<<(std::ostream& os, const Chromosome& chromosome)
{
	os << chromosome.to_string();
	return os;
}

std::string cgp::to_string(const cgp::Chromosome& chromosome)
{
	return chromosome.to_string();
}
