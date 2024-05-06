#include "CGPStream.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace cgp;

CGPOutputStream::CGPOutputStream(std::shared_ptr<CGP> cgp_model, const std::string& out)
	: OutputStream(out) {
	this->cgp_model = cgp_model;
}

CGPOutputStream::CGPOutputStream(std::shared_ptr<CGP> cgp_model, const std::string& out, std::ios_base::openmode mode)
	: OutputStream(out, mode) {
	this->cgp_model = cgp_model;
}

CGPOutputStream::CGPOutputStream(std::shared_ptr<CGP> cgp_model, const std::string& out, std::shared_ptr<std::ostream> default_output)
	: OutputStream(out, default_output) {
	this->cgp_model = cgp_model;
}

CGPOutputStream::CGPOutputStream(std::shared_ptr<CGP> cgp_model, const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode)
	: OutputStream(out, default_output, mode) {
	this->cgp_model = cgp_model;
}

CGPOutputStream::CGPOutputStream(std::shared_ptr<CGP> cgp_model, const std::string& out, const std::unordered_map<std::string, std::string>& variables)
	: OutputStream(out, variables) {
	this->cgp_model = cgp_model;
}

CGPOutputStream::CGPOutputStream(std::shared_ptr<CGP> cgp_model, const std::string& out, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables)
	: OutputStream(out, mode, variables) {
	this->cgp_model = cgp_model;
}

CGPOutputStream::CGPOutputStream(std::shared_ptr<CGP> cgp_model, const std::string& out, std::shared_ptr<std::ostream> default_output, const std::unordered_map<std::string, std::string>& variables)
	: OutputStream(out, default_output, variables) {
	this->cgp_model = cgp_model;
}

CGPOutputStream::CGPOutputStream(std::shared_ptr<CGP> cgp_model, const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables)
	: OutputStream(out, default_output, mode, variables) {
	this->cgp_model = cgp_model;
}

void CGPOutputStream::log_human(size_t run, size_t generation, bool show_chromosome)
{
	log_human(run, generation, cgp_model->get_best_solution());
}

void CGPOutputStream::log_human(size_t run, size_t generation, const CGP::solution_t& solution, bool show_chromosome)
{
	if (is_ignoring_output())
	{
		return;
	}

	const auto& chromosome = CGP::get_chromosome(solution);
	*this
		<< std::setprecision(12)
		<< "[" << (run + 1)
		<< ", " << (generation + 1) << "] MSE: "
		<< error_to_string(CGP::get_error(solution))
		<< ", QEnergy: "
		<< quantized_energy_to_string(CGP::get_quantized_energy(solution))
		<< ", Energy: "
		<< energy_to_string(CGP::get_energy(solution))
		<< ", Area: "
		<< area_to_string(CGP::get_area(solution))
		<< ", QDelay: "
		<< quantized_delay_to_string(CGP::get_quantized_delay(solution))
		<< ", Delay: "
		<< delay_to_string(CGP::get_delay(solution))
		<< ", Depth: "
		<< depth_to_string(CGP::get_depth(solution))
		<< ", Gates: "
		<< gate_count_to_string(CGP::get_gate_count(solution))
#ifdef _DISABLE_ROW_COL_STATS
		<< ", Top row: "
		<< ((chromosome) ? std::to_string(chromosome->get_top_row()) : ("nan"))
		<< ", Bottom row: "
		<< ((chromosome) ? std::to_string(chromosome->get_bottom_row()) : ("nan"))
		<< ", First col: "
		<< ((chromosome) ? std::to_string(chromosome->get_first_column()) : ("nan"))
		<< ", Last col: "
		<< ((chromosome) ? std::to_string(chromosome->get_last_column()) : ("nan"))
#endif
		<< ", Chromosome: "
		<< ((chromosome && show_chromosome) ? chromosome->to_string() : (Chromosome::nan_chromosome_string))
		<< std::endl;
}

void cgp::CGPOutputStream::log_csv_header()
{
	if (is_ignoring_output())
	{
		return;
	}
	*this << "run,generation,timestamp,error,quantized_energy,energy,area,quantized_delay,delay,depth,gate_count,chromosome" << std::endl;
}

void CGPOutputStream::log_csv(size_t run, size_t generation, const std::string& timestamp, bool print_chromosome)
{
	return log_csv(run, generation, timestamp, cgp_model->get_best_solution(), print_chromosome);
}

void CGPOutputStream::log_csv(size_t run, size_t generation, const std::string& timestamp, const CGP::solution_t& solution, bool print_chromosome)
{
	if (is_ignoring_output())
	{
		return;
	}
	*this
		<< std::setprecision(12)
		<< (run + 1)
		<< "," << (generation + 1)
		<< ",\"" << timestamp << "\","
		<< error_to_string(CGP::get_error(solution)) << ","
		<< quantized_energy_to_string(CGP::get_quantized_energy(solution)) << ","
		<< energy_to_string(CGP::get_energy(solution)) << ","
		<< area_to_string(CGP::get_area(solution)) << ","
		<< quantized_delay_to_string(CGP::get_quantized_delay(solution)) << ","
		<< delay_to_string(CGP::get_delay(solution)) << ","
		<< depth_to_string(CGP::get_depth(solution)) << ","
		<< gate_count_to_string(CGP::get_gate_count(solution))
		<< (",\"" + ((print_chromosome) ? (CGP::get_chromosome(solution)->to_string()) : ("")) + "\"")
		<< std::endl;
}

void CGPOutputStream::log_csv(
	size_t run,
	size_t generation,
	std::shared_ptr<Chromosome> chromosome,
	const dataset_t& dataset,
	bool print_chromosome)
{
	if (is_ignoring_output())
	{
		return;
	}

	auto solution = cgp_model->evaluate(dataset, chromosome);
	log_csv(run, generation, "", solution, print_chromosome);
}

void CGPOutputStream::log_weights(const std::vector<weight_input_t>& inputs)
{
	log_weights(cgp_model->get_best_chromosome(), inputs);
}

void CGPOutputStream::log_weights(std::shared_ptr<Chromosome> chromosome, const std::vector<weight_input_t>& inputs)
{
	if (is_ignoring_output())
	{
		return;
	}

	for (int i = 0; i < inputs.size(); i++) {
		auto weights = chromosome->get_weights(inputs[i], i);

		for (size_t i = 0; i < cgp_model->output_count() - 1; i++)
		{
			*this << weight_to_string(weights[i]);
			*this << " ";
		}

		*this << weight_to_string(weights[cgp_model->output_count() - 1]);		
		*this << std::endl;
		stream->flush();
		delete[] weights;
	}
}

void cgp::CGPOutputStream::dump()
{
	if (is_ignoring_output())
	{
		return;
	}
	cgp_model->dump(*stream);
}

void cgp::CGPOutputStream::dump_all()
{
	if (is_ignoring_output())
	{
		return;
	}
	cgp_model->dump_all(*stream);
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in)
	: InputStream(in) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::ios_base::openmode mode)
	: InputStream(in, mode) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input)
	: InputStream(in, default_input) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode)
	: InputStream(in, default_input, mode) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, const std::unordered_map<std::string, std::string>& variables)
	: InputStream(in, variables) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables)
	: InputStream(in, mode, variables) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, const std::unordered_map<std::string, std::string>& variables)
	: InputStream(in, default_input, variables) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables)
	: InputStream(in, default_input, mode, variables) {
	this->cgp_model = cgp_model;
}

CGPCSVRow cgp::CGPInputStream::read_csv_line() const
{
	if (!eof() && !fail())
	{
		CGPCSVRow row;
		std::string line;
		std::getline(*stream, line);
		row.raw_line = line;
		auto begin = line.find_first_of('{');

		if (begin == std::string::npos)
		{
			row.ok = false;
			return row;
		}


		std::istringstream iss(line);
		char comma;

		// Parse fields
		if (!(iss >> row.run >> comma >> row.generation >> comma >> row.error >> comma
			>> row.quantized_energy >> comma >> row.energy >> comma >> row.area
			>> comma >> row.quantized_delay >> comma >> row.delay >> comma
			>> row.depth >> comma >> row.gate_count >> comma >> row.timestamp >> comma >> row.chromosome)) {
			row.ok = false;
			return row;
		}

		// Adjust run and generation
		row.run--;
		row.generation--;

		return row;
	}
	else if (eof())
	{
		throw std::ios::failure("end of line reached");
	}
	else
	{
		throw std::ios::failure("error while reading from the csv");
	}
}

weight_input_t cgp::CGPInputStream::load_input()
{
	weight_repr_value_t weight;
	std::string no_care;
	auto input = new weight_value_t[cgp_model->input_count()];
	for (int i = 0; i < cgp_model->input_count(); i++)
	{
		if (*stream >> weight)
		{
			input[i] = static_cast<weight_value_t>(weight);
			continue;
		}
		else if (stream->eof())
		{
			throw std::invalid_argument("not enough value values passed; got: " + std::to_string(i) + ", need: " + std::to_string(cgp_model->input_count()));
		}
		else
		{
			stream->clear();
		}

		if (*stream >> no_care)
		{
			if (no_care == "x")
			{
				input[i] = 0;
			}
			else
			{
				throw std::invalid_argument("invalit no care value: expecting \"x\"; got: \"" + no_care + "\"");
			}
		}
		else {
			throw std::invalid_argument("invalit unknown input value: expecting weight value or x");
		}
	}
	return input;
}

std::tuple<weight_output_t, int> cgp::CGPInputStream::load_output()
{
	weight_repr_value_t weight;
	std::string no_care;
	auto output = new weight_value_t[cgp_model->output_count()];
	int no_care_index = cgp_model->output_count();
	for (int i = 0; i < cgp_model->output_count(); i++)
	{
		if (*stream >> weight)
		{
			output[i] = static_cast<weight_value_t>(weight);
			continue;
		}
		else if (stream->eof())
		{
			throw std::invalid_argument("not enough value values passed; got: " + std::to_string(i) + ", need: " + std::to_string(cgp_model->output_count()));
		}
		else
		{
			stream->clear();
		}

		if (*stream >> no_care)
		{
			if (no_care == "x")
			{
				no_care_index = i;
				if (no_care_index == 0)
				{
					throw std::invalid_argument("one line in output contains only no care values!");
				}

				break;
			}
			else
			{
				throw std::invalid_argument("invalit no care value: expecting \"x\"; got: \"" + no_care + "\"");
			}
		}
		else {
			throw std::invalid_argument("invalit unknown output value: expecting weight value or x");
		}
	}

	for (int i = no_care_index+1; i < cgp_model->output_count(); i++)
	{
		*stream >> no_care;
	}

	return std::make_tuple(output, no_care_index);
}

dataset_t cgp::CGPInputStream::load_train_data()
{
	std::vector<weight_input_t> inputs;
	std::vector<weight_output_t> outputs;
	std::vector<int> no_care;
	std::array<int, 256> needed_values {0};
	for (size_t i = 0; i < cgp_model->dataset_size(); i++)
	{
		inputs.push_back(load_input());
		auto output_data = load_output();
		outputs.push_back(std::get<0>(output_data));
		no_care.push_back(std::get<1>(output_data));
	}

	for (int i = 0; i < outputs.size(); i++)
	{
		for (int j = 0; j < no_care[i]; j++)
		{
			needed_values[outputs[i][j] + 128] += 1;
		}
	}

	return std::make_tuple(inputs, outputs, no_care, needed_values);
}

std::unique_ptr<CGPConfiguration::gate_parameters_t[]> cgp::CGPInputStream::load_gate_parameters()
{
	int bit_variant = cgp_model->max_multiplexer_bit_variant();
	int function_count = cgp_model->function_count();
	bool has_mux = Chromosome::is_mux(function_count - 2);

	if (has_mux && bit_variant == 0)
	{
		throw std::invalid_argument("multiplexer is unneccesary when dataset size is 1");
	}

	auto costs = std::make_unique<CGPConfiguration::gate_parameters_t[]>(cgp_model->function_count());

	int end = (has_mux) ? (function_count - 2) : (function_count);
	std::string quantized_energy, energy, area, quantized_delay, delay;
	for (int i = 0; i < end; i++)
	{
		auto parameters = CGPConfiguration::get_default_gate_parameters();
		if (*stream >> quantized_energy >> energy >> area >> quantized_delay >> delay)
		{
			CGPConfiguration::set_quantized_energy_parameter(parameters, quantized_energy);
			CGPConfiguration::set_energy_parameter(parameters, energy);
			CGPConfiguration::set_area_parameter(parameters, area);
			CGPConfiguration::set_quantized_delay_parameter(parameters, quantized_delay);
			CGPConfiguration::set_delay_parameter(parameters, delay);
			costs[i] = parameters;
			// Print all parameters in a single row
			std::cout 
				<< std::setprecision(12)
				<< "Quantized Energy: " << quantized_energy << ", "
				<< "Energy: " << energy << ", "
				<< "Area: " << area << ", "
				<< "Quantized Delay: " << quantized_delay << ", "
				<< "Delay: " << delay << std::endl;
		}
		else
		{
			throw std::invalid_argument("could not read all gate parameters on line: " + std::to_string(i + 1));
		}
	}

	// skip mux and demux configurations
	for (int i = end, j = 0; i < end + bit_variant * 2; i++, j = (j == 0) ? (1) : (0))
	{
		auto parameters = CGPConfiguration::get_default_gate_parameters();
		if (*stream >> quantized_energy >> energy >> area >> quantized_delay >> delay)
		{
			CGPConfiguration::set_quantized_energy_parameter(parameters, quantized_energy);
			CGPConfiguration::set_energy_parameter(parameters, energy);
			CGPConfiguration::set_area_parameter(parameters, area);
			CGPConfiguration::set_quantized_delay_parameter(parameters, quantized_delay);
			CGPConfiguration::set_delay_parameter(parameters, delay);
			costs[end + j] = parameters;
			std::cout 
				<< std::setprecision(12)
				<< "Loading " << (Chromosome::is_mux(end + j) ? ("mux") : ("demux")) << "[" << (end + j) << "]\n\t"
				<< "Quantized Energy: " << quantized_energy << ", "
				<< "Energy: " << energy << ", "
				<< "Area: " << area << ", "
				<< "Quantized Delay: " << quantized_delay << ", "
				<< "Delay: " << delay << std::endl;
		}
		else
		{
			throw std::invalid_argument("could not read all gate parameters on line: " + std::to_string(i + 1));
		}
	}

	return costs;
}
