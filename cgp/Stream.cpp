#include "Stream.h"
#include "StringTemplate.h"
#include <iostream>

using namespace cgp;

std::shared_ptr<weight_value_t[]> load_output(std::istream& in, size_t output_size, weight_repr_value_t& min, weight_repr_value_t& max)
{
	weight_repr_value_t weight;
	std::string no_care;
	std::shared_ptr<weight_value_t[]> output = std::make_shared<weight_value_t[]>(output_size);

	for (size_t i = 0; i < output_size; i++)
	{
		if (in >> weight)
		{
			output[i] = static_cast<weight_value_t>(weight);
			min = std::min(min, weight);
			max = std::max(max, weight);
			continue;
		}
		else if (in.eof())
		{
			throw std::invalid_argument("not enough value values passed; got: " + std::to_string(i) + ", need: " + std::to_string(output_size));
		}
		else
		{
			in.clear();
		}

		if (in >> no_care)
		{
			if (no_care == "x")
			{
				// Ignore value when evaluating fitness
				output[i] = CGPConfiguration::no_care_value;
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
	return output;
}

InputStream::InputStream(const std::string& in)
	: InputStream(in, nullptr, std::ios::in)
{
}

InputStream::InputStream(const std::string& in, std::ios_base::openmode mode)
	: InputStream(in, nullptr, mode)
{
}

InputStream::InputStream(const std::string& in, std::shared_ptr<std::istream> default_input)
	: InputStream(in, default_input, std::ios::in)
{
}

InputStream::InputStream(const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode)
	: InputStream(in, default_input, mode, {})
{

}

InputStream::InputStream(const std::string& in, const std::unordered_map<std::string, std::string> &variables) :
	InputStream(in, nullptr, std::ios::in, variables)
{
}

InputStream::InputStream(const std::string& in, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string> &variables)
	: InputStream(in, nullptr, mode, variables)
{
}

InputStream::InputStream(const std::string& in, std::shared_ptr<std::istream> default_input, const std::unordered_map<std::string, std::string> &variables)
	: InputStream(in, default_input, std::ios::in, variables)
{
}

InputStream::InputStream(const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string> &variables)
{
	if (in == "-")
	{
		stream.reset(&std::cin, [](...) {});
	}
	else if (in.empty())
	{
		if (default_input == nullptr)
		{
			throw std::invalid_argument("default input must not be null when input file is not provided");
		}

		stream = default_input;
	}
	else
	{
		const auto& path = replace_string_variables(in, variables);
		auto file = new std::ifstream(path, mode);
		if (!file->is_open())
		{
			delete file;
			throw std::ofstream::failure("could not open input file " + path + " (" + in + ")");
		}
		stream.reset(file);
	}
}

void cgp::InputStream::close()
{
	stream.reset();
}

OutputStream::OutputStream(const std::string& out)
	: OutputStream(out, nullptr, std::ios::out)
{
}

OutputStream::OutputStream(const std::string& out, std::ios_base::openmode mode)
	: OutputStream(out, nullptr, mode)
{
}

OutputStream::OutputStream(const std::string& out, std::shared_ptr<std::ostream> default_output)
	: OutputStream(out, default_output, std::ios::out)
{
}

OutputStream::OutputStream(const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode)
	: OutputStream(out, default_output, mode, {})
{

}

OutputStream::OutputStream(const std::string& out, const std::unordered_map<std::string, std::string> &variables) :
	OutputStream(out, nullptr, std::ios::out, variables)
{
}

OutputStream::OutputStream(const std::string& out, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string> &variables)
	: OutputStream(out, nullptr, mode, variables)
{
}

OutputStream::OutputStream(const std::string& out, std::shared_ptr<std::ostream> default_output, const std::unordered_map<std::string, std::string> &variables)
	: OutputStream(out, default_output, std::ios::out, variables)
{
}

OutputStream::OutputStream(const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string> &variables)
{
	if (out == "-")
	{
		stream.reset(&std::cout, [](...) {});
	}
	else if (out == "+")
	{
		stream.reset(&std::cerr, [](...) {});
	}
	else if (out.empty())
	{
		if (default_output == nullptr)
		{
			throw std::invalid_argument("default input must not be null when output file is not provided");
		}

		stream = default_output;
	}
	else
	{
		const auto& path = replace_string_variables(out, variables);
		auto file = new std::ofstream(out, mode);

		if (!file->is_open())
		{
			delete file;
			throw std::ofstream::failure("could not open output file " + path + " (" + out + ")");
		}
		stream.reset(file);
	}
}

void cgp::OutputStream::close()
{
	stream.reset();
}

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

void CGPOutputStream::log_human(size_t run, size_t generation)
{
	*this
		<< "[" << (run + 1)
		<< ", " << (generation + 1) << "] MSE: "
		<< error_to_string(cgp_model->get_best_error_fitness())
		<< ", Energy: "
		<< energy_to_string(cgp_model->get_best_energy_fitness())
		<< ", Delay: "
		<< delay_to_string(cgp_model->get_best_delay_fitness())
		<< ", Depth: "
		<< depth_to_string(cgp_model->get_best_depth())
		<< ", Gates: "
		<< gate_count_to_string(cgp_model->get_best_gate_count())
		<< std::endl;
}

void CGPOutputStream::log_csv(size_t run, size_t generation, const std::string& timestamp)
{
	*this
		<< (run + 1)
		<< "," << (generation + 1)
		<< ",\"" << timestamp << "\","
		<< error_to_string(cgp_model->get_best_error_fitness()) << ","
		<< energy_to_string(cgp_model->get_best_energy_fitness()) << ","
		<< delay_to_string(cgp_model->get_best_delay_fitness()) << ","
		<< depth_to_string(cgp_model->get_best_depth()) << ","
		<< gate_count_to_string(cgp_model->get_best_gate_count())
		<< ",\"" << cgp_model->get_best_chromosome()->to_string() << "\""
		<< std::endl;
}

void CGPOutputStream::log_csv(size_t run, size_t generation, std::shared_ptr<Chromosome> chromosome, const std::vector<std::shared_ptr<weight_value_t[]>>& inputs, const std::vector<std::shared_ptr<weight_value_t[]>>& outputs)
{
	auto solution = cgp_model->evaluate(inputs, outputs, chromosome);
	*this
		<< (run + 1)
		<< "," << (generation + 1)
		<< ",,"
		<< error_to_string(CGP::get_error(solution)) << ","
		<< energy_to_string(CGP::get_energy(solution)) << ","
		<< delay_to_string(CGP::get_delay(solution)) << ","
		<< depth_to_string(CGP::get_depth(solution)) << ","
		<< gate_count_to_string(CGP::get_gate_count(solution))
		<< ",\"" << chromosome->to_string() << "\""
		<< std::endl;
}

void CGPOutputStream::log_weights(const std::vector<std::shared_ptr<weight_value_t[]>>& inputs)
{
	log_weights(cgp_model->get_best_chromosome(), inputs);
}

void CGPOutputStream::log_weights(std::shared_ptr<Chromosome> chromosome, const std::vector<std::shared_ptr<weight_value_t[]>>& inputs)
{
	for (const auto& in : inputs) {
		auto weights = chromosome->get_weights(in);

		for (size_t i = 0; i < cgp_model->output_count() - 1; i++)
		{
			if (weights[i] == CGPConfiguration::invalid_value)
			{
				*this << "nan";
			}
			else
			{
				*this << weight_to_string(weights[i]);
			}
			*this << " ";
		}

		if (weights[cgp_model->output_count() - 1] == CGPConfiguration::invalid_value)
		{
			*this << "nan";
		}
		else
		{
			*this << weight_to_string(weights[cgp_model->output_count() - 1]);
		}
		*this << std::endl;
	}
}

void cgp::CGPOutputStream::dump()
{
	cgp_model->dump(*stream);
}

void cgp::CGPOutputStream::dump_all()
{
	cgp_model->dump_all(*stream);
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in)
	: InputStream(in) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::ios_base::openmode mode)
	: InputStream(in, mode) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input)
	: InputStream(in, default_input) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode)
	: InputStream(in, default_input, mode) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, const std::unordered_map<std::string, std::string>& variables)
	: InputStream(in, variables) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables)
	: InputStream(in, mode, variables) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, const std::unordered_map<std::string, std::string>& variables)
	: InputStream(in, default_input, variables) {
	this->cgp_model = cgp_model;
}

CGPInputStream::CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables)
	: InputStream(in, default_input, mode, variables) {
	this->cgp_model = cgp_model;
}

std::shared_ptr<weight_value_t[]> cgp::CGPInputStream::load_input()
{
	weight_repr_value_t weight;
	std::string no_care;
	std::shared_ptr<weight_value_t[]> input = std::make_shared<weight_value_t[]>(cgp_model->input_count());
	for (size_t i = 0; i < cgp_model->input_count(); i++)
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
				// Penalize usage of unallowed value
				input[i] = CGPConfiguration::invalid_value;
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

std::shared_ptr<weight_value_t[]> cgp::CGPInputStream::load_output()
{
	weight_repr_value_t weight;
	std::string no_care;
	std::shared_ptr<weight_value_t[]> output = std::make_shared<weight_value_t[]>(cgp_model->output_count());
	for (size_t i = 0; i < cgp_model->output_count(); i++)
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
				// Penalize usage of unallowed value
				output[i] = CGPConfiguration::invalid_value;
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
	return output;
}

std::tuple<std::vector<std::shared_ptr<weight_value_t[]>>, std::vector<std::shared_ptr<weight_value_t[]>>> cgp::CGPInputStream::load_train_data()
{
	std::vector<std::shared_ptr<weight_value_t[]>> inputs(cgp_model->dataset_size()), outputs(cgp_model->dataset_size());
	
	for (size_t i = 0; i < cgp_model->dataset_size(); i++)
	{
		inputs.push_back(load_input());
		outputs.push_back(load_output());
	}

	return std::make_tuple(inputs, outputs);
}

std::vector<std::shared_ptr<weight_value_t[]>> cgp::get_dataset_input(const dataset_t& dataset)
{
	return std::get<0>(dataset);
}

std::vector<std::shared_ptr<weight_value_t[]>> cgp::get_dataset_output(const dataset_t& dataset)
{
	return std::get<1>(dataset);
}
