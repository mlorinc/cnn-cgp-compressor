#include "Stream.h"
#include <iostream>

std::shared_ptr<std::istream> cgp::get_input(const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode) {
	std::shared_ptr<std::istream> stream;
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
		auto file = new std::ifstream(in, mode);
		std::cerr << "attempting to open: " << in << std::endl;
		if (!file->is_open())
		{
			delete file;
			throw std::ofstream::failure("could not open input file " + in);
		}
		std::cerr << "opened: " << in << std::endl;
		stream.reset(file);
	}

	return stream;
}

std::shared_ptr<std::ostream> cgp::get_output(const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode) {
	std::shared_ptr<std::ostream> stream;
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
		std::cerr << "attempting to open: " << out << std::endl;
		auto file = new std::ofstream(out, mode);

		if (!file->is_open())
		{
			delete file;
			throw std::ofstream::failure("could not open output file " + out);
		}
		std::cerr << "opened: " << out << std::endl;
		stream.reset(file);
	}

	return stream;
}

void cgp::log_human(std::ostream& stream, size_t run, size_t generation, cgp::CGP& cgp_model)
{
	stream << "[" << (run + 1) << ", " << (generation + 1) << "] MSE: " << cgp_model.get_best_error_fitness() << ", Energy: " << cgp_model.get_best_energy_fitness() << std::endl;
}

void cgp::log_csv(std::ostream& stream, size_t run, size_t generation, cgp::CGP& cgp_model, const std::string &timestmap)
{
	//",\"" << *cgp_model.get_best_chromosome() << "\""
	stream << (run + 1) << "," << (generation + 1) << ",\"" << timestmap << "\"," << cgp_model.get_best_error_fitness() << "," << cgp_model.get_best_energy_fitness() << ",\"" << cgp_model.get_best_chromosome()->to_string() << "\"" << std::endl;
}

void cgp::log_weights(std::ostream& stream, const std::vector<std::shared_ptr<weight_value_t[]>>& inputs, cgp::CGP& cgp_model)
{
	for (const auto& in : inputs) {
		auto weights = cgp_model.get_best_chromosome()->get_weights(in);

		for (size_t i = 0; i < cgp_model.output_count() - 1; i++)
		{
			if(weights[i] == CGPConfiguration::invalid_value)
			{
				stream << "nan";
			}
			else
			{
				stream << static_cast<weight_repr_value_t>(weights[i]);
			}
			stream << " ";
		}

		if (weights[cgp_model.output_count() - 1] == CGPConfiguration::invalid_value)
		{
			stream << "nan";
		}
		else
		{
			stream << static_cast<weight_repr_value_t>(weights[cgp_model.output_count() - 1]);
		}
		stream << std::endl;
	}
}

std::shared_ptr<cgp::weight_value_t[]> cgp::load_input(std::istream& in, size_t input_size)
{
	weight_repr_value_t weight;
	std::string no_care;
	std::shared_ptr<weight_value_t[]> input = std::make_shared<weight_value_t[]>(input_size);
	
	for (size_t i = 0; i < input_size; i++)
	{
		if (in >> weight)
		{
			input[i] = static_cast<weight_value_t>(weight);
			continue;
		}
		else if (in.eof())
		{
			throw std::invalid_argument("not enough value values passed; got: " + std::to_string(i) + ", need: " + std::to_string(input_size));
		}
		else
		{
			in.clear();
		}

		if (in >> no_care)
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
	std::copy(input.get(), input.get() + input_size, std::ostream_iterator<weight_repr_value_t>(std::cerr, " "));
	std::cerr << std::endl;
	return input;
}

std::shared_ptr<cgp::weight_value_t[]> cgp::load_output(std::istream& in, size_t output_size, weight_repr_value_t& min, weight_repr_value_t& max)
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
