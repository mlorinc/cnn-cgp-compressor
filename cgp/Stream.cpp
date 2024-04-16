#include "Stream.h"
#include "StringTemplate.h"
#include <iostream>
#include <cmath>

using namespace cgp;

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

bool cgp::InputStream::eof() const
{
	return stream->eof();
}

bool cgp::InputStream::fail() const
{
	return stream->fail();
}

std::istream& cgp::InputStream::get_stream()
{
	return *stream;
}

bool cgp::OutputStream::is_ignoring_output() const
{
	return stream == nullptr;
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
	else if (out[0] == '#')
	{
		std::cerr << "warning: ignoring output sent pointed by parameter " + out.substr(1) << std::endl;
		stream = nullptr;
	}
	else if (out.empty())
	{
		if (default_output == nullptr)
		{
			throw std::invalid_argument("default output must not be null when output file is not provided");
		}

		stream = default_output;
	}
	else
	{
		const auto& path = replace_string_variables(out, variables);
		auto file = new std::ofstream(path, mode);

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
