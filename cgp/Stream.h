// Copyright 2024 Marián Lorinc
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     LICENSE.txt file
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Stream.h : Data stream utility class definitions for more straightforward logging capabilities and input parsing.

#pragma once

#include <fstream>
#include <ios>
#include <string>
#include <vector>
#include <unordered_map>
#include "Cgp.h"

namespace cgp
{
	using weight_value_t = CGPConfiguration::weight_value_t;
	using weight_repr_value_t = CGPConfiguration::weight_repr_value_t;

	class InputStream
	{
	protected:
		std::shared_ptr<std::istream> stream;
	public:
		InputStream(const std::string& in);
		InputStream(const std::string& in, std::ios_base::openmode mode);
		InputStream(const std::string& in, std::shared_ptr<std::istream> default_input);
		InputStream(const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode);
		InputStream(const std::string& in, const std::unordered_map<std::string, std::string> &variables);
		InputStream(const std::string& in, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string> &variables);
		InputStream(const std::string& in, std::shared_ptr<std::istream> default_input, const std::unordered_map<std::string, std::string> &variables);
		InputStream(const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string> &variables);
		
		void close();
		bool eof() const;
		bool fail() const;
		std::istream& get_stream();

		template <typename T>
		friend InputStream& operator>>(InputStream& is, T& value)
		{
			if (is.stream)
			{
				(*is.stream) >> value;
			}
			return is;
		}
	};

	class OutputStream
	{
	protected:
		std::shared_ptr<std::ostream> stream;
	public:
		OutputStream(const std::string& out);
		OutputStream(const std::string& out, std::ios_base::openmode mode);
		OutputStream(const std::string& out, std::shared_ptr<std::ostream> default_output);
		OutputStream(const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode);
		OutputStream(const std::string& out, const std::unordered_map<std::string, std::string> &variables);
		OutputStream(const std::string& out, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string> &variables);
		OutputStream(const std::string& out, std::shared_ptr<std::ostream> default_output, const std::unordered_map<std::string, std::string> &variables);
		OutputStream(const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string> &variables);

		void close();
		bool is_ignoring_output() const;

		// Overload the write operator
		template <typename T>
		friend OutputStream& operator<<(OutputStream& os, const T& value)
		{
			if (!os.is_ignoring_output())
			{
				(*os.stream) << value;
			}
			return os;
		}

		// Overload for manipulators that don't require std::endl
		friend OutputStream& operator<<(OutputStream& os, std::ostream& (*manip)(std::ostream&))
		{
			if (!os.is_ignoring_output())
			{
				(*manip)(*os.stream);
			}
			return os;
		}
	};
}