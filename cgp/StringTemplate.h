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
// StringTemplate.h : Add support for string templating using maps.

#pragma once
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <vector>

namespace cgp
{
    class StringTemplateError : public std::invalid_argument {
    public:
        StringTemplateError(const std::vector<std::string>& missing_arguments);
        std::string get_message() const;
    private:
        std::string message;
    };

	/// <summary>
	/// Replace variables withing string surrounded by curly braces.
	/// </summary>
	/// <param name="input">String to perform substitution on.</param>
    /// <param name="variables">Substitution mapping.</param>
	/// <returns>Substituted string.</returns>	
    std::string replace_string_variables(const std::string& input, const std::unordered_map<std::string, std::string>& variables);
}
