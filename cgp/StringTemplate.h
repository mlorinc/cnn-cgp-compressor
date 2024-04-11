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

    std::string replace_string_variables(const std::string& input, const std::unordered_map<std::string, std::string>& variables);
}
