#pragma once
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace cgp
{
    class StringTemplateError : public std::invalid_argument {
    public:
        StringTemplateError(const std::vector<std::string>& missing_arguments);
        inline virtual const char* what() const override;
    private:
        std::string message;
    };

    std::string replace_string_variables(const std::string& input, const std::unordered_map<std::string, std::string>& variables);
}
