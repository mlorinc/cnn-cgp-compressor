#include "Configuration.h"

namespace cgp {

    decltype(CGPConfiguration::function_input_arity_value) CGPConfiguration::function_input_arity() const {
        return function_input_arity_value;
    }

    uint16_t CGPConfiguration::pin_map_size() const {
        return static_cast<uint16_t>(row_count()) * col_count() * function_output_arity() + output_count() + input_count();
    }

    size_t CGPConfiguration::blocks_chromosome_size() const {
        return row_count() * col_count() * (function_input_arity() + 1);
    }
    
    size_t CGPConfiguration::chromosome_size() const {
        return blocks_chromosome_size() + output_count();
    }

    decltype(CGPConfiguration::function_output_arity_value) CGPConfiguration::function_output_arity() const {
        return function_output_arity_value;
    }


    decltype(CGPConfiguration::output_count_val) CGPConfiguration::output_count() const {
        return output_count_val;
    }


    decltype(CGPConfiguration::output_count_val) CGPConfiguration::input_count() const {
        return input_count_val;
    }


    decltype(CGPConfiguration::population_max_value) CGPConfiguration::population_max() const {
        return population_max_value;
    }


    decltype(CGPConfiguration::mutation_max_value) CGPConfiguration::mutation_max() const {
        return mutation_max_value;
    }


    decltype(CGPConfiguration::row_count_value) CGPConfiguration::row_count() const {
        return row_count_value;
    }


    decltype(CGPConfiguration::col_count_value) CGPConfiguration::col_count() const {
        return col_count_value;
    }


    decltype(CGPConfiguration::look_back_parameter_value) CGPConfiguration::look_back_parameter() const {
        return look_back_parameter_value;
    }


    decltype(CGPConfiguration::generation_count_value) CGPConfiguration::generation_count() const {
        return generation_count_value;
    }


    decltype(CGPConfiguration::number_of_runs_value) CGPConfiguration::number_of_runs() const {
        return number_of_runs_value;
    }


    decltype(CGPConfiguration::function_count_value) CGPConfiguration::function_count() const {
        return function_count_value;
    }


    decltype(CGPConfiguration::periodic_log_frequency_value) CGPConfiguration::periodic_log_frequency() const {
        return periodic_log_frequency_value;
    }


    decltype(CGPConfiguration::periodic_log_value) CGPConfiguration::periodic_log() const {
        return periodic_log_value;
    }


    CGPConfiguration& CGPConfiguration::function_input_arity(decltype(function_input_arity_value) value) {
        function_input_arity_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::function_output_arity(decltype(function_output_arity_value) value) {
        function_output_arity_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::output_count(decltype(output_count_val) value) {
        output_count_val = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::input_count(decltype(input_count_val) value) {
        input_count_val = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::population_max(decltype(population_max_value) value) {
        population_max_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::mutation_max(decltype(mutation_max_value) value) {
        mutation_max_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::row_count(decltype(row_count_value) value) {
        row_count_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::col_count(decltype(col_count_value) value) {
        col_count_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::look_back_parameter(decltype(look_back_parameter_value) value) {
        look_back_parameter_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::generation_count(decltype(generation_count_value) value) {
        generation_count_value = value;
        periodic_log_frequency(static_cast<uint32_t>(generation_count_value / 2.0));
        return *this;
    }


    CGPConfiguration& CGPConfiguration::number_of_runs(decltype(number_of_runs_value) value) {
        number_of_runs_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::function_count(decltype(function_count_value) value) {
        function_count_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::periodic_log_frequency(decltype(periodic_log_frequency_value) value) {
        periodic_log_frequency_value = value;
        return *this;
    }


    CGPConfiguration& CGPConfiguration::periodic_log(decltype(periodic_log_value) value) {
        periodic_log_value = value;
        return *this;
    }
}
