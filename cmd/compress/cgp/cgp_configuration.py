class CGPConfiguration:
    def __init__(self, config_file: str = None):
        self._attributes = {}
        self._config_file = config_file
        if config_file:
            self.load(config_file)

    def clone(self, new_config_file: str = None):
        cloned_instance = CGPConfiguration()
        cloned_instance._config_file = new_config_file or self._config_file
        cloned_instance._attributes = self._attributes.copy()
        return cloned_instance

    def load(self, config_file: str = None):
        if config_file is None and self._config_file is None:
            raise ValueError(
                "either config file must be passed to the load function or the class constructor must have been provided a configuration file as argument"
            )
        with open(config_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines
                if line:
                    colon_index = line.index(":")
                    key, value = line[:colon_index], line[colon_index+1:]
                    self._attributes[key.strip()] = self._parse_value(value.strip())

    def save(self, config_file: str = None):
        if config_file is None and self._config_file is None:
            raise ValueError(
                "either config file must be passed to the save function or the class constructor must have been provided a configuration file as argument"
            )

        with open(config_file or self._config_file, "w") as f:
            for key, value in self._attributes.items():
                f.write(f"{key}: {value}\n")

    def __contains__(self, key):
        return key in self._attributes

    def _parse_value(self, value_str: str):
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str

    def get_mse_threshold(self):
        return self._attributes.get("mse_threshold")

    def get_dataset_size(self):
        return self._attributes.get("dataset_size")

    def get_col_count(self):
        return self._attributes.get("col_count")

    def get_row_count(self):
        return self._attributes.get("row_count")

    def get_number_of_runs(self):
        return self._attributes.get("number_of_runs")

    def get_look_back_parameter(self):
        return self._attributes.get("look_back_parameter")

    def get_mutation_max(self):
        return self._attributes.get("mutation_max")

    def get_function_count(self):
        return self._attributes.get("function_count")

    def get_function_input_arity(self):
        return self._attributes.get("function_input_arity")

    def get_function_output_arity(self):
        return self._attributes.get("function_output_arity")

    def get_input_count(self):
        return self._attributes.get("input_count")

    def get_output_count(self):
        return self._attributes.get("output_count")

    def get_population_max(self):
        return self._attributes.get("population_max")

    def get_generation_count(self):
        return self._attributes.get("generation_count")

    def get_periodic_log_frequency(self):
        return self._attributes.get("periodic_log_frequency")

    def get_chromosome_output_file(self):
        return self._attributes.get("chromosome_output_file")

    def get_input_file(self):
        return self._attributes.get("input_file")

    def get_output_file(self):
        return self._attributes.get("output_file")

    def get_cgp_statistics_file(self):
        return self._attributes.get("cgp_statistics_file")

    def set_mse_threshold(self, value):
        self._attributes["mse_threshold"] = value

    def set_dataset_size(self, value):
        self._attributes["dataset_size"] = value

    def set_col_count(self, value):
        self._attributes["col_count"] = value

    def set_row_count(self, value):
        self._attributes["row_count"] = value

    def set_number_of_runs(self, value):
        self._attributes["number_of_runs"] = value

    def set_look_back_parameter(self, value):
        self._attributes["look_back_parameter"] = value

    def set_mutation_max(self, value):
        self._attributes["mutation_max"] = value

    def set_function_count(self, value):
        self._attributes["function_count"] = value

    def set_function_input_arity(self, value):
        self._attributes["function_input_arity"] = value

    def set_function_output_arity(self, value):
        self._attributes["function_output_arity"] = value

    def set_input_count(self, value):
        self._attributes["input_count"] = value

    def set_output_count(self, value):
        self._attributes["output_count"] = value

    def set_population_max(self, value):
        self._attributes["population_max"] = value

    def set_generation_count(self, value):
        self._attributes["generation_count"] = value

    def set_periodic_log_frequency(self, value):
        self._attributes["periodic_log_frequency"] = value

    def set_chromosome_output_file(self, value):
        self._attributes["chromosome_output_file"] = value

    def set_input_file(self, value):
        self._attributes["input_file"] = value

    def set_output_file(self, value):
        self._attributes["output_file"] = value

    def set_cgp_statistics_file(self, value):
        self._attributes["cgp_statistics_file"] = value

    def delete_mse_threshold(self):
        del self._attributes["mse_threshold"]

    def delete_dataset_size(self):
        del self._attributes["dataset_size"]

    def delete_col_count(self):
        del self._attributes["col_count"]

    def delete_row_count(self):
        del self._attributes["row_count"]

    def delete_number_of_runs(self):
        del self._attributes["number_of_runs"]

    def delete_look_back_parameter(self):
        del self._attributes["look_back_parameter"]

    def delete_mutation_max(self):
        del self._attributes["mutation_max"]

    def delete_function_count(self):
        del self._attributes["function_count"]

    def delete_function_input_arity(self):
        del self._attributes["function_input_arity"]

    def delete_function_output_arity(self):
        del self._attributes["function_output_arity"]

    def delete_input_count(self):
        del self._attributes["input_count"]

    def delete_output_count(self):
        del self._attributes["output_count"]

    def delete_population_max(self):
        del self._attributes["population_max"]

    def delete_generation_count(self):
        del self._attributes["generation_count"]

    def delete_periodic_log_frequency(self):
        del self._attributes["periodic_log_frequency"]

    def delete_chromosome_output_file(self):
        del self._attributes["chromosome_output_file"]

    def delete_input_file(self):
        del self._attributes["input_file"]

    def delete_output_file(self):
        del self._attributes["output_file"]

    def delete_cgp_statistics_file(self):
        del self._attributes["cgp_statistics_file"]

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in self._attributes.items()]
        return "\n".join(attributes)

    def has_mse_threshold(self):
        return "mse_threshold" in self._attributes

    def has_dataset_size(self):
        return "dataset_size" in self._attributes

    def has_col_count(self):
        return "col_count" in self._attributes

    def has_row_count(self):
        return "row_count" in self._attributes

    def has_number_of_runs(self):
        return "number_of_runs" in self._attributes

    def has_look_back_parameter(self):
        return "look_back_parameter" in self._attributes

    def has_mutation_max(self):
        return "mutation_max" in self._attributes

    def has_function_count(self):
        return "function_count" in self._attributes

    def has_function_input_arity(self):
        return "function_input_arity" in self._attributes

    def has_function_output_arity(self):
        return "function_output_arity" in self._attributes

    def has_input_count(self):
        return "input_count" in self._attributes

    def has_output_count(self):
        return "output_count" in self._attributes

    def has_population_max(self):
        return "population_max" in self._attributes

    def has_generation_count(self):
        return "generation_count" in self._attributes

    def has_periodic_log_frequency(self):
        return "periodic_log_frequency" in self._attributes

    def has_chromosome_output_file(self):
        return "chromosome_output_file" in self._attributes

    def has_input_file(self):
        return "input_file" in self._attributes

    def has_output_file(self):
        return "output_file" in self._attributes

    def has_cgp_statistics_file(self):
        return "cgp_statistics_file" in self._attributes   

    def to_args(self):
        arguments = []
        for k, v in self._attributes.items():
            if k in ["dataset_size", "periodic_log_frequency"]:
                continue

            key = k.replace("_", "-")
            arguments.append(f"--{key}")
            arguments.append(str(v))
        return arguments
