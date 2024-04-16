#pragma once
#include "Stream.h"

namespace cgp
{
	class CGPStream
	{
	protected:
		std::shared_ptr<CGP> cgp_model;
	};

	class CGPConfigurationStream
	{
	protected:
		std::shared_ptr<CGPConfiguration> cgp_model;
	};

	class CGPOutputStream : public OutputStream, public CGPStream
	{
	public:
		CGPOutputStream(std::shared_ptr<CGP> cgp, const std::string& out);
		CGPOutputStream(std::shared_ptr<CGP> cgp, const std::string& out, std::ios_base::openmode mode);
		CGPOutputStream(std::shared_ptr<CGP> cgp, const std::string& out, std::shared_ptr<std::ostream> default_output);
		CGPOutputStream(std::shared_ptr<CGP> cgp, const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode);
		CGPOutputStream(std::shared_ptr<CGP> cgp, const std::string& out, const std::unordered_map<std::string, std::string>& variables);
		CGPOutputStream(std::shared_ptr<CGP> cgp, const std::string& out, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables);
		CGPOutputStream(std::shared_ptr<CGP> cgp, const std::string& out, std::shared_ptr<std::ostream> default_output, const std::unordered_map<std::string, std::string>& variables);
		CGPOutputStream(std::shared_ptr<CGP> cgp, const std::string& out, std::shared_ptr<std::ostream> default_output, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables);

		/// <summary>
		/// Logs human-readable information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="run">The current run number.</param>
		/// <param name="generation">The current generation number.</param>
		void log_human(size_t run, size_t generation);

		/// <summary>
		/// Logs human-readable information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="run">The current run number.</param>
		/// <param name="generation">The current generation number.</param>
		/// <param name="solution">Solution to log.</param>
		void log_human(size_t run, size_t generation, const CGP::solution_t& solution);

		void log_csv_header();

		/// <summary>
		/// Logs CSV-formatted information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="run">The current run number.</param>
		/// <param name="generation">The current generation number.</param>
		/// <param name="timestamp">The timestamp to include in the log.</param>
		/// <param name="show_chromosome">Flag indicating whether chromosome will be part of the csv file.</param>
		void log_csv(size_t run, size_t generation, const std::string& timestamp, bool show_chromosome = false);

		/// <summary>
		/// Logs CSV-formatted information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="run">The current run number.</param>
		/// <param name="generation">The current generation number.</param>
		/// <param name="timestamp">The timestamp to include in the log.</param>
		/// <param name="solution">Solution to log.</param>
		/// <param name="show_chromosome">Flag indicating whether chromosome will be part of the csv file.</param>
		void log_csv(size_t run, size_t generation, const std::string& timestamp, const CGP::solution_t& solution, bool show_chromosome = false);

		/// <summary>
		/// Logs CSV-formatted information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="run">The current run number.</param>
		/// <param name="generation">The current generation number.</param>
		/// <param name="timestamp">The timestamp to include in the log.</param>
		/// <param name="chromosome">The chromosome to log information about.</param>
		/// <param name="chromosome">The chromosome to log information about.</param>
		void log_csv(
			size_t run,
			size_t generation,
			std::shared_ptr<Chromosome> chromosome,
			const dataset_t& dataset,
			bool show_chromosome = false);

		/// <summary>
		/// Logs weight information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="stream">The output stream to log to.</param>
		/// <param name="inputs">The input values used for evaluation.</param>
		void log_weights(const std::vector<weight_input_t>& inputs);

		/// <summary>
		/// Logs weight information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="stream">The output stream to log to.</param>
		/// <param name="inputs">The input values used for evaluation.</param>
		/// <param name="chromosome">The chromosome to log weights.</param>
		void log_weights(std::shared_ptr<Chromosome> chromosome, const std::vector<weight_input_t>& inputs);

		/// <summary>
		/// Dumps the current state of the CGP model to the output stream.
		/// </summary>
		void dump();

		/// <summary>
		/// Dumps all information related to the CGP model, including internal state and configurations, to the output stream.
		/// </summary>
		void dump_all();
	};

	struct CGPCSVRow
	{
		bool ok;
		size_t run;
		size_t generation;
		std::string timestamp;
		std::string chromosome;
		std::string raw_line;
	};

	class CGPInputStream : public InputStream, public CGPConfigurationStream
	{
	public:
		CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in);
		CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::ios_base::openmode mode);
		CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input);
		CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode);
		CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, const std::unordered_map<std::string, std::string>& variables);
		CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables);
		CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, const std::unordered_map<std::string, std::string>& variables);
		CGPInputStream(std::shared_ptr<CGPConfiguration> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables);

		CGPCSVRow read_csv_line() const;
		weight_input_t load_input();
		weight_output_t load_output();
		dataset_t load_train_data();
		std::unique_ptr<CGPConfiguration::gate_parameters_t[]> load_gate_parameters();
	};
}