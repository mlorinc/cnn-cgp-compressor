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
	using dataset_t = std::tuple<std::vector<std::shared_ptr<weight_value_t[]>>, std::vector<std::shared_ptr<weight_value_t[]>>>;

	std::vector<std::shared_ptr<weight_value_t[]>> get_dataset_input(const dataset_t& dataset);
	std::vector<std::shared_ptr<weight_value_t[]>> get_dataset_output(const dataset_t& dataset);

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

		// Overload the write operator
		template <typename T>
		friend OutputStream& operator<<(OutputStream& os, const T& value)
		{
			if (os.stream)
			{
				(*os.stream) << value;
			}
			return os;
		}

		// Overload for manipulators that don't require std::endl
		friend OutputStream& operator<<(OutputStream& os, std::ostream& (*manip)(std::ostream&))
		{
			if (os.stream)
			{
				(*manip)(*os.stream);
			}
			return os;
		}
	};

	class CGPStream
	{
	protected:
		std::shared_ptr<CGP> cgp_model;
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
		void log_human(size_t run, size_t generation, const CGP::solution_t &solution);

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
		void log_csv(size_t run, size_t generation, const std::string& timestamp, const CGP::solution_t &solution, bool show_chromosome = false);

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
			const std::vector<std::shared_ptr<weight_value_t[]>>& inputs,
			const std::vector<std::shared_ptr<weight_value_t[]>>& outputs,
			bool show_chromosome = false);

		/// <summary>
		/// Logs weight information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="stream">The output stream to log to.</param>
		/// <param name="inputs">The input values used for evaluation.</param>
		void log_weights(const std::vector<std::shared_ptr<weight_value_t[]>>& inputs);

		/// <summary>
		/// Logs weight information about the CGP model to the specified stream.
		/// </summary>
		/// <param name="stream">The output stream to log to.</param>
		/// <param name="inputs">The input values used for evaluation.</param>
		/// <param name="chromosome">The chromosome to log weights.</param>
		void log_weights(std::shared_ptr<Chromosome> chromosome, const std::vector<std::shared_ptr<weight_value_t[]>>& inputs);

		/// <summary>
		/// Dumps the current state of the CGP model to the output stream.
		/// </summary>
		void dump();

		/// <summary>
		/// Dumps all information related to the CGP model, including internal state and configurations, to the output stream.
		/// </summary>
		void dump_all();
	};

	class CGPInputStream : public InputStream, public CGPStream
	{
	public:
		CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in);
		CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::ios_base::openmode mode);
		CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input);
		CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode);
		CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, const std::unordered_map<std::string, std::string>& variables);
		CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables);
		CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, const std::unordered_map<std::string, std::string>& variables);
		CGPInputStream(std::shared_ptr<CGP> cgp_model, const std::string& in, std::shared_ptr<std::istream> default_input, std::ios_base::openmode mode, const std::unordered_map<std::string, std::string>& variables);

		std::shared_ptr<weight_value_t[]> load_input();
		std::shared_ptr<weight_value_t[]> load_output();
		std::tuple<std::vector<std::shared_ptr<weight_value_t[]>>, std::vector<std::shared_ptr<weight_value_t[]>>> load_train_data();
		std::shared_ptr<CGPConfiguration::gate_parameters_t[]> load_gate_parameters();
	};
}