// Unused file. It did not work at all.

#pragma once
#include "Cgp.h"

namespace cgp
{
	struct LearningRates
	{
		double error, energy, delay, gate_count;
	};

	class Learning
	{
	public:
		Learning(double LR, size_t epoch_length, double mse_threshold, size_t period);
		~Learning();
		double get_lr() const;
		bool tick(const CGP::solution_t end_epoch_solution);
		bool finished_period() const;
		bool is_stagnated() const;
		bool is_periodicaly_stagnated(double LR) const;
		LearningRates get_average_learning_rates() const;
		LearningRates get_average_period_learning_rates() const;
	private:
		bool can_access_average = false;
		double lr;
		double mse_threshold;
		size_t iterations = 0;
		size_t epochs = 0, energy_epoch = 0;
		size_t epoch_length;
		size_t period;
		LearningRates current_lr;
		LearningRates lr_global_accumulator;
		LearningRates lr_period_accumulator;
		CGP::solution_t start_epoch_solution, end_epoch_solution;
		LearningRates get_learning_rates() const;
	};
}
