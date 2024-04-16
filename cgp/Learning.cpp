#include "Learning.h"

using namespace cgp;

Learning::Learning(double LR, size_t epoch_length, double mse_threshold, size_t period)
{
	lr = LR;
	this->epoch_length = epoch_length;
	current_lr = { 0 };
	lr_global_accumulator = { 0 };
	lr_period_accumulator = { 0 };
	this->mse_threshold = mse_threshold;
	this->period = period;
}

Learning::~Learning()
{
}

double cgp::Learning::get_lr() const
{
	return lr;
}

bool cgp::Learning::tick(const CGP::solution_t epoch_solution)
{
	if (iterations == 0)
	{
		start_epoch_solution = epoch_solution;

		iterations++;
		epochs++;

		if (epochs % period == 1)
		{
			lr_period_accumulator.error = 0;
			lr_period_accumulator.energy = 0;
		}

		if (CGP::get_error(epoch_solution) < mse_threshold)
		{
			energy_epoch++;
		}

		return false;
	}
	else if (iterations >= epoch_length)
	{
		end_epoch_solution = epoch_solution;
		iterations = 0;
		current_lr = get_learning_rates();
		lr_global_accumulator.error += current_lr.error;
		lr_global_accumulator.energy += current_lr.energy;
		lr_period_accumulator.error += current_lr.error;
		lr_period_accumulator.energy += current_lr.energy;

		//lr_accumulator.delay += current_lr.delay;
		//lr_accumulator.gate_count += current_lr.gate_count;
		return true;
	}
	else
	{
		iterations++;

		if (energy_epoch == 0 && CGP::get_error(epoch_solution) < mse_threshold)
		{
			energy_epoch++;
		}

		return false;
	}
}

bool Learning::finished_period() const
{
	return (iterations == 0) && (epochs % period == 0);
}

bool Learning::is_periodicaly_stagnated(double LR) const
{
	LearningRates rates = get_average_period_learning_rates();

	if (CGP::get_error(end_epoch_solution) > mse_threshold)
	{
		return rates.error < LR;
	}

	return rates.error < LR && rates.energy < LR;
}

bool Learning::is_stagnated() const
{
	LearningRates rates = get_average_learning_rates();

	if (CGP::get_error(end_epoch_solution) > mse_threshold)
	{
		return rates.error < lr;
	}

	return rates.error < lr && rates.energy < lr; // < lr && rates.delay < lr && rates.gate_count < lr;
}

LearningRates cgp::Learning::get_learning_rates() const
{
	LearningRates rates{};
	if (CGP::get_chromosome(start_epoch_solution) == CGP::get_chromosome(end_epoch_solution))
	{
		return rates;
	}

	const auto error_start = CGP::get_error(start_epoch_solution);
	const auto error_end = CGP::get_error(end_epoch_solution);
	//auto delay_start = CGP::get_quantized_delay(start_epoch_solution);
	//auto delay_end = CGP::get_quantized_delay(end_epoch_solution);
	//auto gate_count_start = CGP::get_gate_count(start_epoch_solution);
	//auto gate_count_end = CGP::get_gate_count(end_epoch_solution);

	const auto error_delta = error_start - error_end;

	rates.error = (error_end < error_start) ? (error_start - error_end) : (0);
	//rates.delay = (delay_end < delay_start) ? (delay_start - delay_end) : (0);
	//rates.gate_count = (gate_count_end < gate_count_start) ? (gate_count_start - gate_count_end) : (0);

	//rates.delay = (delay_start != 0) ? (rates.delay / static_cast<double>(delay_start)) : (0);
	//rates.gate_count = (gate_count_start != 0) ? (rates.gate_count / static_cast<double>(gate_count_start)) : (0);
	rates.error = (error_start != 0) ? (rates.error / static_cast<double>(error_start)) : (0);

	if (energy_epoch > 0)
	{
		const auto energy_start = CGP::get_quantized_energy(start_epoch_solution);
		const auto energy_end = CGP::get_quantized_energy(end_epoch_solution);
		const auto energy_delta = energy_start - energy_end;
		rates.energy = (energy_end < energy_start && 1e-6 < energy_delta) ? (energy_start - energy_end) : (0);
		rates.energy = (energy_start != 0) ? (rates.energy / static_cast<double>(energy_start)) : (0);
	}

	return rates;
}

LearningRates cgp::Learning::get_average_learning_rates() const
{
	LearningRates rates{}, period_rates{};
	rates.error = lr_global_accumulator.error / epochs;
	if (energy_epoch > 0)
	{
		rates.energy = lr_global_accumulator.energy / energy_epoch;
	}

	//rates.delay = lr_accumulator.delay / epochs;
	//rates.gate_count = lr_accumulator.gate_count / epochs;
	return rates;
}

LearningRates cgp::Learning::get_average_period_learning_rates() const
{
	LearningRates rates{};
	rates.error = lr_period_accumulator.error / period;
	rates.energy = lr_period_accumulator.energy / period;
	return rates;
}
