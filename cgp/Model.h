#pragma once
#include <string>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>

namespace cgp {
	class Model
	{
	public:
		Model(const std::string& path);
		~Model();
		double evaluate();
		torch::jit::parameter_list parameters();
	private:
		std::string path;
		torch::jit::Module model;
	};
}