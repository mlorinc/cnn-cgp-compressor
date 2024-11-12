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
// Model.h : Load PyTorch models in C++. Unused.

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