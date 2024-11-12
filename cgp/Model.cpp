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
// Model.cpp : Load PyTorch models in C++. Unused.

#include "Model.h"

namespace cgp {

	Model::Model(const std::string& path) : path(path)
	{
		this->model = torch::jit::load(path);
	}

	Model::~Model()
	{
	}
	
	double Model::evaluate()
	{
		auto mnist_path = "";
		model.eval();
		auto dataset = torch::data::datasets::MNIST(mnist_path, torch::data::datasets::MNIST::Mode::kTest)
			.map(torch::data::transforms::Normalize<>(0.5, 0.5))
			.map(torch::data::transforms::Stack<>());

		auto data_loader = torch::data::make_data_loader(
			std::move(dataset),
			torch::data::DataLoaderOptions()
			.batch_size(32)
		);


		torch::nn::CrossEntropyLoss criterion;
		double running_loss = 0;
		int64_t total = 0;
		int64_t correct = 0;
		for (const auto& batch : *data_loader) {
			auto inputs = batch.data.to(torch::kFloat32);
			auto& labels = batch.target;

			// Forward pass
			auto outputs = model.forward({ inputs }).toTensor();

			// Compute loss
			auto loss = criterion(outputs, labels);
			running_loss += loss.item<double>();

			// Compute accuracy
			auto max_result = torch::max(outputs, 1);
			auto predicted = std::get<1>(max_result);
			total += labels.size(0);
			correct += predicted.eq(labels).sum().item<int64_t>();
		}

		// Compute accuracy and loss
		double acc = 100.0 * correct / total;
		double test_loss = running_loss / total;

		// Print the results
		std::cout << "Acc: " << acc << "%, Loss: " << test_loss << std::endl;
		return correct / total;
	}
	torch::jit::parameter_list Model::parameters()
	{
		return model.parameters();
	}
}