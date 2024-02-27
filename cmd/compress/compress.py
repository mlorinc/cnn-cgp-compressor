from typing import Optional
import torch
import torchvision
from models.ptq_quantized_lenet import PTQQuantizedLeNet5
from models.qat_quantized_lenet import QATQuantizedLeNet5
from models.mobilenet_v2 import MobileNetV2
from models.lenet import LeNet5
from models.base import BaseModel
from cgp.cgp_adapter import CGP
from cgp.test_cgp_adapter import TestCGP
import importlib
import argparse

def dequantize_per_channel(x: torch.Tensor, conv_layer: torch.Tensor):
    zero_point = conv_layer.q_per_channel_zero_points()
    scale = conv_layer.q_per_channel_scales()

    dequantized = ((x - zero_point.view(-1, 1, 1)) * scale.view(-1, 1, 1)).float()
    return torch.quantize_per_channel(
        dequantized,
        scale,
        zero_point,
        axis=0,
        dtype=torch.qint8
    )

def dequantize_per_tensor(x: torch.Tensor, scale: torch.float32, zero_point: torch.float32):
    dequantized = ((x - zero_point) * scale).float()
    return torch.quantize_per_tensor(
        dequantized,
        scale,
        zero_point,
        dtype=torch.qint8
    )

def _get_model(model_name: str, model_path: Optional[str] = None) -> BaseModel:
    return importlib.import_module(f"models.{model_name}").init(model_path)

def train_model(model_name: str, model_path: str, base: str):
    print(f"Training model: {model_name}")
    model: BaseModel = _get_model(model_name, model_path)
    if base:
        model.load(base)
        print(f"Loaded from {base}")

    model.fit()
    model.save(model_path)

def evaluate_model(model_name: str, model_path: str):
    print(f"Evaluating model: {model_name}")
    model: BaseModel = _get_model(model_name, model_path)
    if model_name != MobileNetV2.name:
        model = model.load(model_path)
    acc, loss = model.evaluate(max_batches=None)
        
    print(f"acc: {acc:.12f}%, loss {loss:.12f}")

def quantize_model(model_name, model_path, new_path):
    print(f"Quantizing model: {model_name} and saving as {new_path}")
    model: BaseModel = _get_model(model_name, model_path)
    model.load(model_path, quantized=False)
    model.quantize(new_path)
    model.save(new_path)

def optimize_model(model_name: str, model_path: str, cgp_binary_path: str):
    initial_acc = final_acc = 0
    initial_loss = final_loss = 0

    if model_name == LeNet5.name:
        cgp = CGP(cgp_binary_path, 16+1, 5, 3)
        model = LeNet5(model_path)
        model = model.load(model_path)
        model.eval()
        with torch.no_grad():
            initial_acc, initial_loss = model.evaluate()
            print("before compression:")
            print(f"acc: {initial_acc:.12f}, loss {initial_loss:.12f}")
            cgp.add_kernel(model.conv1.weight[0, 0])
            for kernel in model.conv2.weight[:, 0]:
                cgp.add_kernel(kernel)
            cgp.train()

            kernels = cgp.get_kernels()
            model.conv1.weight[0, 0] = kernels[0]
            for i in range(16):
                model.conv2.weight[i, 0] = kernels[i+1]
            final_acc, final_loss = model.evaluate()
    elif model_name == QATQuantizedLeNet5.name:
        # cgp = TestCGP()
        cgp = CGP(cgp_binary_path, 6, 5, 3)
        model = QATQuantizedLeNet5(model_path)
        model = model.load(model_path)
        model.eval()

        with torch.no_grad():
            initial_acc, initial_loss = model.evaluate()
            conv1_biases = model.conv1.bias()
            conv2_biases = model.conv2.bias()
            conv1_fp32_weights = model.conv1.weight().detach()
            conv2_fp32_weights = model.conv2.weight().detach()
            conv1_qint8_weights = model.conv1.weight().detach().int_repr()
            conv2_qint8_weights = model.conv2.weight().detach().int_repr()

            print("before compression:")
            print(f"acc: {initial_acc:.12f}, loss {initial_loss:.12f}")
            
            for kernel in conv1_qint8_weights[:, 0]:
                cgp.add_kernel(kernel)

            cgp.train()
            kernels = cgp.get_kernels()
            conv1_fp32_weights[:, 0] = dequantize_per_channel(torch.stack(kernels), conv1_fp32_weights)
            model.conv1.set_weight_bias(conv1_fp32_weights, conv1_biases)
            final_acc, final_loss = model.evaluate()
    print(f"acc: {initial_acc:.12f}, loss {initial_loss:.12f} | acc: {final_acc:.12f}, loss {final_loss:.12f}")


def main():
    parser = argparse.ArgumentParser(description="Model Training, Evaluation, Quantization, and Optimization")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # model:train
    train_parser = subparsers.add_parser("model:train", help="Train a model")
    train_parser.add_argument("model_name", help="Name of the model to train")
    train_parser.add_argument("model_path", help="Path where trained model will be saved")
    train_parser.add_argument("-b", "--base", type=str, help="Path to the baseline model")

    # model:evaluate
    evaluate_parser = subparsers.add_parser("model:evaluate", help="Evaluate a model")
    evaluate_parser.add_argument("model_name", help="Name of the model to evaluate")
    evaluate_parser.add_argument("model_path", help="Path to the model to evaluate")

    # model:quantize
    quantize_parser = subparsers.add_parser("model:quantize", help="Quantize a model")
    quantize_parser.add_argument("model_name", help="Name of the model to quantize")
    quantize_parser.add_argument("model_path", help="Path where trained model is saved")
    quantize_parser.add_argument("new_path", help="Path of the new quantized model where it will be stored")

    # cgp:optimize
    optimize_parser = subparsers.add_parser("cgp:optimize", help="Optimize a model")
    optimize_parser.add_argument("model_name", help="Name of the model to optimize")
    optimize_parser.add_argument("model_path", help="Path to the model to optimize")
    optimize_parser.add_argument("cgp_binary_path", help="Path to the CGP binary")

    args = parser.parse_args()

    if args.command == "model:train":
        train_model(args.model_name, args.model_path, args.base)
    elif args.command == "model:evaluate":
        evaluate_model(args.model_name, args.model_path)
    elif args.command == "model:quantize":
        quantize_model(args.model_name, args.model_path, args.new_path)
    elif args.command == "cgp:optimize":
        optimize_model(args.model_name, args.model_path, args.cgp_binary_path)
    else:
        print("Invalid command. Use --help for usage information.")

if __name__ == "__main__":
    main()
