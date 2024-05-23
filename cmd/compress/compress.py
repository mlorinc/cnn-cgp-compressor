import argparse
import commands.manager as manager
from commands.evaluate_cgp_model import evaluate_model
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Model Training, Evaluation, Quantization, and Optimization")
    manager.register_commands(parser)
    args = parser.parse_args()
    command = manager.dispatch(args)
    try:
        command()
    except Exception as e:
        raise e
        print(e)
        exit(42)
    
if __name__ == "__main__":
    main()
    # evaluate_model(root=r"C:\Users\Majo\source\repos\TorchCompresser\local_experiments\worst_case\mse_0_256_31", run=1)
