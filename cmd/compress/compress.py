import argparse
import commands.manager as manager
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
