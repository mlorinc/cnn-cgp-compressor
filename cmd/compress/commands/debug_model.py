from models.adapters.model_adapter import ModelAdapter
from models.adapters.model_adapter_factory import create_adapter

def debug_model(model_name: str, model_path: str):
    print(f"Evaluating model: {model_name}")
    model: ModelAdapter = create_adapter(model_name, model_path)
    return
