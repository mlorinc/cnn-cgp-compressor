from commands.train_model import get_model
from models.quantized_model import QuantizedBaseModel

def quantize_model(model_name, model_path, new_path):
    print(f"Quantizing model: {model_name} and saving as {new_path}")
    model = get_model(model_name, model_path)
    assert isinstance(model, QuantizedBaseModel)
    model.load_unquantized(model_path)
    model.quantize(new_path)
    model.save(new_path)
    return model
