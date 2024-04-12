from commands.train_model import get_model
from models.base_model import BaseModel

def quantize_model(model_name, model_path, new_path):
    print(f"Quantizing model: {model_name} and saving as {new_path}")
    model: BaseModel = get_model(model_name, model_path)
    model.load(model_path, quantized=False)
    model.quantize(new_path)
    model.save(new_path)
    return model
