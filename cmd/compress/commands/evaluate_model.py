from commands.datastore import Datastore
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from models.adapters.model_adapter_factory import create_adapter
from typing import Optional

def get_model_adapter(model_name: str, model_path: Optional[str] = None) -> ModelAdapter:
    return BaseAdapter.load_base_model(model_name, model_path)


def evaluate_base_model(model_name: str, model_path: str, archive=False, **kwargs):
    print(f"evaluating model: {model_name}")
    datastore = Datastore()
    save_path = datastore.models(model_name) / f"{model_name}.state_dict.pth"
    model: ModelAdapter = create_adapter(model_name, model_path)
    print(f"archiving:", "yes" if archive else "no")
    if archive:
        if save_path.exists():
            raise FileExistsError(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"saved model to {save_path}")
        model.save(save_path)
        
    acc, loss = model.evaluate(max_batches=None, **kwargs)
    print(acc, loss)
    return acc, loss
