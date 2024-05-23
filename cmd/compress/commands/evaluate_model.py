from commands.datastore import Datastore
from models.adapters.model_adapter import ModelAdapter
from models.adapters.base import BaseAdapter
from models.adapters.model_adapter_factory import create_adapter
from typing import Optional

def get_model_adapter(model_name: str, model_path: Optional[str] = None) -> ModelAdapter:
    """
    Loads a model adapter for the specified model.

    Args:
        model_name (str): The name of the model to load.
        model_path (Optional[str]): The optional path to the model's state dictionary.

    Returns:
        ModelAdapter: An adapter for the specified model.
    """    
    return BaseAdapter.load_base_model(model_name, model_path)


def evaluate_base_model(model_name: str, model_path: str, archive=False, **kwargs):
    """
    Evaluates a base model and optionally archives its state dictionary.

    Args:
        model_name (str): The name of the model to evaluate.
        model_path (str): The path to the model's state dictionary.
        archive (bool, optional): If True, saves the model's state dictionary to the datastore. Defaults to False.
        **kwargs: Additional keyword arguments for the model's evaluate method.

    Returns:
        Tuple: The accuracy and loss of the evaluated model.

    Raises:
        FileExistsError: If the archive flag is set and the model state dictionary already exists in the datastore.
    """    
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
