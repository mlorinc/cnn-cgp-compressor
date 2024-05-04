from commands.datastore import Datastore
from models.adapters.model_adapter_factory import create_adapter
from models.selector import FilterSelector, FilterSelectorCombinations, FilterSelectorCombination
from models.quantization import conv2d_outter, conv2d_core
import pandas as pd

selector_functions = {
    "inner": lambda filter_i, channel_i: conv2d_core([filter_i, channel_i], 5, 3),
    "outter": lambda filter_i, channel_i: conv2d_outter([filter_i, channel_i], 5, 3),
    "all": lambda *_: [(slice(None), slice(None), slice(None), slice(None))]
}

def model_sensitivity(model_name=None, model_path=None, error_type = [], **kwargs):
    data_store = Datastore().derive(model_name)
    state_dict_path = data_store / "state_dict.pth"
    data_store.mkdir(exist_ok=True, parents=True)
    data = []
    adapter = create_adapter(model_name, model_path)
    
    if not state_dict_path.exists():
        adapter.save(str(state_dict_path))
    for t in error_type:
        selector_function = selector_functions[t]
        dest = data_store / "model.{t}.256.csv"
        
        if dest.exists():
            raise FileExistsError(dest)
        
        for e in range(-128, 127, 1):
            adapter = adapter.load(str(state_dict_path))
            for layer in adapter.get_convolution_layers():
                weights = adapter.get_train_weights(layer)
                
                for filter_i, filter_weights in enumerate(weights):
                    for channel_i, channel_filters in enumerate(filter_weights):
                        for selector in selector_function(filter_i, channel_i):
                            weights[*selector] = weights[*selector] + e
                
                combinations = FilterSelectorCombinations()
                combination = FilterSelectorCombination()
                combination.add(FilterSelector(layer, [], [(slice(None), slice(None), slice(None), slice(None))]))
                combinations.add(combination)
                adapter.inject_weights([weights.flatten()], combinations, inline=True)
            top_k, avg_loss = adapter.evaluate(**kwargs)
            data.append([e] + list(top_k.values()) + [avg_loss])
        
        df = pd.DataFrame(data=data, columns=["error", "top-1", "top-5", "loss"])
        df.to_csv(dest, index=False)
