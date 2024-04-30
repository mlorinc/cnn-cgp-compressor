import torch
from models.selector import FilterSelector
from models.selector import FilterSelector, ConstantSelector
from functools import reduce
import operator

def _tensor_iterator_helper(tensor: torch.Tensor, selector):
    if isinstance(selector, slice):
        start = selector.start or 0
        stop = selector.stop or tensor.shape[0]
        step = selector.step or 1
        for i in range(start, stop, step):
            yield (i, tensor[i])
    elif isinstance(selector, int):
        yield (selector, tensor[selector])
    else:
        raise TypeError("unknown selector: " + str(type(selector)))

def tensor_iterator(tensor: torch.Tensor, selectors, input_size: int =None):
    for sel in selectors:    
        if isinstance(sel, ConstantSelector):
            yield torch.tensor(sel.get_values() ,size=(sel.get_size(),)), sel.get_size(), None  
            continue

        if not sel:
            continue
        
        for filter_i, filter_tensor in _tensor_iterator_helper(tensor, sel[0]):
            for channel_tensor_i, channel_tensor in _tensor_iterator_helper(filter_tensor, sel[1]):
                for row_tensor_i, row_tensor in _tensor_iterator_helper(channel_tensor, sel[2]):
                    w = row_tensor[sel[-1]]
                    size = reduce(operator.mul, w.shape)
                    yield w, size, [filter_i, channel_tensor_i, row_tensor_i, sel[-1]]                               

def conv2d_core_slices(kernel_size, core_size):
    # Ensure the core size is valid
    if core_size % 2 == 0 and kernel_size == core_size:
        raise ValueError("Invalid core size. It should be an odd number and not exceed the array size.")

    skip = (kernel_size - core_size) // 2
    c = slice(skip, skip + core_size)
    # Extract the core
    return [c, c]

def conv2d_outter_slices(kernel_size, core_size):
    # Ensure the core size is valid
    if core_size % 2 == 0 and kernel_size == core_size:
        raise ValueError("Invalid core size. It should be an odd number and not exceed the array size.")
    skip = (kernel_size - core_size) // 2

    output_indices = []
    row = 0
    for _ in range(skip):
        output_indices.append((row, slice(None)))
        row += 1

    for _ in range(core_size):
        output_indices.append((row, slice(0, skip)))
        output_indices.append((row, slice(skip+core_size, None)))
        row += 1
    
    for _ in range(skip):
        output_indices.append((row, slice(None)))
        row += 1
    return output_indices

def conv2d_outter(selectors, kernel_size, core_size):
    outter = conv2d_outter_slices(kernel_size, core_size)
    slices = []
    for out in outter:
        slices.append((*selectors, *out))
    return slices

def conv2d_core(selectors, kernel_size, core_size):
    core = conv2d_core_slices(kernel_size, core_size)
    return [(*selectors, *core)]

def conv2d_selector(layer_name: str, selectors, kernel_size, core_size):
    return FilterSelector(layer_name, conv2d_core(selectors, kernel_size, core_size), conv2d_outter(selectors, kernel_size, core_size))

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
