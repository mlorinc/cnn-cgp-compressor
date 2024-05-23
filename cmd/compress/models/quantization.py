import torch
from models.selector import FilterSelector
from models.selector import FilterSelector, ConstantSelector
from functools import reduce
import operator

def _tensor_iterator_helper(tensor: torch.Tensor, selector):
    """
    Helper function to iterate over a tensor based on the provided selector.

    Args:
        tensor (torch.Tensor): The input tensor.
        selector: Selector object indicating the elements to iterate over.

    Yields:
        Tuple[int, torch.Tensor]: Tuple containing the index and the corresponding tensor element.

    Raises:
        TypeError: If the selector type is unknown.
    """    
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

def tensor_iterator(tensor: torch.Tensor, selectors):
    """
    Generator function to iterate over a tensor based on the provided selectors.

    Args:
        tensor (torch.Tensor): The input tensor.
        selectors (list): List of selectors indicating the elements to iterate over.

    Yields:
        Tuple[torch.Tensor, int, list]: Tuple containing the tensor slice, size, and indices.

    """    
    for sel in selectors:    
        if isinstance(sel, ConstantSelector):
            yield torch.tensor(sel.get_values()), sel.get_size(), None  
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
    """
    Generate slices for the core of a 2D convolution operation.

    Args:
        kernel_size (int): Size of the kernel.
        core_size (int): Size of the core.

    Returns:
        list: List of slices for the core.
    """    
    # Ensure the core size is valid
    if core_size % 2 == 0 and kernel_size == core_size:
        raise ValueError("Invalid core size. It should be an odd number and not exceed the array size.")

    skip = (kernel_size - core_size) // 2
    c = slice(skip, skip + core_size)
    # Extract the core
    return [c, c]

def conv2d_outter_slices(kernel_size, core_size):
    """
    Generate slices for the outer region of a 2D convolution operation.

    Args:
        kernel_size (int): Size of the kernel.
        core_size (int): Size of the core.

    Returns:
        list: List of slices for the outer region.
    """    
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
    """
    Generate outer slices for a 2D convolution kernel based on selectors.

    Args:
        selectors (list): List of selectors.
        kernel_size (int): Size of the kernel.
        core_size (int): Size of the core.

    Returns:
        list: List of slices for the outer region.
    """    
    outter = conv2d_outter_slices(kernel_size, core_size)
    slices = []
    for out in outter:
        slices.append((*selectors, *out))
    return slices

def conv2d_core(selectors, kernel_size, core_size):
    """
    Generate core slices for a 2D convolution kernel based on selectors.

    Args:
        selectors (list): List of selectors.
        kernel_size (int): Size of the kernel.
        core_size (int): Size of the core.

    Returns:
        list: List of slices for the core.
    """    
    core = conv2d_core_slices(kernel_size, core_size)
    return [(*selectors, *core)]

def conv2d_selector(layer_name: str, selectors, kernel_size, core_size):
    """
    Generate a FilterSelector object for a 2D convolution kernel.

    Args:
        layer_name (str): Name of the layer.
        selectors (list): List of selectors.
        kernel_size (int): Size of the kernel.
        core_size (int): Size of the core.

    Returns:
        FilterSelector: FilterSelector object for the convolution kernel.
    """    
    return FilterSelector(layer_name, conv2d_core(selectors, kernel_size, core_size), conv2d_outter(selectors, kernel_size, core_size))

def quantize_per_channel(x: torch.Tensor, conv_layer: torch.Tensor):
    """
    Quantize a tensor per channel.

    Args:
        x (torch.Tensor): Input tensor.
        conv_layer (torch.Tensor): Convolutional layer tensor.

    Returns:
        torch.Tensor: Quantized tensor.
    """    
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

def quantize_per_tensor(x: torch.Tensor, scale: torch.float32, zero_point: torch.float32):
    """
    Quantize a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        scale (torch.float32): Scale factor.
        zero_point (torch.float32): Zero point.

    Returns:
        torch.Tensor: Quantized tensor.
    """    
    dequantized = ((x - zero_point) * scale).float()
    return torch.quantize_per_tensor(
        dequantized,
        scale,
        zero_point,
        dtype=torch.qint8
    )
