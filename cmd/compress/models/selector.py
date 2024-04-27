from typing import List

class FilterSelector(object):
    def __init__(self, layer_name: str, inp: List, out: List) -> None:
        self.layer_name = layer_name
        self.inp = inp
        self.out = out

class ZeroSelector(object):
    def __init__(self, size: int) -> None:
        self.size = size
        
class ByteSelector(object):
    def __init__(self) -> None:
        pass