from typing import List

class FilterSelector(object):
    def __init__(self, layer_name: str, inp: List, out: List) -> None:
        self.layer_name = layer_name
        self.inp = inp
        self.out = out
