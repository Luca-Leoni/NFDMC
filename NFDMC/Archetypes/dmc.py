import torch

from torch.nn import Module

class DiaModule(Module):
    __dia_comp = []

    def __init__(self) -> None:
        super().__init__()

    def set_dia_comp(self, dia_comp: list) -> None:
        DiaModule.__dia_comp = dia_comp

    def get_dia_comp(self) -> list:
        return DiaModule.__dia_comp
