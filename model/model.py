from typing import List, Tuple

import torch
import torch.nn as nn

class Model:
    def __init__(self, module: nn.Module) -> None:
        self.batch: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        self.module: nn.Module = module
        self.module.eval()

    def add_sample(self, state: torch.Tensor, p: torch.Tensor, v: float) -> None:
        self.batch.append((state, p, v))

    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            p_v_tuple = self.module(state)
        return p_v_tuple[:-1], p_v_tuple[-1].item()
