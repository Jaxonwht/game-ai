from typing import List, Tuple

import torch
import torch.nn as nn


class Model:
    def __init__(self, module: nn.Module, learning_rate: float, device: str) -> None:
        self.device = device
        self.module: nn.Module = module.to(device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.module.eval()
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.mse_loss(pred[:, -1], target[:, -1])
            - torch.sum(target[:, :-1] * torch.log(pred[:, -1])) / target.size()[0]
        )

    def train_game(
        self,
        state_list: List[torch.Tensor],
        empirical_p_list: List[torch.Tensor],
        empirical_v: int
    ) -> torch.Tensor:
        model_input = torch.stack(state_list)
        model_output = torch.hstack((
            torch.stack(empirical_p_list),
            torch.tensor(empirical_v, device=self.device).repeat(len(empirical_p_list)).unsqueeze(0)
        ))

        self.module.train()

        pred = self.module(model_input)
        loss = self._loss_fn(pred, model_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.module.eval()

        return loss

    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            p_v_tuple = self.module(state.unsqueeze(0).unsqueeze(0)).squeeze(0)
        return p_v_tuple[:-1], p_v_tuple[-1].item()
