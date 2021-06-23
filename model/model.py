from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model:
    def __init__(self, module: nn.Module, learning_rate: float, device: torch.device) -> None:
        self.device = device
        self.module: nn.Module = module.to(self.device)
        self.module.eval()
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.mse_loss(pred[:, -1], target[:, -1])
            - torch.matmul(F.softmax(target[:, :-1], dim=0), torch.log(pred[:, -1])).mean()
        )

    def train_game(
        self,
        state_list: List[torch.Tensor],
        empirical_p_list: List[torch.Tensor],
        empirical_v: int
    ) -> torch.Tensor:
        model_input = torch.stack(state_list).unsqueeze(1).float().to(self.device)

        model_output = torch.hstack((
            torch.stack(empirical_p_list),
            torch.tensor(empirical_v).repeat(len(empirical_p_list)).unsqueeze(1)
        )).float().to(self.device)

        self.module.train()

        pred = self.module(model_input)
        loss = self._loss_fn(pred, model_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.module.eval()

        return loss

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module(state.unsqueeze(0).unsqueeze(0).float().to(self.device)).squeeze(0).detach().cpu()
