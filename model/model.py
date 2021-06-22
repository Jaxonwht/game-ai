from typing import List

import numpy as np
import torch
import torch.nn as nn


class Model:
    def __init__(self, module: nn.Module, learning_rate: float, device: torch.device) -> None:
        self.device = device
        self.module: nn.Module = module.to(device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.mse_loss(pred[:, -1], target[:, -1])
            - torch.sum(target[:, :-1] * torch.log(pred[:, -1])) / target.size()[0]
        )

    def train_game(
        self,
        state_list: List[np.ndarray],
        empirical_p_list: List[np.ndarray],
        empirical_v: int
    ) -> torch.Tensor:
        model_input = torch.from_numpy(np.stack(state_list)).float().to(self.device)

        model_output = torch.from_numpy(
            np.hstack((np.stack(empirical_p_list), np.repeat(np.array([empirical_v]), len(empirical_p_list))))
        ).float().to(self.device)

        pred = self.module(model_input)
        loss = self._loss_fn(pred, model_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.module(
                torch.from_numpy(np.expand_dims(state, (0, 1))).float().to(self.device)
            ).squeeze(0).detach().cpu().numpy()
