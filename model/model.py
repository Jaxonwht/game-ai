import itertools
from copy import deepcopy
from typing import List, Iterable

import torch
import torch.nn as nn


class InterferenceModel:
    # pylint: disable=too-few-public-methods
    def __init__(self, module: nn.Module):
        self.module = module

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module(state.unsqueeze(0).unsqueeze(0).float()).squeeze(0)


class Model:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, module: nn.Module, learning_rate: float, device: torch.device, checkpoint_path: str) -> None:
        self.device = device
        self.inference_module = deepcopy(module).share_memory()
        self.module: nn.Module = module.to(self.device)
        self.module.train()
        self.inference_module.eval()
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.checkpoint_path = checkpoint_path
        self.game_count = 0
        self.epoch_count = 0

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.mse_loss(pred[:, -1], target[:, -1])
            - torch.sum(target[:, :-1] * torch.log(pred[:, :-1])) / target.size()[0]
        )

    def train_game(
        self,
        state_list_iterable: Iterable[List[torch.Tensor]],
        empirical_p_list_iterable: Iterable[List[torch.Tensor]],
        empirical_v_iterable: Iterable[int]
    ) -> torch.Tensor:
        model_input = torch.stack(
            tuple(itertools.chain.from_iterable(state_list_iterable))
        ).unsqueeze(1).float().to(self.device)

        p_v_iterable = zip(empirical_p_list_iterable, empirical_v_iterable)
        model_output = torch.vstack(tuple(
            torch.hstack((
                torch.stack(empirical_p_list),
                torch.tensor(empirical_v).repeat(len(empirical_p_list)).unsqueeze(1)
            )) for empirical_p_list, empirical_v in p_v_iterable
        )).float().to(self.device)

        pred = self.module(model_input)
        loss = self._loss_fn(pred, model_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.inference_module.load_state_dict(self.module.state_dict())  # type: ignore

        return loss.detach().cpu()

    @property
    def inference_model(self) -> InterferenceModel:
        return InterferenceModel(self.inference_module)

    def save_model(self, loss: torch.Tensor) -> None:
        torch.save(
            {
                "epoch": self.epoch_count,
                "game": self.game_count,
                "loss": loss,
                "state_dict": self.module.state_dict(),
                "optim_state_dict": self.optimizer.state_dict()
            },
            self.checkpoint_path
        )

    def load_model(self) -> None:
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception:  # pylint: disable=broad-except
            print(f"Failed to load {self.checkpoint_path}, start new training cycle")
            return
        if "state_dict" not in checkpoint:
            print(f"state_dict missing in {self.checkpoint_path}")
            return
        if "optim_state_dict" not in checkpoint:
            print(f"optim_state_dict missing in {self.checkpoint_path}")
            return
        self.epoch_count = checkpoint.get("epoch", 0)
        self.game_count = checkpoint.get("game", 0)
        self.module.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
        self.inference_module.load_state_dict(self.module.state_dict())  # type: ignore
