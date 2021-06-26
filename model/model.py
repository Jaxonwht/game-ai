import itertools
from typing import Tuple, List, Iterable

import torch
import numpy as np


class Model:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        module_initializer: Tuple,
        learning_rate: float,
        device: torch.device,
        checkpoint_path: str
    ) -> None:
        self.device = device
        self._module_initializer = module_initializer
        self.module = module_initializer[0](*module_initializer[1:]).to(self.device)
        self.module.train()
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self._checkpoint_path = checkpoint_path
        self.game_count = 0
        self.epoch_count = 0
        self.save_model(torch.tensor(0))

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.mse_loss(pred[:, -1], target[:, -1])
            - torch.sum(target[:, :-1] * torch.log(pred[:, :-1])) / target.size()[0]
        )

    @property
    def module_initializer(self) -> Tuple:
        return self._module_initializer

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    def train_game(
        self,
        state_list_iterable: Iterable[List[np.ndarray]],
        empirical_p_list_iterable: Iterable[List[np.ndarray]],
        empirical_v_iterable: Iterable[int]
    ) -> torch.Tensor:
        model_input = torch.from_numpy(
            np.stack(tuple(itertools.chain.from_iterable(state_list_iterable)))
        ).unsqueeze(1).float().to(self.device)

        p_v_iterable = zip(empirical_p_list_iterable, empirical_v_iterable)
        model_output = torch.from_numpy(
            np.vstack(tuple(
                np.hstack((
                    np.stack(p_list),
                    np.expand_dims(np.tile(v, len(p_list)), 1)
                )) for p_list, v in p_v_iterable
            ))
        ).float().to(self.device)

        pred = self.module(model_input)
        loss = self._loss_fn(pred, model_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu()

    def save_model(self, loss: torch.Tensor) -> None:
        torch.save(
            {
                "epoch": self.epoch_count,
                "game": self.game_count,
                "loss": loss,
                "state_dict": self.module.state_dict(),
                "optim_state_dict": self.optimizer.state_dict()
            },
            self._checkpoint_path
        )

    def load_model(self) -> None:
        try:
            checkpoint = torch.load(self._checkpoint_path, map_location=self.device)
        except Exception:  # pylint: disable=broad-except
            print(f"Failed to load {self._checkpoint_path}, start new training cycle")
            return
        if "state_dict" not in checkpoint:
            print(f"state_dict missing in {self._checkpoint_path}")
            return
        if "optim_state_dict" not in checkpoint:
            print(f"optim_state_dict missing in {self._checkpoint_path}")
            return
        self.epoch_count = checkpoint.get("epoch", 0)
        self.game_count = checkpoint.get("game", 0)
        self.module.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
