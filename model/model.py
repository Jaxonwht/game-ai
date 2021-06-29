import itertools
from typing import Tuple, List, Iterable

import torch
import torch.nn as nn
import numpy as np


class Model:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        module: nn.Module,
        learning_rate: float,
        device: torch.device,
        checkpoint_path: str
    ) -> None:
        self._device = device
        self.module = module.to(self._device).share_memory()
        self.module.eval()
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self._checkpoint_path = checkpoint_path
        self.game_count = 0
        self.epoch_count = 0

    @property
    def underlying_module(self) -> nn.Module:
        return self.module

    @property
    def device(self) -> torch.device:
        return self._device

    def _loss_fn(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        game_sizes: Tuple[int, ...]
    ) -> torch.Tensor:
        cumulative_game_sizes = itertools.accumulate(game_sizes)
        start = 0
        loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for end in cumulative_game_sizes:
            loss += (
                self.mse_loss(pred[start: end, -1], target[start: end, - 1])
                - torch.sum(target[start: end, :-1] * torch.log(pred[start: end, :-1]))
                / (end - start)
            )
            start = end
        return loss / len(game_sizes)

    def train_game(
        self,
        state_list_iterable: Iterable[List[np.ndarray]],
        empirical_p_list_iterable: Iterable[List[np.ndarray]],
        empirical_v_iterable: Iterable[int],
        variable_state_dim: bool
    ) -> float:
        # pylint: disable=too-many-locals
        if variable_state_dim:
            total_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            batch_count = 0
            for (
                state_list,
                empirical_p_list,
                empirical_v
            ) in zip(
                state_list_iterable,
                empirical_p_list_iterable,
                empirical_v_iterable
            ):
                per_game_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
                for state, empirical_p in zip(state_list, empirical_p_list):
                    model_input = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    model_output = torch.from_numpy(
                        np.hstack((empirical_p, empirical_v))
                    ).unsqueeze(0).float().to(self.device)

                    pred = self.module(model_input)
                    per_game_loss += self._loss_fn(pred, model_output, (1,))
                total_loss += per_game_loss / len(state_list)
                batch_count += 1

            total_loss /= batch_count

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            return total_loss.item()

        game_sizes = tuple(len(state_list) for state_list in state_list_iterable)

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
        loss = self._loss_fn(pred, model_output, game_sizes)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # type: ignore

    def save_model(self, loss: float) -> None:
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
        loss = checkpoint.get("loss", 0)
        print(f"Load last model, epoch {self.epoch_count}, game {self.game_count}, loss {loss}")
        self.module.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
