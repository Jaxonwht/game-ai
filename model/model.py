from typing import List, Tuple, Type

import torch


class Model:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        module_initializer: Tuple[Type, int, int, int, int],
        learning_rate: float,
        device: torch.device,
        checkpoint_path: str
    ) -> None:
        self.device = device
        self.module_initializer = module_initializer
        self.module = module_initializer[0](*module_initializer[1:]).to(self.device)
        self.module.train()
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.checkpoint_path = checkpoint_path
        self.game_count = 0
        self.epoch_count = 0
        self.save_model(torch.tensor(0))

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.mse_loss(pred[:, -1], target[:, -1])
            - torch.sum(target[:, :-1] * torch.log(pred[:, :-1])) / target.size()[0]
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
