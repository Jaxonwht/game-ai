from typing import List, Tuple

import torch

from config.config import Config
from game_definition.game import Game
from mcts.mcts_controller import MCTSController
from model.model import Model


class GameTrainer:
    # pylint: disable=too-few-public-methods
    def __init__(self, game: Game, model: Model, config: Config) -> None:
        self.model: Model = model
        self.config: Config = config
        self.game: Game = game

    def _one_iteration(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        empirical_p_list: List[torch.Tensor] = []
        state_list: List[torch.Tensor] = []

        self.game.start()

        with torch.no_grad():
            mcts = MCTSController(self.game, self.model)
            while not self.game.over:
                mcts.simulate(self.config.train_playout_times)
                empirical_p_list.append(mcts.empirical_probability)
                state_list.append(self.game.game_state)

                sampled_move = int(mcts.empirical_probability.multinomial(1).item())
                self.game.make_move(sampled_move)
                mcts.confirm_move(sampled_move)

        return state_list, empirical_p_list, self.game.score

    def train(self) -> None:
        for _ in range(self.config.train_iterations):
            loss = self.model.train_game(*self._one_iteration())
            self.model.game_count += 1
            self.model.epoch_count += 1
            self.model.save_model(loss)
            print(f"epoch {self.model.epoch_count}, game {self.model.game_count}, loss {loss.item()}")
