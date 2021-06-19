from typing import List

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

    def _one_iteration(self) -> torch.Tensor:
        empirical_p_list: List[torch.Tensor] = []
        state_list: List[torch.Tensor] = []
        self.game.start()

        while not self.game.over:
            mcts: MCTSController = MCTSController(self.game, self.model)
            mcts.simulate(self.config.train_playout_times)
            empirical_p_list.append(mcts.empirical_p)
            state_list.append(self.game.game_state)

            sampled_move: int = int(mcts.empirical_p.multinomial(1).item())
            self.game.make_move(sampled_move)

        return self.model.train_game(state_list, empirical_p_list, self.game.score)

    def train(self) -> None:
        game_count = 0
        for _ in range(self.config.train_iterations):
            loss: torch.Tensor = self._one_iteration()
            game_count += 1
            print(f"Game count {game_count}, loss {loss.item()}")
