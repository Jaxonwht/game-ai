from typing import List, Tuple

import numpy as np
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
        self.rng = np.random.default_rng()

    def _one_iteration(self) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        empirical_p_list: List[np.ndarray] = []
        state_list: List[np.ndarray] = []

        self.game.start()

        with torch.no_grad():
            mcts = MCTSController(self.game, self.model)
            while not self.game.over:
                mcts.simulate(self.config.train_playout_times, self.config.search_depth_cap)
                empirical_p_list.append(mcts.empirical_probability)
                state_list.append(self.game.game_state)

                sampled_move = self.rng.choice(mcts.empirical_probability.size, p=mcts.empirical_probability)
                self.game.make_move(sampled_move)
                mcts.confirm_move(sampled_move)

        return state_list, empirical_p_list, self.game.score

    def train(self) -> None:
        game_count = 0
        for _ in range(self.config.train_iterations):
            loss = self.model.train_game(*self._one_iteration())
            game_count += 1
            print(f"Game count {game_count}, loss {loss.item()}")
