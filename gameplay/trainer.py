from typing import List, Tuple

import torch

from config.config import Config
from game_definition.game import Game
from mcts.mcts_controller import MCTSController
from model.model import Model


class GameTrainer:
    # pylint: disable=too-few-public-methods
    def __init__(self, game: Game, model: Model, config: Config, device: str) -> None:
        self.model: Model = model
        self.config: Config = config
        self.game: Game = game
        self.device = device

    def _one_iteration(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        empirical_p_list: List[torch.Tensor] = []
        state_list: List[torch.Tensor] = []
        self.game.start()

        with torch.no_grad():
            mcts = MCTSController(self.game, self.model, self.device)
            while not self.game.over:
                mcts.simulate(self.config.train_playout_times, self.config.search_depth_cap)
                empirical_p_list.append(mcts.empirical_probability)
                state_list.append(self.game.game_state)

                sampled_move: int = int(mcts.empirical_probability.multinomial(1).item())
                self.game.make_move(sampled_move)
                mcts.confirm_move(sampled_move)

        return state_list, empirical_p_list, self.game.score

    def train(self) -> None:
        game_count = 0
        for _ in range(self.config.train_iterations):
            state_list, empirical_p_list, game_score = self._one_iteration()
            loss = self.model.train_game(state_list, empirical_p_list, game_score)
            game_count += 1
            print(f"Game count {game_count}, loss {loss.item()}")
