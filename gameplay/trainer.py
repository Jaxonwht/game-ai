import multiprocessing as mp
from typing import List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np

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

    @staticmethod
    def _one_iteration(
        game: Game, module: nn.Module, config: Config, device: torch.device
    ) -> Tuple[List[np.ndarray], List[torch.Tensor], int, Any]:
        empirical_p_list = []
        state_list = []

        with torch.no_grad():
            mcts = MCTSController(game, module, config, device)
            while not game.over:
                mcts.simulate(config.train_playout_times)
                empirical_p_list.append(mcts.empirical_probability)
                state_list.append(game.game_state)

                sampled_move = mcts.empirical_probability.multinomial(1).item()
                game.make_move(sampled_move)  # type: ignore
                mcts.confirm_move(sampled_move)  # type: ignore

        return state_list, empirical_p_list, game.score, game.intermediate_data

    def train(self) -> None:
        for _ in range(self.config.train_iterations):
            self.game.start()
            with mp.Pool() as pool:
                iterable = pool.starmap(
                    GameTrainer._one_iteration,
                    (
                        (
                            self.game,
                            self.model.underlying_module,
                            self.config,
                            self.model.device
                        ) for _ in range(self.config.mcts_batch_size)
                    ),
                    chunksize=self.config.mcts_batch_chunksize
                )
                (
                    state_list_iterable,
                    empirical_p_list_iterable,
                    score_iterable,
                    intermediate_data_iterable
                ) = zip(*iterable)
                self.model.underlying_module.train()
                loss = self.model.train_game(
                    state_list_iterable,  # type: ignore
                    empirical_p_list_iterable,  # type: ignore
                    score_iterable  # type: ignore
                )
                self.model.underlying_module.eval()
                self.model.game_count += self.config.mcts_batch_size
                self.model.epoch_count += 1
                self.model.save_model(loss)
                print(f"epoch {self.model.epoch_count}, game {self.model.game_count}, loss {loss}")
                self.game.collect_intermediate_data(intermediate_data_iterable)
                self.game.save_intermediate_data()
