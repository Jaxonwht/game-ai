import multiprocessing as mp
from typing import List, Tuple, Any
from os.path import isfile

import torch
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
        game: Game, module_initializer: Tuple, checkpoint_path: str, config: Config
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int, Any]:
        empirical_p_list = []
        state_list = []
        rng = np.random.default_rng()

        with torch.no_grad():
            model = module_initializer[0](*module_initializer[1:])
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"])
            mcts = MCTSController(game, model, config)
            while not game.over:
                mcts.simulate(config.train_playout_times)
                empirical_p_list.append(mcts.empirical_probability)
                state_list.append(game.game_state)

                sampled_move = rng.choice(
                    game.number_possible_moves,
                    p=mcts.empirical_probability
                )
                game.make_move(sampled_move)
                mcts.confirm_move(sampled_move)

        return state_list, empirical_p_list, game.score, game.intermediate_data

    def train(self, variable_state_dim: bool) -> None:
        # Initialize model for other processes to access in case the file does not exist
        if not isfile(self.model.checkpoint_path):
            self.model.save_model(0)
        for _ in range(self.config.train_iterations):
            self.game.start()
            with mp.Pool() as pool:
                iterable = pool.starmap(
                    GameTrainer._one_iteration,
                    (
                        (
                            self.game,
                            self.model.module_initializer,
                            self.model.checkpoint_path,
                            self.config
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
                loss = self.model.train_game(
                    state_list_iterable,  # type: ignore
                    empirical_p_list_iterable,  # type: ignore
                    score_iterable,  # type: ignore
                    variable_state_dim
                )
                self.model.game_count += self.config.mcts_batch_size
                self.model.epoch_count += 1
                self.model.save_model(loss)
                print(f"epoch {self.model.epoch_count}, game {self.model.game_count}, loss {loss}")
                self.game.collect_intermediate_data(intermediate_data_iterable)
                self.game.save_intermediate_data()
