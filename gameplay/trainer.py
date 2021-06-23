from typing import List, Tuple

import torch
import torch.multiprocessing as mp

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
    def _one_iteration(args) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        game, inference_model, config = args
        empirical_p_list: List[torch.Tensor] = []
        state_list: List[torch.Tensor] = []

        game.start()

        with torch.no_grad():
            mcts = MCTSController(game, inference_model)
            while not game.over:
                mcts.simulate(config.train_playout_times)
                empirical_p_list.append(mcts.empirical_probability)
                state_list.append(game.game_state)

                sampled_move = int(mcts.empirical_probability.multinomial(1).item())
                game.make_move(sampled_move)
                mcts.confirm_move(sampled_move)

        return state_list, empirical_p_list, game.score

    def train(self) -> None:
        for _ in range(self.config.train_iterations):
            with mp.Pool() as pool:
                iterator = pool.imap_unordered(
                    GameTrainer._one_iteration,
                    (
                        (
                            self.game,
                            self.model.inference_model,
                            self.config
                        ) for _ in range(self.config.mcts_batch_size)
                    ),
                    chunksize=self.config.mcts_batch_chunksize
                )
                loss = self.model.train_game(*zip(*iterator))  # type: ignore
            self.model.game_count += self.config.mcts_batch_size
            self.model.epoch_count += 1
            self.model.save_model(loss)
            print(f"epoch {self.model.epoch_count}, game {self.model.game_count}, loss {loss.item()}")
