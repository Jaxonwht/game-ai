from typing import List, Tuple

import torch
import torch.multiprocessing as mp

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

    @staticmethod
    def _one_iteration(args) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        game, model, device, train_playout_times, search_depth_cap = args
        empirical_p_list: List[torch.Tensor] = []
        state_list: List[torch.Tensor] = []

        with torch.no_grad():
            mcts = MCTSController(game, model, device)
            while not game.over:
                mcts.simulate(train_playout_times, search_depth_cap)
                empirical_p_list.append(mcts.empirical_probability)
                state_list.append(game.game_state)

                sampled_move: int = int(mcts.empirical_probability.multinomial(1).item())
                game.make_move(sampled_move)
                mcts.confirm_move(sampled_move)

        return state_list, empirical_p_list, game.score

    def train(self) -> None:
        game_count = 0
        for _ in range(self.config.train_iterations):
            self.game.start()
            with mp.Pool() as pool:
                iterator = pool.imap_unordered(
                    GameTrainer._one_iteration,
                    (
                        (
                            self.game,
                            self.model,
                            self.device,
                            self.config.train_playout_times,
                            self.config.search_depth_cap
                        ) for _ in range(self.config.mcts_batch_size)
                    ),
                    chunksize=self.config.mcts_batch_chunk_size
                )
                loss = self.model.train_game(*zip(*iterator))  # type: ignore
                game_count += self.config.mcts_batch_size
                print(f"Game count {game_count}, loss {loss.item()}")
