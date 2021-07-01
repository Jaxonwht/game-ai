import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy.random as random

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
    def _simulate_games(
        game: Game,
        module: nn.Module,
        device: torch.device,
        config: Config,
        queue: mp.SimpleQueue,
        rng: random.Generator
    ) -> None:
        # pylint: disable=too-many-arguments
        empirical_p_list = []
        state_list = []

        for _ in range(config.mcts_num_games_per_process):
            with torch.no_grad():
                game.start()
                mcts = MCTSController(game, module, config, device)
                while not game.over:
                    mcts.simulate(config.train_playout_times)
                    empirical_p_list.append(mcts.empirical_probability)
                    state_list.append(game.game_state)

                    sampled_move = rng.choice(game.number_possible_moves, p=mcts.empirical_probability)
                    game.make_move(sampled_move)  # type: ignore
                    mcts.confirm_move(sampled_move)  # type: ignore

            queue.put((state_list, empirical_p_list, game.score, game.intermediate_data))

    def train(self) -> None:
        queue: mp.SimpleQueue = mp.SimpleQueue()
        for _ in range(self.config.train_iterations):
            state_list_iterable = []
            empirical_p_list_iterable = []
            score_iterable = []
            processes = (
                mp.Process(
                    target=GameTrainer._simulate_games,
                    args=(
                        self.game,
                        self.model.underlying_module,
                        self.model.device,
                        self.config,
                        queue,
                        random.default_rng()
                    )
                ) for _ in range(self.config.mcts_num_processes)
            )
            for process in processes:
                process.start()
            for _ in range(self.config.mcts_num_games_per_process * self.config.mcts_num_processes):
                state_list, empirical_p_list, game_score, intermediate_data = queue.get()
                self.model.game_count += 1
                state_list_iterable.append(state_list)
                empirical_p_list_iterable.append(empirical_p_list)
                score_iterable.append(game_score)
                self.game.collect_intermediate_data(intermediate_data)
                self.game.save_intermediate_data()
            for process in processes:
                process.join()

            self.model.underlying_module.train()
            loss = self.model.train_game(
                state_list_iterable,
                empirical_p_list_iterable,
                score_iterable
            )
            self.model.underlying_module.eval()

            self.model.epoch_count += 1
            self.model.save_model(loss)
            print(f"epoch {self.model.epoch_count}, game {self.model.game_count}, loss {loss}")
