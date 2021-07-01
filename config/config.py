from typing import Any, Dict

from yaml import load, CLoader


class Config:
    def __init__(self, args: Dict[str, Any]) -> None:
        self.data = {}
        with open("config/config.yml", "r") as stream:
            self.data = load(stream, Loader=CLoader)
        self.data.update(args)

    @property
    def train_playout_times(self) -> int:
        return self.data.get("playout_times", 100)

    @property
    def train_iterations(self) -> int:
        return self.data.get("train_iterations", 100)

    @property
    def learning_rate(self) -> float:
        return self.data.get("learning_rate", 1.0e-3)

    @property
    def mcts_num_games_per_process(self) -> int:
        return self.data.get("mcts_num_games_per_iteration", 1)

    @property
    def mcts_num_processes(self) -> int:
        return self.data.get("mcts_num_processes", 4)

    @property
    def explore_constant(self) -> float:
        return self.data.get("explore_constant", 1)

    @property
    def mcts_depth_cap(self) -> int:
        return self.data.get("mcts_depth_cap", -1)
