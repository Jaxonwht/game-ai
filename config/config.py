from yaml import load, CLoader


class Config:
    def __init__(self) -> None:
        self.data = {}
        with open("config/config.yml", "r") as stream:
            self.data = load(stream, Loader=CLoader)

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
    def mcts_batch_size(self) -> int:
        return self.data.get("mcts_batch_size", 2)

    @property
    def mcts_batch_chunksize(self) -> int:
        return self.data.get("mcts_batch_chunksize", 1)
