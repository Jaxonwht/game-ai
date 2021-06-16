from yaml import load, CLoader

class Config:
    def __init__(self) -> None:
        self.data = {}
        with open("config/config.yml", "r") as stream:
            self.data = load(stream, Loader=CLoader)

    @property
    def train_playout_times(self) -> int:
        return self.data.get("playout_times", 1)
