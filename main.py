import torch
import torch.nn as nn
import torch.cuda

from config.config import Config
from game_definition.game import Game
from game_definition.landlord_def import Landlord
from gameplay.trainer import GameTrainer
from model.landlord_nn_module import LandLordNN
from model.model import Model

if __name__ == "__main__":
    config: Config = Config()

    print("Initialize game")
    landlord: Game = Landlord()

    state_dim: torch.Size = landlord.game_state.size()
    print("Initialize pytorch nn model")
    landlord_module: nn.Module = LandLordNN(
        10,
        state_dim[0],
        state_dim[1],
        landlord.number_possible_moves + 1
    )

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    landlord_model: Model = Model(landlord_module, config.learning_rate, device)

    print("Initialize game trainer")
    landlord_trainer: GameTrainer = GameTrainer(
        landlord,
        landlord_model,
        config
    )

    print("Start training")
    landlord_trainer.train()
