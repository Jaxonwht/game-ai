import argparse
import multiprocessing as mp

import torch
import torch.cuda

from config.config import Config
from game_definition.landlord_def import Landlord
from gameplay.trainer import GameTrainer
from model.landlord_nn_module import LandLordNN
from model.model import Model

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("name", choices=["landlord"], help="Name of the game to play", metavar="GAME_NAME", type=str)
    parser.add_argument("--recompute_moves", action="store_true", help="Recompute all the moves of Landlord game")
    parser.add_argument(
        "--recompute_valid_moves",
        action="store_true",
        help="Recompute all the valid moves of Landlord game"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain and overwrite checkpoint pt file"
    )
    args = parser.parse_args()
    config: Config = Config()

    if args.name == "landlord":
        mp.set_start_method("spawn", force=True)
        MOVES_BIN = "data/landlord/moves.bin"
        VALID_MOVES_BIN = "data/landlord/valid_moves.bin"
        CHECKPOINT_PT = "data/landlord/model.pt"

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print("Initialize game")
        landlord = Landlord(MOVES_BIN, VALID_MOVES_BIN, args.recompute_moves)

        row_size, col_size = landlord.state_dimension
        print("Initialize pytorch nn model")
        landlord_model: Model = Model(
            (
                LandLordNN,
                10,
                row_size,
                col_size,
                landlord.number_possible_moves + 1
            ),
            config.learning_rate,
            device,
            CHECKPOINT_PT
        )
        if not args.retrain:
            landlord_model.load_model()

        print("Initialize game trainer")
        landlord_trainer: GameTrainer = GameTrainer(
            landlord,
            landlord_model,
            config
        )

        print("Start training")
        landlord_trainer.train()

        print("Saving valid moves data for landlord")
        landlord.save_valid_moves()
