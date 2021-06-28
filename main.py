import argparse
import multiprocessing as mp

import torch
import torch.cuda

from config.config import Config
from game_definition.landlord_def import Landlord
from game_definition.landlord_def_v2 import Landlordv2
from gameplay.trainer import GameTrainer
from model.landlord_nn_module import LandLordNN
from model.landlord_nn_module_v2 import LandLordNNv2
from model.model import Model

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        choices=["landlord", "landlord_v2"],
        help="Name of the game to play", metavar="GAME_NAME", type=str
    )
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
    parser.add_argument(
        "--explore_constant",
        default=1.0,
        type=float,
        help="The exploratory constant used in MCTS"
    )
    args = parser.parse_args()
    config: Config = Config(vars(args))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mp.set_start_method("spawn", force=True)

    if args.name == "landlord":
        MOVES_BIN = "data/landlord/moves.bin"
        VALID_MOVES_BIN = "data/landlord/valid_moves.bin"
        CHECKPOINT_PT = "data/landlord/model.pt"

        print("Initialize landlord game")
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
        else:
            landlord_model.save_model(0)

        print("Initialize game trainer")
        landlord_trainer: GameTrainer = GameTrainer(
            landlord,
            landlord_model,
            config
        )

        print("Start training")
        landlord_trainer.train(variable_state_dim=False)

        print("Saving valid moves data for landlord")
        landlord.save_valid_moves()
    elif args.name == "landlord_v2":
        MOVES_BIN = "data/landlord_v2/moves.bin"
        VALID_MOVES_BIN = "data/landlord_v2/valid_moves.bin"
        CHECKPOINT_PT = "data/landlord_v2/model.pt"

        print("Initialize landlord_v2 game")
        landlord_v2 = Landlordv2(MOVES_BIN, VALID_MOVES_BIN, args.recompute_moves)

        row_size, col_size = landlord_v2.state_dimension

        print("Initialize pytorch nn model")
        landlord_v2_model = Model(
            (
                LandLordNNv2,
                10,
                row_size,
                col_size,
                50,
                16,
                landlord_v2.number_possible_moves + 1
            ),
            config.learning_rate,
            device,
            CHECKPOINT_PT
        )
        if not args.retrain:
            landlord_v2_model.load_model()
        else:
            landlord_v2_model.save_model(0)

        print("Initialize game trainer")
        landlord_v2_trainer = GameTrainer(
            landlord_v2,
            landlord_v2_model,
            config
        )

        print("Start training")
        landlord_v2_trainer.train(variable_state_dim=True)

        print("Saving valid moves data for landlord_v2")
        landlord_v2.save_valid_moves()
