import random
import itertools
from enum import Enum
from typing import Iterable, Dict, Tuple, List

from torch import Tensor

from game_definition.game import Game

class PlayerRole(Enum):
    LANDLORD = 0
    PEASANT_1 = 1
    PEASANT_2 = 2

    def next_role(self):
        return PlayerRole((self.value + 1) % 3)

    def prev_role(self):
        return PlayerRole((self.value - 1) % 3)

class Landlord(Game):
    def __init__(self) -> None:
        self.internal_moves: List[Dict[int, int]] = Landlord._init_moves()

    @staticmethod
    def _init_hands() -> Tuple[List[int], List[int], List[int]]:
        deck = random.sample(range(54), 54)
        return Landlord._convert_cards(deck[:20]), Landlord._convert_cards(deck[20 : 37]), Landlord._convert_cards(deck[37:])

    @staticmethod
    def _convert_cards(cards: Iterable[int]) -> List[int]:
        hand = [0] * 15
        for c in cards:
            hand[c >> 2] += 1
        return hand

    @staticmethod
    def _init_moves() -> List[Dict[int, int]]:
        all_moves = []
        # single card
        for i in range(15):
            all_moves.append({i : 1})
        # double cards
        for i in range(13):
            all_moves.append({i : 2})
        # triple cards
        for i in range(13):
            all_moves.append({i : 3})
        # four cards
        for i in range(13):
            all_moves.append({i : 4})
        for i, j in itertools.permutations(range(13), 2):
            # 3 + 1
            all_moves.append({i : 3, j : 1})
            # 3 + 2
            all_moves.append({i : 3, j : 2})

        return all_moves

    def start(self) -> None:
        self.moves: List[int] = []
        self.played: Tuple[List[int], List[int], List[int]] = ([0] * 15, [0] * 15, [0] * 15)
        self.hands: Tuple[List[int], List[int], List[int]] = Landlord._init_hands()
        self.current_role: PlayerRole = PlayerRole.LANDLORD

    @property
    def number_possible_moves(self) -> int:
        return len(self.internal_moves)

    @property
    def available_moves(self) -> Iterable[int]:
        # TODO
        yield -1

    def make_move(self, move: int) -> None:
        hand = self.hands[self.current_role.value]
        played = self.played[self.current_role.value]
        for card, count in self.internal_moves[move].items():
            hand[card] -= count
            played[card] += count
        self.moves.append(move)
        self.current_role = self.current_role.next_role()

    def undo_move(self, move: int) -> None:
        self.current_role = self.current_role.prev_role()
        hand = self.hands[self.current_role.value]
        played = self.played[self.current_role.value]
        for card, count in self.internal_moves[move].items():
            hand[card] += count
            played[card] -= count
        self.moves.pop()

    @property
    def game_state(self) -> Tensor:
        # TODO
        return

    @property
    def over(self) -> bool:
        return any(all(count == 0 for count in hand) for hand in self.hands)

    @property
    def score(self) -> int:
        # landlord win 1, peasant win -1
        return 1 if self.current_role.prev_role() == PlayerRole.LANDLORD else -1

    @property
    def desire_positive_score(self) -> bool:
        # landlord wants positive score while peasant wants negative score
        return True if self.current_role == PlayerRole.LANDLORD else False
