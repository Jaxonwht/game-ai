from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import Tensor

class Game(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @property
    @abstractmethod
    def state_dimension(self) -> Tuple[int, int]:
        pass

    @property
    @abstractmethod
    def available_moves(self) -> List[int]:
        pass

    @abstractmethod
    def make_move(self, move: int) -> None:
        pass

    @abstractmethod
    def undo_move(self, move: int) -> None:
        pass

    @property
    @abstractmethod
    def game_state(self) -> Tensor:
        pass

    @property
    @abstractmethod
    def over(self) -> bool:
        pass

    @property
    @abstractmethod
    def score(self) -> int:
        pass

    @property
    @abstractmethod
    def number_possible_moves(self) -> int:
        pass

    @property
    @abstractmethod
    def desire_positive_score(self) -> bool:
        pass
