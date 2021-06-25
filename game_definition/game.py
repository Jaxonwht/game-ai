from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Iterable

import numpy as np


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
    def game_state(self) -> np.ndarray:
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

    @abstractmethod
    def save_intermediate_data(self) -> None:
        pass

    @property
    @abstractmethod
    def intermediate_data(self) -> Any:
        pass

    @abstractmethod
    def collect_intermediate_data(self, data_iterable: Iterable) -> None:
        pass
