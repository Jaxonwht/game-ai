from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import numpy as np


class MoveType(Enum):
    SKIP = "Skip"
    SINGLE = "Single"
    DOUBLE = "Double"
    TRIPLE = "Triple"
    FOUR = "Four"
    THREE_PLUS_ONE = "ThreePlusOne"
    THREE_PLUS_TWO = "ThreePlusTwo"
    STRAIGHT = "Straight"
    DOUBLE_STRAIGHT = "DoubleStraight"
    TRIPLE_STRAIGHT = "TripleStraight"
    TRIPLE_STRAIGHT_PLUS_ONES = "TripleStraightPlusOnes"
    TRIPLE_STRAIGHT_PLUS_TWOS = "TripleStraightPlusTwos"
    FOUR_PLUS_TWO = "FourPlusTwo"
    FOUR_PLUS_TWO_PAIRS = "FourPlusTwoPairs"
    DOUBLE_JOKER = "DoubleJoker"


class MoveInternal(ABC):
    def __init__(self, cards: np.ndarray) -> None:
        self._cards = cards
        self._cards.flags.writeable = False

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MoveInternal) and
            self.move_type == other.move_type and
            np.all(self._cards == other.cards)
        )

    def __hash__(self) -> int:
        return hash((self.move_type, self._cards.tobytes()))

    def __repr__(self) -> str:
        return f"MoveInternal(type: {self.move_type.name}, cards: {self._cards})"

    @property
    def cards(self) -> np.ndarray:
        return self._cards

    @property
    def move_type(self) -> MoveType:
        return MoveType(self.__class__.__name__)

    @property
    @abstractmethod
    def dominant_card(self) -> int:
        pass

    @property
    @abstractmethod
    def range(self) -> Tuple[int, int]:
        pass


class Skip(MoveInternal):
    def __init__(self) -> None:
        super().__init__(np.zeros(15, dtype=int))

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class Single(MoveInternal):
    def __init__(self, card: int) -> None:
        self.card: int = card
        array = np.zeros(15, dtype=int)
        array[card] = 1
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Double(MoveInternal):
    def __init__(self, card: int) -> None:
        self.card: int = card
        array = np.zeros(15, dtype=int)
        array[card] = 2
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Triple(MoveInternal):
    def __init__(self, card: int) -> None:
        self.card: int = card
        array = np.zeros(15, dtype=int)
        array[card] = 3
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Four(MoveInternal):
    def __init__(self, card: int) -> None:
        self.card: int = card
        array = np.zeros(15, dtype=int)
        array[card] = 4
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class ThreePlusOne(MoveInternal):
    def __init__(self, three: int, one: int) -> None:
        self.three: int = three
        array = np.zeros(15, dtype=int)
        np.add.at(array, [three, one], (3, 1))
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        return self.three

    @property
    def range(self) -> int:
        raise NotImplementedError()


class ThreePlusTwo(MoveInternal):
    def __init__(self, three: int, two: int) -> None:
        self.three: int = three
        array = np.zeros(15, dtype=int)
        array[[three, two]] = (3, 2)
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        return self.three

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Straight(MoveInternal):
    def __init__(self, start: int, end: int) -> None:
        self.start: int = start
        self.end: int = end
        array = np.zeros(15, dtype=int)
        array[range(start, end + 1)] = 1
        super().__init__(array)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class DoubleStraight(MoveInternal):
    def __init__(self, start: int, end: int) -> None:
        self.start: int = start
        self.end: int = end
        array = np.zeros(15, dtype=int)
        array[range(start, end + 1)] = 2
        super().__init__(array)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class TripleStraight(MoveInternal):
    def __init__(self, start: int, end: int) -> None:
        self.start: int = start
        self.end: int = end
        array = np.zeros(15, dtype=int)
        array[range(start, end + 1)] = 3
        super().__init__(array)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class TripleStraightPlusOnes(MoveInternal):
    def __init__(self, start: int, end: int, array: np.ndarray) -> None:
        self.start: int = start
        self.end: int = end
        super().__init__(array)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class TripleStraightPlusTwos(MoveInternal):
    def __init__(self, start: int, end: int, array: np.ndarray) -> None:
        self.start: int = start
        self.end: int = end
        super().__init__(array)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class FourPlusTwo(MoveInternal):
    def __init__(self, four: int, ones: Tuple[int, int]) -> None:
        self.four: int = four
        array = np.zeros(15, dtype=int)
        np.add.at(array, list((four,) + ones), (4, 1, 1))
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        return self.four

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()


class FourPlusTwoPairs(MoveInternal):
    def __init__(self, four: int, twos: Tuple[int, int]) -> None:
        self.four: int = four
        array = np.zeros(15, dtype=int)
        np.add.at(array, list((four,) + twos), (4, 2, 2))
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        return self.four

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()


class DoubleJoker(MoveInternal):
    def __init__(self) -> None:
        array = np.zeros(15, dtype=int)
        array[13:] = 1
        super().__init__(array)

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()

    @property
    def range(self) -> int:
        raise NotImplementedError()
