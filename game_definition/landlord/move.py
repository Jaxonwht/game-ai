# type: ignore
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import torch


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
    def __init__(self, cards: torch.Tensor) -> None:
        self._cards = cards
        self._cards.share_memory_()

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MoveInternal) and
            self.move_type == other.move_type and
            torch.all(self.cards == other.cards).item()
        )

    def __hash__(self) -> int:
        return hash((self.move_type, str(self._cards)))

    def __repr__(self) -> str:
        return f"MoveInternal(type: {self.move_type.name}, cards: {self.cards})"

    @property
    def cards(self) -> torch.Tensor:
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
        super().__init__(torch.zeros(15, dtype=torch.int64))

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class Single(MoveInternal):
    def __init__(self, card: int) -> None:
        self.card: int = card
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor(card),),
                torch.tensor(1, dtype=torch.int64)
            )
        )

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Double(MoveInternal):
    def __init__(self, card: int) -> None:
        self.card: int = card
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor(card),),
                torch.tensor(2, dtype=torch.int64)
            )
        )

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Triple(MoveInternal):
    def __init__(self, card: int) -> None:
        self.card: int = card
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor(card),),
                torch.tensor(3, dtype=torch.int64)
            )
        )

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Four(MoveInternal):
    def __init__(self, card: int) -> None:
        self.card: int = card
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor(card),),
                torch.tensor(4, dtype=torch.int64)
            )
        )

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class ThreePlusOne(MoveInternal):
    def __init__(self, three: int, one: int) -> None:
        self.three: int = three
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor((three, one)),),
                torch.tensor((3, 1), dtype=torch.int64),
                accumulate=True
            )
        )

    @property
    def dominant_card(self) -> int:
        return self.three

    @property
    def range(self) -> int:
        raise NotImplementedError()


class ThreePlusTwo(MoveInternal):
    def __init__(self, three: int, two: int) -> None:
        self.three: int = three
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor((three, two)),),
                torch.tensor((3, 2), dtype=torch.int64),
                accumulate=True
            )
        )

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
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.arange(start, end + 1),),
                torch.tensor(1, dtype=torch.int64)
            )
        )

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
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.arange(start, end + 1),),
                torch.tensor(2, dtype=torch.int64)
            )
        )

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
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.arange(start, end + 1),),
                torch.tensor(3, dtype=torch.int64)
            )
        )

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class TripleStraightPlusOnes(MoveInternal):
    def __init__(self, start: int, end: int, array: torch.Tensor) -> None:
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
    def __init__(self, start: int, end: int, array: torch.Tensor) -> None:
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
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor((four,) + ones),),
                torch.tensor((4, 1, 1), dtype=torch.int64),
                accumulate=True
            )
        )

    @property
    def dominant_card(self) -> int:
        return self.four

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()


class FourPlusTwoPairs(MoveInternal):
    def __init__(self, four: int, twos: Tuple[int, int]) -> None:
        self.four: int = four
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor((four,) + twos),),
                torch.tensor((4, 2, 2), dtype=torch.int64),
                accumulate=True
            )
        )

    @property
    def dominant_card(self) -> int:
        return self.four

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()


class DoubleJoker(MoveInternal):
    def __init__(self) -> None:
        super().__init__(
            torch.index_put(
                torch.zeros(
                    15,
                    dtype=torch.int64
                ),
                (torch.tensor((13, 14)),),
                torch.tensor((1, 1), dtype=torch.int64)
            )
        )

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()

    @property
    def range(self) -> int:
        raise NotImplementedError()
