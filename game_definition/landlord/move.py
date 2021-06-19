# type: ignore
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple


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
    def __init__(self, dict_form: Dict[int, int]) -> None:
        self._dict_form: Dict[int, int] = dict_form

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MoveInternal) and
            self.move_type == other.move_type and
            self.dict_form == other.dict_form
        )

    def __hash__(self) -> int:
        return hash((
            self.move_type,
            *(self.dict_form.get(i, 0) for i in range(15))
        ))

    @property
    def tuple_form(self) -> Tuple[int, ...]:
        return tuple(self.dict_form.get(i, 0) for i in range(15))

    @property
    def dict_form(self) -> Dict[int, int]:
        return self._dict_form

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
        super().__init__({})

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class Single(MoveInternal):
    def __init__(self, card: int, dict_form: Dict[int, int]) -> None:
        self.card: int = card
        super().__init__(dict_form)

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Double(MoveInternal):
    def __init__(self, card: int, dict_form: Dict[int, int]) -> None:
        self.card: int = card
        super().__init__(dict_form)

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Triple(MoveInternal):
    def __init__(self, card: int, dict_form: Dict[int, int]) -> None:
        self.card: int = card
        super().__init__(dict_form)

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Four(MoveInternal):
    def __init__(self, card: int, dict_form: Dict[int, int]) -> None:
        self.card: int = card
        super().__init__(dict_form)

    @property
    def dominant_card(self) -> int:
        return self.card

    @property
    def range(self) -> int:
        raise NotImplementedError()


class ThreePlusOne(MoveInternal):
    def __init__(self, three: int, dict_form: Dict[int, int]) -> None:
        self.three: int = three
        super().__init__(dict_form)

    @property
    def dominant_card(self) -> int:
        return self.three

    @property
    def range(self) -> int:
        raise NotImplementedError()


class ThreePlusTwo(MoveInternal):
    def __init__(self, three: int, dict_form: Dict[int, int]) -> None:
        self.three: int = three
        super().__init__(dict_form)

    @property
    def dominant_card(self) -> int:
        return self.three

    @property
    def range(self) -> int:
        raise NotImplementedError()


class Straight(MoveInternal):
    def __init__(self, start: int, end: int, dict_form: Dict[int, int]) -> None:
        self.start: int = start
        self.end: int = end
        super().__init__(dict_form)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class DoubleStraight(MoveInternal):
    def __init__(self, start: int, end: int, dict_form: Dict[int, int]) -> None:
        self.start: int = start
        self.end: int = end
        super().__init__(dict_form)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class TripleStraight(MoveInternal):
    def __init__(self, start: int, end: int, dict_form: Dict[int, int]) -> None:
        self.start: int = start
        self.end: int = end
        super().__init__(dict_form)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class TripleStraightPlusOnes(MoveInternal):
    def __init__(self, start: int, end: int, dict_form: Dict[int, int]) -> None:
        self.start: int = start
        self.end: int = end
        super().__init__(dict_form)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class TripleStraightPlusTwos(MoveInternal):
    def __init__(self, start: int, end: int, dict_form: Dict[int, int]) -> None:
        self.start: int = start
        self.end: int = end
        super().__init__(dict_form)

    @property
    def range(self) -> Tuple[int, int]:
        return self.start, self.end

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()


class FourPlusTwo(MoveInternal):
    def __init__(self, four: int, dict_form: Dict[int, int]) -> None:
        self.four: int = four
        super().__init__(dict_form)

    @property
    def dominant_card(self) -> int:
        return self.four

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()


class FourPlusTwoPairs(MoveInternal):
    def __init__(self, four: int, dict_form: Dict[int, int]) -> None:
        self.four: int = four
        super().__init__(dict_form)

    @property
    def dominant_card(self) -> int:
        return self.four

    @property
    def range(self) -> Tuple[int, int]:
        raise NotImplementedError()


class DoubleJoker(MoveInternal):
    def __init__(self) -> None:
        super().__init__({13: 1, 14: 1})

    @property
    def dominant_card(self) -> int:
        raise NotImplementedError()

    @property
    def range(self) -> int:
        raise NotImplementedError()
