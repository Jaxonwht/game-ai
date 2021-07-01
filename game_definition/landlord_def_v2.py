import random
import itertools
import pickle
from enum import Enum
from typing import Iterable, Dict, Tuple, List
from multiprocessing import Manager, Process

import numpy as np

from game_definition.game import Game
from game_definition.landlord.move import (
    MoveType,
    MoveInternal,
    Skip,
    Single,
    Double,
    Triple,
    Four,
    ThreePlusOne,
    ThreePlusTwo,
    Straight,
    DoubleStraight,
    TripleStraight,
    TripleStraightPlusOnes,
    TripleStraightPlusTwos,
    FourPlusTwo,
    FourPlusTwoPairs,
    DoubleJoker
)


class PlayerRole(Enum):
    LANDLORD = 0
    PEASANT_1 = 1
    PEASANT_2 = 2

    def next_role(self):
        return PlayerRole((self.value + 1) % 3)

    def prev_role(self):
        return PlayerRole((self.value - 1) % 3)


class Landlordv2(Game):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, moves_bin: str, valid_moves_bin: str, recompute_moves: bool, history_size: int) -> None:
        self.internal_moves: List[MoveInternal] = []
        self.internal_moves_back_ref: Dict[MoveInternal, int] = {}
        self.moves: List[Tuple[int, PlayerRole]] = []
        self.moves_raw: List[Tuple[int, PlayerRole]] = []
        self.played = np.zeros((3, 15), dtype=int)
        self.hands = np.zeros((3, 15), dtype=int)
        self.current_role: PlayerRole = PlayerRole.LANDLORD
        self.moves_bin = moves_bin
        self.valid_moves_bin = valid_moves_bin
        self.valid_moves: List[List[int]] = []
        self.history_size = history_size
        if recompute_moves:
            print("Does not load moves data from persistent storage")
            self.recompute_moves()
            return
        try:
            with open(moves_bin, "rb") as moves_bin_file:
                self.internal_moves = pickle.load(moves_bin_file)
                self.internal_moves_back_ref = {move: index for index, move in enumerate(self.internal_moves)}
            try:
                with open(valid_moves_bin, "rb") as valid_moves_bin_file:
                    self.valid_moves = pickle.load(valid_moves_bin_file)
            except Exception:  # pylint: disable=broad-except
                print("Does not load valid moves data from persistent storage")
                self.valid_moves = [[] for _ in range(len(self.internal_moves))]
        except Exception:  # pylint: disable=broad-except
            print("Does not load moves data from persistent storage")
            self.recompute_moves()

    @staticmethod
    def _init_hands() -> np.ndarray:
        deck = random.sample(range(54), 54)
        return np.array((
            Landlordv2._convert_cards(deck[:20]),
            Landlordv2._convert_cards(deck[20: 37]),
            Landlordv2._convert_cards(deck[37:])
        ), dtype=int)

    @staticmethod
    def _convert_cards(cards: Iterable[int]) -> List[int]:
        hand = [0] * 15
        for card in cards:
            hand[card >> 2] += 1
        return hand

    @staticmethod
    def _add_simple_combinations(moves: List[MoveInternal]) -> None:
        # skip turn
        moves.append(Skip())
        # single card
        for i in range(15):
            moves.append(Single(i))
        # double cards
        for i in range(13):
            moves.append(Double(i))
        # triple cards
        for i in range(13):
            moves.append(Triple(i))
        # four cards
        for i in range(13):
            moves.append(Four(i))
            moves.append(ThreePlusOne(i, i))
        for i, j in itertools.permutations(range(13), 2):
            # 3 + 1
            moves.append(ThreePlusOne(i, j))
            # 3 + 2
            moves.append(ThreePlusTwo(i, j))
        for i in range(13):
            # 3 + gray joker
            moves.append(ThreePlusOne(i, 13))
            # 3 + color joker
            moves.append(ThreePlusOne(i, 14))
        # double joker
        moves.append(DoubleJoker())

    @staticmethod
    def _add_four_card_combinations(moves: List[MoveInternal]) -> None:
        # 4 + 2
        for i in range(13):
            # 4 + 1 + 1
            for j, k in itertools.combinations(itertools.chain(range(i), range(i + 1, 15)), 2):  # type: ignore
                moves.append(FourPlusTwo(i, (j, k)))
            # 4 + 2
            for j in itertools.chain(range(i), range(i + 1, 13)):
                moves.append(FourPlusTwo(i, (j, j)))
        # 4 + 2 + 2
        for i in range(13):
            for j, k in itertools.combinations_with_replacement(  # type: ignore
                itertools.chain(range(i), range(i + 1, 13)),
                2
            ):
                moves.append(FourPlusTwoPairs(i, (j, k)))

    @staticmethod
    def _add_straight_combinations(moves: List[MoveInternal]) -> None:
        # straight
        for i in range(9):
            for j in range(5, 13 - i):
                moves.append(Straight(i, i + j - 1))
        # double straight
        for i in range(10):
            for j in range(3, min(13 - i, 11)):
                moves.append(DoubleStraight(i, i + j - 1))
        # triple straight
        for i in range(11):
            for j in range(2, min(13 - i, 7)):
                moves.append(TripleStraight(i, i + j - 1))

    @staticmethod
    def _add_airplane_combinations(moves: List[MoveInternal]) -> None:
        # straight 3 + 1s
        for i in range(11):
            for j in range(2, min(13 - i, 6)):
                for k in itertools.combinations_with_replacement(range(15), j):
                    array = np.zeros(15, dtype=int)
                    np.add.at(
                        array,
                        np.concatenate((np.arange(i, i + j), np.array(k))),
                        np.repeat(np.array([3, 1], dtype=int), j)
                    )
                    if np.max(array) <= 4 and array[13] <= 1 and array[14] <= 1:
                        moves.append(TripleStraightPlusOnes(i, i + j - 1, array))
        # straight 3 + 2s
        for i in range(11):
            for j in range(2, min(13 - i, 5)):
                for k in itertools.combinations_with_replacement(
                    itertools.chain(range(i), range(i + j, 13)),
                    j
                ):
                    array = np.zeros(15, dtype=int)
                    np.add.at(
                        array,
                        np.concatenate((np.arange(i, i + j), np.array(k))),
                        np.repeat(np.array([3, 2], dtype=int), j)
                    )
                    if np.max(array) <= 4:
                        moves.append(TripleStraightPlusTwos(i, i + j - 1, array))

    def recompute_moves(self) -> None:
        # pylint: disable=too-many-branches
        print("Recompute moves data for the Landlord game")
        with Manager() as manager:
            moves = manager.list()  # type: ignore
            processes = (
                Process(target=Landlordv2._add_simple_combinations, args=(moves,)),
                Process(target=Landlordv2._add_four_card_combinations, args=(moves,)),
                Process(target=Landlordv2._add_straight_combinations, args=(moves,)),
                Process(target=Landlordv2._add_airplane_combinations, args=(moves,))
            )
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            # remove possible duplicates
            self.internal_moves = list(set(moves))
            self.internal_moves_back_ref = {move: index for index, move in enumerate(self.internal_moves)}

        print("Storing Landlord moves data")
        with open(self.moves_bin, "wb") as file:
            pickle.dump(self.internal_moves, file)

        self.valid_moves = [[] for _ in range(len(self.internal_moves))]

    def start(self) -> None:
        self.moves = []
        self.moves_raw = []
        self.played = np.zeros((3, 15), dtype=int)
        self.hands = Landlordv2._init_hands()
        self.current_role = PlayerRole.LANDLORD

    @property
    def number_possible_moves(self) -> int:
        return len(self.internal_moves)

    @staticmethod
    def hand_contains_move(hand: np.ndarray, move: MoveInternal) -> bool:
        return np.all(hand >= move.cards)  # type: ignore

    def _compute_valid_move(self, move_index: int) -> List[int]:
        # pylint: disable=too-many-branches, too-many-locals, too-many-statements
        last_move: MoveInternal = self.internal_moves[move_index]
        last_move_type: MoveType = last_move.move_type

        if last_move_type == MoveType.SKIP:
            return list(range(len(self.internal_moves)))

        moves = set([self.internal_moves_back_ref[Skip()]])

        # only skip for double joker
        if last_move_type == MoveType.DOUBLE_JOKER:
            return list(moves)

        # double joker always a valid move
        moves.add(self.internal_moves_back_ref[DoubleJoker()])

        if last_move_type == MoveType.FOUR:
            last_card: int = last_move.dominant_card
            # four cards
            for i in range(last_card + 1, 13):
                moves.add(self.internal_moves_back_ref[Four(i)])
            return list(moves)

        # four cards always a valid move
        for i in range(13):
            moves.add(self.internal_moves_back_ref[Four(i)])

        if last_move_type == MoveType.SINGLE:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 15):
                moves.add(self.internal_moves_back_ref[Single(i)])
        elif last_move_type == MoveType.DOUBLE:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 13):
                moves.add(self.internal_moves_back_ref[Double(i)])
        elif last_move_type == MoveType.TRIPLE:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 13):
                moves.add(self.internal_moves_back_ref[Triple(i)])
        elif last_move_type == MoveType.THREE_PLUS_ONE:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 13):
                for j in range(15):
                    moves.add(self.internal_moves_back_ref[ThreePlusOne(i, j)])
        elif last_move_type == MoveType.THREE_PLUS_TWO:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 13):
                for j in itertools.chain(range(last_card), range(last_card + 1, i), range(i + 1, 13)):
                    moves.add(self.internal_moves_back_ref[ThreePlusTwo(i, j)])
        elif last_move_type == MoveType.STRAIGHT:
            start, end = last_move.range
            # last_card_end + i < 12
            for i in range(1, 12 - end):
                moves.add(self.internal_moves_back_ref[Straight(start + i, end + i)])
        elif last_move_type == MoveType.DOUBLE_STRAIGHT:
            start, end = last_move.range
            # last_card_end + i < 12
            for i in range(1, 12 - end):
                moves.add(self.internal_moves_back_ref[DoubleStraight(start + i, end + i)])
        elif last_move_type == MoveType.TRIPLE_STRAIGHT:
            start, end = last_move.range
            # last_card_start + i >= last_card_end + 1
            # last_card_end + i < 12
            for i in range(end - start + 1, 12 - end):
                moves.add(self.internal_moves_back_ref[TripleStraight(start + i, end + i)])
        elif last_move_type == MoveType.TRIPLE_STRAIGHT_PLUS_ONES:
            start, end = last_move.range
            for i in range(end - start + 1, 12 - end):
                for j in itertools.combinations_with_replacement(range(15), end - start + 1):  # type: ignore
                    array = np.zeros(15, dtype=int)
                    np.add.at(
                        array,
                        list(range(start + i, end + i + 1)) + list(j),  # type: ignore
                        np.repeat(np.array([3, 1]), end - start + 1)
                    )
                    if np.max(array) <= 4 and array[13] <= 1 and array[14] <= 1:
                        moves.add(self.internal_moves_back_ref[TripleStraightPlusOnes(
                            start + i,
                            end + i,
                            array
                        )])
        elif last_move_type == MoveType.TRIPLE_STRAIGHT_PLUS_TWOS:
            start, end = last_move.range
            for i in range(end - start + 1, 12 - end):
                for j in itertools.combinations_with_replacement(  # type: ignore
                    itertools.chain(range(start), range(end + i + 1, 13)),
                    end - start + 1
                ):
                    array = np.zeros(15, dtype=int)
                    np.add.at(
                        array,
                        list(range(start + i, end + i + 1)) + list(j),  # type: ignore
                        np.repeat(np.array([3, 2]), end - start + 1)
                    )
                    if np.max(array) <= 4:
                        moves.add(self.internal_moves_back_ref[TripleStraightPlusTwos(
                            start + i,
                            end + i,
                            array
                        )])
        elif last_move_type == MoveType.FOUR_PLUS_TWO:
            card: int = last_move.dominant_card
            for i in range(card + 1, 13):
                for j, k in itertools.combinations(itertools.chain(
                    range(card),
                    range(card + 1, i),
                    range(i + 1, 15)
                ), 2):
                    moves.add(self.internal_moves_back_ref[FourPlusTwo(i, (j, k))])
                for j in itertools.chain(range(card), range(card + 1, i), range(i + 1, 13)):
                    moves.add(self.internal_moves_back_ref[FourPlusTwo(i, (j, j))])
        elif last_move_type == MoveType.FOUR_PLUS_TWO_PAIRS:
            card_four = last_move.dominant_card
            for i in range(card_four + 1, 13):
                for j in itertools.combinations_with_replacement(  # type: ignore
                    itertools.chain(
                        range(card_four),
                        range(card_four + 1, i),
                        range(i + 1, 13)
                    ), 2
                ):
                    moves.add(self.internal_moves_back_ref[FourPlusTwoPairs(i, j)])  # type: ignore
        return list(moves)

    @property
    def available_moves(self) -> List[int]:
        hand = self.hands[self.current_role.value]
        moves = []
        if not self.moves or self.moves[-1][1] == self.current_role:
            for index, internal_move in enumerate(self.internal_moves):
                if Landlordv2.hand_contains_move(hand, internal_move) and internal_move.move_type != MoveType.SKIP:
                    moves.append(index)
            return moves

        last_move_index = self.moves[-1][0]
        if not self.valid_moves[last_move_index]:
            valid_moves = self._compute_valid_move(last_move_index)
            self.valid_moves[last_move_index] = valid_moves
        else:
            valid_moves = self.valid_moves[last_move_index]

        for valid_move_index in valid_moves:
            if Landlordv2.hand_contains_move(hand, self.internal_moves[valid_move_index]):
                moves.append(valid_move_index)
        return moves

    def make_move(self, move: int) -> None:
        self.moves_raw.append((move, self.current_role))
        internal_move = self.internal_moves[move]
        if internal_move == Skip():
            self.current_role = self.current_role.next_role()
            return
        hand = self.hands[self.current_role.value]
        hand -= internal_move.cards
        played = self.played[self.current_role.value]
        played += internal_move.cards
        self.moves.append((move, self.current_role))
        self.current_role = self.current_role.next_role()

    def undo_move(self, move: int) -> None:
        self.current_role = self.current_role.prev_role()
        internal_move = self.internal_moves[move]
        self.moves_raw.pop()
        if internal_move == Skip():
            return
        played = self.played[self.current_role.value]
        played -= internal_move.cards
        hand = self.hands[self.current_role.value]
        hand += internal_move.cards
        self.moves.pop()

    @property
    def state_dimension(self) -> Tuple[int, ...]:
        # Variable state dimension
        return 5, 16, self.history_size

    @property
    def game_state(self) -> np.ndarray:
        state = np.vstack((
            self.played,
            self.hands[self.current_role.value]
        ))
        if not self.moves or self.moves[-1][1] == self.current_role:
            cards_on_field = np.zeros(15, dtype=int)
        else:
            cards_on_field = self.internal_moves[self.moves[-1][0]].cards
        state = np.vstack((state, cards_on_field))

        role_col = np.zeros(5, dtype=int)
        role_col[self.current_role.value] = 1
        state = np.hstack((state, np.expand_dims(role_col, 1)))

        state = np.vstack((
            state,
            *(
                np.concatenate((self.internal_moves[index].cards, np.array((role.value,))))
                for index, role in self.moves_raw[-self.history_size:]
            )
        ))

        if state.shape[0] < 5 + self.history_size:
            return np.pad(state, ((0, 5 + self.history_size - state.shape[0]), (0, 0)), "constant", constant_values=0)
        return state

    @property
    def over(self) -> bool:
        return np.any(np.all(self.hands == 0, axis=1))  # type: ignore

    @property
    def score(self) -> int:
        # landlord win 1, peasant win -1
        return 1 if self.current_role.prev_role() == PlayerRole.LANDLORD else -1

    @property
    def desire_positive_score(self) -> bool:
        # landlord wants positive score while peasant wants negative score
        return self.current_role == PlayerRole.LANDLORD

    def save_valid_moves(self) -> None:
        with open(self.valid_moves_bin, "wb") as file:
            pickle.dump(self.valid_moves, file)

    def save_intermediate_data(self) -> None:
        self.save_valid_moves()

    @property
    def intermediate_data(self) -> List[List[int]]:
        return self.valid_moves

    def collect_intermediate_data(self, valid_moves_iterable: Iterable[List[List[int]]]) -> None:
        # pylint: disable=arguments-differ
        valid_moves_tuple = tuple(valid_moves_iterable)
        for i, moves in enumerate(self.valid_moves):
            if not moves:
                try:
                    # pylint: disable=cell-var-from-loop
                    self.valid_moves[i] = next(filter(lambda x: x, map(lambda x: x[i], valid_moves_tuple)))
                except StopIteration:
                    pass
