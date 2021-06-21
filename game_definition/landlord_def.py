import random
import itertools
import pickle
from enum import Enum
from collections import Counter
from typing import Iterable, Dict, Tuple, List
from multiprocessing import Manager, Process

import torch
from torch import Tensor

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


class Landlord(Game):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, moves_bin: str, valid_moves_bin: str, torch_device: str) -> None:
        self.internal_moves: List[MoveInternal] = []
        self.internal_moves_back_ref: Dict[MoveInternal, int] = {}
        self.moves: List[Tuple[int, PlayerRole]] = []
        self.played: Tuple[List[int], List[int], List[int]] = ([], [], [])
        self.hands: Tuple[List[int], List[int], List[int]] = ([], [], [])
        self.current_role: PlayerRole = PlayerRole.LANDLORD
        self.moves_bin = moves_bin
        self.valid_moves_bin = valid_moves_bin
        self.torch_device = torch_device
        self.valid_moves: List[List[int]] = []
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
    def _init_hands() -> Tuple[List[int], List[int], List[int]]:
        deck = random.sample(range(54), 54)
        return (
            Landlord._convert_cards(deck[:20]),
            Landlord._convert_cards(deck[20: 37]),
            Landlord._convert_cards(deck[37:])
        )

    @staticmethod
    def _convert_cards(cards: Iterable[int]) -> List[int]:
        hand = [0] * 15
        for card in cards:
            hand[card >> 2] += 1
        return hand

    @staticmethod
    def _add_simple_combinations(moves: List[MoveInternal]) -> None:
        # single card
        for i in range(15):
            moves.append(Single(i, {i: 1}))
        # double cards
        for i in range(13):
            moves.append(Double(i, {i: 2}))
        # triple cards
        for i in range(13):
            moves.append(Triple(i, {i: 3}))
        # four cards
        for i in range(13):
            moves.append(Four(i, {i: 4}))
            moves.append(ThreePlusOne(i, {i: 4}))
        for i, j in itertools.permutations(range(13), 2):
            # 3 + 1
            moves.append((ThreePlusOne(i, {i: 3, j: 1})))
            # 3 + 2
            moves.append(ThreePlusTwo(i, {i: 3, j: 2}))
        for i in range(13):
            # 3 + gray joker
            moves.append(ThreePlusOne(i, {i: 3, 13: 1}))
            # 3 + color joker
            moves.append(ThreePlusOne(i, {i: 3, 14: 1}))
        # double joker
        moves.append(DoubleJoker())

    @staticmethod
    def _add_four_card_combinations(moves: List[MoveInternal]) -> None:
        # 4 + 2
        for i in range(13):
            # 4 + 1 + 1
            for j, k in itertools.combinations(itertools.chain(range(i), range(i + 1, 15)), 2):  # type: ignore
                moves.append(FourPlusTwo(i, {i: 4, j: 1, k: 1}))
            # 4 + 2
            for j in itertools.chain(range(i), range(i + 1, 13)):
                moves.append(FourPlusTwo(i, {i: 4, j: 2}))
        # 4 + 2 + 2
        for i in range(13):
            for j in itertools.combinations_with_replacement(  # type: ignore
                itertools.chain(range(i), range(i + 1, 13)),
                2
            ):
                double_pairs: Counter = Counter(j)  # type: ignore
                double_pairs.update(double_pairs)
                double_pairs.update({i: 4})
                moves.append(FourPlusTwoPairs(i, double_pairs))

    @staticmethod
    def _add_straight_combinations(moves: List[MoveInternal]) -> None:
        # straight
        for i in range(9):
            for j in range(5, 13 - i):
                moves.append(Straight(i, i + j - 1, {k: 1 for k in range(i, i + j)}))
        # double straight
        for i in range(10):
            for j in range(3, min(13 - i, 11)):
                moves.append(DoubleStraight(i, i + j - 1, {k: 2 for k in range(i, i + j)}))
        # triple straight
        for i in range(11):
            for j in range(2, min(13 - i, 7)):
                moves.append(TripleStraight(i, i + j - 1, {k: 3 for k in range(i, i + j)}))

    @staticmethod
    def _add_airplane_combinations(moves: List[MoveInternal]) -> None:
        # straight 3 + 1s
        for i in range(11):
            for j in range(2, min(13 - i, 6)):
                for k in itertools.combinations_with_replacement(range(15), j):
                    ones: Counter = Counter(k)
                    ones.update({m: 3 for m in range(i, i + j)})
                    if max(ones.values()) <= 4 and ones[13] <= 1 and ones[14] <= 1:
                        moves.append(TripleStraightPlusOnes(i, i + j - 1, ones))
        # straight 3 + 2s
        for i in range(11):
            for j in range(2, min(13 - i, 5)):
                for k in itertools.combinations_with_replacement(
                    itertools.chain(range(i), range(i + j, 13)),
                    j
                ):
                    twos: Counter = Counter(k)
                    twos.update(twos)
                    twos.update({m: 3 for m in range(i, i + j)})
                    if max(twos.values()) <= 4:
                        moves.append(TripleStraightPlusTwos(i, i + j - 1, twos))

    def recompute_moves(self) -> None:
        # pylint: disable=too-many-branches
        print("Recompute moves data for the Landlord game")
        with Manager() as manager:
            moves = manager.list()  # type: ignore
            processes: List[Process] = []
            # skip move
            moves.append(Skip())
            processes.extend((
                Process(target=Landlord._add_simple_combinations, args=(moves, )),
                Process(target=Landlord._add_four_card_combinations, args=(moves,)),
                Process(target=Landlord._add_straight_combinations, args=(moves,)),
                Process(target=Landlord._add_airplane_combinations, args=(moves,))
            ))
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
        with open(self.valid_moves_bin, "wb") as file:
            pickle.dump(self.valid_moves, file)

    def start(self) -> None:
        self.moves = []
        self.played = ([0] * 15, [0] * 15, [0] * 15)
        self.hands = Landlord._init_hands()
        self.current_role = PlayerRole.LANDLORD

    @property
    def number_possible_moves(self) -> int:
        return len(self.internal_moves)

    @staticmethod
    def hand_contains_move(hand: List[int], move: MoveInternal) -> bool:
        return all(hand[i] >= v for i, v in move.dict_form.items())

    def _compute_valid_move(self, move_index: int) -> List[int]:
        # pylint: disable=too-many-branches, too-many-locals, too-many-statements
        if not move_index:
            return list(range(len(self.internal_moves)))

        last_move: MoveInternal = self.internal_moves[move_index]
        last_move_type: MoveType = last_move.move_type

        moves = set()

        # skip turn
        moves.add(self.internal_moves_back_ref[Skip()])

        # only skip for double joker
        if last_move_type == MoveType.DOUBLE_JOKER:
            return list(moves)

        # double joker always a valid move
        moves.add(self.internal_moves_back_ref[DoubleJoker()])

        if last_move_type == MoveType.FOUR:
            last_card: int = last_move.dominant_card
            # four cards
            for i in range(last_card + 1, 13):
                moves.add(self.internal_moves_back_ref[Four(i, {i: 4})])
            return list(moves)

        # four cards always a valid move
        for i in range(13):
            moves.add(self.internal_moves_back_ref[Four(i, {i: 4})])

        if last_move_type == MoveType.SINGLE:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 15):
                moves.add(self.internal_moves_back_ref[Single(i, {i: 1})])
        elif last_move_type == MoveType.DOUBLE:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 13):
                moves.add(self.internal_moves_back_ref[Double(i, {i: 2})])
        elif last_move_type == MoveType.TRIPLE:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 13):
                moves.add(self.internal_moves_back_ref[Triple(i, {i: 3})])
        elif last_move_type == MoveType.THREE_PLUS_ONE:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 13):
                for j in range(15):
                    counter = Counter((j,))
                    counter.update({i: 3})
                    moves.add(self.internal_moves_back_ref[ThreePlusOne(i, counter)])
        elif last_move_type == MoveType.THREE_PLUS_TWO:
            last_card = last_move.dominant_card
            for i in range(last_card + 1, 13):
                for j in itertools.chain(range(last_card), range(last_card + 1, i), range(i + 1, 13)):
                    moves.add(self.internal_moves_back_ref[ThreePlusTwo(i, {i: 3, j: 2})])
        elif last_move_type == MoveType.STRAIGHT:
            start, end = last_move.range
            # last_card_end + i < 12
            for i in range(1, 12 - end):
                straight: Straight = Straight(
                    start + i,
                    end + i,
                    {j: 1 for j in range(start + i, end + i + 1)}
                )
                moves.add(self.internal_moves_back_ref[straight])
        elif last_move_type == MoveType.DOUBLE_STRAIGHT:
            start, end = last_move.range
            # last_card_end + i < 12
            for i in range(1, 12 - end):
                double_straight: DoubleStraight = DoubleStraight(
                    start + i,
                    end + i,
                    {j: 2 for j in range(start + i, end + i + 1)}
                )
                moves.add(self.internal_moves_back_ref[double_straight])
        elif last_move_type == MoveType.TRIPLE_STRAIGHT:
            start, end = last_move.range
            # last_card_start + i >= last_card_end + 1
            # last_card_end + i < 12
            for i in range(end - start + 1, 12 - end):
                triple_straight: TripleStraight = TripleStraight(
                    start + i,
                    end + i,
                    {j: 3 for j in range(start + i, end + i + 1)}
                )
                moves.add(self.internal_moves_back_ref[triple_straight])
        elif last_move_type == MoveType.TRIPLE_STRAIGHT_PLUS_ONES:
            start, end = last_move.range
            for i in range(end - start + 1, 12 - end):
                for j in itertools.combinations_with_replacement(range(15), end - start + 1):  # type: ignore
                    ones: Counter = Counter(j)  # type: ignore
                    ones.update({k: 3 for k in range(start + i, end + i + 1)})
                    if max(ones.values()) > 4 or ones[13] > 1 or ones[14] > 1:
                        continue
                    straight_ones: TripleStraightPlusOnes = TripleStraightPlusOnes(
                        start + i,
                        end + i,
                        ones
                    )
                    moves.add(self.internal_moves_back_ref[straight_ones])
        elif last_move_type == MoveType.TRIPLE_STRAIGHT_PLUS_TWOS:
            start, end = last_move.range
            for i in range(end - start + 1, 12 - end):
                for j in itertools.combinations_with_replacement(  # type: ignore
                    itertools.chain(range(start), range(end + i + 1, 13)),
                    end - start + 1
                ):
                    twos: Counter = Counter(j)  # type: ignore
                    twos.update(twos)
                    twos.update({k: 3 for k in range(start + i, end + i + 1)})
                    if max(twos.values()) > 4:
                        continue
                    straight_twos: TripleStraightPlusTwos = TripleStraightPlusTwos(
                        start + i,
                        end + i,
                        twos
                    )
                    moves.add(self.internal_moves_back_ref[straight_twos])
        elif last_move_type == MoveType.FOUR_PLUS_TWO:
            card: int = last_move.dominant_card
            for i in range(card + 1, 13):
                for j, k in itertools.combinations(itertools.chain(
                    range(card),
                    range(i + 1, 15)
                ), 2):
                    moves.add(self.internal_moves_back_ref[FourPlusTwo(i, {i: 4, j: 1, k: 1})])
                for j in itertools.chain(range(card), range(i + 1, 13)):
                    moves.add(self.internal_moves_back_ref[FourPlusTwo(i, {i: 4, j: 2})])
        elif last_move_type == MoveType.FOUR_PLUS_TWO_PAIRS:
            card_four = last_move.dominant_card
            for i in range(card_four + 1, 13):
                for j in itertools.combinations_with_replacement(  # type: ignore
                    itertools.chain(range(card_four), range(i + 1, 13)), 2
                ):
                    double_pairs: Counter = Counter(j)  # type: ignore
                    double_pairs.update(double_pairs)
                    double_pairs.update({i: 4})
                    moves.add(self.internal_moves_back_ref[FourPlusTwoPairs(i, double_pairs)])
        return list(moves)

    @property
    def available_moves(self) -> List[int]:
        hand = self.hands[self.current_role.value]
        moves = []
        if not self.moves or self.moves[-1][1] == self.current_role:
            for index, internal_move in enumerate(self.internal_moves):
                if Landlord.hand_contains_move(hand, internal_move):
                    moves.append(index)
            return moves

        last_move_index = self.moves[-1][0]
        if not self.valid_moves[last_move_index]:
            valid_moves = self._compute_valid_move(last_move_index)
            self.valid_moves[last_move_index] = valid_moves
        else:
            valid_moves = self.valid_moves[last_move_index]

        for valid_move_index in valid_moves:
            if Landlord.hand_contains_move(hand, self.internal_moves[valid_move_index]):
                moves.append(valid_move_index)
        return moves

    def make_move(self, move: int) -> None:
        if not move:
            self.current_role = self.current_role.next_role()
            return
        hand = self.hands[self.current_role.value]
        played = self.played[self.current_role.value]
        for card, count in self.internal_moves[move].dict_form.items():
            hand[card] -= count
            played[card] += count
        self.moves.append((move, self.current_role))
        self.current_role = self.current_role.next_role()

    def undo_move(self, move: int) -> None:
        self.current_role = self.current_role.prev_role()
        if not move:
            return
        hand = self.hands[self.current_role.value]
        played = self.played[self.current_role.value]
        for card, count in self.internal_moves[move].dict_form.items():
            hand[card] += count
            played[card] -= count
        self.moves.pop()

    @property
    def state_dimension(self) -> Tuple[int, int]:
        return 5, 16

    @property
    def game_state(self) -> Tensor:
        state: Tensor = torch.tensor((
            *self.played,
            self.hands[self.current_role.value]
        ))
        if not self.moves or self.moves[-1][1] == self.current_role:
            cards_on_field: Tensor = torch.zeros(15, dtype=torch.int64)
        else:
            cards_on_field = torch.tensor(self.internal_moves[self.moves[-1][0]].tuple_form)
        state = torch.vstack((state, cards_on_field.unsqueeze(0)))

        role_col: Tensor = torch.zeros(state.size()[0], dtype=torch.int64)
        role_col[self.current_role.value] = 1

        return torch.hstack((state, role_col.unsqueeze(1))).to(device=self.torch_device, dtype=torch.float)

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
        return self.current_role == PlayerRole.LANDLORD

    def save_valid_moves(self) -> None:
        with open(self.valid_moves_bin, "wb") as file:
            pickle.dump(self.valid_moves, file)
