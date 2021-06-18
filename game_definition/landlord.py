import random
import itertools
from enum import Enum
from collections import Counter
from typing import Iterable, Dict, Tuple, List

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
    def __init__(self) -> None:
        self.internal_moves: List[MoveInternal] = []
        self.internal_moves_back_ref: Dict[MoveInternal, int] = {}
        self._init_moves()

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

    def _add_move(self, move: MoveInternal) -> None:
        index: int = len(self.internal_moves)
        self.internal_moves.append(move)
        self.internal_moves_back_ref[move] = index

    def _init_moves(self) -> None:
        # skip move
        self._add_move(Skip())
        # single card
        for i in range(15):
            self._add_move(Single(i, {i : 1}))
        # double cards
        for i in range(13):
            self._add_move(Double(i, {i: 2}))
        # triple cards
        for i in range(13):
            self._add_move(Triple(i, {i : 3}))
        # four cards
        for i in range(13):
            self._add_move(Four(i, {i: 4}))
        for i, j in itertools.permutations(range(13), 2):
            # 3 + 1
            self._add_move(ThreePlusOne(i, {i: 3, j: 1}))
            # 3 + 2
            self._add_move(ThreePlusTwo(i, {i: 3, j: 2}))
        for i in range(13):
            # 3 + gray joker
            self._add_move(ThreePlusOne(i, {i: 3, 13: 1}))
            # 3 + color joker
            self._add_move(ThreePlusOne(i, {i: 3, 14: 1}))
        # straight
        for i in range(9):
            for j in range(5, 13 - i):
                self._add_move(Straight(i, i + j - 1, {k : 1 for k in range(i, i + j)}))
        # double straight
        for i in range(10):
            for j in range(3, min(13 - i, 11)):
                self._add_move(DoubleStraight(i, i + j - 1, {k: 2 for k in range(i, i + j)}))
        # triple straight
        for i in range(11):
            for j in range(2, min(13 - i, 7)):
                self._add_move(TripleStraight(i, i + j - 1, {k: 3 for k in range(i, i + j)}))
        # straight 3 + 1s
        for i in range(11):
            for j in range(2, min(13 - i, 6)):
                for k in itertools.combinations_with_replacement(range(15), j):
                    ones: Counter = Counter(k)
                    ones.update({m : 3 for m in range(i, i + j)})
                    if max(ones.values()) <= 4 and ones[13] <= 1 and ones[14] <= 1:
                        self._add_move(TripleStraightPlusOnes(
                            i,
                            i + j - 1,
                            ones
                        ))
        # straight 3 + 2s
        for i in range(11):
            for j in range(2, min(13 - i, 5)):
                for k in itertools.combinations_with_replacement(
                    itertools.chain(range(i), range(i + j, 13)),
                    j
                ):
                    twos: Counter = Counter(k)
                    twos.update(twos)
                    twos.update({m : 3 for m in range(i, i + j)})
                    if max(twos.values()) <= 4:
                        self._add_move(TripleStraightPlusTwos(
                            i,
                            i + j - 1,
                            twos
                        ))
        # 4 + 2
        for i in range(13):
            # 4 + 1 + 1
            for j, k in itertools.combinations(itertools.chain(range(i), range(i + 1, 15)), 2):
                self._add_move(FourPlusTwo(i, {i: 4, j: 1, k: 1}))
            # 4 + 2
            for j in itertools.chain(range(i), range(i + 1, 13)):
                self._add_move(FourPlusTwo(i, {i: 4, j: 2}))
        # 4 + 2 + 2
        for i in range(13):
            for j in itertools.combinations_with_replacement(
                itertools.chain(range(i), range(i + 1, 13)),
                2
            ):
                double_pairs: Counter = Counter(j)
                double_pairs.update(double_pairs)
                double_pairs.update({i : 4})
                self._add_move(FourPlusTwoPairs(i, double_pairs))
        # double joker
        self._add_move(DoubleJoker())

    def start(self) -> None:
        self.moves: List[Tuple[int, PlayerRole]] = []
        self.played: Tuple[List[int], List[int], List[int]] = ([0] * 15, [0] * 15, [0] * 15)
        self.hands: Tuple[List[int], List[int], List[int]] = Landlord._init_hands()
        self.current_role: PlayerRole = PlayerRole.LANDLORD

    @property
    def number_possible_moves(self) -> int:
        return len(self.internal_moves)

    @staticmethod
    def hand_contains_move(hand: List[int], move: MoveInternal) -> bool:
        return all(hand[i] >= v for i, v in move.dict_form.items())

    @property
    def available_moves(self) -> List[int]:
        hand: List[int] = self.hands[self.current_role.value]
        moves: List[int] = []
        if not self.moves or self.moves[-1][1] == self.current_role:
            for index, internal_move in enumerate(self.internal_moves):
                if Landlord.hand_contains_move(hand, internal_move):
                    moves.append(index)
            return moves

        last_move_index: int = self.moves[-1][0]
        last_move: MoveInternal = self.internal_moves[last_move_index]
        last_move_type: MoveType = last_move.move_type

        # skip turn
        moves.append(0)

        # only skip for double joker
        if last_move_type == MoveType.DOUBLE_JOKER:
            return moves

        # double joker always a valid move
        double_joker: MoveInternal = DoubleJoker()
        if Landlord.hand_contains_move(hand, double_joker):
            moves.append(self.internal_moves_back_ref[double_joker])

        if last_move_type == MoveType.FOUR:
            last_card: int = last_move.dominant_card
            # four cards
            for i in range(last_card + 1, 13):
                four: Four = Four(i, {i: 4})
                if Landlord.hand_contains_move(hand, four):
                    moves.append(self.internal_moves_back_ref[four])
            return moves

        # four cards always a valid move
        for i in range(13):
            four: Four = Four(i, {i: 4})
            if Landlord.hand_contains_move(hand, four):
                moves.append(self.internal_moves_back_ref[four])

        if last_move_type == MoveType.SINGLE:
            last_card: int = last_move.dominant_card
            for i in range(last_card + 1, 15):
                single: Single = Single(i, {i: 1})
                if Landlord.hand_contains_move(hand, single):
                    moves.append(self.internal_moves_back_ref[single])
        elif last_move_type == MoveType.DOUBLE:
            last_card: int = last_move.dominant_card
            for i in range(last_card + 1, 13):
                double: Double = Double(i, {i: 2})
                if Landlord.hand_contains_move(hand, double):
                    moves.append(self.internal_moves_back_ref[double])
        elif last_move_type == MoveType.TRIPLE:
            last_card: int = last_move.dominant_card
            for i in range(last_card + 1, 13):
                triple: Triple = Triple(i, {i: 3})
                if Landlord.hand_contains_move(hand, triple):
                    moves.append(self.internal_moves_back_ref[triple])
        elif last_move_type == MoveType.THREE_PLUS_ONE:
            last_card: int = last_move.dominant_card
            for i in range(last_card + 1, 13):
                for j in range(15):
                    three_plus_one: ThreePlusOne = ThreePlusOne(i, {i: 3, j: 1})
                    if Landlord.hand_contains_move(hand, three_plus_one):
                        moves.append(self.internal_moves_back_ref[three_plus_one])
        elif last_move_type == MoveType.THREE_PLUS_TWO:
            last_card: int = last_move.dominant_card
            for i in range(last_card + 1, 13):
                for j in itertools.chain(range(last_card), range(last_card + 1, 13)):
                    three_plus_two: ThreePlusTwo = ThreePlusTwo(i, {i: 3, j :2})
                    if Landlord.hand_contains_move(hand, three_plus_two):
                        moves.append(self.internal_moves_back_ref[three_plus_two])
        elif last_move_type == MoveType.STRAIGHT:
            start, end = last_move.range
            # last_card_end + i < 12
            for i in range(1, 12 - end):
                straight: Straight = Straight(
                    start + i,
                    end + i,
                    {j: 1 for j in range(start + i, end + i + 1)}
                )
                if Landlord.hand_contains_move(hand, straight):
                    moves.append(self.internal_moves_back_ref[straight])
        elif last_move_type == MoveType.DOUBLE_STRAIGHT:
            start, end = last_move.range
            # last_card_end + i < 12
            for i in range(1, 12 - end):
                double_straight: DoubleStraight = DoubleStraight(
                    start + i,
                    end + i,
                    {j: 2 for j in range(start + i, end + i + 1)}
                )
                if Landlord.hand_contains_move(hand, double_straight):
                    moves.append(self.internal_moves_back_ref[double_straight])
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
                if Landlord.hand_contains_move(hand, triple_straight):
                    moves.append(self.internal_moves_back_ref[triple_straight])
        elif last_move_type == MoveType.TRIPLE_STRAIGHT_PLUS_ONES:
            start, end = last_move.range
            for i in range(end - start + 1, 12 - end):
                for j in itertools.combinations_with_replacement(range(15), end - start + 1):
                    ones: Counter = Counter(j)
                    ones.update({k: 3 for k in range(start + i, end + i + 1)})
                    if max(ones.values()) > 4 or ones[13] > 1 or ones[14] > 1:
                        continue
                    straight_ones: TripleStraightPlusOnes = TripleStraightPlusOnes(
                        start + i,
                        end + i,
                        ones
                    )
                    if Landlord.hand_contains_move(hand, straight_ones):
                        moves.append(self.internal_moves_back_ref[straight_ones])
        elif last_move_type == MoveType.TRIPLE_STRAIGHT_PLUS_TWOS:
            start, end = last_move.range
            for i in range(end - start + 1, 12 - end):
                for j in itertools.combinations_with_replacement(
                    itertools.chain(range(start), range(end + i + 1, 13)),
                    end - start + 1
                ):
                    twos: Counter = Counter(j)
                    twos.update(twos)
                    twos.update({k: 3 for k in range(start + i, end + i + 1)})
                    if max(twos.values()) > 4:
                        continue
                    straight_twos: TripleStraightPlusTwos = TripleStraightPlusTwos(
                        start + i,
                        end + i,
                        twos
                    )
                    if Landlord.hand_contains_move(hand, straight_twos):
                        moves.append(self.internal_moves_back_ref[straight_twos])
        elif last_move_type == MoveType.FOUR_PLUS_TWO:
            card: int = last_move.dominant_card
            for i in range(card + 1, 13):
                for j, k in itertools.combinations(itertools.chain(
                        range(card),
                        range(i + 1, 15)
                    ),
                2):
                    four_plus_two: FourPlusTwo = FourPlusTwo(i, {i: 4, j: 1, k: 1})
                    if Landlord.hand_contains_move(hand, four_plus_two):
                        moves.append(self.internal_moves_back_ref[four_plus_two])
                for j in itertools.chain(range(card), range(i + 1, 13)):
                    four_plus_pair: FourPlusTwo = FourPlusTwo(i, {i: 4, j: 2})
                    if Landlord.hand_contains_move(hand, four_plus_pair):
                        moves.append(self.internal_moves_back_ref[four_plus_pair])
        elif last_move_type == MoveType.FOUR_PLUS_TWO_PAIRS:
            card: int = last_move.dominant_card
            for i in range(card + 1, 13):
                for j in itertools.combinations_with_replacement(
                    itertools.chain(range(card), range(i + 1, 13)), 2
                ):
                    double_pairs: Counter = Counter(j)
                    double_pairs.update(double_pairs)
                    double_pairs.update({i: 4})
                    four_plus_two_pairs: FourPlusTwoPairs = FourPlusTwoPairs(
                        i, double_pairs
                    )
                    if Landlord.hand_contains_move(hand, four_plus_two_pairs):
                        moves.append(self.internal_moves_back_ref[four_plus_two_pairs])
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
    def game_state(self) -> Tensor:
        state: Tensor = torch.tensor((
            *self.played,
            self.hands[self.current_role.value]
        ))
        if not self.moves or self.moves[-1][1] == self.current_role:
            cards_on_field: Tensor = torch.zeros(15, dtype=torch.int64)
        else:
            cards_on_field: Tensor = torch.tensor(self.internal_moves[self.moves[-1][0]].tuple_form)
        state = torch.stack((state, cards_on_field))

        role_col: Tensor = torch.zeros(state.size()[0], dtype=torch.int64)
        role_col[self.current_role.value] = 1

        return torch.hstack((state, role_col.unsqueeze(1)))

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
