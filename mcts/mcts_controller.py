from typing import Dict
from math import sqrt

import numpy as np

from game_definition.game import Game
from model.model import Model


class StateNode:
    # pylint: disable=too-few-public-methods
    def __init__(self, state: np.ndarray, p_v_tuple: np.ndarray) -> None:
        self.probability = p_v_tuple[:-1]
        self.value = p_v_tuple[-1]
        self.state = state
        self.visit_count: int = 0
        self.value_sum: float = 0
        self.children: Dict[int, StateNode] = {}


class MCTSController:
    def __init__(self, game: Game, model: Model) -> None:
        self.game: Game = game
        self.root = StateNode(game.game_state, model.predict(game.game_state))
        self.model = model

    @property
    def empirical_probability(self) -> np.ndarray:
        probability = np.zeros(self.game.number_possible_moves, dtype=int)
        for move, child_node in self.root.children.items():
            probability[move] = child_node.visit_count
        return probability / np.sum(probability)

    def confirm_move(self, move: int) -> None:
        self.root = self.root.children[move]

    def simulate(self, count: int, search_depth_cap: int) -> None:
        for _ in range(count):
            self._search(self.root, search_depth_cap)

    def _search(self, node: StateNode, search_depth_cap: int) -> float:
        if self.game.over:
            return self.game.score

        if node.visit_count == 0 or search_depth_cap == 0:
            node.value_sum += node.value
            node.visit_count += 1
            return node.value

        desire_positive_score: bool = self.game.desire_positive_score

        max_u, best_move = -float("inf"), -1

        if not node.children:
            for move in self.game.available_moves:
                self.game.make_move(move)
                child_node = StateNode(self.game.game_state, self.model.predict(self.game.game_state))
                node.children[move] = child_node
                self.game.undo_move(move)

        for move, child_node in node.children.items():
            child_node_val = child_node.value if desire_positive_score else -child_node.value
            child_u = (
                child_node_val
                + float(node.probability[move].item()) * sqrt(node.visit_count) / (1 + child_node.visit_count)
            )
            if child_u > max_u:
                max_u, best_move = child_u, move

        self.game.make_move(best_move)
        next_val = self._search(node.children[best_move], search_depth_cap - 1)
        self.game.undo_move(best_move)

        node.value_sum += next_val
        node.visit_count += 1
        return next_val
