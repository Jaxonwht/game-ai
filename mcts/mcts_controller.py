from typing import Dict, Union

import torch
import torch.nn as nn
import numpy as np

from game_definition.game import Game


class StateNode:
    # pylint: disable=too-few-public-methods
    def __init__(self, state: np.ndarray, p_v_tuple: np.ndarray) -> None:
        self.probability = p_v_tuple[:-1]
        self.value = p_v_tuple[-1]
        self.state = state
        self.visit_count = 0
        self.children_visit_count = 0
        self.value_sum: float = 0
        self.children: Dict[int, StateNode] = {}


class MCTSController:
    def __init__(self, game: Game, model: nn.Module) -> None:
        self.game: Game = game
        self.model = model
        self.root = StateNode(game.game_state, self._predict(game.game_state))

    @property
    def empirical_probability(self) -> np.ndarray:
        probability = np.zeros(self.game.number_possible_moves, dtype=int)
        probability[list(self.root.children.keys())] = tuple(x.visit_count for x in self.root.children.values())
        return np.divide(probability, np.sum(probability), dtype=np.float32)

    def confirm_move(self, move: int) -> None:
        self.root = self.root.children[move]

    def simulate(self, count: int) -> None:
        for _ in range(count):
            self._search(self.root)

    def _predict(self, state: np.ndarray) -> np.ndarray:
        return self.model(torch.from_numpy(np.expand_dims(state, (0, 1))).float()).squeeze(0).numpy()

    def _search(self, node: StateNode) -> Union[int, float]:
        if self.game.over:
            node.visit_count += 1
            return self.game.score

        if node.visit_count == 0:
            node.value_sum += node.value
            node.visit_count += 1
            return node.value

        desire_positive_score = self.game.desire_positive_score

        max_u, best_move = -float("inf"), -1

        if not node.children:
            for move in self.game.available_moves:
                self.game.make_move(move)
                child_node = StateNode(self.game.game_state, self._predict(self.game.game_state))
                node.children[move] = child_node
                self.game.undo_move(move)

        for move, child_node in node.children.items():
            if not child_node.visit_count:
                child_node_val: float = 0
            else:
                child_node_val = child_node.value_sum if desire_positive_score else -child_node.value_sum
                child_node_val /= child_node.visit_count
            child_u = (
                child_node_val
                + node.probability[move] * np.sqrt(node.children_visit_count) / (1 + child_node.visit_count)
            )
            if child_u > max_u:
                max_u, best_move = child_u, move

        self.game.make_move(best_move)
        next_val = self._search(node.children[best_move])
        self.game.undo_move(best_move)

        node.value_sum += next_val
        node.children_visit_count += 1
        node.visit_count += 1
        return next_val
