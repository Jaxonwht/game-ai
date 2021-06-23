from typing import Dict

import torch

from game_definition.game import Game
from model.model import Model


class StateNode:
    # pylint: disable=too-few-public-methods
    def __init__(self, state: torch.Tensor, p_v_tuple: torch.Tensor) -> None:
        self.probability = p_v_tuple[:-1]
        self.value = p_v_tuple[-1]
        self.state = state
        self.visit_count = torch.tensor(0, dtype=torch.int)
        self.value_sum = torch.tensor(0, dtype=torch.float)
        self.children: Dict[int, StateNode] = {}


class MCTSController:
    def __init__(self, game: Game, model: Model) -> None:
        self.game: Game = game
        self.root = StateNode(game.game_state, model.predict(game.game_state))
        self.model = model

    @property
    def empirical_probability(self) -> torch.Tensor:
        probability = torch.zeros(self.game.number_possible_moves, dtype=torch.float)
        for move, child_node in self.root.children.items():
            probability[move] = child_node.visit_count
        return probability

    def confirm_move(self, move: int) -> None:
        self.root = self.root.children[move]

    def simulate(self, count: int) -> None:
        for _ in range(count):
            self._search(self.root)

    def _search(self, node: StateNode) -> torch.Tensor:
        if self.game.over:
            node.visit_count += 1
            return torch.tensor(self.game.score)

        if node.visit_count == 0:
            node.value_sum += node.value
            node.visit_count += 1
            return node.value

        desire_positive_score: bool = self.game.desire_positive_score

        max_u, best_move = torch.tensor(-float("inf")), -1

        if not node.children:
            for move in self.game.available_moves:
                self.game.make_move(move)
                child_node = StateNode(self.game.game_state, self.model.predict(self.game.game_state))
                node.children[move] = child_node
                self.game.undo_move(move)

        for move, child_node in node.children.items():
            child_node_val = child_node.value_sum if desire_positive_score else -child_node.value_sum
            child_u = (
                child_node_val
                + node.probability[move] * torch.sqrt(node.visit_count) / (1 + child_node.visit_count)
            )
            if child_u > max_u:
                max_u, best_move = child_u, move

        self.game.make_move(best_move)
        next_val = self._search(node.children[best_move])
        self.game.undo_move(best_move)

        node.value_sum += next_val
        node.visit_count += 1
        return next_val
