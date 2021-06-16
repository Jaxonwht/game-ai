from typing import Dict, Tuple
from math import sqrt

import torch

from game_definitions.game import Game
from model.model import Model

class StateNode:
    def __init__(self, state: torch.Tensor, p_v_tuple: Tuple[torch.Tensor, float]) -> None:
        self.p, self.v = p_v_tuple
        self.state: torch.Tensor = state
        self.visit_count: int = 0
        self.value_sum: float = 0
        self.children: Dict[int, StateNode] = {}

class MCTSController:
    def __init__(self, game: Game, model: Model) -> None:
        self.game: Game = game
        self.root = StateNode(game.game_state, model.predict(game.game_state))
        self.model = model

    @property
    def empirical_p(self) -> torch.Tensor:
        p = torch.empty(self.game.number_possible_moves)
        for move, child_node in self.root.children.items():
            p[move] = child_node.visit_count / self.root.visit_count
        return p

    @property
    def computed_pv(self) -> Tuple[torch.Tensor, float]:
        return self.root.p, self.root.v

    def simulate(self, count: int) -> None:
        for _ in range(count):
            self.search(self.root)

    def search(self, node: StateNode) -> float:
        if self.game.over:
            return self.game.score

        if node.visit_count == 0:
            node.value_sum += node.v
            node.visit_count += 1
            return node.v

        desire_positive_score: bool = self.game.desire_positive_score

        max_u, best_move = -float("inf"), -1
        for move in self.game.available_moves:
            if move in node.children:
                child_node = node.children[move]
            else:
                self.game.make_move(move)
                child_node = StateNode(self.game.game_state, self.model.predict(self.game.game_state))
                node.children[move] = child_node
                self.game.undo_move(move)
            child_node_val = child_node.v if desire_positive_score else -child_node.v
            child_u = child_node_val + float(node.p[move].item()) * sqrt(node.visit_count) / (1 + child_node.visit_count)
            if child_u > max_u:
                max_u, best_move = child_u, move

        self.game.make_move(best_move)
        next_val = self.search(node.children[best_move])
        self.game.undo_move(best_move)

        node.value_sum += next_val
        node.visit_count += 1
        return next_val
