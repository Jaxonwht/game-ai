from typing import Dict, Union

import torch
import torch.nn as nn
import numpy as np

from config.config import Config
from game_definition.game import Game


class StateNode:
    # pylint: disable=too-few-public-methods
    def __init__(self, state: np.ndarray, p_v_tuple: torch.Tensor) -> None:
        self.probability = p_v_tuple[:-1]
        self.value = p_v_tuple[-1]
        self.state = state
        self.visit_count = 0
        self.children_visit_count = 0
        self.value_sum = torch.tensor(0, dtype=torch.float32)
        self.children: Dict[int, StateNode] = {}


class MCTSController:
    def __init__(self, game: Game, model: nn.Module, config: Config, device: torch.device) -> None:
        self.game: Game = game
        self.model = model
        self.config = config
        self.device = device
        self.root = StateNode(game.game_state, self._predict(game.game_state))

    @property
    def empirical_probability(self) -> torch.Tensor:
        probability = torch.zeros(self.game.number_possible_moves, dtype=torch.int64)
        probability[torch.tensor(tuple(self.root.children.keys()))] = torch.tensor(
            tuple(x.visit_count for x in self.root.children.values()), dtype=torch.int64
        )
        return probability / probability.sum()

    def confirm_move(self, move: int) -> None:
        self.root = self.root.children[move]

    def simulate(self, count: int) -> None:
        for _ in range(count):
            self._search(self.root, self.config.mcts_depth_cap)

    def _predict(self, state: np.ndarray) -> torch.Tensor:
        return self.model(
            torch.from_numpy(np.expand_dims(state, (0, 1))).float().to(self.device)
        ).squeeze(0).detach().cpu()

    def _search(self, node: StateNode, depth_cap: int) -> Union[int, torch.Tensor]:
        if self.game.over:
            node.visit_count += 1
            node.value_sum += self.game.score
            return self.game.score

        if node.visit_count == 0 and depth_cap <= 0:
            node.value_sum += node.value
            node.visit_count = 1
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
                child_node_val = torch.tensor(0, dtype=torch.float32)
            else:
                child_node_val = child_node.value_sum if desire_positive_score else -child_node.value_sum
                child_node_val /= child_node.visit_count
            child_u = (
                child_node_val
                + self.config.explore_constant
                * node.probability[move]
                * torch.sqrt(torch.tensor(node.children_visit_count)) / (1 + child_node.visit_count)
            )
            if child_u > max_u:
                max_u, best_move = child_u, move

        self.game.make_move(best_move)
        next_val = self._search(node.children[best_move], depth_cap - 1)
        self.game.undo_move(best_move)

        node.value_sum += next_val
        node.children_visit_count += 1
        node.visit_count += 1
        return next_val
