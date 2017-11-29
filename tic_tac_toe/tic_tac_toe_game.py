import itertools
from typing import NamedTuple

import numpy as np


Position = NamedTuple('Position', [('x', int), ('y', int)])
TTTAction = NamedTuple('TTTAction', [('marker', int), ('position', Position)])
TTTState = NamedTuple('TTTState', [('board', np.ndarray), ('is_terminal', bool)])


def any_abs_sum_is_3(*vals):
    vals = [a.tolist() if isinstance(a, np.ndarray) else [a] for a in vals]
    return any(abs(val) == 3 for val in itertools.chain(*vals))


def winner(*vals):
    vals = [a.tolist() if isinstance(a, np.ndarray) else [a] for a in vals]
    max_ = max(itertools.chain(*vals), key=lambda val: abs(val))
    return max_ // 3 if abs(max_) == 3 else None


class TicTacToeGame(object):
    def __init__(self):
        self.board = self.init_board()

    def init_board(self):
        return np.zeros((3, 3), dtype=np.int8)

    def is_game_over(self):
        has_winner = any_abs_sum_is_3(
            self.board.sum(axis=0),
            self.board.sum(axis=1),
            self.board.diagonal().sum(),
            np.rot90(self.board).diagonal().sum()
        )
        is_draw = (self.board != 0).all()
        return has_winner or is_draw

    def place(self, marker: int, pos: Position):
        self.board[pos.y, pos.x] = marker

    def winner(self):
        return winner(
            self.board.sum(axis=0),
            self.board.sum(axis=1),
            self.board.diagonal().sum(),
            np.rot90(self.board).diagonal().sum()
        )

    def available_positions(self):
        return [Position(x, y) for y, x in np.argwhere(self.board == 0)]

    def reset(self):
        self.board = self.init_board()

    def __repr__(self):
        return self._repr(self.board)

    def _repr(self, board):
        return self.empty_row_str().join(self.row_as_str(row) for row in board)

    def row_as_str(self, row):
        return ' | '.join(self.repr_marker(x) for x in row) + '\n'

    def repr_marker(self, marker):
        repr_map = {-1: 'O', 0: ' ', 1: 'X'}
        return repr_map[marker]

    def empty_row_str(self):
        return '-- --- --\n'

    def _repr_flat(self, flat):
        return self._repr(np.array(flat).reshape(3, 3))
