import numpy as np
from scipy.sparse import csr_matrix


def any_abs_sum_is_3(*vals):
    return any(abs(val) == 3 for val in vals)


class TicTacToe(object):
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.int8)

    def episode_complete(self):
        return any_abs_sum_is_3(
            self.board.sum(axis=0),
            self.board.sum(axis=1),
            self.board.diagonal().sum(),
            np.rot90(self.board).diagonal().sum()
        )

    def interact(self, action):
        reward = 0
        if self.episode_complete():
            return self.state_rep(), reward
        value, x, y = action
        self.place(value, x, y)
        if self.episode_complete():
            reward = 1
        return self.state_rep(), reward

    def state_rep(self):
        return tuple(tuple(row) for row in self.board)

    def place(self, value, x, y):
        self.board[y, x] = value

    def __repr__(self):
        return self.empty_row_str().join(self.row_as_str(row) for row in self.board)

    def row_as_str(self, row):
        repr_map = {-1: 'O', 0: ' ', 1: 'X'}
        return ' | '.join(repr_map[x] for x in row) + '\n'

    def empty_row_str(self):
        return '-- --- --\n'


if __name__ == '__main__':
    ttt = TicTacToe()
    ttt.board[2, 2] = -1
    ttt.board[0, 2] = 1
    print(ttt)