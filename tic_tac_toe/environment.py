import random
from functools import partial

import numpy as np


def any_abs_sum_is_3(*vals):
    return any(abs(val) == 3 for val in vals)


class TicTacToe(object):
    def __init__(self, opponent_agent=None):
        self.board = self.init_board()
        self.opponent_agent = opponent_agent

    def init_board(self):
        return np.zeros((3, 3), dtype=np.int8)

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

        # Take agents action
        self.place(*action)
        if self.episode_complete():
            reward = 1

        # Take opponents action
        opp_action = self.opponent_agent.act(self.state_rep())
        self.place(*opp_action)
        if self.episode_complete():
            reward = -1

        return reward

    def state_rep(self):
        return self.board

    def place(self, value, x, y):
        self.board[y, x] = value

    def reset(self):
        self.board = self.init_board()

    def __repr__(self):
        return self.empty_row_str().join(self.row_as_str(row) for row in self.board)

    def row_as_str(self, row):
        repr_map = {-1: 'O', 0: ' ', 1: 'X'}
        return ' | '.join(repr_map[x] for x in row) + '\n'

    def empty_row_str(self):
        return '-- --- --\n'


class Agent(object):
    def __init__(self, marker='X', epsilon=0.9, random_seed=10):
        self.marker = marker
        self.eligibility = {}
        self.q_values = {}  # {state1: {action1: value}, state2: {action2: value}}
        self.epsilon = epsilon
        self._random = random.Random(random_seed)

    def act(self, state):
        self.choose_egreedy(state)
        return self.marker, 0, 0

    def get_feedback(self, state, action, reward, new_state):
        pass

    def reset(self):
        self.eligibility.clear()

    def choose_egreedy(self, state):
        possible_actions = np.argwhere(state == 0)
        if self.should_explore():
            # TODO What if state is full? This will raise an exception
            return self._random.choice(possible_actions)
        else:
            return max(possible_actions, key=partial(self.get_action_value, state))

    def get_action_value(self, state, action):
        hstate = self.hashable_state(state)
        self.q_values.get(hstate, {}).get(action, 0)

    def hashable_state(self, state):
        return tuple(tuple(row) for row in state)

    def should_explore(self):
        return self._random.random() > self.epsilon



def play_episodes():
    agent = Agent('X')
    opponent = Agent('O')
    env = TicTacToe(opponent)
    for _ in range(10):
        play_episode(agent, env)


def play_episode(agent, env):
    for _ in range(100000):
        state = env.state_rep()
        action = agent.act(state)
        reward = env.interact(action)
        new_state = env.state_rep()

        agent.get_feedback(state, action, reward, new_state)
        if env.episode_complete():
            break
    env.reset()
    agent.reset()


if __name__ == '__main__':
    ttt = TicTacToe()
    ttt.board[2, 2] = -1
    ttt.board[0, 2] = 1
    print(ttt)