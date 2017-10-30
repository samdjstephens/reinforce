import random
from collections import defaultdict
from typing import NamedTuple, Tuple, Any, Dict
from functools import partial

import numpy as np


def any_abs_sum_is_3(*vals):
    return any((abs(val) == 3).any() for val in vals)


Position = NamedTuple('Position', [('x', int), ('y', int)])
TTTAction = NamedTuple('TTTAction', [('marker', int), ('position', Position)])



class TicTacToe(object):
    def __init__(self, opponent_agent=None):
        self.board = self.init_board()
        self.opponent_agent: SarsaAgent = opponent_agent

    def init_board(self):
        return np.zeros((3, 3), dtype=np.int8)

    def episode_complete(self):
        return any_abs_sum_is_3(
            self.board.sum(axis=0),
            self.board.sum(axis=1),
            self.board.diagonal().sum(),
            np.rot90(self.board).diagonal().sum()
        ) or ((self.board != 0).all())

    def interact(self, action: TTTAction) -> int:
        reward = 0
        if self.episode_complete():
            return reward

        # Take agents action
        self.place(action.marker, action.position)
        if self.episode_complete():
            reward = 1
        else:
            # Take opponents action
            opp_action = self.opponent_agent.act(self.state_rep())
            self.place(opp_action.marker, opp_action.position)
            if self.episode_complete():
                reward = -1

        return reward

    def state_rep(self):
        return self.board.copy()

    def place(self, marker: int, pos: Position):
        self.board[pos.y, pos.x] = marker

    def reset(self):
        self.board = self.init_board()

    def __repr__(self):
        return self.empty_row_str().join(self.row_as_str(row) for row in self.board)

    def row_as_str(self, row):
        return ' | '.join(self.repr_marker(x) for x in row) + '\n'

    def repr_marker(self, marker):
        repr_map = {-1: 'O', 0: ' ', 1: 'X'}
        return repr_map[marker]

    def empty_row_str(self):
        return '-- --- --\n'


class SarsaAgent(object):
    def __init__(self,
                 marker: int=1,
                 epsilon: float=0.1,
                 gamma: float=0.9,
                 alpha: float=0.9,
                 lambda_: float=0.9,
                 random_seed: int=10) -> None:
        self.marker = marker
        self.eligibility: Dict[Tuple, float] = defaultdict(float)
        self.q_values: Dict[Tuple, Dict[TTTAction, float]] = defaultdict(dict)  # {state1: {action1: value}, state2: {action2: value}}
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self._random = random.Random(random_seed)

    def act(self, state: np.ndarray) -> TTTAction:
        y, x = self.choose_egreedy(state)
        return TTTAction(self.marker, Position(x, y))

    def get_feedback(self,
                     state: np.ndarray,
                     action: TTTAction,
                     reward: int,
                     new_state: np.ndarray,
                     new_action: TTTAction) -> None:
        td_error = (reward
                    + self.gamma * self.get_action_value(new_state, new_action)
                    - self.get_action_value(state, action))
        self.eligibility[(self.hashable_state(state), action)] += 1
        self.update_q_values(td_error)
        self.update_eligibility_traces()

    def update_eligibility_traces(self) -> None:
        new_eligibility: Dict[Tuple, float] = defaultdict(float)
        for eligibilitykey, eligibilityvalue in self.eligibility.items():
            new_eligibility[eligibilitykey] = self.gamma * self.lambda_ * eligibilityvalue
        self.eligibility = new_eligibility

    def update_q_values(self, td_error: float) -> None:
        new_qvals: Dict[Tuple, Dict[TTTAction, float]] = defaultdict(dict)
        for statekey, action_value_lookup in self.q_values.items():
            for actionkey, actionvalue in action_value_lookup.items():
                new_qvals[statekey][actionkey] = (
                    actionvalue + self.alpha * td_error * self.eligibility[(statekey, actionkey)])
        self.q_values = new_qvals

    def reset(self):
        self.eligibility.clear()

    def choose_egreedy(self, state: np.ndarray) -> TTTAction:
        possible_actions = [TTTAction(self.marker, Position(x, y))
                            for y, x in np.argwhere(state == 0)]
        if self.should_explore():
            # TODO What if state is full? This will raise an exception
            return self._random.choice(possible_actions)
        else:
            return max(possible_actions, key=partial(self.get_action_value,
                                                     state))

    def get_action_value(self, state: np.ndarray, action: TTTAction) -> int:
        hstate = self.hashable_state(state)
        return self.q_values.get(hstate, {}).get(action, 0)

    def hashable_state(self, state: np.ndarray) -> Tuple[Tuple[int]]:
        return tuple(state.flatten())

    def should_explore(self) -> bool:
        return self._random.random() > self.epsilon



def play_episodes(n=100):
    agent = SarsaAgent(1)
    opponent = SarsaAgent(-1)
    env = TicTacToe(opponent)
    for _ in range(n):
        play_episode(agent, env)


def play_episode(agent: SarsaAgent, env: TicTacToe):
    state = env.state_rep()
    action = agent.act(state)
    for turn in range(1, 18):  # Should be done in 9
        reward = env.interact(action)
        new_state = env.state_rep()
        # print(f"######### Turn {turn} ###########\n")
        # print(env)
        if not env.episode_complete():
            new_action = agent.act(new_state)
            agent.get_feedback(state, action, reward, new_state, new_action)
            state = new_state
            action = new_action
        else:
            break

    print(f"Episode complete. {turn} turns taken.")
    print(env, end='\n\n\n\n')
    env.reset()
    agent.reset()


if __name__ == '__main__':
    # TODO: Something broken - its not picking anything in the top row
    play_episodes(100)
