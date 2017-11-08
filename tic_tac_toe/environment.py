import random
import itertools

from collections import defaultdict
from typing import NamedTuple, Tuple, Dict, Callable
from functools import partial

import numpy as np



def any_abs_sum_is_3(*vals):
    vals = [a.tolist() if isinstance(a, np.ndarray) else [a] for a in vals]
    return any(abs(val) == 3 for val in itertools.chain(*vals))

def winner(*vals):
    vals = [a.tolist() if isinstance(a, np.ndarray) else [a] for a in vals]
    max_ = max(itertools.chain(*vals), key=lambda val: abs(val))
    return max_ // 3 if abs(max_) == 3 else None


Position = NamedTuple('Position', [('x', int), ('y', int)])
TTTAction = NamedTuple('TTTAction', [('marker', int), ('position', Position)])

QVALType = Dict[Tuple, Dict[TTTAction, float]]


class TicTacToe(object):
    def __init__(self, opponent_agent=None):
        self.board = self.init_board()
        self.opponent_agent: SarsaAgent = opponent_agent

    def init_board(self):
        return np.zeros((3, 3), dtype=np.int8)

    @staticmethod
    def episode_complete(state):
        has_winner = any_abs_sum_is_3(
            state.sum(axis=0),
            state.sum(axis=1),
            state.diagonal().sum(),
            np.rot90(state).diagonal().sum()
        )
        is_draw = (state != 0).all()
        return has_winner or is_draw

    @staticmethod
    def winner(state):
        return winner(
            state.sum(axis=0),
            state.sum(axis=1),
            state.diagonal().sum(),
            np.rot90(state).diagonal().sum()
        )

    def interact(self, action: TTTAction) -> int:
        reward = 0
        if self.episode_complete(self.state_rep()):
            return reward

        # Take agents action
        self.place(action.marker, action.position)
        if self.episode_complete(self.state_rep()):
            reward = 1
        else:
            # Take opponents action
            self.opponent_play()
            if self.episode_complete(self.state_rep()):
                reward = -1

        return reward

    def opponent_play(self):
        opp_action = self.opponent_agent.act(self.state_rep())
        self.place(opp_action.marker, opp_action.position)

    def state_rep(self):
        return self.board.copy()

    def place(self, marker: int, pos: Position):
        self.board[pos.y, pos.x] = marker

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


class SarsaAgent(object):
    def __init__(self,
                 marker: int=1,
                 epsilon: float=0.1,
                 gamma: float=0.9,
                 alpha: float=0.9,
                 lambda_: float=0.9,
                 random_seed: int=10) -> None:
        # TODO: Check all parameters are sensible
        self.marker = marker
        self.eligibility: Dict[Tuple, float] = defaultdict(float)
        # TODO: Analyse action values - pick out ones which are clearly 'good' because
        # they are 1 away from winning
        self.q_values: QVALType = defaultdict(dict)  # {state1: {action1: value}, state2: {action2: value}}
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self._random = random.Random(random_seed)

    def act(self, state: np.ndarray) -> TTTAction:
        if TicTacToe.episode_complete(state):
            return TTTAction(self.marker, Position(-1, -1))  # Sentinel
        action = self.choose_egreedy(state)
        hashable_state = self.hashable_state(state)
        if action not in self.q_values.get(hashable_state, {}):
            self.q_values[hashable_state][action] = 0.0
        return action

    def get_feedback(self,
                     state: np.ndarray,
                     action: TTTAction,
                     reward: int,
                     new_state: np.ndarray,
                     new_action: TTTAction) -> None:
        # TODO: Check td_error calc
        td_error = (reward
                    + self.gamma * self.get_action_value(new_state, new_action)
                    - self.get_action_value(state, action))
        # TODO: supposed to update eligibility for last state?
        self.eligibility[(self.hashable_state(state), action)] += 1.0
        self.update_q_values(td_error)
        self.update_eligibility_traces()

    def update_eligibility_traces(self) -> None:
        # TODO: Unittest
        new_eligibility: Dict[Tuple, float] = defaultdict(float)
        for eligibilitykey, eligibilityvalue in self.eligibility.items():
            new_eligibility[eligibilitykey] = self.gamma * self.lambda_ * eligibilityvalue
        self.eligibility = new_eligibility

    def update_q_values(self, td_error: float) -> None:
        # TODO: Unittest
        new_qvals: QVALType = defaultdict(dict)
        for statekey, action_value_map in self.q_values.items():
            for actionkey, actionvalue in action_value_map.items():
                new_qvals[statekey][actionkey] = (
                    actionvalue + self.alpha * td_error * self.eligibility[(statekey, actionkey)])
        self.q_values = new_qvals

    def reset(self):
        self.eligibility.clear()

    def choose_egreedy(self, state: np.ndarray) -> TTTAction:
        # TODO: Unittest with known action values
        possible_actions = [TTTAction(self.marker, Position(x, y))
                            for y, x in np.argwhere(state == 0)]
        if self.should_explore():
            choice = self._random.choice(possible_actions)
        else:
            self._random.shuffle(possible_actions)
            choice = self.exploitative_action(possible_actions, state)
        return choice

    def exploitative_action(self, possible_actions, state):
        # TODO: Unittest with known action values
        choice = max(possible_actions, key=partial(self.get_action_value,
                                                   state))
        return choice

    def get_action_value(self, state: np.ndarray, action: TTTAction) -> int:
        if TicTacToe.episode_complete(state):
            return 0
        hstate = self.hashable_state(state)
        return self.q_values[hstate].get(action, 0)

    def hashable_state(self, state: np.ndarray) -> Tuple[int]:
        return tuple(state.flatten())

    def should_explore(self) -> bool:
        return self._random.random() < self.epsilon


def noop_print(text, end='\n'):
    pass


class TestBed(object):
    def __init__(self, logger: Callable=noop_print):
        self.winners = []
        self.logger = logger
        self.agent = SarsaAgent(1)
        self.opponent = SarsaAgent(-1)
        self.env = TicTacToe(self.opponent)

    def play_episodes(self, n=100):
        for _ in range(n):
            winner = self.play_episode()
            self.winners.append(winner)


    def play_episode(self):
        if self.agent._random.random() < 0.5:
            # Let the opponent start 50% of the time
            self.env.opponent_play()

        state = self.env.state_rep()
        action = self.agent.act(state)

        for turn in range(1, 18):  # Should be done in 9
            reward = self.env.interact(action)
            new_state = self.env.state_rep()
            # print(f"######### Turn {turn} ###########\n")
            # print(self.env)

            new_action = self.agent.act(new_state)
            self.agent.get_feedback(state, action, reward, new_state, new_action)
            state = new_state
            action = new_action
            if self.env.episode_complete(state):
                break
        self.logger(f"Episode complete. {turn} turns taken.", end=' ')
        result = self.env.winner(state)
        if result:
            self.logger(f"{self.env.repr_marker(result)} won!")
        else:
            self.logger("It was a draw")
        self.logger(self.env, end='\n\n\n\n')
        self.reset()
        return result

    def reset(self):
        self.env.reset()
        self.agent.reset()
        self.opponent.reset()

if __name__ == '__main__':
    test_bed = TestBed()
    test_bed.play_episodes(1000)