import random
from typing import Callable

from tic_tac_toe.environment import Environment
from tic_tac_toe.tic_tac_toe_game import TicTacToeGame
from tic_tac_toe.sarsa_agent import SarsaAgent


def noop_print(text, end='\n'):
    pass


class TestBed(object):
    def __init__(self, logger: Callable=noop_print, random_seed: int = 54):
        self.winners = []
        self.logger = logger
        self.agent = SarsaAgent(1)
        self.opponent = SarsaAgent(-1)
        self.game = TicTacToeGame()
        self.env = Environment(self.game, self.opponent)
        self._random = random.Random(random_seed)

    def play_episodes(self, n=100):
        print('Episode: ')
        for i in range(n):
            if i % 100 == 0:
                print(f'{i+1}', end=' ')
            winner = self.play_episode()
            self.winners.append(winner)

    def play_episode(self):
        if self._random.random() < 0.5:
            # Let the opponent start 50% of the time
            self.env.opponent_play()

        state = self.env.state_rep()
        action = self.agent.act(state)

        for turn in range(1, 10):  # Should be done in 5
            reward = self.env.interact(action)
            new_state = self.env.state_rep()

            new_action = self.agent.act(new_state)
            self.agent.get_feedback(state, action, reward, new_state, new_action)
            state = new_state
            action = new_action
            if self.env.is_episode_complete:
                break

        self.logger(f"Episode complete. {turn} turns taken.", end=' ')

        result = self.game.winner()

        if result:
            self.logger(f"{self.game.repr_marker(result)} won!")
        else:
            self.logger("It was a draw")
        self.logger(self.env, end='\n\n\n\n')
        self.reset()
        return result

    def reset(self):
        self.game.reset()
        self.agent.reset()
        self.opponent.reset()