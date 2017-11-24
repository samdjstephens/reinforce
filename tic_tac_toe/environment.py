from tic_tac_toe.sarsa_agent import SarsaAgent
from tic_tac_toe.tic_tac_toe_game import TicTacToeGame, TTTAction, TTTState


class Environment(object):
    def __init__(self, game: TicTacToeGame, opponent_agent: SarsaAgent = None) -> None:
        self.game = game
        self.opponent_agent = opponent_agent

    def interact(self, action: TTTAction) -> int:
        reward = 0
        if self.game.is_game_over():
            return reward

        # Take agents action
        self.game.place(action.marker, action.position)

        if self.game.is_game_over() and self.game.winner() is not None:
            reward = 1
        elif self.game.is_game_over():
            reward = 0
        else:
            # Take opponents action
            self.opponent_play()
            if self.game.is_game_over() and self.game.winner() is not None:
                reward = -1
            elif self.game.is_game_over():
                reward = 0

        return reward

    def opponent_play(self) -> None:
        opp_action = self.opponent_agent.act(self.state_rep())
        self.game.place(opp_action.marker, opp_action.position)

    def state_rep(self) -> TTTState:
        return TTTState(self.game.board.copy(), self.is_episode_complete)

    @property
    def is_episode_complete(self) -> bool:
        return self.game.is_game_over()
