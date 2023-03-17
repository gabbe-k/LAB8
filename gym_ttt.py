import gym
from gym import spaces
import tictactoe as te 
import numpy as np 
from mcts_ttt import MCTS
from util_fun import *



def action_to_play_dict(n):
    d = {}
    for i in range(n*n):
        d[i] = (i//n, i%n)
    return d




class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n = 3, n_iter = 150):
        super(TicTacToeEnv, self).__init__()

        self.size = n
        self.ttt = te.Board((n,n),n)

        # 9 placements on the board
        self.action_space = spaces.Discrete(n*n)

        self.curr_player = 2

        self.reward_range = (-1, 1)

        # 9 moves per game
        self.ep_len = n*n 

        # index of current move
        self.ep_index = 0

        self.win_list = create_win_list(n)

        # convert 0 to 8 to (0,0) to (2,2) (3x3 board)
        self.action_to_play = action_to_play_dict(n)

        self.mcts = MCTS(p=1,n_iter=n_iter)

    def reset(self, seed=None, options=None):
        # maybe we want to place a random opponent marker first? 
        # in this case, it should be seeded
        super().reset(seed = seed)

        n = self.size
        self.ttt = te.Board((n,n),n)

        #random play from player 1
        self._next_observation()

        self.curr_player = 2
        self.ep_index = 0
        return self.ttt.board.flatten()
    
    def step(self, action, i):

        self.ep_index += 1
        play = self.action_to_play[action]

        if self.ttt.get_mark_at_position(play) != 0: return ValueError("Illegal move")

        self.ttt.set_mark(play, self.curr_player)

        result = has_won(self.ttt, self.win_list)

        if result != 0:
            self.ep_index = self.ep_len
            obs = self.ttt.board
            reward = result
            return obs.flatten(), reward, True, {}
        else:
            self.ep_index += 1
            obs = self._next_observation()
            reward = has_won(self.ttt, self.win_list)
            return obs.flatten(), reward, False, {}
    
    def _next_observation(self):
        #self.ttt.set_mark(self.random_move(), 1)
        self.ttt.set_mark(self.mcts.search(self.ttt), 1)
        return self.ttt.board
    
    def random_move(self):
        plays = self.ttt.possible_moves()
        a,b = plays[np.random.randint(0, len(plays))]
        return [a,b]

        
