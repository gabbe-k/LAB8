import gym
from gym import spaces
import tictactoe as te 
import numpy as np 
from mcts_ttt import MCTS

def create_win_list(n):
    win_list = []
    # Rows
    for i in range(n):
        for j in range(n - 2):
            win_list.append(list(range(i*n + j, i*n + j + 3)))
    # Columns
    for i in range(n - 2):
        for j in range(n):
            win_list.append(list(range(i*n + j, (i + 3)*n + j, n)))
    # Diagonal 1
    for i in range(n - 2):
        for j in range(n - 2):
            win_list.append(list(range(i*n + j, (i + 3)*n + j + 3, n + 1)))
    # Diagonal 2
    for i in range(n - 2):
        for j in range(2, n):
            win_list.append(list(range(i*n + j, (i + 3)*n + j - 3, n - 1)))
    return win_list



def has_won(board,win_list):

  b_flat = board.board.flatten()

  for w in win_list:
    elems = b_flat[w]

    if all(elems == 1): return -1
    elif all(elems == 2): return 1 

  if np.count_nonzero(b_flat) == b_flat.size:
    return 0.5 

  return 0




def action_to_play_dict(n):
    d = {}
    for i in range(n*n):
        d[i] = (i//n, i%n)
    return d






class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n = 3):
        super(TicTacToeEnv, self).__init__()

        self.size = n
        self.ttt = te.Board((n,n),n)

        # 9 placements on the board
        self.action_space = spaces.Discrete(n*n)

        # three markers (including unmarked), nine spaces
        #self.observation_space = spaces.MultiBinary(n*n*3) 

        self.curr_player = 2

        self.reward_range = (-1, 1)

        # 9 moves per game
        self.ep_len = n*n 

        # index of current move
        self.ep_index = 0

        self.win_list = create_win_list(n)

        # convert 0 to 8 to (0,0) to (2,2) (3x3 board)
        self.action_to_play = action_to_play_dict(n)

        self.ill_counter = 0
        
        self.mcts = MCTS(p=1,n_iter=150)

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
        illegal = False

        if self.ttt.get_mark_at_position(play) != 0: illegal = True

        self.ttt.set_mark(play, self.curr_player)

        result = has_won(self.ttt, self.win_list)

        # Return immediately if illegal move
        if illegal:
            reward = -0.2
            return self.ttt.board.flatten(), reward, True, {}

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

        