import gym
import network_ttt
from network_ttt import Agent
from gym_ttt import TicTacToeEnv
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from main_ttt import *
import tictactoe as te
from gym_ttt import *
from mcts_ttt import *

def has_won(board, win_list):
    b_flat = board.board.flatten()
    for w in win_list:
        elems = b_flat[w]
        if all(elems == 1):
            return 1
        elif all(elems == 2):
            return 2
    if np.count_nonzero(b_flat) == b_flat.size:
        return 0
    
    return -1

n = 3
win_list = create_win_list(n)
acpdict = action_to_play_dict(n)

def deepQVsMCTS(mctsagent, agent1, size=(n,n), verbose=False):
    b = te.Board(size,n)
    firstRound = True
    
    while has_won(b, win_list) == -1:
        
        if verbose: print(b)

        if firstRound:
            moves = b.possible_moves()
            i = np.random.randint(len(moves))
            i2 = np.random.randint(len(moves))
            x,y = moves[i]
            b.set_mark([x,y], 1)
            firstRound = False
        else:
            move = mctsagent.search(b)
            b.set_mark(move, 1)

        if has_won(b, win_list) != -1: return has_won(b, win_list)

        move_Q = agent1.choose_action(b.board.flatten())

        b.set_mark(acpdict[move_Q], 2)

        if has_won(b, win_list) != -1: return has_won(b, win_list)


def MCTSvsdeepQ(mctsagent, agent1, size=(n,n), verbose=False):
    b = te.Board(size,n)
    firstRound = True
    
    while has_won(b, win_list) == -1:
        
        if verbose: print(b)

        if firstRound:
            moves = b.possible_moves()
            i = np.random.randint(len(moves))
            x,y = moves[i]
            b.set_mark([x,y], 2)
            firstRound = False
        else:
            move_Q = agent1.choose_action(b.board.flatten())
            b.set_mark(acpdict[move_Q], 2)

        if has_won(b, win_list) != -1: return has_won(b, win_list)

        move = mctsagent.search(b)
        b.set_mark(move, 1)

        if has_won(b, win_list) != -1: return has_won(b, win_list)


if __name__ == '__main__':
    #load Q-learned model
    network = network_ttt.DeepQNetwork(lr = 0.001, input_dims = [n*n], fc1_dims= 128, fc2_dims = 128, n_actions = n*n)
    network.load_state_dict(t.load('ttt_model_MCTStrain0932000.pt'))
    network.eval()

    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size=64, 
                  n_actions=n*n, eps_end=0.01, input_dims=[n*n], lr=0.001, istraining=False)
    agent.Q_eval = network

    n_iter = 500

    mcts = MCTS(1,n_iter=n_iter)

    #play 100 games and count the number of wins by the DeepQ model
    wins = 0
    draws = 0
    for i in range(100):
      #outcome = deepQVsMCTS(mcts, agent, size=(n,n), verbose=False)
      outcome = MCTSvsdeepQ(mcts, agent, size=(n,n), verbose=True)
      if outcome == 2:
          wins += 1
      elif outcome == 0:
          draws += 1
      
      if i % 10 == 0:
        print("Game: ", i, ", Wins + draws: ", wins + draws) 
    
    print("Win + draw rate of DeepQ model: ", wins/100, "Draw rate: ", draws/100)
    print("Win rate of MCTS model: ", 1 - wins/100 - draws/100)
