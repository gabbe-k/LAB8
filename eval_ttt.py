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

n = 3 
win_list = create_win_list(n)
acpdict = action_to_play_dict(n)
    

def deepQVsMCTS(mctsagent, agent1,  size=(n,n), iter=1000):
  b = te.Board(size,n)

  while has_won(b) == -1:

    print(b)

    move = mctsagent.search(b)
    b.push(move)

    print(b)

    if has_won(b) != -1:
        return has_won(b)
    
    move_Q = agent1.choose_action(b.board.flatten())

    print("Deep Q move:", acpdict[move_Q])
    b.push(acpdict[move_Q])

    if has_won(b) != -1:
        return has_won(b)
     


if __name__ == '__main__':
    #load Q-learned model
    network = network_ttt.DeepQNetwork(lr = 0.001, input_dims = [n*n], fc1_dims= 256, fc2_dims = 256, n_actions = n*n)
    network.load_state_dict(t.load('ttt_model.pt'))
    network.eval()

    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size=64, 
                  n_actions=n*n, eps_end=0.01, input_dims=[n*n], lr=0.001, istraining=False)
    agent.Q_eval = network

    mcts = MCTS(2,n_iter=1000)

    deepQVsMCTS(mcts, agent, size=(n,n), iter=1000)






