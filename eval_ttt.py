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
    

def deepQVsMCTS(mctsagent, agent1,  size=(n,n)):
  b = te.Board(size,n)

  #moves = b.possible_moves()
  #i = np.random.randint(len(moves))
  #x,y = moves[i]

  #b.set_mark([x,y], 1)

  firstRound = True

  while has_won(b) == -1:


    if firstRound:
      moves = b.possible_moves()
      i = np.random.randint(len(moves))
      x,y = moves[i]
      b.set_mark([x,y], 1)
      firstRound = False
       
    else:
      print(b)
      
      move = mctsagent.search(b)
      b.set_mark(move, 1)

      print("MCTS move:")
      print(b) 

    if has_won(b) != -1:
        print(b)
        return has_won(b)
    
    move_Q = agent1.choose_action(b.board.flatten())

    print("Deep Q move:", acpdict[move_Q])
    b.set_mark(acpdict[move_Q], 2)

    if has_won(b) != -1:
        print(b)
        return has_won(b)

  
     


if __name__ == '__main__':
    #load Q-learned model
    network = network_ttt.DeepQNetwork(lr = 0.001, input_dims = [n*n], fc1_dims= 256, fc2_dims = 256, n_actions = n*n)
    network.load_state_dict(t.load('ttt_model.pt'))
    network.eval()

    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size=64, 
                  n_actions=n*n, eps_end=0.01, input_dims=[n*n], lr=0.001, istraining=False)
    agent.Q_eval = network

    n_iter = 200

    mcts = MCTS(1,n_iter=n_iter)

    print(deepQVsMCTS(mcts, agent, size=(n,n)))






