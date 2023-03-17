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

def deepQVsMCTS(mctsagent, agent1, size=(n,n), verbose=False):
    b = te.Board(size,n)
    b.turn = 1
    firstRound = True
    res = -1

    while res == -1:

        if firstRound:
            moves = b.possible_moves()
            i = np.random.randint(len(moves))
            x,y = moves[i]
            b.push([x,y])
            firstRound = False
        else:
            move = mctsagent.search(b)
            b.push(move)

        if has_won(b) != -1: 
            res = has_won(b)
            break

        move_Q = agent1.choose_action(b.board.flatten())
        b.push(acpdict[move_Q])

        if has_won(b) != -1: 
            res = has_won(b)
            break
    
    if verbose: 
        print(b)
        print(res)
    return res


def deepQVsRANDOM(agent1, size=(n,n), verbose=False):
    b = te.Board(size,n)
    b.turn = 1
    res = -1

    while res == -1:

        moves = b.possible_moves()
        i = np.random.randint(len(moves))
        x,y = moves[i]
        b.push([x,y])

        if has_won(b) != -1: 
            res = has_won(b)
            break

        move_Q = agent1.choose_action(b.board.flatten())
        b.push(acpdict[move_Q])

        if has_won(b) != -1: 
            res = has_won(b)
            break
    
    if verbose: 
        print(b)
        print(res)
    return res


if __name__ == '__main__':
    #load Q-learned model
    network = network_ttt.DeepQNetwork(lr = 0.001, input_dims = [n*n], fc1_dims= 256, fc2_dims = 256, n_actions = n*n)
    
    
    #59% draw rate, 16% against mcst 150 iter
    network.load_state_dict(t.load('models/ONE.pt'))
    network.eval()

    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size=512, 
                  n_actions=n*n, eps_end=0.01, input_dims=[n*n], lr=0.001, istraining=False)
    agent.Q_eval = network

    n_iter = 1000

    mcts = MCTS(1,n_iter=n_iter)

    #play 100 games and count the number of wins by the DeepQ model
    wins = 0
    draws = 0
    for i in range(100):
      #outcome = deepQVsMCTS(mcts, agent, size=(n,n), verbose=True)
      outcome = deepQVsRANDOM(agent, size=(n,n), verbose=True)
      if outcome == 2:
          wins += 1
      elif outcome == 0:
          draws += 1
      
      if i % 10 == 0:
        print("Game: ", i, ", Wins + draws: ", wins + draws) 
    
    print("Win + draw rate of DeepQ model: ", wins/100, "Draw rate: ", draws/100)
    print("Win rate of MCTS model: ", 1 - wins/100 - draws/100)
