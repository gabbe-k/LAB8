import numpy as np
import math
import random
import tictactoe as te
from util_fun import *

n = 3
win_list = create_win_list(n)



class Node:
  def __init__(self, board, parent, p, move_done):
    self.board = board
    self.status = has_won(self.board)
    self.parent = parent
    self.N = 0
    self.Q = 0
    self.children = {}

    self.isleaf = self.status >= 0
    self.explored = self.isleaf
    self.move_done = move_done
    self.p = p

    def __lt__(self, other):
      return self.Q < other.Q
    
    
    
    

class MCTS:
  def __init__(self, p, n_iter=1000, c_coeff=2):
    self.p = p
    self.c_coeff = c_coeff
    self.n_iter = n_iter
    

  def search(self, initial_state):
    self.root = Node(initial_state, None, self.p, None)

    for i in range(self.n_iter):
      node = self.select(self.root)
      Q = self.simulate_game(node.board)
      self.backprop(node, Q)

    move = self.get_move(self.root, c=0).move_done
    return move 


  def select(self, node):
      while not node.isleaf:
          node = self.get_move(node, self.c_coeff) if node.explored else self.expand(node)
      return node
  
  
  def expand(self, node):
    possible_moves = node.board.possible_moves()

    for a,b in possible_moves:
      new_board = node.board.copy()

      new_board.set_mark([a,b], node.p)
  
      hash = hashmul(new_board)
      
      if hash not in node.children:
        new_node = Node(new_board, node, 3 - node.p, [a,b])
        node.children[hash] = new_node

        if len(possible_moves) == len(node.children):
          node.explored = True
        
        return new_node

    raise ValueError


  def simulate_game(self, board):
      board = board.copy()
      p = self.p
      
      while (status := has_won(board)) == -1:
                a, b = random.choice(list(board.possible_moves()))
                board.set_mark([a, b], p)
                p = 3 - p

      return 1 if status == self.p else 0 if status == 0 else -1


  def backprop(self, node, Q):
      if node is None:
          return
      node.N += 1
      node.Q += Q
      self.backprop(node.parent, Q)
  
  
  def uct(self, curr_p, node, child_node, c):
    move_Q = curr_p * child_node.Q / child_node.N + c * math.sqrt(math.log(node.N / child_node.N)) 
    return move_Q


  def get_move(self, node, c):
      player_mul = 1 if node.p == self.p else -1

      return np.random.choice([
          child_node for child_node in node.children.values()
          if self.uct(player_mul, node, child_node, c) == max([
              self.uct(player_mul, node, child_node, c)
              for child_node in node.children.values()
          ])
      ])




def getplay():
  s = input().split(",")
  s = ([int(val) for val in s])
  return s




def playGame(iter=3000, verbose=False):
  b = te.Board((n,n),n)

  while has_won(b) == -1:
    print(b)
    print("'row,col'")
    fst,snd = getplay()
    b.set_mark([snd,fst],2)
    if has_won(b) != -1:
      return has_won(b)
    print(b)
    mcts = MCTS(1, n_iter=iter)
    move = mcts.search(b)
    b.set_mark(move,1)

  print("Winner:", has_won(b))  
  print(b)




def randomGame(size=(n,n), iter=1000, verbose=False):
  b = te.Board(size,n)

  while has_won(b) == -1:

    mcts = MCTS(1, n_iter=iter)

    move = mcts.search(b)
    b.push(move)

    #print(b)

    if has_won(b) != -1:
      return has_won(b)
    
    moves = b.possible_moves()
    i = np.random.randint(len(moves))
    b.push(moves[i])

  return has_won(b)



def main():  
  #predict a move
  b = te.Board((n,n),n)
  b.push([1,1])
  mcts = MCTS(p=2, n_iter=1000)
  move = mcts.search(b)


#main loop
if __name__ == "__main__":
  print(randomGame(iter=1000))
  main()
