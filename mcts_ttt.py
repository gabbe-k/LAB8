import numpy as np
import matplotlib.pyplot as plt
import math
import random
import tictactoe as te

n = 3

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

win_list = create_win_list(n)


def has_won(board):

  b_flat = board.board.flatten()

  win = [
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7]
  ]

  for w in win:
    elems = b_flat[w]

    if all(elems == 1): return 1
    elif all(elems == 2): return 2 

  if np.count_nonzero(b_flat) == 9:
    return 0 

  return -1

def has_won(board):

  b_flat = board.board.flatten()

  for w in win_list:
    elems = b_flat[w]

    if all(elems == 1): return 1
    elif all(elems == 2): return 2 

  if np.count_nonzero(b_flat) == 9:
    return 0 

  return -1



def hashmul(board):
  b = board.board.flatten()
  coeff = np.arange(len(b))
  #pairwise multiply and sum
  return np.sum(np.multiply(b, coeff))

def hashmul_np(npboard):
  b = npboard.flatten()
  coeff = np.arange(len(b))
  #pairwise multiply and sum
  return np.sum(np.multiply(b, coeff))




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
    b.push([snd,fst])
    if has_won(b) != -1:
      return has_won(b)
    print(b)
    mcts = MCTS(2, n_iter=iter)
    move = mcts.search(b)
    b.push(move)

  print("Winner:", has_won(b))  
  print(b)

def randomGame(size=(n,n), iter=1000, verbose=False):
  b = te.Board(size,n)

  while has_won(b) == -1:

    moves = b.possible_moves()
    i = np.random.randint(len(moves))
    b.push(moves[i])

    if has_won(b) != -1:
      return has_won(b)
    
    mcts = MCTS(2, n_iter=iter)

    move = mcts.search(b)
    b.push(move)

  return has_won(b)



def main():  

  #predict a move
  b = te.Board((n,n),n)
  b.push([1,1])
  mcts = MCTS(2, n_iter=1000)
  move = mcts.search(b)
  print(move)


#main loop
if __name__ == "__main__":
  
  main()
