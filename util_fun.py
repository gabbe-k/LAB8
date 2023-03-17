import numpy as np 




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




def has_won_np(board, win_list):

  b_flat = board.flatten()

  for w in win_list:
    elems = b_flat[w]

    if all(elems == 1):
        return -1
    elif all(elems == 2):
        return 1

  if np.count_nonzero(b_flat) == b_flat.size:
    return 0.5

  return 0




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

    if all(elems == 1):
        return 1
    elif all(elems == 2):
        return 2

  if np.count_nonzero(b_flat) == 9:
    return 0

  return -1




def has_won_universal(board,win_list):

  b_flat = board.board.flatten()

  for w in win_list:
    elems = b_flat[w]

    if all(elems == 1):
        return 1
    elif all(elems == 2):
        return 2

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
