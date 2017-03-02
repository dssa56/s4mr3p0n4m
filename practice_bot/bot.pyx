import numpy as np


def train_bot(double[:] pr, int[:] mm, int[:] ln, int[:] prev, struct p):
  cdef:
    double[:] book = [[0, 0], [0, 0], [0, 0], [0, 0]]
    double[:] avf = np.ones[2, 4, 2, 30, 20] * 10
    double[:] state = [mm[0], prev[0], ln[0], b_to_ind(book)]
    int i = 0
    int mode = 0
  for i in range(len(pr) - 1):
    sort_book(book, mode)
    newstate, mode, action, reward = (
      transition(state, pr, mm, ln, book, mode, avf, i, p)
    )
    avf = update(avf, state, action, reward)
    state = newstate[:]
  return avf


def transiton(double[:] pr, double[:] mm, double[:] ln, double[:] book,
          double[:] avf, double[:] navf, int i, struct p):
  cdef:
     double reward = 0
     double[4] newstate
     int[:] ava = get_ava(book)
     double[:] avflist = np.empty_like(ava)
     int i = 0
     int bestaction
     int action
  for i in range(len(ava)):
    avflist[i] = get_avf(state, ava[i])
  bestaction = ava[np.argmax(avflist)]
  if np.random.choice([1, 0], p=[1-p.ep*(len(ava)-1),p.ep*(len(ava)-1)]):
    action = bestaction
  else:
    action = np.random.choice(ava)
  reward = update_book(book, action, pr, i)
  newstate[3] = b_to_ind(book)
  newstate[2] = ln[i + 1]
  newstate[1] = prev[i + 1]
  newstate[0] = mm[i + 1]
  return newstate, action, reward


cdef update_book(book, mode, action, pr, i):
  if action == 0:
    return 0
  elif action < 5:
    book[4] = [action, pr[i]]
    return 0
  else:
    if mode == 1:
      if action < 9:
        reward = 1 - book[action-4][0]*book[action-4][1]/pr[i]
        book[action - 4] = [0, 0]
        return reward
      elif action == 9:
        reward = 1 - book[action-4][0]*book[action-4][1]/pr[i]
        book[action - 4] = [0, 0]
        reward += 1 - book[action-3][0]*book[action-3][1]/pr[i]
        book[action - 3] = [0, 0]
        return reward
