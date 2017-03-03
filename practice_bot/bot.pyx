import numpy as np


def train_bot(double[:] pr, int[:] mm, int[:] ln, int[:] prev,
  double alpha, double gamma, double ep):
  cdef:
    double[:] book = [[0, 0], [0, 0], [0, 0], [0, 0]]
    double[:] avf = np.ones[2, 10, 11, 3, 9] * 10
    double[:] state = [mm[0], ln[0], b_to_ind(book), 0]
    double[:] newstate = np.empty_like(state)
    int i = 0
    int mode = 0
    int action
    double reward
  for i in range(len(pr) - 1):
    sort_book(book, mode)
    newstate, action, reward = (
      transition(state, pr, mm, ln, book, mode, avf, i, ep)
    )
    update(avf, state, newstate, action, reward, alpha, gamma)
    state = newstate[:]
  return avf


cdef update(double[:] avf, double[:] state, double[:] newstate,
  int action, double reward, double alpha, double gamma):
  ud = alpha * (reward + gamma * maxq(avf, newstate))
  ud -= alpha * avf[state[0], state[1], state[2], action]
  avf[state[0], state[1], state[2], action] += ud


cdef maxq(double[:] avf, double[:] state):
  cdef:
    int[:] ava = get_ava(book, mode)
    double[:] avflist = np.empty_like(ava)
  for i in range(len(ava)):
    avflist[i] = get_avf(avf, state, ava[i])
  return ava[np.argmax(avflist)]


def transiton(double[:] state, double[:] pr, double[:] mm, double[:] ln,
      double[:] book, double[:] avf, double[:] navf, int mode, int i, double ep):
  cdef:
    double reward = 0
    double[3] newstate
    int[:] ava = get_ava(book, mode)
    double[:] avflist = np.empty_like(ava)
    int i = 0
    int bestaction
    int action
  for i in range(len(ava)):
    avflist[i] = get_avf(avf, state, ava[i])
  bestaction = ava[np.argmax(avflist)]
  if np.random.choice([1, 0], p=[1-ep*(len(ava)-1),ep*(len(ava)-1)]):
    action = bestaction
  else:
    action = np.random.choice(ava)
  reward = update_book(book, action, pr, mode, i)
  newstate[3] = mode
  newstate[2] = b_to_ind(book)
  if action > 8:
    newstate[1] = ln[i]
    newstate[0] = mm[i]
  else:
    newstate[1] = ln[i + 1]
    newstate[0] = mm[i + 1]
  return newstate, action, reward


cdef inline get_avf(avf, state, action):
  return avf[state[0], state[1], state[2], action]


cdef get_ava(book, mode):
  cdef:
     int i = 0
     int pot_out = 0
     int n_pos = 0
  for i in range(len(book)):
    if book[i][0] != 0:
      n_pos += 1
    pot_out += book[i][0]
  if n_pos == 0:
    return [0, 1, 2, 3, 4, 5, 6, 7, 8]
  elif mode == 1:
    if n_pos == 1:
      if pos_out == 1:
        return [0, 1, 2, 3, 9]
      elif pos_out == 2:
        return [0, 1, 2, 9]
      elif pos_out == 3:
        return [0, 1, 9]
      elif pos_out == 4:
        return [0, 9]
    elif n_pos == 2:
      if pos_out == 2:
        return [0, 1, 2, 9, 10]
      elif pos_out == 3:
        return [0, 1, 9, 10]
      else:
        return [0, 9, 10]
    elif n_pos == 3:
      if pos_out == 3:
        return [0, 1, 9, 10, 11]
      else:
        return [0, 9, 10, 11]
    else:
      return [0, 9, 10, 11, 12]
  elif mode == 2:
    if n_pos == 1:
      if pos_out == 1:
        return [0, 5, 6, 7, 9]
      elif pos_out == 2:
        return [0, 5, 6, 9]
      elif pos_out == 3:
        return [0, 5, 9]
      elif pos_out == 4:
        return [0, 9]
    elif n_pos == 2:
      if pos_out == 2:
        return [0, 5, 6, 9, 10]
      elif pos_out == 3:
        return [0, 5, 9, 10]
      else:
        return [0, 9, 10]
    elif n_pos == 3:
      if pos_out == 3:
        return [0, 5, 9, 10, 11]
      else:
        return [0, 9, 10, 11]
    else:
      return [0, 9, 10, 11, 12]


cdef inline long_rew(double[:] book, double pr, int i, int mode):
  if mode == 1:
    r =  1 - book[i][0]*book[i][1]/pr
    book[i] = [0, 0]
    return r
  r = book[i][0]*book[i][1]/pr - 1
  book[i] = [0, 0]
  return r


cdef update_book(double[:] book, int mode, int action, double pr):
  if action == 0:
    return 0
  elif action < 5:
    book[3] = [action, pr]
    mode = 1
    return 0
  elif action < 9:
    book[3] = [action - 4, pr]
    mode = 2
  else:
    r = long_rew(book, pr, action - 5, mode)
    cdef int i = 0, n_pos = 0
    for i in range(4):
      if book[i][0] != 0:
        n_pos += 1
    if n_pos == 0:
      mode = 0
    return r


cdef b_to_ind(book):
  cdef:
     int i = 0
     int pot_out = 0
     int n_pos = 0
  for i in range(len(book)):
    if book[i][0] != 0:
      n_pos += 1
    pot_out += book[i][0]
  if n_pos == 0:
    return 0
  elif n_pos == 1:
    return pot_out
  elif n_pos == 2:
    return 3 + pot_out
  elif n_pos == 3:
    return 5 + pot_out
  else:
    return 10
