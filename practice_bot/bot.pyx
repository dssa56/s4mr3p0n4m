import numpy as np


def train_bot(double[:] pr, int[:] mm, int[:] ln,
  double alpha, double gamma, double ep, int n_epochs):
  cdef:
    double[:,::1] book = np.zeros([4, 2], order='C')
    double[:,:,:,:,::1] avf = np.ones([2, 10, 11, 3, 13], order='C') * 10
    int[:] state = np.array([mm[0], ln[0], b_to_ind(book), 0], dtype=np.int32)
    int[:] newstate = np.empty_like(state)
    int i = 0
    int action
    double reward
    double spread = 0.00015
  for _ in range(n_epochs):
    for i in range(len(pr) - 1):
      sort_book(book, state, pr[i])
      newstate, action, reward = (
        transition(state, pr, mm, ln, book, avf, i, ep, spread)
      )
      update(avf, state, newstate, book, action, reward, alpha, gamma)
      state = newstate[:]
  return avf


def implament_policy(double[:] pr, int[:] mm, int[:] ln,
  double[:,:,:,:,::1] avf):
  cdef:
    double[:,::1] book = np.zeros([4, 2], order='C')
    int[:] state = np.array([mm[0], ln[0], b_to_ind(book), 0], dtype=np.int32)
    int[:] newstate = np.empty_like(state)
    double reward
    double[:] rewards = np.empty([len(pr) - 1])
    int[:] actions = np.empty([len(pr) - 1], dtype=np.int32)
  for i in range(len(pr) - 1):
    sort_book(book, state, pr[i])
    newstate, action, reward = (
     transition(state, pr, mm, ln, book, avf, i, 0, 0.00015)
    )
    actions[i] = action
    rewards[i] = reward
    state = newstate[:]
    return np.asarray(actions), np.asarray(rewards)


def test_sort_book():
  cdef:
    double[:,::1] book = np.array([[1, 0.02], [2, 0.002], [0, 0], [0, 0]])
    int[:] state = np.array([1, 5, b_to_ind(book), 2], dtype=np.int32)
    double pr = 0.3
  sort_book(book, state, pr)
  return np.asarray(book)


cdef sort_book(double[:,::1] book, int[:] state, double pr):
  cdef:
    int i = 0
    int j = 0
    double[4][2] phdbook
  l = np.empty([4])
  for i in range(4):
    if book[i][0] != 0:
      l[i] = -long_rew(book, pr, i, state[3], 0, 0)
    else:
      l[i] = 100
  l = np.argsort(l)
  for i in range(4):
    phdbook[i][0] = book[l[i]][0]
    phdbook[i][1] = book[l[i]][1]
  book[:] = phdbook


def test_update():
  cdef:
    double[:,:,:,:,::1] avf = np.zeros([2, 10, 11, 3, 13], order='C')
    int[:] state = np.array([1, 2, 0, 0], dtype=np.int32)
    int[:] newstate = np.array([1, 3, 0, 0], dtype=np.int32)
    double[:,::1] book = np.zeros([4, 2], order='C')
    int action = 0
    double reward = 0
    double gamma = 0.9
    double alpha = 0.3
  avf[1][3][0][0][1] = 1
  update(avf, state, newstate, book, action, reward, alpha, gamma)
  return avf[1][2][0][0][0]


cdef update(double[:,:,:,:,::1] avf, int[:] state,
  int[:] newstate, double[:,::1] book, int action, double reward, double alpha,
  double gamma):
  ud = alpha * (reward + gamma * maxq(avf, newstate, book))
  ud -= alpha * avf[state[0]][state[1]][state[2]][state[3]][action]
  avf[state[0]][state[1]][state[2]][state[3]][action] += ud


cdef maxq(double[:,:,:,:,::1] avf, int[:] state, double[:,::1] book):
  cdef:
    int[:] ava = np.array(get_ava(book, state[3]), dtype=np.int32)
    double[:] avflist = np.empty_like(ava, dtype=np.float64)
  for i in range(len(ava)):
    avflist[i] = get_avf(avf, state, ava[i])
  return ava[np.argmax(avflist)]


def test_transition():
  cdef:
    double[:,::1] book = np.array([[1, 1.01], [1, 1.05], [0, 0], [0, 0]], order='C')
    int[:] state = np.array([0, 7, b_to_ind(book), 2], dtype=np.int32)
    double[:] pr = np.array([1.02, 1.03])
    int[:] mm = np.array([1, 1], dtype=np.int32)
    int[:] ln = np.array([0, 1], dtype=np.int32)
    double[:,:,:,:,::1] avf = np.zeros([2, 10, 11, 3, 13], order='C')
    int i = 0
    double ep = 0.1
    double spread = 0.001
  avf[0][7][b_to_ind(book)][1][9] = 1
  newstate, action, reward = transition(state, pr, mm, ln, book, avf, i, ep, spread)
  return newstate, action, reward, np.asarray(book)


cdef transition(int[:] state, double[:] pr, int[:] mm, int[:] ln,
  double[:,::1] book, double[:,:,:,:,::1] avf, int i, double ep,
  double spread):
  cdef:
    double reward = 0
    int[:] newstate = np.empty([4], dtype=np.int32)
    int[:] ava = np.array(get_ava(book, state[3]), dtype=np.int32)
    double[:] avflist = np.empty_like(ava, dtype=np.float64)
    int j = 0
    int bestaction
    int action
  for j in range(len(ava)):
    avflist[j] = get_avf(avf, state, ava[j])
  bestaction = ava[np.argmax(avflist)]
  if np.random.choice([1, 0], p=[1-ep*(len(ava)-1),ep*(len(ava)-1)]):
    action = bestaction
  else:
    action = np.random.choice(ava)
  reward = update_book(book, state, newstate, action, pr[i], spread)
  newstate[2] = b_to_ind(book)
  if action > 8:
    newstate[1] = ln[i]
    newstate[0] = mm[i]
  else:
    newstate[1] = ln[i + 1]
    newstate[0] = mm[i + 1]
  return newstate, action, reward


cdef inline get_avf(double[:,:,:,:,::1] avf, int[:] state,
  int action):
  return avf[state[0]][state[1]][state[2]][state[3]][action]


cdef get_ava(double[:,::1] book, int mode):
  cdef:
     int i = 0
     int pot_out = 0
     int n_pos = 0
  for i in range(4):
    if book[i][0] != 0:
      n_pos += 1
    pot_out += int(book[i][0])
  if n_pos == 0:
    return [0, 1, 2, 3, 4, 5, 6, 7, 8]
  elif mode == 1:
    if n_pos == 1:
      if pot_out == 1:
        return [0, 1, 2, 3, 9]
      elif pot_out == 2:
        return [0, 1, 2, 9]
      elif pot_out == 3:
        return [0, 1, 9]
      elif pot_out == 4:
        return [0, 9]
    elif n_pos == 2:
      if pot_out == 2:
        return [0, 1, 2, 9, 10]
      elif pot_out == 3:
        return [0, 1, 9, 10]
      else:
        return [0, 9, 10]
    elif n_pos == 3:
      if pot_out == 3:
        return [0, 1, 9, 10, 11]
      else:
        return [0, 9, 10, 11]
    else:
      return [0, 9, 10, 11, 12]
  elif mode == 2:
    if n_pos == 1:
      if pot_out == 1:
        return [0, 5, 6, 7, 9]
      elif pot_out == 2:
        return [0, 5, 6, 9]
      elif pot_out == 3:
        return [0, 5, 9]
      elif pot_out == 4:
        return [0, 9]
    elif n_pos == 2:
      if pot_out == 2:
        return [0, 5, 6, 9, 10]
      elif pot_out == 3:
        return [0, 5, 9, 10]
      else:
        return [0, 9, 10]
    elif n_pos == 3:
      if pot_out == 3:
        return [0, 5, 9, 10, 11]
      else:
        return [0, 9, 10, 11]
    else:
      return [0, 9, 10, 11, 12]


cdef inline long_rew(double[:,::1] book, double pr, int i, int mode, int upd,
  double spread):
  if mode == 1:
    r =  book[i][0] - book[i][0]*book[i][1]/(pr - spread)
    if upd == 1:
      book[i][0] = 0.
      book[i][1] = 0.
    return r
  r = book[i][0]*book[i][1]/(pr + spread) - book[i][0]
  if upd == 1:
    book[i][0] = 0.
    book[i][1] = 0.
  return r


cdef update_book(double[:,::1] book, int[:] state, int[:] newstate,
  int action, double pr, double spread):
  cdef int i = 0, n_pos = 0
  if action == 0:
    newstate[3] = state[3]
    return 0
  elif action < 5:
    book[3][0] = action
    book[3][1] = pr
    newstate[3] = 1
    return 0
  elif action < 9:
    book[3][0] = action - 4
    book[3][1] = pr
    newstate[3] = 2
    return 0
  else:
    r = long_rew(book, pr, action - 9, state[3], 1, spread)
    for i in range(4):
      if book[i][0] != 0:
        n_pos += 1
    if n_pos == 0:
      newstate[3] = 0
    else:
      newstate[3] = state[3]
    return r


cdef int b_to_ind(double[:,::1] book):
  cdef:
     int i = 0
     int pot_out = 0
     int n_pos = 0
  for i in range(4):
    if book[i][0] != 0:
      n_pos += 1
    pot_out += int(book[i][0])
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
