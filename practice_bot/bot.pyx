import numpy as np


def train_bot(double[:] pr, int[:] mm, int[:] ln,
  double alpha, double gamma, double ep, int n_epochs,
  double[:,:,:,:,::1] avf):
  cdef:
    double[:,::1] book = np.zeros([4, 2], order='C')
    int[:] state = np.array([mm[0], ln[0], b_to_ind(book), 0], dtype=np.int32)
    int[:] newstate = np.empty_like(state)
    int i
    int maxi = len(pr) - 1
    int action
    double reward
    double spread = 0.15
  rewards = []
  for _ in range(n_epochs):
    i = 0
    state = np.array([mm[0], ln[0], b_to_ind(book), 0], dtype=np.int32)
    newstate = np.empty_like(state)
    book = np.zeros([4, 2], order='C')
    while i < maxi:
      sort_book(book, state, pr[i])
      newstate, action, reward, flag = (
        transition(state, pr, mm, ln, book, avf, i, ep, spread)
      )
      if flag == 0:
        i += 1
      newstate[1] = ln[i]
      newstate[0] = mm[i]
      rewards.append(reward)
      ud = update(avf, state, newstate, book, action, reward, alpha, gamma)
      state = newstate[:]
  return avf, rewards


def implament_policy(double[:] pr, int[:] mm, int[:] ln,
  double[:,:,:,:,::1] avf):
  cdef:
    double[:,::1] book = np.zeros([4, 2], order='C')
    int[:] state = np.array([mm[0], ln[0], b_to_ind(book), 0], dtype=np.int32)
    int[:] newstate = np.empty_like(state)
    double reward
    int i = 0
    int maxi = len(pr) - 1
  actions = []
  rewards = []
  uds = []
  while i < maxi:
    sort_book(book, state, pr[i])
    newstate, action, reward, flag = (
     transition(state, pr, mm, ln, book, avf, i, 0, 0.15)
    )
    actions.append(action)
    rewards.append(reward)
    if flag == 0:
      i += 1
    newstate[1] = ln[i]
    newstate[0] = mm[i]
    ud = update(avf, state, newstate, book, action, reward, 0.01, 0.9)
    uds.append(ud)
    state = newstate[:]
  return actions, rewards, uds


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
    int[4] inds = np.empty([4], dtype=np.int32)
  l = np.empty([4])
  for i in range(4):
    if book[i][0] != 0:
      l[i] = -long_rew(book, pr, i, state[3], 0, 0)
    else:
      l[i] = 100
  inds = np.argsort(l)
  for i in range(4):
    phdbook[i][0] = book[inds[i]][0]
    phdbook[i][1] = book[inds[i]][1]
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
  ud = alpha * (reward + gamma * get_avf(avf, newstate, maxq(avf, newstate, book)))
  ud -= alpha * get_avf(avf, state, action)
  avf[state[0]][state[1]][state[2]][state[3]][action] += ud
  return ud


cdef maxq(double[:,:,:,:,::1] avf, int[:] state, double[:,::1] book):
  cdef:
    int[:] ava = np.array(get_ava(book, state[3]), dtype=np.int32)
    double[:] avflist = np.empty_like(ava, dtype=np.float64)
  for i in range(len(ava)):
    avflist[i] = get_avf(avf, state, ava[i])
  return ava[np.argmax(avflist)]

"""
def test_transition():
  cdef:
    double[:,::1] book = np.array([[2, 1.02], [0, 0], [0, 0], [0, 0]], order='C',dtype=np.float64)
    int[:] state = np.array([0, 3, b_to_ind(book), 1], dtype=np.int32)
    double[:] pr = np.array([1.02, 1.03])
    int[:] mm = np.array([0, 0], dtype=np.int32)
    int[:] ln = np.array([3, 4], dtype=np.int32)
    double[:,:,:,:,::1] avf = np.zeros([2, 10, 11, 3, 13], order='C')
    int i = 0
    double ep = 0.
    double spread = 0.001
  avf[0][3][b_to_ind(book)][1][1] = 1
  newstate, action, reward = transition(state, pr, mm, ln, book, avf, i, ep, spread)
  return np.asarray(newstate), action, reward, np.asarray(book), np.asarray(state)
"""

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
    int flag = 0
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
    flag = 1
  return newstate, action, reward, flag


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


def wbti(book):
  cdef double[:,::1] b = np.copy(book, order='C')
  return b_to_ind(b)

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
