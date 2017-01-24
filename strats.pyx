import numpy as np


def get_windows(double[:] prices, double[:] means, double[:] devs, int win,
                double fact):
    cdef:
        int i = 0
        int l = len(prices)
    windows = []
    ms = []
    ds = []
    a_or_b = []
    while i <= l - win:
        if (prices[i] - means[i]) ** 2 > (fact * devs[i]) ** 2 :
            windows.append(prices[i: i + win])
            ms.append(means[i])
            ds.append(devs[i])
            if prices[i] > means[i]:
                a_or_b.append(1)
            else:
                a_or_b.append(0)
            i += win
        else:
            i += 1
    return windows, ms, ds, a_or_b


cdef apply_strat_a(double[:] window, double mean, double dev,
                   double stop_loss, double stop_gain, double money):
    cdef int i
    for i in range(len(window)):
        if ((window[i] > mean + dev * stop_loss)
            or (window[i] < mean + dev * stop_gain)):
            return window[0] / window[i] * money
    return window[0] / window[-1] * money


def wrap_apply_strat_a(win, m, d, sl, sg, mn):
    return apply_strat_a(win, m, d, sl, sg, mn)


cdef apply_strat_b(double[:] window, double mean, double dev,
                   double stop_loss, double stop_gain, double money):
    cdef int i
    for i in range(len(window)):
        if ((window[i] < mean - dev * stop_loss)
             or (window[i] > mean - dev * stop_gain)):
             return window[i] / window[0] * money
    return window[-1] / window[0] * money


def wrap_apply_strat_b(win, m, d, sl, sg, mn):
    return apply_strat_b(win, m, d, sl, sg, mn)


cdef get_loss(double[:] s, int bl, int nb):
    cdef int tot = 0
    cdef double prod = 1
    cdef int i
    cdef int j
    for i in range(nb):
        for j in range(bl):
            prod *= s[bl * i + j]
        if prod < 1:
            tot += 1
        prod = 1
    return tot


def wrap_get_loss(s, bl, nb):
    return get_loss(s, bl, nb)


cdef sl_sg(double[:] prices, md_dict, int md_window, int st_window,
               double fact, double a_sl, double a_sg, double b_sl,
               double b_sg):

    means = md_dict[md_window][0]
    devs = md_dict[md_window][1]
    w, ms, ds, ab = get_windows(prices[md_window:],
                                means, devs, st_window, fact)
    cdef int i
    money = np.empty([len(w)], np.float64)
    for i in range(len(w)):
        if ab[i] == 1:
            money[i] = apply_strat_a(w[i], ms[i], ds[i],
                                  a_sl, a_sg, 1.0)
        else:
            money[i] = apply_strat_b(w[i], ms[i], ds[i],
                                  b_sl, b_sg, 1.0)
    return money.sum()


cdef get_diffs(double[:] prices, md_dict, int md_window, int st_window,
               double fact, double a_sl, double a_sg, double b_sl,
               double b_sg):

    means = md_dict[md_window][0]
    devs = md_dict[md_window][1]
    w, ms, ds, ab = get_windows(prices[md_window:],
                                means, devs, st_window, fact)
    cdef int i
    money = np.empty([len(w) + 1], np.float64)
    money[0] = 1
    for i in range(len(w)):
        if ab[i] == 1:
            money[i + 1] = apply_strat_a(w[i], ms[i], ds[i],
                                         a_sl, a_sg, 1.0)
        else:
            money[i + 1] = apply_strat_b(w[i], ms[i], ds[i],
                                         b_sl, b_sg, 1.0)
    return money


def wrap_get_diffs(double[:] prices, md_dict, int md_window, int st_window,
               double fact, double a_sl, double a_sg, double b_sl,
               double b_sg):

    return get_diffs(prices, md_dict, md_window, st_window,
                   fact, a_sl, a_sg, b_sl, b_sg)


cdef stop_prs(stop_params, double[:] x):
  cdef int n_bins = stop_params[0]
  cdef int bin_len = stop_params[1]
  cdef int loss_tol = stop_params[2]
  cdef int l_s = n_bins * bin_len
  record = []
  cdef int i
  for i in range(l_s):
      record.append(x[i])
  for i in range(len(x) - l_s):
      if get_loss(x[i:i + l_s], bin_len, n_bins) < loss_tol:
          record.append(x[i + l_s])
  return np.prod(record)


def opt_stop_prs(stop_params, double[:] x):
    return 1 / stop_prs(stop_params, x)


def strat(stop_params, x):
  cdef int n_bins = stop_params[0]
  cdef int bin_len = stop_params[1]
  cdef int loss_tol = stop_params[2]
  cdef int l_s = n_bins * bin_len
  record = []
  cdef int i
  for i in range(l_s):
      record.append(x[i])
  for i in range(len(x) - l_s):
      if get_loss(x[i:i + l_s], bin_len, n_bins) < loss_tol:
          record.append(x[i + l_s])
  return np.cumprod(record)
