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


def apply_strat_a(double[:] window, double mean, double dev,
                   double stop_loss, double stop_gain, double money):
    cdef int i
    for i in range(len(window)):
        if ((window[i] > mean + dev * stop_loss)
            or (window[i] < mean + dev * stop_gain)):
            return window[0] / window[i] * money
    return window[0] / window[-1] * money


def apply_strat_b(double[:] window, double mean, double dev,
                   double stop_loss, double stop_gain, double money):
    cdef int i
    for i in range(len(window)):
        if ((window[i] < mean - dev * stop_loss)
             or (window[i] > mean - dev * stop_gain)):
             return window[i] / window[0] * money
    return window[-1] / window[0] * money


def opt_strat(x, prices, md_dict, md_window, st_window, fact, samples, rdn):
    a_sl = x[0]
    a_sg = x[1]
    b_sl = x[2]
    b_sg = x[3]
    means = md_dict[md_window][0]
    devs = md_dict[md_window][1]
    w, ms, ds, ab = get_windows(prices[md_window:],
                                means, devs, st_window, fact)
    if not rdn:
        samples = len(w)
    si = np.random.choice(np.arange(len(w)), samples)
    cdef int i
    money = np.empty([samples], np.float64)
    for i in range(samples):
        if ab[si[i]] == 1:
            money[i] = apply_strat_a(w[si[i]], ms[si[i]], ds[si[i]],
                                  a_sl, a_sg, 1.0)
        else:
            money[i] = apply_strat_b(w[si[i]], ms[si[i]], ds[si[i]],
                                  b_sl, b_sg, 1.0)
    return 1/money.sum()


def strat_ngl(x, prices, md_dict, md_window, st_window, fact, s_len,
              loss_tol):
    a_sl = x[0]
    a_sg = x[1]
    b_sl = x[2]
    b_sg = x[3]
    means = md_dict[md_window][0]
    devs = md_dict[md_window][1]
    w, ms, ds, ab = get_windows(prices[md_window:],
                                means, devs, st_window, fact)
    cdef int i
    cdef int j
    cdef double money = 1.0
    record = []
    s = np.empty([s_len], np.float64)
    for i in range(s_len):
        old_money = money
        if ab[i] == 1:
            money = apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, money)
        else:
            money = apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, money)
        record.append(money)
        s[i] = 1 if money - old_money < 0 else 0
    for i in range(len(w) - s_len):
        i += s_len
        for j in range(s_len - 1):
          s[j] = s[j + 1]
        if sum(s) / s_len > loss_tol:
            if ab[i] == 1:
                s[-1] = (1 if apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, 1.0)
                         < 1.0 else 0)
            else:
                s[-1] = (1 if apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, 1.0)
                         < 1.0 else 0)
        else:
            old_money = money
            if ab[i] == 1:
                money = apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, money)
            else:
                money = apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, money)
            record.append(money)
            s[-1] = 1 if money - old_money < 0 else 0

    return record


def strat_rgl(x, prices, md_dict, md_window, st_window, fact, s_len,
              loss_tol):
    a_sl = x[0]
    a_sg = x[1]
    b_sl = x[2]
    b_sg = x[3]
    means = md_dict[md_window][0]
    devs = md_dict[md_window][1]
    w, ms, ds, ab = get_windows(prices[md_window:],
                                means, devs, st_window, fact)
    cdef int i
    cdef int j
    cdef double money = 1.0
    record = []
    s = np.empty([s_len], np.float64)
    play_money_inds = []

    for i in range(s_len):
        old_money = money
        if ab[i] == 1:
            money = apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, money)
        else:
            money = apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, money)
        record.append(money)
        s[i] = money - old_money

    i = s_len
    while i < len(w) - s_len:
        while (- sum(s)  < loss_tol) and (i < len(w) - s_len):
            for j in range(s_len - 1):
                s[j] = s[j + 1]
            money = record[-1]
            old_money = money
            if ab[i] == 1:
                money = apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, money)
            else:
                money = apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, money)
            record.append(money)
            s[-1] = money - old_money
            i += 1

        for j in range(s_len - 1):
            s[j] = s[j + 1]
        old_money = money
        if ab[i] == 1:
            money = apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, money)
        else:
            money = apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, money)
        s[-1] = money - old_money
        i += 1


    return record
