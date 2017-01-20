def get_windows(double[:] prices, double[:] means, double[:] devs, int win,
                double fact):
    cdef:
        int i = 0
        int l = len(prices)
    windows = []
    ms = []
    ds = []
    a_or_b = []
    while i < l - win:
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


def strat(x, prices, md_dict, md_window, st_window, fact,
          money, opt):
    a_sl = x[0]
    a_sg = x[1]
    b_sl = x[0]
    b_sg = x[1]
    means = md_dict[md_window][0]
    devs = md_dict[md_window][1]
    w, ms, ds, ab = get_windows(prices[md_window:],
                                means, devs, st_window, fact)
    cdef int i
    if opt:
        for i in range(len(w)):
            if ab == 1:
                money = apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, money)
            else:
                money = apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, money)
        return 1/money
    else:
        record = []
        for i in range(len(w)):
            if ab == 1:
                money = apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, money)
            else:
                money = apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, money)
            record.append(money)
        return record
