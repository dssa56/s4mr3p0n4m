def get_windows(double[:] prices, double[:] means, double[:] devs, int win,
                double fact):
    cdef:
        int i = 0
        int l = len(prices)
    a_windows = []
    b_windows = []
    ms_a = []
    ds_a = []
    ms_b = []
    ds_b = []
    while i < l - win:
        if (prices[i] - means[i]) ** 2 > (fact * devs[i]) ** 2 :
            if prices[i] > means[i]:
                a_windows.append(prices[i: i + win])
                ms_a.append(means[i])
                ds_a.append(devs[i])
            else:
                b_windows.append(prices[i: i + win])
                ms_b.append(means[i])
                ds_b.append(devs[i])
            i += win
        else:
            i += 1
    return a_windows, b_windows, ms, ds


def apply_strat_a(double[:] window, double mean, double dev,
                   double stop_loss, double stop_gain, double money):
    cdef int i = 0
    while ((window[i] < mean + dev * stop_loss)
           and (window[i] > mean + dev * stop_gain)):
        i += 1
    return window[0] / window[i] * money


def apply_strat_b(double[:] window, double mean, double dev,
                   double stop_loss, double stop_gain, double money):
    cdef int i = 0
    while ((window[i] > mean - dev * stop_loss)
           and (window[i] < mean - dev * stop_gain)):
        i += 1
    return window[i] / window[0] * money


def strat(prices, md_dict, md_window, st_window, fact, a_sl, a_sg, b_sl, b_sg,
          opt):
    means = md_dict[md_window][0]
    devs = md_dict[md_window][1]
    w_a, w_b, ms, ds = get_windows(prices[md_window:],
                                   means, devs, st_window, fact)
    cdef int i
    cdef int count
    if opt:
        for i in range(len())
