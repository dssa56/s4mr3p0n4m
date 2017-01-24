import pyximport
pyximport.install(reload_support=True);
from strats import wrap_get_diffs, wrap_apply_strat_a, wrap_apply_strat_b
import numpy as np

def get_forward_windows(double[:] prices, double[:] means, double[:] devs,
                        int win, double fact, int bwin):
    cdef:
        int i = bwin
        int l = len(prices)
    windows = []
    ms = []
    ds = []
    a_or_b = []
    while i <= l - win:
        if (prices[i] - means[i]) ** 2 > (fact * devs[i]) ** 2 :
            windows.append(prices[i:win + i])
            ms.append(means[i])
            ds.append(devs[i])
            if prices[i] > means[i]:
                a_or_b.append(1)
            else:
                a_or_b.append(0)
            i += 1
        else:
            i += 1
    return windows, ms, ds, a_or_b


def get_back_windows(double[:] prices, double[:] means, double[:] devs, int win,
                double fact):
    cdef:
        int i = win
        int l = len(prices)
    windows = []
    while i < l:
        if (prices[i] - means[i]) ** 2 > (fact * devs[i]) ** 2 :
            windows.append(prices[i - win: i])
            i += 1
        else:
            i += 1
    return windows


def categorise_windows(prices, md_dict, md_win, win, fact, bwin,
                       a_sl, a_sg, b_sl, b_sg):
    w, ms, ds, ab = get_forward_windows(prices[md_win:], md_dict[md_win][0],
                                        md_dict[md_win][1], win, fact, bwin)
    back_w = get_back_windows(prices[md_win:], md_dict[md_win][0],
                              md_dict[md_win][1], bwin, fact)
    cdef int i
    pos_neg = np.empty([len(w)], np.int32)
    for i in range(len(w)):
        if ab[i] == 1:
            if wrap_apply_strat_a(w[i], ms[i], ds[i], a_sl, a_sg, 1.0) > 1:
                pos_neg[i] = 1
            else:
                pos_neg[i] = 0
        else:
            if wrap_apply_strat_b(w[i], ms[i], ds[i], b_sl, b_sg, 1.0) > 1:
                pos_neg[i] = 1
            else:
                pos_neg[i] = 0
    return np.asarray(back_w), pos_neg
