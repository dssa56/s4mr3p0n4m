import numpy as np
import scipy.stats as st
from sklearn.linear_model import LinearRegression

mdl = LinearRegression()


cdef deviation(double[:] prices, double[:] means):
    cdef:
        int i
        double dev = 0
    for i in range(len(prices)):
        dev += (prices[i] - means[i])**2
    return np.sqrt(dev/len(prices))


def get_regs(double[:] prices, int reg_win):
    n = len(prices) - reg_win
    means = np.empty(n, np.float64)
    r = np.array([[j] for j in range(reg_win)])
    cdef int i
    for i in range(n):
        mdl.fit(r, prices[i : reg_win + i])
        means[i] = mdl.predict([[reg_win - 1]])
    return means


def get_regs_devs(double[:] prices, int dev_win, int reg_win):
    n = len(prices) - reg_win
    means = np.empty(n, np.float64)
    devs = np.empty(n - dev_win, np.float64)
    r_reg = np.array([[j] for j in range(reg_win)])
    cdef int i = 0
    while i < dev_win:
        mdl.fit(r_reg, prices[i : reg_win + i])
        means[i] = mdl.predict([r_reg[-1]])
        i += 1
    while i < n:
        mdl.fit(r_reg, prices[i : reg_win + i])
        means[i] = mdl.predict([r_reg[-1]])
        devs[i - dev_win] = deviation(
                            prices[i + reg_win - 1: i + reg_win + dev_win - 1],
                            means[i - dev_win: i])
        i += 1
    return means[dev_win:], devs


def get_means_devs(double[:] prices, int window):
    n = len(prices) - window
    means = np.empty(n, np.float64)
    devs = np.empty(n, np.float64)
    cdef int i
    for i in range(len(prices) - window ):
        means[i] = np.mean(prices[i : window + i])
        devs[i] = np.sqrt(st.moment(prices[i : window + i], moment=2))
    return means, devs


def generate_data(double[:] prices):
    windows = (np.arange(9) + np.ones([9])) * 10
    windows = np.concatenate((windows, (np.arange(9) + np.ones([9])) * 100))
    windows = np.concatenate((windows, (np.arange(9) + np.ones([9])) * 1000))
    return dict(zip(windows.astype(np.int_), [get_means_devs(prices, w)
                                              for w in windows.astype(np.int_)]
                    ))
