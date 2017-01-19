import numpy as np
import scipy.stats as st


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
