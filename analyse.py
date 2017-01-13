import numpy as np
import scipy.stats as st
import pickle as pkl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn

prices = pkl.load(open('ends.pkl', 'rb'))
times = pkl.load(open('times.pkl', 'rb'))


def get_means_devs(X, window_len):
    means = [np.mean(X[:i+1]) for i in range(window_len)]
    devs = [np.sqrt(st.moment(X[:i+1], moment=2)) for i in range(window_len)]
    means += [np.mean(X[i+1: i+window_len+1])
              for i in range(len(X)-window_len)]
    devs += [np.sqrt(st.moment(X[i+1: i+window_len+1], moment=2))
             for i in range(len(X)-window_len)]
    return np.array(means), np.array(devs)


def plot_section(X, means, devs, t_start, t_end, fac):
    plt.plot(times[t_start: t_end], prices[t_start: t_end])
    plt.plot(times[t_start: t_end], means[t_start: t_end])
    plt.plot(times[t_start: t_end], (means+devs*fac)[t_start: t_end], c='blue')
    plt.plot(times[t_start: t_end], (means-devs*fac)[t_start: t_end], c='blue')
    plt.show()


def generate_data():
    windows = (np.arange(9)+np.ones([9]))*10
    windows = np.concatenate((windows, (np.arange(9)+np.ones([9]))*100))
    windows = np.concatenate((windows, (np.arange(9)+np.ones([9]))*1000))
    return dict(zip(windows.astype(np.int_), [get_means_devs(prices, w)
                                              for w in windows.astype(np.int_)]
                    ))


def lookahead(X, means, devs, fac, stride, skip):
    lower_diffs = [X[i+stride+skip]-X[i+skip]
                   for i in range(len(X)-stride-skip)
                   if (X[i+skip] < (means[i+skip] - fac * devs[i+skip]))
                   and not X[i-1+skip] < (means[i-1+skip]
                                          - fac * devs[i-1+skip])]
    upper_diffs = [X[i+stride+skip]-X[i+skip]
                   for i in range(len(X)-stride-skip)
                   if X[i+skip] > (means[i+skip] + fac * devs[i+skip])
                   and not X[i-1+skip] > (means[i-1+skip]
                                          + fac * devs[i-1+skip])]
    return lower_diffs, upper_diffs


def make_lookahead_array(prices, md_dict):
    windows = (np.arange(9)+np.ones([9]))*10
    windows = np.concatenate((windows, (np.arange(9)+np.ones([9]))*100))
    windows = np.concatenate((windows, (np.arange(9)+np.ones([9]))*1000))
    windows = windows.astype(np.int_)

    strides = (np.arage(9) + np.ones([9]))
    strides = np.concatenate((strides, (np.arange(9)+np.ones([9]))*10))
    strides = np.concatenate((strides, (np.arange(9)+np.ones([9]))*100))
    strides = np.concatenate((strides, (np.arange(9)+np.ones([9]))*1000))
    strides = strides.astype(np.int_)

    facs = [(i+1)*4/30 for i in range(len(30))]
    return [((window, stride, fac),
             lookahead(prices, md_dict[window][0], md_dict[window][1],
                       fac, stride, stride))
            for window in windows
            for fac in facs
            for stride in strides]
