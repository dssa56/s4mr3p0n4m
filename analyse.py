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
    means = [st.gmean(X[:i+1]) for i in range(window_len)]
    devs = [np.sqrt(st.moment(X[:i+1], moment=2)) for i in range(window_len)]
    means += [st.gmean(X[i+1: i+window_len+1])
              for i in range(len(X)-window_len)]
    devs += [np.sqrt(st.moment(X[i+1: i+window_len+1], moment=2))
             for i in range(len(X)-window_len)]
    return np.array(means), np.array(devs)


def plot_section(X, means, devs, t_start, t_end):
    plt.plot(times[t_start: t_end], prices[t_start: t_end])
    plt.plot(times[t_start: t_end], means[t_start: t_end])
    plt.plot(times[t_start: t_end], (means+devs)[t_start: t_end], c='blue')
    plt.plot(times[t_start: t_end], (means-devs)[t_start: t_end], c='blue')
    plt.show()


def generate_data():
    windows = (np.arange(9)+np.ones([9]))*10
    windows = np.concatenate((windows, (np.arange(9)+np.ones([9]))*100))
    windows = np.concatenate((windows, (np.arange(9)+np.ones([9]))*1000))
    return dict(zip(windows.astype(np.int_), [get_means_devs(prices, w)
                                              for w in windows.astype(np.int_)]
                    ))
