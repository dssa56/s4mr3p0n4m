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


if __name__ == '__main__':
    m, d = get_means_devs(prices, 100)
    pkl.dump(m, open('100m.pkl', 'wb'))
    pkl.dump(d, open('100d.pkl', 'wb'))
