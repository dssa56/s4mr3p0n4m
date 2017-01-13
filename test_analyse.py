from analyse import get_means_devs, lookahead
import numpy as np
import scipy.stats as st
import pickle as pkl


def test_means():
    data = np.random.choice(range(100), 1000)
    m, d = get_means_devs(data, 10)
    assert m[30] == st.gmean(data[21:31])
    assert d[100] == np.sqrt(st.moment(data[91:101], moment=2))


m = pkl.load(open('100m.pkl', 'rb'))
d = pkl.load(open('100d.pkl', 'rb'))
p = pkl.load(open('ends.pkl', 'rb'))


def test_data():
    assert m[430] == st.gmean(p[331:431])
    assert d[140] == np.sqrt(st.moment(p[41:141], moment=2))


def test_lookahead():
    prices = [1, 1, 2, 7, 3, 1, -20, -25, 15, 2]
    means = [1, 1, 1.5, 3, 4, 3, 7, 20, 18, 17]
    devs = [1, 1, 1, 1, 2, 2, 1, 3, 4, 20]
    assert lookahead(prices, means, devs, 1, 1, 1) == ([-5], [-4])
    assert lookahead(prices, means, devs, 1, 2, 1) == ([35], [-6])
    assert lookahead(prices, means, devs, 26, 1, 1) == ([-5], [])
    assert lookahead(prices, means, devs, 27, 1, 1) == ([], [])
