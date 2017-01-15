from analyse import get_means_devs, lookahead, make_lookahead_array
import numpy as np
import scipy.stats as st
import pickle as pkl

md_dict = pkl.load(open('windows.pkl', 'rb'))
prices = pkl.load(open('ends.pkl', 'rb'))


def test_means():
    data = np.random.choice(range(100), 1000)
    m, d = get_means_devs(data, 10)
    assert m[30] == np.mean(data[21:31])
    assert d[100] == np.sqrt(st.moment(data[91:101], moment=2))


def test_lookahead():
    prices = [1, 1, 2, 7, 3, 1, -20, -25, 15, 2]
    means = [1, 1, 1.5, 3, 4, 3, 7, 20, 18, 17]
    devs = [1, 1, 1, 1, 2, 2, 1, 3, 4, 20]
    assert lookahead(prices, means, devs, 1, 1, 1) == ([-5], [-4])
    assert lookahead(prices, means, devs, 1, 2, 1) == ([35], [-6])
    assert lookahead(prices, means, devs, 26, 1, 1) == ([-5], [])
    assert lookahead(prices, means, devs, 27, 1, 1) == ([], [])


def test_lha():
    lha = make_lookahead_array(prices, md_dict, 10, 1)
    lha_to_test = [l[1] for l in lha if l[0] == (30, 2, 4)][0]
    assert lha_to_test == lookahead(prices, md_dict[30][0], md_dict[30][1],
                                    4, 2, 30)
