import pyximport
pyximport.install(reload_support=True)
import strats as st
import pickle as pkl
import numpy as np

p = pkl.load(open('ends.pkl', 'rb'))
d = pkl.load(open('md_dict.pkl', 'rb'))


def test_getwin():
    w = st.get_windows(p[10:], d[10][0], d[10][1], 10, 1)
    i = 10
    while abs(p[i] - d[10][0][i - 10]) <= d[10][1][i - 10]:
        i += 1
    if p[i] - d[10][0][i - 10] > 0:
        assert p[i] == w[0][0][0]
    else:
        assert p[i] == w[1][0][0]


def test_strats():
    win = np.array([1, 1, 1, 2, 3, 6, 7, 0]).astype(np.float64)
    assert st.apply_strat_b(win, 1, 1, 3, -3, 100) == 600
    win = np.array([10, 10, 9, 8, 7, 3, 1]).astype(np.float64)
    assert st.apply_strat_b(win, 9, 2, 0.99, -1, 10) == 7
    win = np.array([3, 3, 2.5, 2, 1, 1]).astype(np.float64)
    assert st.apply_strat_a(win, 3, 1.5, 1, -1, 30) == 90
    win = np.array([3, 4, 4.5, 5, 7, 10, 9]).astype(np.float64)
    assert st.apply_strat_a(win, 2, 2, 3, 0, 1) == 3 / 10
