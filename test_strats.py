import pyximport
pyximport.install(reload_support=True)
import strats as st
import pickle as pkl
import numpy as np

p = pkl.load(open('ends_15.pkl', 'rb'))
d = pkl.load(open('md_dict_15.pkl', 'rb'))


def test_getwin():
    w = st.get_windows(p[10:], d[10][0], d[10][1], 10, 1)
    i = 10
    while abs(p[i] - d[10][0][i - 10]) <= d[10][1][i - 10]:
        i += 1
    assert p[i] == w[0][0][0]
    pr = np.array([1, 3, 5, 1, -1, -2, 1]).astype(np.float64)
    ms = np.array([1, 1, 1, 2, 1, 0, 0]).astype(np.float64)
    ds = np.array([1, 1, 2, 1, 0, 0, 0]).astype(np.float64)
    w, m, d2, ab = st.get_windows(pr, ms, ds, 3, 1)
    assert w[0][0] == 3 and w[0][1] == 5
    assert w[1][0] == -1 and w[1][2] == 1
    assert m[0] == 1
    assert m[1] == 1
    assert d2[0] == 1
    assert d2[1] == 0
    assert ab[0] == 1
    assert ab[1] == 0


def test_strats():
    win = np.array([1, 1, 1, 2, 3, 6, 7, 0]).astype(np.float64)
    assert st.apply_strat_b(win, 1, 1, 3, -3, 100) == 600
    win = np.array([10, 10, 9, 8, 7, 3, 1]).astype(np.float64)
    assert st.apply_strat_b(win, 9, 2, 0.99, -1, 10) == 7
    win = np.array([3, 3, 2.5, 2, 1, 1]).astype(np.float64)
    assert st.apply_strat_a(win, 3, 1.5, 1, -1, 30) == 90
    win = np.array([3, 4, 4.5, 5, 7, 10, 9]).astype(np.float64)
    assert st.apply_strat_a(win, 2, 2, 3, 0, 1) == 3 / 10
