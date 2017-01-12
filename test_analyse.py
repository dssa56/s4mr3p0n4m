from analyse import get_means_devs
import numpy as np
import scipy.stats as st
import pickle as pkl


def test_means():
    data = np.random.choice(range(100), 1000)
    m, d = get_means_devs(data, 10)
    assert m[30] == st.gmean(data[20:30])
    assert d[100] == np.sqrt(st.moment(data[90:100], moment=2))


m = pkl.load(open('100m.dat', 'rb'))
d = pkl.load(open('100d.dat', 'rb'))
p = pkl.load(open('ends.pkl', 'rb'))


def test_data():
    assert m[30] == st.gmean(p[19:29])
    assert d[100] == np.sqrt(st.moment(p[90:100], moment=2))
